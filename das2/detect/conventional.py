"""
das2.detect.conventional
========================

What a median-and-standard-deviation check would have said about the same data.

This is not a detector. It emits no `Signal`, nothing downstream consults it,
and it can neither raise an alert nor suppress one. It exists to answer one
question the client asked for directly: when this system reports something,
*would the statistics they already run have found it too?*

    "then verify oh there really is something that is not yet discovered by
     already existing stats calculation he put in"

Answering that honestly means computing their check, not a caricature of it.
So this runs the textbook form -- mean and standard deviation over the same
window, flag anything beyond k sigma -- on the same arrays the detectors saw,
and reports what it finds including when it finds MORE than we did.

Three things it measures, all of which are the client's own numbers
-------------------------------------------------------------------
**Did it cross?** `peak_z` is the largest |v - mean| / sigma in the window. If
that clears 3.0 their check fires and this system has told them nothing new.
That outcome is reported as plainly as the other one; a comparison that can
only come out one way is marketing, not evidence.

**Was sigma defined at all?** On this estate it very often is not. Of the
sensors reporting in a two-hour sample of the real feed, 57% never changed
value at all -- 1,481 of 2,602. For those, sigma is zero, `mean ± 3 sigma` is
a zero-width interval, and the check is not strict: it is undefined. Every
subsequent reading is either exactly normal or infinitely abnormal. Reporting
that as "passed" would be false; it is reported as `undefined`.

**Did the event hide itself?** This is the one worth reading. Sigma is computed
from the same window that contains the event, so the event inflates the
denominator that is supposed to reveal it -- the masking effect, which is why
robust statistics exist at all (Rousseeuw & Leroy 1987, *Robust Regression and
Outlier Detection*; Hampel 1974 for the breakdown-point argument). A single
outlier has unbounded influence on the mean and on sigma, and a sustained
event -- exactly what a regional incident is -- has more.

So `peak_z_excluded` recomputes the same score with the anomalous samples
removed from the baseline. When a sensor scores 2.1 sigma against a window
containing its own event and 7.8 sigma against the hours before it, that is
not a near miss. It is a check being blinded by the thing it is looking for,
and the number is theirs, computed from their data, in the units they already
trust.

**And would noise have crossed it anyway?** This is the half that stops the
comparison from being unfair in the other direction. "Beyond 3 sigma" is a
statement about ONE sample; a window holds hundreds, and the largest of them
is far beyond 3 sigma routinely. At 900 samples the probability that pure
Gaussian noise touches 3 sigma somewhere in the window is **91%**. Across 2,600
sensors that is roughly 2,370 healthy sensors crossing the line every run.

So every crossing is scored against what noise alone would produce at that
sample count -- `1 - (1 - erfc(z/sqrt2))^n`, the standard multiple-comparison
correction. A sensor that reaches 3.4 sigma in 900 samples gets
`p = 0.49`: a coin toss, reported as `chance` rather than as a find. A sensor
that reaches 25 sigma gets `p < 1e-12` and is reported as a genuine alarm --
theirs, not ours, and the report says so.

The fleet number matters too
----------------------------
Per incident this says "0 of 4 sensors would have alarmed". Across the run it
says how many sensors the same check would have flagged in total, next to how
many are explained by noise at that sample count. A check with no null
hypothesis fires at a fixed rate whatever the network is doing, and the two
numbers side by side are the cheapest way to show it.

Deliberately NOT done here
--------------------------
No verdict is drawn from any of this. It does not decide which incidents to
report, does not rank them and does not change a recommendation. It annotates
what was already decided by other means, exactly as the signature layer does,
and for the same reason: a comparison against someone else's method is an
argument, and an argument must never be wired into a dispatch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Sequence

import numpy as np

#: Sigma multiple for the primary comparison. 3.0 because that is what "median,
#: std and all that" means in practice everywhere it is written down, and
#: because a looser one would flatter this system rather than test it.
DEFAULT_K = 3.0

#: The looser variant, reported alongside. Some sites run 2 sigma, and an
#: incident that clears neither is a stronger statement than one that only
#: clears the strict threshold.
LOOSE_K = 2.0

#: Fewer samples than this either side and the comparison is not worth making:
#: a standard deviation from a handful of points is noise about noise.
MIN_SAMPLES = 12

#: Baseline samples required before the masking figure is computed. Removing
#: the event must leave enough behind to describe normal, or `peak_z_excluded`
#: is just a smaller sample with a smaller sigma, which would overstate the
#: effect this module exists to measure.
MIN_BASELINE = 30

#: A standard deviation at or below the instrument's own resolution describes
#: the quantiser, not the process. Treated as undefined rather than as a very
#: strict limit -- this is the `mad.replace(0, nan) -> fillna(0)` blindness
#: that made the incumbent detector silent on every quiet sensor, arrived at
#: from the opposite direction.
DEGENERATE_EPS = 1e-12

#: 1.4826 x MAD estimates sigma for a normal distribution. Carried so the
#: report can show what the same check would say with a robust scale, which is
#: the one-line fix to most of what is measured here.
MAD_TO_SIGMA = 1.4826

#: A crossing this likely under noise alone is not a finding. 0.05 is the
#: conventional line and is used here for exactly the reason it usually is:
#: it needs no justification from this system's own data.
CHANCE_P = 0.05


def peak_p_value(z: float, n: int) -> float:
    """
    Probability that `n` Gaussian samples reach |z| somewhere, by chance.

    `1 - (1 - tail)^n`, evaluated through `log1p`/`expm1` because at the
    magnitudes that matter -- a 25-sigma spike over 900 samples -- the direct
    form underflows to exactly 0 and 1 and loses the distinction between "very
    unlikely" and "impossible".

    This is what makes a sigma threshold interpretable at this sample count. A
    single sample beyond 3 sigma means something; the LARGEST of 900 samples
    beyond 3 sigma means almost nothing, and the difference between those two
    readings is the difference between a check that works and one that reports
    most of the fleet every run.
    """
    if z <= 0 or n <= 0:
        return 1.0
    tail = math.erfc(z / math.sqrt(2.0))
    if tail <= 0.0:
        return 0.0
    if tail >= 1.0:
        return 1.0
    return -math.expm1(n * math.log1p(-tail))


@dataclass(frozen=True)
class ConventionalCheck:
    """What `mean ± k·sigma` makes of one sensor's window."""

    sensor_key: str
    n: int
    mean: float
    median: float
    std: float
    mad: float
    peak_z: float
    peak_z_loose: float
    peak_z_robust: float
    peak_z_excluded: float | None
    crossings: int
    expected_crossings: float
    p_value: float
    degenerate: bool
    k: float = DEFAULT_K

    @property
    def crossed(self) -> bool:
        """Did any sample leave the band? Their check, taken literally."""
        return not self.degenerate and self.peak_z >= self.k

    @property
    def by_chance(self) -> bool:
        """
        It crossed, but noise at this sample count crosses too.

        Kept separate from `crossed` rather than folded into it, because the
        two answer different questions and the report needs both: their check
        DID fire here, and it would have fired on a healthy sensor just the
        same.
        """
        return self.crossed and self.p_value >= CHANCE_P

    @property
    def alarmed(self) -> bool:
        """A crossing that noise does not explain -- a real find, theirs."""
        return self.crossed and not self.by_chance

    @property
    def alarmed_loose(self) -> bool:
        return not self.degenerate and self.peak_z >= LOOSE_K

    @property
    def masked(self) -> bool:
        """
        The event inflated the sigma that was supposed to reveal it.

        True only when removing the event's own samples from the baseline moves
        the sensor from below the threshold to above it. A score that rises but
        stays under 3 sigma is not masking, it is a quiet sensor, and calling
        both the same would make the claim unfalsifiable.
        """
        return (self.peak_z_excluded is not None
                and not self.degenerate
                and self.peak_z < self.k <= self.peak_z_excluded)

    @property
    def verdict(self) -> str:
        if self.degenerate:
            return "undefined"
        if self.by_chance:
            return "chance"
        return "alarmed" if self.alarmed else "missed"

    def why(self) -> str:
        """One line an operator can check against their own spreadsheet."""
        if self.degenerate:
            return (f"sigma is 0 over {self.n} samples — the sensor never "
                    f"moved, so mean ± {self.k:g}σ is a zero-width band and "
                    f"the check has no answer to give")
        if self.by_chance:
            return (f"crosses at {self.peak_z:.1f}σ, but {self.n} samples of "
                    f"pure noise reach that {self.p_value * 100:.0f}% of the "
                    f"time — this one fires on healthy sensors too")
        if self.alarmed:
            return (f"reaches {self.peak_z:.1f}σ, {_odds(self.p_value)} — "
                    f"this one your check finds too")
        if self.masked:
            return (f"{self.peak_z:.1f}σ against a window containing its own "
                    f"event, {self.peak_z_excluded:.1f}σ against the hours "
                    f"before it — the event is in the sample that defines "
                    f"normal")
        if self.peak_z >= LOOSE_K:
            return (f"reaches {self.peak_z:.1f}σ — over {LOOSE_K:g}σ but under "
                    f"{self.k:g}σ")
        return f"peaks at {self.peak_z:.1f}σ, well inside the band"


def _odds(p: float) -> str:
    """
    `'1 in 4,300 windows'`, which reads better than `2.3e-4`.

    Capped, because it is not: a 25-sigma spike gives p around 1e-141, and
    spelling that out produced a 140-digit number in the middle of a sentence
    an operator was meant to read. Past a point the only honest phrasing is
    that noise does not do this.
    """
    if p <= 0 or p < 1e-9:
        return "which noise does not produce at all"
    return f"which noise produces about once in {1.0 / p:,.0f} windows"


def _spans_mask(ts, spans: Iterable[tuple[datetime, datetime]]) -> np.ndarray:
    """Boolean mask of samples falling inside any anomalous span."""
    stamps = np.asarray(ts, dtype="datetime64[ns]")
    mask = np.zeros(stamps.shape, dtype=bool)
    for start, end in spans:
        if start is None or end is None:
            continue
        mask |= ((stamps >= np.datetime64(start, "ns"))
                 & (stamps <= np.datetime64(end, "ns")))
    return mask


def _peak_z(values: np.ndarray, centre: float, scale: float) -> float:
    if scale <= DEGENERATE_EPS or values.size == 0:
        return 0.0
    return float(np.max(np.abs(values - centre)) / scale)


def evaluate(sensor_key: str, ts, values, *,
             spans: Sequence[tuple[datetime, datetime]] = (),
             k: float = DEFAULT_K) -> ConventionalCheck | None:
    """
    Run the classical check over one sensor's window.

    `spans` are the periods this system flagged. They are used only to split
    the window into "the event" and "everything else" for the masking figure;
    the primary score is computed over the whole window exactly as a check that
    knows nothing about our findings would compute it.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < MIN_SAMPLES:
        return None

    mean = float(np.mean(values))
    median = float(np.median(values))
    std = float(np.std(values))
    mad = float(np.median(np.abs(values - median)))
    robust = mad * MAD_TO_SIGMA

    degenerate = std <= DEGENERATE_EPS
    peak = _peak_z(values, mean, std)
    peak_robust = _peak_z(values, median, robust)

    crossings = (0 if degenerate
                 else int(np.count_nonzero(np.abs(values - mean) >= k * std)))
    # What the same threshold yields on noise alone. Two-sided normal tail.
    expected = values.size * math.erfc(k / math.sqrt(2.0))

    peak_excluded: float | None = None
    if spans is not None and len(spans) and not degenerate:
        try:
            inside = _spans_mask(ts, spans)
        except (TypeError, ValueError):                    # pragma: no cover
            inside = np.zeros(values.shape, dtype=bool)
        if inside.shape == values.shape:
            baseline, event = values[~inside], values[inside]
            if baseline.size >= MIN_BASELINE and event.size:
                base_mean = float(np.mean(baseline))
                base_std = float(np.std(baseline))
                if base_std > DEGENERATE_EPS:
                    peak_excluded = _peak_z(event, base_mean, base_std)

    return ConventionalCheck(
        sensor_key=sensor_key,
        n=int(values.size),
        mean=round(mean, 6),
        median=round(median, 6),
        std=round(std, 6),
        mad=round(mad, 6),
        peak_z=round(peak, 3),
        peak_z_loose=round(peak, 3),
        peak_z_robust=round(peak_robust, 3),
        peak_z_excluded=(round(peak_excluded, 3)
                         if peak_excluded is not None else None),
        crossings=crossings,
        expected_crossings=round(expected, 2),
        p_value=peak_p_value(peak, int(values.size)),
        degenerate=degenerate,
        k=k,
    )


def evaluate_all(series: dict[str, tuple[Any, Any]], *,
                 spans_by_sensor: dict[str, list[tuple[datetime, datetime]]]
                 | None = None,
                 k: float = DEFAULT_K) -> dict[str, ConventionalCheck]:
    """Run the check over every sensor analysed this run."""
    spans_by_sensor = spans_by_sensor or {}
    out: dict[str, ConventionalCheck] = {}
    for key, (ts, values) in series.items():
        check = evaluate(key, ts, values,
                         spans=spans_by_sensor.get(key, ()), k=k)
        if check is not None:
            out[key] = check
    return out


def for_incident(incident, checks: dict[str, ConventionalCheck]) -> dict[str, Any]:
    """
    What their check makes of one incident, for the report.

    `headline` is written to be read on its own, because it is the sentence
    that answers "is this actually new?". Where the check would have fired it
    says so first -- the honest answer is the useful one, and an operator who
    catches this overstating itself once will not trust the rest of the page.
    """
    members = [m for m in incident.cluster.members]
    rows: list[dict[str, Any]] = []
    for member in members:
        check = checks.get(member.sensor.sensor_key)
        if check is None:
            continue
        rows.append({
            "sensor": member.sensor.description or member.sensor.sensor_key,
            "equipment": member.sensor.equipment,
            "verdict": check.verdict,
            "peak_z": check.peak_z,
            "peak_z_excluded": check.peak_z_excluded,
            "peak_z_robust": check.peak_z_robust,
            "p_value": check.p_value,
            "masked": check.masked,
            "why": check.why(),
        })
    if not rows:
        return {}

    alarmed = sum(1 for r in rows if r["verdict"] == "alarmed")
    chance = sum(1 for r in rows if r["verdict"] == "chance")
    undefined = sum(1 for r in rows if r["verdict"] == "undefined")
    masked = sum(1 for r in rows if r["masked"])

    if alarmed == len(rows):
        headline = (f"Your current check finds this too — all {alarmed} "
                    f"sensor(s) clear {DEFAULT_K:g}σ by more than noise "
                    f"explains.")
    elif alarmed:
        headline = (f"Your current check finds part of this: {alarmed} of "
                    f"{len(rows)} sensor(s) clear {DEFAULT_K:g}σ by more than "
                    f"noise explains.")
    elif chance:
        headline = (f"Nothing here that a median-and-σ check could rely on: "
                    f"{chance} of {len(rows)} sensor(s) cross {DEFAULT_K:g}σ, "
                    f"but noise crosses just as often at this sample count.")
    else:
        headline = (f"No sensor here reaches {DEFAULT_K:g}σ — a median-and-σ "
                    f"check sees nothing at any of them.")
    return {
        "headline": headline,
        "alarmed": alarmed,
        "chance": chance,
        "undefined": undefined,
        "masked": masked,
        "of": len(rows),
        "found_because": found_because(incident, rows),
        "sensors": sorted(rows, key=lambda r: -r["peak_z"]),
    }


def found_because(incident, rows: Sequence[dict[str, Any]]) -> str:
    """
    Why this system saw it when a per-sensor threshold did not.

    Ordered by how much of the gap each explanation accounts for, and drawn
    only from evidence already computed elsewhere. It names a mechanism rather
    than claiming superiority: "no single sensor is unusual; four of them
    moving together within 31 minutes is" is a statement the reader can check
    against the table underneath it.
    """
    cluster = incident.cluster
    reasons: list[str] = []

    if sum(1 for r in rows if r["masked"]):
        reasons.append("the event is long enough to inflate the σ that would "
                       "have revealed it — the scores in the last column are "
                       "the same sensors measured against the hours before it")
    if len(cluster.sites) > 1:
        spread = ""
        if cluster.start and cluster.end:
            minutes = (cluster.end - cluster.start).total_seconds() / 60.0
            spread = f" within {minutes:.0f} min"
        reasons.append(f"no single sensor is unusual on its own; "
                       f"{len(cluster.members)} of them across "
                       f"{len(cluster.sites)} sites moving together{spread} is")
    if len(cluster.equipment_types) > 1:
        reasons.append(f"{len(cluster.equipment_types)} different parameters "
                       f"moved together, which no per-sensor threshold can see")
    if sum(1 for r in rows if r["verdict"] == "undefined"):
        reasons.append("some of these sensors never move, so σ is 0 and the "
                       "threshold has no width to cross")
    if not reasons:
        reasons.append("the finding is in the shape of the series — a rate, a "
                       "flat line or a contradiction between instruments — "
                       "rather than in the size of any one value")
    return "; ".join(reasons[:3]) + "."


def summarise(checks: dict[str, ConventionalCheck]) -> dict[str, Any]:
    """
    Run-level counts, for the log line and the report header.

    `crossed` against `explained_by_noise` is the pair to read. A check with no
    null hypothesis fires at a fixed rate whatever the network is doing, and
    putting the two numbers next to each other is the cheapest way to show it:
    at 900 samples a healthy sensor touches 3 sigma 91% of the time.
    """
    if not checks:
        return {}
    values = list(checks.values())
    return {
        "sensors": len(values),
        "crossed": sum(1 for c in values if c.crossed),
        "explained_by_noise": sum(1 for c in values if c.by_chance),
        "would_alarm": sum(1 for c in values if c.alarmed),
        "would_alarm_loose": sum(1 for c in values if c.alarmed_loose),
        "undefined": sum(1 for c in values if c.degenerate),
        "masked": sum(1 for c in values if c.masked),
        "crossing_samples": sum(c.crossings for c in values),
        "expected_crossing_samples": round(
            sum(c.expected_crossings for c in values), 1),
    }
