"""
das2.detect.changepoint
=======================

Abrupt, sustained shifts in level -- the detector that makes an area event
visible.

Why this matters more than it sounds
------------------------------------
Every other detector here answers a question about one instrument's health: is
it frozen, silent, out of range, spiking. Those find broken sensors. But the
case the client actually described --

    "four different sensors, at three different sites, all going abnormal at
     03:00 in the same part of the island"

-- is not a sensor-health fault at all. Pressure drops and flow rises across a
district because *the water moved*, and every instrument involved is working
perfectly. Without a level-shift detector that event produces no signal
whatsoever, the clustering layer has nothing to cluster, and the regional map
stays empty however good the rest of the system is. Measured on the fixture
fleet before this module existed: the injected four-sensor regional event
produced **zero clusters**.

Why not CUSUM, in the end
-------------------------
The plan specified CUSUM, and CUSUM is the right tool when the baseline is
flat. This fleet's baseline is not flat: distribution pressure and flow swing
on a daily cycle of 6-30% of their own level, which is far larger than the
noise. A CUSUM tuned to detect a real step accumulates just as happily on the
morning demand ramp, so it fires every day on every healthy sensor.

The fix is not a bigger threshold -- that only trades false alarms for missed
events. It is to measure against a quantity the daily cycle barely affects:

    **sample-to-sample noise**, `MAD(diff(values))`, rather than the spread of
    the values themselves.

A diurnal cycle is slow, so it contributes almost nothing to the difference
between consecutive readings, while a step contributes its whole magnitude.
Measured on the fixture's clean diurnal pressure sensor: over a one-hour
window the daily cycle moves the level by ~1.3x the sample noise, whereas the
injected regional event moves it by ~29x. Two orders of magnitude of
separation, from choosing the right denominator rather than from tuning.

The real answer, and what this is standing in for
-------------------------------------------------
This is a within-window approximation. The correct instrument is the persisted
28-day time-of-day baseline (`das2.profile`), which knows what 03:00 on a
Tuesday normally looks like for each sensor and does not have to infer it from
72 hours. Until that job has run for a month, this is what is available, and it
is honest about being a lower bound: it can only see shifts that are abrupt,
and a slow drift of the same total size is invisible to it by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from das2.detect.profile import SensorProfile
from das2.models import AnomalyType, Signal
from das2.timeutils import to_epoch_seconds

DETECTOR = "changepoint"

#: Window either side of a candidate change, in seconds. Long enough that the
#: median is stable, short enough that the daily cycle barely moves across it.
DEFAULT_COMPARE_WINDOW_S = 3600.0

#: The shift must be at least this many multiples of the sensor's own
#: sample-to-sample noise. This is the gate that rejects the daily cycle:
#: measured on the fixture's clean diurnal sensors, one hour of cycle is ~1.3
#: of these, while a genuine step is ~29.
MIN_SHIFT_SIGMA = 6.0

#: ...and it must also be statistically real, in multiples of the standard
#: error of the difference of the two medians. Both gates are required: the
#: first rejects small-but-clean movements, the second rejects large-but-noisy
#: ones on a sensor with too few samples to tell.
MIN_SHIFT_SE = 8.0

#: ...and it must stand out against how much THIS SENSOR's level normally moves
#: over the same span. This is the gate that makes the detector safe on
#: integrating vessels. A reservoir level is the integral of net flow, which is
#: a random walk: its level genuinely wanders, so shifts that would be alarming
#: on a pressure transducer are simply Tuesday. Measured on a fixture reservoir,
#: the first two gates alone raised LEVEL_SHIFT from 4 to 20 across the fleet,
#: all of the new ones on the tank.
#:
#: Unlike the incumbent's per-sensor quantile -- which always won, so its
#: thresholds never bound and it emitted a fixed rate regardless of reality --
#: this is an ADDITIONAL gate on top of two absolute ones. It can only ever
#: make the detector quieter, never force it to speak.
MIN_SHIFT_MAD = 6.0

#: A step must hold for this long to count. Shorter excursions that return are
#: spikes, and `detect_spike` owns those.
MIN_SUSTAIN_S = 1800.0

#: How far past `min_sustain_s` the "is it still displaced?" window extends.
#: Two, so the check looks at roughly the half-hour to hour after the step --
#: long enough that a spike has already returned, short enough that an event
#: which recovers two hours later is still counted as having happened.
SUSTAIN_TAIL_FACTOR = 2.0

#: A sensor producing more than this many level shifts in one window is
#: DUTY-CYCLING, not faulting, and the detector abstains on it entirely.
#:
#: This is the gate the MAD test cannot provide. A pump's discharge flow is a
#: square wave between 0 and its running rate, so the sliding shift statistic
#: is exactly zero almost everywhere and jumps at each transition -- which
#: makes its MAD zero, which silently disables the MAD gate altogether.
#: Measured on the fixture the moment a duty-cycled pump was added:
#: LEVEL_SHIFT went from 5 to 12, every new one a pump doing its job.
#:
#: A genuine step change is a one-off. Nine of them in three days is a
#: schedule, and reporting a schedule as an anomaly is how an alerting system
#: teaches its operators to ignore it.
MAX_SHIFT_EVENTS = 3

#: Candidate changes closer together than this are one event.
MERGE_GAP_S = 1800.0

#: Sample-to-sample noise cannot be estimated from fewer than this.
MIN_POINTS = 60


# --------------------------------------------------------------------------- #
# Telling a recalibration from a water-level change
# --------------------------------------------------------------------------- #
# They are the same signal. Something stepped and stayed stepped, and nothing
# in the series says which. Every published sensor-fault taxonomy carries
# offset as an instrument fault -- Ni et al. (2009) ACM TOSN 5(3), Sharma et
# al. (2010) ACM TOSN 6(3), Leigh et al. (2019) STOTEN 664 -- and until now we
# had no way to express it, so every recalibration bump was reported as "the
# water moved, go look".
#
# Two properties separate them, and BOTH are required, because either alone is
# ambiguous:
#
#   1. **The transition is instantaneous.** Water has mass. A canal, a wet
#      well or a trunk main takes time to change level, so a real move has a
#      ramp -- samples that sit between the old level and the new one. An
#      engineer entering a new calibration constant produces a step with no
#      transition at all: one sample at the old value, the next at the new.
#
#   2. **It never comes back.** A hydraulic event recedes; a calibration is
#      the new truth. This is the test the literature leans on, and on its own
#      it cannot distinguish an offset from an event that simply has not
#      finished within our window -- which is why property 1 is also required.
#
# Honest status: this is OUR construction. No paper validates these two tests
# in combination, the thresholds below are engineering judgement rather than
# anything inherited, and the verdict is emitted as SUSPECT, never FAIL. The
# thing that would actually settle it is the maintenance work-order log: in
# the one published deployment with full alarm attribution, Leow et al. (2017)
# Environ. Sci.: Water Res. Technol. 3(2), 189 of 219 alarms were maintenance,
# resolved by knowing the schedule rather than by a better algorithm. Asking
# PUB for that log is worth more than any refinement of what follows.

#: How many samples may sit between the old and new level and still count as
#: "no transition". Two allows for one in-flight reading plus the sample the
#: change-point detector picked as the edge.
OFFSET_MAX_TRANSITION_SAMPLES = 2

#: A sample is "between the levels" when it is at least this far from both, as
#: a fraction of the step. Generous, so ordinary noise around either level is
#: not mistaken for a ramp.
OFFSET_LEVEL_TOLERANCE = 0.25

#: Counts as having come back if it ever returns within this fraction of the
#: step towards the old level.
OFFSET_RETURN_FRACTION = 0.5

#: How far either side of the detected edge to look for the transition's real
#: ends. Half an hour: longer than any recalibration and comfortably longer
#: than an urban catchment's 5-30 minute time of concentration.
OFFSET_SEARCH_S = 1800.0


def _median_dt(seconds: np.ndarray) -> float:
    """Typical sampling interval, for turning a duration into a sample count."""
    gaps = np.diff(seconds)
    gaps = gaps[gaps > 0]
    return float(np.median(gaps)) if gaps.size else 120.0


#: Do not call it an offset without at least this much data after the step. A
#: shift near the end of the window has not had the chance to recede, and
#: "never came back" over eleven minutes is not evidence of anything.
OFFSET_MIN_HOLD_S = 3600.0


def _is_instrument_offset(values: np.ndarray, seconds: np.ndarray,
                          peak: int, pre: float, magnitude: float) -> dict | None:
    """
    Evidence that a step is the instrument, not the water. `None` if not.

    Returns the measurements behind the verdict rather than a bare bool, so
    the report can show its working and an operator can disagree with it.
    """
    n = len(values)
    step = abs(magnitude)
    if step <= 0 or peak >= n - 1:
        return None

    held_s = float(seconds[-1] - seconds[peak])
    if held_s < OFFSET_MIN_HOLD_S:
        return None                      # too near the window's edge to judge

    post = pre + magnitude
    tol = OFFSET_LEVEL_TOLERANCE * step

    # 1. Is there a ramp? Measure the transition from where the value LAST sat
    #    at the old level to where it FIRST sits at the new one.
    #
    #    Counting "between" samples in a fixed window around the edge was not
    #    enough, and failed in the direction that matters. The change-point
    #    statistic anchors on the sharpest part of a move, which for a ramp is
    #    its end -- so a fifteen-minute rise had its edge placed where the
    #    value had already arrived, a +/-3 sample window saw only settled
    #    readings, and a genuine regional event was reported as a
    #    recalibration. Bracketing the transition finds its real length
    #    wherever the edge was placed inside it.
    search = max(4, int(OFFSET_SEARCH_S / max(1.0, _median_dt(seconds))))
    lo = max(0, peak - search)
    hi = min(n, peak + search + 1)
    window = values[lo:hi]
    at_pre = np.flatnonzero(np.abs(window - pre) <= tol)
    at_post = np.flatnonzero(np.abs(window - post) <= tol)
    if at_pre.size == 0 or at_post.size == 0:
        return None                      # never clearly at one level or other

    last_pre = int(at_pre[at_pre <= (peak - lo)].max()) if np.any(
        at_pre <= (peak - lo)) else int(at_pre.min())
    after = at_post[at_post > last_pre]
    if after.size == 0:
        return None
    between = int(after.min()) - last_pre - 1
    if between > OFFSET_MAX_TRANSITION_SAMPLES:
        return None                      # it ramped: water, not a constant

    # 2. Does it ever come back towards where it was?
    after = values[peak + 1:]
    if after.size == 0:
        return None
    closest = float(np.min(np.abs(after - pre)))
    if closest <= OFFSET_RETURN_FRACTION * step:
        return None                      # it receded: an event, not an offset

    return {
        "transition_samples": between,
        "held_h": round(held_s / 3600.0, 2),
        "closest_return_fraction": round(closest / step, 3),
    }


def _step_sigma(values: np.ndarray) -> float:
    """
    Per-sample noise, estimated from consecutive differences.

    `MAD(diff)` scaled to a standard deviation, then divided by sqrt(2) because
    a difference of two independent samples has twice the variance of one. The
    point of using differences rather than the spread of the values is that a
    slow trend -- the daily demand cycle -- cancels almost completely here but
    dominates the spread of the values themselves.
    """
    d = np.diff(values)
    d = d[np.isfinite(d)]
    if d.size < 8:
        return float("nan")
    mad = float(np.median(np.abs(d - np.median(d))))
    return 1.4826 * mad / np.sqrt(2.0)


def detect_level_shift(ts: pd.Series, values: np.ndarray,
                       profile: SensorProfile | None = None,
                       unit: str = "", *,
                       compare_window_s: float = DEFAULT_COMPARE_WINDOW_S,
                       min_sustain_s: float = MIN_SUSTAIN_S) -> list[Signal]:
    """
    Abrupt, sustained changes in level.

    Reports the shift in **engineering units** -- "pressure fell 1.47 bar" --
    because that is a statement an engineer can check against the instrument,
    which a dimensionless score is not.
    """
    n = len(values)
    if n < MIN_POINTS:
        return []

    seconds = to_epoch_seconds(ts)
    finite = np.isfinite(values) & np.isfinite(seconds)
    if finite.sum() < MIN_POINTS:
        return []
    values, seconds = values[finite], seconds[finite]
    n = len(values)

    sigma = _step_sigma(values)
    if not np.isfinite(sigma) or sigma <= 0:
        # A perfectly quantised or constant sensor has no sample noise to
        # measure against. That is the flatline detector's territory, not this
        # one's, and inventing a scale here would manufacture shifts.
        return []

    dt = np.diff(seconds)
    dt = dt[dt > 0]
    if dt.size == 0:
        return []
    cadence = float(np.median(dt))
    half = max(8, int(round(compare_window_s / cadence)))
    if n < 2 * half + 2:
        return []

    # Difference of medians either side of every interior point. The median is
    # used rather than the mean so a spike inside the comparison window cannot
    # masquerade as a change of level.
    idx = np.arange(half, n - half)
    before = np.array([np.median(values[i - half:i]) for i in idx])
    after = np.array([np.median(values[i:i + half]) for i in idx])
    shift = after - before

    # Standard error of the difference of two medians of `half` samples each.
    se = 1.253 * sigma * np.sqrt(2.0 / half)

    # How much this sensor's level moves over this span anyway. On a stable
    # sensor almost every sliding comparison is near zero, so a real step is a
    # vast outlier; on a random-walking reservoir level the spread is wide and
    # nothing stands out, which is the correct answer.
    shift_scale = 1.4826 * float(np.median(np.abs(shift - np.median(shift))))

    strong = (np.abs(shift) >= MIN_SHIFT_SIGMA * sigma) & \
             (np.abs(shift) >= MIN_SHIFT_SE * se)
    if shift_scale > 0:
        strong &= np.abs(shift) >= MIN_SHIFT_MAD * shift_scale
    if not strong.any():
        return []

    # Group adjacent candidates: one step produces a run of flagged points as
    # the comparison window slides across it, and the sharpest is the edge.
    candidates = idx[strong]
    groups: list[list[int]] = [[int(candidates[0])]]
    for c in candidates[1:]:
        if seconds[c] - seconds[groups[-1][-1]] <= MERGE_GAP_S:
            groups[-1].append(int(c))
        else:
            groups.append([int(c)])

    if len(groups) > MAX_SHIFT_EVENTS:
        # Routine duty cycling. See MAX_SHIFT_EVENTS.
        return []

    signals: list[Signal] = []
    for group in groups:
        local = np.array(group)
        peak = int(local[np.argmax(np.abs(shift[np.searchsorted(idx, local)]))])
        magnitude = float(shift[int(np.searchsorted(idx, peak))])

        # Sustained? Compare the level well after the change against the level
        # before it. A step that has already reverted is a spike.
        # Sustained means "still displaced a while later", NOT "displaced for
        # the rest of the window".
        #
        # The tail used to run to the end of the series, and that silently
        # discarded every event that recovers. On the fixture's regional event
        # -- four sites dropping together for 2.5 hours and then coming back --
        # the median of everything after the drop is the baseline the sensor
        # returned to, so the drop measured as "not sustained" and was thrown
        # away. Only the RECOVERY survived, which is how a pressure loss came
        # to be reported as a pressure rise. Most real water events are
        # transient; a test that only accepts permanent ones is not a
        # level-shift detector.
        tail_start = np.searchsorted(seconds, seconds[peak] + min_sustain_s)
        tail_end = np.searchsorted(seconds,
                                   seconds[peak] + min_sustain_s * SUSTAIN_TAIL_FACTOR)
        if tail_start >= n:
            tail_start = n - max(4, half // 2)
            tail_end = n
        tail = values[tail_start:max(tail_end, tail_start + 4)]
        if tail.size < 4:
            continue
        pre = float(np.median(values[max(0, peak - half):peak]))
        if abs(float(np.median(tail)) - pre) < MIN_SHIFT_SIGMA * sigma:
            continue

        end_index = min(n - 1, int(np.searchsorted(seconds,
                                                   seconds[peak] + min_sustain_s)))

        # Which of the two is it: the water, or the instrument?
        offset = _is_instrument_offset(values, seconds, peak, pre, magnitude)
        extra = {"offset_evidence": offset} if offset else {}

        signals.append(Signal(
            type=(AnomalyType.INSTRUMENT_OFFSET if offset
                  else AnomalyType.LEVEL_SHIFT),
            start=pd.Timestamp(ts.iloc[max(0, peak - 1)]).to_pydatetime(),
            end=pd.Timestamp(ts.iloc[end_index]).to_pydatetime(),
            detector=DETECTOR,
            magnitude=round(magnitude, 6),
            unit=unit,
            n_points=end_index - peak + 1,
            detail={
                "before": round(pre, 6),
                "after": round(pre + magnitude, 6),
                "sigma": round(abs(magnitude) / sigma, 1),
                "sample_noise": round(sigma, 6),
                "level_wander": round(shift_scale, 6),
                "relative": (round(magnitude / pre, 4) if pre else None),
                **extra,
            },
        ))
    return _collapse_excursions(signals)


#: A return within this share of the departure counts as coming back.
EXCURSION_RETURN_TOLERANCE = 0.35

#: And it has to be of comparable size -- a small drift back is not a recovery.
EXCURSION_COMPARABLE = 0.5


def _collapse_excursions(signals: list[Signal]) -> list[Signal]:
    """
    An event and its own recovery are ONE finding, not two.

    Once the sustain test stopped discarding events that recover -- which it
    used to, silently, by taking the median of everything after the step --
    the fixture's regional event started arriving twice: once for the drop,
    once for the return. Two incidents at the same four sites, two hours
    apart, for one thing that happened. That is worse for an operator than
    missing the onset, because now they have to work out that the second
    message is the first one ending.

    A departure followed by a return to roughly where it started is collapsed
    into one signal spanning both, carrying the DEPARTURE's sign and
    magnitude. The recovery is not lost -- it becomes `recovered_after_h`,
    which is the more useful fact anyway: an excursion that has already ended
    needs a different response from one still running.
    """
    if len(signals) < 2:
        return signals

    ordered = sorted(signals, key=lambda s: s.start)
    used: set[int] = set()
    out: list[Signal] = []

    for i, first in enumerate(ordered):
        if i in used:
            continue
        partner = None
        for j in range(i + 1, len(ordered)):
            if j in used:
                continue
            second = ordered[j]
            if first.magnitude * second.magnitude >= 0:
                continue                      # same direction: not a return
            if abs(second.magnitude) < EXCURSION_COMPARABLE * abs(first.magnitude):
                continue                      # too small to be the recovery
            before = (first.detail or {}).get("before")
            after = (second.detail or {}).get("after")
            if before is None or after is None:
                continue
            if abs(after - before) > EXCURSION_RETURN_TOLERANCE * abs(first.magnitude):
                continue                      # did not come back to where it was
            partner = (j, second)
            break

        if partner is None:
            out.append(first)
            continue

        j, second = partner
        used.add(j)
        held_h = max(0.0, (second.start - first.end).total_seconds() / 3600.0)
        detail = dict(first.detail or {})
        detail["recovered_after_h"] = round(
            (second.end - first.start).total_seconds() / 3600.0, 2)
        detail["held_h"] = round(held_h, 2)
        out.append(Signal(
            type=first.type,
            start=first.start,
            end=second.end,
            detector=first.detector,
            magnitude=first.magnitude,        # the DEPARTURE is the event
            unit=first.unit,
            n_points=(first.n_points or 0) + (second.n_points or 0),
            detail=detail,
        ))
    return sorted(out, key=lambda s: s.start)
