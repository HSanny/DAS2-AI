"""
das2.profile.build
==================

The daily job: long-history baselines, drift and noise-burst.

Why these cannot live in the hourly run
---------------------------------------
Two detectors were originally specified for the hourly pipeline and do not
belong there, for the same underlying reason: **72 hours is not enough data to
separate the fault from the daily cycle.**

* **DRIFT.** A 1%/day calibration drift is 3% across a 72-hour window, while
  diurnal swing in a distribution network is routinely 10-30%. A slope fitted
  over that window is dominated by which phase of the daily cycle the window
  happens to start and end on, not by any drift. It needs at least 14 daily
  medians, and 28 is better.

* **NOISE_BURST.** Its baseline noise level cannot come from the same window
  that contains the burst — that is the same self-contamination that made
  FLATLINE unable to fire, QUANTISATION_COLLAPSE blind to its own signal, and
  STUCK_IN_STATE silent on a pump stuck for 62 of 72 hours. Here it is worse,
  because the burst *is* the noise being measured.

So both move here, to a job that runs once a day over weeks. The same job
builds the **time-of-day baseline** the hourly run scores against — median and
MAD per 15-minute bucket of day, split weekday/weekend — which is what makes
`RESIDUAL_OUTLIER` possible at all.

Why time-of-day buckets rather than STL
---------------------------------------
STL was the obvious choice and is wrong for this fleet. Over 72 hours there are
only three daily cycles, so STL would estimate each seasonal phase from about
three observations, and a genuine 24-hour-scale fault would be absorbed into
the seasonal component precisely where the signal is. Weekly seasonality is
impossible to see at all. And many of these series are duty-cycle square waves
with a median of 0.0, where a "seasonal component" is an artefact of which
hours the pump happened to run.

A stored empirical bucket table has none of those failure modes: it is
gap-tolerant, indifferent to irregular sampling, survives a short window, and
an engineer can read it.

Scale, and why this aggregates rather than loads
-------------------------------------------------
Months x 2,672 sensors x 120 s is on the order of 10^8 rows. The functions here
take a per-sensor frame so they can be driven straight from a server-side
`GROUP BY` rather than pulling the fleet into pandas; `build_from_frame` is the
unit of work, and the caller decides how the rows arrive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from das2.models import AnomalyType, Signal
from das2.timeutils import to_epoch_seconds

DETECTOR = "profile"

#: Minimum history before a long-horizon statement is made at all. Below this,
#: the job abstains rather than reporting a slope fitted to a fortnight of
#: weather.
MIN_DAYS_DRIFT = 14
MIN_DAYS_NOISE = 7

#: Preferred history. The hourly pipeline works with less; these detectors are
#: simply better with more.
PREFERRED_DAYS = 28

#: Bucket width for the time-of-day baseline.
BUCKET_MINUTES = 15
BUCKETS_PER_DAY = 24 * 60 // BUCKET_MINUTES

#: A bucket needs this many samples across the history before it is trusted.
MIN_BUCKET_SAMPLES = 5

#: Drift must exceed this fraction of the sensor's own daily variation, per
#: day, to count. Expressed against the sensor's own spread because "0.1 bar a
#: day" means something entirely different on a 4 bar main and a 0.003 bar one.
DRIFT_MIN_FRACTION_PER_DAY = 0.02

#: ...and the trend must be consistent, not a random walk. Kendall's tau over
#: the daily medians; 0.5 means the ordering is clearly monotone.
#:
#: Measured, and left alone deliberately. Against a stationary sensor -- level
#: fixed, diurnal cycle, white noise -- tau >= 0.5 fired on **0 of 200** clean
#: series, which is the null this gate exists for.
#:
#: It was nearly raised on the strength of a second experiment that looked much
#: worse: against sensors whose LEVEL follows a random walk, tau >= 0.5 fired on
#: 39%, and no threshold up to 0.9 brought that under 0.7% without losing the
#: drifts worth catching. That reading was wrong. Checking what those series had
#: actually done: the ones tau flagged had moved a median of 10x their own noise
#: over three weeks, consistently in one direction, while the ones it rejected
#: had moved a third as far. A sensor whose level has walked 10x its noise away
#: from where it started IS drifting -- that is the finding, not a false alarm.
#: The experiment's "null" was a population of drifting sensors.
#:
#: Worth remembering before anyone tunes this: the null model has to be a
#: HEALTHY sensor, and a random walk is not one.
DRIFT_MIN_TAU = 0.5

#: A noise burst is a day whose within-day spread exceeds the median of the
#: preceding days by this multiple.
NOISE_BURST_MULTIPLE = 4.0


@dataclass
class TimeOfDayBaseline:
    """
    What this sensor normally reads at this time of day.

    Stored per (bucket, weekday/weekend) so the hourly run can score a window
    against it without re-reading weeks of history.
    """

    sensor_key: str
    #: {(bucket_of_day, is_weekend): (median, mad, n)}
    buckets: dict[tuple[int, int], tuple[float, float, int]] = field(default_factory=dict)
    days_observed: int = 0

    @property
    def usable(self) -> bool:
        return len(self.buckets) >= BUCKETS_PER_DAY // 4

    def lookup(self, when: datetime) -> tuple[float, float] | None:
        """Expected value and spread at a moment, or None if not yet learned."""
        bucket = (when.hour * 60 + when.minute) // BUCKET_MINUTES
        weekend = 1 if when.weekday() >= 5 else 0
        entry = self.buckets.get((bucket, weekend))
        if entry is None or entry[2] < MIN_BUCKET_SAMPLES:
            # Fall back to the other day-type before giving up: a sensor with
            # four weeks of weekdays and one weekend has a real baseline, just
            # not a weekend-specific one.
            entry = self.buckets.get((bucket, 1 - weekend))
        if entry is None or entry[2] < MIN_BUCKET_SAMPLES:
            return None
        return entry[0], entry[1]

    def as_rows(self) -> list[dict]:
        """Rows for `das2_sensor_profile`."""
        return [
            {"sensor_key": self.sensor_key, "bucket_of_day": bucket,
             "is_weekend": weekend, "median_value": median,
             "mad_value": mad, "n_samples": n}
            for (bucket, weekend), (median, mad, n) in sorted(self.buckets.items())
        ]


def _as_ts(ts):
    """
    A datetime Series, converting only when it is not one already.

    `pd.to_datetime` on an already-datetime column is not free: it decides
    whether to use its parse cache by ITERATING the values, which profiled at
    32% of the whole daily job -- 50,520 element-wise iterations for twenty
    sensors. Every function here takes a caller-supplied frame and so has to be
    defensive about the column's type; this makes being defensive cost nothing
    in the normal case, where it arrives from the reading store already typed.
    """
    series = ts if isinstance(ts, pd.Series) else pd.Series(ts)
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, errors="coerce")


def build_baseline(sensor_key: str, frame: pd.DataFrame, *,
                   ts_col: str = "ts", value_col: str = "value"
                   ) -> TimeOfDayBaseline:
    """
    Median and MAD per time-of-day bucket, over all the history given.

    Median rather than mean throughout, so one bad day in four weeks does not
    move the baseline that the next four weeks are judged against.
    """
    baseline = TimeOfDayBaseline(sensor_key=sensor_key)
    df = frame[[ts_col, value_col]].dropna()
    if df.empty:
        return baseline

    ts = _as_ts(df[ts_col])
    values = df[value_col].to_numpy(dtype=float)
    bucket = ((ts.dt.hour * 60 + ts.dt.minute) // BUCKET_MINUTES).to_numpy()
    weekend = (ts.dt.weekday >= 5).astype(int).to_numpy()
    baseline.days_observed = int(ts.dt.normalize().nunique())

    for key in set(zip(bucket.tolist(), weekend.tolist())):
        mask = (bucket == key[0]) & (weekend == key[1])
        sample = values[mask]
        sample = sample[np.isfinite(sample)]
        if sample.size == 0:
            continue
        median = float(np.median(sample))
        mad = float(np.median(np.abs(sample - median)))
        baseline.buckets[key] = (median, mad, int(sample.size))
    return baseline


#: A calendar day must COVER this share of a typical day's span to be compared
#: with other days. See `_whole_days`.
WHOLE_DAY_FRACTION = 0.9


def _whole_days(ts: pd.Series, values: np.ndarray):
    """
    Group by calendar day, dropping the partial days at the window's edges.

    This is not tidiness. A 28-day window almost never starts at midnight -- the
    hourly run reads the last 28 days from whenever it happens to run -- so the
    first and last calendar days cover a FRACTION of a day, and that fraction is
    one arbitrary phase of the daily cycle rather than all of it. Their daily
    median is therefore not an estimate of the same quantity as every other
    day's.

    Measured, on a 21-day window offset by twelve hours, sampled at 120 s, with
    a 1.6-wide diurnal cycle and 0.02 of noise:

        whole days      4.011, 3.994, 4.003, 4.000, 3.989 ...  (spread 0.025)
        first day       3.432
        last day        4.560

    An edge offset of 0.565 against a real day-to-day spread of 0.025 -- 22x --
    placed at the two ends of the series, which is exactly where Theil-Sen and
    Kendall's tau are most sensitive. It produced DRIFT on clean sensors, and it
    split a genuine noise burst across two half-days and diluted it below the
    detection multiple so the burst was MISSED. Both failure directions at once,
    from the same cause.

    Judged on TIME COVERED, not on sample count, and the difference is the whole
    correctness of this function. Those are two unrelated properties:

      * a PARTIAL day is short in coverage -- it starts or ends mid-cycle, so
        its median is biased by which hours it happens to contain;
      * a THIN day is sparse in samples but spans the full 24 hours, so its
        median is unbiased. This feed reports at anything from 16 s to 300 s,
        and 57% of its sensors are report-by-exception, so thin days are normal
        and healthy.

    A count-based rule confuses the two in both directions. Tried first at 0.6 of
    the median day's COUNT, it kept a 07:13-to-midnight first day -- 504 samples
    against a typical 720, which passes a count test at 70% while covering only
    16.8 of 24 hours -- and the daily medians still spanned 0.39 instead of 0.025.

    Comparison is against the TYPICAL day's span rather than against 24 hours, so
    a sensor that reports once a day (every span zero, every day alike) keeps all
    of its days instead of losing all of them.
    """
    frame = pd.DataFrame({"ts": _as_ts(ts), "value": values}).dropna()
    if frame.empty:
        return None
    day = frame["ts"].dt.normalize()
    # `agg(["min", "max"])` rather than a lambda over each group: the lambda
    # form is a Python call per calendar day per sensor, and this function runs
    # twice for every sensor in the fleet.
    bounds = frame.groupby(day)["ts"].agg(["min", "max"])
    spans = bounds["max"] - bounds["min"]
    if spans.empty:
        return None
    typical = spans.median()
    keep = spans[spans >= WHOLE_DAY_FRACTION * typical].index
    if len(keep) == 0:
        return None
    return frame[day.isin(keep)].groupby(day)


def _daily_medians(ts: pd.Series, values: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    """
    One median per whole calendar day.

    Collapsing to daily medians is what makes drift measurable: it removes the
    daily cycle completely rather than trying to model it, which is the whole
    reason a 72-hour slope is meaningless and a 28-day one is not. Partial days
    are excluded -- see `_whole_days`, where leaving them in produced DRIFT on
    sensors that were not drifting.
    """
    grouped = _whole_days(ts, values)
    if grouped is None:
        return np.array([]), np.array([])
    medians = grouped["value"].median().dropna()
    if medians.empty:
        return np.array([]), np.array([])
    days = (medians.index - medians.index[0]).days.to_numpy(dtype=float)
    return days, medians.to_numpy(dtype=float)


def _theil_sen(x: np.ndarray, y: np.ndarray) -> float:
    """
    Median of pairwise slopes. Robust to a third of the points being wrong,
    which matters when a couple of the daily medians land on outage days.
    """
    n = len(x)
    if n < 3:
        return float("nan")
    slopes = []
    for i in range(n - 1):
        dx = x[i + 1:] - x[i]
        valid = dx != 0
        if valid.any():
            slopes.append((y[i + 1:][valid] - y[i]) / dx[valid])
    if not slopes:
        return float("nan")
    return float(np.median(np.concatenate(slopes)))


def _kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    """Rank concordance, so a consistent trend is told from a random walk."""
    n = len(x)
    if n < 3:
        return 0.0
    concordant = discordant = 0
    for i in range(n - 1):
        dx = np.sign(x[i + 1:] - x[i])
        dy = np.sign(y[i + 1:] - y[i])
        product = dx * dy
        concordant += int(np.sum(product > 0))
        discordant += int(np.sum(product < 0))
    total = concordant + discordant
    return 0.0 if total == 0 else (concordant - discordant) / total


def detect_drift(sensor_key: str, frame: pd.DataFrame, *,
                 ts_col: str = "ts", value_col: str = "value",
                 unit: str = "") -> list[Signal]:
    """
    A slow, consistent shift in level over weeks — calibration walking away.

    Two conditions, both required. The slope must be **large enough to matter**
    relative to this sensor's own day-to-day variation, and it must be
    **consistent**, which Kendall's tau over the daily medians tests. Without
    tau, any sensor whose level wandered would register a slope: a random walk
    has a perfectly good least-squares trend and means nothing.

    Reports %/day as well as units/day, because "0.06 bar per day" is only
    interpretable next to the sensor's level, and drift is the one finding most
    likely to be read by someone scheduling calibration rather than responding
    to an alarm.
    """
    df = frame[[ts_col, value_col]].dropna()
    days, medians = _daily_medians(df[ts_col], df[value_col].to_numpy(dtype=float))
    if days.size < MIN_DAYS_DRIFT:
        return []

    slope = _theil_sen(days, medians)
    if not np.isfinite(slope) or slope == 0:
        return []

    tau = _kendall_tau(days, medians)
    if abs(tau) < DRIFT_MIN_TAU:
        return []               # wandered, but did not trend

    # Day-to-day variation is the yardstick: drift is only meaningful against
    # how much this sensor moves anyway.
    day_to_day = float(np.median(np.abs(np.diff(medians))))
    level = float(np.median(medians))
    reference = max(day_to_day, abs(level) * 1e-6, 1e-12)
    if abs(slope) < DRIFT_MIN_FRACTION_PER_DAY * reference:
        return []

    span_days = float(days[-1] - days[0])
    total_change = slope * span_days
    percent_per_day = (100.0 * slope / abs(level)) if level else None

    ts = _as_ts(df[ts_col])
    return [Signal(
        type=AnomalyType.DRIFT,
        start=ts.min().to_pydatetime(),
        end=ts.max().to_pydatetime(),
        detector=DETECTOR,
        magnitude=round(total_change, 6),
        unit=unit,
        n_points=int(days.size),
        detail={
            "slope_per_day": round(slope, 8),
            "percent_per_day": (None if percent_per_day is None
                                else round(percent_per_day, 4)),
            "total_change": round(total_change, 6),
            "days_observed": int(days.size),
            "kendall_tau": round(tau, 3),
            "day_to_day_variation": round(day_to_day, 6),
            "level": round(level, 6),
        },
    )]


def detect_noise_burst(sensor_key: str, frame: pd.DataFrame, *,
                       ts_col: str = "ts", value_col: str = "value",
                       unit: str = "") -> list[Signal]:
    """
    The sensor has become much noisier than it normally is.

    Usually a failing transducer, a loose connection or an earthing problem —
    and, like quantisation collapse, it arrives *before* the readings become
    implausible, which is what makes it worth catching.

    Each day's within-day spread is compared against the median of the
    **preceding** days only. Comparing against all days including itself is the
    self-contamination trap that has now bitten four detectors in this project;
    for this one it would be fatal rather than merely weakening, because the
    burst is the noise being measured.

    Known limit: the unit is a CALENDAR day, so a burst that lasts about one day
    and straddles midnight is split between two days, each half-diluted with
    quiet readings, and can fall under the multiple. Measured on a fixture where
    a 30x burst straddles midnight exactly: missed. Accepted rather than fixed
    with a sliding window, because the realistic fault -- a failing transducer or
    a loose earth -- does not become noisy for exactly 24 hours and then stop; it
    stays noisy, and is caught on its first whole day. A sliding window would
    cost four times the work on a once-a-day job to cover a case that needs the
    fault to end on schedule.
    """
    df = frame[[ts_col, value_col]].dropna()
    if df.empty:
        return []
    ts = _as_ts(df[ts_col])
    values = df[value_col].to_numpy(dtype=float)

    # Whole days only, for the same reason DRIFT needs them -- and here the
    # consequence runs the other way. A 21-day window offset by twelve hours
    # splits the burst day across two calendar dates, so each half carries half
    # the burst's samples diluted with half a day of quiet readings, and a real
    # 30x noise burst came in under the 4x multiple and was MISSED. The
    # `sample.size < 10` floor below cannot catch that: a half day on this feed
    # is about 360 samples.
    grouped = _whole_days(ts, values)
    if grouped is None:
        return []

    spreads: list[tuple[pd.Timestamp, float, int]] = []
    for date, index in grouped.groups.items():
        sample = values[df.index.get_indexer(index)]
        sample = sample[np.isfinite(sample)]
        if sample.size < 10:
            continue
        # Noise is measured from CONSECUTIVE DIFFERENCES, not from the spread
        # about the day's median. The spread about the median is dominated by
        # the daily cycle -- on a sensor with a +-1.2 bar diurnal swing and
        # 0.05 bar of noise it reads ~0.85, and multiplying the noise tenfold
        # barely moves it, so a real burst was invisible. A slow cycle cancels
        # almost completely between consecutive samples while noise does not,
        # which is the same reason the level-shift detector uses this scale.
        diffs = np.diff(sample)
        diffs = diffs[np.isfinite(diffs)]
        if diffs.size < 8:
            continue
        sigma = (1.4826 * float(np.median(np.abs(diffs - np.median(diffs))))
                 / np.sqrt(2.0))
        spreads.append((date, sigma, int(sample.size)))

    if len(spreads) < MIN_DAYS_NOISE:
        return []

    spreads.sort(key=lambda row: row[0])
    signals: list[Signal] = []
    for i in range(MIN_DAYS_NOISE - 1, len(spreads)):
        date, spread, n = spreads[i]
        prior = [s for _, s, _ in spreads[:i] if s > 0]
        if len(prior) < MIN_DAYS_NOISE - 1:
            continue
        normal = float(np.median(prior))
        if normal <= 0 or spread < NOISE_BURST_MULTIPLE * normal:
            continue
        signals.append(Signal(
            type=AnomalyType.NOISE_BURST,
            start=date.to_pydatetime(),
            end=(date + pd.Timedelta(days=1)).to_pydatetime(),
            detector=DETECTOR,
            magnitude=round(spread, 6),
            unit=unit,
            n_points=n,
            detail={
                "spread": round(spread, 6),
                "normal_spread": round(normal, 6),
                "ratio": round(spread / normal, 1),
                "days_of_history": i,
            },
        ))
    return signals


@dataclass
class ProfileJobResult:
    baselines: dict[str, TimeOfDayBaseline] = field(default_factory=dict)
    signals: dict[str, list[Signal]] = field(default_factory=dict)
    skipped_short_history: int = 0

    def absorb(self, other: "ProfileJobResult") -> "ProfileJobResult":
        """
        Fold another batch's result into this one.

        The daily job reads history a few hundred sensors at a time -- see
        `store.iter_history` for why -- so the whole-fleet answer is assembled
        from batches. Sensor keys never repeat across batches, so this is a
        plain union rather than a merge with precedence rules.
        """
        self.baselines.update(other.baselines)
        self.signals.update(other.signals)
        self.skipped_short_history += other.skipped_short_history
        return self

    def summary(self) -> dict[str, object]:
        by_type: dict[str, int] = {}
        for signals in self.signals.values():
            for signal in signals:
                by_type[signal.type.value] = by_type.get(signal.type.value, 0) + 1
        return {
            "sensors": len(self.baselines),
            "usable_baselines": sum(1 for b in self.baselines.values() if b.usable),
            "sensors_with_findings": len(self.signals),
            "by_type": by_type,
            "skipped_short_history": self.skipped_short_history,
        }


def run_profile_job(readings: pd.DataFrame, *, key_col: str = "sensor_key",
                    ts_col: str = "ts", value_col: str = "value",
                    units: dict[str, str] | None = None) -> ProfileJobResult:
    """
    The whole daily job over a long-history frame.

    Intended to be driven once a day from the reading store. It is separate
    from the hourly pipeline on purpose: these answers change slowly, cost far
    more to compute, and would otherwise be recomputed every hour to say the
    same thing.
    """
    units = units or {}
    result = ProfileJobResult()

    for key, group in readings.groupby(key_col, sort=False):
        key = str(key)
        unit = units.get(key, "")
        result.baselines[key] = build_baseline(key, group, ts_col=ts_col,
                                               value_col=value_col)

        days = _as_ts(group[ts_col]).dt.normalize().nunique()
        if days < MIN_DAYS_NOISE:
            result.skipped_short_history += 1
            continue

        signals: list[Signal] = []
        signals += detect_drift(key, group, ts_col=ts_col, value_col=value_col,
                                unit=unit)
        signals += detect_noise_burst(key, group, ts_col=ts_col,
                                      value_col=value_col, unit=unit)
        if signals:
            result.signals[key] = signals

    return result
