"""
das2.detect.profile
===================

Per-sensor behavioural profiles, learned from history.

Why this module exists before any detector
-------------------------------------------
Measured on two real hours of the live feed: of 2,602 sensors reporting at least
ten readings, **1,481 never changed value at all** (1,392 of them analog), and
the median sensor had **100% of consecutive readings identical**. The historian
re-reports the same number every 120-second scan.

That single fact invalidates the obvious stuck-sensor rule. "Value unchanged for
30 minutes" describes a frozen instrument *and* it describes 57% of the healthy
fleet, so an absolute rule would raise roughly 1,400 alerts per run and be
switched off within a day.

The only formulation that survives is comparative:

    "This sensor normally changes 40 times a day. It has not changed in three
     days."   -> broken

    "This sensor has not changed once in the 28 days I have watched it."
                                                  -> that is simply what it does

which requires knowing what normal looks like for each sensor individually.
That is what a `SensorProfile` holds. The client confirmed the SQL `data` table
has months of readings, so profiles can be built immediately rather than after a
month of accumulation.

Profiles are deliberately cheap to compute and small to store: a handful of
scalars per sensor, not a model. They can be rebuilt by a daily job from
server-side aggregates, which matters because months x 2,672 sensors x 120 s is
on the order of 10^8 rows -- far too many to pull into pandas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd

from das2.timeutils import to_epoch_seconds


@dataclass(frozen=True)
class SensorProfile:
    """
    What normal looks like for one sensor.

    `change_rate_per_hour` is the load-bearing field: it separates a sensor that
    is supposed to move from one that never does.
    """

    sensor_key: str
    n_readings: int = 0
    span_hours: float = 0.0

    # --- variability ---
    change_rate_per_hour: float = 0.0   # value changes per hour, historically
    distinct_ratio: float = 0.0         # distinct values / readings
    longest_flat_seconds: float = 0.0   # longest historical unchanged run

    # --- level and spread ---
    median: float = float("nan")
    mad: float = float("nan")
    resolution: float = 0.0             # smallest real step the sensor resolves
    value_min: float = float("nan")
    value_max: float = float("nan")

    # --- reporting cadence ---
    median_interval_s: float = float("nan")
    p99_interval_s: float = float("nan")

    @property
    def is_static(self) -> bool:
        """
        True when this sensor has essentially never changed.

        These are the 57%. A flatline detector must stay silent on them, because
        for these sensors a flat line IS the normal behaviour.
        """
        return self.change_rate_per_hour < STATIC_CHANGE_RATE

    @property
    def is_well_observed(self) -> bool:
        """
        Enough history to trust a comparison against it.

        Below this the profile cannot distinguish "does not move" from "has not
        been watched long enough", so detectors that depend on it must abstain
        rather than guess.
        """
        return self.n_readings >= MIN_READINGS and self.span_hours >= MIN_SPAN_HOURS

    @property
    def expected_flat_seconds(self) -> float:
        """
        How long this sensor can normally sit unchanged.

        Derived from its own longest historical flat run, with a floor so a
        briefly-observed sensor is not held to an implausibly tight standard.
        """
        return max(self.longest_flat_seconds, MIN_EXPECTED_FLAT_S)


#: Below this many value-changes per hour a sensor counts as static. 0.2/hour is
#: roughly "changes less than once every five hours".
STATIC_CHANGE_RATE = 0.2

#: Minimum history before a profile is trusted.
MIN_READINGS = 50
MIN_SPAN_HOURS = 12.0

#: Floor on the expected-flat duration, so a thinly-observed sensor is not
#: flagged the moment it exceeds a short observed run.
MIN_EXPECTED_FLAT_S = 3600.0


def _longest_flat_run_seconds(ts: np.ndarray, values: np.ndarray) -> float:
    """Longest stretch, in seconds, over which the value did not change."""
    if len(values) < 2:
        return 0.0
    changed = np.flatnonzero(np.diff(values) != 0)
    if changed.size == 0:
        return float(ts[-1] - ts[0])           # never changed at all
    # Boundaries: start -> first change -> ... -> last change -> end
    edges = np.concatenate(([0], changed + 1, [len(ts) - 1]))
    return float(np.max(np.diff(ts[edges]))) if edges.size > 1 else 0.0


def _resolution(values: np.ndarray, min_active_frac: float = 0.05,
                tol: float = 0.05) -> float:
    """
    Smallest step the sensor actually resolves, or 0.0 when it is not quantised.

    Tests for quantisation rather than assuming it: for a quantised signal every
    change is a whole multiple of the step, so a candidate step must divide the
    other changes. On continuous data a low percentile of the differences is a
    substantial fraction of the real spread, and using it as a scale floor would
    suppress genuine outliers on healthy sensors.
    """
    d = np.abs(np.diff(values))
    d = d[np.isfinite(d)]
    if d.size == 0:
        return 0.0
    nz = d[d > 0]
    if nz.size < 10 or nz.size / d.size < min_active_frac:
        return 0.0
    for candidate in (np.min(nz), np.percentile(nz, 1), np.percentile(nz, 5)):
        if not np.isfinite(candidate) or candidate <= 0:
            continue
        ratios = nz / candidate
        if np.median(np.abs(ratios - np.round(ratios))) < tol:
            return float(candidate)
    return 0.0


def build_profile(sensor_key: str, frame: pd.DataFrame,
                  *, ts_col: str = "ts", value_col: str = "value") -> SensorProfile:
    """
    Summarise one sensor's history.

    `frame` is that sensor's readings. Only positive time intervals count:
    duplicate timestamps (7% of sensors) and out-of-order rows (12%) both occur
    in this feed and would otherwise drag the cadence estimate toward zero.
    """
    df = frame[[ts_col, value_col]].dropna().sort_values(ts_col)
    if df.empty:
        return SensorProfile(sensor_key=sensor_key)

    values = df[value_col].to_numpy(dtype=float)
    ts = to_epoch_seconds(df[ts_col])

    n = len(values)
    span_s = float(ts[-1] - ts[0]) if n > 1 else 0.0
    span_h = span_s / 3600.0

    intervals = np.diff(ts)
    intervals = intervals[intervals > 0]

    n_changes = int(np.count_nonzero(np.diff(values))) if n > 1 else 0
    median = float(np.median(values))

    return SensorProfile(
        sensor_key=sensor_key,
        n_readings=n,
        span_hours=round(span_h, 3),
        change_rate_per_hour=round(n_changes / span_h, 4) if span_h > 0 else 0.0,
        distinct_ratio=round(len(np.unique(values)) / n, 4),
        longest_flat_seconds=_longest_flat_run_seconds(ts, values),
        median=median,
        mad=float(np.median(np.abs(values - median))),
        resolution=_resolution(values),
        value_min=float(np.min(values)),
        value_max=float(np.max(values)),
        median_interval_s=float(np.median(intervals)) if intervals.size else float("nan"),
        p99_interval_s=float(np.percentile(intervals, 99)) if intervals.size else float("nan"),
    )


def build_profiles(readings: pd.DataFrame, *, key_col: str = "sensor_key",
                   ts_col: str = "ts", value_col: str = "value"
                   ) -> dict[str, SensorProfile]:
    """Profiles for every sensor in a readings frame."""
    return {
        key: build_profile(key, grp, ts_col=ts_col, value_col=value_col)
        for key, grp in readings.groupby(key_col, sort=False)
    }


def profile_summary(profiles: dict[str, SensorProfile]) -> dict[str, object]:
    """
    Fleet-level summary, for logging each time profiles are rebuilt.

    The static share is the number to watch: if it is near zero the profiles are
    probably built over too short a window to be meaningful, and any detector
    relying on them will misbehave.
    """
    if not profiles:
        return {"sensors": 0}
    values = list(profiles.values())
    static = [p for p in values if p.is_static]
    observed = [p for p in values if p.is_well_observed]
    rates = sorted(p.change_rate_per_hour for p in values)
    return {
        "sensors": len(values),
        "well_observed": len(observed),
        "static": len(static),
        "static_pct": round(100.0 * len(static) / len(values), 1),
        "median_change_rate_per_hour": rates[len(rates) // 2],
        "median_span_hours": round(
            float(np.median([p.span_hours for p in values])), 1),
    }
