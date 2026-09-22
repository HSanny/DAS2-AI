"""
das2.detect.baseline
====================

Score the current window against what this sensor normally reads *at this time
of day*.

This is the detector that replaces the incumbent's robust-Z, and the
difference is the baseline it compares against. Robust-Z compared each point to
a rolling median of its own neighbours, which means it could only ever see
excursions **shorter than its window**. A sensor sitting 30% low for six hours
looked perfectly normal to it, because the rolling median went with it.

Comparing against a stored time-of-day profile instead makes the sensor's own
past the control, so a sustained departure stays visible for as long as it
lasts, and the daily demand cycle is subtracted rather than mistaken for signal.

Three specific defects of the incumbent that this avoids
--------------------------------------------------------
**It could not see a quiet sensor.** `robust_z` did `mad.replace(0, nan)` then
`fillna(0.0)`, so any window with zero MAD scored exactly 0. Reproduced: 300
zeros followed by a spike to 250.0 scored `max|RZ| = 0.000`. Here the scale is
**floored** — by the stored MAD, the sensor's quantisation step, and an
absolute minimum — so a quiet sensor is not divided by nothing.

**It had no null hypothesis.** `calibrate_flags_per_sensor` took
`max(2.5, q99.7, 4.5)`, and the quantile always won, so the threshold was
really "flag the top 0.3% of points" on every sensor, healthy or not. Measured
on pure Gaussian noise, `P(|RZ| > 4.5)` was 5.95% — four orders of magnitude
above the nominal tail. Here the threshold is a fixed multiple of a stored
spread, so a healthy sensor genuinely produces nothing.

**Its scale factor was applied to the wrong side.** `1.4826` multiplied the
numerator instead of scaling the MAD, inflating every score by 1.4826² ≈ 2.198.
Corrected here, which is why these thresholds are not the incumbent's.

Timing tolerance, and why it is needed
---------------------------------------
A pump that starts twenty minutes late is behaving normally, but against a
fixed time-of-day profile it produces an enormous residual for those twenty
minutes. So each point is scored against the **best** matching bucket within a
tolerance band, which absorbs duty-cycle jitter without blunting a real
sustained departure — a fault does not become acceptable by shifting half an
hour.

Abstains without a baseline
---------------------------
If the profile job has not yet run, or has too little history for this sensor,
this returns nothing. That is the correct behaviour: the whole point is to
compare against a trustworthy past, and inventing one from the current window
would reproduce the rolling-median blindness this module exists to remove.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from das2.models import AnomalyType, Signal
from das2.profile.build import TimeOfDayBaseline
from das2.timeutils import to_epoch_seconds

DETECTOR = "baseline"

#: Residual threshold, in robust sigma of the stored spread. Because the scale
#: is a stored MAD rather than a local one, this is a real tail probability and
#: not a per-sensor quantile in disguise.
RESIDUAL_SIGMA = 6.0

#: A departure must last this long. Shorter excursions belong to `detect_spike`,
#: which is better at them and reports them as what they are.
MIN_DURATION_S = 900.0

#: Tolerance band for duty-cycle timing jitter. A point is scored against the
#: closest-matching bucket within this window either side.
TIMING_TOLERANCE_MIN = 30

#: Floor on the scale, as a fraction of the sensor's level. Stops a sensor whose
#: stored MAD is near zero -- a very steady one -- from flagging on rounding.
MIN_SCALE_FRACTION = 1e-4

#: Minimum points before a window is worth scoring at all.
MIN_POINTS = 12


def _expected(baseline: TimeOfDayBaseline, when: pd.Timestamp,
              tolerance_min: int) -> tuple[float, float] | None:
    """
    Expected value and spread, taking the most forgiving bucket in the band.

    "Most forgiving" means the bucket whose median is closest to nothing in
    particular -- it is resolved per point in `score_window`, because which
    bucket is kindest depends on the observed value. Here the candidates are
    gathered; the choice is made there.
    """
    return baseline.lookup(when.to_pydatetime())


def score_window(ts: pd.Series, values: np.ndarray,
                 baseline: TimeOfDayBaseline | None, *,
                 unit: str = "",
                 resolution: float = 0.0,
                 tolerance_min: int = TIMING_TOLERANCE_MIN
                 ) -> list[Signal]:
    """
    Sustained departures from the stored time-of-day profile.

    Returns one signal per departure, with the deviation reported in
    **engineering units** — "running 1.4 bar below normal for this hour" — which
    an engineer can check against the instrument. The old `Peak_RZ` could not be
    compared between two sensors at all, because each sensor's score was set by
    its own quantisation step.
    """
    if baseline is None or not baseline.usable:
        return []
    if len(values) < MIN_POINTS:
        return []

    times = pd.to_datetime(pd.Series(ts).reset_index(drop=True))
    values = np.asarray(values, dtype=float)
    seconds = to_epoch_seconds(times)

    residual = np.full(len(values), np.nan)
    scale = np.full(len(values), np.nan)

    # Candidate offsets within the tolerance band, in minutes. A point counts as
    # normal if ANY of them explains it, which is what makes duty-cycle jitter
    # survivable.
    offsets = range(-tolerance_min, tolerance_min + 1, 15)

    for i, (when, value) in enumerate(zip(times, values)):
        if not np.isfinite(value):
            continue
        best_abs = None
        for offset in offsets:
            entry = baseline.lookup((when + timedelta(minutes=offset)).to_pydatetime())
            if entry is None:
                continue
            median, mad = entry
            sigma = max(1.4826 * mad, float(resolution),
                        abs(median) * MIN_SCALE_FRACTION, 1e-12)
            deviation = value - median
            if best_abs is None or abs(deviation) < abs(best_abs[0]):
                best_abs = (deviation, sigma)
        if best_abs is not None:
            residual[i], scale[i] = best_abs

    scored = np.isfinite(residual) & np.isfinite(scale) & (scale > 0)
    if scored.sum() < MIN_POINTS:
        return []

    z = np.zeros(len(values))
    z[scored] = residual[scored] / scale[scored]
    flagged = scored & (np.abs(z) >= RESIDUAL_SIGMA)
    if not flagged.any():
        return []

    idx = np.flatnonzero(flagged)
    splits = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[splits + 1]]
    ends = np.r_[idx[splits], idx[-1]]

    signals: list[Signal] = []
    for i, j in zip(starts.tolist(), ends.tolist()):
        duration = seconds[j] - seconds[i]
        if duration < MIN_DURATION_S:
            continue        # a spike, and detect_spike describes it better
        window = residual[i:j + 1]
        peak_index = int(np.nanargmax(np.abs(window)))
        peak = float(window[peak_index])
        signals.append(Signal(
            type=AnomalyType.RESIDUAL_OUTLIER,
            start=times.iloc[i].to_pydatetime(),
            end=times.iloc[j].to_pydatetime(),
            detector=DETECTOR,
            magnitude=round(peak, 6),
            unit=unit,
            n_points=j - i + 1,
            detail={
                "direction": "above" if peak > 0 else "below",
                "peak_sigma": round(float(np.nanmax(np.abs(z[i:j + 1]))), 1),
                "median_deviation": round(float(np.nanmedian(window)), 6),
                "baseline_days": baseline.days_observed,
                "duration_h": round(duration / 3600.0, 2),
            },
        ))
    return signals


def baseline_summary(baselines: dict[str, TimeOfDayBaseline]) -> dict[str, object]:
    """
    Coverage of the stored profiles, for the run log.

    Worth watching: if `usable` is low the L2 layer is silently doing nothing,
    and the system looks quiet for a reason that has nothing to do with the
    sensors.
    """
    if not baselines:
        return {"sensors": 0, "usable": 0}
    usable = [b for b in baselines.values() if b.usable]
    return {
        "sensors": len(baselines),
        "usable": len(usable),
        "usable_pct": round(100.0 * len(usable) / len(baselines), 1),
        "median_days": (round(float(np.median([b.days_observed
                                               for b in baselines.values()])), 1)),
    }
