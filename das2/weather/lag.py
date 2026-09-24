"""
das2.weather.lag
================

How long after the rain does this sensor respond?

The defect this exists for
--------------------------
Rainfall was integrated over an incident's OWN window. A level shift's window
is the few minutes around the step, so a 60 mm/h storm was credited with
whatever fell inside those minutes -- often one five-minute gauge interval, or
none at all. The rule as written asked *"was it raining at the exact instant
the level moved?"*, and water does not work that way. Rain falls on a
catchment, runs off, and arrives at the sensor later.

Why the lag is learned rather than looked up
--------------------------------------------
The obvious fix is a constant. It is also the documented way to get this
badly wrong. Gericke & Smithers (2014), *Hydrological Sciences Journal* 59(11),
reviewed catchment-response-time methods worldwide and concluded that applying
empirical formulas outside their development region must be avoided --
underestimating the time parameter by 80% can overestimate peak discharge by
200%. Every such formula also needs catchment area, slope and flow-path
length, none of which this system has.

So the lag is measured from the data, per sensor, by cross-correlation. That
is a standard, validated technique -- Talei & Chua (2012), *Journal of
Hydrology* 438-439, use exactly it to set the lag for event-based
rainfall-runoff models -- and it needs only the two time series we already
have. It also produces something more valuable than the lag: **which gauge
actually drives this sensor**, learned from behaviour rather than distance.
That is a catchment assignment derived from data, and this system has no
catchment map.

PUB's own *Code of Practice on Surface Water Drainage* gives 5-30 minutes as
the design time of concentration for urban catchments. That is used here as a
sanity prior on the SEARCH WINDOW only -- never as the answer -- because it is
a design convention measured at a design point, not the lag between a gauge
three kilometres away and a canal level sensor.

What is correlated against what
-------------------------------
Rainfall intensity against the level's **rate of change**, not against the
level itself. Rain drives a rise; correlating it against absolute level would
lock onto the level's own slow trend and report the lag of whatever else was
happening that day. `d(level)/dt` is the physically matched pairing.

When it declines to answer
--------------------------
Most sensor-gauge pairs in any window are noise: it did not rain, or the
sensor did not move, or both. A lag returned from noise is worse than no lag,
because everything downstream would treat it as measured. So a result is only
`confident` when there was real rain, the sensor really moved, the peak
correlation clears a floor, and that peak stands clear of the rest of the lag
profile. Anything short of all four returns a lag that reports itself as not
confident, and the caller falls back to a stated default.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger("das2.weather.lag")

#: Grid the two series are put on before correlating. Matches the gauges' own
#: 300 s scan, so a gauge bucket is never split across two grid cells.
BUCKET_S = 300.0

#: How far to look. PUB's Code of Practice puts urban time of concentration at
#: 5-30 minutes; this is deliberately wider, because that figure is measured at
#: a design point while we are measuring from a gauge that may be kilometres
#: from the catchment that drains to the sensor.
MIN_LAG_S = 0.0
MAX_LAG_S = 7200.0

#: A window with less rain than this has nothing to correlate against.
MIN_WET_BUCKETS = 3
MIN_WINDOW_RAIN_MM = 2.0

#: And a sensor that did not move tells us nothing about when it moves.
MIN_RESPONSE_BUCKETS = 3

#: Peak correlation below this is not a response, it is coincidence.
MIN_CORRELATION = 0.35

#: The peak must stand this far above the median of the whole lag profile.
#: Without it, a pair that correlates at 0.4 for EVERY lag -- which is what a
#: shared daily cycle looks like -- would report whichever lag happened to be
#: a hair higher, with full confidence.
MIN_PEAK_PROMINENCE = 0.12


@dataclass(frozen=True)
class RainLag:
    """How long this sensor takes to answer the rain, and how sure we are."""

    seconds: float
    correlation: float
    gauge_key: str = ""
    wet_buckets: int = 0
    confident: bool = False
    reason: str = ""

    @property
    def minutes(self) -> float:
        return self.seconds / 60.0

    def describe(self) -> str:
        if not self.confident:
            return f"lag not measurable ({self.reason})"
        return (f"{self.minutes:.0f} min, learned from gauge "
                f"{self.gauge_key} (r={self.correlation:.2f})")


def _grid(ts: pd.Series, values: np.ndarray, start, end,
          bucket_s: float) -> np.ndarray:
    """
    Put an irregular series on a regular grid, summing within each bucket.

    Summing is right for rainfall (increments) and for the absolute rate of
    change of a level, which is what both sides of this correlation are.
    """
    n = max(1, int((end - start).total_seconds() // bucket_s) + 1)
    out = np.zeros(n, dtype=float)
    if len(ts) == 0:
        return out
    offsets = (pd.to_datetime(ts) - pd.Timestamp(start)).dt.total_seconds()
    index = (offsets // bucket_s).to_numpy()
    ok = np.isfinite(index) & (index >= 0) & (index < n) & np.isfinite(values)
    np.add.at(out, index[ok].astype(int), values[ok])
    return out


def learn_lag(level_ts: pd.Series, level_values: np.ndarray,
              gauge_ts: pd.Series, gauge_mm: np.ndarray,
              start, end, *, gauge_key: str = "",
              bucket_s: float = BUCKET_S,
              max_lag_s: float = MAX_LAG_S) -> RainLag:
    """
    Cross-correlate rainfall against the level's rate of rise.

    Returns the lag of maximum correlation, with a `confident` flag that is
    the whole point: a lag recovered from a dry window or a still sensor is
    noise, and marking it as such is what stops everything downstream from
    treating it as measured.
    """
    rain = _grid(gauge_ts, gauge_mm, start, end, bucket_s)
    if rain.size < 8:
        return RainLag(0.0, 0.0, gauge_key, reason="window too short")

    wet = int(np.count_nonzero(rain > 0.05))
    if wet < MIN_WET_BUCKETS or float(rain.sum()) < MIN_WINDOW_RAIN_MM:
        return RainLag(0.0, 0.0, gauge_key, wet_buckets=wet,
                       reason=f"only {rain.sum():.1f} mm in the window")

    # The sensor's RESPONSE is its rate of change, not its value. Rain makes a
    # level rise; correlating against the level itself would find the lag of
    # whatever slow trend the day happened to have.
    level = _grid(level_ts, np.asarray(level_values, dtype=float),
                  start, end, bucket_s)
    # A bucket with no reading holds 0 from `_grid`, which would read as a
    # genuine value of zero. Carry the last known level across gaps instead.
    seen = _grid(level_ts, np.ones(len(level_ts)), start, end, bucket_s)
    held = np.where(seen > 0, np.divide(level, np.maximum(seen, 1)), np.nan)
    held = pd.Series(held).ffill().bfill().to_numpy()
    response = np.diff(held, prepend=held[0])

    moved = int(np.count_nonzero(np.abs(response) > 0))
    if moved < MIN_RESPONSE_BUCKETS:
        return RainLag(0.0, 0.0, gauge_key, wet_buckets=wet,
                       reason="the sensor did not move")

    lags = np.arange(int(MIN_LAG_S // bucket_s),
                     int(max_lag_s // bucket_s) + 1)
    profile = np.full(lags.size, np.nan)
    for i, lag in enumerate(lags):
        if lag >= rain.size - 4:
            break
        a = rain[:rain.size - lag] if lag else rain
        b = response[lag:]
        size = min(a.size, b.size)
        a, b = a[:size], b[:size]
        if a.std() <= 0 or b.std() <= 0:
            continue
        profile[i] = float(np.corrcoef(a, b)[0, 1])

    if not np.any(np.isfinite(profile)):
        return RainLag(0.0, 0.0, gauge_key, wet_buckets=wet,
                       reason="no usable overlap")

    best = int(np.nanargmax(profile))
    peak = float(profile[best])
    baseline = float(np.nanmedian(profile))
    seconds = float(lags[best] * bucket_s)

    if peak < MIN_CORRELATION:
        return RainLag(seconds, peak, gauge_key, wet_buckets=wet,
                       reason=f"weak response (r={peak:.2f})")
    if peak - baseline < MIN_PEAK_PROMINENCE:
        # Correlating equally at every lag is what a shared daily cycle looks
        # like. Reporting the argmax of that is reporting the argmax of noise.
        return RainLag(seconds, peak, gauge_key, wet_buckets=wet,
                       reason="no clear peak; correlates at every lag alike")

    return RainLag(seconds, peak, gauge_key, wet_buckets=wet, confident=True)


def consensus(lags: list[RainLag]) -> RainLag | None:
    """
    One lag from several sensor-gauge pairs in the same incident.

    The median of the confident ones, attributed to the pair whose correlation
    was strongest. The median rather than the best single pair: one sensor
    correlating at 0.9 with a gauge by coincidence should not set the lag for
    the whole incident, and several pairs agreeing is the actual evidence.
    """
    good = [lag for lag in lags if lag.confident]
    if not good:
        return None
    seconds = float(np.median([lag.seconds for lag in good]))
    strongest = max(good, key=lambda lag: lag.correlation)
    return RainLag(
        seconds=seconds,
        correlation=strongest.correlation,
        gauge_key=strongest.gauge_key,
        wet_buckets=max(lag.wet_buckets for lag in good),
        confident=True,
        reason=f"{len(good)} of {len(lags)} pairs agreed",
    )
