"""
das2.timeutils
==============

Unit-safe conversion from timestamps to epoch seconds.

This module exists because the obvious idiom is wrong, and wrong silently.

    pd.to_datetime(s).astype("int64") / 1e9        # DO NOT

The int64 representation carries whatever resolution pandas inferred for the
column -- seconds, milliseconds, microseconds or nanoseconds. Dividing by 1e9
assumes nanoseconds, so on a millisecond-resolution column every duration comes
out **1000x too small**, and on a second-resolution column 10^9 times too small.

Nothing raises. The numbers just quietly become nonsense, and every downstream
threshold expressed in seconds stops meaning anything. This bug has been
introduced twice in this project:

  * in the rate-of-change channel, where it made every |dv/dt| 1000x too large;
  * in the sensor profiles, where a 72-hour window measured as 0.072 hours, so
    no sensor ever had enough history and the flatline detector silently
    abstained on the entire fleet.

Both were caught by a test asserting a known duration, which is the only
reliable defence: the failure is invisible to inspection.

`.dt.total_seconds()` on a timedelta is exact and resolution-independent, so
that is what is used here, once, in one place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPOCH = pd.Timestamp("1970-01-01")


def to_epoch_seconds(timestamps) -> np.ndarray:
    """
    Timestamps -> float seconds since the Unix epoch.

    Resolution-independent. Accepts anything pd.to_datetime accepts. Unparseable
    entries become NaN rather than raising, so one malformed row cannot discard
    a whole sensor.
    """
    ts = pd.to_datetime(pd.Series(timestamps).reset_index(drop=True), errors="coerce")
    return (ts - EPOCH).dt.total_seconds().to_numpy(dtype=float)


def intervals_seconds(timestamps, *, positive_only: bool = True) -> np.ndarray:
    """
    Gaps between consecutive timestamps, in seconds.

    By default only positive intervals are returned. This feed contains
    duplicate timestamps (7% of sensors) and out-of-order rows (12%), which
    produce zero and negative gaps; including them drags any cadence estimate
    toward zero and turns a naive dv/dt into an infinite rate.
    """
    seconds = to_epoch_seconds(timestamps)
    gaps = np.diff(seconds)
    if positive_only:
        gaps = gaps[np.isfinite(gaps) & (gaps > 0)]
    return gaps


def span_seconds(timestamps) -> float:
    """Total span covered by a set of timestamps, in seconds."""
    seconds = to_epoch_seconds(timestamps)
    seconds = seconds[np.isfinite(seconds)]
    return float(seconds.max() - seconds.min()) if seconds.size > 1 else 0.0
