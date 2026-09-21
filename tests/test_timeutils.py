"""
Tests for das2.timeutils (Phase 4a).

This exists because the same bug has now been introduced twice in this project:

    pd.to_datetime(s).astype("int64") / 1e9

The int64 representation carries whatever resolution pandas inferred, so on a
millisecond column every duration comes out 1000x too small. Nothing raises.

First occurrence made every rate-of-change 1000x too large. Second made a
72-hour window measure as 0.072 hours, so no sensor ever had "enough history"
and the flatline detector silently abstained on the entire fleet -- a detector
that appears to work and finds nothing.

The only reliable defence is asserting a KNOWN duration, which is what these do.

Run:  python3 tests/test_timeutils.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from das2.timeutils import intervals_seconds, span_seconds, to_epoch_seconds  # noqa: E402


def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
    if not cond:
        raise SystemExit(1)


def main():
    print("known durations come back exactly right")
    # 72 hours at 120s -- the real scan rate over the real analysis window.
    n, step = 2160, 120
    base = datetime(2026, 9, 18)
    ts = pd.Series([base + timedelta(seconds=i * step) for i in range(n)])
    span = span_seconds(ts)
    check("72h window measures as 72 hours", abs(span - 259_080) < 1,
          f"({span/3600:.3f} h -- the bug made this 0.072 h)")
    check("intervals are 120 seconds",
          np.allclose(intervals_seconds(ts), 120.0),
          f"(median {np.median(intervals_seconds(ts))}s)")

    print("\nthe bug it defends against, demonstrated")
    buggy = pd.to_datetime(ts).astype("int64").to_numpy() / 1e9
    buggy_span = float(buggy.max() - buggy.min())
    ratio = span / buggy_span if buggy_span else float("inf")
    check("the naive idiom is wrong on this data", abs(ratio - 1.0) > 1e-6,
          f"(off by {ratio:.0f}x -- and nothing raises)")

    print("\nresolution-independent across pandas dtypes")
    for unit in ("s", "ms", "us", "ns"):
        col = pd.Series(pd.to_datetime(ts).values.astype(f"datetime64[{unit}]"))
        check(f"datetime64[{unit}] gives the same span",
              abs(span_seconds(col) - 259_080) < 1,
              f"({span_seconds(col)/3600:.3f} h)")

    print("\nepoch conversion")
    check("the epoch itself is zero",
          to_epoch_seconds(pd.Series([pd.Timestamp("1970-01-01")]))[0] == 0.0)
    one_day = to_epoch_seconds(pd.Series([pd.Timestamp("1970-01-02")]))[0]
    check("one day after the epoch is 86400", one_day == 86400.0)
    check("a real historian timestamp round-trips",
          to_epoch_seconds(pd.Series([pd.Timestamp("2026-09-20 21:00:00")]))[0]
          == pd.Timestamp("2026-09-20 21:00:00").timestamp())

    print("\ndefective intervals are excluded, as the real feed requires")
    # 7% of real sensors have duplicate timestamps, 12% out-of-order rows.
    dirty = pd.to_datetime(pd.Series([
        "2026-09-20 21:00:00",
        "2026-09-20 21:02:00",
        "2026-09-20 21:02:00",   # duplicate   -> dt == 0
        "2026-09-20 21:01:00",   # out of order -> dt < 0
        "2026-09-20 21:04:00",
    ]))
    gaps = intervals_seconds(dirty)
    check("zero and negative gaps dropped", (gaps > 0).all(), f"({gaps})")
    check("the real cadence survives", abs(np.median(gaps) - 120.0) < 60,
          f"(median {np.median(gaps)}s)")
    check("keeping them would drag the estimate down",
          np.median(intervals_seconds(dirty, positive_only=False)) < np.median(gaps))

    print("\ndegenerate input")
    check("empty series", span_seconds(pd.Series([], dtype="datetime64[ns]")) == 0.0)
    check("single timestamp", span_seconds(pd.Series([pd.Timestamp("2026-01-01")])) == 0.0)
    unparseable = to_epoch_seconds(pd.Series(["2026-09-20 21:00:00", "garbage", None]))
    check("unparseable entries become NaN, not an exception",
          np.isfinite(unparseable).sum() == 1)
    check("a non-reset index does not misalign",
          abs(span_seconds(ts.iloc[100:200]) - 99 * 120) < 1)

    print("\nAll timeutils tests passed.")


if __name__ == "__main__":
    main()
