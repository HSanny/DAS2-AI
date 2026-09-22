"""
Tests for time-based detector windows (Phase 0.2).

The defect: every window and gate was counted in SAMPLES, but this feed is
report-by-exception and sensors report at very different rates. Over the same
72h window, ROLL_WIN_Z = 24 was a 6.3-minute baseline on the Dissolved Oxygen
sensor and a 49.8-minute baseline on the Kranji flow meter. MIN_EVENT_LEN = 6
is worse because it is a hard filter: a 2-minute glitch alerted on one sensor
and was silently discarded on the other.

Run:  python3 tests/test_time_windows.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "docker_ready"))
sys.path.insert(0, str(REPO))          # das2, for the scheduler assertions below
import equipment_anomaly_pipeline_visual as det  # noqa: E402


def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
    if not cond:
        raise SystemExit(1)


def stamps(n, dt_s):
    return pd.date_range("2026-01-01", periods=n, freq=f"{dt_s}s").to_series().reset_index(drop=True)


def main():
    # Real rates from abnormal_sensor_backup.csv over a 72h window.
    FAST_DT, SLOW_DT = 15.7, 124.4      # DO sensor vs Kranji1PS flow
    fast = stamps(2000, 16)             # ~16 s
    slow = stamps(2000, 124)            # ~124 s

    print("the defect being fixed")
    check("one sample-count constant meant an 8x different duration",
          abs((24 * SLOW_DT) / (24 * FAST_DT) - 7.9) < 0.3,
          f"({24*FAST_DT/60:.1f} min vs {24*SLOW_DT/60:.1f} min)")

    print("\nrate measurement")
    check("fast sensor rate recovered", abs(det.median_interval_seconds(fast) - 16) < 0.1)
    check("slow sensor rate recovered", abs(det.median_interval_seconds(slow) - 124) < 0.1)

    dirty = pd.to_datetime(pd.Series([
        "2026-01-01 00:00:00", "2026-01-01 00:02:00",
        "2026-01-01 00:02:00",            # duplicate  -> dt == 0
        "2026-01-01 00:01:00",            # backwards  -> dt < 0
        "2026-01-01 00:04:00", "2026-01-01 00:06:00",
    ]))
    check("duplicate and backwards timestamps do not drag the rate to zero",
          det.median_interval_seconds(dirty) > 0,
          f"(got {det.median_interval_seconds(dirty):.0f}s)")
    check("unmeasurable rate returns NaN",
          np.isnan(det.median_interval_seconds(pd.Series([], dtype="datetime64[ns]"))))

    print("\nwindows now mean the same DURATION on every sensor")
    wf = det.compute_sensor_windows(fast)
    ws = det.compute_sensor_windows(slow)
    for key, secs in [("roll_win_z", det.ROLL_WIN_Z_SEC),
                      ("min_event_len", det.MIN_EVENT_SEC),
                      ("cooldown", det.COOLDOWN_SEC)]:
        fast_min = wf[key] * 16 / 60
        slow_min = ws[key] * 124 / 60
        check(f"{key}: same wall-clock span on both sensors",
              abs(fast_min - slow_min) < 0.15 * max(fast_min, slow_min),
              f"({fast_min:.0f} min vs {slow_min:.0f} min, target {secs/60:.0f} min)")

    check("fast sensor gets MORE samples per window", wf["roll_win_z"] > ws["roll_win_z"],
          f"({wf['roll_win_z']} vs {ws['roll_win_z']})")

    print("\nbehaviour preserved for the typical (~120s) sensor")
    typical = det.compute_sensor_windows(stamps(2000, 120))
    for key, legacy in [("roll_win_z", det.ROLL_WIN_Z), ("iso_win", det.ISO_WIN),
                        ("min_event_len", det.MIN_EVENT_LEN), ("merge_gap", det.MERGE_GAP),
                        ("cooldown", det.COOLDOWN), ("step_window", det.STEP_WINDOW_SAMPLES),
                        ("step_window_min", det.STEP_WINDOW_MIN)]:
        check(f"{key} matches the legacy constant", typical[key] == legacy,
              f"({typical[key]} vs {legacy})")

    print("\nclamping and fallback")
    absurd = det.compute_sensor_windows(stamps(2000, 1))       # 1 s sampling
    check("very fast sensor is clamped, not unbounded",
          absurd["roll_win_z"] <= det.WINDOW_MAX_SAMPLES, f"({absurd['roll_win_z']})")
    sparse = det.compute_sensor_windows(stamps(2000, 86400))   # daily
    check("very slow sensor keeps a usable minimum",
          sparse["min_event_len"] >= det.WINDOW_MIN_SAMPLES, f"({sparse['min_event_len']})")

    single = det.compute_sensor_windows(pd.Series(pd.to_datetime(["2026-01-01"])))
    check("unmeasurable rate falls back to the legacy constants",
          single["roll_win_z"] == det.ROLL_WIN_Z and single["min_event_len"] == det.MIN_EVENT_LEN,
          "(sensor still analysed, not skipped)")

    print("\nrollback switch (the shadow-comparison control arm)")
    det.USE_TIME_BASED_WINDOWS = False
    try:
        legacy = det.compute_sensor_windows(fast)
        check("USE_TIME_BASED_WINDOWS=0 restores sample-counted behaviour",
              legacy["roll_win_z"] == det.ROLL_WIN_Z
              and legacy["min_event_len"] == det.MIN_EVENT_LEN
              and legacy["cooldown"] == det.COOLDOWN)
    finally:
        det.USE_TIME_BASED_WINDOWS = True

    print("\nthe scheduler runs on the clock, not on its own start time")
    # The historian writes the hour's HISTORY file at HH:00:00 and HISTCURR at
    # HH:00:02. Sleeping a fixed 60 minutes from whenever the container started
    # drifts to an arbitrary phase, and a run landing at HH:00:0x reads a file
    # still being written -- which is how the first deployment died, with
    # EmptyDataError: No columns to parse from file.
    from datetime import datetime as _dt
    from das2.cli import next_run_at

    check("an hourly run lands at 5 past, whenever it started",
          next_run_at(_dt(2026, 9, 22, 17, 48, 51), 60, 5) == _dt(2026, 9, 22, 18, 5),
          "(started 17:48 -> 18:05, not 18:48)")
    check("a restart does not shift the schedule",
          next_run_at(_dt(2026, 9, 22, 17, 3, 12), 60, 5)
          == next_run_at(_dt(2026, 9, 22, 17, 4, 59), 60, 5) == _dt(2026, 9, 22, 17, 5))
    check("the slot just reached is not repeated",
          next_run_at(_dt(2026, 9, 22, 18, 5, 0), 60, 5) == _dt(2026, 9, 22, 19, 5),
          "(landing exactly on it moves to the next, never sleeps 0)")
    check("it rolls over midnight",
          next_run_at(_dt(2026, 9, 22, 23, 50), 60, 5) == _dt(2026, 9, 23, 0, 5))
    check("a sub-hourly interval keeps the offset",
          next_run_at(_dt(2026, 9, 22, 17, 48), 15, 5) == _dt(2026, 9, 22, 17, 50),
          "(:05 :20 :35 :50)")
    check("offset 0 means on the hour",
          next_run_at(_dt(2026, 9, 22, 17, 48), 30, 0) == _dt(2026, 9, 22, 18, 0))
    check("the next slot is always in the future",
          all(next_run_at(_dt(2026, 9, 22, 17, m, sec), 60, 5)
              > _dt(2026, 9, 22, 17, m, sec)
              for m in range(0, 60, 7) for sec in (0, 59)),
          "(a non-positive sleep would spin the loop)")

    print("\nAll time-window tests passed.")


if __name__ == "__main__":
    main()
