"""
Tests for the sensor-health detectors (Phase 4a).

The headline test is the false-positive one. Measured on two real hours of the
live feed, 57% of actively-reporting sensors never change value, so an absolute
"unchanged for N minutes" rule would raise ~1,400 alerts per run and be switched
off within a day. Everything here exists to make sure that does not happen while
still catching genuinely frozen instruments.

Run:  python3 tests/test_health.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from das2.detect.health import run_health_checks  # noqa: E402
from das2.detect.profile import build_profile, profile_summary  # noqa: E402
from das2.models import AnomalyType  # noqa: E402

T0 = datetime(2026, 9, 18, 0, 0, 0)


def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
    if not cond:
        raise SystemExit(1)


def series(values, dt_s=120, start=T0):
    ts = pd.Series([start + timedelta(seconds=i * dt_s) for i in range(len(values))])
    return ts, np.asarray(values, dtype=float)


def frame(ts, values):
    return pd.DataFrame({"ts": ts, "value": values})


def types(signals):
    return [s.type for s in signals]


def main():
    rng = np.random.default_rng(11)
    # 3 days at 120s, matching the real scan rate.
    N = 3 * 24 * 30

    # ------------------------------------------------------------------ #
    print("THE headline case: a permanently constant sensor is NOT a fault")
    # This is 57% of the real fleet. An absolute flatline rule fires on all.
    ts, vals = series(np.full(N, 3.90))
    prof = build_profile("const", frame(ts, vals))
    check("profile recognises it as static", prof.is_static,
          f"(change rate {prof.change_rate_per_hour}/hour)")
    sig = run_health_checks(ts, vals, prof, range_min=0, range_max=20, unit="bar")
    check("no FLATLINE raised", AnomalyType.FLATLINE not in types(sig),
          "(1,392 real analog sensors look exactly like this)")
    check("no alerts of any kind", sig == [], f"({types(sig)})")

    print("\na sensor that normally MOVES and then freezes IS a fault")
    moving = 45.0 + rng.normal(0, 1.2, N)
    ts, vals = series(moving)
    prof = build_profile("moving", frame(ts, vals))
    check("profile recognises it as active", not prof.is_static,
          f"(change rate {prof.change_rate_per_hour:.0f}/hour)")

    frozen = moving.copy()
    frozen[1500:1900] = frozen[1499]        # ~13 hours stuck
    ts, vals = series(frozen)
    sig = run_health_checks(ts, vals, prof, range_min=0, range_max=2000, unit="L/s")
    flat = [s for s in sig if s.type is AnomalyType.FLATLINE]
    check("FLATLINE raised", len(flat) == 1, f"({len(flat)})")
    check("it spans the frozen stretch", flat[0].magnitude > 40000,
          f"({flat[0].magnitude/3600:.1f} hours)")
    check("evidence names the normal behaviour",
          "normal_change_rate_per_hour" in flat[0].detail,
          "(so an operator can check the reasoning)")

    print("\nwithout a profile the detector ABSTAINS rather than guessing")
    ts, vals = series(np.full(200, 3.90))
    check("no profile -> no flatline",
          AnomalyType.FLATLINE not in types(run_health_checks(ts, vals, None)))
    thin = build_profile("thin", frame(*series(np.full(20, 3.90))))
    check("thin profile is not trusted", not thin.is_well_observed)
    check("thin profile -> no flatline",
          AnomalyType.FLATLINE not in types(run_health_checks(ts, vals, thin)),
          "(cannot distinguish 'never moves' from 'not watched long enough')")

    # ------------------------------------------------------------------ #
    print("\nSTALE is judged against the sensor's OWN cadence")
    fast = 3.9 + rng.normal(0, 0.05, N)                      # 120s scan
    ts_fast, v_fast = series(fast, dt_s=120)
    p_fast = build_profile("fast", frame(ts_fast, v_fast))
    slow_ts, slow_v = series(3.9 + rng.normal(0, 0.05, 300), dt_s=3600)  # hourly
    p_slow = build_profile("slow", frame(slow_ts, slow_v))

    # An hour of silence: stale for the 120s sensor, normal for the hourly one.
    gap_ts = pd.Series(list(ts_fast[:100]) + [ts_fast.iloc[99] + timedelta(hours=2)])
    gap_v = np.append(v_fast[:100], 3.9)
    check("hour-long gap IS stale for a 120s sensor",
          AnomalyType.STALE in types(run_health_checks(gap_ts, gap_v, p_fast)))
    check("the same gap is NOT stale for an hourly sensor",
          AnomalyType.STALE not in types(run_health_checks(gap_ts, gap_v, p_slow)),
          "(a global timeout would page for every slow-scanning sensor)")

    print("\nstill-silent-at-window-end is caught")
    sig = run_health_checks(ts_fast[:100], v_fast[:100], p_fast,
                            window_end=ts_fast.iloc[99] + timedelta(hours=6))
    ongoing = [s for s in sig if s.type is AnomalyType.STALE
               and s.detail.get("ongoing")]
    check("ongoing silence raised", len(ongoing) == 1,
          "(a gap scan alone misses it -- there is no closing row)")

    print("\nFLATLINE and STALE stay distinct")
    # Frozen value but still reporting -> flatline, not stale.
    ts, vals = series(np.concatenate([moving[:500], np.full(400, moving[499])]))
    sig = run_health_checks(ts, vals, prof)
    check("frozen-but-reporting is FLATLINE only",
          AnomalyType.FLATLINE in types(sig) and AnomalyType.STALE not in types(sig),
          "('stuck at a plausible value' != 'RTU offline')")

    # ------------------------------------------------------------------ #
    print("\nRANGE_VIOLATION: the idle-flowmeter regression")
    # Real sensor MRRS-THOMSON has Median_Value 0.001302; raw `value<0` flagged
    # 1079 points and 542 events from one healthy meter.
    idle = rng.normal(0.0, 0.002, N)
    ts, vals = series(idle)
    p_idle = build_profile("idle", frame(ts, vals))
    naive = int((vals < 0).sum())
    sig = run_health_checks(ts, vals, p_idle, range_min=0, range_max=2000,
                            unit="L/s", is_flow=True)
    check("idle meter raises nothing",
          not [s for s in sig if s.type is AnomalyType.RANGE_VIOLATION],
          f"(a raw value<0 test would flag {naive:,} of {N:,} points)")
    check("and no spurious reverse flow either",
          AnomalyType.REVERSE_FLOW not in types(sig))

    print("\n...while genuine violations still fire")
    v = 4.1 + rng.normal(0, 0.06, N); v[800:805] = -5.0
    ts, vals = series(v)
    p = build_profile("p", frame(ts, vals))
    viol = [s for s in run_health_checks(ts, vals, p, range_min=0, range_max=20,
                                         unit="bar")
            if s.type is AnomalyType.RANGE_VIOLATION]
    check("-5 bar on a pressure sensor is caught", len(viol) == 1)
    check("magnitude in engineering units", viol[0].unit == "bar" and viol[0].magnitude > 4,
          f"({viol[0].magnitude:.1f} bar past the bound)")

    v = 28.0 + rng.normal(0, 0.2, N); v[100] = 150.0
    ts, vals = series(v)
    check("150C on a temperature sensor is caught",
          any(s.type is AnomalyType.RANGE_VIOLATION
              for s in run_health_checks(ts, vals, build_profile("t", frame(ts, vals)),
                                         range_min=0, range_max=60, unit="C")))

    print("\nsustained reverse flow is a typed fault, not something to discard")
    v = 45.0 + rng.normal(0, 1.0, N); v[600:640] = -30.0
    ts, vals = series(v)
    sig = run_health_checks(ts, vals, build_profile("f", frame(ts, vals)),
                            range_min=-2000, range_max=2000, unit="L/s", is_flow=True)
    check("REVERSE_FLOW raised", AnomalyType.REVERSE_FLOW in types(sig))

    print("\nSPIKE survives the timestamp defects in this feed")
    # 7% of real sensors have duplicate timestamps, 12% out-of-order rows.
    v = 3.9 + rng.normal(0, 0.05, 200)
    ts = pd.Series([T0 + timedelta(seconds=i * 120) for i in range(200)])
    ts.iloc[50] = ts.iloc[49]                     # duplicate -> dt == 0
    ts.iloc[80], ts.iloc[81] = ts.iloc[81], ts.iloc[80]   # out of order -> dt < 0
    sig = run_health_checks(ts, v, build_profile("s", frame(ts, v)))
    spikes = [s for s in sig if s.type is AnomalyType.SPIKE]
    check("no infinite-rate spikes from bad intervals", len(spikes) == 0,
          f"({len(spikes)} -- a naive dv/dt would fire on both)")

    v[120] = 50.0
    sig = run_health_checks(ts, v, build_profile("s2", frame(ts, v)), unit="bar")
    check("a real step change IS spiked",
          any(s.type is AnomalyType.SPIKE for s in sig))

    # ------------------------------------------------------------------ #
    print("\ncounters and config points are skipped entirely")
    kwh = np.cumsum(np.abs(rng.normal(2, 0.4, N)))     # monotonic
    ts, vals = series(kwh)
    p_kwh = build_profile("kwh", frame(ts, vals))
    check("a kWh counter raises nothing",
          run_health_checks(ts, vals, p_kwh, equipment_kind="counter") == [],
          "(flat = plant idle, drop = rollover; neither is a fault)")
    ts, vals = series(np.full(N, 12.5))
    check("a setpoint raises nothing",
          run_health_checks(ts, vals, build_profile("sp", frame(ts, vals)),
                            equipment_kind="config") == [],
          "(its value is an operator decision)")

    print("\nfleet summary reports the static share")
    profiles = {
        "a": build_profile("a", frame(*series(np.full(N, 3.9)))),
        "b": build_profile("b", frame(*series(np.full(N, 1.0)))),
        "c": build_profile("c", frame(*series(45 + rng.normal(0, 1.2, N)))),
    }
    s = profile_summary(profiles)
    check("static share computed", s["static"] == 2 and s["static_pct"] > 60,
          f"({s['static_pct']}% static)")

    print("\ndegenerate input does not raise")
    for name, (t, v) in [("empty", (pd.Series([], dtype="datetime64[ns]"), np.array([]))),
                         ("single", series([1.0])),
                         ("two points", series([1.0, 2.0]))]:
        check(f"{name}", isinstance(run_health_checks(t, v, None), list))

    print("\nAll health-detector tests passed.")


if __name__ == "__main__":
    main()
