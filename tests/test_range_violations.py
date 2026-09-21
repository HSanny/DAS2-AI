"""
Tests for range-violation alerting and its deadband (Phase 0.5).

Background: the combine step used to SUBTRACT physically impossible readings
(`voted & ~Rule_Based_Invalid`), so a pressure sensor reporting -5 bar was
discarded rather than alerted. Phase 0.5 alerts on them instead.

That change is only safe with a deadband. The fleet-wide bounds test raw
values, so an idle flowmeter sitting at zero with symmetric noise reads
negative about half the time and every one of those samples looks
"physically impossible". A smoke run before the deadband existed produced
1079 flagged points and 542 events from a single healthy idle meter.

Run:  python3 tests/test_range_violations.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docker_ready"))
from equipment_anomaly_pipeline_visual import (  # noqa: E402
    EQUIPMENT_RANGES,
    range_tolerance,
    range_violation_mask,
    resolution_estimate,
)


def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
    if not cond:
        raise SystemExit(1)


def flag(equipment, values):
    tol = range_tolerance(values, resolution_estimate(values))
    return range_violation_mask(equipment, values, tolerance=tol)


def main():
    rng = np.random.default_rng(7)

    print("healthy sensors must not produce violations")
    idle_flow = rng.normal(0.0, 0.001, 2160)       # MRRS-THOMSON: Median_Value 0.001302
    naive = int(((idle_flow < 0) | (idle_flow > 2000)).sum())
    check("idle flowmeter at zero raises none",
          flag("Flowrate", idle_flow).sum() == 0,
          f"(naive bound would flag {naive} of {len(idle_flow)})")

    for name, equip, v in [
        ("pressure at 3.9 bar", "Pressure", rng.normal(3.9, 0.05, 2160)),
        ("voltage on a 414.8V bus", "Voltage", rng.normal(414.8, 0.05, 2160)),
        ("DO at 5.3 mg/L", "Dissolved Oxygen", rng.normal(5.3, 0.08, 2160)),
        ("temperature at 28C", "Temperature", rng.normal(28.0, 0.2, 2160)),
        ("level at 0% (empty tank)", "LevelSensor", rng.normal(0.0, 0.05, 2160)),
        ("level at 100% (full tank)", "LevelSensor", rng.normal(100.0, 0.05, 2160)),
    ]:
        check(f"{name} raises none", flag(equip, v).sum() == 0,
              f"({flag(equip, v).sum()} flagged)")

    print("\ngenuine violations still fire")
    v = rng.normal(4.1, 0.06, 2160); v[500:505] = -5.0
    m = flag("Pressure", v)
    check("-5 bar on a pressure sensor", m.sum() == 5, f"({m.sum()} flagged)")
    check("only the impossible samples", set(np.flatnonzero(m)) == set(range(500, 505)))

    v = rng.normal(28.0, 0.2, 2160); v[100] = 150.0
    check("150 C on a temperature sensor", flag("Temperature", v).sum() == 1)

    v = rng.normal(45.0, 1.2, 2160); v[300:310] = -40.0
    check("sustained reverse flow well beyond noise", flag("Flowrate", v).sum() == 10,
          "(reverse flow is a real fault, not noise)")

    v = rng.normal(414.8, 0.05, 2160); v[7] = 900.0
    check("900 V on a 500 V-max bus", flag("Voltage", v).sum() == 1)

    print("\nthe deadband scales with the sensor, not the fleet")
    quiet = rng.normal(4.0, 0.01, 2160)
    noisy = rng.normal(4.0, 0.80, 2160)
    check("a noisy sensor earns a wider deadband",
          range_tolerance(noisy) > range_tolerance(quiet) * 10,
          f"({range_tolerance(quiet):.4f} vs {range_tolerance(noisy):.4f})")
    check("deadband is never negative", range_tolerance(np.zeros(100)) >= 0)

    print("\nunknown equipment is never flagged")
    check("no bounds defined -> no violations",
          flag("Vibration", rng.normal(2.0, 0.5, 100)).sum() == 0,
          "(a class without a range must not be judged against one)")
    check("Vibration genuinely has no range yet", "Vibration" not in EQUIPMENT_RANGES)

    print("\ndegenerate inputs")
    for name, v in [("empty", np.array([])),
                    ("all NaN", np.full(50, np.nan)),
                    ("single value", np.array([3.0])),
                    ("all identical", np.full(50, 4.0))]:
        m = range_violation_mask("Pressure", v, tolerance=range_tolerance(v))
        check(f"{name}: returns an aligned mask without raising", len(m) == len(v))

    print("\nAll range-violation tests passed.")


if __name__ == "__main__":
    main()
