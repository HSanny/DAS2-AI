"""
Regression tests for the robust-Z scale floor (Phase 0.3).

Pins the two properties that matter:
  1. Excursions on quiet / quantised sensors are now visible (they scored
     exactly 0 before the floor was added).
  2. Behaviour on healthy noisy sensors is UNCHANGED -- the floor must not
     bind there, or the fix would buy sensitivity by raising the false-alarm
     rate, which is the opposite of what is wanted.

Run:  python3 tests/test_robust_z_floor.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docker_ready"))
from equipment_anomaly_pipeline_visual import (  # noqa: E402
    MAD_Z_THRESH,
    RZ_SATURATE,
    resolution_estimate,
    robust_z,
)


def legacy_robust_z(series, window=24):
    """The pre-fix implementation, kept so the tests can show the delta."""
    s = pd.Series(series).astype(float)
    med = s.rolling(window, min_periods=1, center=True).median()
    mad = (s - med).abs().rolling(window, min_periods=1, center=True).median()
    mad = mad.replace(0, np.nan)
    return (1.4826 * (s - med) / mad).fillna(0.0)


def peak(rz):
    return float(np.abs(np.asarray(rz)).max())


def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
    if not cond:
        raise SystemExit(1)


def main():
    rng = np.random.default_rng(1)

    print("previously-invisible excursions are now detected")
    cases = {}
    v = np.zeros(300); v[150] = 250.0
    cases["idle flowmeter + spike to 250"] = v
    v = np.round(rng.normal(414.8, 0.05, 300), 1); v[150] = 380.0
    cases["0.1-quantised 414.8V bus + drop to 380V"] = v
    v = np.full(300, 3.90); v[100:140] = 0.0
    cases["constant 3.90 bar + 40-sample dropout to 0"] = v

    for label, series in cases.items():
        before = peak(legacy_robust_z(series))
        after = peak(robust_z(series)[0])
        check(f"{label}", after > MAD_Z_THRESH,
              f"(was {before:.3f} -> now {after:.2f})")

    print("\nhealthy noisy sensors are unaffected (no new false alarms)")
    for i in range(5):
        noise = rng.normal(0, 1, 2100)
        before = legacy_robust_z(noise)
        after = robust_z(noise)[0]
        # The floor must not bind: scores stay essentially identical.
        agree = np.allclose(before, np.clip(after, -RZ_SATURATE, RZ_SATURATE), atol=1e-9)
        check(f"pure-noise series {i} scores unchanged", agree)

    # Both rates must be measured on the SAME draws, or the comparison is just
    # sampling noise between two different random series.
    series = [rng.normal(0, 1, 2100) for _ in range(10)]
    rate_before = np.mean([np.mean(np.abs(legacy_robust_z(x)) > MAD_Z_THRESH) for x in series])
    rate_after = np.mean([np.mean(np.abs(robust_z(x)[0]) > MAD_Z_THRESH) for x in series])
    check("noise-only flag rate not increased", rate_after <= rate_before + 1e-6,
          f"(before {rate_before:.4%}, after {rate_after:.4%})")

    print("\nscores stay bounded")
    v = np.round(rng.normal(414.8, 0.05, 300), 1); v[150] = 380.0
    check("no absurd magnitudes from dividing by a tiny floor",
          peak(robust_z(v)[0]) <= RZ_SATURATE, f"(peak {peak(robust_z(v)[0]):.1f})")

    print("\nresolution estimator")
    q = np.round(rng.normal(100, 0.5, 500), 1)
    est = resolution_estimate(q)
    check("recovers a 0.1 quantisation step", abs(est - 0.1) < 1e-9, f"(got {est})")

    static = np.zeros(300); static[150] = 250.0
    check("refuses to infer resolution from a near-static series",
          resolution_estimate(static) == 0.0,
          "(otherwise the spike would be its own scale and vanish)")

    check("flat series yields no resolution", resolution_estimate(np.zeros(100)) == 0.0)
    check("handles a one-element series", resolution_estimate(np.array([1.0])) == 0.0)

    print("\ndegenerate inputs do not raise")
    for name, series in [
        ("all NaN", np.full(50, np.nan)),
        ("single value", np.array([5.0])),
        ("empty", np.array([])),
        ("with NaNs", np.concatenate([rng.normal(0, 1, 50), np.full(10, np.nan)])),
        ("all zeros", np.zeros(50)),
        ("huge magnitude", np.full(50, 1e12)),
    ]:
        rz, _, _ = robust_z(series)
        check(f"{name}", np.all(np.isfinite(np.asarray(rz, dtype=float))) or len(series) == 0)

    print("\nAll robust-Z floor tests passed.")


if __name__ == "__main__":
    main()
