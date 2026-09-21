"""
Tests for the rate-of-change channel that replaced sliding DTW (Phase 0.4).

Two things are pinned here:

  1. The DTW channel it replaces was provably NOT a shape comparison. This
     suite re-derives that identity, so the justification for the removal lives
     in the test suite rather than only in a commit message.
  2. The replacement must not reproduce DTW's echo artefact, and must survive
     the duplicate / out-of-order timestamps this report-by-exception feed
     actually contains.

Run:  python3 tests/test_roc_channel.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docker_ready"))
from equipment_anomaly_pipeline_visual import roc_sliding  # noqa: E402


def exact_dtw(a, b):
    """Reference DTW with an L1 cost, used only to prove the legacy identity."""
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = abs(a[i - 1] - b[j - 1]) + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[n, m]


def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
    if not cond:
        raise SystemExit(1)


def times(n, step_s=60):
    return pd.date_range("2026-01-01", periods=n, freq=f"{step_s}s")


def main():
    rng = np.random.default_rng(0)
    w = 24

    print("the removed DTW channel was algebraically a two-term difference")
    for name, v in [("noise", rng.normal(0, 1, 300)),
                    ("with spike", None),
                    ("duty cycle", np.repeat([0.0, 1.0], 150))]:
        if v is None:
            v = rng.normal(0, 1, 300)
            v[100] += 40
        got = np.array([exact_dtw(v[i - w:i], v[i - w + 1:i + 1]) for i in range(w, len(v))])
        pred = np.array([abs(v[i - w] - v[i - w + 1]) + abs(v[i - 1] - v[i])
                         for i in range(w, len(v))])
        check(f"{name}: DTW == |dv(i)| + |dv(i-w)| exactly",
              np.allclose(got, pred, atol=1e-12),
              f"(max diff {np.abs(got - pred).max():.2e})")

    print("\n...and it echoed every transient w samples later")
    v = rng.normal(0, 1, 300)
    v[100] += 40
    dtw = np.array([exact_dtw(v[i - w:i], v[i - w + 1:i + 1]) for i in range(w, len(v))])
    top = sorted((np.argsort(-dtw)[:4] + w).tolist())
    check("legacy DTW peaks at both the spike AND spike+w",
          100 in top and (100 + w) in top, f"(peaks at {top})")

    rate, flags = roc_sliding(v, times(len(v)))
    roc_top = sorted(np.argsort(-np.nan_to_num(rate))[:2].tolist())
    check("replacement peaks only at the spike, no echo",
          all(abs(i - 100) <= 1 for i in roc_top), f"(peaks at {roc_top})")

    print("\nduplicate / out-of-order timestamps (present in this feed)")
    v = np.array([1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0], dtype=float)
    ts = pd.to_datetime([
        "2026-01-01 00:00:00", "2026-01-01 00:01:00",
        "2026-01-01 00:01:00",                        # duplicate -> dt == 0
        "2026-01-01 00:02:00", "2026-01-01 00:03:00",
        "2026-01-01 00:02:30",                        # goes backwards -> dt < 0
        "2026-01-01 00:04:00", "2026-01-01 00:05:00",
        "2026-01-01 00:06:00", "2026-01-01 00:07:00",
    ])
    rate, flags = roc_sliding(v, ts)
    check("no infinities from a zero interval", not np.any(np.isinf(rate)))
    check("duplicate timestamp yields NaN, not a spike", np.isnan(rate[2]))
    check("backwards timestamp yields NaN, not a spike", np.isnan(rate[5]))
    check("no point is flagged purely from a bad interval", not flags[2] and not flags[5])

    print("\nrate is in engineering units per second")
    v = np.array([0.0, 60.0, 120.0])
    rate, _ = roc_sliding(v, times(3, step_s=60))
    check("60 units over 60s reads as 1.0/s", np.allclose(rate[1:], [1.0, 1.0]),
          f"(got {rate[1:]})")

    print("\na real step is flagged, a steady ramp is not")
    steady = np.arange(200, dtype=float)
    rate, flags = roc_sliding(steady, times(200))
    check("constant-slope ramp raises no flags", not flags.any())

    stepped = np.concatenate([np.zeros(100), np.full(100, 50.0)])
    rate, flags = roc_sliding(stepped, times(200))
    check("a sudden step is flagged", flags.any())
    check("only the step transition is flagged", flags.sum() == 1,
          f"(flagged {flags.sum()} points)")

    print("\ndegenerate inputs")
    for name, (v, n) in [("empty", (np.array([]), 0)),
                         ("single point", (np.array([3.0]), 1)),
                         ("all NaN", (np.full(20, np.nan), 20)),
                         ("all identical", (np.full(20, 7.0), 20))]:
        rate, flags = roc_sliding(v, times(n))
        check(f"{name}: returns aligned arrays without raising",
              len(rate) == n and len(flags) == n)

    print("\nAll rate-of-change tests passed.")


if __name__ == "__main__":
    main()
