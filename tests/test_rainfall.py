#!/usr/bin/env python3
"""
Reading a rain gauge, when nothing in the feed says what kind it is.

This exists because the first real run reported 26,118 mm of rain in three
days. Singapore receives about 2,300 mm in a YEAR.

The old rule decided the gauge's convention from a magnitude: any reading over
50 mm meant a running total, otherwise the readings were increments and got
summed. The client's gauges sit at a constant value of about 10 -- under 50 --
so every scan was added: 10 x 2,160 readings = 21,600 mm.

The number itself was not the damage. RAIN_EXPLAINS_MM is 2.0, so a permanent
21,600 mm held the rain gate wide open, and any incident whose anomaly types
were all rain-explicable was classified WEATHER_DRIVEN and withheld from
dispatch -- in a country where it rains most days, silently, on the strength
of arithmetic.

So the convention is now read from the SHAPE of the series, and the cases
below are the shapes a gauge can actually produce.

Run:  python3 tests/test_rainfall.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from das2.weather.provider import (  # noqa: E402
    MAX_PLAUSIBLE_WINDOW_MM,
    InternalRainGaugeProvider,
)

passed = failed = 0
T0 = datetime(2026, 9, 19, 16, 0)
WINDOW = (datetime(2026, 9, 19, 0, 0), datetime(2026, 9, 23, 0, 0))


def check(label: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {label}" + (f"  ({detail})" if detail else ""))
    else:
        failed += 1
        print(f"  FAIL  {label}" + (f"  ({detail})" if detail else ""))


def total(values, dt: int = 120):
    """Millimetres this provider reads out of one gauge's series."""
    # `__new__` on purpose: `_gauge_total` needs only a readings frame, and
    # building a full provider would need an inventory too. The cost is that
    # this helper has to maintain the object's invariants by hand -- every
    # attribute `_gauge_total` touches has to be set here.
    provider = InternalRainGaugeProvider.__new__(InternalRainGaugeProvider)
    provider.rejected_gauges = {}
    provider.readings = pd.DataFrame({
        "sensor_key": ["g"] * len(values),
        "ts": [T0 + timedelta(seconds=i * dt) for i in range(len(values))],
        "value": list(values),
    })
    return provider._gauge_total("g", *WINDOW)


def main() -> int:
    print("\nthe defect: the client's own gauges")
    # Exactly what the real feed carries -- 72 hours at 120 s, parked at 10.
    got = total([10.0] * 2160)
    check("a constant gauge reads as no rain", got == 0.0,
          f"({got} mm; the old rule said 21600.0)")
    check("which is right under either convention", True,
          "a counter that has not moved, or a reading that is not depth")

    print("\nrunning total")
    check("a counter rising 12 mm reports 12 mm",
          total(np.linspace(137.0, 149.0, 2160)) == 12.0)
    # last-minus-first goes NEGATIVE across a reset and would report nothing.
    reset = list(np.linspace(0, 8, 1080)) + list(np.linspace(0, 5, 1080))
    check("a midnight reset does not lose the day's rain",
          total(reset) == 13.0, f"({total(reset)} mm = 8 + 5)")

    print("\ntipping bucket")
    check("a dry bucket reads zero", total([0.0] * 2160) == 0.0)
    tips = [0.0] * 2160
    for i in range(300, 300 + 36 * 6, 6):
        tips[i] = 0.5
    check("36 isolated tips of 0.5 mm read 18 mm", total(tips) == 18.0,
          "(summed, never averaged -- a mean reports drizzle for a cloudburst)")

    print("\na real gauge idles above zero, and idling is not rain")
    from das2.weather.provider import GAUGE_NOISE_FLOOR_MM
    rng = np.random.RandomState(7)
    idle = np.abs(rng.randn(2160)) * 0.01            # a few hundredths, always
    check("an idling bucket reads dry, not unknown", total(idle) == 0.0,
          f"(below the {GAUGE_NOISE_FLOOR_MM} mm resolution of any real bucket)")
    # The same defect as the 26 metres, at a size small enough to be believed:
    # 0.01 mm summed across 2,160 scans is 21.6 mm of invented rain.
    check("and its noise is never summed into rain",
          (total(idle) or 0.0) < 1.0,
          f"(summing it would give {float(idle.sum()):.1f} mm)")
    storm = idle.copy()
    for i in range(900, 900 + 40 * 5, 5):
        storm[i] = 0.5
    check("a storm on top of that idle is still measured",
          abs((total(storm) or 0) - 20.0) < 0.01,
          f"({total(storm)} mm = 40 tips x 0.5)")

    print("\nwhat it refuses to guess at")
    noisy = 10 + np.random.RandomState(0).randn(2160)
    check("a gauge wandering about a non-zero value is unknown",
          total(noisy) is None,
          "(unknown, not dry -- triage then does not apply the rain rule)")
    runaway = np.linspace(0, 5000, 2160)
    check("a physically impossible total is refused",
          total(runaway) is None,
          f"(> {MAX_PLAUSIBLE_WINDOW_MM:.0f} mm is arithmetic, not weather)")
    check("a single reading is not a window", total([3.0]) is None)
    check("no readings at all is unknown", total([]) is None)

    print("\nthe consequence that mattered")
    # 21,600 mm against a 2.0 mm threshold is not a large number, it is an
    # unconditional yes. The gate must be shut on the real data.
    from das2.incident.triage import RAIN_EXPLAINS_MM
    check("the real gauges no longer clear the rain threshold",
          (total([10.0] * 2160) or 0.0) < RAIN_EXPLAINS_MM,
          f"(0.0 mm vs the {RAIN_EXPLAINS_MM} mm gate; it was 21600.0)")

    print(f"\n{passed} passed, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
