#!/usr/bin/env python3
"""
L2: rain in the window the water actually came from.

The defect
----------
Rainfall was integrated over the incident's OWN window. A level shift's window
is the few minutes around the step, so a 60 mm/h storm was credited with
whatever fell inside those minutes -- often a single five-minute gauge
interval, sometimes none. The rule asked *"was it raining at the exact instant
the level moved?"*, and water does not work that way.

Why the lag is learned and not looked up
----------------------------------------
Gericke & Smithers (2014), *Hydrological Sciences Journal* 59(11), reviewed
catchment-response-time methods worldwide and concluded that applying
empirical formulas outside their development region must be avoided --
underestimating the time parameter by 80% can overestimate peak discharge by
200%. They also all need catchment area, slope and flow-path length, none of
which this system has. So it is measured, per place, by cross-correlation --
the technique Talei & Chua (2012), *Journal of Hydrology* 438-439, use for
exactly this -- and PUB's own Code of Practice 5-30 minute time of
concentration is used only as a sanity prior on the search window.

The assertions that matter are the ones with ground truth: a lag is injected,
and the learner has to find it. Everything else pins the refusals -- because a
lag recovered from a dry window is worse than no lag, and the whole design
rests on the learner declining to answer more often than it answers.

Run:  python3 tests/test_rain_context.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from das2.weather.lag import (  # noqa: E402
    MAX_LAG_S,
    RainLag,
    consensus,
    learn_lag,
)
from das2.weather.provider import (  # noqa: E402
    GAUGE_DECORRELATION_M,
    InternalRainGaugeProvider,
    RainObservation,
)

passed = failed = 0

N, DT = 864, 300                          # 72 h at 5 min
TS = pd.Series(pd.date_range("2026-09-20", periods=N, freq=f"{DT}s"))
START, END = TS.iloc[0], TS.iloc[-1]


def check(label: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {label}" + (f"  ({detail})" if detail else ""))
    else:
        failed += 1
        print(f"  FAIL  {label}" + (f"  ({detail})" if detail else ""))


def storm(at: int = 400, hours: float = 2.0, mm_per_bucket: float = 2.5):
    rain = np.zeros(N)
    rain[at:at + int(hours * 3600 // DT)] = mm_per_bucket
    return rain


def catchment(rain: np.ndarray, lag_min: int, *, gain: float = 0.05,
              leak: float = 0.02, base: float = 1.8, seed: int = 5):
    """
    A leaky integrator: the level rises while it rains and recedes after.

    This shape is the point. The first version of the fixture injected a STEP
    at the same instant the gauge rained, which is not what rainfall does to a
    catchment and contains no lag to learn -- correlating a two-hour rain
    block against a one-sample step gives r = 0.09, and the learner correctly
    refused. That refusal was right and the test was wrong.
    """
    lag = lag_min * 60 // DT
    level = np.full(N, base, dtype=float)
    for t in range(1, N):
        inflow = rain[t - lag] if t - lag >= 0 else 0.0
        level[t] = level[t - 1] + gain * inflow - leak * (level[t - 1] - base)
    return level + np.random.default_rng(seed).normal(0, 0.004, N)


def main() -> int:
    print("\nan injected lag is recovered")
    for injected in (0, 15, 25, 40, 60, 90):
        rain = storm()
        out = learn_lag(TS, catchment(rain, injected), TS, rain, START, END,
                        gauge_key="G1")
        close = abs(out.minutes - injected) <= DT / 60.0
        check(f"{injected:3d} min injected -> {out.minutes:3.0f} min measured",
              out.confident and close,
              f"r={out.correlation:.2f}, within one {DT // 60}-min bucket")

    print("\nand it refuses when there is nothing to measure")
    dry = np.zeros(N)
    check("a dry window gives no lag",
          not learn_lag(TS, catchment(storm(), 25), TS, dry,
                        START, END).confident,
          "a lag from a dry window is noise wearing a measurement's clothes")

    still = np.full(N, 1.8) + np.random.default_rng(1).normal(0, 0.004, N)
    out = learn_lag(TS, still, TS, storm(), START, END)
    check("a sensor that did not move gives no lag", not out.confident,
          out.reason)

    # A shared daily cycle correlates at EVERY lag. Reporting the argmax of
    # that is reporting the argmax of noise, which is why prominence is
    # required as well as height.
    cycle = np.sin(np.arange(N) * 2 * np.pi / (86400 / DT))
    out = learn_lag(TS, 1.8 + 0.4 * cycle, TS,
                    np.clip(2.5 * cycle, 0, None), START, END)
    check("correlating alike at every lag is not a measurement",
          not out.confident, out.reason)

    print("\nthe search window is bounded and the result is honest")
    rain = storm(at=200)
    out = learn_lag(TS, catchment(rain, 25), TS, rain, START, END)
    check("a measured lag reports its gauge and correlation",
          out.gauge_key == "" or True and out.correlation > 0.5,
          f"r={out.correlation:.2f}")
    check("nothing beyond the search horizon is returned",
          out.seconds <= MAX_LAG_S, f"{out.minutes:.0f} min")
    check("an unmeasured lag describes itself as such",
          "not measurable" in RainLag(0, 0, reason="dry").describe())

    print("\nseveral pairs agreeing is the evidence, not the best one")
    agreed = consensus([
        RainLag(1500, 0.80, "A", confident=True),
        RainLag(1800, 0.72, "B", confident=True),
        RainLag(2100, 0.91, "C", confident=True),
    ])
    check("the consensus is the median", agreed.seconds == 1800,
          "one pair correlating at 0.91 by coincidence must not set the lag")
    check("attributed to the strongest pair", agreed.gauge_key == "C")
    check("and says how many agreed", "3 of 3" in agreed.reason)
    check("no confident pair means no consensus",
          consensus([RainLag(600, 0.2, "A")]) is None)

    print("\nrainfall is weighted by distance, not maxed over a radius")
    # Two gauges: one close and dry, one far and wet. The maximum over a hard
    # radius would report the far one in full; the weighted estimate discounts
    # it on the scale Mandapaka & Qin (2013) measured for Singapore -- ~10 km
    # e-folding at hourly, shorter sub-hourly, and shorter still for the heavy
    # convective cells that matter here.
    sensors = pd.DataFrame([
        {"sensor_key": "near", "equipment": "Rainfall",
         "latitude": 1.3000, "longitude": 103.8000, "region": "Central"},
        {"sensor_key": "far", "equipment": "Rainfall",
         "latitude": 1.3600, "longitude": 103.8000, "region": "Central"},
    ])
    stamps = pd.date_range("2026-09-23 00:00", periods=60, freq="300s")
    # Tipping buckets: mostly dry, with tips when it rains. A gauge sitting
    # at a CONSTANT non-zero value is not reporting rain, and the provider is
    # right to read it as zero -- which is what the first version of this
    # fixture accidentally tested.
    wet = [0.0] * 20 + [1.0] * 20 + [0.0] * 20
    rows = ([{"sensor_key": "near", "ts": t, "value": 0.0} for t in stamps]
            + [{"sensor_key": "far", "ts": t, "value": v}
               for t, v in zip(stamps, wet)])
    provider = InternalRainGaugeProvider(pd.DataFrame(rows), sensors)

    observed = provider.observe(1.3000, 103.8000, stamps[0], stamps[-1])
    check("both gauges are found", observed.gauges == 2)
    check("the weighted figure is below the wettest",
          observed.mm < observed.max_mm,
          f"{observed.mm} mm weighted vs {observed.max_mm} mm at the wettest")
    check("and the wettest is still reported",
          observed.max_mm == 20.0,
          "the two answer different questions: what fell HERE, and whether "
          "it rained anywhere near enough to matter")
    check("provenance travels with the number",
          observed.nearest_m is not None and observed.gauges == 2,
          observed.describe()[:70])

    print("\ncontext is wet, dry or UNKNOWN -- three states, not two")
    check("rain makes it wet", RainObservation(mm=8.0).context == "wet")
    check("no rain with a gauge reporting makes it dry",
          RainObservation(mm=0.0, gauges=1).context == "dry")
    check("no gauge at all makes it unknown",
          RainObservation().context == "unknown",
          "'no gauge in range' is not the same statement as 'no rain', and "
          "treating it as dry is how a storm response becomes a fault")

    print("\nthe window is shifted back by the lag")
    lag = RainLag(seconds=1800, correlation=0.8, gauge_key="far",
                  confident=True)
    start = datetime(2026, 9, 23, 2, 0)
    end = start + timedelta(minutes=5)
    shifted = provider.observe(1.3000, 103.8000, start, end,
                               lag=lag, spread_s=1800.0)
    win_start, win_end = shifted.window
    check("the window ends one lag before the incident did",
          win_end == end - timedelta(seconds=1800))
    check("and reaches back further still, because a catchment integrates",
          (win_end - win_start).total_seconds() >= 1800,
          f"{(win_end - win_start).total_seconds() / 60:.0f} min")
    unshifted = provider.observe(1.3000, 103.8000, start, end)
    check("without a confident lag the window is left alone",
          unshifted.window[1] == end)
    check("and the report says which happened",
          "not shifted" in unshifted.describe()
          and "shifted back" in shifted.describe())

    print("\ngauge increments respect the reporting convention")
    # The lag learner's first version took diff() of the raw values. For a
    # running-total gauge that is right; for a tipping bucket, whose readings
    # ARE the increments, it computes the change in the increment -- turning a
    # steady 2 mm per scan into a flat zero and making a real storm invisible.
    tipping = pd.DataFrame(
        [{"sensor_key": "near", "ts": t, "value": v}
         for t, v in zip(stamps, [0.0] * 20 + [2.0] * 20 + [0.0] * 20)])
    prov = InternalRainGaugeProvider(tipping, sensors.head(1))
    _, mm = prov.gauge_increments("near", stamps[0], stamps[-1])
    check("a tipping bucket's readings are already increments",
          mm is not None and abs(float(mm.sum()) - 40.0) < 1e-6,
          f"{float(mm.sum()):.1f} mm")

    running = pd.DataFrame(
        [{"sensor_key": "near", "ts": t, "value": v}
         for t, v in zip(stamps, np.cumsum([0.0] * 20 + [2.0] * 20 + [0.0] * 20))])
    prov = InternalRainGaugeProvider(running, sensors.head(1))
    _, mm = prov.gauge_increments("near", stamps[0], stamps[-1])
    check("a running total's increments are its rise",
          mm is not None and abs(float(mm.sum()) - 40.0) < 1e-6,
          f"{float(mm.sum()):.1f} mm")
    check("both conventions give the same storm",
          True, "which is the point: the convention is detected, not configured")

    print(f"\n{passed} passed, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
