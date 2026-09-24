"""
Tests for the detectors added to complete the stack.

Every one of these encodes a bug that was found by running the detector, not by
reading it, and several encode the *same* bug in different modules:

    a statistic computed over a window that contains the fault it is looking
    for will be spoiled by that fault.

It has now appeared five times in this project — in FLATLINE (the longest flat
run set its own threshold), QUANTISATION_COLLAPSE (the collapse inflated the
noise estimate that was supposed to reveal it), STUCK_IN_STATE (a pump stuck
for 62 of 72 hours dragged its own baseline up), NOISE_BURST (the burst *is*
the noise being measured) and DRIFT (invisible under a daily cycle at 72-hour
scale). Each test below asserts the fix, and states what the failure looked
like, because the failure mode is invisible to inspection: the detector simply
returns nothing and looks like a quiet network.

Run:  python3 tests/test_detectors.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from das2.detect.baseline import score_window  # noqa: E402
from das2.detect.digital import (  # noqa: E402
    detect_run_state_inconsistent,
    detect_short_cycling,
    detect_stuck_in_state,
)
from das2.detect.health import (  # noqa: E402
    detect_attenuated_signal,
    detect_quantisation_collapse,
)
from das2.detect.massbalance import BalanceGroup, evaluate_group, find_groups  # noqa: E402
from das2.detect.profile import build_profile  # noqa: E402
from das2.detect.selection import select  # noqa: E402
from das2.models import (  # noqa: E402
    Cluster,
    Incident,
    IncidentClass,
    PhysicalSeverity,
    SensorAnomaly,
    SensorMeta,
    AnomalyType,
)
from das2.profile.build import (  # noqa: E402
    build_baseline,
    detect_drift,
    detect_noise_burst,
)
from das2.spatial.correlation import correlate_anomaly, correlation_summary  # noqa: E402


def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
    if not cond:
        raise SystemExit(1)


def minutes(n, step=120, start=datetime(2026, 9, 19)):
    return pd.Series([start + timedelta(seconds=step * i) for i in range(n)])


def profile_of(ts, values):
    return build_profile("k", pd.DataFrame({"ts": ts, "value": values}))


def main():
    rng = np.random.default_rng(3)

    # ---------------------------------------------------------------- #
    print("QUANTISATION_COLLAPSE — resolution degrading")
    n = 2160
    ts = minutes(n)
    clean = np.round(3.75 + rng.normal(0, 0.04, n), 4)
    quantised = np.round(414.8 + rng.normal(0, 0.05, n), 1)
    check("a continuous sensor is silent",
          not detect_quantisation_collapse(ts, clean, profile_of(ts, clean), "bar"))
    check("a sensor that was ALWAYS quantised is silent",
          not detect_quantisation_collapse(ts, quantised,
                                           profile_of(ts, quantised), "V"),
          "(0.1 V steps are what it has always done)")

    # The realistic shape: starts midway, lasts 20 of 72 hours.
    partial = clean.copy()
    a = int(n * 0.5)
    b = a + int(20 * 3600 / 120)
    partial[a:b] = np.round(partial[a:b] / 0.5) * 0.5
    signals = detect_quantisation_collapse(ts, partial, profile_of(ts, partial), "bar")
    check("a collapse covering part of the window is found", len(signals) == 1,
          "(a half-window split missed this: the later half is a MIXTURE of "
          "coarse and fine steps, and a mixture reads as 'not quantised')")
    check("it reports the coarsened step in engineering units",
          abs(signals[0].magnitude - 0.5) < 1e-9, f"({signals[0].magnitude})")
    check("and localises when it began",
          signals[0].start >= ts.iloc[int(n * 0.4)])

    print("\n  the noise reference must not come from the profile")
    # How badly the profile is spoiled depends on how much of the window the
    # fault covers. A 20-hour collapse in 72 hours barely moves it; a collapse
    # running to the end of the window moves it enormously, and that is the
    # case that blinded the first implementation.
    half = clean.copy()
    half[n // 2:] = np.round(half[n // 2:] / 0.5) * 0.5
    spoiled = profile_of(ts, half)
    honest = profile_of(ts, clean)
    check("a collapse spanning half the window inflates the profile's MAD",
          spoiled.mad > 5 * honest.mad,
          f"(MAD {spoiled.mad:.3f} against a true {honest.mad:.3f}; scoring the "
          f"0.5 step against that put it below threshold, so the detector was "
          f"blinded by the fault it was looking for)")
    check("and the detector still finds it, because it measures noise itself",
          len(detect_quantisation_collapse(ts, half, spoiled, "bar")) >= 1)

    # ---------------------------------------------------------------- #
    print("\nATTENUATED_SIGNAL (QARTOD 10) — alive, reporting, measuring nothing")
    swinging = np.round(50 + 8 * np.sin(np.arange(n) / 120) + rng.normal(0, 0.5, n), 1)
    check("a healthy swinging sensor is silent",
          not detect_attenuated_signal(ts, swinging, profile_of(ts, swinging), "%"))

    stuck = swinging.copy()
    stuck[800:1400] = 50.0 + rng.integers(0, 2, 600) * 0.1
    signals = detect_attenuated_signal(ts, stuck, profile_of(ts, stuck), "%")
    check("a sensor wandering by one LSB for 20 h is found", len(signals) == 1,
          f"({len(signals)})")
    check("its reported range is one resolution step",
          abs(signals[0].magnitude - 0.1) < 1e-6, f"({signals[0].magnitude})")

    frozen = swinging.copy()
    frozen[800:1400] = 50.0
    check("an exactly frozen sensor is NOT reported here",
          not detect_attenuated_signal(ts, frozen, profile_of(ts, frozen), "%"),
          "(that is FLATLINE's; the two must stay disjoint or it is alerted twice)")

    # ---------------------------------------------------------------- #
    print("\nDigital: SHORT_CYCLING, STUCK_IN_STATE, RUN_STATE_INCONSISTENT")
    m = 4320
    dts = minutes(m, step=60)
    duty = np.array([1.0 if (i // 240) % 2 == 0 else 0.0 for i in range(m)])
    check("a healthy 4-hour duty cycle does not short-cycle",
          not detect_short_cycling(dts, duty))
    check("...nor look stuck", not detect_stuck_in_state(dts, duty))

    chattering = duty.copy()
    for i in range(1200, 1380):
        chattering[i] = 1.0 if (i // 3) % 2 == 0 else 0.0
    signals = detect_short_cycling(dts, chattering)
    check("3-minute chatter for 3 hours is found", len(signals) == 1,
          "(real motor wear, and invisible to every value-based detector)")
    check("reported as a rate", signals[0].unit == "cycles/h")

    stuck_on = duty.copy()
    stuck_on[600:] = 1.0
    signals = detect_stuck_in_state(dts, stuck_on)
    check("a pump stuck ON for 62 h is found", len(signals) == 1,
          "(including its own hold in the baseline raised the threshold to "
          "290 h and found nothing — leave-one-out fixes it)")

    standby = np.zeros(m)
    standby[100:140] = 1.0
    standby[300:340] = 1.0
    check("a standby pump sitting OFF for 66 h is NOT flagged",
          not detect_stuck_in_state(dts, standby),
          "(within 72 h, 'seized shut' and 'standby doing its job' are the "
          "same signal; this abstains rather than guessing)")

    flow_ts = minutes(m // 2, step=120)
    flow = np.where(np.array([(i * 2 // 240) % 2 == 0 for i in range(m // 2)]),
                    30.0, 0.0) + rng.normal(0, 0.3, m // 2)
    check("a pump whose flow agrees is silent",
          not detect_run_state_inconsistent(dts, duty, flow_ts, flow, "L/s"))
    broken = flow.copy()
    broken[300:420] = 0.05
    signals = detect_run_state_inconsistent(dts, duty, flow_ts, broken, "L/s")
    check("commanded running with no discharge flow is found", len(signals) >= 1)
    check("and it names both possibilities rather than guessing",
          "flowmeter" in signals[0].detail["verdict"]
          and "not running" in signals[0].detail["verdict"])

    # ---------------------------------------------------------------- #
    print("\nMASS_BALANCE_VIOLATION — three readings that cannot all be true")
    k = 1200
    step = 120
    tank_ts = minutes(k, step=step)
    qin = 0.5 + rng.normal(0, 0.008, k)
    qout = 0.5 + 0.18 * np.sin(np.arange(k) / 50) + rng.normal(0, 0.008, k)
    area = 500.0
    level = 3.0 + np.cumsum((qin - qout) * step / area) + rng.normal(0, 0.002, k)
    group = BalanceGroup(site="T", level_key="L", inlet_keys=["I"], outlet_keys=["O"])
    consistent = {"L": (tank_ts, level), "I": (tank_ts, qin), "O": (tank_ts, qout)}
    check("a consistent tank is silent", not evaluate_group(group, consistent))

    under_reading = qout.copy()
    under_reading[400:580] *= 0.7
    signals = evaluate_group(group, {"L": (tank_ts, level), "I": (tank_ts, qin),
                                     "O": (tank_ts, under_reading)})
    check("an outlet meter under-reading by 30% is found", len(signals) >= 1)
    check("the tank area is FITTED, not guessed",
          400 < signals[0].detail["fitted_area"] < 650,
          f"({signals[0].detail['fitted_area']} against a true 500)")
    check("and the fit quality is reported",
          signals[0].detail["fit_r2"] > 0.5, f"({signals[0].detail['fit_r2']})")

    wrong_tank = {"L": (tank_ts, 3.0 + rng.normal(0, 0.05, k)),
                  "I": (tank_ts, qin), "O": (tank_ts, qout)}
    check("a level sensor on a DIFFERENT tank produces nothing",
          not evaluate_group(group, wrong_tank),
          "(not a closed system, so it abstains rather than reporting a "
          "modelling failure as a fault)")

    print("\n  grouping comes from the inventory, by name")
    sensors = pd.DataFrame([
        {"sensor_key": "1", "site": "SiteA", "equipment": "LevelSensor",
         "description": "SiteA-Reservoir-Level"},
        {"sensor_key": "2", "site": "SiteA", "equipment": "Flowrate",
         "description": "SiteA-Reservoir-Inlet-Flow"},
        {"sensor_key": "3", "site": "SiteA", "equipment": "Flowrate",
         "description": "SiteA-Reservoir-Outlet-Flow"},
        {"sensor_key": "4", "site": "SiteB", "equipment": "Pressure",
         "description": "SiteB-Pump-Pressure"},
    ])
    groups = find_groups(sensors)
    check("one group is found", len(groups) == 1, f"({len(groups)})")
    check("inflow and outflow are told apart by name",
          groups[0].inlet_keys == ["2"] and groups[0].outlet_keys == ["3"],
          "(both are 'Flowrate'; direction exists only in the description)")
    check("a site with no level sensor forms no group",
          all(g.site != "SiteB" for g in groups))

    # ---------------------------------------------------------------- #
    print("\nDRIFT and NOISE_BURST — why they are a DAILY job")
    days, per_day = 28, 144
    long_n = days * per_day
    long_ts = pd.Series([datetime(2026, 8, 1) + timedelta(minutes=10 * i)
                         for i in range(long_n)])
    hour = np.array([t.hour + t.minute / 60 for t in long_ts])
    cycle = 1.2 * np.sin((hour - 6) / 24 * 2 * np.pi)
    healthy = 4.0 + cycle + rng.normal(0, 0.05, long_n)

    def frame(v):
        return pd.DataFrame({"ts": long_ts, "value": v})

    check("a healthy sensor with a strong daily cycle does not drift",
          not detect_drift("k", frame(healthy), unit="bar"),
          "(the +-1.2 bar cycle is 24x the noise; a naive slope follows it)")

    drifting = healthy + np.linspace(0, 0.02 * days, long_n)
    signals = detect_drift("k", frame(drifting), unit="bar")
    check("a 0.5%/day drift over 28 days IS found", len(signals) == 1)
    check("reported as a percentage per day, not a raw slope",
          0.3 < signals[0].detail["percent_per_day"] < 0.7,
          f"({signals[0].detail['percent_per_day']}%/day)")
    check("with the trend consistency that separates it from a random walk",
          signals[0].detail["kendall_tau"] > 0.9,
          f"(tau {signals[0].detail['kendall_tau']})")

    print("\n  and the same drift is INVISIBLE in a 72-hour window")
    short = frame(drifting).iloc[:3 * per_day]
    check("72 hours of the identical series yields nothing",
          not detect_drift("k", short, unit="bar"),
          "(3% of drift under a 30% daily cycle — this is why it is a daily job)")

    check("a healthy sensor has no noise burst",
          not detect_noise_burst("k", frame(healthy), unit="bar"))
    bursting = healthy.copy()
    bursting[20 * per_day:21 * per_day] += rng.normal(0, 0.5, per_day)
    signals = detect_noise_burst("k", frame(bursting), unit="bar")
    check("one day at 10x noise is found", len(signals) == 1)
    check("measured against the PRECEDING days only",
          signals[0].detail["ratio"] > 5,
          f"(ratio {signals[0].detail['ratio']}; the burst cannot be allowed "
          f"into its own baseline)")

    # ---------------------------------------------------------------- #
    print("\nRESIDUAL_OUTLIER — scored against the stored time-of-day baseline")
    baseline = build_baseline("k", frame(healthy))
    check("a usable baseline is built", baseline.usable,
          f"({len(baseline.buckets)} buckets over {baseline.days_observed} days)")
    check("it captures the daily cycle",
          abs(baseline.lookup(datetime(2026, 9, 1, 3, 0))[0]
              - baseline.lookup(datetime(2026, 9, 1, 15, 0))[0]) > 1.0,
          "(03:00 and 15:00 must differ, or the cycle is not being modelled)")

    w = 144
    win_ts = pd.Series([datetime(2026, 8, 29) + timedelta(minutes=10 * i)
                        for i in range(w)])
    win_hour = np.array([t.hour + t.minute / 60 for t in win_ts])
    normal = 4.0 + 1.2 * np.sin((win_hour - 6) / 24 * 2 * np.pi) + rng.normal(0, 0.05, w)
    check("a healthy window scores nothing",
          not score_window(win_ts, normal, baseline, unit="bar"))

    depressed = normal.copy()
    depressed[40:76] -= 0.6
    signals = score_window(win_ts, depressed, baseline, unit="bar")
    check("a sustained 0.6 bar depression IS found", len(signals) >= 1,
          "(a rolling median follows a sustained shift and cannot see it; a "
          "stored baseline does not move)")
    check("reported in engineering units with a direction",
          signals[0].unit == "bar" and signals[0].detail["direction"] == "below")

    late = 4.0 + 1.2 * np.sin((win_hour - 6.5) / 24 * 2 * np.pi) + rng.normal(0, 0.05, w)
    check("a cycle running 30 minutes late is tolerated",
          not score_window(win_ts, late, baseline, unit="bar"),
          "(a pump starting late is normal; without tolerance it is a huge "
          "residual every day)")
    check("no baseline means abstain, not guess",
          not score_window(win_ts, depressed, None, unit="bar"))

    # ---------------------------------------------------------------- #
    print("\nNeighbour correlation — the water, or the instrument?")
    c_n = 1440
    c_ts = pd.Series([datetime(2026, 9, 19, 12) + timedelta(seconds=300 * i)
                      for i in range(c_n)])
    demand = 4.0 + 0.5 * np.sin(np.arange(c_n) / 60)
    t0 = datetime(2026, 9, 20, 3, 0)
    i0 = int(np.searchsorted([t.timestamp() for t in c_ts], t0.timestamp()))
    dip = np.ones(c_n)
    dip[i0:i0 + 60] = 0.65

    def sensor(key, lat):
        return SensorMeta(sensor_key=key, description=f"S{key}", equipment="Pressure",
                          site=f"Site{key}", latitude=lat, longitude=103.9,
                          region="East", unit="bar")

    subject = sensor("A", 1.340)
    fleet = [subject, sensor("B", 1.350), sensor("C", 1.360)]
    anomaly = SensorAnomaly(sensor=subject, start=t0, end=t0 + timedelta(hours=5),
                            dominant_type=AnomalyType.LEVEL_SHIFT,
                            severity=PhysicalSeverity(duration_s=18000))

    real_event = {k: (c_ts, demand * dip + rng.normal(0, 0.02, c_n))
                  for k in ("A", "B", "C")}
    broken = {"A": (c_ts, demand * dip + rng.normal(0, 0.02, c_n)),
              "B": (c_ts, demand + rng.normal(0, 0.02, c_n)),
              "C": (c_ts, demand + rng.normal(0, 0.02, c_n))}

    real_summary = correlation_summary(correlate_anomaly(anomaly, fleet, real_event))
    broken_summary = correlation_summary(correlate_anomaly(anomaly, fleet, broken))
    check("neighbours move together on a real event",
          real_summary["moved_together"] == 2 and real_summary["decoupled"] == 0,
          f"({real_summary})")
    check("and decouple when the instrument is the problem",
          broken_summary["decoupled"] == 2 and broken_summary["moved_together"] == 0,
          f"({broken_summary})")
    check("the two cases are far apart, not marginal",
          real_summary["median_r"] - broken_summary["median_r"] > 0.5,
          f"(r={real_summary['median_r']} vs {broken_summary['median_r']})")

    # ---------------------------------------------------------------- #
    print("\nSelection — budgets, not a top-N rank cut")

    def incident(cls, region, severity, key="x"):
        meta = SensorMeta(sensor_key=key, description=key, equipment="Pressure",
                          site=region, region=region)
        member = SensorAnomaly(sensor=meta, start=t0, end=t0 + timedelta(hours=1),
                               dominant_type=AnomalyType.LEVEL_SHIFT,
                               severity=PhysicalSeverity(duration_s=3600))
        return Incident(incident_id=f"{region}-{key}",
                        cluster=Cluster(members=[member], region=region),
                        incident_class=cls, severity=severity)

    flood = [incident(IncidentClass.SENSOR_FAULT, "East", 60.0, f"e{i}")
             for i in range(12)]
    flood.append(incident(IncidentClass.SENSOR_FAULT, "West", 55.0, "w1"))
    result = select(flood, per_region=5, global_cap=10)
    east = [i for i in result.selected if str(i.cluster.region) == "East"]
    west = [i for i in result.selected if str(i.cluster.region) == "West"]
    check("a bad night in the East respects its regional budget", len(east) == 5,
          f"({len(east)})")
    check("and cannot crowd out the West's one real problem", len(west) == 1,
          "(under a global top-N rank cut it silently would)")
    check("everything held is accounted for, never discarded",
          len(result.selected) + len(result.held) == len(flood))
    check("with a stated reason", all(reason for _, reason in result.held))

    p1 = incident(IncidentClass.REGIONAL_EVENT, "East", 90.0, "p1")
    result = select([p1] + flood, per_region=1, global_cap=1)
    check("a P1 is never held back by a budget",
          any(i.incident_id == "East-p1" for i in result.selected),
          "(a budget that can suppress the worst thing happening is a bug)")

    quiet = select([incident(IncidentClass.TELEMETRY_FANOUT, "East", 8.0)])
    check("a quiet run selects nothing at all", not quiet.selected,
          "(v1 emitted ~10 sensors per run whether healthy or on fire)")
    check("and says why", "fan-out" in quiet.held[0][1])

    print("\nAll detector tests passed.")


if __name__ == "__main__":
    main()
