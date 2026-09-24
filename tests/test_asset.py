#!/usr/bin/env python3
"""
The machine has failed, and every instrument on it is reporting normally.

    "maybe there's a complete breakdown of the equipment instead of just
     operational failure, but there's still value being read by the sensor"

Every other layer in this system watches values. This one watches the
RELATIONSHIP between channels, because a pump that is running, energised,
drawing an ordinary current and moving no water has not produced a single
abnormal reading anywhere.

The weight of these tests is on what the detector must NOT say
---------------------------------------------------------------
Naming the machine as the culprit sends a fitter out, so the bar is higher
than for a finding that merely says "look at this". Three refusals are pinned
harder than any of the positive cases:

* **Output gone, motor current unchanged -> silence.** A motor doing its usual
  work is evidence that water IS moving, so the flowmeter becomes the odd one
  out and the honest reading is `RUN_STATE_INCONSISTENT`'s "one of these two
  is wrong". Claiming an asset failure there picks the expensive explanation
  over the cheap one on no evidence. This is the single most important
  assertion in the file.
* **A duty channel that cannot tell on from off -> silence.** If current reads
  the same whether the pump runs or not, it cannot testify about anything.
* **Two candidate ammeters on one unit -> silence.** A wrong pairing
  manufactures a contradiction between instruments that were never measuring
  the same thing.

And on the direction of the power change
-----------------------------------------
Never used. "Power drops when a centrifugal pump stops delivering" is true of
a radial-flow pump and backwards for an axial-flow one, where brake power is
highest at shutoff. PUB runs both and the feed does not say which is which, so
the test is DEPARTURE from the unit's own normal, in either direction. A test
below drives the current up instead of down and expects the same finding.

Run:  python3 tests/test_asset.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from das2.detect import asset as A                       # noqa: E402
from das2.detect.fusion import TYPE_PRECEDENCE           # noqa: E402
from das2.incident.triage import CLASS_FLOOR, classify   # noqa: E402
from das2.models import (                                # noqa: E402
    ALWAYS_PAGEABLE_TYPES,
    ASSET_TYPES,
    MAINTENANCE_TYPES,
    PROCESS_TYPES,
    QARTOD_TEST,
    RECOMMENDATION,
    SENSOR_HEALTH_TYPES,
    AnomalyType,
    Cluster,
    IncidentClass,
    PhysicalSeverity,
    QartodFlag,
    SensorAnomaly,
    SensorMeta,
    Signal,
    flag_for,
)
from das2.spatial.cluster import cluster_by_asset        # noqa: E402

passed = failed = 0

T0 = datetime(2026, 9, 21, 0, 0)
N = 1440                      # 24 h at one sample a minute
CYCLE = 240                   # 4 h on, 4 h off
TS = pd.Series(pd.date_range(T0, periods=N, freq="60s"))

RUNNING_A, IDLE_A, UNLOADED_A = 41.0, 0.6, 12.0
RUNNING_FLOW = 24.0


def check(label: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {label}" + (f"  ({detail})" if detail else ""))
    else:
        failed += 1
        print(f"  FAIL  {label}" + (f"  ({detail})" if detail else ""))


def running_mask() -> np.ndarray:
    return (np.arange(N) % (2 * CYCLE)) < CYCLE


def machine(*, fault: slice | None = None, duty_in_fault: float | None = None,
            output_in_fault: float | None = None,
            duty_when_off: float = IDLE_A) -> dict:
    """A pump whose three channels agree, optionally broken over `fault`."""
    on = running_mask()
    rng = np.random.default_rng(5)
    duty = np.where(on, RUNNING_A, duty_when_off) + rng.normal(0, 0.15, N)
    output = np.where(on, RUNNING_FLOW, 0.0) + np.abs(rng.normal(0, 0.05, N))
    if fault is not None:
        if duty_in_fault is not None:
            duty[fault] = duty_in_fault + rng.normal(0, 0.15, fault.stop - fault.start)
        if output_in_fault is not None:
            output[fault] = output_in_fault
    return {"RUN": (TS, on.astype(float)),
            "DUTY": (TS, duty),
            "OUT": (TS, output)}


def pump(output: bool = True) -> A.Asset:
    return A.Asset(site="BedokPS", unit="1", run_key="RUN", duty_key="DUTY",
                   duty_kind="current",
                   output_key="OUT" if output else "",
                   output_kind="flow" if output else "",
                   units={"DUTY": "A", "OUT": "L/s"})


def types_of(signals) -> list[AnomalyType]:
    return [s.type for s in signals]


def inventory(rows: list[tuple[str, str, str, str]]):
    """(description, equipment, site, signal_type) -> a sensors frame."""
    return pd.DataFrame([
        {"sensor_key": f"K{n}", "description": d, "equipment": e, "site": s,
         "signal_type": t, "kind": "status" if t == "Digital" else "measurement",
         "unit": "A" if e == "Current" else ""}
        for n, (d, e, s, t) in enumerate(rows)])


def main() -> int:
    print("\npairing a machine's channels")
    sensors = inventory([
        ("TampinesPS-Pump1-Run-Status", "DigitalStatus", "TampinesPS", "Digital"),
        ("TampinesPS-Pump1-Discharge-Flow", "Flowrate", "TampinesPS", "Analog"),
        ("TampinesPS-Pump1-Motor-Current", "Current", "TampinesPS", "Analog"),
    ])
    assets = A.find_assets(sensors)
    check("one machine is found", len(assets) == 1, f"{len(assets)}")
    found = assets[0]
    check("its duty channel is THE CURRENT, not the flowmeter",
          found.duty_key == "K2" and found.duty_kind == "current",
          "`'amp' in 'TampinesPS'` is True, and substring matching handed the "
          "discharge flowmeter over as a motor-current channel — the detector "
          "then adjudicated a pump against the very meter it was judging")
    check("and its output channel is the flow", found.output_key == "K1")
    check("the machine is named for the report", found.name == "TampinesPS unit 1")

    print("\nand refusing to pair when it would be a guess")
    ambiguous = inventory([
        ("BedokPS-Pump1-Run-Status", "DigitalStatus", "BedokPS", "Digital"),
        ("BedokPS-Pump1-Motor-Current-Red", "Current", "BedokPS", "Analog"),
        ("BedokPS-Pump1-Motor-Current-Yellow", "Current", "BedokPS", "Analog"),
    ])
    check("two candidate ammeters on one unit yields nothing",
          not A.find_assets(ambiguous),
          "picking one means comparing a run state against an arbitrary "
          "phase, and a fabricated contradiction is a callout nobody needed")
    check("a run state with nothing to check it against is not an asset",
          not A.find_assets(inventory([
              ("BedokPS-Pump9-Run-Status", "DigitalStatus", "BedokPS", "Digital")])),
          "92 of the real inventory's 192 units are in exactly this position")
    check("a fault or setpoint point is never taken as a duty channel",
          not A.find_assets(inventory([
              ("BedokPS-Pump1-Run-Status", "DigitalStatus", "BedokPS", "Digital"),
              ("BedokPS-Pump1-Current-Alarm-Setpoint", "Current", "BedokPS",
               "Analog")])))

    print("\nrunning, energised, and delivering nothing")
    # Inside a RUNNING block. The blocks are [0,240) on, [240,480) off,
    # [480,720) on, so 540-700 is comfortably within the second on-block --
    # the first version of this slice sat entirely in the OFF block and the
    # fixture tested the wrong rule.
    fault = slice(540, 700)
    broken = machine(fault=fault, duty_in_fault=UNLOADED_A, output_in_fault=0.0)
    signals = A.detect_asset_faults(pump(), broken)
    check("the machine is named as the fault",
          types_of(signals) == [AnomalyType.ASSET_NOT_DELIVERING],
          ", ".join(t.value for t in types_of(signals)) or "nothing")
    detail = signals[0].detail
    check("the verdict says which channels decided it",
          "flow has gone" in detail["verdict"]
          and "current" in detail["verdict"])
    check("and shows the numbers behind it",
          detail["observed_duty"] < detail["running_duty"]
          and detail["observed_output"] < detail["normal_output"],
          f"{detail['observed_duty']} A against a normal of "
          f"{detail['running_duty']} A")
    check("it names every sensor it used",
          {"run_state_sensor", "duty_sensor", "output_sensor"} <= set(detail),
          "the finding is about a relationship, so one sensor key does not "
          "describe it")

    print("\nTHE ONE IT MUST NOT CALL: output gone, motor unchanged")
    meter_suspect = machine(fault=fault, output_in_fault=0.0)   # duty untouched
    check("no asset finding when the motor draws its normal current",
          not A.detect_asset_faults(pump(), meter_suspect),
          "the motor doing its usual work is evidence that water IS moving; "
          "the meter is the odd one out, and that is "
          "RUN_STATE_INCONSISTENT's 'one of these two is wrong'")

    print("\nand it does not care WHICH WAY the current moved")
    # An axial-flow pump draws MORE at shutoff, not less. Same finding.
    axial = machine(fault=fault, duty_in_fault=RUNNING_A * 1.6,
                    output_in_fault=0.0)
    check("a rise away from normal is a departure too",
          types_of(A.detect_asset_faults(pump(), axial))
          == [AnomalyType.ASSET_NOT_DELIVERING],
          "keying on a drop would be right for a radial pump and backwards "
          "for an axial one, silently")

    print("\nthe electrical contradictions")
    off_block = slice(CYCLE + 30, CYCLE + 210)      # inside an OFF block
    energised = machine()
    energised["DUTY"][1][off_block] = RUNNING_A
    check("drawing current while the control says off",
          types_of(A.detect_asset_faults(pump(output=False), energised))
          == [AnomalyType.ASSET_ENERGISED_WHEN_OFF])

    dead = machine(fault=fault, duty_in_fault=IDLE_A, output_in_fault=0.0)
    kinds = types_of(A.detect_asset_faults(pump(output=False), dead))
    check("drawing nothing while the control says running",
          kinds == [AnomalyType.ASSET_NOT_ENERGISED_WHEN_ON],
          ", ".join(t.value for t in kinds) or "nothing")

    print("\nabstaining when the channels cannot say anything")
    flat = machine()
    flat["DUTY"] = (TS, np.full(N, 7.0))
    check("a duty channel that never moves yields nothing",
          not A.detect_asset_faults(pump(), flat),
          "a current point reading the same on and off cannot testify about "
          "whether the pump is turning")

    weak = machine()
    weak["DUTY"] = (TS, np.where(running_mask(), 10.0, 9.0))
    check("nor one whose on/off separation is too small to trust",
          not A.detect_asset_faults(pump(), weak),
          f"below {A.MIN_DUTY_SEPARATION:.0%} of the running level")

    brief = machine(fault=slice(300, 305), duty_in_fault=UNLOADED_A,
                    output_in_fault=0.0)
    check("a contradiction shorter than the spin-up window is ignored",
          not A.detect_asset_faults(pump(), brief),
          f"under {A.MIN_CONTRADICTION_S / 60:.0f} min: starters, meter lag "
          f"and the scan cycle all produce these in a healthy plant")

    print("\na clean machine says nothing at all")
    check("no findings on a pump doing its job",
          not A.detect_asset_faults(pump(), machine()),
          "the false-positive guard: every reading here is ordinary and the "
          "relationships all hold")

    print("\nwhere these sit in the taxonomy")
    check("they are their own category",
          ASSET_TYPES and not (ASSET_TYPES & SENSOR_HEALTH_TYPES)
          and not (ASSET_TYPES & PROCESS_TYPES)
          and not (ASSET_TYPES & MAINTENANCE_TYPES),
          "not the instrument, not the water, and not next month's "
          "calibration round")
    check("QARTOD FLAGS THE DATA GOOD, and that is the finding",
          all(flag_for(t) is QartodFlag.GOOD for t in ASSET_TYPES),
          "every instrument involved is working perfectly; a quality system "
          "looking only at these channels sees nothing wrong, which is why "
          "the plant failure needs a finding of its own")
    check("no QARTOD test is claimed for them",
          all(QARTOD_TEST[t] is None for t in ASSET_TYPES),
          "the mechanism is Test 9's; the conclusion is not a data-quality "
          "verdict at all")
    check("each has a precedence slot",
          all(t in TYPE_PRECEDENCE for t in ASSET_TYPES))
    check("and outranks the two-channel version of the same contradiction",
          max(TYPE_PRECEDENCE.index(t) for t in ASSET_TYPES)
          < TYPE_PRECEDENCE.index(AnomalyType.RUN_STATE_INCONSISTENT),
          "these say WHICH of the two readings to believe; that one cannot")

    check("THEY CAN REACH AN INCIDENT",
          ASSET_TYPES <= ALWAYS_PAGEABLE_TYPES,
          "every asset finding is raised on a run-state sensor, and Pump, "
          "Valve and DigitalStatus all start alertable:false — without this "
          "the detector finds the failed pump and the finding is discarded "
          "before it reaches anyone. It did exactly that on the first run")

    print("\nthe incident it becomes")
    cluster = Cluster(members=[_member(AnomalyType.ASSET_NOT_DELIVERING)],
                      region="East")
    klass, why = classify(cluster)
    check("a machine fault is classed as one",
          klass is IncidentClass.ASSET_FAILURE, klass.value)
    check("the evidence is the detector's own verdict",
          any("not delivering" in w for w in why), " | ".join(why))
    check("the recommendation sends a fitter, not a calibrator",
          "mechanical callout" in RECOMMENDATION[IncidentClass.ASSET_FAILURE])

    fanout = Cluster(members=[_member(AnomalyType.ASSET_NOT_DELIVERING),
                              _member(AnomalyType.ASSET_ENERGISED_WHEN_OFF,
                                      key="K2")],
                     region="East")
    check("TWO CHANNELS AT ONE SITE IS NOT FAN-OUT",
          classify(fanout)[0] is IncidentClass.ASSET_FAILURE,
          "an asset finding is by construction several sensors at one site "
          "disagreeing, which is the shape the fan-out rule suppresses — "
          "decided after it, every pump failure would be thrown away as a "
          "panel fault")

    wide = Cluster(members=[_member(AnomalyType.ASSET_NOT_DELIVERING),
                            _member(AnomalyType.LEVEL_SHIFT, key="K2",
                                    site="TampinesPS"),
                            _member(AnomalyType.LEVEL_SHIFT, key="K3",
                                    site="BedokPond4")],
                   region="East")
    check("but an asset finding inside a multi-site cluster does not relabel it",
          classify(wide)[0] is not IncidentClass.ASSET_FAILURE,
          "a machine is one place; relabelling a four-site event after an "
          "incidental member would hide the regional event underneath")

    print("\nand it is not filed as noise")
    check("a conclusive single-machine fault clears P2",
          CLASS_FLOOR[IncidentClass.ASSET_FAILURE] >= 50.0,
          "three of the four things severity measures are proxies for 'how "
          "much of the network is involved', and a failed pump scores near "
          "zero on all three")

    print("\nhow it reads in the report")
    from das2.incident.parameters import inline_summary
    from das2.report.pdf import _asset_channels

    real = _member(AnomalyType.ASSET_NOT_DELIVERING)
    real.signals[0].detail.update({
        "output_kind": "flow", "output_unit": "L/s", "duty_unit": "A",
        "duty_kind": "current", "normal_output": 23.99,
        "observed_output": 0.03, "running_duty": 40.95, "observed_duty": 11.98,
    })
    incident = _incident([real])
    check("the column names the machine, not the run-state bit",
          inline_summary(incident) == "BedokPS unit 1 — not delivering",
          "grouped by equipment class it reads 'Digital Status 1', which "
          "names the least interesting of the three channels involved")

    channels = _asset_channels(incident)
    labels = [c[0] for c in channels]
    check("the anatomy table is the machine's channels",
          labels == ["flow", "motor current", "run status"], str(labels))
    check("with the numbers that decided it",
          channels[0][1] == "23.99 L/s" and channels[1][2] == "11.98 A",
          f"{channels[0][1]} -> {channels[0][2]}, "
          f"{channels[1][1]} -> {channels[1][2]}")
    check("and it says the instruments were right all along",
          "reported correctly" in channels[2][3],
          "that is the finding, not a footnote")
    check("no channel table for anything else",
          not _asset_channels(_incident([_member(AnomalyType.LEVEL_SHIFT)])))

    print("\nclustered by machine, not by map")
    asset_a = _member(AnomalyType.ASSET_NOT_DELIVERING, asset="BedokPS unit 1")
    asset_b = _member(AnomalyType.ASSET_ENERGISED_WHEN_OFF, key="K2",
                      asset="BedokPS unit 2")
    other = _member(AnomalyType.LEVEL_SHIFT, key="K3")
    clusters, consumed = cluster_by_asset([asset_a, asset_b, other])
    check("one cluster per machine", len(clusters) == 2, f"{len(clusters)}")
    check("two units at one site are not merged",
          all(len(c.members) == 1 for c in clusters),
          "they are two machines and two callouts")
    check("only the asset findings are taken", len(consumed) == 2,
          "everything else still goes through the spatial pass")
    check("a machine is drawn as a point, not an area",
          all(c.radius_m == 0.0 for c in clusters),
          "a radius would put a circle on the map implying an affected area "
          "that does not exist")

    print(f"\n{passed} passed, {failed} failed.")
    return 1 if failed else 0


def _incident(members):
    from types import SimpleNamespace as NS
    return NS(cluster=Cluster(members=members, region="East"))


def _member(atype: AnomalyType, *, key: str = "K1", site: str = "BedokPS",
            asset: str = "BedokPS unit 1") -> SensorAnomaly:
    signal = Signal(type=atype, start=T0, end=T0 + timedelta(hours=3),
                    detector="asset", magnitude=20.0, unit="A",
                    detail={"asset": asset,
                            "verdict": "running and not delivering: flow has "
                                       "gone while the motor draws current "
                                       "below its own normal"})
    return SensorAnomaly(
        sensor=SensorMeta(sensor_key=key, description=f"{site}-Pump-Run-Status",
                          equipment="DigitalStatus", site=site, region="East",
                          latitude=1.33, longitude=103.93),
        start=signal.start, end=signal.end, dominant_type=atype,
        signals=[signal],
        severity=PhysicalSeverity(deviation=20.0, unit="A", duration_s=10800))


if __name__ == "__main__":
    sys.exit(main())
