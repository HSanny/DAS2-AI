"""
Tests for the full-coverage equipment classifier (Phase 3).

Measured against the REAL inventory in docker_ready/processed/dim.csv (13,567
sensors), because that is the input this has to handle and synthetic names
would prove nothing.

Two properties matter equally:
  * coverage must go up -- v1 left 74% of analog sensors unanalysed;
  * precision must not go down -- a sensor filed in the wrong class gets the
    wrong detector profile and the wrong physical range, which is worse than
    leaving it unclassified.

Run:  python3 tests/test_classify.py
"""

import csv
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from das2.io.classify import UNCLASSIFIED, EquipmentClassifier, get_classifier  # noqa: E402

DIM = REPO / "docker_ready" / "processed" / "dim.csv"

#: v1's measured baseline: 1,357 of 5,204 analog sensors classified.
V1_ANALOG_CLASSIFIED = 1357
V1_ANALOG_TOTAL = 5204


def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
    if not cond:
        raise SystemExit(1)


def main():
    clf = get_classifier()
    rows = list(csv.DictReader(open(DIM)))
    classified = [(r["Description"], r["RawType"], clf.classify(r["Description"], r["RawType"]))
                  for r in rows]

    def cls_of(desc):
        return next(c.equipment for d, _, c in classified if d == desc)

    print("coverage against the real inventory")
    rep = clf.coverage_report((r["Description"], r["RawType"]) for r in rows)
    check("whole inventory processed", rep["total"] == len(rows), f"({rep['total']:,})")
    check("analog count matches v1's split", rep["analog"] == V1_ANALOG_TOTAL,
          f"({rep['analog']:,})")
    check("coverage beats v1 by a wide margin",
          rep["analog_classified"] >= 2 * V1_ANALOG_CLASSIFIED,
          f"(v1 {V1_ANALOG_CLASSIFIED:,} = 26.1%  ->  "
          f"das2 {rep['analog_classified']:,} = {rep['analog_coverage_pct']}%)")
    check("unclassified is reported, not hidden",
          rep["unclassified"] > 0 and rep["unclassified_examples"],
          "(coverage is a tracked metric, not a silent default)")

    print("\nthe classes v1 discarded entirely are now recovered")
    for equipment, v1_count, minimum in [
        ("Vibration", 0, 150),     # pump-failure precursor, highest-value PdM signal
        ("Rainfall", 0, 40),       # makes weather-aware triage possible, no external API
        ("Current", 0, 300),
        ("Energy", 0, 400),
        ("Level", 286, 500),
        ("Position", 0, 300),
    ]:
        n = rep["by_equipment"].get(equipment, 0)
        check(f"{equipment:<12} {n:>5} sensors", n >= minimum,
              f"(v1: {v1_count})")

    print("\nconfiguration is separated from measurement")
    # 86 analog setpoints and 8 simulation points exist. Anomaly-detecting a
    # setpoint pages someone for a deliberate engineer action.
    setpoints = [d for d, _, c in classified if c.equipment == "Setpoint"]
    check("setpoints are classified as config", len(setpoints) >= 80, f"({len(setpoints)})")
    check("setpoints never alert", not clf.classes["Setpoint"].alertable)
    check("setpoints are excluded from analysis altogether",
          not clf.classify("MRRS_THOMSON HI SETPOINT", 1).analysable,
          "(its value changes only when an engineer edits it)")
    check("an alarm-limit setpoint beats the measurement rule",
          cls_of("LowerSeletarPS-Raw-Water-Pump3-Current-low-alarm-setpoint") == "Setpoint",
          "(not Current)")
    check("simulation points are config too",
          clf.classify("LowerSeletarTG-Tide-Level-Simulation", 1).meta.is_config)

    print("\ncounters are separated from process values")
    # kWh and run-hours only climb. A flat counter means the plant is idle,
    # which is normal, and a drop is a rollover, not a fault.
    check("kWh is a counter", clf.classes["Energy"].is_counter)
    check("run hours is a counter", clf.classes["Runtime"].is_counter)
    check("neither is treated as a measurement",
          not clf.classes["Energy"].is_measurement
          and not clf.classes["Runtime"].is_measurement)

    print("\nload-bearing rule orderings")
    for desc, expected, why in [
        ("PunggolSerangoon-6.6KV INCOMER 2 F14 KWH", "Energy",
         "KV must not steal an energy meter"),
        ("Kranji2PS-HT-IN-POWER-Current L3", "Current",
         "the trailing noun is what is measured"),
        ("PunggolSerangoon-6.6 KV INCOMER 1 P3 ACTIVE POWER", "Power", ""),
        ("MarinaRWPS-INCOMER2_Power_Factor_1", "PowerFactor",
         "must beat the Power rule"),
        ("LowerSeletarPS-Generator-incomer-kW-Deadband-setpoint", "Setpoint",
         "must beat both Power and Energy"),
    ]:
        check(f"{desc[:48]:<48} -> {expected}",
              cls_of(desc) == expected, why)

    print("\nambiguous SCADA abbreviations resolved by signal type")
    # "DO" is Dissolved Oxygen on analog points and Digital Output on digital
    # ones: 56 real water-quality sensors vs 36 output commands.
    do_analog = [d for d, rt, c in classified
                 if re.search(r"\bdo\b", d, re.I) and c.equipment == "Dissolved Oxygen"]
    do_other = [d for d, rt, c in classified
                if re.search(r"\bdo\b", d, re.I) and c.equipment != "Dissolved Oxygen"]
    check("analog DO points are dissolved oxygen", len(do_analog) >= 50,
          f"({len(do_analog)} -- v1 found only 19)")
    check("digital DO points are NOT dissolved oxygen", len(do_other) >= 30,
          f"({len(do_other)} Digital Output commands correctly excluded)")
    check("a specific output command is not water quality",
          cls_of("Kranji2PS-DO-CMD-P5_6") != "Dissolved Oxygen")
    check("a specific reservoir sensor is",
          cls_of("Marina Bay-DO") == "Dissolved Oxygen")

    busbars = [d for d, _, c in classified if re.search(r"\bbar\b", d, re.I)]
    check("electrical busbars are never pressure",
          all(cls_of(d) != "Pressure" for d in busbars),
          f"({len(busbars)} busbars -- 'bar' as a unit never appears here)")

    print("\nnew classes cannot page anyone until they have earned it")
    # Full coverage roughly quadruples the sensors under detection. Without
    # this, the coverage win would arrive as an alert flood.
    for equipment in ("Vibration", "Current", "Power", "Rainfall", "Position", "Speed"):
        check(f"{equipment} is analysed but not alertable",
              clf.classify(f"X-{equipment}", 1).analysable
              and not clf.classes[equipment].alertable)
    for equipment in ("Pressure", "Flowrate", "Temperature", "Voltage",
                      "Conductivity", "Dissolved Oxygen", "Level"):
        check(f"{equipment} stays alertable", clf.classes[equipment].alertable,
              "(was already in production)")

    print("\nranges are declared where they are meaningful")
    check("Pressure has a range", clf.classes["Pressure"].has_range)
    check("Power has none", not clf.classes["Power"].has_range,
          "(no sensible fleet-wide bound, so it is not range-checked)")
    check("UNCLASSIFIED has none", not clf.classes[UNCLASSIFIED].has_range)

    print("\nbehaviour on edge input")
    for desc in ("", "   ", "???", "12345"):
        r = clf.classify(desc, 1)
        check(f"{desc!r:<10} falls back cleanly", r.equipment == UNCLASSIFIED)
    check("missing RawType is treated as digital",
          clf.classify("Something-Pressure", None).signal_type == "Digital")
    check("RawType 5 is analog", clf.classify("X", 5).signal_type == "Analog")
    check("garbage RawType does not raise",
          clf.classify("X", "not-a-number").signal_type == "Digital")

    print("\nrules are data, editable without a release")
    custom = EquipmentClassifier(
        rules=[{"equipment": "Widget", "patterns": ["widget"]}],
        classes={},
    )
    check("a custom rule table works",
          custom.classify("Site-Widget-1", 1).equipment == "Widget")
    check("an unknown class degrades rather than raising",
          custom.classify("Site-Widget-1", 1).meta.kind == "unknown")

    print("\nAll classifier tests passed.")


if __name__ == "__main__":
    main()
