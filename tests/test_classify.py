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

    # ---------------------------------------------------------------- #
    print("\nunderscore is a word character, and `\\b` does not know it")
    # The defect this section exists for. In every flavour of regex `_` is a
    # WORD character, so `\b` sees no boundary between an underscore and a
    # letter: `\bwl\b` cannot match `CWS001_WL_Alex Canal Sub Drain B`. PUB's
    # CWS/EWS naming is underscore-delimited throughout, and ~1,600 canal
    # water-level sensors -- the single most important parameter on a drainage
    # estate -- sat in UNCLASSIFIED because of it.
    #
    # It was never one rule. FIFTY-TWO patterns across nearly every class had
    # the same blindness; `\bwl\b` was just the one with enough sensors behind
    # it to show up in a coverage report. So this asserts the general
    # property, not the one symptom -- a new rule written with `\b` tomorrow
    # must not reintroduce it.
    from das2.io.classify import expand_boundaries  # noqa: E402

    broken = []
    for equipment, _compiled, raw, _allowed in clf._rules:
        token = re.fullmatch(r"\\b([a-z0-9]+)\\b", raw)
        if not token:
            continue
        probe = f"SITE01_{token.group(1).upper()}_Somewhere Rd"
        if not re.search(expand_boundaries(raw), probe, re.IGNORECASE):
            broken.append((equipment, raw))
    check("every whole-word rule matches its underscore-delimited form",
          not broken,
          f"{len(broken)} still blind" if broken
          else "52 patterns were blind before this fix")

    check("`\\b` still refuses a match mid-word",
          not re.search(expand_boundaries(r"\bflow\b"), "airflowrate", re.I),
          "widening it to match anywhere would classify by substring")
    check("a digit counts as part of the word",
          not re.search(expand_boundaries(r"\bkw\b"), "SITE_KW9_X", re.I))
    check("`\\B` is left alone",
          expand_boundaries(r"a\Bb") == r"a\Bb")
    check("a backspace inside a character class is left alone",
          expand_boundaries(r"[\b]") == r"[\b]",
          "rewriting it there would be a syntax error, not a fix")
    check("an escaped backslash is not mistaken for a boundary",
          expand_boundaries(r"a\\bc") == r"a\\bc")

    print("\ncanal and drain water level is its own parameter")
    # Reported verbatim from the client's run, where every one of these came
    # back UNCLASSIFIED.
    for desc in ("CWS001_WL_Alex Canal Sub Drain B(Prince Phillip Ave)",
                 "EWS008_WL_Seletar Rd/Neram Rd",
                 "CWS018_WL_Geylang River (Paya Laber Rd)",
                 "CWS041_WL_Camp Rd OD(Rochalie Dr)"):
        check(f"{desc[:34]:36s} -> CanalLevel",
              clf.classify(desc, 1).equipment == "CanalLevel")

    check("a reservoir level is NOT folded in with them",
          clf.classify("Kranji1PS-Service-Reservoir-Level", 1).equipment == "Level",
          "both are metres of water; they answer different questions")
    check("CanalLevel is ordered above Level",
          [e for e, _, _, _ in clf._rules].index("CanalLevel")
          < [e for e, _, _, _ in clf._rules].index("Level"),
          "_WL_ matches Level's rule too, so first-match-wins keeps them apart")
    check("a WL setpoint is still config, not a measurement",
          clf.classify("MARINA_WL_SETPOINT_HI", 1).equipment == "Setpoint",
          "an engineer editing a limit must never page anyone")
    check("CanalLevel starts non-alerting",
          not clf.classes["CanalLevel"].alertable,
          "classifying them is the fix; promoting ~1,600 sensors is a "
          "separate, measurable step")
    check("but they are analysed and counted",
          not clf.classes["CanalLevel"].is_config,
          "nothing is hidden while the flag is off")

    canal = [d for d, _, c in classified if c.equipment == "CanalLevel"]
    check("the real inventory's CWS/EWS network is recovered",
          len(canal) >= 250, f"{len(canal)} sensors")
    check("and none of them is still UNCLASSIFIED",
          not [d for d in canal if d.upper().startswith(("CWS", "EWS"))
               and cls_of(d) == UNCLASSIFIED])

    # --- the waterway sonde suite ------------------------------------------ #
    # A 28-station network on the rivers, lakes and basins, all of it landing
    # in UNCLASSIFIED and being dropped before detection -- the only direct
    # measurement of water QUALITY, as opposed to water movement, in the whole
    # estate.
    print("\nwater quality: the sonde suite is classified")
    for desc, expected in (
            ("Bedok-Turb", "Turbidity"),
            ("Kallang Basin-Chl", "Chlorophyll"),
            ("Jurong Lake-BGAlgae", "BlueGreenAlgae"),
            ("Geylang River-NH3", "Ammonia"),
            ("Hougang Ave 7-NH4-N", "Ammonia"),
            ("Kallang River-NO3", "Nitrate"),
            ("Bedok-Salinity", "Salinity"),
            ("Bedok-TDS", "TotalDissolvedSolids"),
            ("Bedok-Depth", "Depth"),
            ("Bedok-RH", "Humidity"),
    ):
        got = clf.classify(desc, 1).equipment
        check(f"{desc} is {expected}", got == expected, f"got {got}")

    check("chlorophyll does not swallow chlorine",
          clf.classify("Pandan2PS-Chlorine-Residual", 1).equipment == "Chlorine",
          "`Chl` and `chlorine` are different measurements, and the boundary "
          "after the l is what keeps them apart")
    check("every new water-quality class starts non-alerting",
          not any(clf.classify(f"X-{p}", 1).meta.alertable for p in
                  ("Turb", "Chl", "BGAlgae", "NH3", "NO3", "Salinity", "TDS",
                   "Depth", "RH")),
          "in the inventory snapshot every one of these reads exactly 5000 "
          "while the temperature beside it reads 25 -- a no-data sentinel, "
          "which alertable would make a RANGE_VIOLATION on every sonde")
    check("sonde depth is not filed with the canal and reservoir levels",
          clf.classify("Bedok-Depth", 1).equipment != "Level",
          "Level is alertable and carries the water fleet; a buoy's "
          "submersion depth belongs in neither")

    print("\nalarm thresholds are configuration, not measurements")
    for desc in ("UpperSeletarPS-SEASON1-STOCK-HA",
                 "Murai-SEASON3-STOCK-LA",
                 "PandanTG-SEASON1-STOCK-TOL"):
        got = clf.classify(desc, 1).equipment
        check(f"{desc.split('-', 1)[1]} is a Setpoint", got == "Setpoint",
              f"got {got} -- `\\bstock\\b` was filing these as reservoir "
              f"levels, and a threshold that never moves is a permanent "
              f"FLATLINE")

    # The negative case, and the one most likely to be "fixed" by someone
    # later. `-SP` looks like the obvious next win: 1,824 points carry it.
    # It is not safe, because it does not mean setpoint consistently --
    # at the MRRS sites nearly every point ends in it, including live valve
    # states and alarm bits, and `MacRitchie-LT-Pump1-Status-SP` is that
    # pump's ONLY status point.
    print("\nand -SP is deliberately NOT treated as one")
    for desc in ("MRRS_ALEX VALVE (Open)-SP",
                 "MRRS_ALEX RTU FAILED-SP",
                 "MacRitchie-LT-Pump1-Status-SP"):
        got = clf.classify(desc, 3).equipment
        check(f"{desc} stays a live point", got != "Setpoint", f"got {got}")
    check("no run-state point is reachable by the Setpoint rule",
          clf.classify("MacRitchie-LT-Pump1-Status-SP", 3).equipment
          != "Setpoint",
          "the asset layer pairs a machine on its run state; taking it into "
          "config would silently remove the machine from detection")

    print("\nrun-hour counters are counters")
    check("HRS-C is Runtime",
          clf.classify("BedokPS-P1-HRS-C", 1).equipment == "Runtime",
          "131 pump hour counters; a counter only ever climbs, so the "
          "ordinary detectors are actively wrong on it")

    print("\nAll classifier tests passed.")


if __name__ == "__main__":
    main()
