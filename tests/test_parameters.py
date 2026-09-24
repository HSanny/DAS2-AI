#!/usr/bin/env python3
"""
Which parameter is driving the event, and which way it moved.

The client's question, in their words: *"in a region, which kind of sensors or
what parameter are triggering the event ... what kind of issue or event, what
sensors are classified or clustered together"*. An incident used to answer
WHERE and HOW CONFIDENT and never WHAT -- twelve sensors at five sites, with
no way to tell canal levels from pump motors without opening the raw data.

Most of what was needed was already on the objects. One thing was not:
**direction**. `LEVEL_SHIFT` recorded that a step happened, never which way,
because fusion took `abs()` of the magnitude. Without the sign the
combinations that carry meaning collapse into one::

    level rising  +  flow rising   ->  drainage responding to rain
    level rising  +  flow falling  ->  something obstructing the channel

Same parameters, same magnitudes, opposite verdicts, opposite decisions.

Run:  python3 tests/test_parameters.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from das2.incident.parameters import (  # noqa: E402
    breakdown,
    display_name,
    inline_summary,
    net_direction,
    region_parameter_matrix,
)
from das2.models import (  # noqa: E402
    AnomalyType,
    PhysicalSeverity,
    QartodFlag,
    SensorAnomaly,
    SensorMeta,
)

passed = failed = 0
T0 = datetime(2026, 9, 23, 2, 0)


def check(label: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {label}" + (f"  ({detail})" if detail else ""))
    else:
        failed += 1
        print(f"  FAIL  {label}" + (f"  ({detail})" if detail else ""))


def anomaly(parameter: str, move: float, unit: str = "m", *,
            site: str = "Bedok PS", region: str = "East",
            atype: AnomalyType = AnomalyType.LEVEL_SHIFT) -> SensorAnomaly:
    return SensorAnomaly(
        sensor=SensorMeta(sensor_key=f"{parameter}{move}{site}",
                          description=f"{site}-{parameter}",
                          equipment=parameter, site=site, region=region,
                          unit=unit),
        start=T0, end=T0 + timedelta(hours=3), dominant_type=atype,
        severity=PhysicalSeverity(deviation=abs(move), signed_deviation=move,
                                  unit=unit, duration_s=10800),
    )


def incident(members) -> NS:
    return NS(cluster=NS(members=members))


def main() -> int:
    print("\nthe sign survives fusion")
    # The defect: `deviation = max(abs(...))` threw the direction away, so a
    # level shift of -0.4 bar and one of +0.4 bar arrived indistinguishable.
    up = PhysicalSeverity(deviation=0.4, signed_deviation=0.4, unit="bar")
    down = PhysicalSeverity(deviation=0.4, signed_deviation=-0.4, unit="bar")
    check("a rise reads as rising", up.direction == "rising")
    check("a fall reads as falling", down.direction == "falling")
    check("they are no longer the same number",
          up.signed_deviation != down.signed_deviation,
          "both were 0.4 before this")
    check("magnitude is unchanged, so ranking is unaffected",
          up.deviation == down.deviation == 0.4)
    check("no movement means no direction, not 'flat'",
          PhysicalSeverity(deviation=0.0).direction == "")

    print("\na drainage incident, broken down")
    # Seven canal levels and three flows rising together, two pressures
    # falling -- the shape of a catchment responding to rain.
    members = ([anomaly("CanalLevel", 0.8 + i * 0.02) for i in range(7)]
               + [anomaly("Flowrate", 24.0 + i, "L/s") for i in range(3)]
               + [anomaly("Pressure", -0.4, "bar") for _ in range(2)])
    groups = breakdown(incident(members))

    check("one group per parameter", [g.parameter for g in groups]
          == ["CanalLevel", "Flowrate", "Pressure"],
          "biggest first")
    check("counts are right", [g.count for g in groups] == [7, 3, 2])
    check("the levels are all rising", groups[0].direction == "rising")
    check("so are the flows", groups[1].direction == "rising")
    check("the pressures are falling", groups[2].direction == "falling")
    check("the typical move keeps its sign and unit",
          groups[0].move_text().startswith("+")
          and groups[0].move_text().endswith(" m"),
          groups[0].move_text())
    check("a falling group reads negative",
          groups[2].move_text() == "-0.4 bar", groups[2].move_text())
    check("direction is words, not only an arrow",
          groups[0].direction_text() == "all rising",
          "a glyph alone fails in greyscale and for colour-blind readers")

    print("\nthe median, not the mean")
    # One transmitter failing to full scale must not set the number the whole
    # group is described by.
    skewed = breakdown(incident(
        [anomaly("CanalLevel", 0.8) for _ in range(6)]
        + [anomaly("CanalLevel", 900.0)]))[0]
    check("one sensor reading full scale does not move the typical figure",
          abs(skewed.typical_move - 0.8) < 1e-9,
          f"median {skewed.typical_move}, mean would be "
          f"{(0.8 * 6 + 900) / 7:.1f}")

    print("\ndisagreement is reported, not averaged away")
    mixed = breakdown(incident(
        [anomaly("Level", 0.5), anomaly("Level", 0.5),
         anomaly("Level", -0.5), anomaly("Level", -0.5)]))[0]
    check("four sensors split two and two is mixed",
          mixed.direction == "mixed")
    check("and says how many disagree",
          "disagree" in mixed.direction_text(), mixed.direction_text())
    check("a mixed group still reports its magnitude",
          mixed.move_text() == "±0.5 m", mixed.move_text())
    check("rather than a dash, which would read as 'nothing moved'",
          mixed.move_text() != "" and not mixed.move_text().startswith("0"))

    two_of_three = breakdown(incident(
        [anomaly("Level", 0.5), anomaly("Level", 0.5),
         anomaly("Level", -0.5)]))[0]
    check("two of three agreeing is a direction, not mixed",
          two_of_three.direction == "rising",
          "the consensus bar is two thirds")

    print("\nparameters with no direction are honest about it")
    stale = breakdown(incident(
        [anomaly("Pump", 0.0, "", atype=AnomalyType.FLATLINE)
         for _ in range(2)]))[0]
    check("a flatlined sensor has not moved either way",
          stale.direction == "" and stale.direction_text() == "no direction")
    check("and reports no engineering move", stale.move_text() == "")
    check("but carries its QARTOD flag", stale.flag is QartodFlag.FAIL,
          "FLATLINE is a fact about the channel")
    check("an inferred finding is only SUSPECT",
          groups[0].flag is QartodFlag.SUSPECT)

    print("\nthe inline form, for a table column")
    summary = inline_summary(incident(members))
    check("it names the parameters in order", summary.startswith("Canal Level 7"))
    check("it carries direction per parameter", "▲" in summary
          and "▼" in summary, summary)
    check("it truncates rather than overflowing",
          "+" in inline_summary(incident(members), limit=2),
          inline_summary(incident(members), limit=2))
    check("camel case is read as words",
          display_name("CanalLevel") == "Canal Level")
    check("and a single word is left alone",
          display_name("UNCLASSIFIED") == "UNCLASSIFIED",
          "otherwise it becomes U N C L A S S I F I E D")

    print("\nregion by parameter, with direction")
    fleet = (members
             + [anomaly("CanalLevel", -0.3, region="North", site="Kranji 1 PS")
                for _ in range(4)]
             + [anomaly("Pump", 0.0, "", region="North", site="Kranji 1 PS",
                        atype=AnomalyType.STALE) for _ in range(3)])
    matrix = region_parameter_matrix(fleet)

    check("regions are rows", set(matrix) == {"East", "North"})
    check("parameters are columns",
          set(matrix["East"]) == {"CanalLevel", "Flowrate", "Pressure"})
    check("each cell splits the three directions",
          matrix["East"]["CanalLevel"] == {"rising": 7, "falling": 0, "flat": 0})
    check("a region pulling the other way is visible",
          matrix["North"]["CanalLevel"]
          == {"rising": 0, "falling": 4, "flat": 0},
          "'the East has 31 level findings' is a count; this is a direction")
    check("findings with no direction land in `flat`",
          matrix["North"]["Pump"]["flat"] == 3,
          "a stale sensor has not risen or fallen; folding it into either "
          "would invent a movement")
    check("net direction is rising minus falling",
          net_direction(matrix["East"]["CanalLevel"]) == 7
          and net_direction(matrix["North"]["CanalLevel"]) == -4)
    check("a balanced cell nets to zero, not to nothing",
          net_direction({"rising": 5, "falling": 5, "flat": 0}) == 0,
          "which is why the page prints the total beside the colour")

    print(f"\n{passed} passed, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
