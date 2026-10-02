#!/usr/bin/env python3
"""
The paging line: sensor health, and nothing else.

The client narrowed the scope after seeing a real run:

    "i only want some detection that makes a suggestion of area where might
     need a close look, we dont want explicit potential like operational
     events being alerted as anomaly, or some extreme weather condition that
     result in abnormal data intake to be alerted as anomaly, we want to
     anticipate the potential unnormal, or abnormal behavior of the sensor
     based on the stats, but not the operational event, or weather condition,
     or close/on valve things like that"

On the run that prompted it, all seven P1 alerts were REGIONAL_EVENT -- the
water moving across an area, correctly detected, and none of it the client's
problem. Under this scope not one of those seven pages.

What is easy to get wrong
-------------------------
"Stop alerting on the weather" and "only alert on sensor health" pull in
opposite directions, and deleting the weather rules satisfies the first while
breaking the second. Rain is what EXCUSES a storm's worth of level excursions;
without it they arrive at the sensor-health rules with nothing to account for
them, and a downpour is reported as a dozen broken level sensors -- which is
the very thing the client asked not to receive.

So the three rules that recognise an operational event survive. They just
return one verdict, OUT_OF_SCOPE, which is recorded with its evidence and
never sent. These tests pin both halves: that the four sensor-health classes
page, and that the rules keeping everything else off the line still fire.

Run:  python3 tests/test_scope.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from das2.incident.triage import (  # noqa: E402
    CORRELATION_STRONG,
    RAIN_EXPLAINS_MM,
    classify,
    severity,
)
from das2.models import (  # noqa: E402
    PAGEABLE_CLASSES,
    AnomalyType,
    Cluster,
    Incident,
    IncidentClass,
    PhysicalSeverity,
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


def member(key: str, site: str, equipment: str, atype: AnomalyType,
           *, minutes: int = 60, rtu: str | None = None) -> SensorAnomaly:
    return SensorAnomaly(
        sensor=SensorMeta(sensor_key=key, description=f"{site}-{equipment}",
                          equipment=equipment, site=site, region="East",
                          rtu_number=rtu),
        start=T0, end=T0 + timedelta(minutes=minutes),
        dominant_type=atype, severity=PhysicalSeverity(),
    )


def area(equipment: str, atype: AnomalyType, n: int = 4) -> Cluster:
    """n sensors at n DIFFERENT sites -- the shape of an area event."""
    return Cluster(
        members=[member(f"k{i}", f"Site{i}", equipment, atype)
                 for i in range(n)],
        region="East", radius_m=2400.0)


def main() -> int:
    print("\nthe four classes that reach a person")
    check("the allow-list is exactly the sensor-health classes",
          PAGEABLE_CLASSES == {IncidentClass.SENSOR_FAULT,
                               IncidentClass.DRIFT_MAINTENANCE,
                               IncidentClass.INSTRUMENT_CONFLICT,
                               IncidentClass.TELEMETRY_OUTAGE},
          "every one is a statement about an instrument or about the path its "
          "readings travel on")
    for klass in IncidentClass:
        inc = Incident(incident_id="x", cluster=area("Level",
                                                    AnomalyType.FLATLINE),
                       incident_class=klass, severity=95.0)
        if klass not in PAGEABLE_CLASSES:
            check(f"{klass.value} does not page even at severity 95",
                  not inc.should_alert)

    print("\nthe water moving across an area is recognised, and not paged")
    klass, why = classify(area("Level", AnomalyType.LEVEL_SHIFT))
    check("four sites, all level shifts, is an area event",
          klass is IncidentClass.OUT_OF_SCOPE, f"({klass.value})")
    check("and the evidence says so in words an operator can overrule",
          any("area event" in w for w in why), f"({why})")
    check("it names the sites, so a reader can check it",
          any("Site0" in w for w in why))
    inc = Incident(incident_id="x", cluster=area("Level",
                                                AnomalyType.LEVEL_SHIFT),
                   incident_class=klass,
                   severity=severity(area("Level", AnomalyType.LEVEL_SHIFT),
                                     klass))
    check("nobody is paged", not inc.should_alert)
    check("and it cannot head the report either",
          inc.priority.value in ("P3", "P4"),
          f"(severity {inc.severity}, {inc.priority.value}; it used to be the "
          f"loudest verdict in the system at weight 1.00)")

    print("\n  and four UNRELATED faults at four sites are still four faults")
    # The distinction that now decides whether anyone is told at all. Before
    # the paging line narrowed, mislabelling these as an area event cost a
    # wasted trip; now it costs the alert entirely.
    mixed = Cluster(members=[
        member("a", "SiteA", "Flowrate", AnomalyType.FLATLINE),
        member("b", "SiteB", "Pressure", AnomalyType.STALE),
        member("c", "SiteC", "Level", AnomalyType.QUANTISATION_COLLAPSE),
        member("d", "SiteD", "Conductivity", AnomalyType.ATTENUATED_SIGNAL),
    ], region="East", radius_m=2400.0)
    klass, why = classify(mixed)
    check("four broken instruments are not an area event",
          klass is IncidentClass.SENSOR_FAULT, f"({klass.value})")
    check("so they reach somebody",
          Incident(incident_id="x", cluster=mixed,
                   incident_class=klass).should_alert,
          "the process-member requirement is what separates these two cases")

    print("\nrain: still measured, because it is what stops a storm paging")
    wet = area("Level", AnomalyType.LEVEL_SHIFT)
    klass, why = classify(wet, rainfall_mm=RAIN_EXPLAINS_MM + 8.0)
    check("a level shift in a downpour is the weather",
          klass is IncidentClass.OUT_OF_SCOPE, f"({klass.value})")
    check("and the millimetres are quoted, so it can be disputed",
          any("mm of rain" in w for w in why), f"({why})")

    klass, _ = classify(wet, rainfall_mm=0.0)
    check("the same cluster in the dry is not excused on rain",
          klass is IncidentClass.OUT_OF_SCOPE
          and not any("rain" in w for w in _),
          "(still an area event -- but for the area reason, not the wet one)")

    print("\n  an over-range reading in a flood is the gauge doing its job")
    # The one case where the two halves of the client's instruction collide.
    # RANGE_VIOLATION is a sensor-health type, so a flood pushing canal levels
    # over their configured high limit lands squarely on the dispatch line --
    # with nothing wrong with any instrument.
    flood = area("CanalLevel", AnomalyType.RANGE_VIOLATION)
    klass, why = classify(flood, rainfall_mm=RAIN_EXPLAINS_MM + 40.0)
    check("a canal over its high limit in 42 mm of rain is not a sensor fault",
          klass is IncidentClass.OUT_OF_SCOPE, f"({klass.value})")

    klass, _ = classify(flood, rainfall_mm=0.0)
    check("but the same reading with no rain IS a sensor fault",
          klass is IncidentClass.SENSOR_FAULT, f"({klass.value})")

    print("\n  and rain is not a blanket excuse: the equipment has to be wet")
    # Rain raises a canal, a flow and a turbidity reading. It does not raise a
    # motor voltage, so an over-range there in the same storm is still real.
    dry_kit = area("Voltage", AnomalyType.RANGE_VIOLATION)
    klass, _ = classify(dry_kit, rainfall_mm=RAIN_EXPLAINS_MM + 40.0)
    check("a motor voltage over its limit in a storm still pages",
          klass is IncidentClass.SENSOR_FAULT, f"({klass.value})")

    print("\n  rain never excuses a transmitter that stopped reporting")
    quiet = area("Level", AnomalyType.STALE)
    klass, _ = classify(quiet, rainfall_mm=RAIN_EXPLAINS_MM + 40.0)
    check("a stale sensor in a downpour is still a broken sensor",
          klass is IncidentClass.SENSOR_FAULT, f"({klass.value})")

    print("\nan operator opening a valve: the neighbours give it away")
    # One sensor, a process-type finding, neighbours moving with it. Not an
    # area event (one site), not rain (dry), not an instrument fault (the type
    # is not a health type) -- the client's "close/on valve things like that".
    valve = Cluster(members=[member("v", "SiteV", "Flowrate",
                                    AnomalyType.REVERSE_FLOW)],
                    region="East")
    klass, why = classify(valve, neighbour_correlation=CORRELATION_STRONG + 0.2)
    check("neighbours moving with it makes it operational",
          klass is IncidentClass.OUT_OF_SCOPE, f"({klass.value})")
    check("and the correlation is quoted",
          any("r=0.8" in w for w in why), f"({why})")

    print("\n  but correlation must never answer ahead of a broken instrument")
    # The ordering that matters most. A frozen transmitter does not stop being
    # frozen because a sensor two kilometres away moved in sympathy, and if
    # this branch ran first every definitive fault in a busy region would be
    # excused as the process.
    frozen = Cluster(members=[member("f", "SiteF", "Flowrate",
                                     AnomalyType.FLATLINE)], region="East")
    klass, _ = classify(frozen, neighbour_correlation=0.95)
    check("a flatlined meter pages even at r=0.95",
          klass is IncidentClass.SENSOR_FAULT, f"({klass.value})")

    print("\nthe three retired verdicts are unreachable")
    names = {k.value for k in IncidentClass}
    for retired in ("REGIONAL_EVENT", "PROCESS_EVENT", "WEATHER_DRIVEN"):
        check(f"{retired} is no longer a verdict", retired not in names)
    check("all three paths now land on one recorded, unsent class",
          IncidentClass.OUT_OF_SCOPE in names
          and "nobody is paged" in Incident(
              incident_id="x", cluster=valve,
              incident_class=IncidentClass.OUT_OF_SCOPE).recommendation)

    print(f"\n{passed} passed, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
