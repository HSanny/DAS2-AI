#!/usr/bin/env python3
"""
Event signatures: what the moving parameters usually mean.

`parameters.py` answers *what moved* -- seven canal levels rose 0.82 m while
three flows rose. This layer answers *what a water engineer would call that*,
and it is the first thing in the system that states an opinion the data does
not contain.

That is why the tests below are weighted the way they are. Only a handful
check that a rule fires; the rest check the fence around it:

* **A signature cannot change a decision.** The class, the priority, the
  severity and the recommendation must be byte-identical with a signature
  attached and without one. A wrong evidence class costs a wasted trip; a
  wrong reading able to stop a dispatch costs a flood. `classify()` takes a
  `Cluster` and never an `Incident`, so it structurally cannot see a
  signature -- this pins that it stays that way.
* **A rule with no falsifier is refused at load.** Not rendered with an empty
  field: refused. The file is meant to be edited by PUB, and the one rule that
  has to survive other people's editing is the one that keeps an unfalsifiable
  claim out of an operator's hands.
* **No match is a valid answer**, and the common one. Inventing a reading to
  avoid an empty field is how a table of rules becomes a horoscope.
* **`unknown` rain is not `dry` rain.** No gauge within 10 km is not evidence
  that it was not raining, and a rule keyed on `dry` must not fire on it.
  Getting this wrong turns "we cannot see the weather" into "the rise has no
  rain to explain it", which is the sentence that sends someone out.

Run:  python3 tests/test_signature.py
"""

import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from das2.incident import signature                    # noqa: E402
from das2.incident.build import build_incident         # noqa: E402
from das2.models import (                              # noqa: E402
    AnomalyType,
    Cluster,
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


def member(equipment: str, move: float, *, site: str, unit: str = "m",
           atype: AnomalyType = AnomalyType.LEVEL_SHIFT) -> SensorAnomaly:
    key = f"{site}-{equipment}-{move}"
    return SensorAnomaly(
        sensor=SensorMeta(sensor_key=key, description=key, equipment=equipment,
                          site=site, region="East", unit=unit,
                          latitude=1.33, longitude=103.93),
        start=T0, end=T0 + timedelta(hours=3), dominant_type=atype,
        severity=PhysicalSeverity(deviation=abs(move), signed_deviation=move,
                                  unit=unit, duration_s=10800),
    )


def incident_of(*members) -> object:
    return build_incident(Cluster(members=list(members), region="East"),
                          now=T0 + timedelta(hours=3))


def write_rules(text: str) -> str:
    handle = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    handle.write(text)
    handle.close()
    signature.load.cache_clear()
    return handle.name


def main() -> int:
    print("\nthe shipped table loads, and every entry can be falsified")
    table = signature.load()
    check("the file is readable and non-empty", len(table) >= 4,
          f"{len(table)} signature(s)")
    check("every entry carries a falsifier",
          all(s.would_change_it for s in table))
    check("every entry carries a status we know how to render",
          all(s.status in signature.LEAD_IN for s in table))
    check("nothing ships as confirmed",
          not any(s.validated for s in table),
          "confirmed means PUB has checked it against real incidents; "
          "nothing here has been")
    check("ids are unique", len({s.id for s in table}) == len(table))

    print("\na rule with no falsifier is refused, not rendered empty")
    path = write_rules(
        "version: 1\n"
        "signatures:\n"
        "  - id: NO_FALSIFIER\n"
        "    name: Something certain\n"
        "    when: {parameters: [{any_of: [Pressure], min_sensors: 1}]}\n"
        "    reads_as: It is definitely a burst.\n"
        "  - id: PROPER\n"
        "    name: Pressure moving\n"
        "    when: {parameters: [{any_of: [Pressure], min_sensors: 1}]}\n"
        "    reads_as: Pressure moved.\n"
        "    would_change_it: Only one sensor moving.\n")
    loaded = signature.load(path)
    check("the unfalsifiable entry is dropped",
          [s.id for s in loaded] == ["PROPER"],
          "an operator cannot check a claim with nothing to check it against")

    print("\nfirst match wins, so file order is the rule author's lever")
    path = write_rules(
        "version: 1\n"
        "signatures:\n"
        "  - id: SPECIFIC\n"
        "    name: Two pressures falling\n"
        "    when:\n"
        "      parameters: [{any_of: [Pressure], direction: falling, "
        "min_sensors: 2}]\n"
        "    would_change_it: One sensor.\n"
        "  - id: GENERAL\n"
        "    name: Any pressure\n"
        "    when: {parameters: [{any_of: [Pressure], min_sensors: 1}]}\n"
        "    would_change_it: No pressure.\n")
    two_falling = incident_of(member("Pressure", -0.8, site="Bedok PS", unit="bar"),
                              member("Pressure", -0.6, site="Tampines PS", unit="bar"))
    found = signature.match(two_falling, path=path)
    check("the specific rule answers first",
          found is not None and found.signature.id == "SPECIFIC")
    one_rising = incident_of(member("Pressure", 0.4, site="Bedok PS", unit="bar"))
    found = signature.match(one_rising, path=path)
    check("and the general one catches what it does not",
          found is not None and found.signature.id == "GENERAL")

    print("\nthe clauses mean what they say")
    path = write_rules(
        "version: 1\n"
        "signatures:\n"
        "  - id: DRY_RISE\n"
        "    name: Rise with no rain\n"
        "    when:\n"
        "      parameters: [{any_of: [CanalLevel, Level], direction: rising, "
        "min_sensors: 2}]\n"
        "      rain_context: dry\n"
        "      min_sites: 2\n"
        "    would_change_it: Rain at a gauge outside the radius.\n")
    two_sites = incident_of(member("CanalLevel", 0.5, site="Bedok Canal"),
                            member("CanalLevel", 0.4, site="Siglap Canal"))
    check("it fires when every clause holds",
          signature.match(two_sites, rain_context="dry", path=path) is not None)
    check("wet rain does not satisfy a dry rule",
          signature.match(two_sites, rain_context="wet", path=path) is None)
    check("and neither does UNKNOWN rain",
          signature.match(two_sites, rain_context="unknown", path=path) is None,
          "no gauge in range is not evidence that it was not raining; "
          "reading it as dry is how 'we cannot see' becomes 'nothing "
          "explains this'")

    one_site = incident_of(member("CanalLevel", 0.5, site="Bedok Canal"),
                           member("CanalLevel", 0.4, site="Bedok Canal"))
    check("min_sites counts SITES, not sensors",
          signature.match(one_site, rain_context="dry", path=path) is None,
          "two instruments in one canal are one place")

    falling = incident_of(member("CanalLevel", -0.5, site="Bedok Canal"),
                          member("CanalLevel", -0.4, site="Siglap Canal"))
    check("a falling group does not satisfy a rising clause",
          signature.match(falling, rain_context="dry", path=path) is None,
          "the sign is the whole finding here")

    mixed = incident_of(member("CanalLevel", 0.5, site="Bedok Canal"),
                        member("CanalLevel", -0.4, site="Siglap Canal"))
    check("a group whose sensors disagree is mixed, and matches neither",
          signature.match(mixed, rain_context="dry", path=path) is None)

    print("\nno match is a real answer")
    pumps = incident_of(member("Vibration", 3.0, site="Bedok PS", unit="mm/s",
                               atype=AnomalyType.NOISE_BURST))
    check("an incident nothing describes gets no reading",
          signature.match(pumps, rain_context="dry", path=path) is None,
          "an empty field is honest; a fabricated sentence is a horoscope")
    check("and an empty incident is refused before any rule is consulted",
          signature.match(incident_of(), path=path) is None)

    print("\na match shows its working, and keeps its hedge")
    found = signature.match(two_sites, rain_context="dry", path=path)
    check("the headline hedges an unvalidated rule",
          found.headline.startswith("Looks like"), found.headline)
    check("the clauses that matched are named",
          any("2" in m for m in found.matched)
          and any("dry" in m for m in found.matched),
          " · ".join(found.matched))
    check("the caveat says it does not decide",
          "does not decide" in found.caveat)

    detail = signature.as_detail(found)
    check("the falsifier travels into the incident detail",
          detail["would_change_it"].startswith("Rain at a gauge"),
          "leaving it in the config file makes the reading uncheckable "
          "exactly where it is read")
    check("and so does the status", detail["status"] == "unvalidated")

    supported = signature.Signature(id="X", name="A supported reading",
                                    status="supported", would_change_it="x")
    check("a supported rule is introduced differently",
          signature.SignatureMatch(signature=supported).headline
          == "Reads as: a supported reading")

    print("\nIT CANNOT CHANGE A DECISION")
    # The whole safety argument, pinned. Everything an operator acts on is
    # computed from the cluster; the reading is attached afterwards and is
    # never read back.
    incident = incident_of(member("Pressure", -0.8, site="Bedok PS", unit="bar"),
                           member("Pressure", -0.6, site="Tampines PS", unit="bar"))
    before = (incident.incident_class, incident.priority, incident.recommendation,
              round(incident.severity, 6), incident.narrative,
              incident.should_alert, incident.should_dispatch)

    found = signature.match(incident)
    check("the fixture does match something, or this test proves nothing",
          found is not None,
          found.signature.id if found else "no match")
    incident.detail[signature.DETAIL_KEY] = signature.as_detail(found)

    after = (incident.incident_class, incident.priority, incident.recommendation,
             round(incident.severity, 6), incident.narrative,
             incident.should_alert, incident.should_dispatch)
    check("class, priority, severity, recommendation and dispatch are unmoved",
          before == after,
          "a reading that could suppress a dispatch is a flood waiting for "
          "the one incident the rule was wrong about")

    from das2.alerting.telegram import compose
    message = compose(incident)
    check("the alert carries the reading",
          "Looks like: pressure falling across sites" in message)
    check("and what would falsify it",
          "Would change this reading" in message)
    check("the recommendation is still read first",
          message.index(incident.recommendation) < message.index("Looks like"),
          "the decision was computed without the reading and outranks it")
    check("the rule's own action is NOT sent to an operator",
          found.signature.action not in message,
          "an unvalidated rule telling someone what to do is the failure "
          "this layer exists to avoid; it stays on the dashboard until the "
          "rule is confirmed")

    check("matching does not mutate the incident it was asked about",
          _unchanged_by_match(),
          "match() reads; the caller decides whether to file the result")

    check("the decision path cannot even see a signature",
          "detail" not in _classify_parameters(),
          "classify() takes a Cluster; there is no argument through which a "
          "reading could reach it")

    print("\nthe layer fails quietly, never loudly")
    signature.load.cache_clear()
    check("a missing file yields no signatures rather than an exception",
          signature.load("/nonexistent/event_signatures.yaml") == ())
    bad = write_rules("signatures: [this is not a mapping\n")
    check("and so does a malformed one", signature.load(bad) == (),
          "a syntax error in an operator-edited file must not take the "
          "hourly run down with it")
    signature.load.cache_clear()

    print("\nrun-level counts")
    counts = signature.summary([found, found, None])
    check("matched and unmatched are both reported",
          counts["matched"] == 2 and counts["unmatched"] == 1)
    check("status is counted, so the report can say how much is unvalidated",
          counts["by_status"].get("unvalidated") == 2)

    print(f"\n{passed} passed, {failed} failed.")
    return 1 if failed else 0


def _unchanged_by_match() -> bool:
    incident = incident_of(member("Pressure", -0.8, site="Bedok PS", unit="bar"),
                           member("Pressure", -0.6, site="Tampines PS", unit="bar"))
    before = dict(incident.detail)
    signature.match(incident)
    return incident.detail == before


def _classify_parameters() -> set[str]:
    import inspect

    from das2.incident.triage import classify

    return set(inspect.signature(classify).parameters)


if __name__ == "__main__":
    sys.exit(main())
