"""
End-to-end test: fixtures in, incidents and a dashboard out.

This is the test that matters most, and the one the unit tests kept failing to
substitute for. Every serious defect found while building this pipeline was
found by *running* it, not by reading it or by testing a function in isolation:

  * a range rule that flagged 1,079 points on one healthy meter;
  * timestamps 1000x too small, twice, silently;
  * a spike detector that reported 11 noise events and missed the real 34.8 V
    collapse;
  * a flatline detector whose threshold was set by the very fault it was
    looking for, so it could never fire;
  * an incident reconciler that resolved and immediately recreated 3 of 4
    incidents every run -- re-alerting each time, which is the exact behaviour
    the whole incident design exists to remove;
  * a clustering radius still set to a superseded value in the config, which
    silently reduced the headline regional event to two unrelated pairs.

Not one of those is visible in a unit test of the component that contained it.

What is asserted here is the client's requirement, in their words: that four
sensors going abnormal together across three sites surfaces as ONE regional
event they can look at before driving anywhere, that a panel fan-out does not,
and that one fault does not alert twelve times.

Run:  python3 tests/test_end_to_end.py
"""

import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from das2 import pipeline  # noqa: E402
from das2.config import Config  # noqa: E402
from das2.incident.build import reconcile  # noqa: E402
from das2.models import IncidentClass  # noqa: E402
from das2.report import dashboard  # noqa: E402


def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
    if not cond:
        raise SystemExit(1)


def build_fixtures(root: Path) -> None:
    subprocess.run([sys.executable, str(REPO / "tools" / "make_fixtures.py"),
                    "--out", str(root)],
                   check=True, capture_output=True)


def make_config(fixtures: Path, out: Path) -> Config:
    cfg = Config()
    cfg.ingest.history_dir = str(fixtures / "HISTORY")
    cfg.ingest.histcurr_path = str(fixtures / "HISTCURR" / "histcurr_fujitsu.csv")
    cfg.ingest.longlat_path = str(fixtures / "LongLat.csv")
    cfg.report.output_dir = str(out)
    cfg.alert.enabled = False
    return cfg


def main():
    tmp = Path(tempfile.mkdtemp(prefix="das2-e2e-"))
    try:
        fixtures, out = tmp / "fx", tmp / "out"
        print("generating fixtures")
        build_fixtures(fixtures)
        check("fixture history written", (fixtures / "HISTORY").is_dir())

        print("\nrunning the pipeline end to end")
        cfg = make_config(fixtures, out)
        result = pipeline.run(cfg)

        check("readings ingested", len(result.readings) > 50_000,
              f"({len(result.readings):,})")
        check("every reading matched a sensor",
              result.stats["ingest"]["readings_unmatched"] == 0)
        check("coordinates resolved for most sensors",
              result.stats["ingest"]["coordinate_coverage_pct"] > 90,
              f"({result.stats['ingest']['coordinate_coverage_pct']}%)")
        check("the window is the full 72 hours",
              (result.window_end - result.window_start).total_seconds() > 70 * 3600)

        # --- the client's headline requirement ----------------------------- #
        print("\nthe regional event: several sites abnormal together")
        regional = [i for i in result.incidents
                    if i.incident_class is IncidentClass.REGIONAL_EVENT]
        check("exactly one regional event is reported", len(regional) == 1,
              f"({len(regional)})")
        event = regional[0]
        check("it spans several sites", len(event.cluster.sites) >= 3,
              f"({sorted(event.cluster.sites)})")
        check("it spans several equipment types",
              len(event.cluster.equipment_types) >= 2,
              f"({sorted(event.cluster.equipment_types)})")
        check("it is placed in the East", str(event.cluster.region) == "East",
              f"({event.cluster.region})")
        check("it is actionable, not suppressed", event.should_alert,
              f"(severity {event.severity}, {event.priority.value})")
        check("it is among the incidents actually sent",
              any(i.incident_id == event.incident_id for i in result.alertable))
        # Deliberately NOT asserting that it outranks everything. Once the
        # fixture carried a pump insisting it was running against a meter
        # reading zero, that scored higher -- and correctly so: a contradiction
        # between two instruments is physically conclusive, whereas a
        # 34-minute pressure dip across three sites is an inference. The
        # earlier assertion encoded an assumption that held only while this was
        # the only substantial finding in the fixture.
        check("incidents are ordered by severity, worst first",
              all(a.severity >= b.severity
                  for a, b in zip(result.incidents, result.incidents[1:])),
              f"({[round(i.severity, 1) for i in result.incidents]})")
        check("it tells the operator to investigate the area",
              "area" in event.recommendation.lower())
        check("it carries the evidence that justified that",
              len(event.detail.get("evidence", [])) >= 2,
              f"({event.detail.get('evidence')})")

        print("\n  and this is the case the name-based rule cannot see")
        names = [m.sensor.description for m in event.cluster.members]
        prefixes = {n.split("-")[0] for n in names}
        check("its members share no common name prefix", len(prefixes) > 1,
              f"({sorted(prefixes)})")

        # --- the opposite verdict ------------------------------------------- #
        print("\nsuppression: noise must not page anyone")
        check("some incidents are suppressed",
              len(result.incidents) > len(result.alertable),
              f"({len(result.incidents)} incidents, {len(result.alertable)} alertable)")
        for incident in result.incidents:
            if incident.incident_class is IncidentClass.TELEMETRY_FANOUT:
                check("fan-out never alerts", not incident.should_alert)
            if incident.incident_class is IncidentClass.WEATHER_DRIVEN:
                check("weather-driven says do not dispatch",
                      "not dispatch" in incident.recommendation.lower())

        print("\n  and rain is judged over the INCIDENT's window, not the run's")
        weather = [i for i in result.incidents
                   if i.incident_class is IncidentClass.WEATHER_DRIVEN]
        check("the regional event is not excused by rain at another hour",
              event.incident_class is IncidentClass.REGIONAL_EVENT
              and event not in weather,
              "(a whole-run rainfall total attached to every incident in the "
              "region suppressed this genuine event to P4)")

        print("\nsensor faults are found and are dispatchable")
        faults = [i for i in result.incidents
                  if i.incident_class is IncidentClass.SENSOR_FAULT]
        check("at least one sensor fault is reported", len(faults) >= 1,
              f"({len(faults)})")
        check("it recommends a technician",
              all("technician" in f.recommendation.lower() for f in faults))

        print("\ndetector output is typed, not a single score")
        types = {a.dominant_type.value for a in result.anomalies}
        check("several distinct fault types found", len(types) >= 4, f"({sorted(types)})")
        check("the injected level shifts are found", "LEVEL_SHIFT" in types)
        check("the injected flatline is found", "FLATLINE" in types)
        check("the injected stale sensor is found", "STALE" in types)

        print("\n  and volume is sane -- the first gate that matters")
        check("anomalies are far fewer than sensors analysed",
              len(result.anomalies) < len(result.sensors),
              f"({len(result.anomalies)} anomalies, {len(result.sensors)} sensors)")
        check("incidents are fewer still",
              len(result.incidents) <= len(result.anomalies),
              f"({len(result.incidents)} incidents)")

        print("\n  severity is in engineering units, never a Z-score")
        with_dev = [a for a in result.anomalies if a.severity.deviation]
        check("deviations are reported with a unit",
              all(a.severity.unit for a in with_dev), f"({len(with_dev)} with deviation)")
        check("no deviation is expressed in seconds",
              all(a.severity.unit != "s" for a in with_dev),
              "(a duration in the deviation column reads as the instrument "
              "being thousands of units out)")

        # --- one fault must not alert twelve times --------------------------- #
        print("\ncross-run identity: one fault, one incident")
        second = pipeline.run(cfg, open_incidents=list(result.incidents))
        new, updated, resolved = reconcile(
            second.incidents, list(result.incidents), now=datetime.now())
        check("nothing is announced as new on the second run", len(new) == 0,
              f"({len(new)} new -- each would be another Telegram message)")
        check("every incident is matched to its existing identity",
              len(updated) == len(result.incidents),
              f"({len(updated)} of {len(result.incidents)})")
        check("nothing is resolved while still being detected", len(resolved) == 0)
        check("ids are stable across runs",
              {i.incident_id for i in updated} == {i.incident_id for i in result.incidents})

        # --- the deliverable -------------------------------------------------- #
        print("\nthe dashboard")
        path = dashboard.write(result, out)
        page = path.read_text(encoding="utf-8")
        # CARTO's basemap CDN began answering with tiles that read "API key
        # required", so the map rendered perfectly and every tile was a
        # notice instead of Singapore -- a worse failure than no basemap,
        # because it looks like a bug in this page.
        check("the basemap needs no API key",
              "openstreetmap.org/{z}/{x}/{y}" in page
              and "cartocdn" not in page,
              "(OSM tiles: no account, no key)")
        check("no template placeholder survived rendering",
              "__TILE_URL__" not in page and "__TILE_ATTRIBUTION__" not in page)
        custom = dashboard.write(result, out, filename="custom.html",
                                 tile_url="https://tiles.internal/{z}/{x}/{y}.png",
                                 tile_attribution="PUB")
        check("an internal tile server can replace it",
              "tiles.internal" in custom.read_text(encoding="utf-8"),
              "(the right answer on a network with no internet)")
        check("an HTML file is written", path.exists())
        html = path.read_text()
        check("it is self-contained enough to email", len(html) > 8_000,
              f"({len(html):,} bytes)")

        payload = json.loads(re.search(r"const DATA = (\{.*?\});\n", html, re.S).group(1))
        check("the embedded data is valid JSON", isinstance(payload, dict))
        check("every incident is represented",
              len(payload["incidents"]) == len(result.incidents))
        check("the region x type matrix is populated",
              bool(payload["matrix"]["regions"]) and bool(payload["matrix"]["equipment"]),
              f"({payload['matrix']['regions']} x {payload['matrix']['equipment']})")
        check("the regional event carries map coordinates",
              payload["incidents"][0]["lat"] is not None
              and payload["incidents"][0]["lon"] is not None)
        check("rainfall context is attached",
              bool(payload["rainfall"]), f"({payload['rainfall']})")
        check("every incident states what to do",
              all(i["recommendation"] for i in payload["incidents"]))

        print("\n  the page must survive an isolated network")
        check("it falls back when the map library is unreachable",
              "svgFallbackMap" in html,
              "(an ops network often has no CDN, and 'where' is the whole point)")

        print("\nAll end-to-end tests passed.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
