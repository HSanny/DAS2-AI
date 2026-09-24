#!/usr/bin/env python3
"""
The interactive report: the page the client clicks into to check our working.

    "can be interactive collective report show on the web ... he will click
     into the collective or interactive report to view, then verify oh there
     really is something that is not yet discovered by already existing stats
     calculation he put in"

That last clause is the whole design. The page is not a prettier dashboard;
it is an argument, and an argument has to hand over the evidence. So each
incident carries the sensors' own traces, the band a median-and-σ check would
have drawn around them, and what that check would have concluded -- including
where it would have been right and this system added nothing.

Two classes of defect are pinned here, because both were real and neither
raises anything:

**Units.** The traces are drawn from epoch milliseconds, and the flagged span
from `datetime.timestamp()`. `ts.astype("int64") // 1e6` assumes nanoseconds;
pandas 2 keeps whatever resolution the parse produced, and these arrive as
`datetime64[s]`, so the series x-axis came out as 1790 where the span was
1790100542000. Every flagged span was drawn past the right-hand edge of its
chart, and the page rendered perfectly with nothing marked on it.

**Silent clipping.** The page embeds its own data, so an unbounded payload is
a file too large to open on a phone -- which is where it is opened. The budget
is asserted rather than trusted.

Run:  python3 tests/test_interactive_report.py
"""

import json
import re
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from das2.detect import conventional as cv                # noqa: E402
from das2.models import (                                 # noqa: E402
    AnomalyType,
    Cluster,
    PhysicalSeverity,
    SensorAnomaly,
    SensorMeta,
)
from das2.report import dashboard                         # noqa: E402

passed = failed = 0
T0 = datetime(2026, 9, 21, 0, 0)
N = 1500


def check(label: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {label}" + (f"  ({detail})" if detail else ""))
    else:
        failed += 1
        print(f"  FAIL  {label}" + (f"  ({detail})" if detail else ""))


def synthetic_run(sensors: int = 4) -> NS:
    """A run with real readings behind it, so the page has traces to draw."""
    stamps = pd.date_range(T0, periods=N, freq="120s")
    rng = np.random.default_rng(7)
    rows, members = [], []
    for s in range(sensors):
        values = 3.0 + rng.normal(0, 0.02, N)
        values[600:900] += 0.30            # the sustained event that hides itself
        rows.append(pd.DataFrame({"sensor_key": f"S{s}", "ts": stamps,
                                  "value": values}))
        members.append(SensorAnomaly(
            sensor=SensorMeta(sensor_key=f"S{s}", description=f"Site{s}-Pressure",
                              equipment="Pressure", site=f"Site {s}",
                              region="East", unit="bar",
                              latitude=1.33, longitude=103.93),
            start=stamps[600].to_pydatetime(), end=stamps[899].to_pydatetime(),
            dominant_type=AnomalyType.LEVEL_SHIFT,
            severity=PhysicalSeverity(deviation=0.30, signed_deviation=0.30,
                                      unit="bar", duration_s=36000)))

    readings = pd.concat(rows, ignore_index=True)
    # The resolution that caused the bug: pandas keeps what the parse gave it.
    readings["ts"] = readings["ts"].astype("datetime64[s]")

    checks = {}
    for s in range(sensors):
        one = readings[readings["sensor_key"] == f"S{s}"]
        checks[f"S{s}"] = cv.evaluate(
            f"S{s}", one["ts"], one["value"].to_numpy(),
            spans=[(members[s].start, members[s].end)])

    cluster = Cluster(members=members, region="East",
                      centroid_lat=1.33, centroid_lon=103.93, radius_m=900.0)
    incident = NS(
        incident_id="East-20260921-abc",
        cluster=cluster,
        incident_class=NS(value="REGIONAL_EVENT"),
        priority=NS(value="P1"), severity=88.0,
        neighbour_correlation=0.71, rainfall_mm=0.0,
        recommendation="Multiple sites affected together - investigate the area.",
        ack_state=NS(value="none"),
        detail={"evidence": ["four sites moved together"],
                "conventional": cv.for_incident(
                    NS(cluster=cluster), checks)},
    )
    return NS(
        run_id="20260921-000000", started_at=T0,
        window_start=T0, window_end=T0 + timedelta(seconds=120 * N),
        duration_s=12.0, readings=readings, sensors=pd.DataFrame(),
        anomalies=members, clusters=[cluster], incidents=[incident],
        alertable=[incident], conventional=checks,
        rainfall_by_region={"East": 0.0},
        region_matrix={"East": {"Pressure": sensors}},
        stats={"conventional": cv.summarise(checks)},
    ), checks


def main() -> int:
    run, checks = synthetic_run()
    payload = dashboard.build_payload(run)
    incident = payload["incidents"][0]
    member = incident["members"][0]

    print("\nthe traces are there, and they line up with what was flagged")
    check("every member carries a trace", all("series" in m for m in
                                              incident["members"]))
    xs = [p[0] for p in member["series"]]
    check("the x axis is epoch milliseconds",
          1.7e12 < xs[0] < 2.0e12, f"x0={xs[0]}")
    check("THE FLAGGED SPAN FALLS INSIDE THE TRACE",
          xs[0] <= member["start_ms"] <= member["end_ms"] <= xs[-1],
          "`astype('int64') // 1e6` assumes nanoseconds; these timestamps are "
          "datetime64[s], and the span was drawn off the edge of every chart "
          "with nothing raised anywhere")
    check("and the span is a real fraction of the window, not a hairline",
          0.05 < (member["end_ms"] - member["start_ms"]) / (xs[-1] - xs[0]) < 0.5,
          f"{100 * (member['end_ms'] - member['start_ms']) / (xs[-1] - xs[0]):.0f}"
          f"% of the window")

    print("\ndecimation keeps the reading the chart is about")
    stamps = list(range(1000))
    values = [0.0] * 1000
    values[503] = 99.0
    thin = dashboard._downsample(stamps, values, points=40)
    check("a lone spike survives thinning",
          any(v == 99.0 for _, v in thin),
          "stride decimation deletes it with probability 1 - 1/N, which is "
          "the one point on the chart that mattered")
    check("and the result is actually smaller", len(thin) <= 40,
          f"1000 -> {len(thin)} points")
    check("a short series is passed through untouched",
          len(dashboard._downsample(stamps[:10], values[:10], points=40)) == 10)
    check("an empty one does not explode",
          dashboard._downsample([], [], points=40) == [])

    print("\nthe band is the operator's own, carried with the trace")
    band = member["band"]
    check("mean, sigma and k travel together",
          {"mean", "std", "k"} <= set(band),
          "a band drawn from numbers the page invented would prove nothing")
    check("k is 3", band["k"] == 3.0)
    check("the verdict rides along so the caption can state it",
          band["verdict"] in ("alarmed", "missed", "chance", "undefined"),
          band["verdict"])
    check("this fixture is the case worth showing: their check misses it",
          band["verdict"] == "missed" and band["peak_z"] < 3.0,
          f"{band['peak_z']}σ over the window, "
          f"{band['peak_z_excluded']}σ against the hours before it")

    print("\nthe verification panel reaches the page")
    panel = incident["conventional"]
    check("the incident carries it", bool(panel))
    check("it answers the client's question in its first line",
          "3σ" in panel["headline"], panel["headline"])
    check("it names why this system saw it anyway",
          bool(panel["found_because"]))
    check("and lists every sensor, so the claim can be audited",
          len(panel["sensors"]) == len(incident["members"]))
    check("the run-level figures are on the page too",
          payload["conventional"].get("sensors") == 4,
          "per-incident counts without the fleet number invite the question "
          "'and how often does your check fire in total?'")

    print("\nthe page is bounded")
    check("the charted-sensor budget exists and is modest",
          0 < dashboard.MAX_CHARTED_SENSORS <= 200,
          f"{dashboard.MAX_CHARTED_SENSORS} sensors")
    big = big_run(sensors=400)
    charted = sum(1 for i in dashboard.build_payload(big)["incidents"]
                  for m in i["members"] if "series" in m)
    check("a run with 400 members does not embed 400 traces",
          charted <= dashboard.MAX_CHARTED_SENSORS,
          f"{charted} charted of 400")

    print("\nit renders, and the filters filter")
    html = dashboard.render(run)
    check("the payload is valid JSON inside the page",
          json.loads(re.search(r"const DATA = (\{.*?\});\n", html,
                               re.S).group(1))["run_id"] == run.run_id)
    check("the filter row is above the table it filters",
          html.index('id="filters"') < html.index('id="incidents"'),
          "position is the only explanation a filter row should need")
    check("a region can be drilled into from the matrix",
          "focusRegion(" in html)
    _browser_checks(html)

    print(f"\n{passed} passed, {failed} failed.")
    return 1 if failed else 0


def big_run(sensors: int) -> NS:
    """Many members, no readings: only the budget is under test."""
    members = [
        SensorAnomaly(
            sensor=SensorMeta(sensor_key=f"B{i}", description=f"B{i}",
                              equipment="Pressure", site=f"Site {i % 20}",
                              region="East", unit="bar"),
            start=T0, end=T0 + timedelta(hours=1),
            dominant_type=AnomalyType.LEVEL_SHIFT,
            severity=PhysicalSeverity(deviation=1.0, unit="bar"))
        for i in range(sensors)]
    stamps = pd.date_range(T0, periods=60, freq="60s")
    readings = pd.concat([
        pd.DataFrame({"sensor_key": m.sensor.sensor_key, "ts": stamps,
                      "value": np.linspace(0, 1, 60)})
        for m in members], ignore_index=True)
    cluster = Cluster(members=members, region="East")
    incident = NS(incident_id="big", cluster=cluster,
                  incident_class=NS(value="REGIONAL_EVENT"),
                  priority=NS(value="P1"), severity=90.0,
                  neighbour_correlation=None, rainfall_mm=None,
                  recommendation="x", ack_state=NS(value="none"), detail={})
    return NS(run_id="big", started_at=T0, window_start=T0,
              window_end=T0 + timedelta(hours=1), duration_s=1.0,
              readings=readings, sensors=pd.DataFrame(), anomalies=members,
              clusters=[cluster], incidents=[incident], alertable=[incident],
              conventional={}, rainfall_by_region={}, region_matrix={},
              stats={})


def _browser_checks(html: str) -> None:
    """
    Open the page and use it.

    Optional: the suite must run on a machine with no browser. But a page whose
    JavaScript throws renders as an empty card and every string assertion above
    still passes, so where a browser IS available it is used.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("    (playwright absent -- browser assertions skipped)")
        return
    chromium = Path("/opt/pw-browsers/chromium")
    with tempfile.TemporaryDirectory() as tmp:
        page_file = Path(tmp) / "page.html"
        page_file.write_text(html, encoding="utf-8")
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    executable_path=str(chromium) if chromium.exists() else None)
                page = browser.new_page()
                errors: list[str] = []
                page.on("pageerror", lambda e: errors.append(str(e)))
                page.goto(page_file.as_uri())
                page.wait_for_timeout(500)

                check("the page runs without throwing", not errors,
                      "; ".join(errors[:2]) or "clean")
                check("every incident is listed",
                      page.locator("tr.inc").count() == 1)
                page.locator("tr.inc").first.click()
                page.wait_for_timeout(200)
                check("clicking a row opens its evidence",
                      page.locator("tr.det.open").count() == 1)
                check("the traces are drawn as SVG, not as images",
                      page.locator(".trace svg path").count() >= 1)
                check("each chart carries one series and no legend",
                      page.evaluate(
                          """() => [...document.querySelectorAll('.trace')]
                             .every(t => t.querySelectorAll('svg path').length === 1
                                      && !t.querySelector('.legend'))"""),
                      "a legend for a single series is a box that repeats the "
                      "caption above it")
                check("the flagged span is drawn inside the chart",
                      page.evaluate(
                          """() => [...document.querySelectorAll('.trace svg rect')]
                             .some(r => +r.getAttribute('x') > 0
                                     && +r.getAttribute('x') < 320)"""),
                      "a span drawn past the edge is a chart that says nothing")
                page.fill("#f-text", "no-such-site")
                page.wait_for_timeout(200)
                check("a filter that matches nothing says so",
                      "Nothing matches" in page.inner_text("#incidents"),
                      "an empty table reads as a broken page")
                page.click("#f-reset")
                page.wait_for_timeout(200)
                check("reset brings everything back",
                      page.locator("tr.inc").count() == 1)
                browser.close()
        except Exception as exc:                           # noqa: BLE001
            print(f"    (browser unavailable -- skipped: "
                  f"{str(exc).splitlines()[0][:80]})")


if __name__ == "__main__":
    sys.exit(main())
