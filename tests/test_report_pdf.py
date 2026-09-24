#!/usr/bin/env python3
"""
The run report, and the map that stopped being readable.

Two defects the client found by using the system, both of which pass every
unit test that checks "did a file get written":

  * **The map.** 198 incidents were drawn as 198 markers -- but coordinates
    come from an `RTUNumber -> LKey` join, so every sensor on one RTU carries
    identical lat/lon. The 198 markers stood in about 40 piles, each hiding the
    ones under it, and the site labels printed on top of one another:
    "Bedok PS, Bedok Pond 2" over "Bedok PS, Bedok Pond 3", neither readable.
    Their words: *"very small and thus not very helpful ... not even all the
    text can be seen"*.
  * **The volume.** A run header, two photos, ten incident messages and a
    digest -- fourteen notifications an hour, for ever. *"the alert messages
    are way too many, can we just give one summary report pdf each time?"*

So the assertions below are about legibility and about count, not about
whether the code ran. A test that only checks the PNG exists would have passed
on every version the client rejected.

Run:  python3 tests/test_report_pdf.py
"""

import sys
import tempfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from das2.incident import parameters, signature  # noqa: E402
from das2.models import AnomalyType, PhysicalSeverity  # noqa: E402
from das2.report import charts, pdf, theme  # noqa: E402
from das2.spatial.regions import Region     # noqa: E402

passed = failed = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {label}" + (f"  ({detail})" if detail else ""))
    else:
        failed += 1
        print(f"  FAIL  {label}" + (f"  ({detail})" if detail else ""))


# --------------------------------------------------------------------------- #
# A run shaped like the one the client sent back
# --------------------------------------------------------------------------- #
SITES = [
    ("Bedok PS", 1.34286, 103.91977, Region.EAST),
    ("Bedok Pond 2", 1.31697, 103.93443, Region.EAST),
    ("Bedok Pond 3", 1.32100, 103.93900, Region.EAST),
    ("Tampines Pond B", 1.34960, 103.95680, Region.EAST),
    ("Sg Mandai Intake", 1.41800, 103.78000, Region.NORTH),
    ("Kranji 1 PS", 1.41567, 103.72871, Region.NORTH),
    ("Lower Seletar PS", 1.40500, 103.86500, Region.NORTH_EAST),
    ("Pandan 2 PS", 1.31214, 103.74723, Region.WEST),
    ("Murai PS", 1.39000, 103.68500, Region.WEST),
    ("Marina Barrage", 1.28050, 103.87100, Region.CENTRAL),
    ("Arthur Road TG", 1.30100, 103.89000, Region.CENTRAL),
    ("MacRitchie PS", 1.34235, 103.83571, Region.CENTRAL),
]


#: Parameter, unit, typical move. Shaped like a drainage incident: canal
#: levels and flows rising together, pressure falling.
PARAMS = [("CanalLevel", "m", 0.82), ("Flowrate", "L/s", 24.0),
          ("Pressure", "bar", -0.4), ("Pump", "", 0.0)]


def member(site: str, n: int, region: str = "East") -> NS:
    """One anomalous sensor, with the fields the report and composer read."""
    name, unit, move = PARAMS[n % len(PARAMS)]
    return NS(sensor=NS(description=f"{site}-{name}-{n}",
                        sensor_key=f"K{n:05d}", equipment=name,
                        site=site, unit=unit, region=region),
              dominant_type=AnomalyType.LEVEL_SHIFT,
              score=40.0,
              severity=PhysicalSeverity(deviation=abs(move),
                                        signed_deviation=move,
                                        unit=unit, duration_s=10800))


def incident(idx: int, site_idx: int, severity: float, cls: str) -> NS:
    name, lat, lon, region = SITES[site_idx % len(SITES)]
    priority = ("P1" if severity >= 80 else "P2" if severity >= 60
                else "P3" if severity >= 35 else "P4")
    built = NS(
        incident_id=f"INC{idx:04d}",
        cluster=NS(centroid_lat=lat, centroid_lon=lon, region=region,
                   members=[member(name, k, str(region.value))
                            for k in range(1 + idx % 12)],
                   sites={name},
                   radius_m=800.0,
                   start=datetime(2026, 9, 23, 2), end=datetime(2026, 9, 23, 5)),
        incident_class=NS(value=cls),
        priority=NS(value=priority), severity=severity,
        neighbour_correlation=0.42, rainfall_mm=3.1, detail={},
        recommendation="Multiple sites affected together - investigate the "
                       "area, not one sensor.",
    )
    # The reading the pipeline attaches, through the same code path the
    # pipeline uses. Built here rather than hand-written so a rule edited in
    # `event_signatures.yaml` cannot drift away from what this pins.
    found = signature.match(built, rain_context="dry")
    if found is not None:
        built.detail[signature.DETAIL_KEY] = signature.as_detail(found)
    return built


def busy_run(n: int = 198) -> NS:
    classes = ["REGIONAL_EVENT", "SENSOR_FAULT", "TELEMETRY_FANOUT",
               "WATCH", "DRIFT_MAINTENANCE"]
    incidents = []
    for k in range(n):
        # 5 P1, 13 P2, 31 P3, the rest P4 -- the shape of the client's
        # own run, so the assertions below are about their volume.
        severity = (88.0 - k * 2.0 if k < 5 else
                    78.0 - (k - 5) * 1.3 if k < 18 else
                    58.0 - (k - 18) * 0.7 if k < 49 else
                    30.0 - (k % 25))
        incidents.append(incident(k, k, max(severity, 4.0), classes[k % 5]))
    alertable = sorted([i for i in incidents if i.severity >= 35],
                       key=lambda i: -i.severity)[:24]
    return NS(
        run_id="20260923-145443",
        window_start=datetime(2026, 9, 20, 15),
        window_end=datetime(2026, 9, 23, 14),
        duration_s=375.3,
        incidents=incidents, alertable=alertable,
        anomalies=[m for i in incidents for m in i.cluster.members],
        clusters=[i.cluster for i in incidents],
        held=[(i, "telemetry fan-out, not a site visit")
              for i in incidents if i not in alertable][:90],
        rainfall_by_region={"North": 121.8, "East": 0.0, "West": 0.2},
        region_matrix={"East": {"Level": 31, "Flowrate": 12},
                       "North": {"Level": 22, "Temperature": 14},
                       "West": {"Voltage": 17}},
        stats={
            "lifecycle": {"new": 33, "updated": 251, "resolved": 87},
            "selection": {"selected": 10, "held": 274, "held_reasons": {
                "telemetry fan-out, not a site visit": 116,
                "run cap of 10 reached; on the dashboard": 86,
                "evidence too weak or conflicting": 44}},
            "detection": {"anomalies": 3368, "sensors": 1771, "by_type": {
                "STALE": 1791, "SPIKE": 690, "FLATLINE": 443}},
            "ingest": {"history_files": 48, "history_files_skipped": 0,
                       "history_rows": 5496090, "history_rows_dropped": 57553,
                       "inventory_rows": 13982,
                       "coordinate_coverage_pct": 85.7,
                       "sensors_without_coords": 2001, "missing_hours": 24,
                       "missing_hours_range": "2026-09-20 23:00 .. 22:00"},
            "coverage": {"analog_coverage_pct": 68.7, "unclassified": 1642},
            "baselines": {"sensors": 0, "usable": 0},
        },
    )


WIDE = ["CanalLevel", "Level", "Flowrate", "Pressure", "Pump", "Valve",
        "Vibration", "Current", "Voltage", "Temperature", "Rainfall",
        "Conductivity"]


def crowded_run() -> NS:
    """Two incidents, each carrying every parameter class we classify."""
    run = busy_run(n=6)
    for idx, inc in enumerate(run.incidents[:2]):
        inc.severity = 90.0 - idx
        inc.priority = NS(value="P1")
        inc.cluster.members = [
            NS(sensor=NS(description=f"Site{idx}-{name}", sensor_key=f"W{k}",
                         equipment=name, site=f"Site {idx}", unit="m",
                         region="East"),
               dominant_type=AnomalyType.LEVEL_SHIFT, score=40.0,
               severity=PhysicalSeverity(deviation=0.5, signed_deviation=-0.5,
                                         unit="m", duration_s=10800))
            for k, name in enumerate(WIDE)]
        found = signature.match(inc, rain_context="dry")
        if found is not None:
            inc.detail[signature.DETAIL_KEY] = signature.as_detail(found)
    run.alertable = run.incidents[:2]
    return run


def _check_crowded_page() -> None:
    try:
        import pymupdf
    except ImportError:
        print("    (pymupdf absent -- page-bounds assertions skipped)")
        return

    run = crowded_run()
    with tempfile.TemporaryDirectory() as tmp:
        out = pdf.write(run, Path(tmp) / "crowded.pdf", stamp="test")
        doc = pymupdf.open(out)
        page = next((p for p in doc if "made of" in p.get_text()), None)
        if page is None:
            check("the crowded incidents reach an anatomy page", False)
            doc.close()
            return

        text = page.get_text()
        height = page.rect.height
        footer = ("Direction comes from the signed deviation of each sensor's "
                  "dominant finding. A parameter whose sensors disagree is "
                  "reported as mixed rather than resolved by majority. A "
                  "reading describes the incident; it never changed how it "
                  "was classified or what was recommended.")
        stray = [line for line, rect in _lines(page)
                 if rect.y1 > height * 0.955 and line.strip() not in footer]
        check("the only thing in the bottom margin is the footnote",
              not stray, "; ".join(stray[:2]) or "clean")
        check("both incidents still get their heading",
              text.count("P1  ") >= 2, f"{text.count('P1  ')} heading(s)")
        check("a table too tall for its share says what it dropped",
              "more parameter(s)" in text,
              "twelve parameters in half a page cannot all be shown; "
              "cutting them off without saying so is the silent loss")
        check("and the reading survives the squeeze",
              "Would change this reading" in text)
        check("the rule's own ACTION is not printed beside the real one",
              "Log it." not in text,
              "'log it, this is the trip not worth making' next to a computed "
              "'investigate the area' lets an unchecked rule countermand a "
              "decision in the reader's head")
        doc.close()


def _lines(page):
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line["spans"])
            yield text, __import__("pymupdf").Rect(line["bbox"])


def main() -> int:
    run = busy_run()

    print("\nthe map draws places, not repeated coordinates")
    groups = charts._site_groups(run.incidents)
    check("198 incidents collapse to one marker per place",
          len(groups) == len(SITES),
          f"{len(run.incidents)} incidents -> {len(groups)} markers")
    check("no two markers share a position",
          len({(round(g['lat'], 3), round(g['lon'], 3)) for g in groups})
          == len(groups))
    check("every incident is accounted for in some marker",
          sum(len(g["incidents"]) for g in groups) == len(run.incidents),
          "aggregation must not lose any")
    check("a marker carries the WORST priority at its place",
          all(g["priority"] == max(g["incidents"],
                                   key=lambda i: i.severity).priority.value
              for g in groups),
          "a P1 hidden under a P4 dot is the defect being fixed")
    check("markers are ordered most severe first",
          [g["severity"] for g in groups]
          == sorted((g["severity"] for g in groups), reverse=True),
          "so the labelled ones are the ones worth naming")

    print("\nlabels never overlap, and never leave the map")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.4, 6.2))
    ax.set_xlim(charts.ISLAND_VIEW[0], charts.ISLAND_VIEW[1])
    ax.set_ylim(charts.ISLAND_VIEW[2], charts.ISLAND_VIEW[3])
    boxes = charts._place_labels(ax, groups, limit=12, fontsize=7.5,
                                 bounds=charts.ISLAND_VIEW)
    plt.close(fig)

    overlaps = [(a, b) for n, a in enumerate(boxes) for b in boxes[n + 1:]
                if not (a[2] < b[0] or a[0] > b[2]
                        or a[3] < b[1] or a[1] > b[3])]
    check("no two labels overlap", not overlaps,
          f"{len(boxes)} label(s) placed, {len(overlaps)} collision(s)")
    lon0, lon1, lat0, lat1 = charts.ISLAND_VIEW
    check("every label is inside the map",
          all(lon0 <= b[0] and b[2] <= lon1 and lat0 <= b[1] and b[3] <= lat1
              for b in boxes),
          "one drawn off the edge is a label nobody can read")
    check("some labels were actually placed", len(boxes) >= 4,
          f"{len(boxes)} of 12 requested")

    print("\nthe map renders bigger when it is the page")
    with tempfile.TemporaryDirectory() as tmp:
        small = charts.region_map_png(run.incidents, Path(tmp) / "s.png")
        big = charts.region_map_png(run.incidents, Path(tmp) / "b.png",
                                    figsize=(10.4, 6.2), dpi=170,
                                    view=charts.ISLAND_VIEW)
        check("a full-page map carries more pixels than the thumbnail",
              big.stat().st_size > small.stat().st_size * 1.3,
              f"{small.stat().st_size // 1024} KB -> "
              f"{big.stat().st_size // 1024} KB")

    print("\nthe report is one document, and it is complete")
    with tempfile.TemporaryDirectory() as tmp:
        out = pdf.write(run, Path(tmp) / "report.pdf", stamp="das2 source test")
        check("a PDF is written", out.exists(),
              f"{out.stat().st_size // 1024} KB")
        check("and it is under Telegram's 50 MB document limit",
              out.stat().st_size < 50 * 1024 * 1024)

        raw = out.read_bytes()
        check("it is a real PDF", raw.startswith(b"%PDF-"))
        pages = raw.count(b"/Type /Page\n") or raw.count(b"/Type /Page")
        check("it has a page per section", pages >= 6, f"{pages} page(s)")

        try:
            import pymupdf                                # noqa: F401
            doc = pymupdf.open(out)
            text = "\n".join(page.get_text() for page in doc)
            doc.close()
        except ImportError:
            text = ""
            print("    (pymupdf absent -- text assertions skipped)")

        if text:
            for phrase in ("this round of analysis", "What this run found",
                           "Act first", "What each one is made of",
                           "By region and by parameter", "Where",
                           "What to act on", "held back",
                           "Data quality and coverage"):
                check(f"the report says {phrase!r}", phrase in text)
            check("the headline count is the one an operator acts on",
                  str(sum(1 for i in run.incidents
                          if i.priority.value in ("P1", "P2"))) in text)
            check("regions are named, not repr'd",
                  "North-East" in text and "Region.NORTH_EAST" not in text,
                  "str(enum) gives Region.NORTH_EAST, which is not a place")
            check("what was held back is reported with its reasons",
                  "telemetry fan-out" in text,
                  "a system that shows only what it chose to tell you "
                  "is indistinguishable from one that missed the rest")
            check("the missing-hours caveat travels with the numbers",
                  "Missing hours" in text,
                  "24 missing hours inflate STALE; a reader must know")
            check("the explanation names which parameters are moving",
                  "parameters carrying this run" in text,
                  "'18 incidents' without 'on canal level' does not say what "
                  "kind of problem it is")
            check("the breakdown reaches the incident pages",
                  "Canal Level" in text and "typical move" in text.lower())
            check("the reading is offered, hedged",
                  "Looks like:" in text,
                  "an unvalidated rule asserted flatly is how the system "
                  "loses the argument the first time it is wrong")
            check("and never without what would falsify it",
                  "Would change this reading" in text,
                  "a hedged sentence with nothing to check it against is a "
                  "horoscope; this line is how PUB corrects the rule")
            check("the reading does not displace the decision",
                  text.index("Investigate the area")
                  < text.index("Looks like:"),
                  "the class and the action were computed without it and "
                  "must be read first")
            check("the explanation is continued, never truncated",
                  "never alerted on" in text,
                  "the caveats sit last, so an overflowing page drops the "
                  "limits and keeps the counts they invalidate")

    print("\nnothing is drawn off the bottom of the paper")
    # Matplotlib clips at the figure edge in silence, so a page that grew by a
    # sentence loses its last line with no error anywhere. Two twelve-parameter
    # incidents to a page is the case that overruns: without a height budget
    # the first block takes what it likes and the second is drawn into the
    # margin, which is a silent loss of exactly what this page exists to show.
    _check_crowded_page()

    print("\nthe explanation says what happened, in sentences")
    from das2.report import narrative

    paras = narrative.paragraphs(run)
    check("the run is explained in prose", len(paras) >= 3,
          f"{len(paras)} paragraph(s)")
    check("the first sentence leads with the decision",
          "need a decision now" in paras[0] or "Nothing needs" in paras[0],
          paras[0][:70])
    joined = " ".join(paras)
    check("it names the mechanism, not just the class",
          "neighbours" in joined or "neighbour" in joined,
          "'REGIONAL_EVENT' is a label; 'the neighbours moved too' is a reason")
    check("the caveats travel with the findings",
          "missing" in joined and "STALE" in joined,
          "a 24-hour feed gap inflates STALE across the fleet")
    check("what was held back is explained, not just counted",
          "held back" in joined)
    check("no sentence asserts a severity word the data cannot carry",
          not any(w in joined.lower()
                  for w in ("critical", "alarming", "urgent!", "severe")),
          "it is P1, or it is 12 sensors across 4 sites")

    quiet = busy_run(0)
    quiet.stats["selection"] = {"held": 0, "held_reasons": {}}
    quiet.stats["lifecycle"] = {"new": 0, "updated": 0, "resolved": 0}
    quiet_text = " ".join(narrative.paragraphs(quiet))
    check("a quiet run is explained too, not left blank",
          "Nothing was detected" in quiet_text,
          "silence has to read as a result, not as a crashed job")

    print("\ntwo artefacts per run: the map, then the analysis")
    sent: list[tuple[str, str]] = []

    class FakeClient:
        def __init__(self, config):
            self.config = config

        def send_document(self, path, caption="", **kw):
            sent.append(("document", caption))
            return {"ok": True}

        def send_photo(self, path, caption="", **kw):
            sent.append(("photo", caption))
            return {"ok": True}

        def send_message(self, text, **kw):
            sent.append(("message", text))
            return {"ok": True}

    from das2.alerting import telegram as tg

    real = tg.TelegramClient
    tg.TelegramClient = FakeClient
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = pdf.write(run, Path(tmp) / "r.pdf")
            config = tg.TelegramConfig(token="t", chat_id="c", enabled=True)

            map_png = Path(tmp) / "map.png"
            charts.region_map_png(run.incidents, map_png,
                                  view=charts.ISLAND_VIEW)

            sent.clear()
            report = tg.send_report(run, config, report_pdf=out,
                                    map_png=map_png)
            check("exactly two notifications are sent", len(sent) == 2,
                  f"{len(sent)}: {[k for k, _ in sent]}")
            check("the map comes first", sent[0][0] == "photo",
                  "it renders inline on a phone; the PDF must be opened")
            check("the analysis comes second", sent[1][0] == "document")
            check("the numbers are said once, not twice",
                  sent[0][1] != sent[1][1],
                  "two identical captions read as a duplicate send")
            check("every alertable incident counts as delivered",
                  len(report.sent) == len(run.alertable),
                  "the document carried them all; reporting fewer would "
                  "make the next run re-announce them")

            caption = sent[0][1]
            check("the caption says how many need a decision",
                  "need a decision" in caption)
            check("the caption says how many were held back",
                  "held back" in caption)
            check("the caption offers the worst incident's reading, hedged",
                  "Looks like:" in caption and "unvalidated" in caption,
                  "read on a lock screen, so it carries the headline and "
                  "the hedge and nothing else")
            check("but the decision count is still read first",
                  caption.index("need a decision") < caption.index("Looks like"))
            check("the caption fits Telegram's limit",
                  len(caption) <= tg.CAPTION_MAX,
                  f"{len(caption)} of {tg.CAPTION_MAX}")

            sent.clear()
            tg.send_report(run, config, report_pdf=out, map_png=map_png,
                           p1_detail_messages=True)
            p1 = sum(1 for i in run.alertable if i.priority.value == "P1")
            check("P1 detail messages are opt-in and bounded",
                  len(sent) == 2 + p1,
                  f"map + report + {p1} P1 message(s), not "
                  f"{len(run.alertable)}")

            # A failed photo must not cost the analysis: the map is also a
            # page of the document, so losing the picture loses the glance,
            # not the delivery.
            sent.clear()
            missing = Path(tmp) / "not-here.png"
            report = tg.send_report(run, config, report_pdf=out,
                                    map_png=missing)
            check("a missing map still delivers the report",
                  len(sent) == 1 and sent[0][0] == "document",
                  f"{[k for k, _ in sent]}")
            check("and the report still counts as delivered",
                  len(report.sent) == len(run.alertable))

            sent.clear()
            tg.send_run(run, config, charts={}, max_incidents=10)
            check("the old per-incident mode still works",
                  len(sent) > 5, f"{len(sent)} message(s)")
            check("and it is the mode the client asked to be rid of",
                  len(sent) > 10,
                  "which is exactly why 'report' is now the default")
    finally:
        tg.TelegramClient = real

    print("\nthe palette keeps its promises")
    check("priority colours are distinct", len(set(
        theme.PRIORITY_COLOR[p] for p in theme.PRIORITY_ORDER)) == 4)
    check("every priority carries a meaning in words",
          all(theme.PRIORITY_MEANING.get(p) for p in theme.PRIORITY_ORDER),
          "colour is never the only channel")
    check("the sequential ramp is ordered light to dark",
          all(sum(int(theme.SEQUENTIAL[i][j:j + 2], 16) for j in (1, 3, 5))
              > sum(int(theme.SEQUENTIAL[i + 1][j:j + 2], 16) for j in (1, 3, 5))
              for i in range(len(theme.SEQUENTIAL) - 1)),
          "a ramp that is not monotonic in lightness is a rainbow")
    check("the map's shading stops short of the ramp's dark end",
          all(sum(int(c[j:j + 2], 16) for j in (1, 3, 5)) > 420
              for c in theme.CHOROPLETH),
          "it sits under saturated markers and must stay context")

    print(f"\n{passed} passed, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
