"""
das2.report.dashboard
=====================

One self-contained HTML file per run: the regional map the client asked for.

    "a heatmap, or a regional map, that he is able to see cluster of abnormal
     sensors ... and observe the anomaly by region and by type ... before
     actually sending someone down to the site"

What the page has to answer, in order
-------------------------------------
1. *Is anything wrong right now?*        -> the KPI strip
2. *Where?*                              -> the map
3. *Is it the area or one instrument?*   -> cluster markers vs single markers
4. *Do we drive there?*                  -> the recommendation on every row

The design follows from that ordering. The map leads because "where" is the
question the client phrased the whole requirement around, and the
recommendation is repeated on every incident row rather than being hidden
behind a click, because the decision is the deliverable.

Self-contained on purpose
-------------------------
All data is embedded as JSON in the page. No server, no API, no build step --
the file can be emailed, opened from a network share, or archived as the record
of what the system saw at a point in time. Only the map tiles come from the
network, and the page degrades to the matrix and tables when they are
unreachable, so a machine without internet still gets everything except the
basemap.

Honest about precision
----------------------
Coordinates come from an RTU-level join, so every sensor at one site shares a
position and the map cannot resolve within a site. The page says so, in the
legend, rather than drawing precise-looking pins that imply a precision the
data does not have.
"""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from das2.models import Incident, IncidentClass, Priority

#: Priority colours. Ordered, single-hue-per-step ramp so the severity ordering
#: survives greyscale printing and the common forms of colour blindness -- an
#: alert page that only works for some readers is not an alert page.
PRIORITY_COLOR = {
    "P1": "#b2182b",
    "P2": "#ef8a62",
    "P3": "#f7d08a",
    "P4": "#9fb8c8",
}

#: Class badge colours, chosen so the two opposite verdicts are visually
#: opposite: red for "go and look", blue-grey for "do not drive there".
CLASS_COLOR = {
    IncidentClass.REGIONAL_EVENT.value: "#b2182b",
    IncidentClass.SENSOR_FAULT.value: "#d6604d",
    IncidentClass.PROCESS_EVENT.value: "#4393c3",
    IncidentClass.DRIFT_MAINTENANCE.value: "#92c5de",
    IncidentClass.WEATHER_DRIVEN.value: "#2166ac",
    IncidentClass.INSTRUMENT_CONFLICT.value: "#c0504d",
    # The machine, not the instrument. A distinct warm hue, because the
    # response is a different trade with different tools.
    IncidentClass.ASSET_FAILURE.value: "#a63603",
    IncidentClass.WATCH.value: "#999999",
    IncidentClass.TELEMETRY_FANOUT.value: "#777777",
}

REGION_ORDER = ["Central", "East", "North", "North-East", "West", "Unknown"]

#: Points kept per charted sensor. The trace exists so a reader can SEE that the
#: event sits inside the band their own check draws, which needs shape, not
#: every sample: 240 points is more than a 900px-wide chart can resolve.
#: Decimation keeps the min and the max of each bucket so a spike survives --
#: taking every Nth sample would delete the one reading the chart is about.
SERIES_POINTS = 240

#: Sensors charted per run. A 198-incident run carries ~1,800 members, and
#: embedding all of them produced a file too large to open on a phone. The
#: budget goes to the incidents that need a decision; the rest keep their
#: tables.
MAX_CHARTED_SENSORS = 120


def _downsample(stamps: Sequence[float], values: Sequence[float], *,
                points: int = SERIES_POINTS) -> list[list[float]]:
    """
    Thin a series to something a chart can draw, keeping the extremes.

    Min-and-max per bucket rather than every Nth sample. A spike is one
    reading; stride decimation deletes it with probability `1 - 1/N`, which
    would quietly remove the single most important point on a chart whose job
    is to show an operator what the instrument actually did.
    """
    n = len(values)
    if n == 0:
        return []
    if n <= points:
        return [[float(t), round(float(v), 4)] for t, v in zip(stamps, values)]

    buckets = max(1, points // 2)
    size = n / buckets
    out: list[list[float]] = []
    for b in range(buckets):
        lo, hi = int(b * size), max(int(b * size) + 1, int((b + 1) * size))
        chunk = values[lo:hi]
        if not len(chunk):
            continue
        times = stamps[lo:hi]
        i_min = int(np.argmin(chunk))
        i_max = int(np.argmax(chunk))
        for i in sorted({i_min, i_max}):
            out.append([float(times[i]), round(float(chunk[i]), 4)])
    return out


def _series_for(result, keys: set[str]) -> dict[str, list[list[float]]]:
    """Decimated traces for the sensors the page will chart."""
    readings = getattr(result, "readings", None)
    if readings is None or not len(readings) or not keys:
        return {}
    try:
        wanted = readings[readings["sensor_key"].isin(keys)]
    except (KeyError, TypeError):                          # pragma: no cover
        return {}

    out: dict[str, list[list[float]]] = {}
    for key, group in wanted.groupby("sensor_key", sort=False):
        group = group.sort_values("ts")
        # NOT `.astype("int64") // 1e6`. That assumes nanosecond resolution,
        # and pandas 2 keeps whatever resolution the parse produced -- these
        # timestamps arrive as datetime64[s], so the naive form returned 1790
        # where it meant 1790100542000, and every flagged span was drawn off
        # the right-hand edge of its chart. Subtracting the epoch and dividing
        # by a Timedelta is unit-agnostic by construction.
        stamps = ((group["ts"] - pd.Timestamp("1970-01-01"))
                  // pd.Timedelta("1ms")).to_numpy()
        out[str(key)] = _downsample(stamps, group["value"].to_numpy(dtype=float))
    return out


def _incident_json(incident: Incident, *,
                   series: dict[str, list[list[float]]] | None = None,
                   checks: dict[str, Any] | None = None) -> dict[str, Any]:
    c = incident.cluster
    series = series or {}
    checks = checks or {}
    return {
        "id": incident.incident_id,
        "cls": incident.incident_class.value,
        "priority": incident.priority.value,
        "severity": incident.severity,
        "region": str(c.region or "Unknown"),
        "lat": c.centroid_lat,
        "lon": c.centroid_lon,
        "radius_m": c.radius_m,
        "sites": sorted(c.sites),
        "equipment": sorted(c.equipment_types),
        "start": c.start.isoformat() if c.start else None,
        "end": c.end.isoformat() if c.end else None,
        "recommendation": incident.recommendation,
        "evidence": incident.detail.get("evidence", []),
        # The reading, if one fits. Carried whole -- headline, sentence and
        # falsifier -- because a hedged claim shown without what would
        # disprove it is the one form of this that is worse than silence.
        "signature": incident.detail.get("signature") or None,
        # What a median-and-sigma check makes of the same sensors. This is the
        # panel the client verifies the system with -- "is this actually new?"
        "conventional": incident.detail.get("conventional") or None,
        "correlation": incident.neighbour_correlation,
        "rainfall_mm": incident.rainfall_mm,
        "ack": incident.ack_state.value,
        "members": [
            _member_json(m, series.get(m.sensor.sensor_key),
                         checks.get(m.sensor.sensor_key))
            for m in c.members
        ],
    }


def _member_json(m, trace: list[list[float]] | None, check: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "key": m.sensor.sensor_key,
        "desc": m.sensor.description,
        "equipment": m.sensor.equipment,
        "site": m.sensor.site,
        "type": m.dominant_type.value,
        "score": round(m.score, 1),
        "deviation": m.severity.deviation,
        "unit": m.severity.unit,
        "duration_s": m.severity.duration_s,
        "start": m.start.isoformat(),
        "end": m.end.isoformat(),
        "start_ms": int(m.start.timestamp() * 1000),
        "end_ms": int(m.end.timestamp() * 1000),
        "detectors": sorted({s.detector for s in m.signals}),
    }
    if trace:
        out["series"] = trace
    if check is not None:
        # The band the operator's own check would have drawn, so the chart can
        # show the event sitting inside it rather than asserting that it does.
        out["band"] = {
            "mean": check.mean, "std": check.std, "k": check.k,
            "peak_z": check.peak_z, "peak_z_excluded": check.peak_z_excluded,
            "verdict": check.verdict,
        }
    return out


def _charted_keys(result) -> set[str]:
    """
    Which sensors get a trace embedded, worst incident first.

    Budgeted rather than exhaustive. The incidents that need a decision are
    charted; a suppressed fan-out of 116 sensors is not, because nobody is
    going to scroll through 116 traces of the same dead RTU.
    """
    ranked = sorted(result.incidents, key=lambda i: -i.severity)
    keys: set[str] = set()
    for incident in ranked:
        for member in incident.cluster.members:
            if len(keys) >= MAX_CHARTED_SENSORS:
                return keys
            keys.add(member.sensor.sensor_key)
    return keys


def build_payload(result) -> dict[str, Any]:
    """The JSON the page renders. Also useful on its own, for tests and the API."""
    series = _series_for(result, _charted_keys(result))
    checks = getattr(result, "conventional", None) or {}
    incidents = [_incident_json(i, series=series, checks=checks)
                 for i in result.incidents]
    matrix = result.region_matrix
    equipment = sorted({e for row in matrix.values() for e in row})
    regions = [r for r in REGION_ORDER if r in matrix] + \
              sorted(r for r in matrix if r not in REGION_ORDER)

    return {
        "run_id": result.run_id,
        "generated_at": result.started_at.isoformat(),
        "window": {
            "start": result.window_start.isoformat() if result.window_start else None,
            "end": result.window_end.isoformat() if result.window_end else None,
        },
        "duration_s": result.duration_s,
        "stats": result.stats,
        "incidents": incidents,
        "matrix": {"regions": regions, "equipment": equipment, "counts": matrix},
        "rainfall": result.rainfall_by_region,
        "conventional": (result.stats or {}).get("conventional") or {},
        "kpi": {
            "incidents": len(result.incidents),
            "alertable": len(result.alertable),
            "suppressed": len(result.incidents) - len(result.alertable),
            "p1": sum(1 for i in result.incidents if i.priority is Priority.P1),
            "sensors": len({a.sensor.sensor_key for a in result.anomalies}),
            "clusters": len(result.clusters),
            "multi_site": sum(1 for c in result.clusters if len(c.sites) > 1),
        },
    }


#: OpenStreetMap's own tiles: no account, no key, no registration.
#:
#: Two defaults have now failed, differently, and neither failure was
#: visible from the code:
#:
#:   CARTO's basemap CDN began answering with tiles that READ "API key
#:   required", so the map drew perfectly out of error notices.
#:
#:   OSM answers 403 -- "not following the tile usage policy" -- to a page
#:   opened from file://, because there is no Referer identifying the app.
#:   Serving the output directory over HTTP is enough to satisfy it.
#:
#: Hence the tileerror handler below: whichever provider is configured, the
#: page must stay readable when it refuses.
#:
#: Alternatives for DAS2_REPORT_MAP_TILE_URL, none of which I can reach from
#: the machine this was written on, so all are unverified:
#:
#:   OneMap (Singapore Land Authority) -- the natural choice for a PUB
#:   system: national coverage with canals, drains and reservoirs drawn
#:   properly. Check whether it now wants a token.
#:     https://www.onemap.gov.sg/maps/tiles/Default/{z}/{x}/{y}.png
#:
#:   Esri, which serves {z}/{y}/{x} -- note the order -- and has historically
#:   allowed use without a key:
#:     https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}
#:
#: On a network with no internet, an internal tile server is the only real
#: answer; nothing public is reachable and no default can help.
DEFAULT_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
DEFAULT_TILE_ATTRIBUTION = (
    '&copy; <a href="https://www.openstreetmap.org/copyright">'
    'OpenStreetMap</a> contributors'
)


def render(result, *, title: str = "DAS2 — Sensor Intelligence",
           tile_url: str = "", tile_attribution: str = "") -> str:
    """Render the full HTML page as a string."""
    payload = build_payload(result)
    data = json.dumps(payload, default=str, separators=(",", ":"))
    return (_TEMPLATE
            .replace("__TITLE__", html.escape(title))
            .replace("__TILE_URL__",
                     json.dumps(tile_url or DEFAULT_TILE_URL))
            .replace("__TILE_ATTRIBUTION__",
                     json.dumps(tile_attribution or DEFAULT_TILE_ATTRIBUTION))
            .replace("__PAYLOAD__", data.replace("</", "<\\/")))


def write(result, out_dir: str | Path, *,
          title: str = "DAS2 — Sensor Intelligence",
          filename: str = "",
          tile_url: str = "", tile_attribution: str = "") -> Path:
    """
    Write the dashboard HTML and return its path.

    `filename` defaults to `dashboard_<run_id>.html`. A caller writing into a
    per-run directory passes `dashboard.html`, since the directory already
    carries the run id and repeating it reads as a mistake.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (filename or f"dashboard_{result.run_id}.html")
    path.write_text(
        render(result, title=title, tile_url=tile_url,
               tile_attribution=tile_attribution),
        encoding="utf-8")
    return path


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
  :root {
    --bg: #f6f7f9; --card: #ffffff; --ink: #14181d; --muted: #667080;
    --line: #dfe3e8; --accent: #b2182b;
    /* Status, not series. "new" = their check missed it and this one did not;
       "known" = it would have fired too. Each ships with a word beside it, so
       the colour is never the only carrier. */
    --new: #1a6b45; --known: #8c5000; --band: rgba(102,112,128,.16);
  }
  @media (prefers-color-scheme: dark) {
    /* Stepped for the dark surface rather than flipped: #1a6b45 on #1c2128 is
       below any usable contrast, and an automatic inversion would leave it
       technically present and practically invisible. */
    :root { --bg:#14181d; --card:#1c2128; --ink:#e8ecf1; --muted:#98a3b3;
            --line:#2c333c; --new:#63c39a; --known:#e0a95f;
            --band:rgba(152,163,179,.18); }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
  .wrap { max-width:1280px; margin:0 auto; padding:24px 16px 64px; }
  h1 { font-size:22px; margin:0 0 4px; }
  .sub { color:var(--muted); font-size:13px; margin-bottom:20px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px;
          padding:16px; margin-bottom:20px; }
  .card h2 { font-size:15px; margin:0 0 12px; letter-spacing:.02em;
             text-transform:uppercase; color:var(--muted); }
  .kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr));
          gap:12px; margin-bottom:20px; }
  .kpi { background:var(--card); border:1px solid var(--line); border-radius:10px;
         padding:14px 16px; }
  .kpi .n { font-size:28px; font-weight:650; line-height:1.1; }
  .kpi .l { font-size:12px; color:var(--muted); margin-top:2px; }
  .kpi.alarm .n { color:var(--accent); }
  #map { height:440px; border-radius:8px; z-index:0; }
  /* No basemap: a plain ground so the markers read as positions on nothing,
     rather than as a map that failed to finish drawing. */
  #map.nobasemap { background:#eef1f5; }
  #map.nobasemap .leaflet-tile-pane { display:none; }
  .note { font-size:12px; color:var(--muted); margin-top:10px; }
  table { border-collapse:collapse; width:100%; font-size:13px; }
  th, td { text-align:left; padding:7px 9px; border-bottom:1px solid var(--line);
           vertical-align:top; }
  th { color:var(--muted); font-weight:600; font-size:11px;
       text-transform:uppercase; letter-spacing:.04em; }
  .badge { display:inline-block; padding:2px 8px; border-radius:999px;
           font-size:11px; font-weight:650; color:#fff; white-space:nowrap; }
  .mx td.cell { text-align:center; font-variant-numeric:tabular-nums;
                font-weight:600; color:#14181d; }
  .mx td.zero { color:var(--muted); font-weight:400; }
  .inc { cursor:pointer; }
  .inc:hover { background:rgba(127,127,127,.07); }
  .det { display:none; background:rgba(127,127,127,.05); }
  .det.open { display:table-row; }
  .det td { padding:14px 16px; }
  .ev { margin:0 0 10px; padding-left:18px; }
  .ev li { margin:2px 0; color:var(--muted); }
  .rec { font-weight:600; margin-bottom:10px; }
  /* The reading sits BELOW the recommendation and is set apart from it, so
     it cannot be mistaken for the verdict it describes. */
  .sig { margin:0 0 10px; padding:8px 12px; border-left:3px solid var(--muted);
         background:rgba(127,127,127,.06); }
  .sig .muted { font-size:12px; }
  .muted { color:var(--muted); }
  .suppressed td { opacity:.5; }
  .empty { text-align:center; padding:40px; color:var(--muted); }
  .legend { font-size:12px; color:var(--muted); display:flex; gap:14px;
            flex-wrap:wrap; margin-top:10px; }
  .legend span b { display:inline-block; width:10px; height:10px;
                   border-radius:50%; margin-right:5px; }

  /* Filters sit in ONE row directly above the thing they filter, so the
     relationship is positional and needs no explaining. */
  .filters { display:flex; gap:14px; flex-wrap:wrap; align-items:flex-end;
             margin-bottom:12px; font-size:12px; color:var(--muted); }
  .filters label { display:flex; flex-direction:column; gap:4px; }
  .filters select, .filters input, .filters button {
      font:inherit; font-size:13px; color:var(--ink); background:var(--card);
      border:1px solid var(--line); border-radius:6px; padding:5px 8px; }
  .filters button { cursor:pointer; }
  .filters button:hover { border-color:var(--muted); }

  /* The verification panel: what their own check says about these sensors. */
  .cv { margin:12px 0; padding:10px 12px; border:1px solid var(--line);
        border-radius:8px; }
  .cv h4 { margin:0 0 6px; font-size:13px; }
  .cv .because { color:var(--muted); font-size:12px; margin:6px 0 8px; }
  .cv table { font-size:12px; }
  .v-alarmed   { color:var(--known); font-weight:600; }
  .v-missed    { color:var(--new); font-weight:600; }
  .v-chance    { color:var(--muted); font-weight:600; }
  .v-undefined { color:var(--muted); font-weight:600; }

  /* Charts. One series each, so no legend: the caption names the sensor. */
  .traces { display:grid; gap:12px; margin-top:12px;
            grid-template-columns:repeat(auto-fit, minmax(320px, 1fr)); }
  .trace { border:1px solid var(--line); border-radius:8px; padding:8px 10px; }
  .trace .cap { font-size:12px; margin-bottom:2px; }
  .trace .sub { font-size:11px; color:var(--muted); margin-bottom:4px; }
  .trace svg { display:block; width:100%; height:110px; overflow:visible; }
  .tip { position:fixed; pointer-events:none; z-index:9999; display:none;
         background:var(--ink); color:var(--card); font-size:11px;
         padding:4px 7px; border-radius:5px; white-space:nowrap; }
  .tl { font-size:12px; margin-top:10px; }
  .tl td { padding:3px 8px 3px 0; border:0; }
  /* `display:block` is load-bearing: a `span` is inline, and height on an
     inline box is ignored, so the first version drew the delays with no bars
     beside them. */
  .tl .bar { display:block; width:100%; min-width:120px; height:8px;
             background:var(--line); border-radius:3px; position:relative; }
  .tl .bar i { position:absolute; top:0; height:8px; border-radius:3px;
               background:var(--accent); display:block; min-width:3px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Water Sensor Intelligence</h1>
  <div class="sub" id="sub"></div>
  <div class="kpis" id="kpis"></div>

  <div class="card">
    <h2>Where — incident map</h2>
    <div id="map"></div>
    <div class="legend" id="legend"></div>
    <div class="note" id="maphint"></div>
    <div class="note">
      Marker size reflects how many sensors are involved; colour is priority.
      Coordinates come from an RTU-level join, so all sensors at one site share
      a position — the map resolves <b>sites</b>, not individual instruments.
    </div>
  </div>

  <div class="card">
    <h2>By region and by type</h2>
    <div id="matrix"></div>
    <div class="note">
      A column lit across every region is a fleet-wide equipment problem; a row
      lit across every type is something wrong with that area.
    </div>
  </div>

  <div class="card">
    <h2>Incidents — what to do</h2>
    <div class="filters" id="filters">
      <label>Region <select id="f-region"></select></label>
      <label>Priority <select id="f-priority"></select></label>
      <label>Class <select id="f-class"></select></label>
      <label>Parameter <select id="f-equipment"></select></label>
      <label>Search <input id="f-text" type="search" placeholder="site or sensor"></label>
      <button id="f-reset" type="button">Reset</button>
      <span class="muted" id="f-count"></span>
    </div>
    <div id="incidents"></div>
  </div>
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const DATA = __PAYLOAD__;
const PC = {"P1":"#b2182b","P2":"#ef8a62","P3":"#f7d08a","P4":"#9fb8c8"};
const CC = {"REGIONAL_EVENT":"#b2182b","SENSOR_FAULT":"#d6604d",
            "PROCESS_EVENT":"#4393c3","DRIFT_MAINTENANCE":"#92c5de",
            "WEATHER_DRIVEN":"#2166ac","INSTRUMENT_CONFLICT":"#c0504d",
            "ASSET_FAILURE":"#a63603",
            "WATCH":"#999999",
            "TELEMETRY_FANOUT":"#777777"};
const esc = s => String(s == null ? "" : s).replace(/[&<>"']/g,
  c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

/* ---- header ---- */
const w = DATA.window || {};
document.getElementById("sub").textContent =
  `Run ${DATA.run_id} · window ${(w.start||"?").replace("T"," ").slice(0,16)} → ` +
  `${(w.end||"?").replace("T"," ").slice(0,16)} · analysed in ${DATA.duration_s}s`;

const k = DATA.kpi;
document.getElementById("kpis").innerHTML = [
  [k.p1, "P1 — act now", k.p1 > 0],
  [k.alertable, "incidents to action", false],
  [k.suppressed, "suppressed as noise", false],
  [k.multi_site, "multi-site clusters", false],
  [k.sensors, "sensors abnormal", false],
].map(([n, l, alarm]) =>
  `<div class="kpi${alarm ? " alarm" : ""}"><div class="n">${n}</div>
   <div class="l">${l}</div></div>`).join("");

/* ---- map ---- */
/* Leaflet and its tiles come from a CDN. An operations network is quite
   likely to have neither, and "where?" is the question this page exists to
   answer, so an unreachable CDN must not cost the map entirely. When Leaflet
   is missing the same incidents are drawn as an inline SVG scatter over
   Singapore's extent: no basemap, no interactivity, but the geography is
   still legible and the page stays genuinely self-contained. */
const placed = DATA.incidents.filter(i => i.lat != null && i.lon != null);
const SG = {lon0: 103.60, lon1: 104.10, lat0: 1.20, lat1: 1.48};

function svgFallbackMap(el, incidents) {
  const W = 1000, H = 560, PAD = 34;
  const x = lon => PAD + (lon - SG.lon0) / (SG.lon1 - SG.lon0) * (W - 2 * PAD);
  // SVG y grows downward; latitude grows upward.
  const y = lat => H - PAD - (lat - SG.lat0) / (SG.lat1 - SG.lat0) * (H - 2 * PAD);

  const grid = [];
  for (let lon = 103.6; lon <= 104.1001; lon += 0.1)
    grid.push(`<line x1="${x(lon)}" y1="${PAD}" x2="${x(lon)}" y2="${H - PAD}"/>`
            + `<text x="${x(lon)}" y="${H - PAD + 16}" class="ax">${lon.toFixed(1)}</text>`);
  for (let lat = 1.2; lat <= 1.4801; lat += 0.1)
    grid.push(`<line x1="${PAD}" y1="${y(lat)}" x2="${W - PAD}" y2="${y(lat)}"/>`
            + `<text x="4" y="${y(lat) + 4}" class="ax">${lat.toFixed(1)}</text>`);

  const dots = incidents.slice().sort((a, b) => a.members.length - b.members.length)
    .map(i => {
      const r = 7 + Math.sqrt(i.members.length) * 4;
      const label = (i.sites.slice(0, 2).join(", ") || i.region);
      return `<g>
        ${i.radius_m > 100 ? `<circle cx="${x(i.lon)}" cy="${y(i.lat)}"
           r="${Math.max(r, i.radius_m / (SG.lon1 - SG.lon0) / 111320 * (W - 2 * PAD))}"
           fill="${PC[i.priority]}" fill-opacity=".08"
           stroke="${PC[i.priority]}" stroke-opacity=".4"/>` : ""}
        <circle cx="${x(i.lon)}" cy="${y(i.lat)}" r="${r}"
                fill="${PC[i.priority] || "#999"}" fill-opacity=".88"
                stroke="#fff" stroke-width="1.6"><title>${esc(i.priority)} ${esc(i.cls)}
${esc(i.region)} — ${i.members.length} sensor(s)
${esc(i.recommendation)}</title></circle>
        <text x="${x(i.lon)}" y="${y(i.lat) - r - 6}" class="lb">${esc(label)}</text>
      </g>`;
    }).join("");

  el.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" width="100%" height="100%"
         preserveAspectRatio="xMidYMid meet" role="img"
         aria-label="Incident locations across Singapore">
      <style>
        .gr line { stroke:#dde2e8; stroke-width:1 }
        .ax { fill:#8b96a5; font-size:11px; text-anchor:middle }
        .lb { fill:#14181d; font-size:12px; text-anchor:middle; font-weight:600;
              paint-order:stroke; stroke:#fff; stroke-width:3px }
      </style>
      <rect x="0" y="0" width="${W}" height="${H}" fill="#f4f6f8" rx="8"/>
      <g class="gr">${grid.join("")}</g>
      ${dots}
    </svg>`;
}

const mapEl = document.getElementById("map");
if (!placed.length) {
  mapEl.style.height = "auto";
  mapEl.innerHTML = '<div class="empty">No incident has coordinates this run.</div>';
} else if (typeof L === "undefined") {
  svgFallbackMap(mapEl, placed);
  document.getElementById("maphint").textContent =
    "Map tiles are unreachable, so this is drawn from the coordinates alone — " +
    "positions are exact, there is simply no basemap behind them.";
} else {
  try {
    const map = L.map("map").setView([1.3521, 103.8198], 11);
    // The basemap, and a way out when it fails.
    //
    // Two providers have now failed differently: CARTO began serving tiles
    // that read "API key required", and OpenStreetMap answers 403 to a page
    // opened from file:// because there is no Referer to identify the app.
    // Both produced a map made entirely of error tiles, which is worse than
    // no basemap -- it reads as a bug in this page.
    //
    // So tile failures are counted, and past a handful the layer is dropped
    // and the page says what happened. Markers, zoom, pan and popups all keep
    // working; only the geography behind them is missing, which is exactly
    // the state svgFallbackMap was written for. No provider's change of terms
    // can make this page unreadable again.
    const tiles = L.tileLayer(__TILE_URL__,
      {maxZoom: 19, attribution: __TILE_ATTRIBUTION__});
    let tileErrors = 0;
    tiles.on("tileerror", () => {
      if (++tileErrors < 5 || !map.hasLayer(tiles)) return;
      map.removeLayer(tiles);
      mapEl.classList.add("nobasemap");
      document.getElementById("maphint").textContent =
        "The tile provider refused (blocked, rate-limited or needs a key), " +
        "so there is no basemap behind these positions — they are still " +
        "exact. Opening this page over HTTP rather than from a file, or " +
        "setting DAS2_REPORT_MAP_TILE_URL to an internal tile server, " +
        "restores it.";
    });
    tiles.addTo(map);

    const group = [];
    placed.forEach(i => {
      // Area scales with member count, so ten sensors do not look a hundred
      // times worse than one.
      const r = 7 + Math.sqrt(i.members.length) * 4;
      L.circleMarker([i.lat, i.lon], {
        radius: r, color: "#fff", weight: 1.5,
        fillColor: PC[i.priority] || "#999", fillOpacity: .85
      }).addTo(map).bindPopup(
        `<b>${esc(i.priority)} · ${esc(i.cls)}</b><br>` +
        `${esc(i.region)} — ${i.members.length} sensor(s) at ${i.sites.length} site(s)<br>` +
        `<span style="color:#667080">${esc(i.sites.join(", "))}</span><br><br>` +
        `<b>${esc(i.recommendation)}</b>`);
      // A cluster's true extent, so a wide one cannot be mistaken for a pin.
      if (i.radius_m > 100) {
        L.circle([i.lat, i.lon], {radius: i.radius_m, color: PC[i.priority],
          weight: 1, opacity: .5, fillOpacity: .06}).addTo(map);
      }
      group.push([i.lat, i.lon]);
    });
    if (group.length) map.fitBounds(group, {padding: [50, 50], maxZoom: 13});
  } catch (e) {
    svgFallbackMap(mapEl, placed);
  }
}
document.getElementById("legend").innerHTML =
  Object.entries(PC).map(([p, c]) =>
    `<span><b style="background:${c}"></b>${p}</span>`).join("") +
  `<span class="muted">circle = cluster extent</span>`;

/* ---- region x equipment matrix ---- */
const mx = DATA.matrix;
if (!mx.equipment.length) {
  document.getElementById("matrix").innerHTML =
    '<div class="empty">No anomalies this run.</div>';
} else {
  const max = Math.max(1, ...mx.regions.flatMap(
    r => mx.equipment.map(e => (mx.counts[r] || {})[e] || 0)));
  const rows = mx.regions.map(r => {
    const rain = DATA.rainfall[r];
    const label = esc(r) + (rain ? ` <span class="muted">(${rain} mm)</span>` : "");
    const cells = mx.equipment.map(e => {
      const v = (mx.counts[r] || {})[e] || 0;
      // Lightness carries the count; a sequential single-hue ramp keeps the
      // ordering readable in greyscale and for colour-blind readers.
      const t = v / max;
      const bg = v ? `background:rgba(178,24,43,${0.10 + 0.75 * t})` : "";
      const fg = t > 0.55 ? "color:#fff" : "";
      return `<td class="cell ${v ? "" : "zero"}" style="${bg};${fg}">${v || "·"}</td>`;
    }).join("");
    const total = mx.equipment.reduce((s, e) => s + ((mx.counts[r] || {})[e] || 0), 0);
    // The row header drills into the region, which is the "observe the anomaly
    // by region" half of the original ask made clickable.
    return `<tr><th><a href="#" onclick="focusRegion('${esc(r)}');return false"
            >${label}</a></th>${cells}<td class="cell">${total}</td></tr>`;
  }).join("");
  document.getElementById("matrix").innerHTML =
    `<table class="mx"><thead><tr><th></th>` +
    mx.equipment.map(e => `<th>${esc(e)}</th>`).join("") +
    `<th>Total</th></tr></thead><tbody>${rows}</tbody></table>`;
}

/* ---- the verification panel -------------------------------------------- *
 * The client's own test of this system: "is there really something here my
 * existing statistics did not find?". So his check is run on the same sensors
 * and the answer is printed whichever way it comes out.                      */
function conventionalPanel(i) {
  const c = i.conventional;
  if (!c) return "";
  const rows = c.sensors.map(s => `
    <tr><td>${esc(s.sensor)}</td>
        <td class="v-${esc(s.verdict)}">${esc(s.verdict)}</td>
        <td>${s.peak_z.toFixed(1)}σ</td>
        <td>${s.peak_z_excluded != null
              ? s.peak_z_excluded.toFixed(1) + "σ"
              : '<span class="muted">—</span>'}</td>
        <td class="muted">${esc(s.why)}</td></tr>`).join("");
  return `<div class="cv">
    <h4>Would your median ± 3σ check have found this?</h4>
    <div>${esc(c.headline)}</div>
    <div class="because">Found here because ${esc(c.found_because)}</div>
    <table><thead><tr><th>Sensor</th><th>Their check</th><th>Peak σ</th>
      <th>σ vs before the event</th><th>Why</th></tr></thead>
      <tbody>${rows}</tbody></table>
    <div class="note">“Peak σ” uses the whole window, as a check run over this
      window would. The next column recomputes it with the event's own samples
      taken out of the baseline — where the two disagree, the event inflated
      the σ meant to reveal it.</div>
  </div>`;
}

/* ---- onset order -------------------------------------------------------- *
 * Which sensor moved first is the cheapest root-cause evidence available, and
 * it needs no model: it is in the timestamps already.                        */
function timeline(i) {
  const ms = i.members.filter(m => m.start_ms);
  if (ms.length < 2) return "";
  const t0 = Math.min(...ms.map(m => m.start_ms));
  const t1 = Math.max(...ms.map(m => m.end_ms));
  const span = Math.max(1, t1 - t0);
  const rows = ms.slice().sort((a, b) => a.start_ms - b.start_ms).map((m, n) => {
    const left = 100 * (m.start_ms - t0) / span;
    const width = Math.max(1.5, 100 * (m.end_ms - m.start_ms) / span);
    const delay = Math.round((m.start_ms - t0) / 60000);
    return `<tr><td class="muted">${n + 1}</td><td>${esc(m.desc)}</td>
      <td class="muted">${esc(m.equipment)}</td>
      <td style="width:100%"><span class="bar">
        <i style="left:${left}%;width:${width}%"></i></span></td>
      <td class="muted">${delay ? "+" + delay + " min" : "first"}</td></tr>`;
  }).join("");
  return `<div class="tl"><b>Order of onset</b>
    <table class="tl"><tbody>${rows}</tbody></table>
    <div class="note">Ordering only. Which instrument reacted first is not
      proof of where the cause is — a sensor nearer the source reports sooner,
      and so does one that simply samples faster.</div></div>`;
}

/* ---- one sensor's trace ------------------------------------------------- *
 * A single series, so no legend: the caption names it. The band is the
 * operator's own mean ± 3σ, drawn so the page shows the event sitting inside
 * it rather than asserting that it does.                                     */
const TIP = document.createElement("div");
TIP.className = "tip";
document.body.appendChild(TIP);

function trace(m) {
  if (!m.series || m.series.length < 2) return "";
  const W = 320, H = 110, PADL = 4, PADR = 4, PADT = 8, PADB = 8;
  const xs = m.series.map(p => p[0]), ys = m.series.map(p => p[1]);
  const b = m.band;
  // Scale to the DATA, then admit as much of the band as fits within twice
  // that range. Scaling to the band instead flattened every trace into a
  // straight line whenever sigma was large -- which is exactly the sensor
  // whose shape the reader needs to see, since a large sigma is why their
  // check missed it. The band is clipped and the caption says so.
  let lo = Math.min(...ys), hi = Math.max(...ys);
  const pad = (hi - lo) * 0.08 || Math.abs(hi) * 0.01 || 1;
  lo -= pad; hi += pad;
  let clipped = false;
  if (b && b.std > 0) {
    const room = (hi - lo);
    const want = [b.mean - b.k * b.std, b.mean + b.k * b.std];
    const nlo = Math.max(lo - room, Math.min(lo, want[0]));
    const nhi = Math.min(hi + room, Math.max(hi, want[1]));
    clipped = (want[0] < nlo - 1e-9) || (want[1] > nhi + 1e-9);
    lo = nlo; hi = nhi;
  }
  if (hi - lo < 1e-9) { hi = lo + 1; lo = lo - 1; }
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const sx = t => PADL + (W - PADL - PADR) * (t - x0) / Math.max(1, x1 - x0);
  const sy = v => PADT + (H - PADT - PADB) * (1 - (v - lo) / (hi - lo));

  const path = m.series.map((p, n) =>
    (n ? "L" : "M") + sx(p[0]).toFixed(1) + " " + sy(p[1]).toFixed(1)).join(" ");
  let band = "";
  if (b && b.std > 0) {
    const top = Math.max(0, sy(b.mean + b.k * b.std));
    const bot = Math.min(H, sy(b.mean - b.k * b.std));
    const mid = sy(b.mean);
    band = `<rect x="0" y="${top.toFixed(1)}" width="${W}"
             height="${Math.max(0.5, bot - top).toFixed(1)}"
             fill="var(--band)"></rect>` +
           (mid >= 0 && mid <= H
            ? `<line x1="0" x2="${W}" y1="${mid.toFixed(1)}"
                y2="${mid.toFixed(1)}" stroke="var(--muted)"
                stroke-width="1" stroke-dasharray="3 3"
                vector-effect="non-scaling-stroke"></line>` : "");
  }
  // The flagged span gets edges as well as a tint. A 12%-opacity fill laid
  // over the sigma band was invisible in every chart it mattered in, and a
  // span whose extent cannot be seen is not evidence of anything.
  const ex0 = sx(m.start_ms), ex1 = sx(m.end_ms);
  const hue = PC[m._pri] || "#b2182b";
  const flagged = `<rect x="${ex0.toFixed(1)}" y="0"
      width="${Math.max(1.5, ex1 - ex0).toFixed(1)}" height="${H}"
      fill="${hue}" opacity=".18"></rect>
    <line x1="${ex0.toFixed(1)}" x2="${ex0.toFixed(1)}" y1="0" y2="${H}"
      stroke="${hue}" stroke-width="1" opacity=".75"
      vector-effect="non-scaling-stroke"></line>
    <line x1="${ex1.toFixed(1)}" x2="${ex1.toFixed(1)}" y1="0" y2="${H}"
      stroke="${hue}" stroke-width="1" opacity=".75"
      vector-effect="non-scaling-stroke"></line>`;
  // The scale goes in the caption, not in the SVG. The chart is stretched to
  // the column width with `preserveAspectRatio="none"`, which distorts any
  // glyph drawn inside it; strokes escape that through `non-scaling-stroke`,
  // text has no equivalent.
  const range = `${Math.min(...ys).toPrecision(4)} – ` +
                `${Math.max(...ys).toPrecision(4)}`;
  const verdict = b ? ` · their check: <span class="v-${esc(b.verdict)}">` +
                      `${esc(b.verdict)}</span> at ${b.peak_z.toFixed(1)}σ` : "";
  return `<div class="trace">
    <div class="cap">${esc(m.desc)}</div>
    <div class="sub">${esc(m.equipment)}${m.unit ? " · " + esc(m.unit) : ""}
      · ${esc(range)}${verdict}</div>
    <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none"
         data-series='${esc(JSON.stringify(m.series))}'
         data-unit="${esc(m.unit || "")}"
         data-x0="${x0}" data-x1="${x1}">
      ${band}${flagged}
      <path d="${path}" fill="none" stroke="var(--ink)" stroke-width="2"
            vector-effect="non-scaling-stroke"
            stroke-linejoin="round" stroke-linecap="round"></path>
      <line class="cross" y1="0" y2="${H}" stroke="var(--muted)"
            stroke-width="1" vector-effect="non-scaling-stroke"
            style="display:none"></line>
    </svg>
    <div class="note">Shaded band: mean ± 3σ over this window${clipped
      ? ", wider than this view and clipped to it" : ""}. Tinted span: what
      this system flagged.</div>
  </div>`;
}

/* Crosshair and tooltip. An SVG chart in a page IS interactive; a trace with
   no readout makes the reader estimate values off a 110px axis. */
function armTraces(root) {
  root.querySelectorAll(".trace svg").forEach(svg => {
    let pts = null;
    const line = svg.querySelector(".cross");
    svg.addEventListener("pointermove", ev => {
      if (!pts) { try { pts = JSON.parse(svg.dataset.series); } catch (e) { pts = []; } }
      if (!pts.length) return;
      const box = svg.getBoundingClientRect();
      const frac = (ev.clientX - box.left) / Math.max(1, box.width);
      const t = +svg.dataset.x0 + frac * (+svg.dataset.x1 - +svg.dataset.x0);
      let best = pts[0];
      for (const p of pts) if (Math.abs(p[0] - t) < Math.abs(best[0] - t)) best = p;
      line.setAttribute("x1", frac * 320);
      line.setAttribute("x2", frac * 320);
      line.style.display = "";
      TIP.textContent = new Date(best[0]).toISOString().replace("T", " ").slice(5, 16)
                      + "  ·  " + best[1] + " " + svg.dataset.unit;
      TIP.style.display = "block";
      TIP.style.left = (ev.clientX + 12) + "px";
      TIP.style.top = (ev.clientY - 28) + "px";
    });
    svg.addEventListener("pointerleave", () => {
      line.style.display = "none";
      TIP.style.display = "none";
    });
  });
}

/* ---- incident table, filtered ------------------------------------------- */
const inc = DATA.incidents;
const F = {region: "", priority: "", cls: "", equipment: "", text: ""};

function options(sel, values, label) {
  sel.innerHTML = `<option value="">${label}</option>` +
    values.map(v => `<option value="${esc(v)}">${esc(v)}</option>`).join("");
}

function matches(i) {
  if (F.region && i.region !== F.region) return false;
  if (F.priority && i.priority !== F.priority) return false;
  if (F.cls && i.cls !== F.cls) return false;
  if (F.equipment && !i.equipment.includes(F.equipment)) return false;
  if (F.text) {
    const hay = (i.sites.join(" ") + " " + i.id + " " +
                 i.members.map(m => m.desc).join(" ")).toLowerCase();
    if (!hay.includes(F.text.toLowerCase())) return false;
  }
  return true;
}

function renderIncidents() {
  const host = document.getElementById("incidents");
  const shown = inc.filter(matches);
  document.getElementById("f-count").textContent =
    `${shown.length} of ${inc.length} shown`;
  if (!inc.length) {
    host.innerHTML = '<div class="empty">No incidents. Every sensor behaved ' +
      'within its own normal range this run.</div>';
    return;
  }
  if (!shown.length) {
    host.innerHTML = '<div class="empty">Nothing matches these filters.</div>';
    return;
  }
  const rows = shown.map((i, n) => {
    const quiet = i.cls === "TELEMETRY_FANOUT" || i.cls === "WATCH";
    const members = i.members.map(m =>
      `<tr><td>${esc(m.desc)}</td><td>${esc(m.equipment)}</td>
       <td>${esc(m.type)}</td>
       <td>${m.deviation ? esc(m.deviation) + " " + esc(m.unit) : "—"}</td>
       <td>${Math.round(m.duration_s / 60)} min</td>
       <td>${esc(m.detectors.join(", "))}</td></tr>`).join("");
    const charts = i.members.filter(m => m.series).slice(0, 6)
      .map(m => trace(Object.assign({_pri: i.priority}, m))).join("");
    return `
    <tr class="inc${quiet ? " suppressed" : ""}" data-det="d${n}">
      <td><span class="badge" style="background:${PC[i.priority]}">${esc(i.priority)}</span></td>
      <td><span class="badge" style="background:${CC[i.cls] || "#888"}">${esc(i.cls)}</span></td>
      <td>${esc(i.region)}</td>
      <td>${i.members.length}</td>
      <td>${i.sites.length}</td>
      <td>${esc(i.sites.join(", ")) || '<span class="muted">unnamed</span>'}</td>
      <td>${i.severity}</td>
      <td>${esc(i.recommendation)}</td>
    </tr>
    <tr class="det" id="d${n}"><td colspan="8">
      <div class="rec">${esc(i.recommendation)}</div>
      <ul class="ev">${i.evidence.map(e => `<li>${esc(e)}</li>`).join("")}</ul>
      ${i.signature ? `<div class="sig"><b>${esc(i.signature.headline)}</b>
        <div>${esc(i.signature.reads_as)}</div>
        <div class="muted">Would change this reading:
          ${esc(i.signature.would_change_it)}</div>
        <div class="muted">${esc(i.signature.caveat)}</div></div>` : ""}
      ${i.rainfall_mm != null ? `<div class="muted">Rainfall nearby: ${i.rainfall_mm} mm</div>` : ""}
      ${i.correlation != null ? `<div class="muted">Neighbour correlation: r=${i.correlation}</div>` : ""}
      ${conventionalPanel(i)}
      ${timeline(i)}
      <table style="margin-top:10px">
        <thead><tr><th>Sensor</th><th>Equipment</th><th>Fault</th>
        <th>Deviation</th><th>Duration</th><th>Found by</th></tr></thead>
        <tbody>${members}</tbody></table>
      ${charts ? `<div class="traces">${charts}</div>` : ""}
      <div class="note">Incident ${esc(i.id)}</div>
    </td></tr>`;
  }).join("");
  host.innerHTML =
    `<table><thead><tr><th>Pri</th><th>Class</th><th>Region</th><th>Sensors</th>
     <th>Sites</th><th>Where</th><th>Sev</th><th>Recommendation</th></tr></thead>
     <tbody>${rows}</tbody></table>
     <div class="note">Click a row for the evidence behind the recommendation,
     what your own median ± 3σ check makes of the same sensors, and the traces
     it is all drawn from. Dimmed rows are suppressed and will not page
     anyone.</div>`;
  host.querySelectorAll("tr.inc").forEach(row => {
    row.addEventListener("click", () => {
      const det = document.getElementById(row.dataset.det);
      det.classList.toggle("open");
      if (det.classList.contains("open")) armTraces(det);
    });
  });
}

(function initFilters() {
  const uniq = f => [...new Set(inc.map(f).flat())].filter(Boolean).sort();
  options(document.getElementById("f-region"), uniq(i => i.region), "all");
  options(document.getElementById("f-priority"), uniq(i => i.priority), "all");
  options(document.getElementById("f-class"), uniq(i => i.cls), "all");
  options(document.getElementById("f-equipment"), uniq(i => i.equipment), "all");
  const bind = (id, key) => document.getElementById(id)
    .addEventListener("input", ev => { F[key] = ev.target.value; renderIncidents(); });
  bind("f-region", "region"); bind("f-priority", "priority");
  bind("f-class", "cls"); bind("f-equipment", "equipment");
  bind("f-text", "text");
  document.getElementById("f-reset").addEventListener("click", () => {
    Object.keys(F).forEach(k => F[k] = "");
    document.querySelectorAll(".filters select, .filters input")
      .forEach(el => el.value = "");
    renderIncidents();
  });
})();

/* Drilling into a region from the map or the matrix sets the same filter the
   controls do, so there is one state and not two. */
function focusRegion(region) {
  F.region = region;
  document.getElementById("f-region").value = region;
  renderIncidents();
  document.getElementById("incidents").scrollIntoView({behavior: "smooth",
                                                       block: "start"});
}
window.focusRegion = focusRegion;

renderIncidents();
</script>
</body>
</html>
"""
