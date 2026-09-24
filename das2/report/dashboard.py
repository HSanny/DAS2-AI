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
from typing import Any

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
    IncidentClass.WATCH.value: "#999999",
    IncidentClass.TELEMETRY_FANOUT.value: "#777777",
}

REGION_ORDER = ["Central", "East", "North", "North-East", "West", "Unknown"]


def _incident_json(incident: Incident) -> dict[str, Any]:
    c = incident.cluster
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
        "correlation": incident.neighbour_correlation,
        "rainfall_mm": incident.rainfall_mm,
        "ack": incident.ack_state.value,
        "members": [
            {
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
                "detectors": sorted({s.detector for s in m.signals}),
            }
            for m in c.members
        ],
    }


def build_payload(result) -> dict[str, Any]:
    """The JSON the page renders. Also useful on its own, for tests and the API."""
    incidents = [_incident_json(i) for i in result.incidents]
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
  }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#14181d; --card:#1c2128; --ink:#e8ecf1; --muted:#98a3b3;
            --line:#2c333c; }
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
    return `<tr><th>${label}</th>${cells}<td class="cell">${total}</td></tr>`;
  }).join("");
  document.getElementById("matrix").innerHTML =
    `<table class="mx"><thead><tr><th></th>` +
    mx.equipment.map(e => `<th>${esc(e)}</th>`).join("") +
    `<th>Total</th></tr></thead><tbody>${rows}</tbody></table>`;
}

/* ---- incident table ---- */
const inc = DATA.incidents;
if (!inc.length) {
  document.getElementById("incidents").innerHTML =
    '<div class="empty">No incidents. Every sensor behaved within its own ' +
    'normal range this run.</div>';
} else {
  const rows = inc.map((i, n) => {
    const quiet = i.cls === "TELEMETRY_FANOUT" || i.cls === "WATCH";
    const members = i.members.map(m =>
      `<tr><td>${esc(m.desc)}</td><td>${esc(m.equipment)}</td>
       <td>${esc(m.type)}</td>
       <td>${m.deviation ? esc(m.deviation) + " " + esc(m.unit) : "—"}</td>
       <td>${Math.round(m.duration_s / 60)} min</td>
       <td>${esc(m.detectors.join(", "))}</td></tr>`).join("");
    return `
    <tr class="inc${quiet ? " suppressed" : ""}" onclick="
        document.getElementById('d${n}').classList.toggle('open')">
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
      <table style="margin-top:10px">
        <thead><tr><th>Sensor</th><th>Equipment</th><th>Fault</th>
        <th>Deviation</th><th>Duration</th><th>Found by</th></tr></thead>
        <tbody>${members}</tbody></table>
      <div class="note">Incident ${esc(i.id)}</div>
    </td></tr>`;
  }).join("");
  document.getElementById("incidents").innerHTML =
    `<table><thead><tr><th>Pri</th><th>Class</th><th>Region</th><th>Sensors</th>
     <th>Sites</th><th>Where</th><th>Sev</th><th>Recommendation</th></tr></thead>
     <tbody>${rows}</tbody></table>
     <div class="note">Click a row for the evidence behind the recommendation.
     Dimmed rows are suppressed and will not page anyone.</div>`;
}
</script>
</body>
</html>
"""
