"""
das2.report.charts
==================

Static PNGs: the map and the matrix, drawn once and used twice.

`das2.report.pdf` puts the map on a full page of the run report; these
functions also write standalone files, which is what the `messages` alerting
mode sends and what a reader wants when they only need the one picture. An
operator reading a phone at 3 a.m. gets one image, so each chart answers
exactly one question and is legible at phone width:

  * `region_map_png`    -- where, on a real outline of Singapore
  * `region_matrix_png` -- by region and by type, the client's literal request
  * `sensor_detail_png` -- what the offending sensor actually did

Design constraints that are not negotiable here
-----------------------------------------------
* **Readable on a phone.** Large type, few elements, no legend that requires
  squinting. A chart that needs pinch-zoom has failed.
* **Colour is never the only channel.** Priority is encoded as colour *and*
  marker size *and* text, so the image survives greyscale and colour blindness.
* **No tiles.** The map is a real outline of Singapore, but it is drawn from a
  shoreline that ships inside the package -- see `das2.report.basemap` for why
  tile servers are not an option here. No internet, no API key, no 403.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # no display in a container
import matplotlib.patheffects as pe        # noqa: E402
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                         # noqa: E402

from das2.models import Incident            # noqa: E402
from das2.report import basemap, theme      # noqa: E402

theme.install()

#: Kept as a module-level name because other modules import it. The values
#: live in `theme` so the PDF and the PNGs cannot drift apart.
PRIORITY_COLOR = theme.PRIORITY_COLOR

#: Singapore's bounding box, so a single incident does not render on a map
#: zoomed so far in that the location is meaningless. Taken from the vendored
#: shoreline so the outline and the axes cannot disagree.
SG_BOUNDS = basemap.bounds()   # lon_min, lon_max, lat_min, lat_max

PHONE_FIGSIZE = (7.2, 5.4)
DPI = 150

#: A tighter crop than the vendored extract, for when the map is the page.
#: The shoreline file carries a margin of Johor and Riau so the island does
#: not read as floating, but at full-page size that margin is a third of the
#: paper spent on two coastlines nobody has a sensor on. This keeps enough of
#: them to place Singapore and gives the rest to the island.
ISLAND_VIEW = (103.590, 104.120, 1.195, 1.495)


def _style(ax, title: str, subtitle: str = "", *, scale: float = 1.0) -> None:
    # The title's pad has to clear the subtitle, which is drawn just above the
    # axes. With the default pad the two overlap and the figure is unreadable
    # exactly where it is meant to be most readable.
    ax.set_title(title, fontsize=theme.SIZE_HEADING * scale,
                 fontweight=theme.WEIGHT_BOLD, loc="left", color=theme.INK,
                 pad=28 * scale if subtitle else 12)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes,
                fontsize=theme.SIZE_SMALL * scale, color=theme.INK_MUTED,
                va="bottom")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme.AXIS)
    ax.tick_params(colors=theme.INK_MUTED, labelsize=theme.SIZE_SMALL * scale,
                   length=0)


def _site_groups(incidents: list) -> list[dict]:
    """
    Collapse incidents that share a position into one marker per place.

    This is the difference between a readable map and the one the client sent
    back. Coordinates come from an `RTUNumber -> LKey` join, so every sensor on
    one RTU carries *identical* lat/lon -- 198 incidents across ~40 RTUs draw as
    198 markers stacked in 40 piles, each hiding the ones beneath it. The
    stack looked like a busy map; it was one marker's worth of information
    repeated.

    So a place gets ONE marker: sized by how many incidents are there, coloured
    by the worst of them, labelled with the sites it covers. Nothing is hidden,
    and the count that was invisible before is now the thing the size encodes.
    """
    buckets: dict[tuple, dict] = {}
    for incident in incidents:
        lat, lon = incident.cluster.centroid_lat, incident.cluster.centroid_lon
        if lat is None or lon is None:
            continue
        # Round to ~60 m before grouping. Two RTUs on one site are sometimes
        # surveyed a few metres apart, which is far below what this map can
        # resolve and far above zero, so exact equality would leave them
        # overlapping instead of merged.
        key = (round(lat, 3), round(lon, 3))
        group = buckets.setdefault(key, {
            "lat": lat, "lon": lon, "incidents": [], "sites": set(),
            "sensors": 0, "region": incident.cluster.region,
        })
        group["incidents"].append(incident)
        group["sites"].update(incident.cluster.sites)
        group["sensors"] += len(incident.cluster.members)

    groups = []
    for group in buckets.values():
        worst = max(group["incidents"], key=lambda i: i.severity)
        group["worst"] = worst
        group["priority"] = worst.priority.value
        group["severity"] = worst.severity
        groups.append(group)
    return sorted(groups, key=lambda g: -g["severity"])


def _place_labels(ax, groups: list[dict], *, limit: int, fontsize: float,
                  bounds: tuple[float, float, float, float],
                  ) -> list[tuple[float, float, float, float]]:
    """
    Label the most severe places, moving each one off its neighbours.

    Labels are the part of a map that fails first as density rises. In the run
    the client sent back, "Bedok PS, Bedok Pond 2" and "Bedok PS, Bedok Pond 3"
    were printed on top of each other and neither could be read: a fixed offset
    above the marker cannot work when two markers are 300 m apart.

    So each label tries eight positions around its marker and takes the first
    that clears every label already placed AND stays inside the map. A label
    that cannot be placed is dropped rather than drawn over something -- an
    unreadable label is worse than none, because it also hides what is
    underneath it.

    The limit is low on purpose. Naming forty places on one island is not a map,
    it is a list drawn badly; the list belongs in a table, where it can be
    sorted and read. The map's job is to show WHERE the weight is.
    """
    lon0, lon1, lat0, lat1 = bounds
    span = lon1 - lon0
    step = span * 0.020
    offsets = [(0, 1.0), (0, -1.0), (1.15, 0), (-1.15, 0),
               (1.0, 0.85), (-1.0, 0.85), (1.0, -0.85), (-1.0, -0.85)]
    taken: list[tuple[float, float, float, float]] = []

    for group in groups[:limit]:
        text = ", ".join(sorted(group["sites"])[:2]) or str(group["region"] or "")
        if len(group["sites"]) > 2:
            text += f" +{len(group['sites']) - 2}"
        n = len(group["incidents"])
        if n > 1:
            text += f" ({n})"
        # Character width is about 0.55 em at this font; the figure is 7.2 in
        # wide for `span` degrees, so a point is span/(72*7.2) degrees.
        deg_per_pt = span / (72 * 7.2)
        half_w = len(text) * fontsize * 0.58 * deg_per_pt * 0.5 + span * 0.006
        half_h = fontsize * 1.5 * deg_per_pt * 0.5

        for dx, dy in offsets:
            cx = group["lon"] + dx * (step + half_w)
            cy = group["lat"] + dy * (step + half_h)
            box = (cx - half_w, cy - half_h, cx + half_w, cy + half_h)
            if box[0] < lon0 or box[2] > lon1 or box[1] < lat0 or box[3] > lat1:
                continue                      # would run off the map
            if any(not (box[2] < o[0] or box[0] > o[2]
                        or box[3] < o[1] or box[1] > o[3]) for o in taken):
                continue                      # would sit on another label
            taken.append(box)
            ax.annotate(
                text, xy=(group["lon"], group["lat"]), xytext=(cx, cy),
                ha="center", va="center", fontsize=fontsize, color=theme.INK,
                zorder=7, annotation_clip=False,
                bbox=dict(boxstyle="round,pad=0.22", fc="white",
                          ec=theme.HAIRLINE, alpha=.95),
                arrowprops=dict(arrowstyle="-", color=theme.INK_MUTED,
                                linewidth=0.7, shrinkA=0, shrinkB=5),
            )
            break
    return taken


def _region_names(ax, count: dict, boxes: list, scale: float,
                  bounds: tuple[float, float, float, float],
                  markers: list | None = None) -> None:
    """
    Write each region's name where nothing else is already written.

    Deliberately quiet: this layer says which fifth of the island you are
    looking at, not what to act on. Secondary ink, the smallest size in the
    report, under a white halo so it survives whatever shade it lands on.

    A region whose every candidate position is occupied gets no name. The
    choropleth still shows the five areas, and a name printed over a site label
    destroys two pieces of information to deliver one.
    """
    avoid = list(markers or [])
    for x0, y0, x1, y1 in boxes:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        avoid += [(x0, cy), (cx, cy), (x1, cy), (cx, y0), (cx, y1)]

    span = bounds[1] - bounds[0]
    fontsize = theme.SIZE_TINY * min(scale, 1.2)
    deg_per_pt = span / (72 * 7.2)
    taken = list(boxes)

    for region, spots in basemap.region_anchors(avoid, candidates=5).items():
        n = count.get(region, 0)
        label = str(region.value).upper() + (f"\n{n}" if n else "")
        half_w = len(str(region.value)) * fontsize * 0.58 * deg_per_pt * 0.5
        half_h = fontsize * (2.6 if n else 1.4) * deg_per_pt * 0.5
        for alon, alat in spots:
            if not (bounds[0] < alon < bounds[1]
                    and bounds[2] < alat < bounds[3]):
                continue                     # cropped out of view
            box = (alon - half_w, alat - half_h, alon + half_w, alat + half_h)
            if any(not (box[2] < o[0] or box[0] > o[2]
                        or box[3] < o[1] or box[1] > o[3]) for o in taken):
                continue
            taken.append(box)
            ax.text(alon, alat, label, ha="center", va="center",
                    fontsize=fontsize, color=theme.INK_SECONDARY,
                    fontweight=theme.WEIGHT_BOLD, linespacing=1.3, alpha=.85,
                    zorder=3,
                    path_effects=[pe.withStroke(linewidth=3.0,
                                                foreground="white")])
            break


def draw_map(ax, incidents: list, *, scale: float = 1.0,
             label_limit: int = 8, legend: bool = True,
             view: tuple[float, float, float, float] | None = None,
             ) -> list[dict]:
    """
    Draw the island, the region shading and one marker per place onto `ax`.

    Separated from `region_map_png` so the PDF can put the map beside a table
    of the same places -- the map answers "where is the weight", the table
    answers "which ones, exactly", and neither does the other's job well.
    Returns the place groups, so the caller builds that table from the same
    data the markers came from rather than recomputing it differently.
    """
    ax.set_aspect("equal", adjustable="box")
    bounds = view or SG_BOUNDS

    # Shade by severity rather than by count: five P4 telemetry faults in one
    # region must not outweigh a single P1 somewhere else, which is exactly
    # what a headcount would do.
    load: dict = {}
    count: dict = {}
    for incident in incidents:
        region = incident.cluster.region
        if region is not None:
            load[region] = load.get(region, 0.0) + incident.severity
            count[region] = count.get(region, 0) + 1
    basemap.shade_regions(ax, load)
    basemap.draw_island(ax)
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])

    groups = _site_groups(incidents)
    if not groups:
        ax.text(0.5, 0.06, "No placed incidents this run",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=theme.SIZE_BODY * scale, color=theme.INK_SECONDARY,
                zorder=6,
                bbox=dict(boxstyle="round,pad=0.4", fc="white",
                          ec=theme.HAIRLINE))
        _region_names(ax, count, [], scale, bounds)
        return groups

    for priority in reversed(theme.PRIORITY_ORDER):       # severe drawn last
        members = [g for g in groups if g["priority"] == priority]
        if not members:
            continue
        ax.scatter(
            [g["lon"] for g in members], [g["lat"] for g in members],
            s=[(34 + 46 * np.sqrt(len(g["incidents"]))) * scale for g in members],
            c=theme.PRIORITY_COLOR[priority], edgecolors="white",
            linewidths=1.2 * scale, alpha=.95, zorder=5,
            label=f"{priority}  {theme.PRIORITY_MEANING[priority]}"
                  f"   {sum(len(g['incidents']) for g in members)}",
        )

    # Site labels go down BEFORE the region names, and the region names then
    # dodge the boxes they actually occupy. The other order -- which is what
    # the first version did -- positioned "NORTH / 54 incidents" against the
    # marker coordinates and then dropped a site label on top of it, which is
    # exactly the collision this ordering exists to prevent.
    boxes = _place_labels(ax, groups, limit=label_limit, bounds=bounds,
                          fontsize=theme.SIZE_TINY * min(scale, 1.25))
    _region_names(ax, count, boxes, scale, bounds,
                  markers=[(g["lon"], g["lat"]) for g in groups])

    if legend:
        box = ax.legend(
            loc="lower left", frameon=True, facecolor="white",
            edgecolor=theme.HAIRLINE, framealpha=.94,
            fontsize=theme.SIZE_TINY * min(scale, 1.2),
            title="one marker per place, sized by incidents there",
            title_fontsize=theme.SIZE_TINY * min(scale, 1.1),
            handletextpad=.4, borderpad=.45, labelspacing=.35,
            borderaxespad=0.4)
        box.set_zorder(8)
        box.get_title().set_color(theme.INK_MUTED)
    return groups


def region_map_png(incidents: list[Incident], out_path: str | Path, *,
                   title: str = "Abnormal sensor clusters",
                   figsize: tuple[float, float] = PHONE_FIGSIZE,
                   dpi: int = DPI, label_limit: int = 8,
                   view: tuple[float, float, float, float] | None = None,
                   ) -> Path:
    """
    Where the incidents are, on the island.

    Four layers, in the order an operator reads them: the island, so a position
    means something at a glance; the region shading, so the part of Singapore
    carrying the run is visible before any marker is read; one marker per
    PLACE, not per incident; then labels on the places worth naming.

    Marker area scales with the square root of the incident count, so ten
    incidents at one site read as bigger than one without reading as ten times
    bigger.
    """
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    scale = max(1.0, figsize[0] / PHONE_FIGSIZE[0])
    groups = draw_map(ax, incidents, scale=scale,
                      label_limit=label_limit, view=view)

    placed = sum(len(g["incidents"]) for g in groups)
    _style(ax, title,
           f"{placed} incident(s) at {len(groups)} location(s) · "
           "shading is severity by region · positions are site-level (per RTU)",
           scale=min(scale, 1.3))
    # No lat/lon axes. On a recognisable outline they are decoration, and the
    # figure is read on a phone where every line of furniture costs map.
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor=theme.PAGE)
    plt.close(fig)
    return out_path


def region_matrix_png(matrix: dict[str, dict[str, int]], out_path: str | Path, *,
                      title: str = "Anomalies by region and by type") -> Path:
    """
    The client's literal request, as a heatmap.

    A sequential single-hue ramp, because the quantity is a count with a
    meaningful zero and no midpoint -- a diverging palette here would invent a
    centre that does not exist. Counts are printed in every cell, so the figure
    is exact rather than merely indicative.
    """
    out_path = Path(out_path)
    regions = sorted(matrix)
    equipment = sorted({e for row in matrix.values() for e in row})

    fig, ax = plt.subplots(figsize=(max(6.4, 1.05 * len(equipment) + 2.6),
                                    max(3.0, 0.62 * len(regions) + 2.0)), dpi=DPI)

    if not regions or not equipment:
        ax.text(0.5, 0.5, "No anomalies this run", transform=ax.transAxes,
                ha="center", va="center", fontsize=13, color="#667080")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return out_path

    grid = np.array([[matrix.get(r, {}).get(e, 0) for e in equipment]
                     for r in regions], dtype=float)

    im = ax.imshow(grid, cmap="Reds", aspect="auto",
                   vmin=0, vmax=max(1.0, grid.max()))

    ax.set_xticks(range(len(equipment)))
    ax.set_xticklabels(equipment, rotation=38, ha="right", fontsize=9)
    ax.set_yticks(range(len(regions)))
    ax.set_yticklabels(regions, fontsize=10)

    threshold = grid.max() * 0.55
    for r in range(len(regions)):
        for c in range(len(equipment)):
            value = int(grid[r, c])
            if value == 0:
                ax.text(c, r, "·", ha="center", va="center",
                        color="#b6bdc6", fontsize=11)
            else:
                ax.text(c, r, str(value), ha="center", va="center",
                        fontsize=10, fontweight="600",
                        color="white" if grid[r, c] > threshold else "#14181d")

    ax.set_xticks(np.arange(-.5, len(equipment), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(regions), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.tick_params(which="minor", length=0)
    for side in ax.spines.values():
        side.set_visible(False)

    ax.set_title(title, fontsize=13.5, fontweight="600", loc="left", pad=30)
    ax.text(0, 1.015, "a column lit across regions is a fleet-wide equipment "
                      "problem; a row lit across types is that area",
            transform=ax.transAxes, fontsize=8.8, color="#667080", va="bottom")
    fig.colorbar(im, ax=ax, shrink=.75, pad=.02).set_label(
        "sensors abnormal", fontsize=9, color="#667080")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def sensor_detail_png(ts, values, anomaly, out_path: str | Path) -> Path:
    """
    One sensor's series with the flagged window shaded.

    The single most useful image in an alert: it lets an operator confirm or
    reject the machine's verdict in about two seconds, which is the whole
    reason the old system attached a plot at all.
    """
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=PHONE_FIGSIZE, dpi=DPI)

    ax.plot(ts, values, linewidth=1.1, color="#2166ac", zorder=2)
    ax.axvspan(anomaly.start, anomaly.end, color="#b2182b", alpha=.14, zorder=1)
    ax.axvline(anomaly.start, color="#b2182b", linewidth=1.1, alpha=.7)
    ax.axvline(anomaly.end, color="#b2182b", linewidth=1.1, alpha=.7)

    unit = anomaly.severity.unit or anomaly.sensor.unit or ""
    _style(ax,
           anomaly.sensor.description or anomaly.sensor.sensor_key,
           f"{anomaly.dominant_type.value} · "
           f"{anomaly.severity.duration_s/60:.0f} min · "
           f"{anomaly.sensor.region or 'unplaced'}")
    ax.set_ylabel(unit, fontsize=9.5, color="#667080")
    ax.grid(True, color="#eef1f4", linewidth=.8)
    ax.set_axisbelow(True)
    fig.autofmt_xdate()

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def run_charts(result, out_dir: str | Path, *,
               stamped: bool = True) -> dict[str, Path]:
    """
    Every PNG for one run. Returns {name: path}.

    `stamped` puts the run id in each filename, which is what a flat output
    directory needs. A caller writing into a per-run directory passes False:
    the directory already carries the run id, and repeating it in every file
    inside it reads as a mistake.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{result.run_id}" if stamped else ""
    return {
        "map": region_map_png(result.incidents,
                              out_dir / f"map{suffix}.png"),
        "matrix": region_matrix_png(result.region_matrix,
                                    out_dir / f"matrix{suffix}.png"),
    }
