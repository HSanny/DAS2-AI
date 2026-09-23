"""
das2.report.charts
==================

Static PNGs for Telegram, which cannot render HTML inline.

The dashboard is the full picture; these are the glance. An operator reading a
phone at 3 a.m. gets one image, so each chart answers exactly one question and
is legible at phone width:

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
from das2.report import basemap             # noqa: E402

PRIORITY_COLOR = {"P1": "#b2182b", "P2": "#ef8a62",
                  "P3": "#e8c56a", "P4": "#5c7183"}

#: Singapore's bounding box, so a single incident does not render on a map
#: zoomed so far in that the location is meaningless. Taken from the vendored
#: shoreline so the outline and the axes cannot disagree.
SG_BOUNDS = basemap.bounds()   # lon_min, lon_max, lat_min, lat_max

PHONE_FIGSIZE = (7.2, 5.4)
DPI = 150


def _style(ax, title: str, subtitle: str = "") -> None:
    # The title's pad has to clear the subtitle, which is drawn just above the
    # axes. With the default pad the two overlap and the figure is unreadable
    # exactly where it is meant to be most readable.
    ax.set_title(title, fontsize=14, fontweight="600", loc="left",
                 pad=28 if subtitle else 12)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=9.5,
                color="#667080", va="bottom")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9d0d8")
    ax.tick_params(colors="#667080", labelsize=9)


def region_map_png(incidents: list[Incident], out_path: str | Path, *,
                   title: str = "Abnormal sensor clusters") -> Path:
    """
    Where the incidents are, on the island.

    Three layers, in the order an operator reads them: the island itself, so
    the location means something at a glance; the region shading, so the part
    of Singapore carrying the run's load is visible before any marker is read;
    then the incidents themselves.

    Marker area scales with member count rather than radius, so a ten-sensor
    incident reads as bigger than a one-sensor incident without reading as a
    hundred times bigger.
    """
    out_path = Path(out_path)
    placed = [i for i in incidents
              if i.cluster.centroid_lat is not None
              and i.cluster.centroid_lon is not None]

    fig, ax = plt.subplots(figsize=PHONE_FIGSIZE, dpi=DPI)
    # Latitude/longitude are not interchangeable units; at 1.35 degrees north
    # one degree of longitude is very nearly one degree of latitude in metres,
    # so an equal aspect is honest here and shapes are not distorted.
    ax.set_aspect("equal", adjustable="box")

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

    # Region names have to dodge the site labels, not just the dots. A site
    # label is a white box roughly 0.045 deg wide sitting above its marker, so
    # avoiding the marker's own coordinate alone still lets "WEST" land under
    # "Pandan1PS". Spanning the box's footprint costs nothing and settles it.
    avoid = []
    for i in placed:
        lon, lat = i.cluster.centroid_lon, i.cluster.centroid_lat
        avoid.append((lon, lat))
        avoid += [(lon + dx, lat + 0.006) for dx in (-.022, -.011, 0, .011, .022)]
    anchors = basemap.region_anchors(avoid)
    for region, (alon, alat) in anchors.items():
        n = count.get(region, 0)
        label = str(region.value).upper()
        if n:
            label += f"\n{n} incident" + ("s" if n > 1 else "")
        ax.text(alon, alat, label, ha="center", va="center",
                fontsize=8, color="#44525f", fontweight="600",
                linespacing=1.4, alpha=.9, zorder=3,
                path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])

    if not placed:
        ax.text(0.5, 0.06, "No placed incidents this run",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color="#44525f", zorder=6,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#dfe3e8"))
    else:
        for priority in ("P4", "P3", "P2", "P1"):      # severe drawn last, on top
            group = [i for i in placed if i.priority.value == priority]
            if not group:
                continue
            ax.scatter(
                [i.cluster.centroid_lon for i in group],
                [i.cluster.centroid_lat for i in group],
                s=[70 + 55 * np.sqrt(len(i.cluster.members)) for i in group],
                c=PRIORITY_COLOR[priority], edgecolors="white", linewidths=1.4,
                alpha=.95, zorder=5, label=f"{priority} ({len(group)})",
            )
        # Label only the ones worth driving to; labelling everything would
        # produce an unreadable pile at phone size.
        for i in sorted(placed, key=lambda x: -x.severity)[:6]:
            label = ", ".join(sorted(i.cluster.sites)[:2]) or str(i.cluster.region or "")
            ax.annotate(label,
                        (i.cluster.centroid_lon, i.cluster.centroid_lat),
                        textcoords="offset points", xytext=(0, 13),
                        ha="center", fontsize=8.5, color="#14181d", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.22", fc="white",
                                  ec="#dfe3e8", alpha=.9))
        ax.legend(loc="lower left", frameon=True, facecolor="white",
                  edgecolor="#dfe3e8", framealpha=.9, fontsize=9,
                  handletextpad=.3, borderpad=.4).set_zorder(6)

    _style(ax, title,
           f"{len(placed)} placed incident(s) · shading is severity by region · "
           "positions are site-level (per RTU)")
    # No lat/lon axes. On a recognisable outline they are decoration, and the
    # figure is read on a phone where every line of furniture costs map.
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
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
