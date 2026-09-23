"""
das2.report.basemap
===================

Singapore, drawn from a coastline shipped inside this package.

Why not tiles
-------------
The obvious way to get a recognisable map is to fetch raster tiles from
OpenStreetMap or a commercial provider. Both were tried against the client's
deployment and both failed, in different ways:

  * CARTO answered every tile with **"API key required"**;
  * OpenStreetMap's own tiles answered **403 Access blocked** -- their tile
    usage policy does not allow an unattended server pulling tiles on a
    schedule, which is exactly what this is.

Neither is a bug to fix. A tile fetch is a network dependency in the alert
path: it needs internet from a container on a utility's network, it needs a
provider who permits automated use, and when it fails it fails at 3 a.m. in
the one image the duty operator was going to look at. So the island is drawn
from vector data that ships with the code. It renders identically on a machine
with no internet at all, which is the deployment this is heading for.

What is drawn
-------------
`COAST_FILE` holds the GSHHG full-resolution shoreline clipped to Singapore:
the main island, 27 offshore islands (Ubin, Tekong, Sentosa, the Southern
Islands, the Jurong group) and the neighbouring Johor and Riau coastlines for
context. See `tools/extract_coastline.py` for how it was produced and
`singapore_coast.json`'s own `source` field for provenance.

**It is a shoreline, not a cadastral map**, and it predates the most recent
reclamation -- Tuas, parts of Jurong Island and Marina South read smaller than
they are today. It is accurate enough to recognise where a cluster is, which
is the whole job; it is not a survey.

Regions
-------
The region shading is not a separate dataset. It is the *same* nearest-
centroid rule `das2.spatial.regions` uses to decide which region a sensor
belongs to, evaluated over a grid and clipped to land. So the coloured areas
on the map are literally the decision boundaries the system used -- if a site
sits near an edge, the map shows you that it does, rather than implying a
precision the assignment does not have.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from das2.spatial.regions import PLANNING_AREAS, Region

COAST_FILE = Path(__file__).resolve().parent.parent / "spatial" / "singapore_coast.json"

#: Grid for the region shading. ~55 m a cell at this box, which is finer than
#: the figure's own pixels at phone size, so the boundaries look like lines
#: rather than stairs.
GRID_W, GRID_H = 1200, 728

#: Drawing order matters here: regions are painted, then boundaries drawn over
#: them, then markers over that. These sit below all of it.
SEA = "#dce9f2"
NEIGHBOUR_LAND = "#dfe2e5"
NEIGHBOUR_EDGE = "#c8cdd2"
COAST_EDGE = "#5f6b76"


@lru_cache(maxsize=1)
def coastline() -> dict:
    """The vendored shoreline, as {main, islands, neighbours, box}."""
    return json.loads(COAST_FILE.read_text())


def bounds() -> tuple[float, float, float, float]:
    """lon_min, lon_max, lat_min, lat_max of the vendored extract."""
    return tuple(coastline()["box"])                       # type: ignore[return-value]


@lru_cache(maxsize=1)
def _region_grid() -> tuple[np.ndarray, list[Region]]:
    """
    Region index per grid cell, `-1` for sea.

    Cached because it depends on nothing that changes between runs, and it is
    the only part of drawing the map that costs anything.
    """
    from matplotlib.path import Path as MplPath

    lon0, lon1, lat0, lat1 = bounds()
    lons = np.linspace(lon0, lon1, GRID_W)
    lats = np.linspace(lat0, lat1, GRID_H)
    mesh_lon, mesh_lat = np.meshgrid(lons, lats)
    points = np.column_stack([mesh_lon.ravel(), mesh_lat.ravel()])

    # Land first: everything outside it is never coloured.
    coast = coastline()
    land = np.zeros(points.shape[0], dtype=bool)
    for ring in [coast["main"], *coast["islands"]]:
        land |= MplPath(np.asarray(ring)).contains_points(points)

    # Nearest planning-area centroid, one centroid at a time. The obvious
    # (cells x centroids) distance matrix is 874k x 55 and wants ~400 MB; a
    # running minimum wants two arrays and is no slower.
    #
    # Degrees are used directly rather than metres: at 1.35 N a degree of
    # longitude is 111.3 km against 110.6 km for latitude, a 0.6% difference
    # that cannot move a cell across a boundary tens of kilometres wide.
    regions = list(dict.fromkeys(a.region for a in PLANNING_AREAS))
    best = np.full(points.shape[0], np.inf)
    owner = np.zeros(points.shape[0], dtype=np.int16)
    for area in PLANNING_AREAS:
        d = (points[:, 0] - area.lon) ** 2 + (points[:, 1] - area.lat) ** 2
        closer = d < best
        best = np.where(closer, d, best)
        owner = np.where(closer, regions.index(area.region), owner)

    owner = np.where(land, owner, -1).reshape(GRID_H, GRID_W)
    return owner, regions


def region_extent() -> tuple[float, float, float, float]:
    """The imshow extent matching `_region_grid`."""
    lon0, lon1, lat0, lat1 = bounds()
    return (lon0, lon1, lat0, lat1)


def draw_island(ax, *, sea: bool = True) -> None:
    """Sea, neighbouring land, then Singapore's own shoreline."""
    coast = coastline()
    lon0, lon1, lat0, lat1 = bounds()
    if sea:
        ax.set_facecolor(SEA)
    for ring in coast["neighbours"]:
        xs, ys = zip(*ring)
        ax.fill(xs, ys, color=NEIGHBOUR_LAND, ec=NEIGHBOUR_EDGE,
                linewidth=.6, zorder=1)
    for ring in [coast["main"], *coast["islands"]]:
        xs, ys = zip(*ring)
        ax.fill(xs, ys, color="none", ec=COAST_EDGE, linewidth=.9, zorder=4)
    ax.set_xlim(lon0, lon1)
    ax.set_ylim(lat0, lat1)


def region_anchors(avoid: list[tuple[float, float]] | None = None,
                   ) -> dict[Region, tuple[float, float]]:
    """
    Somewhere inside each region to write its name.

    Not the centroid. Singapore's regions are concave enough that the mean of
    their cells lands in the sea -- North-East's falls in the Johor Strait.

    Not simply the point furthest from a marker either: that hands the label
    to whichever offshore islet happens to belong to the region, and "WEST"
    written on an islet in the Singapore Strait identifies nothing.

    So: erode each region until it nearly vanishes, which leaves the deep
    interior of its largest body, and pick the point in that interior that is
    furthest from any marker. Deep enough to read as belonging to the region,
    clear enough not to sit under an incident.
    """
    owner, regions = _region_grid()
    lon0, lon1, lat0, lat1 = bounds()
    step = 6                                   # ~330 m; finer buys nothing here
    grid = owner[::step, ::step]
    lons = np.linspace(lon0, lon1, owner.shape[1])[::step]
    lats = np.linspace(lat0, lat1, owner.shape[0])[::step]
    mesh_lon, mesh_lat = np.meshgrid(lons, lats)

    if avoid:
        clearance = np.full(grid.shape, np.inf)
        for alon, alat in avoid:
            np.minimum(clearance,
                       (mesh_lon - alon) ** 2 + (mesh_lat - alat) ** 2,
                       out=clearance)
    else:
        clearance = np.zeros(grid.shape)

    anchors: dict[Region, tuple[float, float]] = {}
    for idx, region in enumerate(regions):
        mask = grid == idx
        if not mask.any():
            continue
        # Depth by erosion, in pure numpy -- scipy is not a dependency of this
        # project and one label placement is not the reason to make it one.
        depth = mask.astype(np.int16)
        layer = mask
        while True:
            shrunk = (layer
                      & np.roll(layer, 1, 0) & np.roll(layer, -1, 0)
                      & np.roll(layer, 1, 1) & np.roll(layer, -1, 1))
            shrunk[0], shrunk[-1], shrunk[:, 0], shrunk[:, -1] = (
                False, False, False, False)
            if not shrunk.any():
                break
            depth += shrunk
            layer = shrunk
        interior = mask & (depth >= max(1, int(depth.max() * 0.45)))
        scored = np.where(interior, clearance, -np.inf)
        y, x = np.unravel_index(int(np.argmax(scored)), scored.shape)
        anchors[region] = (float(mesh_lon[y, x]), float(mesh_lat[y, x]))
    return anchors


def shade_regions(ax, weight: dict[Region, float], *,
                  ramp: tuple[tuple[float, float, float], ...] = (
                      (0.949, 0.961, 0.969),      # #f2f5f7  nothing here
                      (0.839, 0.882, 0.910),      # #d6e1e8
                      (0.702, 0.792, 0.847),      # #b3cad8
                      (0.541, 0.682, 0.769),      # #8aaec4
                      (0.396, 0.573, 0.694),      # #6592b1
                  )) -> None:
    """
    Paint each region by `weight`.

    A choropleth rather than five fixed colours: the client asked to *see*
    which part of the island is in trouble, and a fixed palette says only
    where the regions are. The ramp is deliberately desaturated blue-grey so
    that the priority colours on the markers -- which are the urgent thing --
    are the only saturated ink on the figure.
    """
    owner, regions = _region_grid()
    h, w = owner.shape
    rgba = np.zeros((h, w, 4), dtype=float)

    top = max(weight.values(), default=0.0)
    for idx, region in enumerate(regions):
        mask = owner == idx
        if not mask.any():
            continue
        share = 0.0 if top <= 0 else max(0.0, weight.get(region, 0.0)) / top
        step = ramp[min(len(ramp) - 1, int(round(share * (len(ramp) - 1))))]
        rgba[mask] = (*step, 1.0)

    # Region boundaries: the cells whose neighbour belongs to someone else.
    # Drawing them into the same image keeps them exactly on the colour edge,
    # which a separate contour pass does not manage.
    edge = np.zeros_like(owner, dtype=bool)
    edge[:, :-1] |= (owner[:, :-1] != owner[:, 1:]) & (owner[:, :-1] >= 0) & (owner[:, 1:] >= 0)
    edge[:-1, :] |= (owner[:-1, :] != owner[1:, :]) & (owner[:-1, :] >= 0) & (owner[1:, :] >= 0)
    rgba[edge] = (1.0, 1.0, 1.0, .95)

    ax.imshow(rgba, extent=region_extent(), origin="lower",
              interpolation="nearest", zorder=2)
