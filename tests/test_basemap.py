#!/usr/bin/env python3
"""
The map is a map.

Two tile providers were tried against the client's deployment and both refused
an unattended server: CARTO answered "API key required", OpenStreetMap
answered 403 Access blocked. So the island is drawn from vector data vendored
in the package, and these assertions are what stop that data or the code that
reads it from quietly degrading back into a scatter plot on a grey rectangle.

The geometric checks are deliberately about *recognisability*: the right land
area in the right place, the offshore islands present, the regions arranged
the way Singapore's regions actually are. A map that is subtly wrong is worse
than no map, because an operator will believe it.

Run:  python3 tests/test_basemap.py
"""

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from das2.report import basemap  # noqa: E402
from das2.report.charts import region_map_png  # noqa: E402
from das2.spatial.regions import Region  # noqa: E402

passed = failed = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {label}" + (f"  ({detail})" if detail else ""))
    else:
        failed += 1
        print(f"  FAIL  {label}" + (f"  ({detail})" if detail else ""))


def incident(lat, lon, region, priority, severity, members, site):
    return NS(cluster=NS(centroid_lat=lat, centroid_lon=lon, region=region,
                         members=list(range(members)), sites={site}),
              priority=NS(value=priority), severity=severity)


def main() -> int:
    print("\nthe shoreline ships with the code")
    check("the coastline file is in the package",
          basemap.COAST_FILE.exists(),
          str(basemap.COAST_FILE.relative_to(Path(__file__).parent.parent)))
    coast = basemap.coastline()
    check("it names where it came from",
          "GSHHG" in coast.get("source", ""), coast.get("source", "")[:48])
    check("and admits what it is out of date about",
          "reclamation" in coast.get("note", "").lower(),
          "Tuas and Jurong Island read small")
    check("it is small enough to live in git",
          basemap.COAST_FILE.stat().st_size < 150_000,
          f"{basemap.COAST_FILE.stat().st_size // 1024} KB")

    print("\nthe main island is the main island")
    main_ring = np.asarray(coast["main"])
    check("it has enough points to have a shape", len(main_ring) > 400,
          f"{len(main_ring)} points -- the 27 of GSHHG intermediate is a blob")
    lon0, lon1 = main_ring[:, 0].min(), main_ring[:, 0].max()
    lat0, lat1 = main_ring[:, 1].min(), main_ring[:, 1].max()
    check("it spans the right longitudes", 103.60 < lon0 and lon1 < 104.05,
          f"{lon0:.3f}-{lon1:.3f}, against Tuas 103.61 and Changi 104.01")
    check("it spans the right latitudes", 1.20 < lat0 and lat1 < 1.49,
          f"{lat0:.3f}-{lat1:.3f}, against Woodlands 1.47")
    # Roughly 41 km east-west against 23 km north-south: a wide, flat island.
    check("it is wider than it is tall, as Singapore is",
          1.5 < (lon1 - lon0) / (lat1 - lat0) < 2.3,
          f"aspect {(lon1 - lon0) / (lat1 - lat0):.2f}")

    print("\nthe islands an operator would look for are there")
    boxes = [(np.asarray(r)[:, 0].min(), np.asarray(r)[:, 0].max(),
              np.asarray(r)[:, 1].min(), np.asarray(r)[:, 1].max())
             for r in coast["islands"]]

    def has(name, lon, lat):
        check(f"{name} is drawn",
              any(a <= lon <= b and c <= lat <= d for a, b, c, d in boxes),
              f"{lon}, {lat}")

    has("Pulau Tekong", 104.055, 1.415)
    has("Pulau Ubin", 103.960, 1.412)
    has("Sentosa", 103.830, 1.252)
    check("and the offshore islets are not swamping the file",
          len(coast["islands"]) < 60, f"{len(coast['islands'])} island(s)")
    check("neighbouring land is drawn for context",
          len(coast["neighbours"]) > 0,
          "an island with nothing around it reads as floating")

    print("\nregions are laid out the way Singapore's regions are")
    anchors = basemap.region_anchors()
    for region in (Region.CENTRAL, Region.EAST, Region.NORTH,
                   Region.NORTH_EAST, Region.WEST):
        check(f"{region.value} has a place to put its name", region in anchors)
    west, east = anchors[Region.WEST], anchors[Region.EAST]
    north, central = anchors[Region.NORTH], anchors[Region.CENTRAL]
    check("West is west of East", west[0] < east[0],
          f"{west[0]:.3f} < {east[0]:.3f}")
    check("North is north of Central", north[1] > central[1],
          f"{north[1]:.3f} > {central[1]:.3f}")
    check("North-East sits between them",
          north[0] < anchors[Region.NORTH_EAST][0] < east[0] + .02)

    print("\nevery label lands on land, not in the strait")
    # The obvious choice -- the mean of a region's cells -- puts North-East's
    # label in the Johor Strait, because the region wraps around Seletar.
    owner, regions = basemap._region_grid()
    lon_axis = np.linspace(*basemap.bounds()[:2], owner.shape[1])
    lat_axis = np.linspace(*basemap.bounds()[2:], owner.shape[0])
    for region, (alon, alat) in anchors.items():
        cell = owner[int(np.abs(lat_axis - alat).argmin()),
                     int(np.abs(lon_axis - alon).argmin())]
        check(f"{region.value}'s label is on land, in its own region",
              cell == regions.index(region),
              f"cell {cell}, expected {regions.index(region)}")

    print("\nlabels dodge the markers they would otherwise sit under")
    busy = [(103.7056, 1.3404), (103.9302, 1.3345)]
    moved = basemap.region_anchors(busy)
    for lon, lat in busy:
        nearest = min(((a[0] - lon) ** 2 + (a[1] - lat) ** 2) ** .5
                      for a in moved.values())
        check(f"no region name within 2 km of the marker at {lon}, {lat}",
              nearest > 0.018, f"{nearest * 111:.1f} km")

    print("\nthe PNG renders, with incidents and without")
    incidents = [
        incident(1.3404, 103.7056, Region.WEST, "P1", 88, 27, "Pandan1PS"),
        incident(1.4382, 103.7622, Region.NORTH, "P2", 61, 6, "Kranji1PS"),
        incident(1.3345, 103.9302, Region.EAST, "P2", 55, 4, "BedokPond4"),
        incident(1.3110, 103.8720, Region.CENTRAL, "P3", 33, 2, "Bidadari"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        busy_png = region_map_png(incidents, Path(tmp) / "map.png")
        check("a run with incidents renders", busy_png.exists(),
              f"{busy_png.stat().st_size // 1024} KB")
        check("and is under Telegram's 10 MB photo limit",
              busy_png.stat().st_size < 10 * 1024 * 1024)

        quiet_png = region_map_png([], Path(tmp) / "quiet.png")
        check("a quiet run still renders the island", quiet_png.exists(),
              "silence must look deliberate, not like a crashed job")
        # The quiet map has no markers and no choropleth, so it compresses far
        # smaller. If the two are the same size, something is not being drawn.
        check("the two are visibly different images",
              quiet_png.stat().st_size != busy_png.stat().st_size)

        # An incident with no coordinates must not take the map down with it:
        # RTUNumber is dirty in this feed and the LongLat join misses for some.
        unplaced = [incident(None, None, Region.WEST, "P2", 40, 3, "Nowhere")]
        no_coords = region_map_png(unplaced, Path(tmp) / "unplaced.png")
        check("an incident with no coordinates does not break the map",
              no_coords.exists(), "the RTU->LongLat join misses for some")

    print(f"\n{passed} passed, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
