#!/usr/bin/env python3
"""
Carve Singapore out of the GSHHG shoreline and vendor it into the package.

Run once. Its output, `das2/spatial/singapore_coast.json`, is committed, so
nothing at runtime needs this file, the source data, or a network. It lives
here so the map's geometry has a provenance rather than being a blob someone
found: every coordinate the system draws came out of the steps below.

    pip download basemap-data-hires --no-deps -d /tmp/bm
    cd /tmp/bm && unzip -q basemap_data_hires-*.whl
    python3 tools/extract_coastline.py \
        /tmp/bm/mpl_toolkits/basemap_data \
        das2/spatial/singapore_coast.json

Why this source: GSHHG (Wessel & Smith, LGPL) is the only global shoreline
that ships on PyPI as data rather than being fetched at import time, so it can
be pulled from a machine that cannot reach a mapping provider. Its full
resolution renders the main island in 715 points -- the intermediate one gives
27, which is a blob, not an island.

Its limitation, which the JSON repeats and `basemap.py` repeats again: GSHHG
predates the most recent reclamation. Tuas, parts of Jurong Island and Marina
South are smaller than they are today. That is fine for reading where a
cluster is and wrong for anything that needs a real boundary.

An SLA-style source (URA planning-area polygons from data.gov.sg) would give
true region boundaries instead of the nearest-centroid partition drawn now.
It is a drop-in replacement for the `main`/`islands` rings plus a per-region
polygon list; nothing else changes.
"""
import json
import sys

import numpy as np

BOX = (103.55, 104.16, 1.13, 1.50)      # lon0, lon1, lat0, lat1
MIN_ISLAND_KM2 = 0.15
base = sys.argv[1]

def clip(poly, box):
    """Sutherland-Hodgman against an axis-aligned rectangle."""
    lon0, lon1, lat0, lat1 = box
    def inside(p, edge):
        return {0: p[0] >= lon0, 1: p[0] <= lon1,
                2: p[1] >= lat0, 3: p[1] <= lat1}[edge]
    def cut(a, b, edge):
        if edge in (0, 1):
            x = lon0 if edge == 0 else lon1
            t = (x - a[0]) / (b[0] - a[0])
            return (x, a[1] + t * (b[1] - a[1]))
        y = lat0 if edge == 2 else lat1
        t = (y - a[1]) / (b[1] - a[1])
        return (a[0] + t * (b[0] - a[0]), y)
    out = [tuple(p) for p in poly]
    for edge in range(4):
        if not out:
            return []
        buf, prev = [], out[-1]
        for cur in out:
            if inside(cur, edge):
                if not inside(prev, edge):
                    buf.append(cut(prev, cur, edge))
                buf.append(cur)
            elif inside(prev, edge):
                buf.append(cut(prev, cur, edge))
            prev = cur
        out = buf
    return out

meta = f"{base}/gshhsmeta_f.dat"
data = open(f"{base}/gshhs_f.dat", "rb")
main, islands, neighbours = None, [], []

for line in open(meta):
    f = line.split()
    lvl, area, npts = int(f[0]), float(f[1]), int(f[2])
    south, north, off, nb = float(f[3]), float(f[4]), int(f[5]), int(f[6])
    if lvl != 1 or south > BOX[3] or north < BOX[2]:
        continue
    data.seek(off)
    pts = np.frombuffer(data.read(nb), dtype="<f4").reshape(-1, 2).astype(float)
    if (pts[:, 0].max() < BOX[0] or pts[:, 0].min() > BOX[1]
            or pts[:, 1].max() < BOX[2] or pts[:, 1].min() > BOX[3]):
        continue
    ring = [(round(x, 4), round(y, 4)) for x, y in clip(pts, BOX)]
    if len(ring) < 4:
        continue
    if abs(area - 547.83) < 0.01:
        main = ring
    elif area > 5000:                    # Asia mainland (Johor) / Sumatra
        neighbours.append(ring)
    elif BOX[0] <= pts[:, 0].min() and pts[:, 0].max() <= BOX[1] \
            and pts[:, 1].min() >= 1.20:
        if area >= MIN_ISLAND_KM2:
            islands.append(ring)
    elif area >= MIN_ISLAND_KM2:
        neighbours.append(ring)

out = {
    "source": "GSHHG full resolution (Wessel & Smith), shipped in the PyPI "
              "package basemap-data-hires 2.0.0 as gshhs_f.dat; LGPL.",
    "note": "Clipped to the box below and rounded to 4 dp (~11 m). GSHHG "
            "predates the most recent reclamation, so Tuas and parts of "
            "Jurong Island and Marina South read smaller than today.",
    "box": BOX,
    "main": main,
    "islands": islands,
    "neighbours": neighbours,
}
print("main:", len(main), "pts;", len(islands), "islands;",
      len(neighbours), "neighbour masses")
json.dump(out, open(sys.argv[2], "w"), separators=(",", ":"))
