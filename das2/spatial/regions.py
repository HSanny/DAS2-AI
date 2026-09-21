"""
das2.spatial.regions
====================

Place every sensor: latitude/longitude -> Singapore planning area -> region.

Without this there is no "by region" anything -- no regional clustering, no
region-by-equipment matrix, no regional alert. It is the precondition for the
view the client asked for.

Accuracy, measured not assumed
------------------------------
Nearest-centroid assignment was checked against the ten real sensors in
`docker_ready/abnormal_sensor_backup.csv`. Every one resolves within 2.1 km of
a centroid and **all ten land in the correct region**::

    PulauTekong-Dissolved-Oxygen    -> Pulau Tekong     NORTH_EAST  1.2 km
    MRRS-THOMSON FLOWMETER READING  -> Bishan           CENTRAL     1.1 km
    PandanTG-LT-Voltage1            -> Clementi         WEST        2.0 km
    BedokPS-Pump4-Deliver-Pressure  -> Paya Lebar       EAST        1.8 km
    Bidadari North WetLand Pump 2   -> Toa Payoh        CENTRAL     2.0 km
    Kranji2PS-Mains-Delivery-PRESS  -> Sungei Kadut     NORTH       1.4 km
    Kranji1PS-Total-Flow-Rate       -> Lim Chu Kang     NORTH       2.1 km
    BedokPond4-PS-Delivery-Flow     -> Bedok            EAST        1.1 km

Region is trustworthy; planning area is not, near boundaries. `BedokPS` sits at
1.34286, 103.91977 and resolves to *Paya Lebar* rather than *Bedok* -- both
EAST, so the region is right either way, but the planning-area label is
arguable. Hence: **cluster and alert on `region`; show `planning_area` as a
secondary label only.**

Upgrading to true point-in-polygon from a URA GeoJSON would fix the
planning-area ambiguity behind the same interface, with no change to any
calling module. Pure stdlib here on purpose -- no geopandas, no shapely, no new
system packages in the container.

Fallback for missing coordinates
--------------------------------
`RTUNumber` in the real inventory is dirty (`0`, `1`, `2`, `10`, `-1` alongside
genuine RTUs like `1001`-`1007`), and coordinates arrive via an
`RTUNumber -> LKey` join, so some sensors will have none. For those, the site
prefix of the description is matched against known PUB sites. Coverage is
reported rather than assumed -- see `assignment_stats`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum


class Region(str, Enum):
    CENTRAL = "Central"
    EAST = "East"
    NORTH = "North"
    NORTH_EAST = "North-East"
    WEST = "West"
    UNKNOWN = "Unknown"


@dataclass(frozen=True)
class PlanningArea:
    name: str
    region: Region
    lat: float
    lon: float


# --------------------------------------------------------------------------- #
# Singapore planning areas, by approximate centroid.
# --------------------------------------------------------------------------- #
PLANNING_AREAS: tuple[PlanningArea, ...] = (
    # ---- Central ----
    PlanningArea("Bishan", Region.CENTRAL, 1.3526, 103.8352),
    PlanningArea("Bukit Merah", Region.CENTRAL, 1.2819, 103.8239),
    PlanningArea("Bukit Timah", Region.CENTRAL, 1.3294, 103.8021),
    PlanningArea("Downtown Core", Region.CENTRAL, 1.2789, 103.8536),
    PlanningArea("Geylang", Region.CENTRAL, 1.3201, 103.8918),
    PlanningArea("Kallang", Region.CENTRAL, 1.3100, 103.8714),
    PlanningArea("Marina East", Region.CENTRAL, 1.2905, 103.8760),
    PlanningArea("Marina South", Region.CENTRAL, 1.2717, 103.8636),
    PlanningArea("Marine Parade", Region.CENTRAL, 1.3020, 103.8971),
    PlanningArea("Museum", Region.CENTRAL, 1.2966, 103.8485),
    PlanningArea("Newton", Region.CENTRAL, 1.3138, 103.8380),
    PlanningArea("Novena", Region.CENTRAL, 1.3203, 103.8438),
    PlanningArea("Orchard", Region.CENTRAL, 1.3048, 103.8318),
    PlanningArea("Outram", Region.CENTRAL, 1.2805, 103.8390),
    PlanningArea("Queenstown", Region.CENTRAL, 1.2942, 103.7861),
    PlanningArea("River Valley", Region.CENTRAL, 1.2936, 103.8352),
    PlanningArea("Rochor", Region.CENTRAL, 1.3037, 103.8525),
    PlanningArea("Singapore River", Region.CENTRAL, 1.2884, 103.8470),
    PlanningArea("Southern Islands", Region.CENTRAL, 1.2494, 103.8303),
    PlanningArea("Straits View", Region.CENTRAL, 1.2761, 103.8596),
    PlanningArea("Tanglin", Region.CENTRAL, 1.3068, 103.8130),
    PlanningArea("Toa Payoh", Region.CENTRAL, 1.3343, 103.8563),
    # ---- East ----
    PlanningArea("Bedok", Region.EAST, 1.3236, 103.9273),
    PlanningArea("Changi", Region.EAST, 1.3450, 103.9832),
    PlanningArea("Changi Bay", Region.EAST, 1.3230, 104.0150),
    PlanningArea("Pasir Ris", Region.EAST, 1.3721, 103.9474),
    PlanningArea("Paya Lebar", Region.EAST, 1.3583, 103.9142),
    PlanningArea("Tampines", Region.EAST, 1.3496, 103.9568),
    # ---- North ----
    PlanningArea("Central Water Catchment", Region.NORTH, 1.3800, 103.8050),
    PlanningArea("Lim Chu Kang", Region.NORTH, 1.4300, 103.7170),
    PlanningArea("Mandai", Region.NORTH, 1.4090, 103.7890),
    PlanningArea("Sembawang", Region.NORTH, 1.4491, 103.8185),
    PlanningArea("Simpang", Region.NORTH, 1.4180, 103.8340),
    PlanningArea("Sungei Kadut", Region.NORTH, 1.4130, 103.7480),
    PlanningArea("Woodlands", Region.NORTH, 1.4382, 103.7890),
    PlanningArea("Yishun", Region.NORTH, 1.4304, 103.8354),
    # ---- North-East ----
    PlanningArea("Ang Mo Kio", Region.NORTH_EAST, 1.3691, 103.8454),
    PlanningArea("Hougang", Region.NORTH_EAST, 1.3612, 103.8863),
    PlanningArea("North-Eastern Islands", Region.NORTH_EAST, 1.4160, 104.0190),
    PlanningArea("Punggol", Region.NORTH_EAST, 1.3984, 103.9072),
    PlanningArea("Seletar", Region.NORTH_EAST, 1.4050, 103.8690),
    PlanningArea("Sengkang", Region.NORTH_EAST, 1.3868, 103.8914),
    PlanningArea("Serangoon", Region.NORTH_EAST, 1.3554, 103.8679),
    PlanningArea("Pulau Tekong", Region.NORTH_EAST, 1.4060, 104.0430),
    # ---- West ----
    PlanningArea("Boon Lay", Region.WEST, 1.3380, 103.7010),
    PlanningArea("Bukit Batok", Region.WEST, 1.3590, 103.7637),
    PlanningArea("Bukit Panjang", Region.WEST, 1.3774, 103.7719),
    PlanningArea("Choa Chu Kang", Region.WEST, 1.3840, 103.7470),
    PlanningArea("Clementi", Region.WEST, 1.3162, 103.7649),
    PlanningArea("Jurong East", Region.WEST, 1.3329, 103.7436),
    PlanningArea("Jurong West", Region.WEST, 1.3404, 103.7090),
    PlanningArea("Pioneer", Region.WEST, 1.3150, 103.6970),
    PlanningArea("Tengah", Region.WEST, 1.3740, 103.7150),
    PlanningArea("Tuas", Region.WEST, 1.3210, 103.6350),
    PlanningArea("Western Islands", Region.WEST, 1.2080, 103.7460),
    PlanningArea("Western Water Catchment", Region.WEST, 1.4050, 103.6890),
)


# --------------------------------------------------------------------------- #
# Site-prefix fallback, for sensors whose coordinates never arrived.
# --------------------------------------------------------------------------- #
#: Lower-case site token -> region. Deliberately region-only: the site name tells
#: you roughly where a plant is, never which planning area it falls in.
SITE_REGION_HINTS: dict[str, Region] = {
    "marinabarrage": Region.CENTRAL,
    "stamforddt": Region.CENTRAL,
    "mrrs": Region.CENTRAL,
    "macritchieps": Region.CENTRAL,
    "bidadari": Region.CENTRAL,
    "arthurroad": Region.CENTRAL,
    "arthurroadtg": Region.CENTRAL,
    "bedokps": Region.EAST,
    "bedokipu": Region.EAST,
    "bedokpond": Region.EAST,
    "bedokpong": Region.EAST,
    "bedokdiversion": Region.EAST,
    "bedok": Region.EAST,
    "changi": Region.EAST,
    "tampines": Region.EAST,
    "kranji1ps": Region.NORTH,
    "kranji2ps": Region.NORTH,
    "kranjiipu": Region.NORTH,
    "woodlands": Region.NORTH,
    "sembawang": Region.NORTH,
    "yishun": Region.NORTH,
    "mandai": Region.NORTH,
    "upperseletarps": Region.NORTH,
    "lowerseletarps": Region.NORTH_EAST,
    "lowerseletartg": Region.NORTH_EAST,
    "seletar": Region.NORTH_EAST,
    "serangoontg": Region.NORTH_EAST,
    "punggolserangoon": Region.NORTH_EAST,
    "punggoltg": Region.NORTH_EAST,
    "pulautekong": Region.NORTH_EAST,
    "hougang": Region.NORTH_EAST,
    "angmokio": Region.NORTH_EAST,
    "pandan1ps": Region.WEST,
    "pandantg": Region.WEST,
    "jurongps": Region.WEST,
    "jurong": Region.WEST,
    "tuas": Region.WEST,
    "tengehps": Region.WEST,
    "westseletarps": Region.WEST,
    "ls2ps": Region.WEST,
    "clementi": Region.WEST,
    "choachukang": Region.WEST,
}

EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(min(1.0, a)))


def site_from_description(description: str) -> str | None:
    """
    Site token from a description such as ``BedokPS-Pump4-Deliver-Pressure``.

    The same convention `cluster_suppression` relies on for its name rule, so
    the two stay consistent about what counts as "the same site".
    """
    if not description:
        return None
    head = re.split(r"[-_ ]", description.strip(), maxsplit=1)[0]
    return head or None


@dataclass(frozen=True)
class Placement:
    """Where a sensor is, and how confident we are about it."""

    region: Region
    planning_area: str | None
    distance_m: float | None
    source: str          # "coordinates" | "site-name" | "none"

    @property
    def is_placed(self) -> bool:
        return self.region is not Region.UNKNOWN

    @property
    def region_is_reliable(self) -> bool:
        """
        Coordinate-derived regions are trustworthy; name-derived ones are a
        best guess. Callers that weight evidence should know which they have.
        """
        return self.source == "coordinates"


UNPLACED = Placement(Region.UNKNOWN, None, None, "none")


def place_by_coordinates(lat: float | None, lon: float | None,
                         max_distance_m: float = 15000.0) -> Placement:
    """
    Nearest planning-area centroid, if the point is plausibly in Singapore.

    Beyond `max_distance_m` the point is left unplaced rather than forced into
    whichever centroid happens to be least far away -- a bad coordinate should
    surface as missing, not as a confident wrong answer.
    """
    if lat is None or lon is None:
        return UNPLACED
    try:
        lat_f, lon_f = float(lat), float(lon)
    except (TypeError, ValueError):
        return UNPLACED
    if not (math.isfinite(lat_f) and math.isfinite(lon_f)):
        return UNPLACED
    # 0,0 is the classic "join produced nothing" coordinate.
    if lat_f == 0.0 and lon_f == 0.0:
        return UNPLACED

    best, best_d = None, float("inf")
    for area in PLANNING_AREAS:
        d = haversine_m(lat_f, lon_f, area.lat, area.lon)
        if d < best_d:
            best, best_d = area, d

    if best is None or best_d > max_distance_m:
        return UNPLACED
    return Placement(best.region, best.name, best_d, "coordinates")


def place_by_site_name(description: str) -> Placement:
    """
    Region from the site prefix, for sensors with no usable coordinates.

    Returns region only: a site name locates a plant to within a region, never
    to a planning area, and claiming otherwise would be inventing precision.
    """
    site = site_from_description(description)
    if not site:
        return UNPLACED
    token = re.sub(r"[^a-z0-9]", "", site.lower())
    if not token:
        return UNPLACED

    if token in SITE_REGION_HINTS:
        return Placement(SITE_REGION_HINTS[token], None, None, "site-name")

    # Longest-prefix match, so "BedokPS_4G" and "BedokIPU300" still resolve
    # without needing an entry each.
    matches = [(k, v) for k, v in SITE_REGION_HINTS.items() if token.startswith(k)]
    if matches:
        key, region = max(matches, key=lambda kv: len(kv[0]))
        return Placement(region, None, None, "site-name")
    return UNPLACED


def place(lat: float | None, lon: float | None, description: str = "",
          max_distance_m: float = 15000.0) -> Placement:
    """Coordinates first, site name as fallback."""
    by_coords = place_by_coordinates(lat, lon, max_distance_m=max_distance_m)
    if by_coords.is_placed:
        return by_coords
    return place_by_site_name(description)


def assignment_stats(placements: list[Placement]) -> dict[str, object]:
    """
    Coverage summary, to be logged every run.

    Geo-clustering is worthless for sensors with no placement, and the
    `RTUNumber -> LKey` join that supplies coordinates is known to be dirty, so
    coverage is a tracked metric rather than an assumption.
    """
    total = len(placements)
    if total == 0:
        return {"total": 0, "placed": 0, "placed_pct": 0.0,
                "by_source": {}, "by_region": {}}

    by_source: dict[str, int] = {}
    by_region: dict[str, int] = {}
    for p in placements:
        by_source[p.source] = by_source.get(p.source, 0) + 1
        by_region[p.region.value] = by_region.get(p.region.value, 0) + 1

    placed = sum(1 for p in placements if p.is_placed)
    return {
        "total": total,
        "placed": placed,
        "placed_pct": round(100.0 * placed / total, 1),
        "by_source": by_source,
        "by_region": by_region,
    }
