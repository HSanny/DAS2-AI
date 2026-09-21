"""
Tests for Singapore region placement (Phase 1).

The headline test replays the ten real sensors in
docker_ready/abnormal_sensor_backup.csv and asserts every one lands in the
correct region. That file is the only real coordinate data in the repo, so it
is the only honest accuracy check available.

Run:  python3 tests/test_regions.py
"""

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from das2.spatial.regions import (  # noqa: E402
    PLANNING_AREAS,
    Region,
    assignment_stats,
    haversine_m,
    place,
    place_by_coordinates,
    place_by_site_name,
    site_from_description,
)


def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
    if not cond:
        raise SystemExit(1)


# Expected region per sensor in the shipped sample. Region only -- planning area
# is ambiguous near boundaries by design (BedokPS resolves to Paya Lebar; both
# are East), so asserting on it would be pinning a known limitation as if it
# were a guarantee.
EXPECTED_REGION = {
    "PulauTekong-Dissolved-Oxygen": Region.NORTH_EAST,
    "MRRS-THOMSON FLOWMETER READING": Region.CENTRAL,
    "PandanTG-LT-Voltage1": Region.WEST,
    "PandanTG-LT-Voltage2": Region.WEST,
    "PandanTG-LT-Voltage3": Region.WEST,
    "BedokPS-Pump4-Deliver-Pressure": Region.EAST,
    "Bidadari North WetLand Submersible Pump 2 Flow": Region.CENTRAL,
    "Kranji2PS-Mains-Delivery-PRESSURE": Region.NORTH,
    "Kranji1PS-Total-Flow-Rate": Region.NORTH,
    "BedokPond4-PS-Delivery-Flow-Rate": Region.EAST,
}


def main():
    print("real sensors from abnormal_sensor_backup.csv")
    backup = REPO / "docker_ready" / "abnormal_sensor_backup.csv"
    rows = list(csv.DictReader(open(backup)))
    check("sample file present and populated", len(rows) == 10, f"({len(rows)} rows)")

    placements = []
    for r in rows:
        desc = r["Description"]
        p = place(float(r["Latitude"]), float(r["Longitude"]), desc)
        placements.append(p)
        expected = EXPECTED_REGION[desc]
        check(f"{desc[:44]:44} -> {p.region.value}",
              p.region is expected,
              f"({p.planning_area}, {p.distance_m/1000:.1f} km)")

    check("every sample sensor resolves within 2.5 km of a centroid",
          all(p.distance_m is not None and p.distance_m < 2500 for p in placements),
          f"(max {max(p.distance_m for p in placements)/1000:.1f} km)")
    check("all placed from coordinates, not the name fallback",
          all(p.source == "coordinates" for p in placements))
    check("coordinate placements are marked reliable",
          all(p.region_is_reliable for p in placements))

    print("\nknown limitation is documented, not silently wrong")
    bedok = place_by_coordinates(1.34286, 103.91977)
    check("BedokPS lands in Paya Lebar, not Bedok",
          bedok.planning_area == "Paya Lebar",
          "(nearest-centroid boundary ambiguity)")
    check("...but the REGION is still correct, which is what clustering uses",
          bedok.region is Region.EAST)

    print("\nbad coordinates are refused rather than forced")
    for name, (lat, lon) in [
        ("null-island 0,0 (failed join)", (0.0, 0.0)),
        ("London", (51.5, -0.12)),
        ("Kuala Lumpur", (3.14, 101.69)),
        ("None", (None, None)),
        ("NaN", (float("nan"), float("nan"))),
    ]:
        p = place_by_coordinates(lat, lon)
        check(f"{name} is not placed", not p.is_placed)

    check("a point just outside the radius is refused",
          not place_by_coordinates(1.35, 103.85, max_distance_m=1.0).is_placed)

    print("\nsite-name fallback, for sensors whose coordinates never arrived")
    for desc, expected in [
        ("BedokPS-Pump4-Deliver-Pressure", Region.EAST),
        ("Kranji2PS-Mains-Delivery-PRESSURE", Region.NORTH),
        ("PandanTG-LT-Voltage1", Region.WEST),
        ("PulauTekong-Dissolved-Oxygen", Region.NORTH_EAST),
        ("MARINABARRAGE-CG1 TOTAL FLOW", Region.CENTRAL),
        ("SerangoonTG-SRGSR DAILY FLOW", Region.NORTH_EAST),
        ("BedokIPU300_4G-something", Region.EAST),       # prefix match
        ("Pandan1PS-Rainfall", Region.WEST),
    ]:
        p = place_by_site_name(desc)
        check(f"{desc[:40]:40} -> {p.region.value}", p.region is expected)

    p = place_by_site_name("BedokPS-Pump4")
    check("name fallback claims no planning area", p.planning_area is None,
          "(a site name cannot locate a planning area)")
    check("name fallback is marked unreliable", not p.region_is_reliable)
    check("unknown site is not placed", not place_by_site_name("ZZZUnknown-Thing").is_placed)
    check("empty description is not placed", not place_by_site_name("").is_placed)

    print("\nplace() prefers coordinates and falls back cleanly")
    good = place(1.4157, 103.7287, "Kranji1PS-Total-Flow-Rate")
    check("with coordinates, uses them", good.source == "coordinates")
    fallback = place(None, None, "Kranji1PS-Total-Flow-Rate")
    check("without coordinates, uses the name", fallback.source == "site-name")
    check("both agree on the region", good.region is fallback.region is Region.NORTH)
    check("bad coordinates still fall back to the name",
          place(0.0, 0.0, "Kranji1PS-Total-Flow-Rate").region is Region.NORTH)

    print("\nsite parsing")
    for desc, expected in [
        ("BedokPS-Pump4-Deliver-Pressure", "BedokPS"),
        ("Bidadari North WetLand Submersible Pump 2", "Bidadari"),
        ("MRRS-THOMSON FLOWMETER READING", "MRRS"),
        ("Pandan1PS_AIN_C2_P1", "Pandan1PS"),
        ("NoSeparator", "NoSeparator"),
    ]:
        check(f"{desc[:40]:40} -> {expected}", site_from_description(desc) == expected)
    check("empty description yields no site", site_from_description("") is None)

    print("\ncentroid table sanity")
    check("all five regions are represented",
          {a.region for a in PLANNING_AREAS} == {
              Region.CENTRAL, Region.EAST, Region.NORTH, Region.NORTH_EAST, Region.WEST})
    check("every centroid is inside Singapore's bounding box",
          all(1.15 < a.lat < 1.50 and 103.6 < a.lon < 104.1 for a in PLANNING_AREAS))
    check("no duplicate planning-area names",
          len({a.name for a in PLANNING_AREAS}) == len(PLANNING_AREAS))

    print("\nhaversine")
    check("zero distance for identical points", haversine_m(1.3, 103.8, 1.3, 103.8) == 0.0)
    # Changi to Tuas is about 48 km across the island.
    d = haversine_m(1.3450, 103.9832, 1.3210, 103.6350)
    check("Changi to Tuas is ~39 km", 36_000 < d < 42_000, f"({d/1000:.1f} km)")
    check("distance is symmetric",
          abs(haversine_m(1.3, 103.8, 1.4, 103.9) - haversine_m(1.4, 103.9, 1.3, 103.8)) < 1e-6)

    print("\ncoverage reporting")
    stats = assignment_stats(placements)
    check("total counted", stats["total"] == 10)
    check("all ten placed", stats["placed"] == 10 and stats["placed_pct"] == 100.0)
    check("regions tallied", sum(stats["by_region"].values()) == 10)
    mixed = placements + [place_by_site_name("ZZZ"), place_by_site_name("BedokPS-x")]
    check("unplaced sensors are counted, not hidden",
          assignment_stats(mixed)["placed"] == 11,
          "(geo-clustering is worthless for unplaced sensors, so coverage is tracked)")
    check("empty input does not raise", assignment_stats([])["total"] == 0)

    print("\nAll region tests passed.")


if __name__ == "__main__":
    main()
