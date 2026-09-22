"""
Tests for das2.spatial.cluster (Phase 5).

The two cases that matter are opposites, and the whole point of the module is
telling them apart:

    several sites, same time  -> a real area event   -> go and look
    one site, many sensors    -> panel fan-out       -> nobody drives anywhere

The coordinates used here are the REAL ones from the fixture fleet, which are
themselves real PUB site positions, so the distances exercised are the
distances the production join actually produces -- the East sites are 3.3 to
4.4 km apart, which is why a 2 km radius was wrong.

Run:  python3 tests/test_cluster.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from das2.models import AnomalyType, PhysicalSeverity, SensorAnomaly, SensorMeta  # noqa: E402
from das2.spatial.cluster import (  # noqa: E402
    ClusterParams,
    cluster_anomalies,
    cluster_summary,
    is_single_site,
    region_equipment_matrix,
    unclustered,
    windows_overlap,
)
from das2.spatial.regions import haversine_m  # noqa: E402

T0 = datetime(2026, 9, 20, 3, 0, 0)

# Real PUB coordinates, as used by the fixture generator.
SITE = {
    "BedokPS":     (1.34286, 103.91977, "East"),
    "BedokPond4":  (1.31697, 103.93443, "East"),
    "TampinesPS":  (1.34960, 103.95680, "East"),
    "Kranji1PS":   (1.41567, 103.72871, "North"),
    "PandanTG":    (1.31214, 103.74723, "West"),
    "PulauTekong": (1.40396, 104.05330, "North-East"),
}


def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
    if not cond:
        raise SystemExit(1)


def anomaly(key, site, equipment, *, start_min=0, dur_min=150,
            atype=AnomalyType.LEVEL_SHIFT, span_fraction=0.35, placed=True,
            lat=None, lon=None, site_name=..., region=...):
    """
    One SensorAnomaly. `SensorMeta` is frozen, so every variation a test needs
    (a missing site name, a nudged position, a region straddling a boundary)
    has to be passed in here rather than patched afterwards.
    """
    slat, slon, sregion = SITE[site] if placed else (None, None, None)
    meta = SensorMeta(
        sensor_key=key,
        description=f"{site}-{equipment}",
        equipment=equipment,
        site=(site if placed else None) if site_name is ... else site_name,
        latitude=slat if lat is None else lat,
        longitude=slon if lon is None else lon,
        region=sregion if region is ... else region,
    )
    return SensorAnomaly(
        sensor=meta,
        start=T0 + timedelta(minutes=start_min),
        end=T0 + timedelta(minutes=start_min + dur_min),
        dominant_type=atype,
        severity=PhysicalSeverity(span_fraction=span_fraction,
                                  duration_s=dur_min * 60, window_fraction=0.1),
    )


def main():
    print("the geometry these tests rest on")
    d_bedok_pond = haversine_m(*SITE["BedokPS"][:2], *SITE["BedokPond4"][:2])
    d_bedok_tamp = haversine_m(*SITE["BedokPS"][:2], *SITE["TampinesPS"][:2])
    check("the East sites are 3-5 km apart, not 2",
          3000 < d_bedok_pond < 5000 and 3000 < d_bedok_tamp < 5000,
          f"(BedokPS-BedokPond4 {d_bedok_pond:.0f} m, BedokPS-TampinesPS {d_bedok_tamp:.0f} m)")
    check("East to West is far beyond any radius",
          haversine_m(*SITE["BedokPS"][:2], *SITE["PandanTG"][:2]) > 19000,
          f"({haversine_m(*SITE['BedokPS'][:2], *SITE['PandanTG'][:2])/1000:.1f} km)")

    # --- the case v1 is structurally blind to ------------------------------- #
    print("\nthe regional event: 4 sensors, 3 East sites, 2 equipment types")
    regional = [
        anomaly("r1", "BedokPS", "Pressure"),
        anomaly("r2", "BedokPond4", "Flowrate", start_min=10),
        anomaly("r3", "TampinesPS", "Pressure", start_min=5),
        anomaly("r4", "TampinesPS", "Flowrate", start_min=15),
    ]
    clusters = cluster_anomalies(regional)
    check("they form exactly one cluster", len(clusters) == 1,
          f"({len(clusters)} cluster(s))")
    c = clusters[0]
    check("all four are in it", len(c.members) == 4)
    check("it is reported as East", str(c.region) == "East", f"({c.region})")
    check("it spans several sites", len(c.sites) == 3, f"({sorted(c.sites)})")
    check("it is NOT single-site, so it cannot be fan-out", not is_single_site(c))
    check("it carries both equipment types", c.equipment_types == {"Pressure", "Flowrate"})
    check("its radius is site-scale, not island-scale", 500 < c.radius_m < 5000,
          f"({c.radius_m:.0f} m)")
    check("the centroid sits among the members",
          1.30 < c.centroid_lat < 1.36 and 103.90 < c.centroid_lon < 103.97,
          f"({c.centroid_lat:.5f}, {c.centroid_lon:.5f})")

    print("\n  and the old 2 km guess would have missed it entirely")
    tight = cluster_anomalies(regional, ClusterParams(radius_m=2000.0))
    joined = max((len(x.members) for x in tight), default=0)
    check("at 2 km radius it collapses to the one co-located pair", joined <= 2,
          f"(largest group {joined} of 4 -- this is why the radius was measured)")

    # --- the opposite verdict ------------------------------------------------ #
    print("\nthe fan-out: 3 sensors on ONE site")
    fanout = [anomaly(f"f{i}", "PandanTG", "Current", start_min=i * 3)
              for i in range(3)]
    clusters = cluster_anomalies(fanout)
    check("they form one cluster", len(clusters) == 1)
    c = clusters[0]
    check("all three are in it", len(c.members) == 3)
    check("its radius is exactly zero -- RTU-level coordinates", c.radius_m == 0.0)
    check("it IS single-site, the suppressible shape", is_single_site(c))

    print("\n  single-site detection does not depend on the site NAME")
    # description had no parseable site prefix, so `site` is None for all three
    nameless = [anomaly(f"n{i}", "PandanTG", "Current", site_name=None)
                for i in range(3)]
    c = cluster_anomalies(nameless)[0]
    check("co-located sensors with no site name are still one site",
          is_single_site(c), f"(radius {c.radius_m} m)")

    print("\n  two genuinely distinct sites are not collapsed into one")
    # Lower Peirce PS and MRRS Kallang are 174 m apart in the real feed.
    near = [anomaly("p1", "BedokPS", "Pressure"),
            anomaly("p2", "BedokPS", "Flowrate", site_name="MRRSGeylang",
                    lat=SITE["BedokPS"][0] + 0.00157)]          # ~174 m north
    c = cluster_anomalies(near)[0]
    check("sites 174 m apart cluster together", len(c.members) == 2)
    check("but are NOT reported as a single site", not is_single_site(c),
          f"(radius {c.radius_m:.0f} m)")

    # --- time is a real gate, not decoration --------------------------------- #
    print("\ntime separation splits what distance would join")
    same_place_apart = [
        anomaly("t1", "BedokPS", "Pressure", start_min=0, dur_min=30),
        anomaly("t2", "BedokPS", "Flowrate", start_min=600, dur_min=30),
    ]
    check("two faults 10 hours apart at one site do not cluster",
          cluster_anomalies(same_place_apart) == [])
    check("windows_overlap agrees",
          not windows_overlap(same_place_apart[0], same_place_apart[1], 30))
    print("  but the tolerance absorbs detector edge disagreement")
    edges = [
        anomaly("e1", "BedokPS", "Pressure", start_min=0, dur_min=60),
        anomaly("e2", "BedokPond4", "Flowrate", start_min=75, dur_min=60),
    ]
    check("a 15-minute offset still counts as simultaneous",
          len(cluster_anomalies(edges)) == 1)
    check("a 15-minute offset would NOT, with zero tolerance",
          cluster_anomalies(edges, ClusterParams(time_tolerance_min=0)) == [])

    # --- unplaced sensors ----------------------------------------------------- #
    print("\nunplaced sensors are excluded, never lumped together")
    mixed = regional + [
        anomaly("u1", "BedokPS", "LevelSensor", placed=False),
        anomaly("u2", "BedokPS", "LevelSensor", placed=False),
    ]
    clusters = cluster_anomalies(mixed)
    keys = {m.sensor.sensor_key for c in clusters for m in c.members}
    check("no cluster contains an unplaced sensor", not ({"u1", "u2"} & keys))
    check("the placed event is unaffected", len(clusters) == 1 and len(clusters[0].members) == 4)
    loose = unclustered(mixed, clusters)
    check("they resurface as unclustered instead of vanishing",
          {a.sensor.sensor_key for a in loose} == {"u1", "u2"},
          f"({[a.sensor.sensor_key for a in loose]})")

    print("\na lone anomaly is not a cluster")
    solo = [anomaly("s1", "Kranji1PS", "Flowrate")]
    check("one sensor produces no cluster", cluster_anomalies(solo) == [])
    check("and is returned as unclustered", len(unclustered(solo, [])) == 1)

    # --- chaining is bounded --------------------------------------------------- #
    print("\nan over-wide component IS split, when a split exists")
    # Two tight groups ~5.9 km apart, bridged by a single site in the middle.
    # Single linkage joins all seven through the bridge, giving a component
    # wider than one crew could cover; removing the bridge separates it
    # cleanly, so the cap can do its job without destroying anything.
    bridged = ([anomaly(f"a{i}", "BedokPS", "Pressure", site_name=f"A{i}",
                        lat=1.3000 + i * 0.0004) for i in range(3)]
               + [anomaly("bridge", "BedokPS", "Pressure", site_name="Bridge",
                          lat=1.3260)]
               + [anomaly(f"b{i}", "BedokPS", "Flowrate", site_name=f"B{i}",
                          lat=1.3520 + i * 0.0004) for i in range(3)])
    ends = haversine_m(bridged[0].sensor.latitude, bridged[0].sensor.longitude,
                       bridged[-1].sensor.latitude, bridged[-1].sensor.longitude)
    check("ungoverned, the chain would span past the cap", ends > 5000,
          f"({ends/1000:.1f} km end to end)")
    clusters = cluster_anomalies(bridged)
    check("it is split into two dispatchable clusters", len(clusters) == 2,
          f"({len(clusters)} clusters of sizes {[len(x.members) for x in clusters]})")
    check("each is well within the cap",
          all(x.radius_m < 500 for x in clusters),
          f"(radii {[round(x.radius_m) for x in clusters]} m)")
    loose = unclustered(bridged, clusters)
    check("the bridge site is not deleted, it becomes unclustered",
          [a.sensor.sensor_key for a in loose] == ["bridge"],
          f"({[a.sensor.sensor_key for a in loose]})")
    check("every one of the seven is still accounted for",
          sum(len(x.members) for x in clusters) + len(loose) == 7)

    print("\nbut a split that would dissolve the group is refused")
    # Eight sites in an evenly spaced line, 2.5 km apart, spanning 17.5 km.
    # There is no natural cut: the moment the radius drops below the spacing
    # the whole thing becomes singletons. A recursion that always tightened
    # would turn eight real anomalies into zero clusters, so it stops instead
    # and reports the true width -- which is itself the useful signal, since a
    # 17 km line of failures is a transmission-main event, not a site visit.
    chain = [anomaly(f"c{i}", "BedokPS", "Pressure", site_name=f"Site{i}",
                     lat=1.30 + i * 0.02247)            # ~2.5 km steps
             for i in range(8)]
    span = haversine_m(chain[0].sensor.latitude, chain[0].sensor.longitude,
                       chain[-1].sensor.latitude, chain[-1].sensor.longitude)
    check("the chain really does span the island", span > 15000,
          f"({span/1000:.1f} km end to end)")
    clusters = cluster_anomalies(chain)
    check("all eight are kept, not dissolved into nothing",
          len(clusters) == 1 and len(clusters[0].members) == 8,
          f"({len(clusters)} cluster(s), sizes {[len(x.members) for x in clusters]})")
    check("and its true width is reported, over the cap and visibly so",
          clusters[0].radius_m > 5000,
          f"(radius {clusters[0].radius_m/1000:.1f} km -- an operator can see this "
          f"is not one site visit)")

    print("\n  a tight group is never split")
    tight_group = [anomaly(f"g{i}", "BedokPS", "Pressure", site_name=f"Near{i}",
                           lat=1.34286 + i * 0.0018)   # ~200 m steps, 800 m total
                   for i in range(5)]
    clusters = cluster_anomalies(tight_group)
    check("five sites within 800 m stay one cluster",
          len(clusters) == 1 and len(clusters[0].members) == 5)

    # --- region gating is available, and has the cost the docs claim ---------- #
    print("\nregion gating is off by default, and this is why")
    # a region boundary drawn straight through one event
    cross = [anomaly("x1", "BedokPS", "Pressure"),
             anomaly("x2", "BedokPond4", "Flowrate", region="Central")]
    check("ungated, the two still cluster", len(cluster_anomalies(cross)) == 1)
    check("gated, the same event is split in two",
          cluster_anomalies(cross, ClusterParams(require_same_region=True)) == [])

    # --- the views the dashboard needs ---------------------------------------- #
    print("\nregion x equipment matrix -- the literal 'by region and by type' view")
    matrix = region_equipment_matrix(regional + fanout + [
        anomaly("m1", "Kranji1PS", "Pressure"),
    ])
    check("East holds both types", matrix["East"] == {"Pressure": 2, "Flowrate": 2},
          f"({matrix['East']})")
    check("West holds the fan-out", matrix["West"] == {"Current": 3})
    check("North holds the lone one", matrix["North"] == {"Pressure": 1})
    unplaced_matrix = region_equipment_matrix([anomaly("z", "BedokPS", "Flowrate", placed=False)])
    check("an unplaced sensor is counted as Unknown, not dropped",
          unplaced_matrix == {"Unknown": {"Flowrate": 1}}, f"({unplaced_matrix})")

    print("\nrun summary")
    all_anoms = regional + fanout + solo
    clusters = cluster_anomalies(all_anoms)
    loose = unclustered(all_anoms, clusters)
    summary = cluster_summary(clusters, loose)
    check("two clusters found", summary["clusters"] == 2, f"({summary})")
    check("one of them is multi-site -- the one worth driving to",
          summary["multi_site_clusters"] == 1)
    check("seven sensors clustered", summary["clustered_sensors"] == 7)
    check("the lone one is counted loose", summary["unclustered_sensors"] == 1)
    check("regions are reported as plain strings",
          summary["regions_affected"] == ["East", "West"],
          f"({summary['regions_affected']})")

    print("\nordering: the biggest cluster is presented first")
    check("regional event outranks the fan-out",
          len(clusters[0].members) >= len(clusters[1].members))

    print("\ndegenerate input")
    check("empty input", cluster_anomalies([]) == [])
    check("empty summary", cluster_summary([], [])["clusters"] == 0)
    check("all unplaced", cluster_anomalies([
        anomaly("q1", "BedokPS", "Pressure", placed=False),
        anomaly("q2", "BedokPS", "Flowrate", placed=False)]) == [])

    print("\nAll cluster tests passed.")


if __name__ == "__main__":
    main()
