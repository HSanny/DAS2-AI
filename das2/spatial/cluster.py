"""
das2.spatial.cluster
====================

Group abnormal sensors that went wrong **near each other, at the same time**.

This is the thing the client actually asked for:

    "a heatmap, or a regional map, where he can see clusters of abnormal
     sensors classified by location in Singapore ... before actually sending
     someone down to the site"

and it is the one capability neither existing codebase has. A grep across the
production system finds zero files mentioning latitude, longitude, geo,
haversine, cluster-by-place or correlation.

Why place matters more than name
--------------------------------
The existing suppression rule groups sensors by NAME similarity, which
correctly silences panel fan-out -- `MARINABARRAGE-CG1` and `-CG7` failing
together is one telemetry fault, not seven. But it is structurally blind to the
opposite case:

    four DIFFERENT sensors, at three DIFFERENT sites, all going abnormal at
    03:00 in the same part of the island

Nothing links those by name, so nothing links them at all. That case is exactly
the one worth driving to, and it is invisible today.

Clustering here answers the complementary question -- *did several places go
wrong together?* -- and the two verdicts are opposites:

    one site, similar names   -> fan-out    -> suppress, nobody drives anywhere
    several sites, same time  -> real event -> investigate the area

Honest limit: this groups SITES, not sensors
---------------------------------------------
Coordinates come from an `RTUNumber -> LKey` join, so every sensor on one RTU
carries identical latitude and longitude. Distance between two sensors at the
same site is exactly zero, and there is no sub-site resolution to be had. That
is the right granularity for deciding where to send someone, but the map must
not imply more precision than exists, and any design premised on sensor-level
distance would be built on nothing.

The radius was measured, not guessed
------------------------------------
Every parameter below comes from the real `LongLat.csv` (76 distinct located
sites, the 7 fake "Unused" rows excluded). Two measurements decided the design.

**Site spacing.** Nearest-neighbour distance between real PUB sites:

    p10 69 m · p25 445 m · median 1,049 m · p75 2,107 m · p95 3,224 m

So sites are typically about a kilometre apart, and a radius under ~2 km leaves
a large minority of sites with no reachable neighbour at all -- they could never
join any cluster, whatever happened:

    radius 1,000 m -> 49% of sites have a neighbour
    radius 2,000 m -> 68%
    radius 3,000 m -> 91%
    radius 4,000 m -> 97%

**Percolation.** Against that, single linkage chains, so too large a radius
fuses the island into one meaningless blob. Taking the worst case -- every site
abnormal at once -- and measuring the largest connected component:

    radius   sites in largest component      its diameter
     2,000 m            11  (14%)                3,778 m
     2,500 m            14  (18%)                8,319 m
     3,000 m            17  (22%)               12,555 m
     3,500 m            57  (75%)               26,666 m   <- percolates
     4,000 m            64  (84%)               28,477 m
     6,000 m            75  (99%)               35,688 m

There is a sharp threshold between 3.0 and 3.5 km: above it, the answer to
"which sites went wrong together" is always "the whole island", which is no
answer at all.

That looks like an argument for a radius below 3 km, and it was, until the two
measurements were put side by side. They are in direct conflict: real events
span 3-4 km, and 3.5 km percolates. **No single radius satisfies both**, so
the radius cannot be the thing that bounds a cluster. The diameter cap is,
which is why the radius is free to be set at 5 km -- large enough to link a
real spread-out event -- while the cap keeps the result dispatchable. The
percolation table above is therefore not a limit on the radius; it is the
evidence that the cap has to exist.

Why region is *not* used as a hard gate
---------------------------------------
Requiring both sensors to be in the same region is a very effective brake on
chaining (worst component 20 sites / 14.3 km even at a 6 km radius, versus 75
sites / 35.7 km ungated). It is still wrong as a default, because it splits
real pairs. Of site pairs within 3 km, **8.6% straddle a region boundary**, and
the closest of them are plainly one place:

    Lower Peirce PS (North-East) <->  174 m -> MRRS Kallang (Central)
    Lower Seletar PS (North-East) <-> 1,025 m -> Lower Seletar TG (North)

Calling two instruments 174 m apart "separate events" to protect a boundary
drawn for urban planning would be an obvious error to any operator reading the
map. Chaining is bounded by the diameter cap instead, which is physical.
`require_same_region` remains available for anyone who wants the stricter
behaviour, with its measured cost recorded here.

Chaining is bounded in TIME as well as space
--------------------------------------------
Overlap is transitive; simultaneity is not. If A overlaps B and B overlaps C,
single linkage puts all three together even when A and C are nineteen hours
apart -- which is not what "these went wrong together" claims, and is a
reliable way to manufacture a severe incident out of unrelated faults. Members
of a cluster are therefore required to share one common instant, not merely a
chain of overlaps.

Chaining is bounded by splitting, and splitting can refuse
----------------------------------------------------------
Single linkage can chain (A near B, B near C, A far from C), which for a fault
propagating along a main is correct behaviour rather than a defect. But a
cluster 12 km across is not somewhere you can send one crew, so a component
wider than `max_diameter_m` is **re-split at a tighter radius** until it fits.

Splitting rather than trimming matters: trimming the outliers off a sprawling
component would silently delete real anomalies from the run. And the split
itself can decline. An evenly spaced chain of sites fragments into nothing but
singletons the instant the radius drops below the spacing, so a recursion that
always tightened would turn eight real anomalies into zero clusters. A
tightened radius is therefore accepted only when it produces at least two
pieces that are themselves clusters; otherwise the group is kept whole and its
true width reported, for the operator to judge.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np

from das2.models import Cluster, SensorAnomaly
from das2.spatial.regions import haversine_m

#: How far apart two sites may be and still be linked. Set equal to the
#: dispatch-area cap below, because that is the same question asked twice:
#: there is no point linking at a distance wider than the cluster is allowed to
#: be. Measured consequences at 5 km: a neighbour is reachable for 97% of real
#: sites, and the three East sites of the fixture fleet -- 3.3, 4.2 and 4.4 km
#: apart, all real PUB positions -- link as the one event they are. A 2 km or
#: 3 km radius links none of them, and a 4 km radius links only two of three.
#:
#: The percolation risk that would normally rule out a radius this large is
#: handled by the diameter cap, not by the radius. See below.
DEFAULT_RADIUS_M = 5000.0

#: A cluster wider than this is re-split at a tighter radius. Singapore is
#: ~40 km across, so 5 km is a coherent dispatch area -- roughly a quarter-hour
#: drive end to end -- while the 28 km worst-case component at the default
#: radius plainly is not.
#:
#: This is the parameter that actually controls the output. Re-running the
#: worst case (all 76 sites abnormal at once) shows the cap dominating the
#: radius almost completely:
#:
#:     radius   cap     clusters   largest   max diameter
#:      3 km    5 km       15        11         4,941 m
#:      4 km    5 km       15        11         4,941 m
#:      5 km    5 km       16        11         4,941 m
#:      5 km    3 km       19         8         2,618 m
#:      5 km    8 km       14        12         7,256 m
#:
#: so the radius can be set by "what should be able to link", and the cap by
#: "how far apart may one crew's targets be", independently.
DEFAULT_MAX_DIAMETER_M = 5000.0

#: Re-splitting stops here. Below this, sites are close enough to be one place
#: (25% of real sites have a neighbour within 445 m), so tightening further
#: would only separate instruments in the same compound.
DEFAULT_MIN_RADIUS_M = 250.0

#: Each re-split step multiplies the radius by this. 0.7 reaches the floor from
#: 3 km in six steps, so the recursion is shallow and bounded.
RADIUS_SHRINK = 0.7

#: Windows this far apart still count as simultaneous. Sensors report on
#: independent 120 s scans and detectors pick slightly different edges, so
#: demanding exact overlap would split one event in two.
DEFAULT_TIME_TOLERANCE_MIN = 30

#: Below this a "cluster" is just one sensor, which is not a cluster.
DEFAULT_MIN_SIZE = 2

#: Two positions closer than this are the same site. Not zero, because the
#: closest genuinely distinct sites in the real data are 30 m apart, and an
#: exact-equality test would be defeated by any future coordinate jitter.
SAME_SITE_M = 50.0


@dataclass(frozen=True)
class ClusterParams:
    radius_m: float = DEFAULT_RADIUS_M
    max_diameter_m: float = DEFAULT_MAX_DIAMETER_M
    min_radius_m: float = DEFAULT_MIN_RADIUS_M
    time_tolerance_min: int = DEFAULT_TIME_TOLERANCE_MIN
    min_size: int = DEFAULT_MIN_SIZE
    #: Refuse to cluster across region boundaries. Off by default: measured at
    #: 3 km, this would wrongly split 8.6% of neighbouring site pairs, one of
    #: them 174 m apart. See the module docstring.
    require_same_region: bool = False


def windows_overlap(a: SensorAnomaly, b: SensorAnomaly, tolerance_min: int) -> bool:
    """True when two anomaly windows overlap, within a tolerance."""
    if None in (a.start, a.end, b.start, b.end):
        return False
    gap = timedelta(minutes=tolerance_min)
    return (a.start - gap) <= b.end and (b.start - gap) <= a.end


def _distance_matrix(anomalies: list[SensorAnomaly]) -> np.ndarray:
    n = len(anomalies)
    d = np.zeros((n, n), dtype=float)
    for i in range(n):
        si = anomalies[i].sensor
        for j in range(i + 1, n):
            sj = anomalies[j].sensor
            d[i, j] = d[j, i] = haversine_m(si.latitude, si.longitude,
                                            sj.latitude, sj.longitude)
    return d


def _compatibility_matrix(anomalies: list[SensorAnomaly],
                          params: ClusterParams) -> np.ndarray:
    """
    Everything about a pair EXCEPT distance: time overlap, and region if gated.

    Kept separate from distance because re-splitting varies only the radius, and
    recomputing the time test at every recursion level would be wasted work.
    """
    n = len(anomalies)
    ok = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = anomalies[i], anomalies[j]
            if params.require_same_region and a.sensor.region != b.sensor.region:
                continue
            if not windows_overlap(a, b, params.time_tolerance_min):
                continue
            ok[i, j] = ok[j, i] = True
    return ok


def _components(members: list[int], compat: np.ndarray, dist: np.ndarray,
                radius: float) -> list[list[int]]:
    """Connected components of `members` under (compatible AND within radius)."""
    pool = set(members)
    seen: set[int] = set()
    out: list[list[int]] = []
    for start in members:
        if start in seen:
            continue
        stack, comp = [start], []
        seen.add(start)
        while stack:
            node = stack.pop()
            comp.append(node)
            for other in pool:
                if other in seen:
                    continue
                if compat[node, other] and dist[node, other] <= radius:
                    seen.add(other)
                    stack.append(other)
        out.append(comp)
    return out


def _diameter(comp: list[int], dist: np.ndarray) -> float:
    if len(comp) < 2:
        return 0.0
    idx = np.asarray(comp)
    return float(dist[np.ix_(idx, idx)].max())


def _split_oversized(comp: list[int], compat: np.ndarray, dist: np.ndarray,
                     radius: float, params: ClusterParams) -> list[list[int]]:
    """
    Break one over-wide component up, or leave it alone if it cannot be broken
    up without destroying it.

    A tightened radius is accepted only when it yields **at least two pieces
    that are themselves clusters**. That condition is what stops the obvious
    failure: an evenly-spaced chain of sites fragments into nothing but
    singletons the moment the radius drops below the spacing, and since
    singletons are not clusters, a naive recursion would dissolve eight real
    anomalies into zero output. Refusing such a split keeps the group and
    reports its true width instead, which an operator can judge for themselves.
    """
    tighter = radius * RADIUS_SHRINK
    while tighter >= params.min_radius_m:
        pieces = _components(comp, compat, dist, tighter)
        if sum(1 for p in pieces if len(p) >= params.min_size) >= 2:
            out: list[list[int]] = []
            for piece in pieces:
                if len(piece) < 2 or _diameter(piece, dist) <= params.max_diameter_m:
                    out.append(piece)
                else:
                    out.extend(_split_oversized(piece, compat, dist, tighter, params))
            return out
        tighter *= RADIUS_SHRINK
    return [comp]


def _bounded_components(members: list[int], compat: np.ndarray, dist: np.ndarray,
                        radius: float, params: ClusterParams) -> list[list[int]]:
    """
    Components at `radius`, re-split until each is narrower than the cap.

    Terminates because the radius strictly shrinks toward `min_radius_m`. A
    component that still will not fit at the floor is returned intact rather
    than discarded -- the members are real, and an honestly over-sized cluster
    is better than silently dropping anomalies. Its `radius_m` says so.
    """
    out: list[list[int]] = []
    for comp in _components(members, compat, dist, radius):
        if len(comp) < 2 or _diameter(comp, dist) <= params.max_diameter_m:
            out.append(comp)
        else:
            out.extend(_split_oversized(comp, compat, dist, radius, params))
    return out


def _share_an_instant(members: list[SensorAnomaly], tolerance_min: int) -> bool:
    """True when every member's window covers one common moment."""
    gap = timedelta(minutes=tolerance_min)
    latest_start = max(m.start for m in members)
    earliest_end = min(m.end for m in members)
    return latest_start - gap <= earliest_end + gap


def _split_by_time(members: list[SensorAnomaly],
                   tolerance_min: int) -> list[list[SensorAnomaly]]:
    """
    Partition a component into groups that are genuinely SIMULTANEOUS.

    Time overlap is transitive under single linkage and simultaneity is not,
    and the difference is not academic. Measured on the fixture: a
    quantisation collapse whose window legitimately spans 19.7 hours overlapped
    a stale sensor, a pump contradiction, a reverse-flow event and the real
    four-sensor level shift -- none of which overlapped each other. Chained
    together they formed one 8-member "REGIONAL_EVENT", scored P1, and would
    have sent a crew to investigate an area event that never happened. Five
    independent faults, presented as the most severe thing on the map.

    So a cluster is required to share a common instant, which is what "these
    happened together" actually claims. The greedy sweep below takes the
    earliest-ending window, groups everything that has started by then, and
    repeats -- the standard interval-stabbing construction, and it yields
    groups each of which genuinely overlaps at a point.
    """
    if len(members) < 2 or _share_an_instant(members, tolerance_min):
        return [members]

    gap = timedelta(minutes=tolerance_min)
    remaining = sorted(members, key=lambda m: m.end)
    groups: list[list[SensorAnomaly]] = []
    while remaining:
        pivot = remaining[0].end + gap
        group = [m for m in remaining if m.start - gap <= pivot]
        groups.append(group)
        keep = {id(m) for m in group}
        remaining = [m for m in remaining if id(m) not in keep]
    return groups


def _centroid(members: list[SensorAnomaly]) -> tuple[float | None, float | None]:
    placed = [m.sensor for m in members if m.sensor.has_coords]
    if not placed:
        return None, None
    return (sum(s.latitude for s in placed) / len(placed),
            sum(s.longitude for s in placed) / len(placed))


def _radius(members: list[SensorAnomaly], lat: float | None,
            lon: float | None) -> float:
    if lat is None or lon is None:
        return 0.0
    return max((haversine_m(lat, lon, m.sensor.latitude, m.sensor.longitude)
                for m in members if m.sensor.has_coords), default=0.0)


def _dominant_region(members: list[SensorAnomaly]) -> str | None:
    """
    Most common region among members, ties broken by name so runs are stable.

    Region rather than planning area, because nearest-centroid placement is
    reliable at region level and ambiguous at planning-area level near
    boundaries (BedokPS resolves to Paya Lebar, not Bedok -- both East).
    """
    counts: dict[str, int] = {}
    for m in members:
        if m.sensor.region:
            counts[m.sensor.region] = counts.get(m.sensor.region, 0) + 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))[0][0]


def cluster_anomalies(anomalies: list[SensorAnomaly],
                      params: ClusterParams | None = None) -> list[Cluster]:
    """
    Group anomalies into spatio-temporal clusters.

    Unplaced sensors are excluded rather than lumped together: without
    coordinates there is no evidence they are near anything, and a
    "cluster of everything we cannot place" would be a fiction. They still
    appear individually elsewhere -- see `unclustered`.

    Returns clusters ordered by member count, largest first.
    """
    params = params or ClusterParams()
    placed = [a for a in anomalies if a.sensor.has_coords]
    if len(placed) < 2:
        return []

    dist = _distance_matrix(placed)
    compat = _compatibility_matrix(placed, params)
    components = _bounded_components(list(range(len(placed))), compat, dist,
                                     params.radius_m, params)

    clusters: list[Cluster] = []
    for comp in components:
        # Spatial components are bounded by the diameter cap; they must also be
        # bounded in TIME, or one long-running fault bridges every unrelated
        # anomaly near it into a single severe-looking incident.
        for members in _split_by_time([placed[i] for i in comp],
                                      params.time_tolerance_min):
            if len(members) < params.min_size:
                continue
            lat, lon = _centroid(members)
            clusters.append(Cluster(
                members=sorted(members, key=lambda m: -m.score),
                region=_dominant_region(members),
                centroid_lat=lat,
                centroid_lon=lon,
                radius_m=round(_radius(members, lat, lon), 1),
            ))

    clusters.sort(key=lambda c: (-len(c.members), -max(m.score for m in c.members)))
    return clusters


def unclustered(anomalies: list[SensorAnomaly],
                clusters: list[Cluster]) -> list[SensorAnomaly]:
    """
    Anomalies that are in no cluster: lone faults, and everything unplaced.

    These are not less important -- a single frozen sensor with no neighbours
    reacting is the clearest possible dispatch case. They simply are not a
    *cluster*, and folding them into one would be dishonest.
    """
    clustered = {m.sensor.sensor_key for c in clusters for m in c.members}
    return [a for a in anomalies if a.sensor.sensor_key not in clustered]


# --------------------------------------------------------------------------- #
# Views the dashboard needs
# --------------------------------------------------------------------------- #
def is_single_site(cluster: Cluster) -> bool:
    """
    All members at one place -- the fan-out shape.

    Tested on geometry as well as names, because `site` is parsed from the
    description and is missing for some sensors: a cluster whose members are
    all within `SAME_SITE_M` of its centroid is one site whatever it is called.
    Both tests must agree, so two genuinely distinct sites 174 m apart are not
    collapsed, and two differently-named points on one RTU are not treated as
    two places.

    Not a verdict on its own: triage combines this with name similarity and
    shared RTU. But a cluster spanning several sites cannot be panel fan-out,
    which is the distinction that matters most.
    """
    return len(cluster.sites) <= 1 and cluster.radius_m <= SAME_SITE_M


def region_equipment_matrix(anomalies: list[SensorAnomaly]
                            ) -> dict[str, dict[str, int]]:
    """
    Counts by region x equipment type.

    This is the literal "observe the anomaly by region and by type" request,
    and the view that makes a pattern visible at a glance: one column lighting
    up across every region is a fleet-wide equipment problem, one row lighting
    up across every type is something wrong with that area.
    """
    matrix: dict[str, dict[str, int]] = {}
    for a in anomalies:
        region = a.sensor.region or "Unknown"
        equipment = a.sensor.equipment or "UNCLASSIFIED"
        matrix.setdefault(region, {})
        matrix[region][equipment] = matrix[region].get(equipment, 0) + 1
    return matrix


def cluster_summary(clusters: list[Cluster], loose: list[SensorAnomaly]
                    ) -> dict[str, object]:
    """One-line-per-run summary, for logs and the dashboard header."""
    multi_site = [c for c in clusters if not is_single_site(c)]
    return {
        "clusters": len(clusters),
        "multi_site_clusters": len(multi_site),
        "clustered_sensors": sum(len(c.members) for c in clusters),
        "unclustered_sensors": len(loose),
        "largest_cluster": max((len(c.members) for c in clusters), default=0),
        "widest_cluster_m": max((c.radius_m for c in clusters), default=0.0),
        "regions_affected": sorted({str(c.region) for c in clusters if c.region}),
    }
