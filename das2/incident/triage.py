"""
das2.incident.triage
====================

Decide what an incident *is*, and therefore what to do about it.

This module is the answer to the client's actual question:

    "...before actually sending someone down to the site to check on the
     sensors."

Everything upstream finds abnormal numbers. This is where abnormal numbers
become a recommendation an operator can act on, and the recommendation that
matters most is the negative one -- *do not drive there* -- because that is the
cost the client is trying to avoid.

Deterministic on purpose
------------------------
Every rule below is an explicit, inspectable condition on evidence the run
already has. No model, no learned weights, no score threshold pulled from
nowhere. Two reasons:

  * There are no labels yet. The Telegram feedback buttons started collecting
    them only recently, and until there are enough, any trained classifier
    would be fitting noise while looking authoritative.
  * An operator who is told "do not dispatch" must be able to see *why*. A
    recommendation nobody can audit will be ignored the first time it is wrong,
    and then ignored permanently.

The classification order is the order of *certainty*, not severity. Fan-out is
tested first because it is the one verdict that can be established structurally
-- sensors sharing an RTU that fail together share a cause, and that is a fact
about the wiring rather than an inference about the water.
"""

from __future__ import annotations

from das2.models import (
    MAINTENANCE_TYPES,
    SENSOR_HEALTH_TYPES,
    AnomalyType,
    Cluster,
    Incident,
    IncidentClass,
)
from das2.spatial.cluster import is_single_site

#: A cluster this wide or wider, over this many sites, is an area event rather
#: than a site fault. Two sites is not an area -- it is a coincidence worth
#: looking at, which is what WATCH is for.
REGIONAL_MIN_SITES = 2
REGIONAL_MIN_MEMBERS = 3

#: Neighbours moving with the suspect sensor means the water moved. Neighbours
#: sitting still while one instrument swings means the instrument is lying.
#: These are the thresholds on Pearson r over the incident window.
CORRELATION_STRONG = 0.6
CORRELATION_WEAK = 0.2

#: Rain at nearby gauges over the incident window, in mm, above which a
#: flow/level excursion is adequately explained by the weather. Deliberately
#: low: this only ever downgrades an alert to "monitor", and the cost of a
#: missed dispatch here is bounded by the fact that the event is still shown.
RAIN_EXPLAINS_MM = 2.0

#: Types that rain can plausibly explain. Rain moves water; it does not make a
#: transmitter stop reporting, so a STALE sensor in a downpour is still a
#: broken sensor.
RAIN_EXPLICABLE: frozenset[AnomalyType] = frozenset({
    AnomalyType.LEVEL_SHIFT,
    AnomalyType.SPIKE,
    AnomalyType.RESIDUAL_OUTLIER,
    AnomalyType.MASS_BALANCE_VIOLATION,
})


def _shares_rtu(cluster: Cluster) -> bool:
    """All members on one RTU: a shared cause in the telemetry, not the water."""
    rtus = {m.sensor.rtu_number for m in cluster.members if m.sensor.rtu_number}
    return len(rtus) == 1 and len(cluster.members) > 1


def classify(cluster: Cluster, *,
             neighbour_correlation: float | None = None,
             rainfall_mm: float | None = None) -> tuple[IncidentClass, list[str]]:
    """
    Classify one cluster, returning the class and the evidence behind it.

    The evidence list is not decoration. It is rendered verbatim into the alert
    and the dashboard, so an operator can see the reasoning and overrule it.
    """
    members = cluster.members
    if not members:
        return IncidentClass.WATCH, ["no members"]

    types = {m.dominant_type for m in members}
    sites = cluster.sites
    equipment = cluster.equipment_types
    why: list[str] = []

    # --- 1. Fan-out: structural, so it is decided first --------------------- #
    # Several sensors at one place, failing together, is one cause. Whether
    # that cause is a dead RTU or a pulled fuse, it is not several faults and
    # it is not an area event.
    if len(members) > 1 and (is_single_site(cluster) or _shares_rtu(cluster)):
        if _shares_rtu(cluster):
            why.append(f"all {len(members)} sensors share RTU "
                       f"{next(iter({m.sensor.rtu_number for m in members}))}")
        else:
            why.append(f"all {len(members)} sensors are at one site "
                       f"({', '.join(sorted(sites)) or 'unnamed'})")
        why.append("a shared cause in the telemetry, not several faults")
        return IncidentClass.TELEMETRY_FANOUT, why

    # --- 2. Rain explains it ------------------------------------------------ #
    # Checked before the area rules, because a regional flow excursion during a
    # downpour is the single most common false dispatch in a water network.
    if (rainfall_mm is not None and rainfall_mm >= RAIN_EXPLAINS_MM
            and types and types <= RAIN_EXPLICABLE):
        why.append(f"{rainfall_mm:.1f} mm of rain at nearby gauges during the window")
        why.append(f"affected types ({', '.join(sorted(t.value for t in types))}) "
                   f"are all rain-explicable")
        return IncidentClass.WEATHER_DRIVEN, why

    # --- 3. Area event ------------------------------------------------------ #
    if len(members) >= REGIONAL_MIN_MEMBERS and len(sites) >= REGIONAL_MIN_SITES:
        why.append(f"{len(members)} sensors across {len(sites)} sites "
                   f"({', '.join(sorted(sites))})")
        if len(equipment) > 1:
            why.append(f"{len(equipment)} equipment types affected "
                       f"({', '.join(sorted(equipment))}) -- unlikely to be one "
                       f"instrument failing")
        if cluster.radius_m:
            why.append(f"spread {cluster.radius_m/1000:.1f} km around "
                       f"{cluster.region or 'an unplaced centroid'}")
        if neighbour_correlation is not None and neighbour_correlation >= CORRELATION_STRONG:
            why.append(f"neighbouring sensors correlate (r={neighbour_correlation:.2f}) "
                       f"-- the water moved, not the instruments")
        return IncidentClass.REGIONAL_EVENT, why

    # --- 4. Maintenance ----------------------------------------------------- #
    if types and types <= MAINTENANCE_TYPES:
        why.append(f"gradual {', '.join(sorted(t.value for t in types))} "
                   f"with no abrupt failure")
        return IncidentClass.DRIFT_MAINTENANCE, why

    # --- 5. Instrument fault vs the process --------------------------------- #
    # This is the dispatch decision, and correlation is what decides it.
    if types and types <= SENSOR_HEALTH_TYPES:
        why.append(f"sensor-health fault ({', '.join(sorted(t.value for t in types))})")
        if neighbour_correlation is None:
            why.append("no neighbours available to corroborate -- treat the "
                       "instrument as suspect")
            return IncidentClass.SENSOR_FAULT, why
        if neighbour_correlation <= CORRELATION_WEAK:
            why.append(f"neighbours did not react (r={neighbour_correlation:.2f}) "
                       f"-- the instrument, not the water")
            return IncidentClass.SENSOR_FAULT, why
        why.append(f"but neighbours moved too (r={neighbour_correlation:.2f}), "
                   f"which a pure instrument fault cannot explain")
        return IncidentClass.WATCH, why

    if neighbour_correlation is not None and neighbour_correlation >= CORRELATION_STRONG:
        why.append(f"neighbours moved together (r={neighbour_correlation:.2f})")
        why.append("consistent with a real change in the process")
        return IncidentClass.PROCESS_EVENT, why

    why.append(f"{len(members)} sensor(s), "
               f"{', '.join(sorted(t.value for t in types))}, "
               f"no corroborating evidence either way")
    return IncidentClass.WATCH, why


# --------------------------------------------------------------------------- #
# Severity
# --------------------------------------------------------------------------- #
#: Weights for the severity blend. Spread across sites and equipment types is
#: weighted heavily because those are what distinguish "the area is in trouble"
#: from "one box broke", which is the distinction that decides a dispatch.
W_MEMBER_SEVERITY = 0.40
W_SITE_SPREAD = 0.25
W_TYPE_DIVERSITY = 0.15
W_SIZE = 0.20

#: A cluster touching this many sites is as spread as the score needs to care
#: about; beyond it the incident is already "the whole area".
SITES_SATURATE = 4
TYPES_SATURATE = 3
MEMBERS_SATURATE = 8

#: Class multipliers. Fan-out is driven to the floor because acting on it is
#: exactly the wasted trip this system exists to prevent.
CLASS_WEIGHT: dict[IncidentClass, float] = {
    IncidentClass.REGIONAL_EVENT: 1.00,
    IncidentClass.SENSOR_FAULT: 0.85,
    IncidentClass.PROCESS_EVENT: 0.55,
    IncidentClass.DRIFT_MAINTENANCE: 0.40,
    IncidentClass.WEATHER_DRIVEN: 0.30,
    IncidentClass.WATCH: 0.35,
    IncidentClass.TELEMETRY_FANOUT: 0.15,
}


def severity(cluster: Cluster, incident_class: IncidentClass) -> float:
    """
    Blend the physical severity of the members with the shape of the cluster.

    Deliberately not `max(member severity)`: one sensor pegged at a wild value
    is usually a broken sensor, whereas four sensors each moderately off across
    three sites is usually a real event. The old system ranked on the former
    and so led with its least trustworthy findings.
    """
    members = cluster.members
    if not members:
        return 0.0

    # Mean of the top three, so one extreme member cannot carry the incident
    # but a genuinely severe one is not averaged away either.
    top = sorted((m.score for m in members), reverse=True)[:3]
    member_component = sum(top) / len(top) / 100.0

    site_component = min(1.0, len(cluster.sites) / SITES_SATURATE)
    type_component = min(1.0, len(cluster.equipment_types) / TYPES_SATURATE)
    size_component = min(1.0, len(members) / MEMBERS_SATURATE)

    raw = (W_MEMBER_SEVERITY * member_component
           + W_SITE_SPREAD * site_component
           + W_TYPE_DIVERSITY * type_component
           + W_SIZE * size_component)

    return round(100.0 * raw * CLASS_WEIGHT.get(incident_class, 0.5), 1)


def narrate(incident: Incident, why: list[str]) -> str:
    """
    The human-readable summary that leads the Telegram message.

    Written so the first line alone is enough to decide whether to keep
    reading, because that is how an alert at 3 a.m. is actually consumed.
    """
    c = incident.cluster
    where = incident.cluster.region or "an unplaced area"
    sites = ", ".join(sorted(c.sites)) or "unnamed sites"
    head = (f"{incident.priority.value} {incident.incident_class.value} in {where}: "
            f"{len(c.members)} sensor(s) across {len(c.sites)} site(s) [{sites}]")
    body = "\n".join(f"  - {line}" for line in why)
    return f"{head}\n{body}\n\n{incident.recommendation}"
