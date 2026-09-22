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
    PROCESS_TYPES,
    CROSS_SIGNAL_TYPES,
    DEFINITIVE_INSTRUMENT_FAULTS,
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

#: When "many sensors, all quiet" stops meaning broken instruments and starts
#: meaning a broken feed.
#:
#: 20 is deliberately well above anything a site visit could plausibly fix and
#: well below what the client's real run produced (116, 123 and 115 sensors
#: across six sites each, every one of them recommending a technician). It is
#: a guess in the same sense the other thresholds are -- nobody has labelled
#: data for it -- but the cost is asymmetric: calling a genuine multi-site
#: instrument failure an outage costs one wrong line in an alert that still
#: pages someone, while calling an outage an instrument fault sends crews to
#: six sites. Raise it once real outages have been seen and counted.
OUTAGE_MIN_MEMBERS = 20
OUTAGE_MIN_SITES = 2

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

    # --- 2. The telemetry path, not the instruments ------------------------- #
    # Structural, like fan-out above, so it is decided early -- and it MUST
    # precede the instrument-fault rules, which match STALE outright and
    # would otherwise answer first. That is not hypothetical: placed after
    # them, this branch was unreachable and the test below caught it.
    #
    # On the client's first real run this branch did not exist, and three of
    # the five P1 alerts were "Instrument fault ... dispatch a technician" over
    # 116, 123 and 115 sensors spanning six sites each. A hundred instruments
    # do not fail together across kilometres. The path carrying their readings
    # does -- and in that run the cause was visible in the ingest stats: a
    # 24-hour hole in the historian feed, which makes every sensor look STALE
    # at once.
    #
    # Sending crews to six sites for a comms outage is precisely the wasted
    # trip this system exists to prevent, so the generic dispatch line must not
    # be reachable at this scale. TELEMETRY_FANOUT already covers one panel or
    # one RTU; this covers the wider case, where the remedy is a link or a
    # feed rather than a fuse.
    if (len(members) >= OUTAGE_MIN_MEMBERS and len(sites) >= OUTAGE_MIN_SITES
            and types and types <= SENSOR_HEALTH_TYPES):
        why.append(f"{len(members)} sensors across {len(sites)} sites stopped "
                   f"reporting together "
                   f"({', '.join(sorted(t.value for t in types))})")
        why.append("instruments do not fail in this number across this "
                   "distance -- the telemetry path is the suspect, not the "
                   "sensors")
        why.append("check the comms link, the RTU group and the historian "
                   "feed before dispatching anyone")
        return IncidentClass.TELEMETRY_OUTAGE, why

    # --- 3. Rain explains it ------------------------------------------------ #
    # Checked before the area rules, because a regional flow excursion during a
    # downpour is the single most common false dispatch in a water network.
    if (rainfall_mm is not None and rainfall_mm >= RAIN_EXPLAINS_MM
            and types and types <= RAIN_EXPLICABLE):
        why.append(f"{rainfall_mm:.1f} mm of rain at nearby gauges during the window")
        why.append(f"affected types ({', '.join(sorted(t.value for t in types))}) "
                   f"are all rain-explicable")
        return IncidentClass.WEATHER_DRIVEN, why

    # --- 4. Area event ------------------------------------------------------ #
    # An area event means the WATER moved, so it needs enough members whose
    # anomaly is about a process rather than an instrument. Without this test,
    # any three independent broken sensors at nearby sites whose windows happen
    # to overlap are reported as a regional event -- and on the fixture that is
    # exactly what happened: a quantisation collapse, a stale transmitter, a
    # pump contradiction and a brief reverse flow, four unrelated faults with
    # nothing in common but a shared instant, scored P2 and recommended for
    # area investigation.
    process_members = [m for m in members if m.dominant_type in PROCESS_TYPES]
    if (len(members) >= REGIONAL_MIN_MEMBERS and len(sites) >= REGIONAL_MIN_SITES
            and len(process_members) >= REGIONAL_MIN_MEMBERS):
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

    # --- 5. Instruments contradicting each other ----------------------------- #
    # Checked before the correlation-dependent rules, because a contradiction
    # between two instruments is already conclusive. Neighbour correlation
    # cannot make "level, inflow and outflow disagree" go away -- it can only
    # say which reading to doubt, and that is a question for the technician
    # standing in front of them.
    if types & CROSS_SIGNAL_TYPES:
        conflicting = sorted(t.value for t in types & CROSS_SIGNAL_TYPES)
        why.append(f"{', '.join(conflicting)}: instruments that should agree "
                   f"do not")
        for member in members:
            verdict = next((s.detail.get("verdict") for s in member.signals
                            if s.detail.get("verdict")), None)
            if verdict:
                why.append(verdict)
                break
        return IncidentClass.INSTRUMENT_CONFLICT, why

    # --- 6. Maintenance ----------------------------------------------------- #
    if types and types <= MAINTENANCE_TYPES:
        why.append(f"gradual {', '.join(sorted(t.value for t in types))} "
                   f"with no abrupt failure")
        return IncidentClass.DRIFT_MAINTENANCE, why

    # --- 7. Definitively broken instruments ---------------------------------- #
    # Not correlation-dependent, deliberately. A sensor that has stopped
    # reporting, or frozen, or lost its resolution, is broken whatever its
    # neighbours did -- correlation says something about a sensor's VALUE, and
    # these faults are not about the value.
    if types and types <= DEFINITIVE_INSTRUMENT_FAULTS:
        why.append(f"{', '.join(sorted(t.value for t in types))}: the "
                   f"instrument itself has failed")
        why.append("not a judgement about the water, so neighbour behaviour "
                   "does not change it")
        return IncidentClass.SENSOR_FAULT, why

    # --- 8. Instrument fault vs the process --------------------------------- #
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
    # A physical contradiction is as certain as this system gets: two readings
    # cannot both be true, so there is nothing probabilistic left to discount.
    IncidentClass.INSTRUMENT_CONFLICT: 0.90,
    IncidentClass.SENSOR_FAULT: 0.85,
    # Real and worth fixing -- a feed that has stopped blinds the whole
    # detector -- but it is one job for whoever owns the link, not a severity
    # that should outrank a genuine regional event just because it swept up a
    # hundred sensors. Member count is already part of the raw score, so left
    # unweighted an outage would dominate every run it appears in.
    IncidentClass.TELEMETRY_OUTAGE: 0.45,
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
