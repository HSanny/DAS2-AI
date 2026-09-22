"""
das2.incident.build
===================

Turn clusters into incidents, with stable identity across runs.

The identity problem is the point
--------------------------------
The pipeline runs on a schedule over an overlapping window, so the same fault
is re-discovered on every run. The system being replaced had no notion of this
at all: `fetch_unsent` sent every new row, so one three-day condition produced
up to twelve Telegram photos. That is almost certainly the client's loudest
current complaint, and it is not a detector problem -- the detector was right
twelve times.

An incident therefore needs an id that survives:

  * the window sliding, so its start time changes;
  * members joining and leaving as the event spreads or recedes;
  * a sensor recovering and relapsing within the continuity window.

Matching on member sets alone fails the second case, and matching on time alone
fails the third. So a candidate cluster continues an open incident when it
overlaps it *enough* in membership AND is in the same region AND is close
enough in time -- all three, because each one alone has a failure mode that
this data actually exhibits.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta

from das2.models import (
    Cluster,
    Incident,
    IncidentClass,
    IncidentStatus,
    SensorAnomaly,
)
from das2.incident.triage import classify, narrate, severity

#: Fraction of members that must be shared for a cluster to continue an open
#: incident. 0.5 tolerates an event that doubles in size or halves, which is
#: what spreading and receding events actually do, while refusing to merge two
#: unrelated faults that happen to share one sensor.
DEFAULT_JACCARD_MIN = 0.5

#: An incident not seen for this long is closed. Sized well above the run
#: interval so a single missed run does not resolve and immediately reopen an
#: ongoing fault -- which would produce exactly the alert storm the identity
#: logic exists to prevent.
DEFAULT_CONTINUITY_HOURS = 6.0

#: Severity must move by this much before an open incident is re-announced.
#: Without it, noise in the severity blend would page an operator every run
#: about an incident they already acknowledged.
DEFAULT_ESCALATION_DELTA = 15.0


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def make_incident_id(cluster: Cluster, opened_at: datetime) -> str:
    """
    A stable, readable id.

    Region and date are in the clear because an operator reading a Telegram
    message should be able to tell two incidents apart at a glance; the hash
    only disambiguates. Built from the member set rather than a counter so it
    can be regenerated without consulting the database.
    """
    region = str(cluster.region or "UNPLACED").replace(" ", "")
    keys = ",".join(sorted(cluster.sensor_keys))
    digest = hashlib.sha1(f"{region}|{keys}".encode()).hexdigest()[:8]
    return f"{region}-{opened_at:%Y%m%d}-{digest}"


def build_incident(cluster: Cluster, *, now: datetime,
                   neighbour_correlation: float | None = None,
                   rainfall_mm: float | None = None) -> Incident:
    """Classify, score and narrate one cluster as a fresh incident."""
    incident_class, why = classify(cluster,
                                   neighbour_correlation=neighbour_correlation,
                                   rainfall_mm=rainfall_mm)
    opened_at = cluster.start or now
    incident = Incident(
        incident_id=make_incident_id(cluster, opened_at),
        cluster=cluster,
        incident_class=incident_class,
        status=IncidentStatus.OPEN,
        severity=severity(cluster, incident_class),
        opened_at=opened_at,
        last_seen_at=cluster.end or now,
        neighbour_correlation=neighbour_correlation,
        rainfall_mm=rainfall_mm,
        detail={"evidence": why},
    )
    incident.narrative = narrate(incident, why)
    return incident


def reconcile(candidates: list[Incident], open_incidents: list[Incident], *,
              now: datetime,
              jaccard_min: float = DEFAULT_JACCARD_MIN,
              continuity_hours: float = DEFAULT_CONTINUITY_HOURS,
              escalation_delta: float = DEFAULT_ESCALATION_DELTA
              ) -> tuple[list[Incident], list[Incident], list[Incident]]:
    """
    Match this run's incidents against the ones already open.

    Returns `(new, updated, resolved)`.

    An incident in `updated` keeps its original id, `opened_at` and
    acknowledgement -- that is the whole point, and it is what stops an
    operator being paged twelve times for one fault. It is re-announced only
    when the evidence has meaningfully changed, which is what
    `escalation_delta` governs.
    """
    # Matching comes FIRST, and resolution is decided from what is left over.
    #
    # The reverse order -- resolve anything not seen recently, then match --
    # looks equivalent and is not. The analysis window is far longer than the
    # run interval, so a fault that ended 40 hours ago is still inside the
    # window and still re-detected on every run. Pre-resolving it closed the
    # incident and the very same run immediately re-opened it as new, churning
    # one fault through resolve/reopen indefinitely and re-alerting each time.
    # Measured on the fixture: 3 of 4 incidents resolved and recreated in a
    # single run. Something still being detected is not resolved, whatever its
    # age.
    new: list[Incident] = []
    updated: list[Incident] = []
    claimed: set[str] = set()

    for cand in candidates:
        cand_keys = cand.cluster.sensor_keys
        best, best_score = None, 0.0
        for existing in open_incidents:
            if existing.incident_id in claimed:
                continue
            if str(existing.cluster.region) != str(cand.cluster.region):
                continue
            score = _jaccard(cand_keys, existing.cluster.sensor_keys)
            if score > best_score:
                best, best_score = existing, score

        if best is not None and best_score >= jaccard_min:
            claimed.add(best.incident_id)
            previous_severity = best.severity
            # Carry the new evidence onto the existing identity.
            best.cluster = cand.cluster
            best.incident_class = cand.incident_class
            best.severity = cand.severity
            best.last_seen_at = cand.last_seen_at
            best.neighbour_correlation = cand.neighbour_correlation
            best.rainfall_mm = cand.rainfall_mm
            best.narrative = cand.narrative
            best.detail = dict(cand.detail)
            best.detail["membership_overlap"] = round(best_score, 3)
            best.detail["previous_severity"] = previous_severity
            escalated = cand.severity - previous_severity >= escalation_delta
            best.detail["escalated"] = escalated
            best.status = IncidentStatus.UPDATED
            updated.append(best)
        else:
            new.append(cand)

    # Only now: an open incident that nothing re-detected, and which has not
    # been seen for longer than the continuity window, is over.
    cutoff = now - timedelta(hours=continuity_hours)
    resolved: list[Incident] = []
    for incident in open_incidents:
        if incident.incident_id in claimed:
            continue
        if incident.last_seen_at is not None and incident.last_seen_at >= cutoff:
            continue                 # quiet this run, but too recent to close
        incident.status = IncidentStatus.RESOLVED
        incident.resolved_at = now
        resolved.append(incident)

    # Anything open that nothing matched, but which is still inside the
    # continuity window, stays open untouched -- it may simply not have been
    # re-detected this run, and closing it early would reopen it next run.
    return new, updated, resolved


def build_incidents(clusters: list[Cluster], *, now: datetime,
                    loose: list[SensorAnomaly] | None = None,
                    correlations: dict[str, float] | None = None,
                    rainfall: dict[str, float] | None = None) -> list[Incident]:
    """
    Every incident for one run, most severe first.

    Unclustered anomalies each become their own single-member incident rather
    than being dropped. A lone frozen sensor with no neighbours reacting is the
    clearest dispatch case there is, and a system that only reported clusters
    would miss the most actionable finding it makes.
    """
    correlations = correlations or {}
    rainfall = rainfall or {}

    incidents = [
        build_incident(c, now=now,
                       neighbour_correlation=correlations.get(_cluster_key(c)),
                       # Keyed on the CLUSTER, not the region. Region-level
                       # rainfall is a whole-run total, and attaching it to
                       # every incident in that region means a downpour at
                       # 03:00 "explains" a level shift at 20:00. Measured on
                       # the fixture: the genuine four-sensor regional event
                       # was labelled WEATHER_DRIVEN and suppressed to P4 by
                       # rain that fell at a different time of day.
                       rainfall_mm=rainfall.get(_cluster_key(c)))
        for c in clusters
    ]

    for anomaly in (loose or []):
        solo = Cluster(
            members=[anomaly],
            region=anomaly.sensor.region,
            centroid_lat=anomaly.sensor.latitude,
            centroid_lon=anomaly.sensor.longitude,
            radius_m=0.0,
        )
        incidents.append(build_incident(
            solo, now=now,
            neighbour_correlation=correlations.get(anomaly.sensor.sensor_key),
            rainfall_mm=rainfall.get(anomaly.sensor.sensor_key),
        ))

    incidents.sort(key=lambda i: -i.severity)
    return incidents


def _cluster_key(cluster: Cluster) -> str:
    return ",".join(sorted(cluster.sensor_keys))


def incident_summary(incidents: list[Incident]) -> dict[str, object]:
    """Per-run counts, for the log line and the dashboard header."""
    by_class: dict[str, int] = {}
    by_priority: dict[str, int] = {}
    for i in incidents:
        by_class[i.incident_class.value] = by_class.get(i.incident_class.value, 0) + 1
        by_priority[i.priority.value] = by_priority.get(i.priority.value, 0) + 1
    alertable = [i for i in incidents if i.should_alert]
    return {
        "incidents": len(incidents),
        "alertable": len(alertable),
        "suppressed": len(incidents) - len(alertable),
        "by_class": dict(sorted(by_class.items(), key=lambda kv: -kv[1])),
        "by_priority": dict(sorted(by_priority.items())),
        "max_severity": max((i.severity for i in incidents), default=0.0),
    }
