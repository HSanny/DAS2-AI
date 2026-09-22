"""
das2.detect.selection
=====================

Which incidents actually page someone.

Why a fixed top-N is the wrong shape
------------------------------------
The system being replaced ended every run with `MAX_ABNORMAL_SENSORS = 10`, a
global rank cut on `Peak_RZ`. Combined with a detector whose thresholds never
bound — the per-sensor quantile always won, so it was really "flag the top 0.3%
of points on every sensor" — that produced **about ten sensors per run whether
the network was healthy or on fire**. Two failures at once:

* On a quiet night it manufactured ten findings from noise. That is why the
  heartbeat feature exists at all: "zero anomalies" was a rare state rather
  than the normal one, so silence had to be explained separately.
* On a bad night it silently discarded everything past the tenth. A genuine
  ten-site regional event would have been truncated with nothing saying so.

And it ranked by `Peak_RZ`, which is not comparable between sensors: a 1.8%
voltage excursion scored 109.7 because that channel's MAD was pinned at its
0.1 V quantisation step, while a full-scale flow event scored 306.9. The cut
was effectively sorting by each sensor's quantisation step.

What replaces it
----------------
A budget, not a rank cut, applied per region and with severity able to override
it:

1. **P1 always goes.** A budget that can suppress the worst thing happening is
    not a budget, it is a bug waiting to be discovered at the worst moment.
2. **Each region gets its own allowance.** A bad night in the West cannot
    crowd out the East's one real problem — under a global cut it silently
    would, and the operator would have no way to know.
3. **A global cap remains as a backstop**, so a pathological run cannot flood
    the chat.
4. **Nothing is ever discarded.** Everything not selected stays on the
    dashboard and in the database, and the run summary counts it. The
    difference between "suppressed" and "deleted" is the difference between a
    system you can audit and one you cannot.

Quiet means quiet
-----------------
When nothing clears the bar, this selects nothing, and that is a real state
rather than an embarrassment to be padded out. Alert volume is the first gate
the client will judge this on: a system emitting four hundred alerts a day is
dead whatever its accuracy, and one emitting ten every night regardless of
reality teaches its operators to ignore it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from das2.models import Incident, IncidentClass, Priority

#: Priorities in descending urgency, for ordering and comparison.
PRIORITY_ORDER = (Priority.P1, Priority.P2, Priority.P3, Priority.P4)
_PRIORITY_RANK = {p: i for i, p in enumerate(PRIORITY_ORDER)}

#: Never held back by a budget. If the system cannot get its most severe
#: finding out, nothing else it does matters.
ALWAYS_ALERT = (Priority.P1,)

#: Per-region allowance per run.
DEFAULT_PER_REGION = 5

#: Absolute backstop across all regions.
DEFAULT_GLOBAL_CAP = 10

#: Nothing below this pages anyone, whatever the budget allows.
DEFAULT_MIN_PRIORITY = Priority.P3


@dataclass
class SelectionResult:
    selected: list[Incident] = field(default_factory=list)
    held: list[tuple[Incident, str]] = field(default_factory=list)

    @property
    def held_incidents(self) -> list[Incident]:
        return [incident for incident, _ in self.held]

    def summary(self) -> dict[str, object]:
        reasons: dict[str, int] = {}
        for _, reason in self.held:
            reasons[reason] = reasons.get(reason, 0) + 1
        by_region: dict[str, int] = {}
        for incident in self.selected:
            region = str(incident.cluster.region or "Unplaced")
            by_region[region] = by_region.get(region, 0) + 1
        return {
            "selected": len(self.selected),
            "held": len(self.held),
            "by_region": dict(sorted(by_region.items())),
            "held_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        }


def _rank(incident: Incident) -> tuple[int, float]:
    return (_PRIORITY_RANK.get(incident.priority, len(PRIORITY_ORDER)),
            -incident.severity)


def select(incidents: list[Incident], *,
           per_region: int = DEFAULT_PER_REGION,
           global_cap: int = DEFAULT_GLOBAL_CAP,
           min_priority: Priority = DEFAULT_MIN_PRIORITY) -> SelectionResult:
    """
    Decide what to send, and record why everything else was held.

    The reason strings are not decoration: they are what lets an operator ask
    "why did nobody tell me about the East?" and get an answer from the run
    record rather than from reading this file.
    """
    result = SelectionResult()
    min_rank = _PRIORITY_RANK.get(min_priority, len(PRIORITY_ORDER))

    # Class suppression first. A fan-out or a WATCH is not competing for budget
    # -- it is not a candidate at all, and counting it against a region's
    # allowance would let noise crowd out a real finding.
    candidates: list[Incident] = []
    for incident in sorted(incidents, key=_rank):
        if not incident.should_alert:
            reason = ("telemetry fan-out, not a site visit"
                      if incident.incident_class is IncidentClass.TELEMETRY_FANOUT
                      else "evidence too weak or conflicting")
            result.held.append((incident, reason))
        elif _PRIORITY_RANK.get(incident.priority, 99) > min_rank:
            result.held.append(
                (incident, f"below the {min_priority.value} alerting threshold"))
        else:
            candidates.append(incident)

    used_per_region: dict[str, int] = {}
    for incident in candidates:
        region = str(incident.cluster.region or "Unplaced")

        if incident.priority in ALWAYS_ALERT:
            # Deliberately bypasses both budgets. See ALWAYS_ALERT.
            result.selected.append(incident)
            used_per_region[region] = used_per_region.get(region, 0) + 1
            continue

        if len(result.selected) >= global_cap:
            result.held.append(
                (incident, f"run cap of {global_cap} reached; on the dashboard"))
            continue

        if used_per_region.get(region, 0) >= per_region:
            result.held.append(
                (incident,
                 f"{region} already has {per_region} this run; on the dashboard"))
            continue

        result.selected.append(incident)
        used_per_region[region] = used_per_region.get(region, 0) + 1

    return result
