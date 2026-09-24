"""
das2.incident.parameters
========================

Which parameter is driving an incident, and which way it moved.

The gap this closes
-------------------
An incident used to read::

    P1  REGIONAL_EVENT  East  12 sensor(s)  Bedok Diversion, Bedok PS, ...
        -> Multiple sites affected together - investigate the area.

That says WHERE and HOW CONFIDENT, and never WHAT. It does not say whether
those twelve sensors are canal levels, flows, pressures or pump motors, so it
cannot say what kind of physical event it is, and the operator still has to
open the raw data to find out. "Investigate the area" is not a dispatch
instruction; it is an admission that the system stopped one step short.

Everything needed was already on the objects -- `sensor.equipment` is the
parameter, `severity.deviation` the size, `dominant_type` the behaviour -- and
none of it reached the report.

Direction is the piece that was genuinely missing
-------------------------------------------------
`LEVEL_SHIFT` says a step happened; it never said which way, because fusion
took `abs()` of the magnitude. Without the sign the combinations that carry
meaning collapse into one:

    level rising  +  flow rising   ->  the drainage system responding to rain
    level rising  +  flow falling  ->  something obstructing the channel

Same parameters, same magnitudes, opposite verdicts and opposite decisions.
`PhysicalSeverity.signed_deviation` now carries it.

What this module deliberately does NOT do
-----------------------------------------
It does not name the event. Turning "levels up, flows up, raining" into
"stormwater response" is an inference, the rules for it would be ours rather
than anything validated, and a wrong confident label sends a crew looking for
the wrong thing. This module reports the measured facts -- parameter, count,
direction, typical move, QARTOD flag -- and leaves the naming to a later layer
that can be argued with in a config file.

One honest limit on the direction consensus: a parameter whose sensors
disagree about direction is reported as `mixed`, not resolved by majority. Four
levels up and three down in one cluster is a fact worth seeing, and averaging
it away would manufacture agreement that is not there.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from statistics import median

from das2.models import (
    AnomalyType,
    QartodFlag,
    SensorAnomaly,
    aggregate_flags,
    flag_for,
)

#: Below this share of a parameter's sensors agreeing, call it mixed.
#: Two thirds, so 2-of-3 agrees and 2-of-4 does not.
DIRECTION_CONSENSUS = 0.66


def display_name(parameter: str) -> str:
    """
    `'Canal Level'` from `'CanalLevel'`.

    Left alone when it is not camel case, so UNCLASSIFIED does not come out
    as U N C L A S S I F I E D.
    """
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", parameter)


@dataclass
class ParameterGroup:
    """One parameter's contribution to an incident."""

    parameter: str
    members: list[SensorAnomaly] = field(default_factory=list)

    # -- what it is ------------------------------------------------------- #
    @property
    def count(self) -> int:
        return len(self.members)

    @property
    def sites(self) -> set[str]:
        return {m.sensor.site for m in self.members if m.sensor.site}

    @property
    def display(self) -> str:
        return display_name(self.parameter)

    @property
    def unit(self) -> str:
        return next((m.severity.unit for m in self.members
                     if m.severity.unit), "")

    # -- what it did ------------------------------------------------------ #
    @property
    def direction(self) -> str:
        """`'rising'`, `'falling'`, `'mixed'`, or `''` if none has a direction."""
        directions = [m.severity.direction for m in self.members
                      if m.severity.direction]
        if not directions:
            return ""
        rising = directions.count("rising")
        share = max(rising, len(directions) - rising) / len(directions)
        if share < DIRECTION_CONSENSUS:
            return "mixed"
        return "rising" if rising * 2 > len(directions) else "falling"

    @property
    def typical_move(self) -> float:
        """
        Median signed deviation. The median, not the mean, because one sensor
        reading full-scale after a transmitter failure would otherwise set the
        number the whole group is described by.
        """
        moves = [m.severity.signed_deviation for m in self.members
                 if m.severity.signed_deviation]
        return median(moves) if moves else 0.0

    @property
    def behaviours(self) -> list[AnomalyType]:
        """The distinct dominant types, most common first."""
        counts: dict[AnomalyType, int] = {}
        for m in self.members:
            counts[m.dominant_type] = counts.get(m.dominant_type, 0) + 1
        return [t for t, _ in sorted(counts.items(), key=lambda kv: -kv[1])]

    @property
    def flag(self) -> QartodFlag:
        """The worst QARTOD disposition among the members."""
        return aggregate_flags(flag_for(m.dominant_type) for m in self.members)

    # -- how it reads ----------------------------------------------------- #
    @property
    def arrow(self) -> str:
        return {"rising": "▲", "falling": "▼",
                "mixed": "↕"}.get(self.direction, "—")

    def move_text(self) -> str:
        """
        `'+0.82 m'`, or `'±0.82 m'` when the sensors moved opposite ways.

        A mixed group needs the second form. Its median signed deviation is
        near zero by construction -- that is what mixed means -- and printing
        a dash there would say "nothing moved" about a parameter where half
        the sensors went up and half went down. The magnitude is real even
        when the direction does not resolve.
        """
        unit = f" {self.unit}" if self.unit else ""
        if self.direction == "mixed":
            sizes = [abs(m.severity.signed_deviation) for m in self.members
                     if m.severity.signed_deviation]
            return f"\u00b1{median(sizes):.4g}{unit}" if sizes else ""
        move = self.typical_move
        if not move:
            return ""
        return f"{move:+.4g}{unit}"

    def direction_text(self) -> str:
        """Words, never the arrow alone -- the glyph is a second channel."""
        if not self.direction:
            return "no direction"
        if self.direction == "mixed":
            return f"mixed ({self.count} sensors disagree)"
        return f"all {self.direction}" if self.count > 1 else self.direction

    def summary(self) -> str:
        """`'Canal level 7▲'` -- for an inline column in a table."""
        return f"{self.display} {self.count}{self.arrow}"


def breakdown(incident) -> list[ParameterGroup]:
    """
    An incident's members grouped by parameter, biggest group first.

    Ties break on total severity rather than alphabetically, so the parameter
    doing the most work sorts above one that merely has as many sensors.
    """
    groups: dict[str, ParameterGroup] = {}
    for member in incident.cluster.members:
        name = member.sensor.equipment or "UNCLASSIFIED"
        groups.setdefault(name, ParameterGroup(name)).members.append(member)

    return sorted(
        groups.values(),
        key=lambda g: (-g.count, -sum(m.score for m in g.members), g.parameter),
    )


#: How an asset finding reads in one column.
ASSET_PHRASE = {
    "ASSET_NOT_DELIVERING": "not delivering",
    "ASSET_ENERGISED_WHEN_OFF": "energised while off",
    "ASSET_NOT_ENERGISED_WHEN_ON": "drawing nothing",
}


def asset_summary(incident) -> str:
    """
    `'TampinesPS unit 1 — not delivering'`, or `''` when it is not one.

    A machine failure has no useful parameter breakdown. Grouped by equipment
    class it reads "Digital Status 1", because the finding is raised on the
    run-state bit -- which names the least interesting of the three channels
    involved and tells an operator nothing. The unit and what it is doing is
    the answer to "which parameter is triggering this", for this class.
    """
    from das2.models import ASSET_TYPES

    for member in incident.cluster.members:
        if member.dominant_type not in ASSET_TYPES:
            continue
        name = next((str(s.detail.get("asset")) for s in member.signals
                     if s.detail.get("asset")), "")
        phrase = ASSET_PHRASE.get(member.dominant_type.value, "faulted")
        return f"{name or member.sensor.site} — {phrase}"
    return ""


def inline_summary(incident, limit: int = 3) -> str:
    """
    `'Canal level 7▲, Flowrate 3▲, Pressure 2▼'`.

    For the column in an incident table, where the full breakdown will not
    fit but "which parameter" still has to be answerable without a page turn.
    """
    asset = asset_summary(incident)
    if asset:
        return asset
    groups = breakdown(incident)
    shown = ", ".join(g.summary() for g in groups[:limit])
    if len(groups) > limit:
        shown += f" +{len(groups) - limit}"
    return shown


def region_parameter_matrix(anomalies: list[SensorAnomaly]
                            ) -> dict[str, dict[str, dict[str, int]]]:
    """
    `{region: {parameter: {'rising': n, 'falling': n, 'flat': n}}}`.

    The whole-run view. The existing region-by-type matrix answers "the East
    has 31 level findings"; this answers "31, and 28 of them rising", which is
    the difference between a count and a direction of travel. `flat` counts
    findings with no direction at all -- a stale or flatlined sensor has not
    moved either way, and folding those into one of the two would invent a
    movement that did not happen.
    """
    matrix: dict[str, dict[str, dict[str, int]]] = {}
    for a in anomalies:
        region = a.sensor.region or "Unknown"
        parameter = a.sensor.equipment or "UNCLASSIFIED"
        cell = matrix.setdefault(region, {}).setdefault(
            parameter, {"rising": 0, "falling": 0, "flat": 0})
        cell[a.severity.direction or "flat"] += 1
    return matrix


def net_direction(cell: dict[str, int]) -> int:
    """Rising minus falling. The signed value a diverging scale needs."""
    return cell.get("rising", 0) - cell.get("falling", 0)
