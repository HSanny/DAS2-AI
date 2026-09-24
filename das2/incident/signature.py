"""
das2.incident.signature
=======================

What the moving parameters usually mean, in a sentence.

The layer above the breakdown. `parameters.py` reports that seven canal levels
rose 0.82 m while three flows rose and two pressures fell; this says what a
water engineer would call that, and what would change their mind.

It describes, it does not decide
--------------------------------
The incident CLASS -- REGIONAL_EVENT, SENSOR_FAULT, TELEMETRY_FANOUT -- drives
priority and dispatch, and is computed from evidence the system can measure:
how many sites, how many parameters, whether the neighbours moved, what the
rain did. A signature sits beside that verdict and explains it. It never
suppresses an alert and never raises one.

That separation is the whole safety argument. A wrong evidence class costs a
wasted trip; a wrong signature with the power to stop a dispatch costs a
flood. Until these rules have been checked against real incidents they get to
explain and not to decide, and `applies_to_decision` is deliberately absent
from this module's interface.

The rules are not mine to own
-----------------------------
They live in `das2/data/event_signatures.yaml`, which a PUB engineer can edit
without touching code. This system has no labelled incidents, no hydraulic
model and no catchment map, so the rules cannot be derived from data -- they
are hydraulic reasoning, and the people who know whether they hold on this
network are the people operating it. Their corrections are the ground truth
the system does not otherwise have, and the file says so in its own header.

Every match carries its status
------------------------------
`supported` means a published method backs the reading. `unvalidated` means
plausible hydraulics, unchecked here -- which is most of them, and the report
hedges accordingly. `confirmed` means PUB has agreed, and nothing ships as
that. A signature that cannot be falsified is not a finding, so
`would_change_it` is required on every entry and travels with every match.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from das2.incident.parameters import ParameterGroup, breakdown

log = logging.getLogger("das2.incident.signature")

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "event_signatures.yaml"

#: How the report introduces a match, by status. The hedge is not politeness:
#: an unvalidated rule asserted flatly is how a system loses the argument the
#: first time it is wrong.
LEAD_IN = {
    "supported": "Reads as",
    "unvalidated": "Looks like",
    "confirmed": "This is",
}


@dataclass(frozen=True)
class Signature:
    """One rule from the table."""

    id: str
    name: str
    status: str = "unvalidated"
    when: dict[str, Any] = field(default_factory=dict)
    reads_as: str = ""
    action: str = ""
    would_change_it: str = ""
    basis: str = ""

    @property
    def validated(self) -> bool:
        return self.status == "confirmed"


@dataclass(frozen=True)
class SignatureMatch:
    """A signature, the incident it matched, and why."""

    signature: Signature
    matched: list[str] = field(default_factory=list)

    @property
    def headline(self) -> str:
        lead = LEAD_IN.get(self.signature.status, "Looks like")
        return f"{lead}: {self.signature.name.lower()}"

    @property
    def caveat(self) -> str:
        if self.signature.status == "confirmed":
            return ""
        if self.signature.status == "supported":
            return "reading backed by published method; not yet checked here"
        return "unvalidated reading — describes, does not decide"


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())


@lru_cache(maxsize=4)
def load(path: str | None = None) -> tuple[Signature, ...]:
    """The signature table, in file order. First match wins, so order matters."""
    target = Path(path) if path else DEFAULT_PATH
    try:
        import yaml
    except ImportError:                                    # pragma: no cover
        log.warning("PyYAML is absent; no event signatures will be reported")
        return ()
    try:
        data = yaml.safe_load(target.read_text()) or {}
    except (OSError, Exception) as exc:                    # noqa: BLE001
        log.warning("could not read %s (%s); no signatures", target, exc)
        return ()

    out: list[Signature] = []
    for entry in data.get("signatures") or []:
        if not entry.get("id") or not entry.get("would_change_it"):
            # A rule with no falsifier is not a finding. Refusing it here
            # rather than rendering it is what keeps that rule enforceable
            # once the file is being edited by someone else.
            log.warning("signature %r skipped: needs an id and a "
                        "would_change_it", entry.get("id") or entry.get("name"))
            continue
        out.append(Signature(
            id=str(entry["id"]),
            name=str(entry.get("name") or entry["id"]),
            status=str(entry.get("status") or "unvalidated"),
            when=entry.get("when") or {},
            reads_as=_text(entry.get("reads_as")),
            action=_text(entry.get("action")),
            would_change_it=_text(entry.get("would_change_it")),
            basis=_text(entry.get("basis")),
        ))
    return tuple(out)


def _clause_matches(clause: dict[str, Any],
                    groups: list[ParameterGroup]) -> str | None:
    """`'CanalLevel 7 rising'` when the clause is satisfied, else `None`."""
    wanted = {str(name) for name in (clause.get("any_of") or [])}
    direction = str(clause.get("direction") or "any")
    minimum = int(clause.get("min_sensors") or 1)

    hits = [g for g in groups if g.parameter in wanted
            and (direction == "any" or g.direction == direction)]
    total = sum(g.count for g in hits)
    if total < minimum:
        return None
    names = "/".join(sorted({g.display for g in hits}))
    way = "" if direction == "any" else f" {direction}"
    return f"{names} {total}{way}"


def match(incident, *, rain_context: str = "unknown",
          path: str | None = None) -> SignatureMatch | None:
    """
    The first signature this incident satisfies, or `None`.

    `None` is a perfectly good answer and the common one. An incident with no
    matching signature keeps its evidence class and its recommendation; the
    report simply has one fewer sentence to offer. Inventing a reading to
    avoid an empty field is how a table of rules becomes a horoscope.
    """
    groups = breakdown(incident)
    if not groups:
        return None

    sites = len(incident.cluster.sites)

    for signature in load(path):
        when = signature.when or {}

        wanted_rain = str(when.get("rain_context") or "any")
        if wanted_rain != "any" and wanted_rain != rain_context:
            continue

        if sites < int(when.get("min_sites") or 0):
            continue

        matched: list[str] = []
        for clause in when.get("parameters") or []:
            hit = _clause_matches(clause, groups)
            if hit is None:
                matched = []
                break
            matched.append(hit)
        if not matched:
            continue

        if wanted_rain != "any":
            matched.append(f"{wanted_rain} conditions")
        return SignatureMatch(signature=signature, matched=matched)

    return None


#: The key a match is filed under on `Incident.detail`. One name, in one place,
#: because three modules read it and a typo in any of them fails silently as
#: "this incident had no signature".
DETAIL_KEY = "signature"


def as_detail(found: SignatureMatch) -> dict[str, Any]:
    """
    The match, flattened for `Incident.detail` -- and for the database.

    `would_change_it` travels with it deliberately. A reading whose falsifier
    is left behind in the config file is a reading the operator cannot check,
    and the whole argument for showing an unvalidated rule at all is that the
    person looking at it can see what would make it wrong.
    """
    return {
        "id": found.signature.id,
        "name": found.signature.name,
        "status": found.signature.status,
        "headline": found.headline,
        "reads_as": found.signature.reads_as,
        "action": found.signature.action,
        "would_change_it": found.signature.would_change_it,
        "caveat": found.caveat,
        "matched": list(found.matched),
    }


def attached(incident) -> dict[str, Any]:
    """The signature already filed on an incident, or `{}`."""
    return dict((getattr(incident, "detail", None) or {}).get(DETAIL_KEY) or {})


def summary(matches: list[SignatureMatch | None]) -> dict[str, Any]:
    """Run-level counts, for the log line and the report."""
    counts: dict[str, int] = {}
    by_status: dict[str, int] = {}
    unmatched = 0
    for m in matches:
        if m is None:
            unmatched += 1
            continue
        counts[m.signature.id] = counts.get(m.signature.id, 0) + 1
        by_status[m.signature.status] = by_status.get(m.signature.status, 0) + 1
    return {
        "matched": sum(counts.values()),
        "unmatched": unmatched,
        "by_signature": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "by_status": by_status,
    }
