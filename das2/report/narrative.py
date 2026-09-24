"""
das2.report.narrative
=====================

The run, in sentences.

Every other part of this report is a number, a bar or a dot. Those answer
*what*; none of them answers *so what*, and the client asked for "an
explanation on the case or this round of analysis" -- which is a different
thing from a chart of it.

Deterministic, not generated
----------------------------
Templates over an LLM, deliberately. This text is the part a duty engineer
will quote when they justify sending a crew out at 2 a.m., so it has to say
only what the run actually measured, say the same thing every time the numbers
are the same, and be reviewable line by line by someone who does not trust it
yet. A model that paraphrases well would also, occasionally, assert something
the data does not support, and there is no way to tell which sentence that was
after the fact.

The rules it follows
--------------------
* **Lead with the decision.** The first paragraph says whether anyone needs to
  act and where, because on most runs that is all that gets read.
* **Name the mechanism, not the label.** "Four sites moved together and the
  neighbouring sensors moved with them" is actionable; "REGIONAL_EVENT" is a
  class name.
* **Say what was NOT concluded.** A run with no baselines, or a 24-hour hole in
  the feed, is a run whose silence means less than it appears to. Those
  caveats travel in the same paragraph as the findings they weaken, never in a
  footnote nobody reads.
* **No adjectives the data cannot carry.** Nothing is "critical" or "alarming"
  here; it is P1, or it is 12 sensors across 4 sites.
"""

from __future__ import annotations

from typing import Any


def _plural(n: int, one: str, many: str | None = None) -> str:
    return one if n == 1 else (many or one + "s")


def _join(items: list[str]) -> str:
    """Oxford-comma join, because "North, East and West" is read aloud."""
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _region_of(incident) -> str | None:
    region = incident.cluster.region
    if region is None:
        return None
    return str(getattr(region, "value", region))


def headline(result) -> str:
    """One sentence: does anyone need to do anything, and where."""
    urgent = [i for i in result.incidents if i.priority.value in ("P1", "P2")]
    if not urgent:
        total = len(result.incidents)
        if not total:
            return ("Nothing was detected this run. Every sensor the system "
                    "could read behaved within its own history.")
        return (f"Nothing needs a decision now. {total} "
                f"{_plural(total, 'incident')} {_plural(total, 'is', 'are')} "
                f"open at P3 or below, which means schedule or log, not "
                f"dispatch.")

    regions: dict[str, int] = {}
    for incident in urgent:
        name = _region_of(incident)
        if name:
            regions[name] = regions.get(name, 0) + 1
    where = _join([f"{n} in the {r}" for r, n in
                   sorted(regions.items(), key=lambda kv: -kv[1])])
    p1 = sum(1 for i in urgent if i.priority.value == "P1")
    lead = (f"{len(urgent)} {_plural(len(urgent), 'incident')} "
            f"{_plural(len(urgent), 'needs', 'need')} a decision now")
    if p1:
        lead += f", {p1} of them at P1"
    return f"{lead}" + (f" — {where}." if where else ".")


def evidence(result) -> str:
    """What the run's dominant verdicts mean in plain terms."""
    counts: dict[str, int] = {}
    for incident in result.incidents:
        name = incident.incident_class.value
        counts[name] = counts.get(name, 0) + 1

    # Phrased as the mechanism, because the class name is a label and the
    # mechanism is what tells a reader whether to believe it.
    MEANING = {
        "REGIONAL_EVENT": ("several sites moved together and their neighbours "
                           "moved with them, which is water, not instruments"),
        "SENSOR_FAULT": ("the sensor moved and nothing around it did, which is "
                         "the instrument, not the water"),
        "TELEMETRY_FANOUT": ("sensors on one RTU went at the same instant, "
                             "which is a telemetry path, not a site"),
        "TELEMETRY_OUTAGE": ("a block of sensors across separate sites stopped "
                             "reporting together, which is the link or the "
                             "historian feed"),
        "WEATHER_DRIVEN": ("levels and flows rose while it was raining on the "
                           "same catchment"),
        "PROCESS_EVENT": ("a level shift the neighbours corroborate — "
                          "operational, worth watching, not a fault"),
        "DRIFT_MAINTENANCE": ("a slow drift with no abrupt failure, which is a "
                              "calibration job rather than a callout"),
        "INSTRUMENT_CONFLICT": ("readings that cannot all be true at once — "
                                "level, inflow and outflow disagree"),
        "ASSET_FAILURE": ("a machine whose own channels contradict each "
                          "other while every instrument on it reports "
                          "normally — the plant, not the telemetry"),
        "WATCH": ("evidence too weak or too mixed to act on yet"),
    }
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:3]
    if not top:
        return ""
    parts = [f"{n} where {MEANING[name]}" if name in MEANING
             else f"{n} classed {name}"
             for name, n in top]
    return ("Of the open incidents, " + _join(parts) + ". "
            "That split is the whole point of the triage step: it is what "
            "separates a trip worth making from one that is not.")


def driving_parameters(result) -> str:
    """
    Which parameters are actually moving, and which way.

    The explanation was answering "how many" and "where" and never "what of".
    A reader could finish the page knowing eighteen incidents needed a
    decision without learning whether the estate's canal levels or its pump
    motors were the thing misbehaving -- which is the first question anyone
    asks before deciding who to send.
    """
    from das2.incident.parameters import display_name, net_direction, \
        region_parameter_matrix

    matrix = region_parameter_matrix(getattr(result, "anomalies", []) or [])
    if not matrix:
        return ""

    totals: dict[str, dict[str, int]] = {}
    for row in matrix.values():
        for parameter, cell in row.items():
            acc = totals.setdefault(
                parameter, {"rising": 0, "falling": 0, "flat": 0})
            for key, n in cell.items():
                acc[key] += n

    ranked = sorted(totals.items(), key=lambda kv: -sum(kv[1].values()))[:3]
    if not ranked:
        return ""

    parts = []
    for parameter, cell in ranked:
        total = sum(cell.values())
        net = net_direction(cell)
        moving = cell["rising"] + cell["falling"]
        if not moving:
            way = "none of them moving in either direction"
        elif abs(net) < moving * 0.34:
            way = f"{cell['rising']} rising and {cell['falling']} falling"
        else:
            way = (f"{max(cell['rising'], cell['falling'])} of them "
                   f"{'rising' if net > 0 else 'falling'}")
        parts.append(f"{total} on {display_name(parameter).lower()}, {way}")

    text = "The parameters carrying this run are " + _join(parts) + "."
    if any(abs(net_direction(c)) >= sum(c.values()) * 0.66
           for _, c in ranked if sum(c.values()) >= 4):
        text += (" A parameter whose sensors nearly all move the same way is "
                 "the network responding to something, not instruments "
                 "failing independently.")
    return text


def movement(result) -> str:
    """What changed since the last run."""
    lifecycle = result.stats.get("lifecycle") or {}
    new = lifecycle.get("new", 0)
    updated = lifecycle.get("updated", 0)
    resolved = lifecycle.get("resolved", 0)
    if not (new or updated or resolved):
        return ""
    text = (f"{new} {_plural(new, 'incident')} "
            f"{_plural(new, 'is', 'are')} new since the last run, "
            f"{updated} carried over and {resolved} closed.")
    if updated > new:
        text += (" Most of what is open was already open: the same fault is "
                 "not being re-announced, it is being tracked.")
    return text


def suppression(result) -> str:
    """Why the number sent is smaller than the number found."""
    selection = result.stats.get("selection") or {}
    held = selection.get("held", 0)
    if not held:
        return ""
    reasons = selection.get("held_reasons") or {}
    top = sorted(reasons.items(), key=lambda kv: -kv[1])[:2]
    detail = _join([f"{n} as {reason}" for reason, n in top])
    return (f"{held} {_plural(held, 'incident')} "
            f"{_plural(held, 'was', 'were')} deliberately held back"
            + (f" — {detail}." if detail else ".")
            + " Every one of them is listed later in this report with its "
              "reason, so silence here is a decision you can audit rather "
              "than an absence you have to trust.")


def caveats(result) -> str:
    """
    What would make these numbers wrong, stated beside them.

    This paragraph is the one most likely to be cut and the one least safe to
    cut. A 24-hour hole in the feed inflates STALE across the fleet, and a
    reader who does not know that reads an instrument-fault count that is
    mostly missing files.
    """
    stats = result.stats
    ingest = stats.get("ingest") or {}
    baselines = stats.get("baselines") or {}
    coverage = stats.get("coverage") or {}
    notes: list[str] = []

    missing = ingest.get("missing_hours", 0)
    if missing:
        notes.append(
            f"{missing} of the window's hourly files were missing "
            f"({ingest.get('missing_hours_range', 'see the coverage page')}). "
            f"A sensor that is silent across that gap reads as STALE, so the "
            f"instrument-fault count is inflated while this persists and is "
            f"not evidence of a fleet-wide failure")

    if not baselines.get("usable", 0):
        notes.append(
            "no time-of-day baselines are stored yet, so the layer that "
            "scores a reading against what that sensor normally does at this "
            "hour abstained entirely. DRIFT and NOISE_BURST produce nothing "
            "until the daily profile job has run over a fortnight of history")

    without = ingest.get("sensors_without_coords", 0)
    if without:
        notes.append(
            f"{without:,} sensors have no coordinates, so they cannot appear "
            f"on the map or be grouped with their neighbours; a fault on one "
            f"of them is still detected but arrives without a place")

    unclassified = coverage.get("unclassified", 0)
    if unclassified:
        notes.append(
            f"{unclassified:,} sensors are still UNCLASSIFIED — analysed, "
            f"visible in the counts, never alerted on")

    if not notes:
        return ""
    # One sentence each, not one joined clause. Strung together with commas
    # and "and", four caveats of this length become a single sixty-word
    # sentence that a reader skips -- which defeats the point of putting the
    # limits beside the findings rather than in a footnote.
    return ("Read the above knowing what this run could not see. "
            + " ".join(n[0].upper() + n[1:] + "." for n in notes))


def paragraphs(result) -> list[str]:
    """The whole explanation, in reading order. Empty entries are dropped."""
    return [p for p in (headline(result), driving_parameters(result),
                        evidence(result), movement(result),
                        suppression(result), caveats(result)) if p]


def summary_line(result) -> str:
    """One line, for a Telegram caption or a log."""
    return headline(result)
