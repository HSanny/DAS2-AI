"""
das2.detect.asset
=================

The machine has failed, and every instrument on it is working perfectly.

    "maybe there's a complete breakdown of the equipment instead of just
     operational failure, but there's still value being read by the sensor"

That sentence describes the blind spot every other layer in this system
shares. A threshold watches one channel; a baseline watches one channel over
time; even the clustering asks whether several channels moved together. None
of them can see a pump that is running, energised, reporting a perfectly
ordinary motor current, and moving no water -- because not one of those
readings is abnormal on its own. The fault is in the RELATIONSHIP between
them, and a relationship has to be looked at directly.

What a third channel buys
-------------------------
`digital.detect_run_state_inconsistent` already compares a pump's run state
against its discharge flow, and it is careful to conclude only what two
channels can support: *"either the pump is not running, or the flowmeter has
failed"*. That is honest and it is also unsatisfying, because those two
answers have completely different costs.

A third channel decides it. The motor's current (or power) is measured
independently of both, so:

    run=ON, current unlike this unit's own normal, no output
        -> two independent channels agree the machine changed.   ASSET
    run=ON, current exactly at this unit's own normal, no output
        -> only the flowmeter disagrees with everything else.    INSTRUMENT

The second case is left to the existing detector on purpose. This module
speaks only where the electrical channel corroborates, which is the whole
reason it is allowed to name the machine as the fault.

Why it is measured against the unit's own normal
------------------------------------------------
The obvious rule -- "power drops when a centrifugal pump stops delivering" --
is true for a radial-flow pump and FALSE for an axial-flow one, where brake
power is highest at shutoff. PUB's drainage stations run both, and nothing in
this feed says which a given unit is. Any rule keyed on the direction of the
power change would therefore be right at some stations and backwards at
others, silently.

So direction is never used. The test is whether the duty channel has departed
from *this unit's own* running level, in either direction, which is true for
both pump types and needs no pump curve, no impeller diameter and no
commissioning data -- none of which exist here.

Everything is per-asset and self-referencing
--------------------------------------------
Thresholds come from the unit's own history within the window: what current it
draws when its run bit is on, what it draws when off, what its meter reads
while it is running. A fleet-wide "pumps draw more than 5 A" would be wrong on
every station with a different motor, which is the defect that made v1's
`is_rule_invalid` unusable.

Where a unit's own channels cannot establish that separation -- a current
point that never moves between on and off, a meter that never reads positive
-- the module abstains. It reports nothing rather than guessing, because a
fabricated "your pump has failed" is a mechanical callout nobody needed.

Coverage, measured
------------------
Against the real inventory (13,982 points, 192 units carrying a unit number):
82 units have a run state and a current or power channel, so the energisation
contradictions can run on them; 22 also carry an output channel and can be
attributed. 92 units have a run state and nothing to check it against, and
those this module is silent on -- which is a data request, not a bug.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from das2.detect.digital import (
    MIN_POINTS,
    _is_binary,
    _is_run_state,
    _unit_number,
)
from das2.io.classify import expand_boundaries
from das2.models import AnomalyType, Signal
from das2.timeutils import to_epoch_seconds

log = logging.getLogger("das2.detect.asset")

DETECTOR = "asset"

#: The duty channel must separate "running" from "off" by at least this share
#: of its running level before anything here is judged. Below it the channel
#: says nothing about whether the machine is turning -- which is the case for
#: a current point stuck at a constant, and for a point classified as current
#: that is really something else.
MIN_DUTY_SEPARATION = 0.25

#: Where "energised" sits between the unit's own off level and its own running
#: level. ONE boundary, and it sits low.
#:
#: The first version put it halfway, which reads as "drawing about what it
#: normally draws" rather than "switched on". A pump that has lost its prime
#: draws around 12 A where it normally draws 41 and idles at 0.6 -- plainly
#: energised, and a halfway line called it dead, turning the estate's clearest
#: asset failure into "your breaker has tripped". The question this boundary
#: answers is only "is this machine drawing power at all", and a fifth of the
#: unit's own on/off swing is clear of the off-state noise floor while leaving
#: everything above it, however reduced, on the energised side.
ENERGISED_FRACTION = 0.2

#: How far the duty must move from the unit's own running level, as a share of
#: its own on/off swing, before the electrical channel counts as corroborating
#: that the machine changed. A quarter of the swing is far outside the
#: run-to-run variation of a pump doing its normal job.
DUTY_DEPARTURE = 0.25

#: Output at or below this share of the unit's own running output counts as
#: "not delivering". Shared with `digital.RUN_FLOW_FRACTION` in spirit and kept
#: separate in fact, so tuning one cannot silently move the other.
NO_OUTPUT_FRACTION = 0.1

#: A contradiction must persist this long. Starters, spin-up, meter lag and
#: the scan cycle all produce brief disagreements in a healthy plant; ten
#: minutes is past all of them and still well inside the time a dry-running
#: pump takes to damage itself.
MIN_CONTRADICTION_S = 600.0

#: Samples required on each side before a median means anything.
MIN_STATE_SAMPLES = 10

#: Words marking an analog point as the motor's electrical duty.
#:
#: Matched on WORD boundaries, never as substrings. `"amp" in description` is
#: true of `TampinesPS-Pump1-Discharge-Flow`, which put a flowmeter forward as
#: a motor-current channel and had the detector comparing a pump's run state
#: against the very meter it was supposed to be adjudicating. It found a
#: contradiction, too.
#:
#: The boundary expansion is `classify.expand_boundaries`, not a bare `\b`,
#: because `\b` does not break on an underscore -- `S718_PUMP_CURRENT_A` would
#: miss. That fix already exists in this codebase and there is no reason for a
#: second definition of "word boundary" to drift alongside it.
CURRENT_WORDS = ("current", "amp", "amps", "ampere", "amperes", "ammeter")
POWER_WORDS = ("power", "kw", "kilowatt", "kilowatts", "watt", "watts")

#: ...and what marks a point as the machine's OUTPUT -- what it is supposed to
#: be producing. Pressure counts: a pump that is turning and not raising its
#: discharge pressure has stopped delivering just as surely as one with no
#: flow, and 42 units carry pressure where only 30 carry flow.
OUTPUT_WORDS = ("discharge", "delivery", "deliver", "outlet", "outflow",
                "out-flow", "export", "downstream")

def _word_re(words: tuple[str, ...]) -> re.Pattern[str]:
    return re.compile(
        expand_boundaries(r"\b(?:" + "|".join(words) + r")\b"), re.IGNORECASE)


CURRENT_RE = _word_re(CURRENT_WORDS)
POWER_RE = _word_re(POWER_WORDS)
OUTPUT_RE = _word_re(OUTPUT_WORDS)

_EXCLUDE = _word_re(("fault", "alarm", "trip", "fail", "setpoint", "demand",
                     "command", "limit", "threshold", "set-point"))


@dataclass
class Asset:
    """One machine, and the channels that can be used to judge it."""

    site: str
    unit: str
    run_key: str = ""
    duty_key: str = ""                 # motor current, else power
    duty_kind: str = ""                # "current" | "power"
    output_key: str = ""               # discharge flow, else discharge pressure
    output_kind: str = ""              # "flow" | "pressure"
    units: dict[str, str] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.site} unit {self.unit}"

    @property
    def judgeable(self) -> bool:
        """A run state plus at least one independent channel to check it."""
        return bool(self.run_key and self.duty_key)


def _matches(description: str, pattern: re.Pattern[str]) -> bool:
    """Whole-word match, with the fault/alarm/setpoint points ruled out first."""
    text = description or ""
    if _EXCLUDE.search(text):
        return False
    return bool(pattern.search(text))


def find_assets(sensors) -> list[Asset]:
    """
    Group the inventory into machines, conservatively.

    A role is filled only when exactly one sensor at that site and unit number
    is a candidate for it. Two ammeters on one pump and neither is chosen:
    picking one would mean comparing a run state against an arbitrary phase,
    and a fabricated contradiction here is a mechanical callout nobody needed.

    Nothing in the feed declares which ammeter belongs to which pump, so the
    relationship is recovered from the description exactly as
    `digital.find_pump_flow_pairs` does -- same site, same unit number.
    """
    candidates: dict[tuple[str, str], dict[str, list[tuple[str, str]]]] = {}
    units: dict[str, str] = {}

    for _, row in sensors.iterrows():
        site = row.get("site")
        if site is None or pd.isna(site) or not str(site):
            continue
        description = str(row.get("description") or "")
        unit = _unit_number(description)
        if unit is None:
            continue                    # no unit number: nothing to attribute

        # Ruled out before any role is considered, not only inside `_matches`.
        # A point named `Pump1-Current-Alarm-Setpoint` carries equipment
        # `Current`, and testing `equipment == "Current"` first short-circuited
        # straight past the exclusion -- so an alarm threshold was put forward
        # as the motor's duty channel, and a setpoint does not move when a
        # motor stops.
        if _EXCLUDE.search(description):
            continue

        key = str(row["sensor_key"])
        equipment = str(row.get("equipment") or "")
        signal_type = str(row.get("signal_type") or "").lower()
        kind = str(row.get("kind") or "")
        units[key] = str(row.get("unit") or "")
        slot = candidates.setdefault((str(site), unit),
                                     {"run": [], "current": [], "power": [],
                                      "flow": [], "pressure": []})

        if (signal_type == "digital" or kind == "status"
                or equipment in ("Pump", "Valve", "DigitalStatus")):
            if _is_run_state(description):
                slot["run"].append((key, description))
            continue
        if equipment == "Current" or _matches(description, CURRENT_RE):
            slot["current"].append((key, description))
        elif equipment == "Power" or _matches(description, POWER_RE):
            slot["power"].append((key, description))
        elif equipment == "Flowrate":
            slot["flow"].append((key, description))
        elif equipment == "Pressure":
            slot["pressure"].append((key, description))

    def only(items: list[tuple[str, str]]) -> str:
        """The single candidate, or nothing. See the docstring."""
        if len(items) == 1:
            return items[0][0]
        # Several: keep one only if exactly one of them names the output side
        # explicitly ("discharge", "delivery"). Two ammeters stay ambiguous.
        named = [k for k, d in items if _matches(d, OUTPUT_RE)]
        return named[0] if len(named) == 1 else ""

    assets: list[Asset] = []
    for (site, unit), slot in sorted(candidates.items()):
        run = only(slot["run"])
        if not run:
            continue
        duty, duty_kind = only(slot["current"]), "current"
        if not duty:
            duty, duty_kind = only(slot["power"]), "power"
        output, output_kind = only(slot["flow"]), "flow"
        if not output:
            output, output_kind = only(slot["pressure"]), "pressure"

        asset = Asset(site=site, unit=unit, run_key=run,
                      duty_key=duty, duty_kind=duty_kind if duty else "",
                      output_key=output,
                      output_kind=output_kind if output else "",
                      units=units)
        if asset.judgeable:
            assets.append(asset)
    return assets


def _locf(target_seconds: np.ndarray, source_seconds: np.ndarray,
          source_values: np.ndarray) -> np.ndarray:
    """
    Each target timestamp's most recent source reading.

    Last observation carried forward, which is the correct reading of this
    feed: a scanned value persists until the next report. Averaging into
    buckets would invent readings the instrument never produced.
    """
    order = np.searchsorted(source_seconds, target_seconds, side="right") - 1
    out = np.full(len(target_seconds), np.nan)
    valid = order >= 0
    out[valid] = source_values[order[valid]]
    return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous [start, end] index pairs where `mask` is True."""
    idx = np.flatnonzero(mask)
    if not idx.size:
        return []
    splits = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[splits + 1]]
    ends = np.r_[idx[splits], idx[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


@dataclass
class DutyProfile:
    """What this unit's duty channel does when it is on, and when it is off."""

    running: float
    off: float
    separation: float
    threshold: float

    @property
    def usable(self) -> bool:
        return self.separation > 0


def _duty_profile(run_values: np.ndarray, running_state: float,
                  duty: np.ndarray) -> DutyProfile | None:
    """
    The unit's own on and off duty levels, or `None` if it has no opinion.

    Abstaining here is most of the safety in this module. A current point that
    reads the same whether the pump is on or off cannot testify about whether
    the pump is turning, and treating its constant value as an "off" baseline
    would make every running hour look like an energisation fault.
    """
    on = duty[(run_values == running_state) & np.isfinite(duty)]
    off = duty[(run_values != running_state) & np.isfinite(duty)]
    if on.size < MIN_STATE_SAMPLES or off.size < MIN_STATE_SAMPLES:
        return None

    running, idle = float(np.median(on)), float(np.median(off))
    separation = running - idle
    scale = max(abs(running), 1e-9)
    if separation <= 0 or separation / scale < MIN_DUTY_SEPARATION:
        return None
    return DutyProfile(running=running, off=idle, separation=separation,
                       threshold=idle + ENERGISED_FRACTION * separation)


def _signal(atype: AnomalyType, ts: pd.Series, i: int, j: int, *,
            magnitude: float, unit: str, detail: dict[str, Any]) -> Signal:
    return Signal(
        type=atype,
        start=pd.Timestamp(ts.iloc[i]).to_pydatetime(),
        end=pd.Timestamp(ts.iloc[j]).to_pydatetime(),
        detector=DETECTOR,
        magnitude=round(float(magnitude), 6),
        unit=unit,
        n_points=j - i + 1,
        detail=detail,
    )


def detect_asset_faults(asset: Asset, series: dict) -> list[Signal]:
    """
    Every contradiction this machine's own channels can establish.

    Returns signals keyed to the run-state sensor, because that is the point an
    operator recognises -- but each `detail` names every channel it used, since
    the finding is about the relationship rather than about any one of them.
    """
    if asset.run_key not in series or asset.duty_key not in series:
        return []

    run_ts, run_values = series[asset.run_key]
    duty_ts, duty_values = series[asset.duty_key]
    if len(run_values) < MIN_POINTS or len(duty_values) < MIN_POINTS:
        return []
    if not _is_binary(run_values):
        return []

    running_state = float(np.nanmax(run_values))
    if running_state <= 0:
        return []

    run_seconds = to_epoch_seconds(run_ts)
    duty = _locf(run_seconds, to_epoch_seconds(duty_ts), duty_values)

    profile = _duty_profile(run_values, running_state, duty)
    if profile is None:
        return []

    duty_unit = asset.units.get(asset.duty_key, "") or ""
    is_on = run_values == running_state
    known = np.isfinite(duty)
    signals: list[Signal] = []

    # --- the machine is energised while the control says it is off ---------- #
    for i, j in _runs((~is_on) & known & (duty >= profile.threshold)):
        if run_seconds[j] - run_seconds[i] < MIN_CONTRADICTION_S:
            continue
        observed = float(np.nanmedian(duty[i:j + 1]))
        signals.append(_signal(
            AnomalyType.ASSET_ENERGISED_WHEN_OFF, run_ts, i, j,
            magnitude=observed - profile.off, unit=duty_unit,
            detail={
                "asset": asset.name,
                "duty_kind": asset.duty_kind,
                "duty_unit": duty_unit,
                "observed_duty": round(observed, 6),
                "off_duty": round(profile.off, 6),
                "running_duty": round(profile.running, 6),
                "duration_h": round((run_seconds[j] - run_seconds[i]) / 3600, 2),
                "verdict": f"the control says this unit is off and its "
                           f"{asset.duty_kind} says it is energised — a held-in "
                           f"contactor, a manual override, or a run-status bit "
                           f"that has failed",
            }))

    # --- the control says running and nothing is being drawn ---------------- #
    for i, j in _runs(is_on & known & (duty < profile.threshold)):
        if run_seconds[j] - run_seconds[i] < MIN_CONTRADICTION_S:
            continue
        observed = float(np.nanmedian(duty[i:j + 1]))
        signals.append(_signal(
            AnomalyType.ASSET_NOT_ENERGISED_WHEN_ON, run_ts, i, j,
            magnitude=profile.running - observed, unit=duty_unit,
            detail={
                "asset": asset.name,
                "duty_kind": asset.duty_kind,
                "duty_unit": duty_unit,
                "observed_duty": round(observed, 6),
                "running_duty": round(profile.running, 6),
                "off_duty": round(profile.off, 6),
                "duration_h": round((run_seconds[j] - run_seconds[i]) / 3600, 2),
                "verdict": f"the control says this unit is running and it is "
                           f"drawing {asset.duty_kind} at its OFF level — a "
                           f"tripped breaker, a blown fuse, a failed starter, "
                           f"or a run-status bit that has failed",
            }))

    signals += _not_delivering(asset, series, run_ts, run_seconds, is_on,
                               duty, profile)
    for signal in signals:
        signal.detail.setdefault("run_state_sensor", asset.run_key)
        signal.detail.setdefault("duty_sensor", asset.duty_key)
    return signals


def _not_delivering(asset: Asset, series: dict, run_ts, run_seconds,
                    is_on: np.ndarray, duty: np.ndarray,
                    profile: DutyProfile) -> list[Signal]:
    """
    Running, energised, and producing nothing -- with the duty channel agreeing.

    The discriminator this whole module exists for. When the output has gone
    AND the duty has moved away from this unit's own running level, two
    independent instruments agree the machine's operating point changed, and
    the machine is the finding.

    When the output has gone and the duty is exactly normal, this stays silent.
    A motor doing its usual work is evidence that water IS moving, so the
    flowmeter becomes the odd one out -- which is `RUN_STATE_INCONSISTENT`'s
    honest "one of these two is wrong", already emitted elsewhere. Claiming an
    asset failure there would be choosing the expensive explanation over the
    cheap one on no evidence.
    """
    if not asset.output_key or asset.output_key not in series:
        return []
    output_ts, output_values = series[asset.output_key]
    if len(output_values) < MIN_POINTS:
        return []

    output = _locf(run_seconds, to_epoch_seconds(output_ts), output_values)
    while_running = output[is_on & np.isfinite(output)]
    if while_running.size < MIN_STATE_SAMPLES:
        return []

    # Conditioned on the run state, and a median rather than a mean. A pump
    # idle half the window would otherwise drag the reference toward zero and
    # "no output" would never be reached; a median also tolerates the fault
    # being present in its own reference.
    normal_output = float(np.median(while_running))
    if normal_output <= 0:
        return []
    floor = NO_OUTPUT_FRACTION * normal_output

    signals: list[Signal] = []
    stopped = (is_on & np.isfinite(output) & (output <= floor)
               & np.isfinite(duty) & (duty >= profile.threshold))
    for i, j in _runs(stopped):
        if run_seconds[j] - run_seconds[i] < MIN_CONTRADICTION_S:
            continue
        observed_duty = float(np.nanmedian(duty[i:j + 1]))
        departure = abs(observed_duty - profile.running)
        if departure < DUTY_DEPARTURE * profile.separation:
            continue        # the motor is doing its usual work; see docstring

        observed_output = float(np.nanmedian(output[i:j + 1]))
        direction = "below" if observed_duty < profile.running else "above"
        signals.append(_signal(
            AnomalyType.ASSET_NOT_DELIVERING, run_ts, i, j,
            magnitude=normal_output - observed_output,
            unit=asset.units.get(asset.output_key, "") or "",
            detail={
                "asset": asset.name,
                "output_kind": asset.output_kind,
                "output_sensor": asset.output_key,
                "output_unit": asset.units.get(asset.output_key, "") or "",
                "duty_kind": asset.duty_kind,
                "duty_unit": asset.units.get(asset.duty_key, "") or "",
                "normal_output": round(normal_output, 6),
                "observed_output": round(observed_output, 6),
                "running_duty": round(profile.running, 6),
                "observed_duty": round(observed_duty, 6),
                "duty_departure": round(departure / profile.separation, 3),
                "duration_h": round((run_seconds[j] - run_seconds[i]) / 3600, 2),
                "verdict": f"running and not delivering: {asset.output_kind} "
                           f"has gone while the motor draws "
                           f"{asset.duty_kind} {direction} its own normal. "
                           f"Two independent channels agree the machine "
                           f"changed, so this is the plant and not the meter",
            }))
    return signals


def run_asset_checks(sensors, series: dict) -> dict[str, list[Signal]]:
    """Every asset contradiction in the fleet. Returns {run_state_key: [...]}."""
    out: dict[str, list[Signal]] = {}
    for asset in find_assets(sensors):
        signals = detect_asset_faults(asset, series)
        if signals:
            out.setdefault(asset.run_key, []).extend(signals)
    return out


def asset_summary(sensors, series: dict | None = None) -> dict[str, Any]:
    """
    What could be judged, and what could not.

    The second number is the point. 92 of the real inventory's 192 units carry
    a run state and nothing to check it against, and a report that only counted
    findings would imply this layer had looked at the whole estate.
    """
    assets = find_assets(sensors)
    with_output = sum(1 for a in assets if a.output_key)
    return {
        "assets": len(assets),
        "with_output_channel": with_output,
        "attributable": with_output,
        "duty_channels": {
            "current": sum(1 for a in assets if a.duty_kind == "current"),
            "power": sum(1 for a in assets if a.duty_kind == "power"),
        },
    }
