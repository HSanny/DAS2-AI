"""
das2.detect.digital
===================

Detectors for pump and valve state signals.

Why these exist as a separate module
------------------------------------
Roughly **1,567 Pump and Valve sensors** in this inventory are binary state
signals, and the system being replaced ran all of them through its analog
stack. `DETECTOR_PROFILE` had no `Pump` or `Valve` key, so they fell through to
the default — robust-Z, IsolationForest and DTW — over a series that only ever
takes the values 0 and 1. On a 0/1 signal a rolling median is 0 or 1, a MAD is
almost always exactly 0, and the whole apparatus produces either silence or
noise. They were also absent from `SKIP_UNCATEGORIZED_EQUIPMENT`, so the
compute was spent and the output was meaningless.

The faults that matter on these signals are not statistical at all. They are
counting and consistency questions, and each is nearly free to compute:

  * **SHORT_CYCLING** — a pump starting and stopping far more often than it
    should. This is real, expensive motor wear that is completely invisible to
    every other detector here, and the alarm feed shows it happening now:
    `PolderPS-OS-02-Flap-Valve-1-Status` cycles Open/Close every few seconds,
    and `YankitDiversion-Barrage-Gate-Position-Bad-IO` flaps ~60 times in 20
    minutes.
  * **STUCK_IN_STATE** — a duty pump that has not changed state in days, which
    usually means a seized actuator or a signal that has come adrift, not a
    plant that has been idle all week.
  * **RUN_STATE_INCONSISTENT** — commanded running, discharge flow near zero.
    A cross-signal physical contradiction, which is about as close to a
    zero-false-positive fault as this domain offers: either the pump is not
    running or the flowmeter is lying, and both are worth a visit.

Everything here is judged against the sensor's own history
----------------------------------------------------------
"More than six starts an hour" is meaningless fleet-wide: a booster on
pressure control may cycle every few minutes by design, while a transfer pump
runs for eight hours at a time. So the cycling rate is compared against that
sensor's own normal rate, exactly as `FLATLINE` compares against its own normal
flat run — and where there is not enough history to know, the detector
abstains.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from das2.detect.profile import SensorProfile
from das2.models import AnomalyType, Signal
from das2.timeutils import to_epoch_seconds

DETECTOR = "digital"

#: A digital signal must be this close to two-valued before these run. Some
#: points classified as status are really small integers (a mode or a step
#: number); counting "cycles" on those would be meaningless.
MAX_DISTINCT_STATES = 4

#: Cycling counts as short when it exceeds the sensor's own normal rate by this
#: multiple. Comparative rather than absolute, because normal cycling rates
#: across this fleet differ by orders of magnitude.
SHORT_CYCLE_MULTIPLE = 4.0

#: ...and an absolute floor, so a pump that normally starts twice a day is not
#: flagged for starting nine times. Six starts an hour is already hard on a
#: motor; this is not a tight bound.
SHORT_CYCLE_MIN_PER_HOUR = 6.0

#: Window over which the cycling rate is measured.
SHORT_CYCLE_WINDOW_S = 3600.0

#: A state held this many times longer than the sensor's own normal longest
#: hold counts as stuck.
STUCK_MULTIPLE = 5.0

#: ...with an absolute floor. Below a day, a held state is much more likely to
#: be a quiet plant than a seized actuator.
STUCK_FLOOR_S = 86400.0

#: Below this many readings, none of these can be judged.
MIN_POINTS = 30

#: Flow at or below this fraction of the paired meter's own running flow counts
#: as "not flowing" while the pump says it is running.
RUN_FLOW_FRACTION = 0.1

#: The contradiction must persist this long. A pump takes seconds to spin up
#: and a flowmeter takes seconds to respond, so a brief disagreement at a start
#: is expected behaviour rather than a fault.
RUN_STATE_MIN_S = 600.0


def _is_binary(values: np.ndarray) -> bool:
    finite = values[np.isfinite(values)]
    return finite.size > 0 and np.unique(finite).size <= MAX_DISTINCT_STATES


def _transitions(values: np.ndarray) -> np.ndarray:
    """Indices where the state changes."""
    return np.flatnonzero(np.diff(values) != 0) + 1


def detect_short_cycling(ts: pd.Series, values: np.ndarray,
                         profile: SensorProfile | None = None) -> list[Signal]:
    """
    The pump is starting and stopping far more often than it should.

    Motor wear from short cycling is cumulative, expensive and silent: nothing
    about any individual reading is wrong, so no threshold on the *value* can
    ever see it. Only the rate of change of state gives it away.

    Measured in a sliding one-hour window rather than over the whole window, so
    a two-hour bout of chattering is not averaged into invisibility across
    three days.
    """
    if len(values) < MIN_POINTS or not _is_binary(values):
        return []

    seconds = to_epoch_seconds(ts)
    changes = _transitions(values)
    if changes.size < 2:
        return []

    change_times = seconds[changes]
    total_hours = (seconds[-1] - seconds[0]) / 3600.0
    if total_hours <= 0:
        return []

    # The sensor's own normal rate. Built from this window for now; the daily
    # profile job will supply a better one from 28 days.
    # Contaminated by the bout it is looking for, as every within-window
    # baseline here is. It errs safe -- a chattering sensor inflates its own
    # "normal" and so raises its own threshold -- and the absolute floor is
    # what actually catches the case. The daily profile job will supply an
    # uncontaminated rate from 28 days.
    normal_per_hour = changes.size / total_hours
    threshold = max(SHORT_CYCLE_MIN_PER_HOUR,
                    normal_per_hour * SHORT_CYCLE_MULTIPLE)

    # Sliding one-hour count over the change times.
    signals: list[Signal] = []
    flagged: list[tuple[float, float, int]] = []
    left = 0
    for right in range(change_times.size):
        while change_times[right] - change_times[left] > SHORT_CYCLE_WINDOW_S:
            left += 1
        count = right - left + 1
        if count >= threshold:
            flagged.append((change_times[left], change_times[right], count))

    if not flagged:
        return []

    # Merge overlapping hot windows into one event per bout.
    merged: list[list[float]] = [list(flagged[0])]
    for start, end, count in flagged[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
            merged[-1][2] = max(merged[-1][2], count)
        else:
            merged.append([start, end, count])

    for start, end, count in merged:
        i = int(np.searchsorted(seconds, start))
        j = int(min(len(seconds) - 1, np.searchsorted(seconds, end)))
        hours = max((end - start) / 3600.0, 1e-9)
        signals.append(Signal(
            type=AnomalyType.SHORT_CYCLING,
            start=pd.Timestamp(ts.iloc[i]).to_pydatetime(),
            end=pd.Timestamp(ts.iloc[j]).to_pydatetime(),
            detector=DETECTOR,
            magnitude=round(count / hours, 2),
            unit="cycles/h",
            n_points=int(count),
            detail={
                "state_changes": int(count),
                "normal_per_hour": round(normal_per_hour, 2),
                "threshold_per_hour": round(threshold, 2),
            },
        ))
    return signals


def detect_stuck_in_state(ts: pd.Series, values: np.ndarray,
                          profile: SensorProfile | None = None) -> list[Signal]:
    """
    A state signal that has not moved in far longer than it normally does.

    Only the ACTIVE state is judged, and that limitation is the honest one.

    A pump stuck ON for 64 hours when it normally cycles every 4 is
    unambiguous: it should have stopped and did not. A pump stuck OFF is not,
    because within a 72-hour window "seized shut" and "standby pump correctly
    idle" produce the identical signal -- a flat zero. Measured on a synthetic
    standby that ran twice and then sat off for 66 hours, an unrestricted rule
    flagged it, and there is nothing in the data that makes that call wrong so
    much as unknowable.

    Telling those apart needs to know whether this pump normally runs, which
    needs weeks rather than days. The daily profile job is where stuck-OFF
    becomes answerable; until then this abstains on it rather than paging
    someone to look at a standby pump doing its job.
    """
    if len(values) < MIN_POINTS or not _is_binary(values):
        return []

    seconds = to_epoch_seconds(ts)
    changes = _transitions(values)
    if changes.size < 2:
        # Never changed in the whole window. Without history there is no way to
        # tell a seized actuator from a signal that simply does not move, and
        # 57% of this fleet does not move -- so abstain rather than guess.
        return []

    edges = np.concatenate(([0], changes, [len(values) - 1]))
    holds = np.diff(seconds[edges])
    # Three holds minimum, so "normal" is estimated from at least two others.
    if holds.size < 3:
        return []

    # The active state, against which "stuck" is decidable. See the docstring.
    active_state = float(np.nanmax(values))

    signals: list[Signal] = []
    for k, hold in enumerate(holds):
        if float(values[int(edges[k])]) != active_state:
            continue            # stuck-OFF is not distinguishable from standby
        # Leave-one-out: the candidate hold is excluded from the baseline it is
        # measured against. Including it defeats the detector whenever holds
        # are few, which they always are -- a 72-hour window on a pump that
        # cycles twice a day yields three or four. Measured on a pump stuck ON
        # for 62 hours out of 72: the stuck hold dragged the 95th percentile to
        # 58 h, the threshold to 290 h, and the detector found nothing. Third
        # instance of a statistic being spoiled by the anomaly it is looking
        # for, after FLATLINE and QUANTISATION_COLLAPSE.
        others = np.delete(holds, k)
        normal_hold = float(np.percentile(others, 95))
        threshold = max(normal_hold * STUCK_MULTIPLE, STUCK_FLOOR_S)
        if hold < threshold:
            continue
        i, j = int(edges[k]), int(edges[k + 1])
        signals.append(Signal(
            type=AnomalyType.STUCK_IN_STATE,
            start=pd.Timestamp(ts.iloc[i]).to_pydatetime(),
            end=pd.Timestamp(ts.iloc[j]).to_pydatetime(),
            detector=DETECTOR,
            magnitude=round(hold, 1),
            unit="s",
            n_points=j - i + 1,
            detail={
                "state": float(values[i]),
                "held_hours": round(hold / 3600.0, 2),
                "normal_hold_hours": round(normal_hold / 3600.0, 2),
                "threshold_hours": round(threshold / 3600.0, 2),
            },
        ))
    return signals


def detect_run_state_inconsistent(run_ts: pd.Series, run_values: np.ndarray,
                                  flow_ts: pd.Series, flow_values: np.ndarray,
                                  unit: str = "") -> list[Signal]:
    """
    The pump says it is running and its discharge is not flowing.

    A contradiction between two independent instruments, which makes it one of
    the most trustworthy findings available: statistics cannot explain it away,
    and exactly one of two physical things is true — the pump is not actually
    running, or the flowmeter has failed. Both warrant a visit, and the
    incident says so rather than guessing which.

    The two series are on independent scan cycles and share almost no
    timestamps, so the flow is read at each run-state sample by holding the
    last reported value forward. That is the correct interpretation for this
    feed: a scanned value persists until the next report, and averaging into
    buckets would invent readings the instrument never produced.
    """
    if len(run_values) < MIN_POINTS or len(flow_values) < MIN_POINTS:
        return []
    if not _is_binary(run_values):
        return []

    run_seconds = to_epoch_seconds(run_ts)
    flow_seconds = to_epoch_seconds(flow_ts)

    running_state = float(np.nanmax(run_values))
    if running_state <= 0:
        return []

    # Last-observation-carried-forward: for each run sample, the most recent
    # flow reading at or before it.
    order = np.searchsorted(flow_seconds, run_seconds, side="right") - 1
    valid = order >= 0
    aligned = np.full(len(run_seconds), np.nan)
    aligned[valid] = flow_values[order[valid]]

    # What this meter reads WHILE THE PUMP SAYS IT IS RUNNING. The reference
    # has to be conditioned on the run state: taking the median of every
    # positive reading instead lets the pump's off-periods dominate it, and on
    # a pump that is idle half the time that median collapses toward the noise
    # floor -- so "near zero" became "near zero flow", nothing ever cleared it,
    # and the detector was silent on a fault it had been given directly.
    #
    # The median tolerates the fault being present in its own reference: a
    # four-hour contradiction inside thirty-six hours of running does not move
    # a median. A mean would.
    while_running = aligned[(run_values == running_state) & np.isfinite(aligned)]
    if while_running.size < 10:
        return []
    running_flow = float(np.median(while_running))
    if running_flow <= 0:
        return []
    floor = RUN_FLOW_FRACTION * running_flow

    bad = (run_values == running_state) & np.isfinite(aligned) & (aligned <= floor)
    if not bad.any():
        return []

    idx = np.flatnonzero(bad)
    splits = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[splits + 1]]
    ends = np.r_[idx[splits], idx[-1]]

    signals: list[Signal] = []
    for i, j in zip(starts.tolist(), ends.tolist()):
        duration = run_seconds[j] - run_seconds[i]
        if duration < RUN_STATE_MIN_S:
            continue        # spin-up and meter lag, not a fault
        signals.append(Signal(
            type=AnomalyType.RUN_STATE_INCONSISTENT,
            start=pd.Timestamp(run_ts.iloc[i]).to_pydatetime(),
            end=pd.Timestamp(run_ts.iloc[j]).to_pydatetime(),
            detector=DETECTOR,
            magnitude=round(running_flow - float(np.nanmedian(aligned[i:j + 1])), 6),
            unit=unit,
            n_points=j - i + 1,
            detail={
                "expected_flow": round(running_flow, 6),
                "observed_flow": round(float(np.nanmedian(aligned[i:j + 1])), 6),
                "duration_h": round(duration / 3600.0, 2),
                "verdict": "either the pump is not running, or the flowmeter "
                           "has failed",
            },
        ))
    return signals


#: Words marking a digital point as a pump/valve RUN state rather than some
#: other status bit (a fault flag, a mode, a local/remote selector). Counting
#: "the pump is running" against a flowmeter only means anything if the point
#: really is a run state.
RUN_STATE_WORDS = ("run", "running", "started", "on-status", "status")

#: ...and words that mark a status point as something else entirely, checked
#: first. "Pump1-Fault-Status" contains "status" but says nothing about whether
#: the pump is turning.
NOT_RUN_STATE_WORDS = ("fault", "alarm", "trip", "fail", "mode", "local",
                       "remote", "auto", "available", "healthy", "comms")

#: Words marking a flow meter as measuring what a pump discharges.
DISCHARGE_WORDS = ("discharge", "delivery", "deliver", "outlet", "outflow",
                   "out-flow", "export")

#: Pump/unit identifier, so `Pump3-Run-Status` is paired with `Pump3` flow
#: rather than with `Pump1`'s.
_UNIT_PATTERN = re.compile(r"(?:pump|unit|p|set)\s*[-_ ]?(\d+)", re.IGNORECASE)


def _unit_number(description: str) -> str | None:
    match = _UNIT_PATTERN.search(description or "")
    return match.group(1) if match else None


def _is_run_state(description: str) -> bool:
    text = (description or "").lower()
    if any(word in text for word in NOT_RUN_STATE_WORDS):
        return False
    return any(word in text for word in RUN_STATE_WORDS)


def _is_discharge_flow(description: str) -> bool:
    text = (description or "").lower()
    return any(word in text for word in DISCHARGE_WORDS)


def find_pump_flow_pairs(sensors) -> list[tuple[str, str]]:
    """
    Pair each pump run-state signal with the flow meter on its discharge.

    Returns `[(run_state_key, flow_key), ...]`.

    This pairing is the reason RUN_STATE_INCONSISTENT needs a step of its own:
    it is the only detector here that reads two different sensors, and nothing
    in the feed declares which flowmeter belongs to which pump. The only place
    that relationship exists is the description, so it is recovered from there
    -- same site first, then a matching unit number, then a discharge word.

    Deliberately conservative. Where a site has several pumps and one
    unattributable flow meter, no pair is made: a wrong pairing would report a
    contradiction between two instruments that were never meant to agree, and
    a false "your pump is running dry" is exactly the wasted trip this system
    exists to prevent.
    """
    import pandas as pd

    runs, flows = [], []
    for _, row in sensors.iterrows():
        site = row.get("site")
        if site is None or pd.isna(site) or not str(site):
            continue
        description = str(row.get("description") or "")
        key, site = str(row["sensor_key"]), str(site)
        equipment = str(row.get("equipment") or "")
        signal_type = str(row.get("signal_type") or "")
        kind = str(row.get("kind") or "")

        if equipment == "Flowrate":
            flows.append((key, site, description))
        elif (signal_type.lower() == "digital" or kind == "status"
              or equipment in ("Pump", "Valve", "DigitalStatus")):
            if _is_run_state(description):
                runs.append((key, site, description))

    pairs: list[tuple[str, str]] = []
    for run_key, site, run_description in runs:
        same_site = [f for f in flows if f[1] == site]
        if not same_site:
            continue

        unit = _unit_number(run_description)
        if unit:
            # A run state that names its unit may ONLY pair with a meter naming
            # the same one. Falling back to "any discharge meter at this site"
            # when no unit matches looks harmless and is not: on the fixture it
            # paired Pump3's run state with Pump2's discharge meter, which
            # would report a contradiction between two instruments that were
            # never measuring the same thing -- a fabricated "your pump is
            # running dry", which is exactly the wasted trip this system exists
            # to prevent.
            candidates = [f for f in same_site if _unit_number(f[2]) == unit]
        else:
            # No unit named, so the site's single discharge meter is the only
            # reasonable reading of it.
            candidates = [f for f in same_site if _is_discharge_flow(f[2])]

        # Exactly one candidate, or the pairing is a guess. See the docstring.
        if len(candidates) == 1:
            pairs.append((run_key, candidates[0][0]))
    return pairs


def run_digital_checks(ts: pd.Series, values: np.ndarray,
                       profile: SensorProfile | None = None) -> list[Signal]:
    """
    Every SINGLE-series digital detector.

    RUN_STATE_INCONSISTENT is not here: it compares a pump against its
    flowmeter, so it needs the pairing from `find_pump_flow_pairs` and is run
    separately by the pipeline.
    """
    if len(values) == 0:
        return []
    signals: list[Signal] = []
    signals += detect_short_cycling(ts, values, profile)
    signals += detect_stuck_in_state(ts, values, profile)
    return signals


def run_pump_flow_checks(sensors, series: dict) -> dict[str, list[Signal]]:
    """
    Every pump/flow contradiction in the fleet. Returns {run_state_key: [...]}.

    Keyed on the run-state sensor, because that is the point an operator will
    recognise -- but `detail` names the flowmeter too, since the whole finding
    is that one of the two is wrong and the data cannot say which.
    """
    import pandas as pd

    out: dict[str, list[Signal]] = {}
    units = dict(zip(sensors["sensor_key"].astype(str),
                     sensors.get("unit", pd.Series([""] * len(sensors)))))
    for run_key, flow_key in find_pump_flow_pairs(sensors):
        if run_key not in series or flow_key not in series:
            continue
        run_ts, run_values = series[run_key]
        flow_ts, flow_values = series[flow_key]
        signals = detect_run_state_inconsistent(
            run_ts, run_values, flow_ts, flow_values,
            unit=str(units.get(flow_key) or ""))
        for signal in signals:
            signal.detail["run_state_sensor"] = run_key
            signal.detail["flow_sensor"] = flow_key
        if signals:
            out.setdefault(run_key, []).extend(signals)
    return out
