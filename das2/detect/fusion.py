"""
das2.detect.fusion
==================

Collapse many detector Signals into one SensorAnomaly per sensor per event.

Why this replaces the old voting gate
-------------------------------------
The system being replaced required `Methods_Used >= 2` before a sensor could be
reported. That rule is fatal to typed detection, and would have silently
emitted nothing at all: a FLATLINE is found by the flatline detector and by
nothing else, *by construction*. There is no second opinion to be had, so a
two-vote gate discards every unambiguous sensor-health fault while admitting
only the statistical findings that two correlated estimators happened to agree
on.

Worse, the three "independent" voters were not independent. The DTW channel was
algebraically a two-term first difference and IsolationForest's feature vector
keyed on the same `diff`, so two of the three votes were near-duplicates of one
another and the gate mostly measured agreement between a quantity and itself.

Fusion here does the opposite: a single detector is enough to report, because
each detector answers a *different physical question*. Corroboration still
matters, but as evidence that raises severity, never as a licence to speak.

Choosing the dominant type
--------------------------
One sensor can carry several overlapping signals at once -- a failed
transmitter goes STALE, and the last held value may also sit out of range. The
operator needs one answer to "what is wrong with this thing", so signals are
grouped into events by time overlap and each event gets a dominant type by a
fixed precedence:

    a sensor that is not reporting        beats
    a sensor reporting impossible values  beats
    a sensor frozen at a plausible value  beats
    everything statistical

The ordering is by *how certain the fault is*, not by how alarming it looks.
STALE outranks RANGE_VIOLATION because a sensor that has stopped talking is a
fact about the telemetry, whereas an out-of-range value is a fact about a
number whose provenance is already in doubt.

Severity is physical, never a Z-score
-------------------------------------
The old ranking used `Peak_RZ`, which is not comparable between sensors: a
PandanTG voltage channel scored 109.7 for a 1.8% excursion, because its MAD was
pinned at the 0.1 V quantisation step, while a pump on a 208 median scored 306.9
for a full-scale event. Ranking by that number ranks by each sensor's
quantisation step. `PhysicalSeverity` instead carries deviation in engineering
units, fraction of instrument span, duration in seconds, and fraction of the
window affected -- all of which an engineer can check against the instrument.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from das2.models import (
    AnomalyType,
    PhysicalSeverity,
    SensorAnomaly,
    SensorMeta,
    Signal,
)

#: Precedence when one event carries several types. Earlier wins. Ordered by
#: how certain the underlying fault is, not by how dramatic it sounds.
TYPE_PRECEDENCE: tuple[AnomalyType, ...] = (
    AnomalyType.STALE,                    # not reporting at all
    AnomalyType.FLATLINE,                 # reporting, but frozen
    AnomalyType.RANGE_VIOLATION,          # physically impossible value
    AnomalyType.REVERSE_FLOW,             # physically meaningful, and specific
    AnomalyType.QUANTISATION_COLLAPSE,    # resolution has degraded
    AnomalyType.DITHERING_DEAD,
    AnomalyType.SHORT_CYCLING,
    AnomalyType.RUN_STATE_INCONSISTENT,
    AnomalyType.STUCK_IN_STATE,
    AnomalyType.MASS_BALANCE_VIOLATION,
    AnomalyType.SPIKE,
    AnomalyType.LEVEL_SHIFT,
    AnomalyType.DRIFT,
    AnomalyType.NOISE_BURST,
    AnomalyType.RESIDUAL_OUTLIER,         # weakest: "the statistics disliked it"
)

_PRECEDENCE_INDEX = {t: i for i, t in enumerate(TYPE_PRECEDENCE)}

#: Signals this far apart still belong to one event. Detectors disagree about
#: edges -- the flatline detector marks the frozen run, the stale detector marks
#: the silence that follows it -- and splitting those into two incidents would
#: report one broken transmitter twice.
DEFAULT_MERGE_GAP_MIN = 30

#: Units that express a duration rather than a deviation from expected value.
TIME_UNITS = frozenset({"s", "sec", "seconds", "min", "h"})

#: Corroboration bonus. Two detectors answering different physical questions and
#: agreeing is real evidence, but it is a modifier on severity, never a gate.
CORROBORATION_BONUS = 1.15


def _precedence(t: AnomalyType) -> int:
    return _PRECEDENCE_INDEX.get(t, len(TYPE_PRECEDENCE))


def group_signals(signals: list[Signal],
                  merge_gap_min: int = DEFAULT_MERGE_GAP_MIN) -> list[list[Signal]]:
    """
    Partition one sensor's signals into events by time overlap.

    A sensor can fail twice in one window for unrelated reasons, and merging
    those into a single 72-hour "event" would misreport both the duration and
    the severity. Signals are merged only when they actually overlap, within a
    tolerance for detector edge disagreement.
    """
    if not signals:
        return []
    ordered = sorted(signals, key=lambda s: (s.start, s.end))
    gap = timedelta(minutes=merge_gap_min)

    events: list[list[Signal]] = [[ordered[0]]]
    horizon = ordered[0].end
    for sig in ordered[1:]:
        if sig.start - gap <= horizon:
            events[-1].append(sig)
            horizon = max(horizon, sig.end)
        else:
            events.append([sig])
            horizon = sig.end
    return events


def _dominant(signals: list[Signal]) -> AnomalyType:
    """
    The single type that best describes the event.

    Ties on precedence are broken by duration, because when two detectors of
    equal standing disagree the one that saw the condition for longer is
    describing the sensor's actual state rather than a moment of it.
    """
    return min(signals,
               key=lambda s: (_precedence(s.type), -s.duration_s)).type


def _severity(signals: list[Signal], sensor: SensorMeta,
              window_seconds: float) -> PhysicalSeverity:
    """
    Physical severity for one event.

    `span_fraction` is the component that makes sensors comparable, so it is
    computed only when the instrument's span is actually known -- guessing a
    span would reintroduce exactly the incomparability `Peak_RZ` suffered from.
    Where the span is unknown the severity falls back to duration and coverage,
    which are always meaningful.
    """
    start = min(s.start for s in signals)
    end = max(s.end for s in signals)
    duration_s = max(0.0, (end - start).total_seconds())

    # Largest magnitude among the signals that reported one in engineering
    # units. Signals that cannot express a magnitude contribute duration
    # instead, which is what actually matters for them.
    #
    # Time-valued magnitudes are excluded explicitly. STALE and FLATLINE report
    # how LONG the condition lasted, in seconds, and letting that through made
    # a 4-hour silence render as "deviation: 14538 s" in the dashboard's
    # deviation column -- a duration wearing an engineering-deviation label,
    # which an operator would read as the instrument being 14,538 units out.
    with_magnitude = [s for s in signals
                      if s.magnitude and s.unit and s.unit not in TIME_UNITS]
    deviation = max((abs(s.magnitude) for s in with_magnitude), default=0.0)
    unit = next((s.unit for s in with_magnitude
                 if abs(s.magnitude) == deviation), sensor.unit or "")

    span = sensor.span
    span_fraction = min(1.0, deviation / span) if (span and deviation) else None

    window_fraction = (min(1.0, duration_s / window_seconds)
                       if window_seconds > 0 else 0.0)

    return PhysicalSeverity(
        deviation=round(deviation, 6),
        unit=unit or "",
        span_fraction=span_fraction,
        duration_s=duration_s,
        window_fraction=window_fraction,
    )


def fuse_sensor(sensor: SensorMeta, signals: list[Signal], *,
                window_start: datetime | None = None,
                window_end: datetime | None = None,
                merge_gap_min: int = DEFAULT_MERGE_GAP_MIN) -> list[SensorAnomaly]:
    """
    Every distinct event on one sensor.

    Returns a list, not a single anomaly: a sensor that spikes at 02:00 and
    goes stale at 20:00 has two things wrong with it, and collapsing them would
    report a twenty-hour fault that never happened.
    """
    if not signals:
        return []

    window_seconds = 0.0
    if window_start and window_end:
        window_seconds = max(0.0, (window_end - window_start).total_seconds())

    out: list[SensorAnomaly] = []
    for event in group_signals(signals, merge_gap_min):
        start = min(s.start for s in event)
        end = max(s.end for s in event)
        severity = _severity(event, sensor, window_seconds)

        # Corroboration from a genuinely different detector raises severity.
        # Counted by detector, not by signal, so one detector emitting three
        # runs of the same fault does not look like agreement.
        if len({s.detector for s in event}) > 1:
            severity = PhysicalSeverity(
                deviation=severity.deviation,
                unit=severity.unit,
                span_fraction=(min(1.0, severity.span_fraction * CORROBORATION_BONUS)
                               if severity.span_fraction is not None else None),
                duration_s=severity.duration_s,
                window_fraction=min(1.0, severity.window_fraction * CORROBORATION_BONUS),
            )

        out.append(SensorAnomaly(
            sensor=sensor,
            start=start,
            end=end,
            dominant_type=_dominant(event),
            signals=sorted(event, key=lambda s: (_precedence(s.type), s.start)),
            severity=severity,
        ))
    return out


def fuse_all(signals_by_sensor: dict[str, tuple[SensorMeta, list[Signal]]], *,
             window_start: datetime | None = None,
             window_end: datetime | None = None,
             merge_gap_min: int = DEFAULT_MERGE_GAP_MIN) -> list[SensorAnomaly]:
    """Fuse a whole run, highest severity first."""
    out: list[SensorAnomaly] = []
    for sensor, signals in signals_by_sensor.values():
        out.extend(fuse_sensor(sensor, signals, window_start=window_start,
                               window_end=window_end, merge_gap_min=merge_gap_min))
    out.sort(key=lambda a: -a.score)
    return out


def fusion_summary(anomalies: list[SensorAnomaly]) -> dict[str, object]:
    """Per-run counts by type, for the log line and the dashboard header."""
    by_type: dict[str, int] = {}
    for a in anomalies:
        key = a.dominant_type.value
        by_type[key] = by_type.get(key, 0) + 1
    return {
        "anomalies": len(anomalies),
        "sensors": len({a.sensor.sensor_key for a in anomalies}),
        "by_type": dict(sorted(by_type.items(), key=lambda kv: -kv[1])),
        "corroborated": sum(1 for a in anomalies
                            if len({s.detector for s in a.signals}) > 1),
    }
