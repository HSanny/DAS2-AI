"""
das2.models
===========

Typed domain objects shared by the whole pipeline.

Two design decisions are load-bearing here, and both come from defects measured
in the system being replaced.

**1. Anomalies carry a TYPE, not a score.**
The old detector fused three channels into a boolean and ranked sensors by
`Peak_RZ`. That number is not comparable between sensors: `PandanTG-LT-Voltage2`
reported `Peak_RZ = 109.7` for a 1.8% excursion (its MAD was pinned at the 0.1 V
quantisation step), while `Bidadari...Pump 2` reported 306.9 for a full-scale
event. The ranking that decided which ten sensors reached the client was driven
by each sensor's quantisation step rather than by how wrong it was.

A dominant `AnomalyType` is actionable in a way a score never is: FLATLINE means
send a technician, LEVEL_SHIFT with correlated neighbours means the water moved.
Triage keys off the type; severity only orders within it.

**2. Severity is physical first, derived second.**
`PhysicalSeverity` holds facts an engineer can check -- deviation in engineering
units, fraction of instrument span, duration in seconds. `score()` derives a
0-100 number from those for ordering, and is explicitly a convenience, not the
source of truth.

Deliberately absent: a fused 0-1 "confidence". It cannot be calibrated without
labels, and "what does 0.72 mean?" has no defensible answer until the operator
feedback buttons shipped in Phase 0 have produced some.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


# --------------------------------------------------------------------------- #
# Anomaly taxonomy
# --------------------------------------------------------------------------- #
class AnomalyType(str, Enum):
    """
    What is wrong, not how wrong.

    Inherits from str so values serialise to JSON/CSV/SQL as plain strings.
    """

    # --- L1: sensor health. Physical facts, no quantile calibration needed. ---
    FLATLINE = "FLATLINE"                          # value unchanged while still reporting
    STALE = "STALE"                                # stopped reporting altogether
    RANGE_VIOLATION = "RANGE_VIOLATION"            # outside physical plausibility
    SPIKE = "SPIKE"                                # |dv/dt| beyond physical limit
    REVERSE_FLOW = "REVERSE_FLOW"                  # sustained negative flow
    QUANTISATION_COLLAPSE = "QUANTISATION_COLLAPSE"  # resolution degraded
    DITHERING_DEAD = "DITHERING_DEAD"              # oscillating in the last bit

    # --- L2: behavioural ---
    RESIDUAL_OUTLIER = "RESIDUAL_OUTLIER"          # vs time-of-day baseline
    LEVEL_SHIFT = "LEVEL_SHIFT"                    # CUSUM change point
    MASS_BALANCE_VIOLATION = "MASS_BALANCE_VIOLATION"  # dLevel/dt*A != Qin - Qout

    # --- L2: digital (pump/valve) ---
    SHORT_CYCLING = "SHORT_CYCLING"                # excessive starts/hour: motor wear
    STUCK_IN_STATE = "STUCK_IN_STATE"
    RUN_STATE_INCONSISTENT = "RUN_STATE_INCONSISTENT"  # running but no flow

    # --- Daily job (needs history longer than one analysis window) ---
    DRIFT = "DRIFT"                                # slow calibration drift
    NOISE_BURST = "NOISE_BURST"                    # spread grew vs baseline


#: Types that mean the INSTRUMENT is faulty -- these justify dispatching someone.
SENSOR_HEALTH_TYPES: frozenset[AnomalyType] = frozenset({
    AnomalyType.FLATLINE,
    AnomalyType.STALE,
    AnomalyType.RANGE_VIOLATION,
    AnomalyType.QUANTISATION_COLLAPSE,
    AnomalyType.DITHERING_DEAD,
    AnomalyType.NOISE_BURST,
})

#: Types that mean the PROCESS moved -- the water, not the instrument. These are
#: the ones worth correlating against neighbours before sending anyone anywhere.
PROCESS_TYPES: frozenset[AnomalyType] = frozenset({
    AnomalyType.RESIDUAL_OUTLIER,
    AnomalyType.LEVEL_SHIFT,
    AnomalyType.SPIKE,
    AnomalyType.REVERSE_FLOW,
    AnomalyType.MASS_BALANCE_VIOLATION,
})

#: Maintenance-scheduling rather than urgent-response.
MAINTENANCE_TYPES: frozenset[AnomalyType] = frozenset({
    AnomalyType.DRIFT,
    AnomalyType.SHORT_CYCLING,
})

#: Faults where the instrument is definitively broken, and no amount of
#: agreement from the neighbours changes that.
#:
#: This distinction matters because correlation is only informative about a
#: sensor's *value*. A transmitter that has stopped reporting has stopped
#: reporting; a frozen reading is frozen; a collapsed ADC has collapsed. None
#: of those become acceptable because a sensor two kilometres away happened to
#: move in sympathy. Measured on the fixture, treating these as
#: correlation-dependent downgraded a genuine frozen flowmeter from
#: SENSOR_FAULT to a suppressed WATCH, purely because an unrelated conductivity
#: sensor at the next site correlated weakly with the flat line.
DEFINITIVE_INSTRUMENT_FAULTS: frozenset[AnomalyType] = frozenset({
    AnomalyType.FLATLINE,
    AnomalyType.STALE,
    AnomalyType.QUANTISATION_COLLAPSE,
    AnomalyType.DITHERING_DEAD,
})

#: Findings strong enough to alert even on an equipment class that is still
#: marked `alertable: false`.
#:
#: The `alertable` flag exists to stop newly classified equipment from paging
#: anyone before its false-positive rate has been seen -- full coverage put
#: roughly four times as many sensors under detection, and admitting them all
#: at once would bury the operator. It is the right default for a generic
#: status bit.
#:
#: But Pump, Valve and DigitalStatus are all `alertable: false`, and they are
#: exactly the classes the digital and cross-signal detectors were written for.
#: Left as-is, a pump insisting it is running against a meter reading zero was
#: detected, scored, put on the dashboard -- and silently prevented from ever
#: reaching anyone, which makes the detector pointless.
#:
#: These types are exempt because their confidence does not depend on the
#: equipment class being well understood. A contradiction between two
#: instruments is a physical impossibility whatever they are attached to, and
#: short cycling is a count of state changes, not a statistical inference.
#: Everything else on those classes still waits its turn.
ALWAYS_PAGEABLE_TYPES: frozenset["AnomalyType"] = frozenset()   # filled below

#: Contradictions between two or more instruments. Physically impossible rather
#: than statistically unusual, so they are never held for more evidence: the
#: evidence is already conclusive, and only the culprit is unknown.
CROSS_SIGNAL_TYPES: frozenset[AnomalyType] = frozenset({
    AnomalyType.MASS_BALANCE_VIOLATION,
    AnomalyType.RUN_STATE_INCONSISTENT,
})

ALWAYS_PAGEABLE_TYPES = CROSS_SIGNAL_TYPES | frozenset({
    # Real, cumulative, expensive motor wear that no value-based detector can
    # see, and which the client's own alarm feed shows happening now.
    AnomalyType.SHORT_CYCLING,
})


class Priority(str, Enum):
    P1 = "P1"   # act now
    P2 = "P2"   # act this shift
    P3 = "P3"   # schedule
    P4 = "P4"   # record only

    @classmethod
    def from_score(cls, score: float) -> "Priority":
        if score >= 75:
            return cls.P1
        if score >= 50:
            return cls.P2
        if score >= 25:
            return cls.P3
        return cls.P4


# --------------------------------------------------------------------------- #
# Sensors and readings
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SensorMeta:
    """
    Identity and placement of one sensor.

    `latitude`/`longitude` come from an RTU-level join, so every sensor on one
    RTU shares identical coordinates. Spatial resolution is therefore SITE
    level, which is the right granularity for deciding where to send someone but
    cannot resolve position within a site. `coords_are_site_level` exists so the
    dashboard can say so rather than implying a precision that is not there.
    """

    sensor_key: str
    description: str
    equipment: str
    signal_type: str = "Analog"           # Analog | Digital
    rtu_number: str | None = None
    site: str | None = None               # site prefix parsed from description
    latitude: float | None = None
    longitude: float | None = None
    planning_area: str | None = None      # indicative only, see spatial.regions
    region: str | None = None             # the level clustering actually trusts
    unit: str | None = None
    span_min: float | None = None
    span_max: float | None = None
    alertable: bool = True

    @property
    def has_coords(self) -> bool:
        return self.latitude is not None and self.longitude is not None

    @property
    def coords_are_site_level(self) -> bool:
        return True

    @property
    def span(self) -> float | None:
        if self.span_min is None or self.span_max is None:
            return None
        width = float(self.span_max) - float(self.span_min)
        return width if width > 0 else None


# --------------------------------------------------------------------------- #
# Detector output
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Signal:
    """
    One detector's typed finding over a time span.

    Detectors emit these; `detect.fusion` combines them into a SensorAnomaly.
    `magnitude` is in engineering units wherever the detector can express it
    that way, because a number an engineer can sanity-check is worth more than a
    dimensionless score they cannot.
    """

    type: AnomalyType
    start: datetime
    end: datetime
    detector: str
    magnitude: float = 0.0
    unit: str = ""
    n_points: int = 0
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return max(0.0, (self.end - self.start).total_seconds())

    def overlaps(self, other: "Signal", tolerance_s: float = 0.0) -> bool:
        gap = tolerance_s
        return (self.start.timestamp() - gap) <= other.end.timestamp() and \
               (other.start.timestamp() - gap) <= self.end.timestamp()


@dataclass(frozen=True)
class PhysicalSeverity:
    """
    How bad it is, in terms an engineer can verify against the instrument.

    `score()` is a derived ordering convenience. The fields above it are the
    real content, and are what the dashboard and the alert should show.
    """

    deviation: float = 0.0          # engineering units away from expected
    unit: str = ""
    span_fraction: float | None = None   # deviation / instrument span, 0..1
    duration_s: float = 0.0
    window_fraction: float = 0.0    # fraction of the analysis window affected

    def score(self) -> float:
        """
        Derived 0-100 ordering score.

        Weighted towards fraction-of-span, because that is the one component
        comparable between a 500 V bus and a 20 bar main. Sensors with no known
        span fall back to duration and coverage alone rather than to a raw
        deviation that means nothing across equipment classes.
        """
        parts: list[tuple[float, float]] = []   # (value 0..1, weight)

        if self.span_fraction is not None:
            parts.append((min(1.0, max(0.0, self.span_fraction)), 0.5))

        # 4 hours of a fault is treated as "as long as it needs to be".
        parts.append((min(1.0, self.duration_s / (4 * 3600.0)), 0.3))
        parts.append((min(1.0, max(0.0, self.window_fraction)), 0.2))

        total_weight = sum(w for _, w in parts)
        if total_weight <= 0:
            return 0.0
        return 100.0 * sum(v * w for v, w in parts) / total_weight


@dataclass
class SensorAnomaly:
    """
    Fused per-sensor finding: several Signals collapsed to one dominant type.

    Replaces the old boolean `Combined_Anomaly` plus `Peak_RZ` ranking. Note
    `signals` is retained in full -- the evidence that produced the verdict has
    to survive into the dashboard, or an operator cannot check the machine's
    reasoning.
    """

    sensor: SensorMeta
    start: datetime
    end: datetime
    dominant_type: AnomalyType
    signals: list[Signal] = field(default_factory=list)
    severity: PhysicalSeverity = field(default_factory=PhysicalSeverity)

    @property
    def types(self) -> set[AnomalyType]:
        return {s.type for s in self.signals}

    @property
    def is_sensor_health(self) -> bool:
        return self.dominant_type in SENSOR_HEALTH_TYPES

    @property
    def is_process(self) -> bool:
        return self.dominant_type in PROCESS_TYPES

    @property
    def score(self) -> float:
        return self.severity.score()

    @property
    def duration_s(self) -> float:
        return max(0.0, (self.end - self.start).total_seconds())


# --------------------------------------------------------------------------- #
# Spatial grouping
# --------------------------------------------------------------------------- #
@dataclass
class Cluster:
    """
    Sensors that went abnormal near each other at the same time.

    This is the object the client actually asked for -- "a cluster of abnormal
    sensors classified altogether, based on locations". Because coordinates are
    RTU-level, a cluster groups SITES rather than individual sensor positions.
    """

    members: list[SensorAnomaly] = field(default_factory=list)
    region: str | None = None
    centroid_lat: float | None = None
    centroid_lon: float | None = None
    radius_m: float = 0.0

    @property
    def start(self) -> datetime | None:
        return min((m.start for m in self.members), default=None)

    @property
    def end(self) -> datetime | None:
        return max((m.end for m in self.members), default=None)

    @property
    def equipment_types(self) -> set[str]:
        return {m.sensor.equipment for m in self.members}

    @property
    def sites(self) -> set[str]:
        return {m.sensor.site for m in self.members if m.sensor.site}

    @property
    def sensor_keys(self) -> set[str]:
        return {m.sensor.sensor_key for m in self.members}

    @property
    def dominant_types(self) -> set[AnomalyType]:
        return {m.dominant_type for m in self.members}


# --------------------------------------------------------------------------- #
# Incidents
# --------------------------------------------------------------------------- #
class IncidentClass(str, Enum):
    """What the evidence says is going on, which determines what to do."""

    TELEMETRY_FANOUT = "TELEMETRY_FANOUT"        # one panel/RTU, many sensors -> suppress
    # Many instruments, several SITES, all gone quiet together. Instruments do
    # not fail in hundreds across kilometres; the path carrying their readings
    # does. Separate from FANOUT, which is one panel, because the remedy is
    # different: a link or an RTU group, not a fuse.
    TELEMETRY_OUTAGE = "TELEMETRY_OUTAGE"
    REGIONAL_EVENT = "REGIONAL_EVENT"            # several sites, several types -> escalate
    SENSOR_FAULT = "SENSOR_FAULT"                # instrument is broken -> dispatch
    DRIFT_MAINTENANCE = "DRIFT_MAINTENANCE"      # schedule calibration
    PROCESS_EVENT = "PROCESS_EVENT"              # the water moved -> monitor
    WEATHER_DRIVEN = "WEATHER_DRIVEN"            # rain explains it -> do not dispatch
    # Two instruments cannot both be right. Physically conclusive, so it is a
    # dispatch rather than a WATCH -- what is unknown is which one to believe,
    # not whether something is wrong.
    INSTRUMENT_CONFLICT = "INSTRUMENT_CONFLICT"
    WATCH = "WATCH"                              # weak/conflicting -> re-evaluate


class IncidentStatus(str, Enum):
    OPEN = "OPEN"
    UPDATED = "UPDATED"
    RESOLVED = "RESOLVED"


class AckState(str, Enum):
    NONE = "NONE"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    DISPATCHED = "DISPATCHED"
    FALSE_ALARM = "FALSE_ALARM"


#: What each class means operationally. Kept beside the enum so the alert text,
#: the dashboard and the triage rules cannot drift apart.
RECOMMENDATION: dict[IncidentClass, str] = {
    IncidentClass.TELEMETRY_FANOUT:
        "Telemetry fan-out from shared equipment - no site visit needed.",
    IncidentClass.REGIONAL_EVENT:
        "Multiple sites affected together - investigate the area, not one sensor.",
    IncidentClass.SENSOR_FAULT:
        "Instrument fault with no corroboration from neighbours - dispatch a technician.",
    IncidentClass.TELEMETRY_OUTAGE:
        "Many sensors across several sites stopped reporting together - check "
        "the comms path or the historian feed. Do NOT dispatch per sensor.",
    IncidentClass.DRIFT_MAINTENANCE:
        "Gradual drift - schedule recalibration, not urgent.",
    IncidentClass.PROCESS_EVENT:
        "Neighbouring sensors moved together - this looks like the process, not a fault.",
    IncidentClass.WEATHER_DRIVEN:
        "Rainfall nearby explains this - monitor only, do not dispatch.",
    IncidentClass.WATCH:
        "Evidence is weak or conflicting - re-evaluate on the next run.",
    IncidentClass.INSTRUMENT_CONFLICT:
        "Instruments contradict each other - check all of them; one is wrong.",
}


@dataclass
class Incident:
    """
    The unit of alerting: an event in a place, persisting across runs.

    The system being replaced alerted per sensor per run. With a 72h window
    re-run every 6h, consecutive runs overlap by 66 hours, so one three-day
    fault produced up to twelve Telegram messages. An Incident holds a stable
    id across runs so it is announced once, updated only when it meaningfully
    changes, and closed when it clears.
    """

    incident_id: str
    cluster: Cluster
    incident_class: IncidentClass = IncidentClass.WATCH
    status: IncidentStatus = IncidentStatus.OPEN
    severity: float = 0.0                 # 0-100
    opened_at: datetime | None = None
    last_seen_at: datetime | None = None
    resolved_at: datetime | None = None
    ack_state: AckState = AckState.NONE
    ack_by: str | None = None
    narrative: str = ""
    # Neighbour correlation: the single most decision-relevant signal for
    # "is it the water or the instrument?". None means it was not computed.
    neighbour_correlation: float | None = None
    rainfall_mm: float | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def priority(self) -> Priority:
        return Priority.from_score(self.severity)

    @property
    def recommendation(self) -> str:
        return RECOMMENDATION[self.incident_class]

    @property
    def should_alert(self) -> bool:
        """
        Fan-out is noise by construction, and an unacknowledged WATCH is not
        worth anyone's attention until the evidence firms up.
        """
        if self.incident_class is IncidentClass.TELEMETRY_FANOUT:
            return False
        if self.incident_class is IncidentClass.WATCH:
            return False
        return True

    @property
    def should_dispatch(self) -> bool:
        return self.incident_class in (
            IncidentClass.SENSOR_FAULT,
            IncidentClass.REGIONAL_EVENT,
        )

    @property
    def sensor_keys(self) -> set[str]:
        return self.cluster.sensor_keys
