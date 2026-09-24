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

    Names follow IOOS QARTOD where QARTOD has a name for the thing
    -- see `QARTOD_TEST` below. That is not cosmetic. QARTOD's
    *Manual for Real-Time Quality Control of Water Level Data* is the
    operational standard for exactly this estate, and a finding an operator
    can look up in a published manual is worth more than one that exists only
    in our source. Where we have no standard equivalent the name is ours and
    `QARTOD_TEST` says so, so nothing invented is ever mistaken for inherited.
    """

    # --- L1: sensor health. Physical facts, no quantile calibration needed. ---
    FLATLINE = "FLATLINE"                          # value unchanged while still reporting
    STALE = "STALE"                                # stopped reporting altogether
    RANGE_VIOLATION = "RANGE_VIOLATION"            # outside physical plausibility
    SPIKE = "SPIKE"                                # excursion that returns
    REVERSE_FLOW = "REVERSE_FLOW"                  # sustained negative flow
    QUANTISATION_COLLAPSE = "QUANTISATION_COLLAPSE"  # resolution degraded
    #: Alive, reporting, and measuring nothing -- the reading dithers in the
    #: last bit or two and never moves. Renamed from the invented
    #: `DITHERING_DEAD`: this is QARTOD Test 10, and our detector (rolling
    #: range below a ceiling) is already the `check_type="range"` form of it.
    #: The old spelling still deserialises; see `_missing_`.
    ATTENUATED_SIGNAL = "ATTENUATED_SIGNAL"

    # --- L2: behavioural ---
    RESIDUAL_OUTLIER = "RESIDUAL_OUTLIER"          # vs time-of-day baseline
    LEVEL_SHIFT = "LEVEL_SHIFT"                    # the PROCESS stepped
    #: The same shape, the other cause: the INSTRUMENT stepped.
    #:
    #: Split out of LEVEL_SHIFT because a post-maintenance recalibration and a
    #: genuine water-level change are the same signal, and every published
    #: taxonomy carries offset as a sensor fault -- Ni et al. (2009) ACM TOSN
    #: 5(3), Sharma et al. (2010) ACM TOSN 6(3), Leigh et al. (2019) STOTEN
    #: 664. Before this, every recalibration bump was routed to "the water
    #: moved, go look".
    INSTRUMENT_OFFSET = "INSTRUMENT_OFFSET"
    MASS_BALANCE_VIOLATION = "MASS_BALANCE_VIOLATION"  # dLevel/dt*A != Qin - Qout

    # --- L2: digital (pump/valve) ---
    SHORT_CYCLING = "SHORT_CYCLING"                # excessive starts/hour: motor wear
    STUCK_IN_STATE = "STUCK_IN_STATE"
    RUN_STATE_INCONSISTENT = "RUN_STATE_INCONSISTENT"  # running but no flow

    # --- Daily job (needs history longer than one analysis window) ---
    DRIFT = "DRIFT"                                # slow calibration drift
    NOISE_BURST = "NOISE_BURST"                    # spread grew vs baseline

    @classmethod
    def _missing_(cls, value):
        """
        Accept names this project used before it aligned with QARTOD.

        `das2_sensor_anomaly.dominant_type` is a string column with months of
        rows in it. Renaming the enum without this turns every historical
        `DITHERING_DEAD` row into a ValueError the first time the profile job
        reads it back.
        """
        legacy = {"DITHERING_DEAD": cls.ATTENUATED_SIGNAL}
        return legacy.get(str(value).upper())


#: Where each finding sits in IOOS QARTOD's numbered test list.
#:
#: `None` means we have no standard equivalent and the name is ours. Keeping
#: that explicit is the point of the table: it is the difference between "this
#: is Test 8, Flat Line, and here is the manual" and "we called it something".
#:
#: Two gaps worth naming, because they are absent rather than renamed:
#:
#:   * **Test 7, Rate of Change.** We do not have it. Our SPIKE requires the
#:     excursion to come back, which makes it Test 6 -- a fast move that does
#:     NOT return is caught only if it happens to be a clean step. Adding it is
#:     easy and deliberately deferred: an unconditioned rate test on a drainage
#:     estate fires on every storm, so it belongs after rainfall conditioning
#:     exists, not before.
#:   * **Tests 2, 3 and 5** (Syntax, Location, Climatology). Climatology needs
#:     more history than a 72-hour window holds.
QARTOD_TEST: dict["AnomalyType", tuple[int, str] | None] = {
    AnomalyType.STALE: (1, "Gap"),
    AnomalyType.RANGE_VIOLATION: (4, "Gross Range"),
    AnomalyType.SPIKE: (6, "Spike"),
    AnomalyType.FLATLINE: (8, "Flat Line"),
    AnomalyType.MASS_BALANCE_VIOLATION: (9, "Multi-Variate"),
    AnomalyType.RUN_STATE_INCONSISTENT: (9, "Multi-Variate"),
    AnomalyType.ATTENUATED_SIGNAL: (10, "Attenuated Signal"),
    # Ours. No QARTOD equivalent, and saying so is the point.
    AnomalyType.REVERSE_FLOW: None,        # a signed gross-range special case
    AnomalyType.QUANTISATION_COLLAPSE: None,   # measurement basis: Thornhill 2004
    AnomalyType.RESIDUAL_OUTLIER: None,
    AnomalyType.LEVEL_SHIFT: None,
    AnomalyType.INSTRUMENT_OFFSET: None,
    AnomalyType.SHORT_CYCLING: None,
    AnomalyType.STUCK_IN_STATE: None,
    AnomalyType.DRIFT: None,
    AnomalyType.NOISE_BURST: None,
}



class QartodFlag(int, Enum):
    """
    IOOS QARTOD's disposition vocabulary, which is orthogonal to `AnomalyType`.

    `AnomalyType` says WHY a reading is suspect; this says HOW BADLY, in the
    four values every QARTOD-conformant system emits. Reporting both means an
    operator can filter on a standard severity without us having to invent one,
    and a future consumer of this data does not have to learn our taxonomy to
    use it.
    """

    GOOD = 1
    UNKNOWN = 2        # test not applicable, or not run
    SUSPECT = 3        # failed a secondary criterion; use with caution
    FAIL = 4           # failed the primary criterion
    MISSING = 9


#: Aggregation precedence, verbatim from QARTOD's own reference implementation
#: (`ioos_qc.qartod.aggregate`): later in this tuple wins. Note it is NOT
#: numeric order -- MISSING is 9 but loses to FAIL, because "we have no data"
#: is a weaker statement than "the data we have is wrong".
FLAG_PRECEDENCE: tuple[QartodFlag, ...] = (
    QartodFlag.MISSING,
    QartodFlag.UNKNOWN,
    QartodFlag.GOOD,
    QartodFlag.SUSPECT,
    QartodFlag.FAIL,
)

_FLAG_RANK = {flag: i for i, flag in enumerate(FLAG_PRECEDENCE)}


def aggregate_flags(flags) -> QartodFlag:
    """The worst flag among several, by QARTOD's precedence."""
    worst = QartodFlag.UNKNOWN
    for flag in flags:
        if _FLAG_RANK.get(flag, -1) > _FLAG_RANK.get(worst, -1):
            worst = flag
    return worst


#: The QARTOD disposition each finding carries.
#:
#: FAIL is reserved for facts about the channel -- it is not reporting, it is
#: frozen, the value is physically impossible. SUSPECT is for everything that
#: is an inference, however strong, because QARTOD's own definition of FAIL is
#: "failed the primary criterion" and a statistical verdict is not that.
#:
#: This is why a two-tier gross range matters and is the next thing to build:
#: instrument span gives a defensible FAIL tier without PUB's commissioned
#: alarm limits, which we have asked for and not received.
TYPE_FLAG: dict["AnomalyType", QartodFlag] = {
    AnomalyType.STALE: QartodFlag.MISSING,
    AnomalyType.FLATLINE: QartodFlag.FAIL,
    AnomalyType.RANGE_VIOLATION: QartodFlag.FAIL,
    AnomalyType.QUANTISATION_COLLAPSE: QartodFlag.FAIL,
    AnomalyType.ATTENUATED_SIGNAL: QartodFlag.FAIL,
    AnomalyType.RUN_STATE_INCONSISTENT: QartodFlag.FAIL,
    AnomalyType.MASS_BALANCE_VIOLATION: QartodFlag.FAIL,
}


def flag_for(anomaly_type: "AnomalyType") -> QartodFlag:
    """The QARTOD flag for a finding. Anything inferred is SUSPECT."""
    return TYPE_FLAG.get(anomaly_type, QartodFlag.SUSPECT)


def qartod_label(anomaly_type: "AnomalyType") -> str:
    """`'FLATLINE (QARTOD 8 Flat Line)'`, or just the name when it is ours."""
    test = QARTOD_TEST.get(anomaly_type)
    if not test:
        return anomaly_type.value
    return f"{anomaly_type.value} (QARTOD {test[0]} {test[1]})"


#: Types that mean the INSTRUMENT is faulty -- these justify dispatching someone.
SENSOR_HEALTH_TYPES: frozenset[AnomalyType] = frozenset({
    AnomalyType.FLATLINE,
    AnomalyType.STALE,
    AnomalyType.RANGE_VIOLATION,
    AnomalyType.QUANTISATION_COLLAPSE,
    AnomalyType.ATTENUATED_SIGNAL,
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
    # Maintenance, not dispatch -- and deliberately NOT in SENSOR_HEALTH_TYPES.
    #
    # The response to an instrument offset is "check the calibration", the same
    # as DRIFT, not "send a crew to look at the water". Keeping it out of the
    # sensor-health set also keeps it out of the TELEMETRY_OUTAGE rule, which
    # fires on a block of health faults across sites: a maintenance sweep
    # recalibrating twenty instruments in an afternoon would otherwise be
    # reported as the comms link having failed.
    AnomalyType.INSTRUMENT_OFFSET,
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
    AnomalyType.ATTENUATED_SIGNAL,
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
