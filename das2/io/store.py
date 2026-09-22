"""
das2.io.store
=============

Persistence for runs, anomalies and incidents.

Deliberately plain SQL over SQLAlchemy Core -- no ORM, no models layer
duplicating `das2.models`. Three reasons:

  * The schema is owned by `migrations/010_das2_schema.sql`, which the client
    runs by hand against their SQL Server. An ORM that also thinks it owns the
    schema will eventually disagree with that file, and the file is the one the
    DBA has seen.
  * Every statement here is one the client can run in SSMS to check what the
    system did. That matters for a system whose job is to justify a dispatch.
  * It keeps SQL Server and SQLite behaving identically, which is what lets the
    whole thing be tested offline.

The one piece of real logic is `load_open_incidents`, because incident identity
across runs is the mechanism that stops one fault producing twelve alerts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from das2.models import (
    AckState,
    AnomalyType,
    Cluster,
    Incident,
    IncidentClass,
    IncidentStatus,
    PhysicalSeverity,
    SensorAnomaly,
    SensorMeta,
)

log = logging.getLogger("das2.io.store")

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent.parent / "migrations"

#: Error fragments meaning "this migration has already been applied". Matched
#: case-insensitively across SQL Server and SQLite wording.
ALREADY_APPLIED = (
    "already exists",
    "there is already an object",
    "duplicate column",
    "column names in each table must be unique",
)

#: Migrations that define the das2 schema. The 000-002 files belong to the v1
#: pipeline and are applied separately; listing them here would couple the new
#: system's setup to the old one's.
DAS2_MIGRATIONS = ("010_das2_schema.sql", "011_sensor_kind.sql")


def make_engine(url: str, *, echo: bool = False) -> Engine:
    """
    Engine for either SQL Server (pyodbc) or SQLite.

    `pool_pre_ping` is on because this process runs on a schedule with long
    idle gaps, and a SQL Server connection that has been idle for an hour is
    routinely dead by the time the next run starts. Without it the first
    statement of every run fails.
    """
    return create_engine(url, echo=echo, pool_pre_ping=True, future=True)


def _split_statements(sql: str) -> list[str]:
    """
    Split a migration file into statements.

    Naive `sql.split(";")` is wrong here and fails loudly: these migrations are
    heavily commented, and one of the comments reads

        -- ... neighbours moving together means the water moved; neighbours
        -- flat means the instrument is lying.

    whose semicolon cut a CREATE TABLE in half and produced
    `OperationalError: near "neighbours"`. So comments are stripped first, and
    the scan tracks quoting so a semicolon inside a string literal is not a
    statement boundary either.

    The `GO` batch separator is also honoured, because a DBA may well paste
    these files into SSMS and the files are written to work either way.
    """
    out: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    i, n = 0, len(sql)

    while i < n:
        ch = sql[i]

        if quote:
            buf.append(ch)
            if ch == quote:
                # '' inside a quoted string is an escaped quote, not the end.
                if i + 1 < n and sql[i + 1] == quote:
                    buf.append(sql[i + 1])
                    i += 2
                    continue
                quote = None
            i += 1
            continue

        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            i += 1
            continue

        if ch == "-" and sql.startswith("--", i):
            i = sql.find("\n", i)
            if i == -1:
                break
            continue

        if ch == "/" and sql.startswith("/*", i):
            end = sql.find("*/", i + 2)
            i = n if end == -1 else end + 2
            continue

        if ch == ";":
            out.append("".join(buf))
            buf = []
            i += 1
            continue

        buf.append(ch)
        i += 1

    out.append("".join(buf))

    statements: list[str] = []
    for raw in out:
        for part in _split_go_batches(raw):
            cleaned = part.strip()
            if cleaned:
                statements.append(cleaned)
    return statements


def _split_go_batches(block: str) -> list[str]:
    """Split on a lone `GO` line, T-SQL's batch separator."""
    parts, current = [], []
    for line in block.splitlines():
        if line.strip().upper() == "GO":
            parts.append("\n".join(current))
            current = []
        else:
            current.append(line)
    parts.append("\n".join(current))
    return parts


def apply_migrations(engine: Engine, *,
                     files: Iterable[str] = DAS2_MIGRATIONS,
                     directory: Path | None = None) -> list[str]:
    """
    Apply the idempotent schema migrations. Safe to run on every start.

    Returns the statements that were executed, so a deployment log shows what
    actually touched the database rather than just "migrations ran".
    """
    directory = directory or MIGRATIONS_DIR
    executed: list[str] = []
    with engine.begin() as conn:
        for name in files:
            path = directory / name
            if not path.exists():
                log.warning("migration %s not found at %s", name, path)
                continue
            for statement in _split_statements(path.read_text(encoding="utf-8")):
                try:
                    conn.execute(text(statement))
                    executed.append(statement.split("\n")[0][:90])
                except Exception as exc:                 # noqa: BLE001
                    # These migrations are meant to be re-run on every start,
                    # so "it is already there" is the expected outcome, not a
                    # failure. SQL Server and SQLite word it differently, and
                    # ALTER TABLE ADD has no portable IF NOT EXISTS at all.
                    message = str(exc).lower()
                    if any(hint in message for hint in ALREADY_APPLIED):
                        continue
                    log.error("migration statement failed: %s -- %s",
                              statement[:120], exc)
                    raise
    log.info("migrations applied: %d statement(s)", len(executed))
    return executed


# --------------------------------------------------------------------------- #
# Writing a run
# --------------------------------------------------------------------------- #
def _uid() -> str:
    return uuid.uuid4().hex[:32]


def _py_dt(value):
    """
    pandas Timestamp -> datetime.

    The DB-API drivers bind `datetime`, not `pandas.Timestamp`, and sqlite3
    rejects the latter outright with "type 'Timestamp' is not supported".
    Window bounds come straight off a DataFrame column, so they arrive as
    Timestamps and have to be converted before binding.
    """
    if value is None:
        return None
    to_pydatetime = getattr(value, "to_pydatetime", None)
    return to_pydatetime() if callable(to_pydatetime) else value


def save_run(engine: Engine, result, *, detector_version: str = "das2") -> None:
    """Persist one run: the run row, its anomalies, its incidents and members."""
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO das2_detection_run
              (run_id, started_at, finished_at, window_start, window_end,
               sensors_analysed, anomalies_found, incidents_open,
               detector_version, status, notes)
            VALUES
              (:run_id, :started_at, :finished_at, :window_start, :window_end,
               :sensors_analysed, :anomalies_found, :incidents_open,
               :detector_version, :status, :notes)
        """), {
            "run_id": result.run_id,
            "started_at": result.started_at,
            "finished_at": datetime.now(),
            "window_start": _py_dt(result.window_start),
            "window_end": _py_dt(result.window_end),
            "sensors_analysed": int(len(result.sensors)),
            "anomalies_found": len(result.anomalies),
            "incidents_open": len(result.incidents),
            "detector_version": detector_version,
            "status": "OK",
            "notes": json.dumps(result.stats, default=str)[:1000],
        })

        anomaly_ids: dict[str, str] = {}
        for anomaly in result.anomalies:
            aid = _uid()
            anomaly_ids[f"{anomaly.sensor.sensor_key}|{anomaly.start.isoformat()}"] = aid
            conn.execute(text("""
                INSERT INTO das2_sensor_anomaly
                  (anomaly_id, run_id, sensor_key, start_ts, end_ts, dominant_type,
                   deviation, deviation_unit, span_fraction, duration_s,
                   window_fraction, severity_score, signals_json)
                VALUES
                  (:anomaly_id, :run_id, :sensor_key, :start_ts, :end_ts, :dominant_type,
                   :deviation, :deviation_unit, :span_fraction, :duration_s,
                   :window_fraction, :severity_score, :signals_json)
            """), {
                "anomaly_id": aid,
                "run_id": result.run_id,
                "sensor_key": anomaly.sensor.sensor_key,
                "start_ts": _py_dt(anomaly.start),
                "end_ts": _py_dt(anomaly.end),
                "dominant_type": anomaly.dominant_type.value,
                "deviation": anomaly.severity.deviation,
                "deviation_unit": anomaly.severity.unit,
                "span_fraction": anomaly.severity.span_fraction,
                "duration_s": anomaly.severity.duration_s,
                "window_fraction": anomaly.severity.window_fraction,
                "severity_score": anomaly.score,
                "signals_json": json.dumps(
                    [{"type": s.type.value, "detector": s.detector,
                      "start": s.start.isoformat(), "end": s.end.isoformat(),
                      "magnitude": s.magnitude, "unit": s.unit,
                      "n_points": s.n_points, "detail": s.detail}
                     for s in anomaly.signals], default=str),
            })

        for incident in result.incidents:
            _upsert_incident(conn, incident)
            for member in incident.cluster.members:
                key = f"{member.sensor.sensor_key}|{member.start.isoformat()}"
                conn.execute(text("""
                    DELETE FROM das2_incident_member
                     WHERE incident_id = :incident_id AND sensor_key = :sensor_key
                """), {"incident_id": incident.incident_id,
                       "sensor_key": member.sensor.sensor_key})
                conn.execute(text("""
                    INSERT INTO das2_incident_member
                      (incident_id, sensor_key, anomaly_id, first_seen_at,
                       last_seen_at, contribution)
                    VALUES
                      (:incident_id, :sensor_key, :anomaly_id, :first_seen_at,
                       :last_seen_at, :contribution)
                """), {
                    "incident_id": incident.incident_id,
                    "sensor_key": member.sensor.sensor_key,
                    "anomaly_id": anomaly_ids.get(key),
                    "first_seen_at": _py_dt(member.start),
                    "last_seen_at": _py_dt(member.end),
                    "contribution": member.score,
                })

            conn.execute(text("""
                INSERT INTO das2_incident_event
                  (event_id, incident_id, ts, event_type, detail)
                VALUES (:event_id, :incident_id, :ts, :event_type, :detail)
            """), {
                "event_id": _uid(),
                "incident_id": incident.incident_id,
                "ts": datetime.now(),
                "event_type": incident.status.value.lower(),
                "detail": json.dumps(incident.detail, default=str)[:2000],
            })

    log.info("run %s persisted: %d anomalies, %d incidents",
             result.run_id, len(result.anomalies), len(result.incidents))


def _upsert_incident(conn, incident: Incident) -> None:
    """
    Insert or update one incident, preserving its acknowledgement.

    Written as delete-then-insert of the mutable columns rather than a MERGE,
    because MERGE syntax differs between SQL Server and SQLite and this code
    has to run identically on both. The ack columns are deliberately NOT
    overwritten on update: an operator who acknowledged an incident an hour ago
    must not be re-paged because the severity moved by a point.
    """
    c = incident.cluster
    params = {
        "incident_id": incident.incident_id,
        "incident_class": incident.incident_class.value,
        "status": incident.status.value,
        "severity": incident.severity,
        "priority": incident.priority.value,
        "region": str(c.region or ""),
        "centroid_lat": c.centroid_lat,
        "centroid_lon": c.centroid_lon,
        "radius_m": c.radius_m,
        "sensor_count": len(c.members),
        "site_count": len(c.sites),
        "equipment_types": ",".join(sorted(c.equipment_types))[:400],
        "neighbour_correlation": incident.neighbour_correlation,
        "rainfall_mm": incident.rainfall_mm,
        "narrative": incident.narrative,
        "recommendation": incident.recommendation[:500],
        "opened_at": _py_dt(incident.opened_at),
        "last_seen_at": _py_dt(incident.last_seen_at),
        "resolved_at": _py_dt(incident.resolved_at),
    }
    existing = conn.execute(
        text("SELECT incident_id FROM das2_incident WHERE incident_id = :incident_id"),
        {"incident_id": incident.incident_id}).fetchone()

    if existing:
        conn.execute(text("""
            UPDATE das2_incident SET
              incident_class = :incident_class, status = :status,
              severity = :severity, priority = :priority, region = :region,
              centroid_lat = :centroid_lat, centroid_lon = :centroid_lon,
              radius_m = :radius_m, sensor_count = :sensor_count,
              site_count = :site_count, equipment_types = :equipment_types,
              neighbour_correlation = :neighbour_correlation,
              rainfall_mm = :rainfall_mm, narrative = :narrative,
              recommendation = :recommendation, last_seen_at = :last_seen_at,
              resolved_at = :resolved_at
            WHERE incident_id = :incident_id
        """), params)
    else:
        conn.execute(text("""
            INSERT INTO das2_incident
              (incident_id, incident_class, status, severity, priority, region,
               centroid_lat, centroid_lon, radius_m, sensor_count, site_count,
               equipment_types, neighbour_correlation, rainfall_mm, narrative,
               recommendation, opened_at, last_seen_at, resolved_at, ack_state)
            VALUES
              (:incident_id, :incident_class, :status, :severity, :priority, :region,
               :centroid_lat, :centroid_lon, :radius_m, :sensor_count, :site_count,
               :equipment_types, :neighbour_correlation, :rainfall_mm, :narrative,
               :recommendation, :opened_at, :last_seen_at, :resolved_at, 'NONE')
        """), params)


# --------------------------------------------------------------------------- #
# Reading back
# --------------------------------------------------------------------------- #
def load_open_incidents(engine: Engine) -> list[Incident]:
    """
    Incidents still open, for the next run to match against.

    Reconstructs enough of each incident for `reconcile` to work: the member
    sensor keys, the region, and the identity fields. The full member anomalies
    are NOT rebuilt -- matching only needs the key set, and rebuilding detector
    output from storage would risk the stored copy disagreeing with what the
    detector would say now.
    """
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT incident_id, incident_class, status, severity, region,
                   centroid_lat, centroid_lon, radius_m, opened_at, last_seen_at,
                   ack_state, ack_by, narrative, neighbour_correlation, rainfall_mm
              FROM das2_incident
             WHERE status <> 'RESOLVED'
        """)).mappings().all()

        out: list[Incident] = []
        for row in rows:
            members = conn.execute(text("""
                SELECT sensor_key, first_seen_at, last_seen_at, contribution
                  FROM das2_incident_member
                 WHERE incident_id = :incident_id
            """), {"incident_id": row["incident_id"]}).mappings().all()

            cluster = Cluster(
                members=[_stub_member(m) for m in members],
                region=row["region"] or None,
                centroid_lat=row["centroid_lat"],
                centroid_lon=row["centroid_lon"],
                radius_m=row["radius_m"] or 0.0,
            )
            out.append(Incident(
                incident_id=row["incident_id"],
                cluster=cluster,
                incident_class=_enum(IncidentClass, row["incident_class"],
                                     IncidentClass.WATCH),
                status=_enum(IncidentStatus, row["status"], IncidentStatus.OPEN),
                severity=row["severity"] or 0.0,
                opened_at=_dt(row["opened_at"]),
                last_seen_at=_dt(row["last_seen_at"]),
                ack_state=_enum(AckState, row["ack_state"], AckState.NONE),
                ack_by=row["ack_by"],
                narrative=row["narrative"] or "",
                neighbour_correlation=row["neighbour_correlation"],
                rainfall_mm=row["rainfall_mm"],
            ))
    log.info("loaded %d open incident(s) from the previous run(s)", len(out))
    return out


def _stub_member(row) -> SensorAnomaly:
    """
    A member reconstructed only as far as matching needs.

    The sensor key and window are real; the type is a placeholder, because
    storage is not the authority on what the detector would call this today.
    """
    start = _dt(row["first_seen_at"]) or datetime.now()
    end = _dt(row["last_seen_at"]) or start
    return SensorAnomaly(
        sensor=SensorMeta(sensor_key=str(row["sensor_key"]),
                          description="", equipment=""),
        start=start, end=end,
        dominant_type=AnomalyType.RESIDUAL_OUTLIER,
        severity=PhysicalSeverity(duration_s=(end - start).total_seconds()),
    )


def record_ack(engine: Engine, incident_ref: str, state: AckState,
               operator: str, *, note: str = "") -> bool:
    """
    Apply an operator's button press.

    `incident_ref` may be a suffix, because `callback_data` is capped at 64
    bytes and long ids are truncated when the keyboard is built. Matched by
    suffix rather than exact id for that reason.
    """
    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT incident_id FROM das2_incident
             WHERE incident_id = :ref OR incident_id LIKE :like
             ORDER BY last_seen_at DESC
        """), {"ref": incident_ref, "like": f"%{incident_ref}"}).fetchone()
        if not row:
            log.warning("ack for unknown incident %r", incident_ref)
            return False
        incident_id = row[0]

        conn.execute(text("""
            UPDATE das2_incident
               SET ack_state = :state, ack_by = :operator,
                   ack_at = :ts, ack_note = :note
             WHERE incident_id = :incident_id
        """), {"state": state.value, "operator": operator, "ts": datetime.now(),
               "note": note[:500], "incident_id": incident_id})

        conn.execute(text("""
            INSERT INTO das2_incident_event
              (event_id, incident_id, ts, event_type, detail)
            VALUES (:event_id, :incident_id, :ts, 'acknowledged', :detail)
        """), {"event_id": _uid(), "incident_id": incident_id,
               "ts": datetime.now(),
               "detail": json.dumps({"state": state.value, "by": operator})})

        # A false-alarm tap is a label, and labels are the scarcest thing this
        # system has. Recorded separately so the feedback table stays the one
        # place to look for ground truth.
        if state is AckState.FALSE_ALARM:
            conn.execute(text("""
                INSERT INTO das2_feedback
                  (feedback_id, incident_id, label, operator, created_at, note)
                VALUES (:fid, :incident_id, 'noise', :operator, :ts, :note)
            """), {"fid": _uid(), "incident_id": incident_id,
                   "operator": operator, "ts": datetime.now(), "note": note[:500]})
        elif state is AckState.DISPATCHED:
            conn.execute(text("""
                INSERT INTO das2_feedback
                  (feedback_id, incident_id, label, operator, created_at, note)
                VALUES (:fid, :incident_id, 'real', :operator, :ts, :note)
            """), {"fid": _uid(), "incident_id": incident_id,
                   "operator": operator, "ts": datetime.now(), "note": note[:500]})
    log.info("ack recorded: %s -> %s by %s", incident_id, state.value, operator)
    return True


def record_delivery(engine: Engine, incident_id: str, channel: str, *,
                    message_id: str | None = None, payload: str = "",
                    suppressed: bool = False, reason: str = "") -> None:
    """
    Log what was actually sent.

    `payload_hash` is what makes delivery dedup possible: an incident whose
    alert text has not changed since the last send does not need sending again,
    which is the last line of defence against the twelve-messages-per-fault
    behaviour.
    """
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO das2_alert_delivery
              (delivery_id, incident_id, channel, sent_at, message_id,
               payload_hash, suppressed, suppress_reason)
            VALUES
              (:delivery_id, :incident_id, :channel, :sent_at, :message_id,
               :payload_hash, :suppressed, :suppress_reason)
        """), {
            "delivery_id": _uid(), "incident_id": incident_id, "channel": channel,
            "sent_at": datetime.now(), "message_id": message_id,
            "payload_hash": hashlib.sha256(payload.encode()).hexdigest()[:64],
            "suppressed": 1 if suppressed else 0,
            "suppress_reason": reason[:200],
        })


def already_delivered(engine: Engine, incident_id: str, payload: str,
                      channel: str = "telegram") -> bool:
    """True when this exact alert text was already sent for this incident."""
    digest = hashlib.sha256(payload.encode()).hexdigest()[:64]
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT 1 FROM das2_alert_delivery
             WHERE incident_id = :incident_id AND channel = :channel
               AND payload_hash = :digest AND suppressed = 0
        """), {"incident_id": incident_id, "channel": channel,
               "digest": digest}).fetchone()
    return row is not None


def upsert_sensors(engine: Engine, sensors) -> int:
    """Refresh the sensor inventory. Cheap, and keeps the DB self-describing."""
    written = 0
    with engine.begin() as conn:
        for _, row in sensors.iterrows():
            params = {
                "sensor_key": str(row["sensor_key"]),
                "description": str(row.get("description") or "")[:400],
                "equipment": str(row.get("equipment") or "")[:64],
                "signal_type": str(row.get("signal_type") or "")[:16],
                "kind": str(row.get("kind") or "")[:16],
                "unit": str(row.get("unit") or "")[:32],
                "rtu_number": _str_or_none(row.get("rtu_number")),
                "site": _str_or_none(row.get("site")),
                "latitude": _float_or_none(row.get("latitude")),
                "longitude": _float_or_none(row.get("longitude")),
                "region": _str_or_none(row.get("region")),
                "planning_area": _str_or_none(row.get("planning_area")),
                "alertable": 1 if row.get("alertable", True) else 0,
            }
            exists = conn.execute(
                text("SELECT 1 FROM das2_sensor WHERE sensor_key = :sensor_key"),
                {"sensor_key": params["sensor_key"]}).fetchone()
            if exists:
                conn.execute(text("""
                    UPDATE das2_sensor SET description=:description,
                      equipment=:equipment, signal_type=:signal_type, kind=:kind,
                      unit=:unit, rtu_number=:rtu_number, site=:site,
                      latitude=:latitude, longitude=:longitude, region=:region,
                      planning_area=:planning_area, alertable=:alertable
                     WHERE sensor_key=:sensor_key
                """), params)
            else:
                conn.execute(text("""
                    INSERT INTO das2_sensor
                      (sensor_key, description, equipment, signal_type, kind, unit,
                       rtu_number, site, latitude, longitude, region,
                       planning_area, alertable)
                    VALUES
                      (:sensor_key, :description, :equipment, :signal_type, :kind,
                       :unit, :rtu_number, :site, :latitude, :longitude, :region,
                       :planning_area, :alertable)
                """), params)
            written += 1
    return written


# --------------------------------------------------------------------------- #
def _enum(cls, value, default):
    try:
        return cls(value)
    except (ValueError, TypeError):
        return default


def _dt(value) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _str_or_none(value) -> str | None:
    if value is None:
        return None
    text_value = str(value)
    return None if text_value in ("nan", "NaT", "<NA>", "") else text_value[:200]


def _float_or_none(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out else out          # NaN check without importing math
