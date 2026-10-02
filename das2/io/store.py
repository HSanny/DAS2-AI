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
import re
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from sqlalchemy import String, Text, bindparam, create_engine, text
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

#: How much of the audit trail each column can hold.
#:
#: `das2_detection_run.notes` is VARCHAR(1000) and cannot be widened without a
#: migration. `das2_incident_event.detail` is TEXT, so its budget is a choice
#: rather than a limit -- bounded only so one pathological incident cannot
#: write megabytes into a row nobody will read.
NOTES_LIMIT = 1000
DETAIL_LIMIT = 8000


def fit_json(payload: Any, limit: int, *, keep: Iterable[str] = ()) -> str:
    """
    JSON that fits `limit` characters and can still be read back.

    Slicing the string does not truncate the data, it destroys it:
    `json.dumps(...)[:1000]` ends in the middle of a key and nothing can parse
    the row afterwards. Both audit columns in this schema were written that
    way, and both overflowed the moment incidents started carrying the
    verification panel -- the run summary at 2,925 characters into a
    1,000-character column, and the largest incident's detail at 2,790 into a
    2,000-character slice. It was the biggest, most important incidents whose
    record was destroyed, and nothing raised.

    So the payload is shrunk STRUCTURALLY: drop whole top-level entries,
    largest first, until what remains fits, and record what went in `_omitted`
    so a reader knows the row is partial rather than wondering. `keep` names
    the entries to sacrifice last.

    The result always parses. That is the whole contract.
    """
    if not isinstance(payload, dict):
        payload = {"value": payload}

    protected = set(keep)
    working = dict(payload)
    omitted: list[str] = []

    while True:
        candidate: dict[str, Any] = dict(working)
        if omitted:
            candidate["_omitted"] = omitted
        rendered = json.dumps(candidate, default=str)
        if len(rendered) <= limit:
            return rendered
        if not working:
            # Nothing left to drop and the bookkeeping alone is too big. A
            # count always fits, and still says the row is partial.
            return json.dumps({"_omitted_all": len(omitted)})
        droppable = [k for k in working if k not in protected] or list(working)
        victim = max(droppable,
                     key=lambda k: len(json.dumps(working[k], default=str)))
        working.pop(victim)
        omitted.append(victim)

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
DAS2_MIGRATIONS = ("010_das2_schema.sql", "011_sensor_kind.sql",
                   "012_profile_days.sql")


#: Seconds to allow for the SQL Server LOGIN handshake.
#:
#: ODBC Driver 18 defaults to 15, and on the client's historian that is not
#: enough: every connection failed at exactly 15 seconds with
#:
#:     ('HYT00', '... Login timeout expired (0) (SQLDriverConnect)')
#:
#: while a plain TCP connect to 1433 from the same container succeeded
#: immediately. A login taking longer than fifteen seconds is unusual -- a
#: loaded server, a slow directory lookup, a reverse-DNS wait -- but it is the
#: server's business, not a reason for an hourly job to give up.
#:
#: 60 rather than 30 because the cost is asymmetric. A server that is merely
#: slow costs one longer wait; a run abandoned for want of a few seconds costs
#: the whole hour, and the incident layer needs the database to tell a new
#: incident from one already sent. A genuinely dead server still fails, just
#: later.
DEFAULT_LOGIN_TIMEOUT_S = 60


def make_engine(url: str, *, echo: bool = False,
                login_timeout_s: int = DEFAULT_LOGIN_TIMEOUT_S) -> Engine:
    """
    Engine for either SQL Server (pyodbc) or SQLite.

    `pool_pre_ping` is on because this process runs on a schedule with long
    idle gaps, and a SQL Server connection that has been idle for an hour is
    routinely dead by the time the next run starts. Without it the first
    statement of every run fails.

    `login_timeout_s` reaches pyodbc as its `timeout` argument, which sets
    SQL_ATTR_LOGIN_TIMEOUT. It is applied only to pyodbc URLs; SQLite's driver
    has no such parameter and rejects it.
    """
    kwargs: dict = {}
    if url.startswith("mssql+pyodbc:") and login_timeout_s:
        kwargs["connect_args"] = {"timeout": int(login_timeout_s)}
    engine = create_engine(url, echo=echo, pool_pre_ping=True, future=True,
                           **kwargs)

    # Bulk inserts, or das2_reading takes all night.
    #
    # Without this, pyodbc sends an executemany as one round trip PER ROW. The
    # client's first scheduled run wrote 5,627,250 readings at 165 rows a
    # second and took NINE AND A HALF HOURS -- so the 19:05 through 04:05 runs
    # never happened, and the system managed two runs in a day instead of
    # twenty-four. fast_executemany packs the parameters into arrays and sends
    # them in blocks.
    if url.startswith("mssql+pyodbc:"):
        from sqlalchemy import event

        @event.listens_for(engine, "before_cursor_execute")
        def _bulk(conn, cursor, statement, parameters, context, executemany):
            if executemany:
                cursor.fast_executemany = True

    return engine


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


#: Rewrites that turn the portable migration SQL into T-SQL.
#:
#: The .sql files are written in the SQLite dialect because that is what the
#: tests run against, and 010's header claimed the result was portable. It was
#: not. Three constructs in it are wrong on SQL Server, and every one of them
#: stayed invisible until the system met a real server:
#:
#:   CREATE TABLE IF NOT EXISTS    No such form in T-SQL, in any version, and
#:                                 `DROP TABLE IF EXISTS` existing makes it
#:                                 look supported. Confirmed against the
#:                                 client's SQL Server:
#:                                   [42000] Incorrect syntax near the
#:                                   keyword 'IF'. (156)
#:
#:   TIMESTAMP                     NOT a datetime on SQL Server. It is a
#:                                 deprecated synonym for ROWVERSION, an
#:                                 auto-generated binary(8) row counter, so the
#:                                 tables would be created happily and then
#:                                 reject every INSERT with "Cannot insert an
#:                                 explicit value into a timestamp column."
#:                                 This is the dangerous one: it fails late,
#:                                 far from its cause, and only after the
#:                                 migration has reported success.
#:
#:   TEXT                          Deprecated since 2005 and unusable with
#:                                 normal string comparison.
#:
#: Comments are stripped before this runs, and the migrations contain exactly
#: one string literal ('NONE'), so a word-boundary substitution cannot corrupt
#: anything. No column is named `text` or `timestamp`; a test asserts that,
#: because the day one is added this has to become a parser.
_MSSQL_REWRITES = (
    (r"(?is)\b(CREATE\s+(?:UNIQUE\s+)?(?:TABLE|INDEX))\s+IF\s+NOT\s+EXISTS\s+", r"\1 "),
    (r"(?i)\bTIMESTAMP\b", "DATETIME2"),
    (r"(?i)\bTEXT\b", "VARCHAR(MAX)"),
)

#: Existence guards, so the T-SQL is idempotent in its own right.
#:
#: Dropping IF NOT EXISTS alone would leave re-runs depending on
#: ALREADY_APPLIED matching the text of the error -- and SQL Server localises
#: its messages. On a non-English server "There is already an object named ..."
#: never appears, the match fails, and the second run raises instead of
#: continuing. Matching English error strings is not a sound way to decide
#: whether a table exists.
#:
#: It also makes the printed script safe to paste into SSMS twice, which is
#: what anyone reviewing it will do.
_RE_CREATE_TABLE = re.compile(r"(?is)^\s*CREATE\s+TABLE\s+(\w+)\s*\(")
_RE_CREATE_INDEX = re.compile(
    r"(?is)^\s*CREATE\s+(?:UNIQUE\s+)?INDEX\s+(\w+)\s+ON\s+(\w+)\s*\(")
_RE_ALTER_ADD = re.compile(r"(?is)^\s*ALTER\s+TABLE\s+(\w+)\s+ADD\s+(\w+)\b")


def _guard_mssql(statement: str) -> str:
    """Prefix a T-SQL existence test, so the statement is a no-op if applied."""
    match = _RE_CREATE_TABLE.match(statement)
    if match:
        return (f"IF OBJECT_ID(N'{match.group(1)}', N'U') IS NULL\n"
                f"{statement}")

    match = _RE_CREATE_INDEX.match(statement)
    if match:
        index, table = match.group(1), match.group(2)
        return (f"IF NOT EXISTS (SELECT 1 FROM sys.indexes\n"
                f"               WHERE name = N'{index}'\n"
                f"                 AND object_id = OBJECT_ID(N'{table}'))\n"
                f"{statement}")

    match = _RE_ALTER_ADD.match(statement)
    if match:
        table, column = match.group(1), match.group(2)
        return (f"IF COL_LENGTH(N'{table}', N'{column}') IS NULL\n"
                f"{statement}")

    return statement


def translate_sql(statement: str, dialect: str) -> str:
    """
    Adapt one portable migration statement to `dialect`.

    Everything but `mssql` is returned untouched: SQLite is the dialect the
    files are written in.
    """
    if dialect != "mssql":
        return statement
    for pattern, replacement in _MSSQL_REWRITES:
        statement = re.sub(pattern, replacement, statement)
    return _guard_mssql(statement)


def render_migrations(dialect: str, *,
                      files: Iterable[str] = DAS2_MIGRATIONS,
                      directory: Path | None = None) -> list[str]:
    """
    The statements `apply_migrations` would run, without running them.

    Shares the split-and-translate path with `apply_migrations`, so a printed
    script and an applied one cannot disagree. That matters more than it
    sounds: the alternative is a hand-maintained T-SQL copy of the schema,
    which drifts the first time a column is added and is then wrong in a way
    nobody notices until a run fails on a missing column.
    """
    directory = directory or MIGRATIONS_DIR
    out: list[str] = []
    for name in files:
        path = directory / name
        if not path.exists():
            log.warning("migration %s not found at %s", name, path)
            continue
        for raw in _split_statements(path.read_text(encoding="utf-8")):
            out.append(translate_sql(raw, dialect))
    return out


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
            for raw in _split_statements(path.read_text(encoding="utf-8")):
                statement = translate_sql(raw, engine.dialect.name)
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


#: Rows per round trip. Matches `save_readings`, for the same reason: pyodbc
#: packs an executemany into parameter arrays, and the array has to fit in
#: memory on both ends.
WRITE_CHUNK = 1000

#: Columns wide enough that the driver must be told their type.
#:
#: With `fast_executemany` on, pyodbc sizes a string parameter from what it
#: infers for the batch rather than from the column, and a batch whose first row
#: carries a 200-character narrative followed by a row carrying 4,000 characters
#: is the documented way to get "String data, right truncation" -- or, worse, a
#: silently shortened value. Declaring the type makes SQLAlchemy emit
#: `setinputsizes`, so the driver sizes the parameter from the schema instead of
#: from the first row it happens to see.
#:
#: Only the wide ones are listed. Keys, timestamps and floats are fixed-width
#: and infer correctly.
_WIDE_TEXT = {
    "signals_json": Text(),
    "detail": Text(),
    "narrative": Text(),
    "notes": Text(),
    "recommendation": String(500),
    "equipment_types": String(400),
}


def _typed(sql: str):
    """A statement with its wide text parameters typed. See `_WIDE_TEXT`."""
    stmt = text(sql)
    present = {name: type_ for name, type_ in _WIDE_TEXT.items()
               if f":{name}" in sql}
    if present:
        stmt = stmt.bindparams(*(bindparam(name, type_=type_)
                                 for name, type_ in present.items()))
    return stmt


def _write_many(conn, sql: str, rows: list[dict], *,
                chunk: int = WRITE_CHUNK) -> None:
    """
    Execute one statement over many rows, in as few round trips as possible.

    This is the whole fix for the persistence speed. `save_run` used to call
    `conn.execute` once per anomaly, once per incident member and once per
    incident event, which on the client's run was about 3,500 statements and
    3,500 network round trips to SQL Server: 12 minutes 45 seconds, roughly
    4.6 rows a second. The readings path in the same run wrote 659,006 rows in
    2 minutes 1 second -- 5,450 a second, to the same database over the same
    link -- because it hands the driver a list and lets it batch.

    The per-row fallback is not decoration. `fast_executemany` has real driver
    quirks around parameter typing, and the typed binds above address the one
    that bites hardest, but a run that fails to record what it found is worse
    than a run that records it slowly. A batch that raises is retried row by
    row, and the loss is speed with a line in the log, not the run.
    """
    if not rows:
        return
    for start in range(0, len(rows), chunk):
        batch = rows[start:start + chunk]
        stmt = _typed(sql)
        try:
            conn.execute(stmt, batch)
        except Exception as exc:                              # noqa: BLE001
            log.warning("batched write of %d row(s) failed (%s); "
                        "falling back to row-by-row",
                        len(batch), str(exc)[:160])
            for row in batch:
                conn.execute(stmt, row)


def save_run(engine: Engine, result, *, detector_version: str = "das2") -> None:
    """
    Persist one run: the run row, its anomalies, its incidents and members.

    Every statement below is executed over a LIST of rows rather than once per
    row -- see `_write_many` for the measurement that motivated it.
    """
    with engine.begin() as conn:
        conn.execute(_typed("""
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
            # The four an operator asks about first: what was found, what was
            # sent, what was held, and what the run could see at all.
            "notes": fit_json(result.stats, NOTES_LIMIT,
                              keep=("detection", "incidents", "selection",
                                    "ingest")),
        })

        anomaly_ids: dict[str, str] = {}
        anomaly_rows: list[dict] = []
        for anomaly in result.anomalies:
            aid = _uid()
            anomaly_ids[f"{anomaly.sensor.sensor_key}|{anomaly.start.isoformat()}"] = aid
            anomaly_rows.append({
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
        _write_many(conn, """
            INSERT INTO das2_sensor_anomaly
              (anomaly_id, run_id, sensor_key, start_ts, end_ts, dominant_type,
               deviation, deviation_unit, span_fraction, duration_s,
               window_fraction, severity_score, signals_json)
            VALUES
              (:anomaly_id, :run_id, :sensor_key, :start_ts, :end_ts, :dominant_type,
               :deviation, :deviation_unit, :span_fraction, :duration_s,
               :window_fraction, :severity_score, :signals_json)
        """, anomaly_rows)

        _upsert_incidents(conn, result.incidents)

        member_keys: list[dict] = []
        member_rows: list[dict] = []
        event_rows: list[dict] = []
        now = datetime.now()
        for incident in result.incidents:
            for member in incident.cluster.members:
                key = f"{member.sensor.sensor_key}|{member.start.isoformat()}"
                member_keys.append({"incident_id": incident.incident_id,
                                    "sensor_key": member.sensor.sensor_key})
                member_rows.append({
                    "incident_id": incident.incident_id,
                    "sensor_key": member.sensor.sensor_key,
                    "anomaly_id": anomaly_ids.get(key),
                    "first_seen_at": _py_dt(member.start),
                    "last_seen_at": _py_dt(member.end),
                    "contribution": member.score,
                })
            event_rows.append({
                "event_id": _uid(),
                "incident_id": incident.incident_id,
                "ts": now,
                "event_type": incident.status.value.lower(),
                "detail": fit_json(incident.detail, DETAIL_LIMIT,
                                   keep=("evidence", "signature",
                                         "rain_evidence")),
            })

        # Delete every membership first, then insert every membership. Phased
        # rather than interleaved per member, which is what lets each phase be
        # one batch -- and it is safe because the two sets are identical: a row
        # deleted here is a row re-inserted below.
        _write_many(conn, """
            DELETE FROM das2_incident_member
             WHERE incident_id = :incident_id AND sensor_key = :sensor_key
        """, member_keys)
        _write_many(conn, """
            INSERT INTO das2_incident_member
              (incident_id, sensor_key, anomaly_id, first_seen_at,
               last_seen_at, contribution)
            VALUES
              (:incident_id, :sensor_key, :anomaly_id, :first_seen_at,
               :last_seen_at, :contribution)
        """, member_rows)
        _write_many(conn, """
            INSERT INTO das2_incident_event
              (event_id, incident_id, ts, event_type, detail)
            VALUES (:event_id, :incident_id, :ts, :event_type, :detail)
        """, event_rows)

    log.info("run %s persisted: %d anomalies, %d incidents, "
             "%d member row(s)",
             result.run_id, len(result.anomalies), len(result.incidents),
             len(member_rows))


def _incident_params(incident: Incident) -> dict:
    c = incident.cluster
    return {
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


def _upsert_incidents(conn, incidents) -> None:
    """
    Insert or update a run's incidents, preserving their acknowledgements.

    Update-or-insert rather than a MERGE, because MERGE syntax differs between
    SQL Server and SQLite and this code has to run identically on both. The ack
    columns are deliberately NOT overwritten on update: an operator who
    acknowledged an incident an hour ago must not be re-paged because the
    severity moved by a point.

    Which incidents already exist is answered in ONE query over the whole run
    rather than one per incident. That matters more than it looks: the SELECT
    was the only statement here that had to come back before the next could be
    sent, so on the client's run it was 97 serial round trips before a single
    row was written.
    """
    incidents = list(incidents)
    if not incidents:
        return

    ids = [i.incident_id for i in incidents]
    existing: set[str] = set()
    # Chunked because an IN list is parameterised, and SQL Server rejects a
    # statement with more than 2,100 parameters -- a limit a busy run reaches.
    for start in range(0, len(ids), 500):
        batch = ids[start:start + 500]
        names = [f"id{n}" for n in range(len(batch))]
        rows = conn.execute(
            text("SELECT incident_id FROM das2_incident WHERE incident_id IN ("
                 + ", ".join(f":{n}" for n in names) + ")"),
            dict(zip(names, batch))).fetchall()
        existing.update(str(r[0]) for r in rows)

    updates = [_incident_params(i) for i in incidents
               if i.incident_id in existing]
    inserts = [_incident_params(i) for i in incidents
               if i.incident_id not in existing]

    if updates:
        _write_many(conn, """
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
        """, updates)

    if inserts:
        _write_many(conn, """
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
        """, inserts)


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
    """
    Refresh the sensor inventory. Keeps the DB self-describing.

    Batched, like `save_run` and `save_baselines`, and this one runs on EVERY
    run rather than once a day: a row-per-statement version is three statements
    per sensor -- a SELECT that has to come back before either write can be
    sent, then an UPDATE or an INSERT -- which on the 2,672 analysed sensors is
    about 8,000 serial round trips to do a job whose answer barely changes
    between runs.
    """
    rows: list[dict] = []
    for _, row in sensors.iterrows():
        rows.append({
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
        })

    if not rows:
        return 0

    with engine.begin() as conn:
        # Which keys are already there, in one query per 500 rather than one per
        # sensor. Chunked because an IN list is parameterised and SQL Server
        # caps a statement at 2,100 parameters.
        existing: set[str] = set()
        keys = [r["sensor_key"] for r in rows]
        for start in range(0, len(keys), 500):
            batch = keys[start:start + 500]
            names = [f"k{n}" for n in range(len(batch))]
            found = conn.execute(
                text("SELECT sensor_key FROM das2_sensor WHERE sensor_key IN ("
                     + ", ".join(f":{n}" for n in names) + ")"),
                dict(zip(names, batch))).fetchall()
            existing.update(str(r[0]) for r in found)

        _write_many(conn, """
            UPDATE das2_sensor SET description=:description,
              equipment=:equipment, signal_type=:signal_type, kind=:kind,
              unit=:unit, rtu_number=:rtu_number, site=:site,
              latitude=:latitude, longitude=:longitude, region=:region,
              planning_area=:planning_area, alertable=:alertable
             WHERE sensor_key=:sensor_key
        """, [r for r in rows if r["sensor_key"] in existing])
        _write_many(conn, """
            INSERT INTO das2_sensor
              (sensor_key, description, equipment, signal_type, kind, unit,
               rtu_number, site, latitude, longitude, region,
               planning_area, alertable)
            VALUES
              (:sensor_key, :description, :equipment, :signal_type, :kind,
               :unit, :rtu_number, :site, :latitude, :longitude, :region,
               :planning_area, :alertable)
        """, [r for r in rows if r["sensor_key"] not in existing])
    return len(rows)


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


# --------------------------------------------------------------------------- #
# Time-of-day baselines, written by the daily profile job
# --------------------------------------------------------------------------- #
def _as_datetime(value):
    """
    A datetime from whatever the driver returned for MAX(ts).

    SQL Server hands back a real datetime; SQLite stores timestamps as TEXT
    and hands back a string. Subtracting a timedelta from that string raises,
    and the caller's except swallowed it -- so the incremental filter silently
    did nothing on SQLite and the whole window was re-sent, which is exactly
    the behaviour it was written to remove. A test now proves the reduction
    rather than the intent.
    """
    if value is None or isinstance(value, datetime):
        return value
    try:
        import pandas as pd
        parsed = pd.to_datetime(value, errors="coerce")
        return None if parsed is None or parsed is pd.NaT else parsed.to_pydatetime()
    except Exception:                                      # noqa: BLE001
        return None


#: How far back before the newest stored reading to re-offer rows.
#:
#: 12% of sensors in this feed carry out-of-order timestamps, so a reading for
#: 14:03 can arrive after one for 14:07. Filtering strictly on MAX(ts) would
#: drop those permanently. An hour of overlap catches them, and the duplicate
#: guard makes re-sending the rest harmless.
LATE_ARRIVAL_GRACE_HOURS = 1


def save_readings(engine: Engine, readings, *, chunk: int = 5000) -> int:
    """
    Persist this run's readings, so the daily job has history to work from.

    Without this nothing ever populates `das2_reading`, and DRIFT,
    NOISE_BURST and the time-of-day baselines would have no source except the
    v1 `data` table -- which may not be readable by this login and is not
    guaranteed to stay. The hourly run is the only thing that sees the CSVs, so
    it is the only thing that can fill the store.

    Existing rows are left alone. Consecutive runs overlap by most of the
    window, so the great majority of what arrives here has been seen before;
    skipping duplicates by primary key is far cheaper than rewriting them, and
    keeps the first-seen timestamp honest.
    """
    if readings is None or len(readings) == 0:
        return 0

    # Only the tail. This is the difference between a run and an overnight job.
    #
    # Each run re-reads the whole 72-hour window, so consecutive runs overlap
    # by 71 of 72 hours and ~98.6% of what arrives here is already stored.
    # Sending all of it and letting NOT EXISTS discard the duplicates makes the
    # database do 5.6 million index lookups an hour to keep 78,000 rows. On the
    # client's first scheduled run that took nine and a half hours, and the
    # next nine hourly runs simply never happened.
    #
    # The overlap below is deliberate: filtering strictly on MAX(ts) would drop
    # any reading that arrives late for an hour already written, and this feed
    # has out-of-order timestamps in 12% of sensors. An hour of slack lets
    # those through, and NOT EXISTS still makes re-sending them harmless.
    try:
        with engine.connect() as conn:
            last = conn.execute(
                text("SELECT MAX(ts) FROM das2_reading")).scalar()
        last = _as_datetime(last)
        if last is not None:
            cutoff = last - timedelta(hours=LATE_ARRIVAL_GRACE_HOURS)
            before = len(readings)
            readings = readings[readings["ts"] > cutoff]
            log.info("readings to persist: %d of %d (the rest predate %s "
                     "and are already stored)",
                     len(readings), before, cutoff)
            if len(readings) == 0:
                return 0
    except Exception as exc:                              # noqa: BLE001
        # A first run has no table content to compare against, and a failed
        # probe must not stop the write -- it only costs the old behaviour.
        log.debug("could not read the last stored reading (%s); "
                  "persisting the whole window", str(exc)[:120])

    rows = [
        {"sensor_key": str(r["sensor_key"]),
         "ts": _py_dt(r["ts"]),
         "value": (None if r["value"] is None or r["value"] != r["value"]
                   else float(r["value"]))}
        for r in readings[["sensor_key", "ts", "value"]].to_dict("records")
    ]

    dialect = engine.dialect.name
    if dialect == "sqlite":
        sql = ("INSERT OR IGNORE INTO das2_reading (sensor_key, ts, value) "
               "VALUES (:sensor_key, :ts, :value)")
    elif dialect == "mssql":
        # No INSERT IGNORE in T-SQL. NOT EXISTS is portable, index-backed on the
        # primary key, and avoids MERGE -- which needs more privilege and has a
        # long history of surprising locking behaviour.
        sql = ("INSERT INTO das2_reading (sensor_key, ts, value) "
               "SELECT :sensor_key, :ts, :value WHERE NOT EXISTS ("
               "SELECT 1 FROM das2_reading WHERE sensor_key = :sensor_key "
               "AND ts = :ts)")
    else:
        sql = ("INSERT INTO das2_reading (sensor_key, ts, value) "
               "VALUES (:sensor_key, :ts, :value) ON CONFLICT DO NOTHING")

    written = 0
    for start in range(0, len(rows), chunk):
        batch = rows[start:start + chunk]
        try:
            with engine.begin() as conn:
                conn.execute(text(sql), batch)
            written += len(batch)
        except Exception as exc:                          # noqa: BLE001
            # Never fatal. Losing history costs the daily job accuracy; losing
            # the run costs the alert.
            log.warning("could not persist readings batch: %s", str(exc)[:160])
            break
    log.info("persisted %d reading(s) to das2_reading", written)
    return written


def prune_readings(engine: Engine, retention_days: int) -> int:
    """
    Drop readings older than the retention window.

    Without this the table grows without bound: months x 2,672 sensors x 120 s
    is on the order of 10^8 rows, and an operations database quietly filling up
    is the sort of failure that takes the monitoring down along with it.

    The retention window has to stay comfortably above what the profile job
    wants -- it reads 28 days -- so pruning below that would silently disable
    DRIFT and the baselines. Returns the number of rows removed.
    """
    if retention_days <= 0:
        return 0
    cutoff = datetime.now() - timedelta(days=retention_days)
    try:
        with engine.begin() as conn:
            result = conn.execute(
                text("DELETE FROM das2_reading WHERE ts < :cutoff"),
                {"cutoff": cutoff})
            removed = int(result.rowcount or 0)
    except Exception as exc:                              # noqa: BLE001
        log.warning("could not prune das2_reading: %s", str(exc)[:160])
        return 0
    if removed:
        log.info("pruned %d reading(s) older than %d days", removed, retention_days)
    return removed


def save_baselines(engine: Engine, baselines) -> int:
    """
    Replace the stored time-of-day profiles.

    Replace rather than merge: the daily job recomputes each sensor's whole
    table from the full history window, so a merge would leave buckets behind
    from a period that is no longer in scope and quietly age the baseline.

    Batched, and this is the largest instance of the per-row write that cost
    `save_run` thirteen minutes. A sensor's profile is up to 96 buckets x
    weekday/weekend = 192 rows, so the full fleet is 2,672 x 192 = **513,024
    rows**. One statement each, at the 4.6 rows/sec this database was measured
    at over this link, is **31 hours** -- a daily job that cannot finish in a
    day, and which would have been found only by someone watching it not
    finish.
    """
    deletes = []
    rows: list[dict] = []
    now = datetime.now()
    for baseline in baselines.values():
        batch = baseline.as_rows()
        if not batch:
            continue
        deletes.append({"sensor_key": baseline.sensor_key})
        for row in batch:
            row["updated_at"] = now
            row["days_observed"] = baseline.days_observed
            rows.append(row)

    if not rows:
        return 0

    with engine.begin() as conn:
        _write_many(conn, "DELETE FROM das2_sensor_profile "
                          "WHERE sensor_key = :sensor_key", deletes)
        _write_many(conn, """
            INSERT INTO das2_sensor_profile
              (sensor_key, bucket_of_day, is_weekend, median_value,
               mad_value, n_samples, updated_at, days_observed)
            VALUES
              (:sensor_key, :bucket_of_day, :is_weekend, :median_value,
               :mad_value, :n_samples, :updated_at, :days_observed)
        """, rows)
    log.info("stored %d baseline bucket(s) for %d sensor(s)",
             len(rows), len(baselines))
    return len(rows)


def load_baselines(engine: Engine) -> dict:
    """
    Stored time-of-day baselines, for the hourly run's L2 layer.

    Returns an empty dict when the daily job has not run. The L2 detector
    abstains on that rather than inventing a baseline from the current window,
    which is the behaviour that made the incumbent's rolling median unable to
    see any excursion longer than its own window.
    """
    from das2.profile.build import TimeOfDayBaseline

    out: dict[str, TimeOfDayBaseline] = {}
    with engine.connect() as conn:
        try:
            rows = conn.execute(text("""
                SELECT sensor_key, bucket_of_day, is_weekend, median_value,
                       mad_value, n_samples, days_observed
                  FROM das2_sensor_profile
            """)).mappings().all()
        except Exception as exc:                          # noqa: BLE001
            log.warning("could not read stored baselines: %s", exc)
            return {}

    for row in rows:
        key = str(row["sensor_key"])
        baseline = out.setdefault(key, TimeOfDayBaseline(sensor_key=key))
        baseline.days_observed = max(baseline.days_observed,
                                     int(row["days_observed"] or 0))
        baseline.buckets[(int(row["bucket_of_day"]), int(row["is_weekend"]))] = (
            float(row["median_value"] or 0.0),
            float(row["mad_value"] or 0.0),
            int(row["n_samples"] or 0),
        )
    log.info("loaded time-of-day baselines for %d sensor(s)", len(out))
    return out


def load_history(engine: Engine, days: int = 28, *,
                 fallback_table: str = ""):
    """
    Long history for the daily profile job.

    Reads `das2_reading`, which the hourly run fills. That is the whole source
    unless a fallback is configured, and starting empty is a deliberate choice
    rather than an oversight: DRIFT needs 14 days and the baselines want 28, so
    on a fresh install the long-horizon detectors produce nothing for the first
    few weeks and say so.

    `fallback_table` is OFF by default. An earlier version reached into
    `dbo.data` automatically whenever `das2_reading` was empty, which is a
    surprising thing for a job to do to a database it was not pointed at --
    and on an installation deliberately started from scratch it would quietly
    reintroduce the history that was just cleared. Set
    DAS2_DATABASE_HISTORY_FALLBACK_TABLE to opt in.

    The fallback is expected to expose sensor_key, ts and value, by those names
    or through a view. Anything else is rejected rather than guessed at.
    """
    import pandas as pd

    cutoff = datetime.now() - timedelta(days=days)
    sources = [("SELECT sensor_key, ts, value FROM das2_reading "
                "WHERE ts >= :cutoff", "das2_reading")]
    if fallback_table:
        sources.append((
            f"SELECT sensor_key, ts, value FROM {fallback_table} "
            f"WHERE ts >= :cutoff", fallback_table))

    for sql, label in sources:
        try:
            with engine.connect() as conn:
                frame = pd.read_sql(text(sql), conn, params={"cutoff": cutoff})
            if not frame.empty:
                # format="mixed" is required, not merely tidy. SQLite stores
                # timestamps as text, and rows written at different times carry
                # different precision -- "2026-09-19 02:00:00" alongside
                # "2026-09-19 02:00:00.123456". pandas infers a single format
                # from the first row and then raises on every row that does not
                # match it, which failed the whole daily job with a message
                # about strftime rather than about the data.
                frame["ts"] = pd.to_datetime(frame["ts"], format="mixed",
                                             errors="coerce")
                frame = frame.dropna(subset=["ts"])
                frame["sensor_key"] = frame["sensor_key"].astype(str)
                if frame.empty:
                    continue
                log.info("read %d row(s) of history from %s", len(frame), label)
                return frame
        except Exception as exc:                          # noqa: BLE001
            log.info("history not available from %s (%s)", label, str(exc)[:100])
    return pd.DataFrame(columns=["sensor_key", "ts", "value"])


#: Sensors per history batch for the daily job.
#:
#: 250 x 28 days x 120 s is about 5 million rows in flight, a few hundred MB in
#: pandas. The fleet is 2,672 sensors, so this is roughly eleven batches.
HISTORY_SENSOR_BATCH = 250


def history_span_days(engine: Engine, table: str, days: int) -> float:
    """How many days of history `table` holds inside the window. 0.0 if none."""
    cutoff = datetime.now() - timedelta(days=days)
    try:
        with engine.connect() as conn:
            lo, hi = conn.execute(
                text(f"SELECT MIN(ts), MAX(ts) FROM {table} "
                     f"WHERE ts >= :cutoff"),
                {"cutoff": cutoff}).fetchone()
    except Exception as exc:                                  # noqa: BLE001
        log.info("history not available from %s (%s)", table, str(exc)[:100])
        return 0.0
    lo, hi = _as_datetime(lo), _as_datetime(hi)
    if lo is None or hi is None:
        return 0.0
    return max(0.0, (hi - lo).total_seconds() / 86400.0)


def history_sensors(engine: Engine, days: int = 28, *,
                    fallback_table: str = "") -> tuple[list[str], str]:
    """
    Which sensors have history in the window, and which table it came from.

    Answered with a `SELECT DISTINCT` so the row count never reaches this
    process. Returns `([], "")` when there is no history at all.

    When a fallback table is configured, the table with the LONGER history wins
    rather than `das2_reading` simply winning whenever it is non-empty. That
    ordering mattered: `das2_reading` holds whatever the runs so far have
    written, so one run is enough to make it non-empty -- and then an older
    readings table holding months of history was never consulted again. The
    effect was that DRIFT, which needs 14 days, waited a fortnight on a
    database that already had the data, with nothing to say why.

    A fallback is only ever consulted because the operator pointed
    DAS2_DATABASE_HISTORY_FALLBACK_TABLE at it, so preferring it when it knows
    more is doing what they asked rather than reaching somewhere unexpected.
    """
    candidates = ["das2_reading"] + ([fallback_table] if fallback_table else [])
    if len(candidates) > 1:
        spans = {t: history_span_days(engine, t, days) for t in candidates}
        candidates.sort(key=lambda t: -spans[t])
        if spans[candidates[0]] > 0:
            log.info("history sources: %s — reading from %s",
                     ", ".join(f"{t} {spans[t]:.0f}d" for t in spans),
                     candidates[0])

    for table in candidates:
        try:
            with engine.connect() as conn:
                rows = conn.execute(
                    text(f"SELECT DISTINCT sensor_key FROM {table} "
                         f"WHERE ts >= :cutoff"),
                    {"cutoff": datetime.now() - timedelta(days=days)}).fetchall()
            keys = sorted({str(r[0]) for r in rows if r[0] is not None})
            if keys:
                return keys, table
        except Exception as exc:                              # noqa: BLE001
            log.info("history not available from %s (%s)", table, str(exc)[:100])
    return [], ""


def iter_history(engine: Engine, days: int = 28, *,
                 fallback_table: str = "",
                 batch_sensors: int = HISTORY_SENSOR_BATCH):
    """
    The same history as `load_history`, a batch of sensors at a time.

    This is what makes the daily job survivable at fleet scale, and the module
    it feeds says so in its own docstring: *"Months x 2,672 sensors x 120 s is
    on the order of 10^8 rows... the functions here take a per-sensor frame so
    they can be driven straight from a server-side GROUP BY rather than pulling
    the fleet into pandas."* `load_history` did exactly what that warns
    against -- one `SELECT` of the whole window into one frame -- which is fine
    on the fixture and is tens of gigabytes on the real estate.

    Batching by SENSOR rather than by time, because every statistic the job
    computes is per-sensor and needs that sensor's whole history at once: a
    28-day Theil-Sen slope cannot be assembled from one day at a time, and a
    time-of-day bucket needs every day's visit to it. Splitting by sensor is
    the only axis along which the arithmetic is unchanged.

    Yields `(frame, batch_number, total_batches)`. Yields nothing at all when
    there is no history, which the caller must treat as the fresh-install state
    rather than as a failure.
    """
    import pandas as pd

    keys, table = history_sensors(engine, days, fallback_table=fallback_table)
    if not keys:
        return

    cutoff = datetime.now() - timedelta(days=days)
    total = (len(keys) + batch_sensors - 1) // batch_sensors
    for n, start in enumerate(range(0, len(keys), batch_sensors), start=1):
        batch = keys[start:start + batch_sensors]
        # Named parameters rather than an expanding IN clause: SQL Server caps a
        # statement at 2,100 parameters, and the batch size has to stay clear of
        # it. At 250 sensors plus the cutoff there is room to spare.
        names = [f"k{i}" for i in range(len(batch))]
        sql = (f"SELECT sensor_key, ts, value FROM {table} "
               f"WHERE ts >= :cutoff AND sensor_key IN ("
               + ", ".join(f":{name}" for name in names) + ")")
        params = {"cutoff": cutoff}
        params.update(zip(names, batch))
        with engine.connect() as conn:
            frame = pd.read_sql(text(sql), conn, params=params)
        if frame.empty:
            continue
        # format="mixed" is required, not merely tidy -- see `load_history`.
        frame["ts"] = pd.to_datetime(frame["ts"], format="mixed",
                                     errors="coerce")
        frame = frame.dropna(subset=["ts"])
        frame["sensor_key"] = frame["sensor_key"].astype(str)
        if frame.empty:
            continue
        log.info("history batch %d/%d: %d row(s), %d sensor(s) from %s",
                 n, total, len(frame), frame["sensor_key"].nunique(), table)
        yield frame, n, total


def baseline_age_hours(engine: Engine) -> float | None:
    """
    How long since the daily profile job last wrote anything, in hours.

    `None` means it has never run. This exists so the hourly run can notice
    that for itself: on the client's deployment the profile job was documented,
    containerised and never scheduled, so `baselines: {'sensors': 0}` appeared
    in every run log and DRIFT, NOISE_BURST and RESIDUAL_OUTLIER -- the whole
    "anticipate the sensor going bad" family -- silently produced nothing for
    weeks. A dependency that fails by being quiet needs to be checked by the
    thing that depends on it, not by a line in a compose file.
    """
    try:
        with engine.connect() as conn:
            newest = conn.execute(
                text("SELECT MAX(updated_at) FROM das2_sensor_profile")).scalar()
    except Exception as exc:                                  # noqa: BLE001
        log.debug("could not read the baseline age (%s)", str(exc)[:120])
        return None
    newest = _as_datetime(newest)
    if newest is None:
        return None
    return max(0.0, (datetime.now() - newest).total_seconds() / 3600.0)
