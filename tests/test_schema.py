"""
Tests for migrations/010_das2_schema.sql (Phase 1).

Runs the real migration file against SQLite. That is not SQL Server, so it
proves syntax and shape rather than production behaviour -- the SQL Server path
still needs a smoke test on the client's box. What it does prove, cheaply and
every run, is that the file parses, is genuinely idempotent, and that a full
incident round-trip fits the columns the code actually writes.

Run:  python3 tests/test_schema.py
"""

import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MIGRATION = REPO / "migrations" / "010_das2_schema.sql"

EXPECTED_TABLES = {
    "das2_sensor", "das2_reading", "das2_sensor_profile", "das2_detection_run",
    "das2_sensor_anomaly", "das2_incident", "das2_incident_member",
    "das2_incident_event", "das2_neighbour_correlation", "das2_rain_observation",
    "das2_alert_delivery", "das2_feedback",
}


def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
    if not cond:
        raise SystemExit(1)


def tables(conn):
    return {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}


def columns(conn, table):
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def main():
    sql = MIGRATION.read_text(encoding="utf-8")

    print("migration applies")
    conn = sqlite3.connect(":memory:")
    conn.executescript(sql)
    got = tables(conn)
    check("every expected table created", EXPECTED_TABLES <= got,
          f"(missing {sorted(EXPECTED_TABLES - got)})" if EXPECTED_TABLES - got else "")

    print("\nidempotent")
    conn.executescript(sql)
    conn.executescript(sql)
    check("re-running twice more does not raise", EXPECTED_TABLES <= tables(conn))

    print("\nthe v1 tables are untouched")
    # Compare executable SQL only: the file discusses the v1 tables at length in
    # comments, which is documentation rather than a reference to them.
    statements = "\n".join(
        line.split("--", 1)[0] for line in sql.splitlines()
    )
    check("no v1 table is referenced by any statement",
          not any(t in statements for t in
                  ("dim", "inst", "linktable", "dateDim", "alarmevent",
                   "abnormal_sensor_history")),
          "(both systems must run side by side during the shadow comparison)")
    check("no DROP statements at all", "DROP " not in statements.upper())
    check("no ALTER of existing tables", "ALTER " not in statements.upper())
    check("SCADA alarm events stay out of scope", "alarm" not in statements.lower())
    check("every created table is namespaced das2_",
          all(t.startswith("das2_") for t in (EXPECTED_TABLES & tables(conn))))

    print("\nthe decision-relevant columns exist")
    check("anomalies carry a type, not only a score",
          "dominant_type" in columns(conn, "das2_sensor_anomaly"),
          "(Peak_RZ was not comparable between sensors)")
    check("severity is stored in engineering units",
          {"deviation", "deviation_unit", "span_fraction"} <= columns(conn, "das2_sensor_anomaly"))
    check("incident records neighbour correlation",
          "neighbour_correlation" in columns(conn, "das2_incident"),
          "(the water moved, or the instrument is lying)")
    check("incident records rainfall from internal gauges",
          "rainfall_mm" in columns(conn, "das2_incident"))
    check("sensor records how it was placed",
          "placement_source" in columns(conn, "das2_sensor"),
          "(coordinate coverage must be measured, not assumed)")
    check("new equipment classes can be analysed without paging anyone",
          "alertable" in columns(conn, "das2_sensor"))
    check("shadow mode can share the run table",
          "detector_version" in columns(conn, "das2_detection_run"))
    check("delivery dedup is possible",
          {"payload_hash", "suppressed"} <= columns(conn, "das2_alert_delivery"))

    print("\nan incident round-trip fits the schema")
    conn.execute(
        "INSERT INTO das2_sensor (sensor_key, description, equipment, site, "
        "latitude, longitude, region, placement_source, alertable) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        ("k1", "BedokPS-Pump4-Deliver-Pressure", "Pressure", "BedokPS",
         1.34286, 103.91977, "East", "coordinates", 1))
    conn.execute(
        "INSERT INTO das2_detection_run (run_id, window_start, window_end, "
        "sensors_analysed, detector_version, status) VALUES (?,?,?,?,?,?)",
        ("run-1", "2026-03-01T00:00:00", "2026-03-04T00:00:00", 1, "v2", "ok"))
    conn.execute(
        "INSERT INTO das2_sensor_anomaly (anomaly_id, run_id, sensor_key, start_ts, "
        "end_ts, dominant_type, deviation, deviation_unit, span_fraction, duration_s, "
        "severity_score, signals_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        ("a1", "run-1", "k1", "2026-03-01T06:00:00", "2026-03-01T07:00:00",
         "FLATLINE", 0.0, "bar", 0.0, 3600.0, 62.0, '[{"type":"FLATLINE"}]'))
    conn.execute(
        "INSERT INTO das2_incident (incident_id, incident_class, status, severity, "
        "priority, region, sensor_count, neighbour_correlation, ack_state) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        ("i1", "SENSOR_FAULT", "OPEN", 62.0, "P2", "East", 1, 0.05, "NONE"))
    conn.execute(
        "INSERT INTO das2_incident_member (incident_id, sensor_key, anomaly_id) "
        "VALUES (?,?,?)", ("i1", "k1", "a1"))
    conn.execute(
        "INSERT INTO das2_incident_event (event_id, incident_id, ts, event_type) "
        "VALUES (?,?,?,?)", ("e1", "i1", "2026-03-01T07:05:00", "opened"))
    conn.commit()

    row = conn.execute("""
        SELECT i.incident_id, i.incident_class, s.description, a.dominant_type,
               a.deviation_unit, i.neighbour_correlation
          FROM das2_incident i
          JOIN das2_incident_member m ON m.incident_id = i.incident_id
          JOIN das2_sensor s          ON s.sensor_key  = m.sensor_key
          JOIN das2_sensor_anomaly a  ON a.anomaly_id  = m.anomaly_id
    """).fetchone()
    check("incident joins to its member sensor and anomaly", row is not None)
    check("class and type survive the round-trip",
          row[1] == "SENSOR_FAULT" and row[3] == "FLATLINE", f"({row})")
    check("engineering unit preserved", row[4] == "bar")
    check("uncorrelated neighbours recorded", abs(row[5] - 0.05) < 1e-9,
          "(nothing else moved -> the instrument, not the water)")

    print("\nkeys behave")
    try:
        conn.execute("INSERT INTO das2_incident (incident_id) VALUES ('i1')")
        duplicated = True
    except sqlite3.IntegrityError:
        duplicated = False
    check("incident_id is unique", not duplicated)

    conn.execute("INSERT INTO das2_reading VALUES ('k1','2026-03-01T06:00:00',3.9)")
    try:
        conn.execute("INSERT INTO das2_reading VALUES ('k1','2026-03-01T06:00:00',4.0)")
        dup_reading = True
    except sqlite3.IntegrityError:
        dup_reading = False
    check("one reading per sensor per timestamp", not dup_reading,
          "(duplicate timestamps in this feed must not fan out at join time)")

    conn.close()
    print("\nAll schema tests passed.")


if __name__ == "__main__":
    main()
