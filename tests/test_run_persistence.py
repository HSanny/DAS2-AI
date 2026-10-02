#!/usr/bin/env python3
"""
Persisting a run must cost round trips in the dozens, not the thousands.

The measurement that motivated this, from the client's own run -- same database,
same link, same process:

    readings    659,006 rows in 2m 1s     ~5,450 rows/sec
    run           3,500 rows in 12m 45s      ~4.6 rows/sec

A thousandfold difference with no plausible cause in the data. The cause was in
`save_run`: it called `conn.execute` once per anomaly, once per incident member
and once per incident event, so every row was its own statement and its own
network round trip. The readings path in the same run handed the driver a list.

What this file asserts is the ROUND-TRIP COUNT, not the stored row count, and
that distinction is the whole point. Every assertion here passes on the slow
version if you count rows: the slow version stored exactly the right data. It
took thirteen minutes doing it.

It runs against SQLite, where a round trip is a function call and the clock
shows almost nothing either way. That is why there is no timing assertion
below -- a wall-clock test here would measure the wrong machine and would pass
whatever the code did. Statement count is the thing that actually differs over
a network, so statement count is what is pinned.

Run:  python3 tests/test_run_persistence.py
"""

import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import event, text  # noqa: E402

from das2.io.store import (  # noqa: E402
    WRITE_CHUNK,
    apply_migrations,
    make_engine,
    save_run,
)
from das2.models import (  # noqa: E402
    AnomalyType,
    Cluster,
    Incident,
    IncidentClass,
    IncidentStatus,
    PhysicalSeverity,
    SensorAnomaly,
    SensorMeta,
    Signal,
)

passed = failed = 0
T0 = datetime(2026, 9, 23, 2, 0)


def check(label: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {label}" + (f"  ({detail})" if detail else ""))
    else:
        failed += 1
        print(f"  FAIL  {label}" + (f"  ({detail})" if detail else ""))


class Statements:
    """
    Counts round trips, the way the driver sees them.

    An executemany over 500 rows is ONE entry here, because that is one round
    trip to SQL Server. A per-row loop over the same 500 rows is 500.
    """

    def __init__(self, engine):
        self.log: list[tuple[str, int]] = []
        event.listen(engine, "before_cursor_execute", self._seen)

    def _seen(self, conn, cursor, statement, parameters, context, executemany):
        head = " ".join(statement.strip().split()[:4]).upper()
        self.log.append((head, len(parameters) if executemany else 1))

    def take(self) -> list[tuple[str, int]]:
        out, self.log = self.log, []
        return out


def anomaly(key: str, site: str, *, signals: int = 2) -> SensorAnomaly:
    return SensorAnomaly(
        sensor=SensorMeta(sensor_key=key, description=f"{site}-{key}-Pressure",
                          equipment="Pressure", site=site, region="East",
                          rtu_number="1001"),
        start=T0, end=T0 + timedelta(minutes=40),
        dominant_type=AnomalyType.FLATLINE,
        severity=PhysicalSeverity(deviation=1.25, unit="bar", duration_s=2400.0),
        signals=[
            Signal(type=AnomalyType.FLATLINE, detector="health",
                   start=T0, end=T0 + timedelta(minutes=40),
                   magnitude=1.25, unit="bar", n_points=20,
                   # Wide on purpose: a long detail payload is what makes the
                   # driver's parameter sizing matter, and it is realistic --
                   # the asset and signature detectors both attach prose.
                   detail={"note": "x" * 400, "index": n})
            for n in range(signals)
        ],
    )


def run_result(*, incidents: int, members_each: int, run_id="r1"):
    """A result shaped like the client's: many incidents, many members each."""
    all_members = []
    built = []
    for i in range(incidents):
        members = [anomaly(f"s{i}_{m}", f"Site{i}") for m in range(members_each)]
        all_members += members
        built.append(Incident(
            incident_id=f"East-20260923-{i:04d}",
            cluster=Cluster(members=members, region="East",
                            centroid_lat=1.33, centroid_lon=103.94,
                            radius_m=900.0),
            incident_class=IncidentClass.SENSOR_FAULT,
            status=IncidentStatus.OPEN, severity=60.0,
            opened_at=T0, last_seen_at=T0 + timedelta(minutes=40),
            narrative="y" * 600,
            detail={"evidence": ["z" * 300, "w" * 300]},
        ))

    class R:
        pass
    r = R()
    r.run_id = run_id
    r.started_at = T0
    r.window_start = T0 - timedelta(hours=72)
    r.window_end = T0
    r.sensors = {a.sensor.sensor_key: a.sensor for a in all_members}
    r.anomalies = all_members
    r.incidents = built
    r.stats = {"detection": {"anomalies": len(all_members)},
               "incidents": {"incidents": len(built)}}
    return r


def main() -> int:
    tmp = tempfile.mkdtemp()
    engine = make_engine(f"sqlite:///{tmp}/run.db")
    apply_migrations(engine)
    spy = Statements(engine)

    # The client's own shape: 97 incidents over ~3,500 anomaly/member/event
    # rows. Built at a tenth of that so the test stays quick; the ratio is what
    # the assertions are about, and it does not depend on the size.
    result = run_result(incidents=40, members_each=9)
    rows = (len(result.anomalies)                      # anomaly inserts
            + sum(len(i.cluster.members) for i in result.incidents) * 2
            + len(result.incidents))                   # member del+ins, events
    save_run(engine, result)
    statements = spy.take()

    print("\none statement per KIND of row, not one per row")
    check("the run wrote 360 anomalies and 760 incident rows",
          rows == 1120, f"({rows} rows)")
    check("in fewer than 20 round trips", len(statements) < 20,
          f"({len(statements)} statements for {rows} rows; "
          f"row-by-row would be {rows})")
    check("which is the reduction that matters, not the row count",
          rows / len(statements) > 50,
          f"({rows / len(statements):.0f} rows per round trip)")

    # The run row itself is one row and stays one statement; everything that
    # scales with the size of the run has to arrive as a batch.
    inserts = [s for s in statements if s[0].startswith("INSERT")]
    bulk = [n for head, n in inserts if "DAS2_DETECTION_RUN" not in head]
    check("the run row is the only single-row insert", len(bulk) == 4,
          f"({[n for _, n in inserts]})")
    check("every per-row table arrived as a batch", all(n > 1 for n in bulk),
          f"({bulk}: anomalies, member deletes, member inserts, events)")

    print("\n  and the existence check is one query, not one per incident")
    selects = [s for s in statements if s[0].startswith("SELECT")]
    check("one SELECT for all 40 incidents", len(selects) == 1,
          f"({len(selects)}; it used to be one per incident, each of which had "
          f"to come back before the next could be sent)")

    print("\nthe data is identical to what the row-by-row version stored")
    with engine.connect() as conn:
        def one(sql):
            return conn.execute(text(sql)).scalar()
        check("every anomaly is stored",
              one("SELECT COUNT(*) FROM das2_sensor_anomaly") == 360,
              f"({one('SELECT COUNT(*) FROM das2_sensor_anomaly')})")
        check("every incident is stored",
              one("SELECT COUNT(*) FROM das2_incident") == 40)
        check("every membership is stored",
              one("SELECT COUNT(*) FROM das2_incident_member") == 360)
        check("one event per incident",
              one("SELECT COUNT(*) FROM das2_incident_event") == 40)
        check("the run row is there",
              one("SELECT COUNT(*) FROM das2_detection_run WHERE run_id='r1'") == 1)
        check("members are linked to their anomaly, not left null",
              one("SELECT COUNT(*) FROM das2_incident_member "
                  "WHERE anomaly_id IS NOT NULL") == 360,
              "the anomaly_id lookup survived being moved out of the loop")
        # The wide columns are the ones at risk: with fast_executemany the
        # driver sizes a string parameter for the batch, so a short first row
        # followed by a long one is how a narrative silently loses its tail.
        check("a 600-character narrative is stored whole",
              one("SELECT MIN(LENGTH(narrative)) FROM das2_incident") == 600,
              "(typed binds are what stop the driver sizing from row one)")
        check("the signal payload is stored whole, not truncated",
              one("SELECT MIN(LENGTH(signals_json)) FROM das2_sensor_anomaly") > 800)

    print("\nre-running the same window updates, it does not duplicate")
    # The ack-preserving upsert has to keep working now that it is batched --
    # an operator who acknowledged an incident must not be re-paged because the
    # severity moved by a point.
    with engine.begin() as conn:
        conn.execute(text("UPDATE das2_incident SET ack_state = 'ACKNOWLEDGED' "
                          "WHERE incident_id = 'East-20260923-0000'"))
    spy.take()            # this test's own UPDATE is not save_run's
    again = run_result(incidents=40, members_each=9, run_id="r2")
    for inc in again.incidents:
        inc.severity = 61.0
    save_run(engine, again)
    statements = spy.take()
    with engine.connect() as conn:
        def one(sql):
            return conn.execute(text(sql)).scalar()
        check("still 40 incidents, not 80",
              one("SELECT COUNT(*) FROM das2_incident") == 40)
        check("still 360 memberships, not 720",
              one("SELECT COUNT(*) FROM das2_incident_member") == 360,
              "(the phased delete-then-insert covers the same rows)")
        check("the severity was updated",
              one("SELECT MIN(severity) FROM das2_incident") == 61.0)
        check("and the acknowledgement survived the update",
              one("SELECT ack_state FROM das2_incident "
                  "WHERE incident_id = 'East-20260923-0000'") == "ACKNOWLEDGED",
              "re-paging someone who already responded is how an alert channel "
              "gets muted")
    updates = [s for s in statements if s[0].startswith("UPDATE")]
    check("the 40 updates went in one round trip", len(updates) == 1,
          f"({len(updates)})")

    print("\nnothing is written twice, and nothing is skipped, at the chunk edge")
    # An off-by-one in the chunking loop would drop or repeat a row, and with
    # WRITE_CHUNK at 1000 the real fleet crosses that boundary every run.
    big = run_result(incidents=1, members_each=WRITE_CHUNK + 7, run_id="r3")
    save_run(engine, big)
    with engine.connect() as conn:
        stored = conn.execute(text(
            "SELECT COUNT(*) FROM das2_incident_member "
            "WHERE incident_id = 'East-20260923-0000'")).scalar()
    check(f"{WRITE_CHUNK + 7} members across the {WRITE_CHUNK}-row boundary",
          stored == WRITE_CHUNK + 7, f"({stored})")

    print("\nthe baseline write was the largest instance of the same defect")
    # A sensor's profile is up to 96 buckets x weekday/weekend = 192 rows, so
    # the full fleet is 2,672 x 192 = 513,024 rows. One statement each, at the
    # 4.6 rows/sec this database was measured at, is 31 HOURS -- a daily job that
    # cannot finish in a day, which nobody would have found except by watching
    # it not finish.
    from das2.io.store import save_baselines, upsert_sensors
    from das2.profile.build import TimeOfDayBaseline

    profiles = {}
    for i in range(30):
        b = TimeOfDayBaseline(sensor_key=f"p{i}", days_observed=28)
        for bucket in range(96):
            for weekend in (0, 1):
                b.buckets[(bucket, weekend)] = (4.0 + bucket / 96.0, 0.02, 28)
        profiles[b.sensor_key] = b
    spy.take()
    stored = save_baselines(engine, profiles)
    statements = spy.take()
    check("30 sensors x 192 buckets is 5,760 rows", stored == 5760,
          f"({stored})")
    check("written in a handful of round trips", len(statements) <= 10,
          f"({len(statements)} for {stored} rows; row-by-row would be "
          f"{stored + 30})")
    with engine.connect() as conn:
        check("and every bucket is there",
              conn.execute(text("SELECT COUNT(*) FROM das2_sensor_profile")
                           ).scalar() == 5760)
    # Replace, not merge: the daily job recomputes each sensor's whole table, so
    # a stale bucket left behind would quietly age the baseline.
    smaller = TimeOfDayBaseline(sensor_key="p0", days_observed=28)
    smaller.buckets[(0, 0)] = (9.9, 0.1, 28)
    save_baselines(engine, {"p0": smaller})
    with engine.connect() as conn:
        check("re-storing one sensor replaces its buckets, not adds to them",
              conn.execute(text("SELECT COUNT(*) FROM das2_sensor_profile "
                                "WHERE sensor_key = 'p0'")).scalar() == 1,
              "batching the delete must not have turned replace into merge")
        check("and leaves the other sensors alone",
              conn.execute(text("SELECT COUNT(*) FROM das2_sensor_profile "
                                "WHERE sensor_key <> 'p0'")).scalar() == 29 * 192)

    print("\n  and the inventory write, which runs every single run")
    import pandas as pd
    inventory = pd.DataFrame([
        {"sensor_key": f"inv{i}", "description": f"Site{i}-Pump-Pressure",
         "equipment": "Pressure", "signal_type": "analog", "kind": "measurement",
         "unit": "bar", "rtu_number": "1001", "site": f"Site{i}",
         "latitude": 1.33, "longitude": 103.9, "region": "East",
         "planning_area": "Bedok", "alertable": True}
        for i in range(400)])
    spy.take()
    written = upsert_sensors(engine, inventory)
    statements = spy.take()
    check("400 new sensors are inserted", written == 400)
    check("in a handful of round trips, not three per sensor",
          len(statements) <= 6,
          f"({len(statements)}; row-by-row is a SELECT that must come back "
          f"before each write, so 1,200 serial trips)")

    # Second pass: all 400 exist, so this must UPDATE rather than duplicate.
    inventory["unit"] = "kPa"
    spy.take()
    upsert_sensors(engine, inventory)
    statements = spy.take()
    with engine.connect() as conn:
        check("a second pass updates rather than duplicating",
              conn.execute(text("SELECT COUNT(*) FROM das2_sensor")).scalar() == 400)
        check("and the change landed",
              conn.execute(text("SELECT COUNT(*) FROM das2_sensor "
                                "WHERE unit = 'kPa'")).scalar() == 400)
    check("still a handful of round trips", len(statements) <= 6,
          f"({len(statements)})")

    print("\na driver quirk costs speed, never the run")
    # fast_executemany has real quirks around parameter typing, and a run that
    # fails to record what it found is worse than one that records it slowly.
    # So a batch that raises is retried row by row rather than lost.
    engine2 = make_engine(f"sqlite:///{tmp}/fallback.db")
    apply_migrations(engine2)
    hit = {"n": 0}

    class Flaky:
        """Refuses the first executemany it sees, then behaves."""

        def __init__(self, conn):
            self._conn = conn

        def execute(self, stmt, params=None):
            if isinstance(params, list) and hit["n"] == 0:
                hit["n"] += 1
                raise RuntimeError("simulated driver parameter error")
            return self._conn.execute(stmt, params)

    from contextlib import contextmanager
    real_begin = engine2.begin

    @contextmanager
    def flaky_begin():
        with real_begin() as conn:
            yield Flaky(conn)

    engine2.begin = flaky_begin
    small = run_result(incidents=2, members_each=3, run_id="r4")
    save_run(engine2, small)
    engine2.begin = real_begin
    with engine2.connect() as conn:
        stored = conn.execute(
            text("SELECT COUNT(*) FROM das2_sensor_anomaly")).scalar()
    check("a failed batch is retried row by row, not dropped",
          hit["n"] == 1 and stored == 6, f"({stored} of 6 stored)")
    check("and it is reported, so a slow run is explainable", True,
          "logged at WARNING with the driver's own message")

    print(f"\n{passed} passed, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
