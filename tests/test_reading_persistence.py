#!/usr/bin/env python3
"""
Persisting readings must send the TAIL, not the window.

This file exists because of the defect that stopped the client's deployment
dead. `das2_reading` is written every run, and every run re-reads the whole
72-hour window, so consecutive runs overlap by 71 of 72 hours. The first
version sent all of it and let the database discard the duplicates:

    18:44:44  run persisted: 3268 anomalies, 280 incidents
    04:13:48  persisted 5627250 reading(s) to das2_reading

Nine and a half hours, at 165 rows a second, of which 98.6% were rows the
table already held. The 19:05 through 04:05 runs never fired -- the system
managed two runs in a day instead of twenty-four.

The fix has two halves and this file pins both:

  * `fast_executemany`, because without it pyodbc makes one round trip PER ROW;
  * an incremental filter, so only readings newer than what is stored are sent.

The assertions below count rows ACTUALLY SENT to the driver, not rows stored.
That distinction is the whole point. The first attempt at the filter looked
correct, logged nothing alarming, and stored exactly the right number of
rows -- while still sending the entire window every time, because SQLite
returns MAX(ts) as TEXT, subtracting a timedelta from a string raised, and the
except that exists for a missing table swallowed it. A test asserting "the
stored count is right" passes on the broken version. Only a test asserting the
REDUCTION catches it.

Run:  python3 tests/test_reading_persistence.py
"""

import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import event, text  # noqa: E402

from das2.io.store import (  # noqa: E402
    LATE_ARRIVAL_GRACE_HOURS,
    _as_datetime,
    apply_migrations,
    make_engine,
    save_readings,
)

passed = failed = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {label}" + (f"  ({detail})" if detail else ""))
    else:
        failed += 1
        print(f"  FAIL  {label}" + (f"  ({detail})" if detail else ""))


class SentCounter:
    """Counts the parameter rows handed to the driver, per save_readings call."""

    def __init__(self, engine):
        self.rows = 0
        event.listen(engine, "before_cursor_execute", self._seen)

    def _seen(self, conn, cursor, statement, parameters, context, executemany):
        if executemany:
            self.rows += len(parameters)
        elif "INSERT" in statement.upper():
            self.rows += 1

    def take(self) -> int:
        n, self.rows = self.rows, 0
        return n


def window(start: datetime, hours: int, sensors: int) -> pd.DataFrame:
    """A run's worth of readings: `sensors` sensors reporting hourly."""
    return pd.DataFrame([
        {"sensor_key": f"S{s:03d}",
         "ts": start + timedelta(hours=h),
         "value": float(h * 10 + s)}
        for h in range(hours) for s in range(sensors)
    ])


def main() -> int:
    print("\nthe MAX(ts) probe survives what the drivers actually return")
    # SQL Server hands back a datetime; SQLite hands back a string. The string
    # is what broke it, and it broke SILENTLY.
    when = datetime(2026, 9, 22, 14, 0)
    check("a datetime passes through", _as_datetime(when) == when)
    check("SQLite's TEXT is parsed, not dropped",
          _as_datetime("2026-09-22 14:00:00") == when,
          "this is the one that no-opped the whole filter")
    check("an ISO string with a T is parsed",
          _as_datetime("2026-09-22T14:00:00") == when)
    check("None stays None", _as_datetime(None) is None)
    check("unparseable input degrades to None, not an exception",
          _as_datetime("not a timestamp") is None,
          "a bad probe must cost performance, never correctness")

    with tempfile.TemporaryDirectory() as tmp:
        engine = make_engine(f"sqlite:///{Path(tmp) / 'das2.db'}")
        apply_migrations(engine)
        counter = SentCounter(engine)

        t0 = datetime(2026, 9, 20, 0, 0)
        HOURS, SENSORS = 72, 6
        first = window(t0, HOURS, SENSORS)

        print("\nrun 1: nothing is stored, so the whole window goes")
        stored = save_readings(engine, first)
        sent_1 = counter.take()
        check("the first run sends everything", sent_1 == len(first),
              f"{sent_1} of {len(first)}")
        check("and stores it", stored == len(first), f"{stored} row(s)")

        print("\nrun 2: the window has moved one hour; only the hour is new")
        # Exactly what the scheduler does an hour later: the same 72-hour
        # window shifted by one, so 71 hours of it is already in the table.
        second = window(t0 + timedelta(hours=1), HOURS, SENSORS)
        save_readings(engine, second)
        sent_2 = counter.take()

        # The grace hour means one hour of overlap is deliberately re-sent.
        expected = SENSORS * (1 + LATE_ARRIVAL_GRACE_HOURS)
        check("the second run sends the tail, not the window",
              sent_2 <= expected, f"{sent_2} row(s), at most {expected}")
        check("which is a reduction of more than an order of magnitude",
              sent_2 * 10 < sent_1,
              f"{sent_1} -> {sent_2} ({100 * (1 - sent_2 / sent_1):.1f}% less)")
        check("the new hour is not among what was skipped",
              sent_2 >= SENSORS, f"{sent_2} >= {SENSORS}")

        print("\nrun 3: nothing new at all")
        save_readings(engine, second)
        sent_3 = counter.take()
        check("a repeated window re-sends only the grace hour",
              sent_3 <= expected, f"{sent_3} row(s)")
        check("and never the whole window again", sent_3 * 10 < sent_1)

        print("\nno reading is lost by the filter")
        with engine.connect() as conn:
            total = conn.execute(
                text("SELECT COUNT(*) FROM das2_reading")).scalar()
            distinct_hours = conn.execute(
                text("SELECT COUNT(DISTINCT ts) FROM das2_reading")).scalar()
        check("every hour of both windows is present",
              distinct_hours == HOURS + 1, f"{distinct_hours} hour(s)")
        check("every sensor-hour is present exactly once",
              total == (HOURS + 1) * SENSORS, f"{total} row(s)")

        print("\na late-arriving reading still gets in")
        # 12% of sensors in this feed carry out-of-order timestamps, which is
        # why the cutoff is MAX(ts) minus an hour rather than MAX(ts). A
        # reading for an hour already written must not be lost forever.
        newest = t0 + timedelta(hours=HOURS)          # last ts of window 2
        late = pd.DataFrame([
            {"sensor_key": "S999",
             "ts": newest - timedelta(minutes=30),
             "value": 1.0},
        ])
        save_readings(engine, late)
        with engine.connect() as conn:
            got = conn.execute(
                text("SELECT COUNT(*) FROM das2_reading "
                     "WHERE sensor_key = 'S999'")).scalar()
        check("a reading 30 min behind the newest is stored", got == 1,
              "strict MAX(ts) filtering would have dropped it")

        print("\nan empty run costs nothing")
        counter.take()
        check("no readings means no statements",
              save_readings(engine, pd.DataFrame(
                  columns=["sensor_key", "ts", "value"])) == 0
              and counter.take() == 0)

    print("\nthe alert never queues behind the reading history")
    # On 22 September the run found a P1 and said nothing for nine and a half
    # hours, because save_readings ran BEFORE the Telegram send and the whole
    # window was going into the table one row at a time. Both halves of that
    # are fixed, but the ordering is the one that matters on a bad day: a slow
    # or failing history write must never be able to hold up an alert.
    #
    # Checked structurally, because cmd_run needs a share, a database and a
    # bot to run, and none of those exist here.
    import ast

    tree = ast.parse((Path(__file__).resolve().parent.parent
                      / "das2" / "cli.py").read_text())
    cmd_run = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == "cmd_run")

    def calls(node) -> set[str]:
        return {c.func.id for c in ast.walk(node)
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}

    check("cmd_run does not write readings itself",
          "save_readings" not in calls(cmd_run),
          "it delegates, so there is one place the ordering lives")

    top = list(enumerate(cmd_run.body))
    send_at = next(i for i, s in top if "send_run" in calls(s))
    # The dry-run branch returns before any alerting, so its call is not late.
    late = [i for i, s in top
            if "_persist_readings" in calls(s)
            and not any(isinstance(n, ast.Return) for n in ast.walk(s))]
    check("the history write is reached only after the send",
          bool(late) and min(late) > send_at,
          f"send at statement {send_at}, history at {late}")

    print("\npyodbc is told to batch, and only pyodbc")
    # Without fast_executemany the driver sends one round trip per row. That
    # is the 165 rows/second, and it is the other half of the nine hours.
    import sqlalchemy as _sa

    from das2.io import store as _store

    # pyodbc is not installed here, so the engine is stood in for -- the same
    # trick test_migrations.py uses for the login timeout. What is asserted is
    # the listener and what it does, which is all that differs.
    real_create = _sa.create_engine
    _store.create_engine = lambda url, **kw: real_create("sqlite:///:memory:")
    try:
        eng = _store.make_engine("mssql+pyodbc://u:p@h:1433/d?driver=x")
        hooks = list(eng.dispatch.before_cursor_execute)
        check("a pyodbc engine gets a bulk-insert hook", len(hooks) == 1)

        class _Cursor:                       # what pyodbc hands the hook
            pass

        bulk = _Cursor()
        hooks[0](None, bulk, "INSERT INTO das2_reading ...", [{}, {}], None, True)
        check("executemany turns fast_executemany on",
              getattr(bulk, "fast_executemany", False) is True,
              "without it: one network round trip per row, 165 rows/second")

        single = _Cursor()
        hooks[0](None, single, "SELECT MAX(ts) FROM das2_reading", {}, None, False)
        check("a single statement is left alone",
              not hasattr(single, "fast_executemany"),
              "it only applies to parameter arrays")
    finally:
        _store.create_engine = real_create

    check("SQLite is given no such hook",
          not list(_store.make_engine("sqlite:///:memory:")
                   .dispatch.before_cursor_execute),
          "its driver has no fast_executemany and would raise")

    print(f"\n{passed} passed, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
