#!/usr/bin/env python3
"""
The migration SQL, on both dialects.

This file exists because of a defect the rest of the suite could not catch.
The migrations are written in the SQLite dialect, every test ran against
SQLite, and 010's own header claimed the result was portable to SQL Server. It
was not. Three constructs in it are wrong there, and none is visible from a
passing SQLite test:

  CREATE TABLE IF NOT EXISTS   no such form in T-SQL, in any version
  TIMESTAMP                    a synonym for ROWVERSION, NOT a datetime
  TEXT                         deprecated, unusable in comparisons

The first was confirmed against the client's SQL Server, which answered
`[42000] Incorrect syntax near the keyword 'IF'. (156)` on the very first
statement. The second is the one to fear: it fails late. The tables would be
created, the migration would report success, and then every INSERT would die
with "Cannot insert an explicit value into a timestamp column" -- an error
pointing nowhere near the file that caused it.

So the assertions below check the *translated* SQL, which is the only thing a
SQL Server deployment ever sees.
"""

import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text  # noqa: E402

from das2.io.store import (  # noqa: E402
    DAS2_MIGRATIONS,
    MIGRATIONS_DIR,
    _split_statements,
    apply_migrations,
    make_engine,
    render_migrations,
    translate_sql,
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


def main() -> int:
    print("\nmigrations: the T-SQL carries nothing SQL Server rejects")
    mssql = render_migrations("mssql")
    check("every migration produced statements", len(mssql) > 20,
          f"{len(mssql)} statement(s)")

    joined = "\n".join(mssql)
    check("no CREATE ... IF NOT EXISTS",
          not re.search(r"(?i)\bIF\s+NOT\s+EXISTS\s+\w+\s*\(", joined),
          "the real server answered: Incorrect syntax near the keyword 'IF'")
    check("no bare TIMESTAMP", not re.search(r"(?i)\bTIMESTAMP\b", joined),
          "there it is ROWVERSION, and rejects every INSERT")
    check("no bare TEXT",
          not re.search(r"(?i)\bTEXT\b", joined.replace("VARCHAR(MAX)", "")),
          "deprecated since 2005")
    check("datetimes became DATETIME2", "DATETIME2" in joined)
    check("text columns became VARCHAR(MAX)", "VARCHAR(MAX)" in joined)

    print("\nmigrations: the T-SQL is idempotent WITHOUT catching errors")
    # Re-running must not depend on ALREADY_APPLIED matching an exception's
    # text, because SQL Server localises its messages: on a non-English server
    # "There is already an object named ..." never appears, the match fails,
    # and the second run raises instead of continuing.
    unguarded = [s for s in mssql if not s.lstrip().upper().startswith("IF ")]
    check("every statement is guarded by an existence test", not unguarded,
          unguarded[0][:60] if unguarded else f"{len(mssql)}/{len(mssql)}")
    check("tables guarded with OBJECT_ID",
          "OBJECT_ID(N'das2_sensor', N'U')" in joined)
    check("indexes guarded against sys.indexes", "FROM sys.indexes" in joined)
    check("added columns guarded with COL_LENGTH", "COL_LENGTH(" in joined)

    print("\nmigrations: the substitution cannot corrupt an identifier")
    # The rewrite is a word-boundary regex, which is only safe while no column
    # is NAMED text or timestamp and no string literal contains either word.
    raw = "\n".join(
        s for name in DAS2_MIGRATIONS
        for s in _split_statements((MIGRATIONS_DIR / name).read_text())
    )
    check("no column is named `text` or `timestamp`",
          not re.search(r"(?im)^\s*(text|timestamp)\s", raw),
          "if one is ever added, the rewrite must become a parser")
    literals = re.findall(r"'([^']*)'", raw)
    check("no string literal contains either word",
          not any(re.search(r"(?i)text|timestamp", lit) for lit in literals),
          f"literals: {literals}")

    print("\nthe login timeout must outlast a slow server")
    # Driver 18 allows 15 seconds for the LOGIN handshake. On the client's
    # historian that was not enough: every connection died at exactly 15 s
    # with HYT00 Login timeout expired, while a plain TCP connect to 1433
    # from the same container returned instantly. The one connection that DID
    # succeed during debugging carried `Login Timeout=30` -- and it also
    # carried Encrypt=no, so two things changed at once and the success was
    # credited to the wrong one. The timeout is the variable that matters.
    import sqlalchemy as _sa

    from das2.io import store as _store

    seen = {}
    real_create = _sa.create_engine

    def _spy(url, **kw):
        seen["connect_args"] = kw.get("connect_args")
        if str(url).startswith("mssql"):
            # A real engine, not a sentinel: make_engine attaches the bulk-insert
            # listener (see test_reading_persistence.py) and an `object()` has no
            # events to attach it to.
            return real_create("sqlite:///:memory:")
        return real_create(url, **kw)

    _store.create_engine = _spy
    try:
        _store.make_engine("mssql+pyodbc://u:p@h:1433/d?driver=x")
        check("SQL Server gets a login timeout",
              (seen["connect_args"] or {}).get("timeout") == _store.DEFAULT_LOGIN_TIMEOUT_S,
              f"({_store.DEFAULT_LOGIN_TIMEOUT_S}s, against the driver default of 15)")
        check("and it is well clear of the 15 s that failed",
              _store.DEFAULT_LOGIN_TIMEOUT_S >= 30)
        _store.make_engine("sqlite:///x.db")
        check("SQLite is not given one", seen["connect_args"] is None,
              "(its driver has no such parameter and rejects it)")
    finally:
        _store.create_engine = real_create

    print("\nmigrations: SQLite is untouched, and still applies")
    sqlite_sql = render_migrations("sqlite")
    check("sqlite statements pass through verbatim",
          all(translate_sql(s, "sqlite") == s for s in sqlite_sql))
    check("sqlite keeps IF NOT EXISTS",
          "IF NOT EXISTS" in "\n".join(sqlite_sql),
          "valid there, and the tests rely on it")

    with tempfile.TemporaryDirectory() as tmp:
        engine = make_engine(f"sqlite:///{Path(tmp) / 'das2.db'}")
        applied = apply_migrations(engine)
        check("migrations apply to a fresh database", len(applied) > 0,
              f"{len(applied)} statement(s)")
        apply_migrations(engine)
        check("and are safe to re-run", True, "second pass raised nothing")
        with engine.connect() as conn:
            for table in ("das2_sensor", "das2_incident", "das2_detection_run"):
                got = conn.execute(
                    text("SELECT name FROM sqlite_master "
                         "WHERE type='table' AND name=:t"), {"t": table}
                ).scalar()
                check(f"{table} exists", got == table)

    print(f"\n{passed} passed, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
