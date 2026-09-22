"""
das2.cli
========

Command line for the whole system.

    python -m das2.cli migrate      # create the tables (idempotent)
    python -m das2.cli check        # verify config, data and DB before anything else
    python -m das2.cli run          # one analysis run: detect, report, alert
    python -m das2.cli run --dry-run    # everything except Telegram and the DB
    python -m das2.cli ack-worker   # long-poll for acknowledge button presses
    python -m das2.cli schedule     # run on an interval, forever

`check` exists because the three things most likely to be wrong on a new
deployment -- the CSV path, the database credentials and the bot token -- all
fail in ways that look like "the system found nothing". Being able to
distinguish "healthy and quiet" from "broken and silent" is worth a command of
its own, and it is the first thing to run after `docker compose up`.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from das2.config import Config, load_config


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)-22s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


log = logging.getLogger("das2.cli")


# --------------------------------------------------------------------------- #
def cmd_migrate(config: Config, args) -> int:
    from das2.io.store import apply_migrations, make_engine, render_migrations

    # --print-sql emits exactly what would be executed and connects to nothing,
    # so a DBA can review it, or run it in SSMS themselves, without this system
    # holding credentials at all. It renders through the same translation the
    # migration path uses, so the printed script cannot drift from the applied
    # one -- which a hand-maintained copy of the schema inevitably would.
    if getattr(args, "print_sql", False):
        for statement in render_migrations(args.dialect or "mssql"):
            print(statement.rstrip().rstrip(";") + ";\n")
        return 0

    engine = make_engine(config.database.sqlalchemy_url())
    executed = apply_migrations(engine)
    print(f"Applied {len(executed)} statement(s) against {config.database.safe_url}")
    for statement in executed:
        print(f"  {statement}")
    return 0


def cmd_check(config: Config, args) -> int:
    """
    Pre-flight. Every check prints PASS/FAIL and why, and nothing is fatal --
    the point is to show the whole picture, not to stop at the first problem.
    """
    ok = True

    def report(label: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}"
              f"{('  — ' + detail) if detail else ''}")

    print("\nInput data")
    history = Path(config.ingest.history_dir)
    report("history directory exists", history.is_dir(), str(history))

    # A CIFS mount that failed authentication or pointed at the wrong path
    # still APPEARS as a directory -- it is simply empty. That is the single
    # most common way this goes wrong, and "directory exists" alone would pass
    # it straight through to a run that finds no readings and looks like a
    # quiet network. So an existing-but-empty directory is called out as its
    # own failure, with the likely cause.
    if history.is_dir():
        entries = list(history.iterdir())
        files = sorted(history.glob("*HISTORY*.csv"))
        if not entries:
            report("history directory is not empty", False,
                   "the directory exists but contains NOTHING. On a CIFS mount "
                   "that usually means wrong credentials, a wrong share path, "
                   "or cifs-utils missing on the host -- not a missing folder. "
                   "Check: docker compose run --rm das2 ls -la "
                   f"{history}")
        else:
            report("history directory is not empty", True,
                   f"{len(entries)} entr(ies)")
            report("history CSVs present", bool(files),
                   f"{len(files)} file(s)" if files
                   else f"{len(entries)} entries but no *HISTORY*.csv — "
                        f"wrong folder?")
            if files:
                newest = max(files, key=lambda f: f.stat().st_mtime)
                age_h = (time.time() - newest.stat().st_mtime) / 3600.0
                report("the feed is current", age_h < 6,
                       f"newest file is {age_h:.1f} h old ({newest.name})")
    else:
        report("history CSVs present", False, "directory not mounted")

    # Resolved rather than merely tested for existence, because the inventory
    # is an HOURLY export -- `hts_HISTCURR_2026Sep22-130000` -- not the single
    # pre-merged file the old pipeline used. Checking the configured path
    # literally reported FAIL while the inventory sat in the same directory
    # under a different name, which is the least useful thing a pre-flight can
    # say.
    from das2.io.ingest import resolve_inventory_path
    try:
        inventory = resolve_inventory_path(config.ingest.inventory_path)
        report("inventory file exists", True, str(inventory))
    except FileNotFoundError as exc:
        report("inventory file exists", False, str(exc)[:200])
    longlat = Path(config.ingest.longlat_path) if config.ingest.longlat_path else None
    report("LongLat.csv exists", bool(longlat and longlat.exists()),
           str(longlat) if longlat
           else "not configured — no map, no geo-clustering, no by-region view")

    print("\nDatabase")
    try:
        from sqlalchemy import text
        from das2.io.store import make_engine
        engine = make_engine(config.database.sqlalchemy_url())
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        report("connection", True, config.database.safe_url)
        with engine.connect() as conn:
            try:
                n = conn.execute(text("SELECT COUNT(*) FROM das2_incident")).scalar()
                report("das2 tables present", True, f"das2_incident has {n} row(s)")
            except Exception:                              # noqa: BLE001
                report("das2 tables present", False,
                       "run `python -m das2.cli migrate` first")
    except Exception as exc:                               # noqa: BLE001
        report("connection", False, str(exc)[:160])

    print("\nTelegram")
    if not config.alert.telegram_token or not config.alert.telegram_chat_id:
        report("credentials configured", False,
               "set DAS2_ALERT_TELEGRAM_TOKEN and DAS2_ALERT_TELEGRAM_CHAT_ID")
    else:
        from das2.alerting.telegram import TelegramClient, TelegramConfig
        client = TelegramClient(TelegramConfig(
            token=config.alert.telegram_token,
            chat_id=config.alert.telegram_chat_id))
        try:
            me = client._call("getMe", {})
            report("bot reachable", bool(me.get("ok")),
                   f"@{me.get('result', {}).get('username', '?')}")
            client.send_message("✅ DAS2 connectivity check — "
                                "this chat will receive incident alerts.")
            report("test message delivered to chat", True,
                   config.alert.telegram_chat_id)
        except Exception as exc:                           # noqa: BLE001
            report("bot reachable", False, str(exc)[:160])

    print("\nClassifier")
    try:
        from das2.io.classify import get_classifier
        classifier = get_classifier(config.ingest.equipment_rules_path or None)
        report("rules loaded", True,
               f"{len(classifier.classes)} equipment classes")
    except Exception as exc:                               # noqa: BLE001
        report("rules loaded", False, str(exc)[:160])

    print(f"\n{'All checks passed.' if ok else 'Some checks FAILED — see above.'}\n")
    return 0 if ok else 1


def cmd_run(config: Config, args) -> int:
    from das2 import pipeline
    from das2.report import charts, dashboard

    engine = None
    open_incidents: list = []
    baselines: dict = {}
    if not args.dry_run:
        from sqlalchemy import text

        from das2.io.store import load_open_incidents, make_engine

        # Build the engine and probe it before anything else, and STOP if
        # either fails. Construction is inside the guard because it is not
        # merely lazy bookkeeping: create_engine imports the DBAPI, so a
        # missing pyodbc or an unparseable URL raises here, and an uncaught
        # traceback out of a scheduled container is a poor way to say
        # "the database settings are wrong".
        #
        # The loads below degrade gracefully, which is right for a table that
        # is empty or a query that failed once. It is wrong for a database that
        # cannot be reached at all: the run would carry on, treat every
        # incident as new because it could not read the open ones, fail to
        # persist them, and then alert -- repeating the whole set every run
        # forever. Suppressing exactly that repetition is what the incident
        # layer is for, so a database outage must not be allowed to turn it off
        # quietly. Better to run nothing than to flood the operators' chat.
        try:
            engine = make_engine(config.database.sqlalchemy_url())
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception as exc:                           # noqa: BLE001
            log.error("cannot reach the database at %s: %s",
                      config.database.safe_url, exc)
            log.error("run `docker compose run --rm das2-check` for the full "
                      "pre-flight, or `--dry-run` to analyse without a "
                      "database.")
            return 3

        try:
            open_incidents = load_open_incidents(engine)
        except Exception as exc:                           # noqa: BLE001
            log.warning("could not load open incidents (%s) — "
                        "every incident will be treated as new", exc)
        try:
            from das2.io.store import load_baselines
            baselines = load_baselines(engine)
        except Exception as exc:                           # noqa: BLE001
            log.warning("could not load time-of-day baselines (%s) — "
                        "the L2 residual layer will abstain", exc)

    result = pipeline.run(config, open_incidents=open_incidents,
                          baselines=baselines)

    if not result.anomalies and result.readings.empty:
        log.error("no readings were analysed — check the history directory")
        return 2

    out_dir = Path(config.report.output_dir)
    html_path = dashboard.write(result, out_dir)
    print(f"\nDashboard: {html_path}")

    chart_paths = {}
    try:
        chart_paths = charts.run_charts(result, out_dir)
        for name, path in chart_paths.items():
            print(f"Chart ({name}): {path}")
    except Exception as exc:                               # noqa: BLE001
        log.error("chart generation failed (alerting continues): %s", exc)

    _print_summary(result)

    if engine is not None:
        from das2.io.store import (prune_readings, save_readings, save_run,
                                   upsert_sensors)
        try:
            upsert_sensors(engine, result.sensors)
            save_run(engine, result)
            if config.database.store_readings:
                # Feeds the daily profile job. Without it DRIFT, NOISE_BURST
                # and the L2 baselines have no source of history.
                save_readings(engine, result.readings)
                # ...and bound it, or the table grows without limit.
                prune_readings(engine, config.database.reading_retention_days)
            print("Persisted to the database.")
        except Exception as exc:                           # noqa: BLE001
            log.error("persistence failed: %s", exc)

    if args.dry_run or args.no_alert:
        print("\nAlerting skipped (dry run).")
        return 0

    from das2.alerting.telegram import TelegramConfig, send_run
    report = send_run(
        result,
        TelegramConfig(token=config.alert.telegram_token,
                       chat_id=config.alert.telegram_chat_id,
                       enabled=config.alert.enabled),
        charts=chart_paths,
        dashboard_url=config.report.public_url or None,
        max_incidents=config.alert.max_incidents_per_run,
    )
    print(f"Telegram: {report.as_dict()}")

    if engine is not None:
        from das2.alerting.telegram import compose
        from das2.io.store import record_delivery
        for incident in result.incidents:
            sent = incident.incident_id in report.sent
            try:
                record_delivery(engine, incident.incident_id, "telegram",
                                payload=compose(incident), suppressed=not sent,
                                reason="" if sent else incident.incident_class.value)
            except Exception:                              # noqa: BLE001
                pass
    return 0


def cmd_demo(config: Config, args) -> int:
    """
    Generate synthetic data with known faults and run against it.

    The point is to separate "the system is installed correctly" from "the
    system found nothing in your data", which otherwise look identical. The
    fixture contains faults whose answers are known, so if this prints a
    regional event across three East sites then ingest, classification,
    detection, clustering, triage and reporting are all working, and any
    silence on real data is a statement about the data rather than the install.

    Touches nothing: no database, no Telegram, and a temporary directory it
    cleans up unless asked to keep it.
    """
    import subprocess
    import tempfile

    from das2 import pipeline
    from das2.report import charts, dashboard

    root = Path(args.keep) if args.keep else Path(tempfile.mkdtemp(prefix="das2-demo-"))
    fixtures, out = root / "fixtures", Path(config.report.output_dir)
    print(f"Generating synthetic data with known faults in {fixtures} ...")
    subprocess.run([sys.executable, "-m", "tools.make_fixtures",
                    "--out", str(fixtures)], check=True,
                   capture_output=True, cwd=str(Path(__file__).resolve().parent.parent))

    demo = load_config(args.config)
    demo.ingest.history_dir = str(fixtures / "HISTORY")
    demo.ingest.histcurr_path = str(fixtures / "HISTCURR" / "histcurr_fujitsu.csv")
    demo.ingest.longlat_path = str(fixtures / "LongLat.csv")
    demo.report.output_dir = str(out)

    result = pipeline.run(demo)
    _print_summary(result)
    print(f"\nDashboard: {dashboard.write(result, out)}")
    try:
        for name, path in charts.run_charts(result, out).items():
            print(f"Chart ({name}): {path}")
    except Exception as exc:                               # noqa: BLE001
        log.error("chart generation failed: %s", exc)

    from das2.models import IncidentClass
    regional = [i for i in result.incidents
                if i.incident_class is IncidentClass.REGIONAL_EVENT]
    print("\n" + "=" * 68)
    if regional:
        sites = ", ".join(sorted(regional[0].cluster.sites))
        print(f"OK — the injected regional event was found across {sites}.")
        print("Ingest, classification, detection, clustering, triage and the")
        print("dashboard are all working. Point it at your real data next.")
    else:
        print("PROBLEM — the injected regional event was NOT found.")
        print("Something upstream is broken; the detail above says where.")
    print("=" * 68)

    if not args.keep:
        import shutil
        shutil.rmtree(root, ignore_errors=True)
    else:
        print(f"\nFixtures kept in {root}")
    return 0 if regional else 1


def cmd_profile(config: Config, args) -> int:
    """
    The daily job: long-history baselines, DRIFT and NOISE_BURST.

    Separate from the hourly run because these answers change slowly, cost far
    more to compute, and genuinely cannot be derived from a 72-hour window --
    a 1%/day drift is 3% across it while the daily demand cycle is 10-30%.
    Run it once a day; the hourly run scores against what it stores.
    """
    from das2.io.store import load_history, make_engine, save_baselines
    from das2.profile.build import MIN_DAYS_DRIFT, PREFERRED_DAYS, run_profile_job

    engine = make_engine(config.database.sqlalchemy_url())
    days = args.days or PREFERRED_DAYS
    print(f"Reading up to {days} days of history ...")
    history = load_history(
        engine, days=days,
        fallback_table=config.database.history_fallback_table)

    if history.empty:
        print("\nNo history yet — das2_reading is empty.")
        print("This is the expected state on a fresh install, not a failure. "
              "The hourly run fills das2_reading as it goes; come back once it "
              "has been running.")
        print(f"\n  DRIFT          needs {MIN_DAYS_DRIFT} days (prefers "
              f"{PREFERRED_DAYS})")
        print("  NOISE_BURST    needs 7 days")
        print("  the L2 baseline layer needs enough days to fill its "
              "time-of-day buckets")
        print("\nUntil then those three produce nothing, and the hourly run "
              "says so in its `baselines:` line.")
        if not config.database.history_fallback_table:
            print("\nIf you have an existing readings table you would rather "
                  "learn from, point DAS2_DATABASE_HISTORY_FALLBACK_TABLE at "
                  "it; it must expose sensor_key, ts and value.")
        return 2

    observed = history.groupby("sensor_key")["ts"].apply(
        lambda s: s.dt.normalize().nunique())
    print(f"{len(history):,} readings, {history['sensor_key'].nunique()} sensors, "
          f"median {observed.median():.0f} days each")

    result = run_profile_job(history)
    print(f"\n{result.summary()}")

    if observed.median() < MIN_DAYS_DRIFT:
        print(f"\nNOTE: median history is under {MIN_DAYS_DRIFT} days, so DRIFT "
              f"is not being computed. It is not a failure -- the slope simply "
              f"cannot be separated from the daily cycle over a shorter span.")

    save_baselines(engine, result.baselines)
    print("Baselines stored. The hourly run will score against them from now on.")
    return 0


def cmd_ack_worker(config: Config, args) -> int:
    from das2.alerting.telegram import TelegramConfig, run_ack_worker
    from das2.io.store import make_engine, record_ack

    engine = make_engine(config.database.sqlalchemy_url())

    def on_ack(ref, state, user):
        record_ack(engine, ref, state, user)

    print("Acknowledge worker running. Ctrl-C to stop.")
    run_ack_worker(
        TelegramConfig(token=config.alert.telegram_token,
                       chat_id=config.alert.telegram_chat_id),
        on_ack,
        offset_file=Path(config.report.output_dir) / "telegram_offset.json",
        stop_after=args.cycles,
    )
    return 0


#: Touched after every run. The container healthcheck reads its age, because a
#: scheduler that has silently died looks exactly like a quiet network from the
#: outside -- and "no alerts" is the state this system is supposed to produce
#: most of the time, so that ambiguity would otherwise hide a dead job for days.
HEARTBEAT_PATH = Path("/data/logs/heartbeat")


def _touch_heartbeat(config: Config) -> None:
    for candidate in (HEARTBEAT_PATH, Path(config.report.output_dir) / "heartbeat"):
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            candidate.touch()
            return
        except OSError:
            continue        # not fatal: a missing heartbeat must not stop a run


def cmd_schedule(config: Config, args) -> int:
    """
    Run forever on an interval.

    A failed run logs and waits for the next tick rather than exiting: a
    malformed CSV at 02:00 must not take the monitoring system down until
    somebody notices in the morning. The heartbeat is touched either way, since
    the process is alive and the failure is already in the log -- a healthcheck
    that restarts the container on a bad input file would just lose the log.
    """
    interval = args.interval_minutes or config.run_interval_minutes
    print(f"Scheduler started — a run every {interval} minute(s). Ctrl-C to stop.")
    while True:
        started = time.time()
        try:
            cmd_run(config, args)
        except KeyboardInterrupt:
            print("\nStopped.")
            return 0
        except Exception:                                  # noqa: BLE001
            log.exception("run failed — continuing to the next interval")
        _touch_heartbeat(config)
        elapsed = time.time() - started
        sleep_for = max(30.0, interval * 60 - elapsed)
        log.info("next run in %.0f minute(s)", sleep_for / 60)
        time.sleep(sleep_for)


def _print_summary(result) -> None:
    print("\n" + "=" * 68)
    print(f"RUN {result.run_id}   ({result.duration_s}s)")
    if result.window_start:
        print(f"Window: {result.window_start} -> {result.window_end}")
    print("-" * 68)
    for key in ("ingest", "coverage", "profiles", "baselines", "detection",
                "mass_balance", "correlation", "clustering", "incidents",
                "lifecycle", "selection"):
        if key in result.stats:
            print(f"{key:>12}: {result.stats[key]}")
    print("-" * 68)
    if not result.incidents:
        print("No incidents. Every sensor behaved within its own normal range.")
    for incident in result.incidents[:15]:
        flag = " " if incident.should_alert else "~"
        sites = ", ".join(sorted(incident.cluster.sites)) or "unnamed"
        print(f"{flag}{incident.priority.value} {incident.incident_class.value:<18} "
              f"{str(incident.cluster.region or '-'):<11} "
              f"{len(incident.cluster.members):>2} sensor(s)  {sites}")
        print(f"    -> {incident.recommendation}")
    print("(~ = suppressed, will not page anyone)")
    print("=" * 68)


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="das2", description="Water sensor anomaly intelligence")
    parser.add_argument("--config", help="path to config.yaml")
    parser.add_argument("--log-level", default=os.environ.get("DAS2_LOG_LEVEL", "INFO"),
                        help="DEBUG | INFO | WARNING | ERROR "
                             "(or set DAS2_LOG_LEVEL)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_migrate = sub.add_parser("migrate",
                               help="create/update the database tables")
    p_migrate.add_argument("--print-sql", action="store_true",
                           help="print the SQL instead of running it, for "
                                "review or for pasting into SSMS. Connects to "
                                "nothing.")
    p_migrate.add_argument("--dialect", default="mssql",
                           choices=["mssql", "sqlite"],
                           help="dialect for --print-sql (default: mssql)")
    sub.add_parser("check", help="verify data, database and Telegram")

    demo = sub.add_parser(
        "demo", help="run against synthetic data with known faults")
    demo.add_argument("--keep", metavar="DIR",
                      help="keep the generated fixtures in DIR")

    profile = sub.add_parser(
        "profile", help="daily job: baselines, DRIFT and NOISE_BURST")
    profile.add_argument("--days", type=int, default=None,
                         help="how much history to read (default 28)")

    run = sub.add_parser("run", help="one analysis run")
    run.add_argument("--dry-run", action="store_true",
                     help="no database writes, no Telegram")
    run.add_argument("--no-alert", action="store_true",
                     help="persist, but do not send Telegram messages")

    worker = sub.add_parser("ack-worker", help="consume acknowledge buttons")
    worker.add_argument("--cycles", type=int, default=None,
                        help="stop after N poll cycles (for testing)")

    sched = sub.add_parser("schedule", help="run on an interval, forever")
    sched.add_argument("--interval-minutes", type=int, default=None)
    sched.add_argument("--dry-run", action="store_true")
    sched.add_argument("--no-alert", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.log_level)
    config = load_config(args.config)

    handlers = {
        "migrate": cmd_migrate,
        "check": cmd_check,
        "demo": cmd_demo,
        "profile": cmd_profile,
        "run": cmd_run,
        "ack-worker": cmd_ack_worker,
        "schedule": cmd_schedule,
    }
    try:
        return handlers[args.command](config, args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
