"""
Tests for das2.config and das2.models (Phase 1).

These pin the two rules the rewrite depends on:

  * config durations are SECONDS, never sample counts -- the defect that made
    ROLL_WIN_Z=24 mean 6.3 minutes on one sensor and 49.8 on another;
  * severity is PHYSICAL -- the defect that let Peak_RZ=109.7 (a 1.8%
    excursion) outrank Peak_RZ=306.9 (a full-scale event) because the first
    sensor had a finer quantisation step.

Run:  python3 tests/test_config_models.py
"""

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from das2.config import Config, DatabaseConfig  # noqa: E402
from das2.models import (  # noqa: E402
    AckState,
    AnomalyType,
    Cluster,
    Incident,
    IncidentClass,
    PhysicalSeverity,
    Priority,
    SENSOR_HEALTH_TYPES,
    SensorAnomaly,
    SensorMeta,
    Signal,
)


def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
    if not cond:
        raise SystemExit(1)


T0 = datetime(2026, 3, 1, 6, 0, 0)


def sensor(**kw):
    base = dict(sensor_key="k1", description="BedokPS-Pump4-Deliver-Pressure",
                equipment="Pressure", site="BedokPS", region="East")
    base.update(kw)
    return SensorMeta(**base)


def anomaly(equipment="Pressure", key="k1", atype=AnomalyType.FLATLINE,
            start=T0, minutes=60, score_fields=None, **sensor_kw):
    sev = PhysicalSeverity(**(score_fields or {}))
    return SensorAnomaly(
        sensor=sensor(sensor_key=key, equipment=equipment, **sensor_kw),
        start=start, end=start + timedelta(minutes=minutes),
        dominant_type=atype, severity=sev,
    )


def main():
    # ------------------------------------------------------------------ #
    print("config: defaults are durations, not sample counts")
    cfg = Config()
    check("no field is named like a sample count",
          not any("win" in f and not f.endswith(("_s", "_seconds", "_min", "_minutes",
                                                 "_hours", "_days"))
                  for f in vars(cfg.baseline)),
          "(the ROLL_WIN_Z=24 defect must not reappear)")
    check("resample uses step-hold, not mean", cfg.resample.method == "locf",
          "(averaging fabricates values report-by-exception never held)")
    check("flatline threshold is in seconds", cfg.health.flatline_min_seconds == 1800)
    check("range tolerance exists at all", cfg.health.range_tolerance_sigma > 0,
          "(without it an idle flowmeter yields ~1000 false violations)")
    check("baseline needs real history", cfg.baseline.min_profile_days >= 14)
    check("window_hours and continuity are independent knobs",
          cfg.ingest.window_hours == 72 and cfg.incident.continuity_gap_min == 180,
          "(no cadence assumption)")

    print("\nconfig: layering")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "c.json"
        p.write_text(json.dumps({
            "spatial": {"cluster_radius_m": 750.0},
            "incident": {"regional_min_sensors": 5},
            "bogus_section": {"x": 1},
            "alert": {"not_a_real_field": 1, "enabled": False},
        }))
        cfg = Config.load(p, use_env=False)
        check("file overrides a default", cfg.spatial.cluster_radius_m == 750.0)
        check("second section applied", cfg.incident.regional_min_sensors == 5)
        check("untouched defaults survive", cfg.spatial.min_cluster_size == 2)
        check("unknown section ignored, not fatal", not hasattr(cfg, "bogus_section"))
        check("unknown field ignored but siblings applied", cfg.alert.enabled is False)

    print("\nconfig: env overrides win over file")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "c.json"
        p.write_text(json.dumps({"spatial": {"cluster_radius_m": 750.0}}))
        os.environ["DAS2_SPATIAL_CLUSTER_RADIUS_M"] = "1234.5"
        os.environ["DAS2_ALERT_ENABLED"] = "false"
        os.environ["DAS2_INCIDENT_REGIONAL_MIN_SENSORS"] = "7"
        try:
            cfg = Config.load(p)
            check("float from env", cfg.spatial.cluster_radius_m == 1234.5)
            check("bool from env", cfg.alert.enabled is False)
            check("int from env", cfg.incident.regional_min_sensors == 7)
        finally:
            for k in ("DAS2_SPATIAL_CLUSTER_RADIUS_M", "DAS2_ALERT_ENABLED",
                      "DAS2_INCIDENT_REGIONAL_MIN_SENSORS"):
                os.environ.pop(k, None)

    print("\nconfig: legacy DAS_* credentials still work")
    os.environ["DAS_MSSQL_HOST"] = "10.0.0.5"
    os.environ["DAS_MSSQL_PORT"] = "14330"
    try:
        cfg = Config.load(None)
        check("existing container env is honoured",
              cfg.database.host == "10.0.0.5" and cfg.database.port == 14330,
              "(drops into the current deployment unchanged)")
    finally:
        os.environ.pop("DAS_MSSQL_HOST", None)
        os.environ.pop("DAS_MSSQL_PORT", None)

    print("\nconfig: credentials never leak into logs")
    cfg = Config()
    cfg.database.password = "hunter2"
    check("to_dict masks the password", cfg.to_dict()["database"]["password"] == "***")
    check("repr does not contain it", "hunter2" not in repr(cfg))
    check("but the real URL still uses it", "hunter2" in cfg.database.sqlalchemy_url())
    check("safe_url masks it", "hunter2" not in cfg.database.safe_url)

    # The leak that mattered: with the whole URL in DAS2_DATABASE_URL -- which
    # is how DEPLOY.md recommends configuring SQL Server -- `password` is empty
    # and masking it alone is a no-op, so the credential went into every log
    # line that printed safe_url, including the connection-failure error.
    leaky = DatabaseConfig(url="mssql+pyodbc://sa:hunter2@10.0.0.5:1433/anomaly_db")
    check("a password inside DAS2_DATABASE_URL is masked too",
          "hunter2" not in leaky.safe_url, leaky.safe_url)
    check("and the rest of the URL survives masking",
          "sa:***@10.0.0.5:1433/anomaly_db" in leaky.safe_url)
    escaped = DatabaseConfig(url="mssql+pyodbc://sa:p%40ss%3Aword@h/db")
    check("a percent-escaped password is masked", "p%40ss" not in escaped.safe_url)
    cfg.database.url = "sqlite:///x.db"
    check("explicit url wins", cfg.database.sqlalchemy_url() == "sqlite:///x.db")

    # ------------------------------------------------------------------ #
    # A hand-written DAS2_DATABASE_URL that omits ?driver= is accepted by
    # SQLAlchemy and fails only at connect time, as
    #   IM002 ... Data source name not found and no default driver specified
    # which names neither the setting at fault nor the file it lives in. It
    # cost a real deployment its first run. The image ships exactly one ODBC
    # driver, so the value is never ambiguous -- fill it in rather than
    # report it.
    print("\nconfig: a URL without ?driver= is completed, not left to fail")
    cfg = Config()
    cfg.database.url = "mssql+pyodbc://sa:pass@10.0.0.5:1433/anomaly_db"
    completed = cfg.database.sqlalchemy_url()
    check("driver is supplied", "driver=ODBC+Driver+18+for+SQL+Server" in completed)
    check("TrustServerCertificate too", "TrustServerCertificate=yes" in completed,
          "(the container trusts no CA the SQL Server was issued under)")
    # Driver 18 encrypts by default, and TrustServerCertificate only skips
    # CERTIFICATE VALIDATION -- it does not turn encryption off. On the
    # client's server the handshake stalled until the login timer expired, so
    # the error said "Login timeout expired" and looked like an unreachable
    # host while a plain TCP connect to 1433 succeeded.
    check("the connection is encrypted by default", "Encrypt=yes" in completed,
          "(the password crosses a network either way)")
    off = DatabaseConfig(host="h", database="d", username="u", password="p",
                         encrypt="no").sqlalchemy_url()
    check("and encryption can be turned off when a server cannot do TLS",
          "Encrypt=no" in off, "(last resort: this sends the password in clear)")
    pinned = DatabaseConfig(url="mssql+pyodbc://u:p@h/d?Encrypt=yes",
                            encrypt="no").sqlalchemy_url()
    check("an Encrypt already in the URL is never overridden",
          "Encrypt=yes" in pinned and "Encrypt=no" not in pinned)
    # ------------------------------------------------------------------ #
    # The defect that cost an evening. The client's password contains `@`:
    #
    #   mssql+pyodbc://flotech:P@ssword1234@192.168.25.16:1433/anomaly_db
    #
    # SQLAlchemy parses the password as [^@]*, stopping at the FIRST `@`, so
    # pyodbc received Server=ssword1234@192.168.25.16,1433 and PWD=P. Every
    # run then waited on a host that does not exist and reported HYT00 Login
    # timeout expired -- which is what an unreachable SERVER looks like. A
    # plain TCP connect to the real address succeeded throughout, so the
    # diagnosis went to the network, then TLS, then the login timeout.
    print("\nconfig: a raw @ in the password does not become the hostname")
    import sqlalchemy as _sa

    raw = DatabaseConfig(
        url="mssql+pyodbc://flotech:P@ssword1234@192.168.25.16:1433/anomaly_db")
    parsed = _sa.engine.url.make_url(raw.sqlalchemy_url())
    check("the host is the host", parsed.host == "192.168.25.16",
          f"({parsed.host!r}; it was 'ssword1234@192.168.25.16')")
    check("and the whole password survives", parsed.password == "P@ssword1234",
          f"({parsed.password!r}; it was 'P')")

    already = DatabaseConfig(
        url="mssql+pyodbc://flotech:P%40ssword1234@192.168.25.16:1433/anomaly_db")
    ap = _sa.engine.url.make_url(already.sqlalchemy_url())
    check("a correctly escaped password is left alone",
          ap.host == "192.168.25.16" and ap.password == "P@ssword1234")

    many = DatabaseConfig(url="mssql+pyodbc://u:a@b@c@192.168.25.16:1433/d")
    mp = _sa.engine.url.make_url(many.sqlalchemy_url())
    check("several @ still split at the last one",
          mp.host == "192.168.25.16" and mp.password == "a@b@c",
          f"({mp.password!r})")

    # urlsplit and SQLAlchemy disagree about where userinfo ends, and safe_url
    # used the wrong one -- so every log line showed a healthy URL for a
    # connection that had never reached the server. A diagnostic that parses
    # differently from the thing it describes is worse than none.
    check("safe_url reports the host that will actually be used",
          "192.168.25.16" in raw.safe_url
          and "ssword1234@" not in raw.safe_url,
          raw.safe_url)
    check("and still hides the password", "ssword1234" not in raw.safe_url)

    check("the default driver matches the one the Dockerfile installs",
          DatabaseConfig().driver == "ODBC Driver 18 for SQL Server",
          "(it said 17 for a while; the image has never carried 17)")

    cfg.database.url = ("mssql+pyodbc://sa:pass@h:1433/db"
                        "?driver=ODBC+Driver+17+for+SQL+Server")
    check("an explicit driver is never overridden",
          "Driver+17" in cfg.database.sqlalchemy_url()
          and "Driver+18" not in cfg.database.sqlalchemy_url())

    for untouched in ("sqlite:////data/output/das2.db",
                      "postgresql://u:p@h/db",
                      "mssql+pyodbc://sa:pass@SomeDSN",
                      "mssql+pyodbc:///?odbc_connect=DRIVER%3D%7BODBC+18%7D"):
        cfg.database.url = untouched
        check(f"passed through unchanged: {untouched[:38]}",
              cfg.database.sqlalchemy_url() == untouched)

    # ------------------------------------------------------------------ #
    print("\nmodels: severity is physical, and comparable across equipment")
    # The real regression: a 1.8% voltage excursion must NOT outrank a
    # full-scale flow event, which is exactly what Peak_RZ did.
    small_v = PhysicalSeverity(deviation=7.4, unit="V", span_fraction=0.018,
                               duration_s=600, window_fraction=0.01)
    big_flow = PhysicalSeverity(deviation=200.0, unit="L/s", span_fraction=0.96,
                                duration_s=7200, window_fraction=0.30)
    check("full-scale flow event outranks a 1.8% voltage blip",
          big_flow.score() > small_v.score(),
          f"({big_flow.score():.0f} vs {small_v.score():.0f})")
    check("scores stay within 0..100",
          0 <= small_v.score() <= 100 and 0 <= big_flow.score() <= 100)
    check("an empty severity scores 0", PhysicalSeverity().score() == 0.0)
    check("unknown span does not crash the score",
          PhysicalSeverity(duration_s=3600, window_fraction=0.5).score() > 0,
          "(falls back to duration and coverage)")
    check("longer duration scores higher, all else equal",
          PhysicalSeverity(span_fraction=0.5, duration_s=7200).score() >
          PhysicalSeverity(span_fraction=0.5, duration_s=60).score())

    print("\nmodels: priority banding")
    check("P1 for severe", Priority.from_score(90) is Priority.P1)
    check("P2 mid", Priority.from_score(60) is Priority.P2)
    check("P3 low", Priority.from_score(30) is Priority.P3)
    check("P4 trivial", Priority.from_score(5) is Priority.P4)

    print("\nmodels: anomaly typing drives the decision")
    flat = anomaly(atype=AnomalyType.FLATLINE)
    shift = anomaly(atype=AnomalyType.LEVEL_SHIFT)
    check("FLATLINE is a sensor-health fault", flat.is_sensor_health)
    check("LEVEL_SHIFT is a process signal", shift.is_process)
    check("the two are disjoint", not flat.is_process and not shift.is_sensor_health)
    check("STALE counts as sensor health", AnomalyType.STALE in SENSOR_HEALTH_TYPES)
    check("duration is derived from the span", flat.duration_s == 3600)

    print("\nmodels: signals retain the evidence")
    s1 = Signal(AnomalyType.FLATLINE, T0, T0 + timedelta(minutes=30), "health", 0.0, "bar", 12)
    s2 = Signal(AnomalyType.SPIKE, T0 + timedelta(minutes=20),
                T0 + timedelta(minutes=25), "health", 3.2, "bar/s", 2)
    s3 = Signal(AnomalyType.DRIFT, T0 + timedelta(hours=5),
                T0 + timedelta(hours=6), "profile", 0.4, "bar", 60)
    a = anomaly()
    a.signals = [s1, s2]
    check("types are collected", a.types == {AnomalyType.FLATLINE, AnomalyType.SPIKE})
    check("overlapping signals detected", s1.overlaps(s2))
    check("non-overlapping signals detected", not s1.overlaps(s3))
    # s1 ends at T0+30min, s3 starts at T0+5h -> a 4.5h gap.
    check("a tolerance narrower than the gap does not bridge it",
          not s1.overlaps(s3, tolerance_s=4 * 3600))
    check("a tolerance wider than the gap does",
          s1.overlaps(s3, tolerance_s=5 * 3600))
    check("signal duration", s1.duration_s == 1800)

    print("\nmodels: clusters group sites, not sensor positions")
    c = Cluster(members=[
        anomaly(key="a", equipment="Pressure", site="BedokPS"),
        anomaly(key="b", equipment="Flowrate", site="BedokPond4",
                start=T0 + timedelta(minutes=10)),
        anomaly(key="c", equipment="Flowrate", site="BedokPS"),
    ], region="East")
    check("member keys collected", c.sensor_keys == {"a", "b", "c"})
    check("equipment diversity visible", c.equipment_types == {"Pressure", "Flowrate"})
    check("distinct sites visible", c.sites == {"BedokPS", "BedokPond4"})
    check("cluster window spans its members",
          c.start == T0 and c.end == T0 + timedelta(minutes=70))
    check("empty cluster has no window", Cluster().start is None)

    print("\nmodels: incident decides the action")
    inc = Incident(incident_id="i1", cluster=c,
                   incident_class=IncidentClass.SENSOR_FAULT, severity=80.0)
    check("P1 at severity 80", inc.priority is Priority.P1)
    check("sensor fault is dispatchable", inc.should_dispatch)
    check("sensor fault alerts", inc.should_alert)
    check("recommendation is human-readable", "technician" in inc.recommendation.lower())

    fan = Incident(incident_id="i2", cluster=c,
                   incident_class=IncidentClass.TELEMETRY_FANOUT, severity=90.0)
    check("fan-out never alerts, even at severity 90", not fan.should_alert,
          "(it is noise by construction)")
    check("fan-out is not dispatchable", not fan.should_dispatch)

    watch = Incident(incident_id="i3", cluster=c, incident_class=IncidentClass.WATCH)
    check("WATCH does not alert", not watch.should_alert)

    rain = Incident(incident_id="i4", cluster=c,
                    incident_class=IncidentClass.WEATHER_DRIVEN, severity=70.0)
    check("weather-driven alerts but does not dispatch",
          rain.should_alert and not rain.should_dispatch,
          "(the operator is told, and told not to drive out)")

    check("every class has a recommendation",
          all(Incident(incident_id="x", cluster=c, incident_class=k).recommendation
              for k in IncidentClass))
    check("ack defaults to none", inc.ack_state is AckState.NONE)

    # ------------------------------------------------------------------ #
    print("\nthe source fingerprint survives a Windows checkout")
    # Hashing raw bytes made the fingerprint depend on git's core.autocrlf:
    # the same commit read b94d624cd922 on Windows and 14e611e38aba on Linux.
    # A fingerprint that differs between two CORRECT checkouts reports a
    # mismatch precisely when there is none -- worse than not reporting at
    # all, since it sent a deployment chasing a rebuild it had already done.
    import hashlib as _hl
    from pathlib import Path as _P

    import das2 as _das2

    root = _P(_das2.__file__).resolve().parent

    def _fingerprint(convert) -> str:
        digest = _hl.sha256()
        for path in sorted(
            q for pat in ("*.py", "*.yaml", "*.yml")
            for q in root.rglob(pat) if "__pycache__" not in q.parts
        ):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(convert(path.read_bytes()))
        return digest.hexdigest()[:12]

    lf = _fingerprint(lambda b: b.replace(b"\r\n", b"\n"))
    crlf = _fingerprint(
        lambda b: b.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
                   .replace(b"\r\n", b"\n"))
    check("LF and CRLF checkouts agree", lf == crlf, lf)
    check("and it matches what the package reports",
          _das2.source_fingerprint() == lf)
    check("build_stamp carries the fingerprint",
          _das2.source_fingerprint() in _das2.build_stamp())

    # ------------------------------------------------------------------ #
    print("\ntriage: a hundred quiet sensors is a feed, not a hundred faults")
    # On the client's first real run, three of five P1 alerts read
    # "Instrument fault ... dispatch a technician" over 116, 123 and 115
    # sensors spanning six sites each. The cause was visible in the same run's
    # ingest stats: a 24-hour hole in the historian feed, which makes every
    # sensor STALE at once. Sending crews to six sites for a comms outage is
    # exactly the wasted trip this system exists to prevent.
    from das2.incident.triage import (  # noqa: E402
        OUTAGE_MIN_MEMBERS,
        classify,
    )

    def cluster_of(n, sites, atype=AnomalyType.STALE):
        members = []
        for i in range(n):
            site = sites[i % len(sites)]
            members.append(anomaly(key=f"k{i}", atype=atype, site=site,
                                   rtu_number=str(1000 + i)))
        return Cluster(members=members)

    big = cluster_of(116, ["Jurong PS", "Jurong TG", "Pandan 1 PS",
                           "Pandan 2 PS", "Pandan TG", "Pandan 1 Ext"])
    klass, why = classify(big)
    check("116 stale sensors over 6 sites is an outage",
          klass is IncidentClass.TELEMETRY_OUTAGE, f"({klass.value})")
    inc = Incident(incident_id="x", cluster=big, incident_class=klass)
    check("it still alerts -- a dead feed blinds the detector",
          inc.should_alert)
    check("but nobody is dispatched", not inc.should_dispatch,
          "(the remedy is a link, not a truck)")
    check("and the recommendation says so",
          "Do NOT dispatch per sensor" in inc.recommendation)
    check("the evidence names the reasoning",
          any("telemetry path" in w for w in why))

    # The rule must not swallow the case it was carved out of.
    small = cluster_of(4, ["Bedok PS", "Tampines Pond B"])
    check("a handful of stale sensors is still a sensor fault",
          classify(small)[0] is IncidentClass.SENSOR_FAULT,
          f"(n=4, below the threshold of {OUTAGE_MIN_MEMBERS})")
    one_site = cluster_of(40, ["Bedok PS"])
    check("many sensors at ONE site remains fan-out",
          classify(one_site)[0] is IncidentClass.TELEMETRY_FANOUT,
          "(one panel, already handled, and never paged)")
    process = cluster_of(30, ["Jurong PS", "Jurong TG", "Pandan TG"],
                         atype=AnomalyType.LEVEL_SHIFT)
    check("a large PROCESS event is not mistaken for an outage",
          classify(process)[0] is not IncidentClass.TELEMETRY_OUTAGE,
          "(the water moving is not the feed failing)")

    print("\nAll config/model tests passed.")


if __name__ == "__main__":
    main()
