#!/usr/bin/env python3
"""
The daily job: it has to RUN, and it has to survive the real fleet.

Three detectors depend entirely on this job -- DRIFT, NOISE_BURST and
RESIDUAL_OUTLIER -- and they are precisely the ones that answer the client's
narrowed scope:

    "we want to anticipate the potential unnormal, or abnormal behavior of the
     sensor based on the stats"

On the client's deployment all three produced nothing, for weeks, and the only
sign of it was one quiet line in every run log:

    baselines: {'sensors': 0, 'usable': 0}

The job was written, tested, containerised and documented. It was never
scheduled, because scheduling it was a sentence in a compose-file comment. So
this file pins two separate things:

  * **it runs.** The hourly run checks the age of the stored baselines itself
    and rebuilds them when they are overdue, rather than depending on a cron
    nobody set up. A dependency that fails by being silent has to be checked by
    the thing that depends on it.

  * **it scales.** `load_history` read the whole window into one DataFrame --
    28 days x 2,672 sensors x 120 s is on the order of 50 million rows, which is
    fine on the fixture and tens of gigabytes on the estate. The job now reads a
    few hundred sensors at a time, and the assertions below check that the
    batching changes the memory profile WITHOUT changing the arithmetic.

The last section is the one that matters most: it proves a baseline, once
stored, actually makes the L2 layer fire. Everything else here could pass while
the feature stayed useless.

Run:  python3 tests/test_profile_job.py
"""

import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text  # noqa: E402

from das2.cli import _refresh_baselines_if_stale, build_profiles  # noqa: E402
from das2.config import Config  # noqa: E402
from das2.detect.baseline import score_window  # noqa: E402
from das2.io.store import (  # noqa: E402
    apply_migrations,
    baseline_age_hours,
    history_sensors,
    iter_history,
    load_baselines,
    load_history,
    make_engine,
    save_readings,
)
from das2.models import AnomalyType  # noqa: E402
from das2.profile.build import MIN_DAYS_NOISE, run_profile_job  # noqa: E402

passed = failed = 0

#: 21 days: past DRIFT's 14-day floor and NOISE_BURST's 7.
DAYS = 21

#: 120 s, which is the real feed's median inter-arrival. It matters to one
#: assertion below and not just to realism: NOISE_BURST measures noise from
#: CONSECUTIVE DIFFERENCES, so the sampling step has to be short enough that the
#: daily cycle barely moves between samples. At 120 s the cycle contributes 0.007
#: per step against 0.028 of noise; at 30 min it contributes 0.105 and swamps it,
#: and a real burst becomes invisible. A fixture coarser than the feed would have
#: reported that as a detector bug.
STEP_MIN = 2

#: Deliberately NOT midnight. The hourly run reads the last 28 days from
#: whenever it happens to run, so a window aligned to midnight is the one case
#: production never produces -- and it is the case that hides the partial-day
#: defect that `_whole_days` exists to fix.
START = datetime(2026, 9, 10, 7, 13)
NOW = START + timedelta(days=DAYS)


def check(label: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {label}" + (f"  ({detail})" if detail else ""))
    else:
        failed += 1
        print(f"  FAIL  {label}" + (f"  ({detail})" if detail else ""))


def series(key: str, *, drift_per_day: float = 0.0,
           burst_date: datetime | None = None, seed: int = 0) -> pd.DataFrame:
    """
    A sensor with a real daily cycle, optionally drifting or bursting.

    A daily cycle is the point: it is what makes a 72-hour slope meaningless and
    a 21-day one measurable, and it is what the time-of-day buckets exist to
    model.

    The burst is keyed to a CALENDAR day rather than to a window-relative one,
    because NOISE_BURST's unit is the calendar day. A burst straddling midnight
    is a known limit of that detector, documented on it, and pinned separately
    below rather than smuggled into this fixture.
    """
    rng = np.random.RandomState(seed)
    steps = DAYS * 24 * 60 // STEP_MIN
    stamps = pd.to_datetime(
        [START + timedelta(minutes=i * STEP_MIN) for i in range(steps)])
    hour_of_day = stamps.hour + stamps.minute / 60.0
    day = (stamps.normalize() - pd.Timestamp(START).normalize()).days.to_numpy()

    value = 4.0 + 0.8 * np.sin(2 * np.pi * hour_of_day.to_numpy() / 24.0)
    value = value + drift_per_day * day
    noise = np.full(steps, 0.02)
    if burst_date is not None:
        noise[stamps.normalize() == pd.Timestamp(burst_date).normalize()] = 0.6
    value = value + rng.randn(steps) * noise
    return pd.DataFrame({"sensor_key": key, "ts": stamps, "value": value})


def seed_history(engine, n_sensors: int) -> pd.DataFrame:
    frames = [series(f"quiet{i}", seed=i) for i in range(n_sensors - 2)]
    frames.append(series("drifter", drift_per_day=0.09, seed=101))
    frames.append(series("burster",
                        burst_date=START + timedelta(days=DAYS - 3), seed=202))
    history = pd.concat(frames, ignore_index=True)
    save_readings(engine, history)
    return history


def main() -> int:
    tmp = tempfile.mkdtemp()
    engine = make_engine(f"sqlite:///{tmp}/profile.db")
    apply_migrations(engine)
    config = Config.load()

    print("\nbefore the job has ever run")
    check("the baseline age is unknown, not zero",
          baseline_age_hours(engine) is None,
          "None means 'never run'; 0.0 would mean 'ran just now'")
    check("and the hourly run would score against nothing",
          load_baselines(engine) == {},
          "which is the state the client's deployment was in for weeks")
    result, batches = build_profiles(config, engine)
    check("the job reports no history rather than failing", batches == 0,
          "a fresh install is the expected state, not an error")

    print("\nwith history in the store")
    history = seed_history(engine, 12)
    stored = history_sensors(engine)
    check("the sensors are found with one DISTINCT, not by reading the rows",
          len(stored[0]) == 12 and stored[1] == "das2_reading",
          f"({len(stored[0])} sensors from {stored[1]})")

    print("\n  the read is batched, and the batches cover everything exactly once")
    batch_sizes = []
    seen_keys: list[str] = []
    seen_rows = 0
    for frame, n, of in iter_history(engine, batch_sensors=5):
        batch_sizes.append(frame["sensor_key"].nunique())
        seen_keys += frame["sensor_key"].unique().tolist()
        seen_rows += len(frame)
    check("12 sensors arrive in 3 batches of at most 5", batch_sizes == [5, 5, 2],
          f"({batch_sizes})")
    check("every sensor appears exactly once across the batches",
          sorted(seen_keys) == sorted(history["sensor_key"].unique()),
          "a sensor split across two batches would have its 21-day slope "
          "fitted twice, to half the data each time")
    check("and no reading is lost or repeated", seen_rows == len(history),
          f"({seen_rows} of {len(history)})")

    print("\n  batching does not change the arithmetic")
    # The claim being tested is that splitting by SENSOR is the one axis along
    # which the statistics are unchanged. Compare the batched job against the
    # old whole-fleet read, which is still there for exactly this.
    whole = run_profile_job(load_history(engine))
    batched, batches = build_profiles(config, engine, batch_sensors=5)
    check("the same sensors get baselines",
          set(batched.baselines) == set(whole.baselines),
          f"({len(batched.baselines)} vs {len(whole.baselines)})")
    check("with identical buckets, value for value",
          all(batched.baselines[k].buckets == whole.baselines[k].buckets
              for k in whole.baselines),
          "the bucket medians are what the hourly run scores against")
    check("and the same findings, sensor for sensor",
          {k: [s.type for s in v] for k, v in batched.signals.items()}
          == {k: [s.type for s in v] for k, v in whole.signals.items()},
          f"({sorted(batched.signals)})")
    check("and it really did arrive in several batches", batches == 3,
          f"({batches}) — an identical answer from one batch would prove "
          f"nothing about batching")

    print("\nthe long-horizon detectors fire, which is the whole point")
    found = {k: {s.type for s in v} for k, v in batched.signals.items()}
    check("the drifting sensor is reported as DRIFT",
          AnomalyType.DRIFT in found.get("drifter", set()),
          f"(0.09/day over {DAYS} days against a 1.6-wide daily cycle — "
          f"invisible in any 72-hour window)")
    check("the bursting sensor is reported as NOISE_BURST",
          AnomalyType.NOISE_BURST in found.get("burster", set()),
          "its baseline spread cannot come from the window that holds the burst")
    check("and the burst is not mistaken for drift",
          AnomalyType.DRIFT not in found.get("burster", set()),
          "a noisier day moves no level; reporting it as drift would send "
          "someone to recalibrate a loose connection")
    check("and the quiet sensors are silent",
          not any(k.startswith("quiet") for k in found),
          f"({sorted(found)}) — a job that flags healthy sensors is worse "
          f"than one that never ran")
    check("short history is counted, not silently skipped",
          isinstance(batched.summary()["skipped_short_history"], int))
    check(f"NOISE_BURST's own floor is {MIN_DAYS_NOISE} days", MIN_DAYS_NOISE == 7)

    print("\nan older readings table is used when it knows more")
    # Why this is not just a nicety. `das2_reading` holds whatever the runs so
    # far have written, so ONE run makes it non-empty -- and the old precedence
    # was "das2_reading first, fallback only if empty", which meant a database
    # already holding months of history in its v1 table was never consulted
    # again after the first run. DRIFT needs 14 days, so it waited a fortnight
    # on data that was sitting there, with nothing in the output to say why.
    from das2.io.store import history_span_days

    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE IF NOT EXISTS v1_data (sensor_key "
                          "VARCHAR(200), ts TIMESTAMP, value DOUBLE PRECISION)"))
    older = pd.concat([series(f"quiet{i}", seed=500 + i) for i in range(3)],
                      ignore_index=True)
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO v1_data (sensor_key, ts, value) "
                          "VALUES (:sensor_key, :ts, :value)"),
                     older.assign(ts=older["ts"].dt.to_pydatetime())
                          .to_dict("records"))

    # Make das2_reading the SHORTER source, which is the real situation after a
    # fresh install has done a handful of runs.
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM das2_reading WHERE ts < :cut"),
                     {"cut": NOW - timedelta(days=3)})
    short = history_span_days(engine, "das2_reading", 28)
    long_ = history_span_days(engine, "v1_data", 28)
    check("das2_reading is now the shorter source",
          short < long_, f"({short:.0f}d vs {long_:.0f}d)")

    keys, table = history_sensors(engine, fallback_table="v1_data")
    check("so the longer one is read instead", table == "v1_data",
          f"({table}) — the old rule took das2_reading whenever it had a row, "
          f"and DRIFT then waited 14 days for history already in the database")
    keys, table = history_sensors(engine, fallback_table="")
    check("and a fallback nobody configured is still never touched",
          table == "das2_reading",
          "it is opt-in: a job that reaches into a table it was not pointed "
          "at is a surprising thing to do to someone's database")
    check("an unreadable fallback does not break the job",
          history_span_days(engine, "no_such_table", 28) == 0.0,
          "a wrong table name costs the longer history, not the run")

    # Put the readings back for the sections below.
    seed_history(engine, 12)

    print("\nthe partial days at the window's edges are excluded")
    # The defect this section exists for. A 28-day window almost never starts at
    # midnight, so the first and last calendar days hold a FRACTION of a day's
    # samples covering one arbitrary phase of the daily cycle. Their median is
    # not an estimate of the same quantity as every other day's, and leaving
    # them in broke both detectors at once, in opposite directions.
    from das2.profile.build import (
        WHOLE_DAY_FRACTION,
        _daily_medians,
        _whole_days,
        detect_drift,
        detect_noise_burst,
    )

    clean = series("edge", seed=77)
    day = clean["ts"].dt.normalize()
    counts = clean.groupby(day).size()
    spans = clean.groupby(day)["ts"].agg(lambda s: s.max() - s.min())
    hours = spans.dt.total_seconds() / 3600.0
    check("the window really does have two partial days",
          hours.iloc[0] < 20 and hours.iloc[-1] < 20,
          f"(first covers {hours.iloc[0]:.1f}h, last {hours.iloc[-1]:.1f}h, "
          f"typical {hours.median():.1f}h — the run starts at {START:%H:%M})")

    days, medians = _daily_medians(clean["ts"], clean["value"].to_numpy())
    check("both are left out of the daily medians",
          len(medians) == len(counts) - 2,
          f"({len(medians)} medians from {len(counts)} calendar days)")
    check("so every median left is comparable with every other",
          float(np.ptp(medians)) < 0.1,
          f"(spread {float(np.ptp(medians)):.4f}; with the partials in, the "
          f"edges sat 0.565 away from a day-to-day spread of 0.025 — 22x, at "
          f"the two ends of the series, where Theil-Sen and Kendall's tau are "
          f"most sensitive)")

    check("a clean sensor is no longer reported as drifting",
          detect_drift("edge", clean) == [],
          "the partial days alone produced DRIFT on sensors with no drift")

    print("\n  judged on time COVERED, not on sample count")
    # The two are unrelated, and a count-based rule confuses them both ways. The
    # first version used 0.6 of the median day's COUNT and kept this very
    # window's first day: 504 samples against a typical 720 passes a count test
    # at 70% while covering only 16.8 of 24 hours. The medians still spanned
    # 0.39 instead of 0.025.
    check("a day can pass on count while failing on coverage",
          counts.iloc[0] / counts.median() > 0.6 and hours.iloc[0] / 24 < 0.8,
          f"({counts.iloc[0]} of {counts.median():.0f} samples = "
          f"{counts.iloc[0] / counts.median():.0%}, but only "
          f"{hours.iloc[0]:.1f}h of 24 — which is why the rule is on span)")

    # A THIN day spans the full 24h with few samples: unbiased, and normal here.
    # 57% of this fleet is report-by-exception, so dropping sparse days would
    # drop most of the fleet.
    sparse = clean[::20]
    grouped = _whole_days(sparse["ts"], sparse["value"].to_numpy())
    check("a sensor reporting 36x less often keeps its days",
          grouped is not None and len(grouped) == len(counts) - 2,
          f"({0 if grouped is None else len(grouped)} days kept from "
          f"{len(counts)}; one twentieth of the samples, same coverage)")

    # The degenerate case the span rule has to survive: once a day, every span
    # zero. Comparing against the TYPICAL span rather than against 24 hours is
    # what keeps these.
    once = clean[clean["ts"].dt.hour.eq(9) & clean["ts"].dt.minute.eq(13)]
    grouped = _whole_days(once["ts"], once["value"].to_numpy())
    check("a once-a-day sensor keeps every day, not none",
          grouped is not None and len(grouped) == len(once),
          f"({0 if grouped is None else len(grouped)} of {len(once)} days; "
          f"against a fixed 24h rule every one would be dropped)")
    check("and the fraction is a share of the typical day, not of 24 hours",
          0.0 < WHOLE_DAY_FRACTION < 1.0, f"({WHOLE_DAY_FRACTION})")

    print("\n  and the known limit is a limit, not a surprise")
    # Documented on the detector: the unit is a calendar day, so a burst that
    # lasts about a day and straddles midnight is split in half and can fall
    # under the multiple. Pinned so it stays a decision rather than becoming a
    # mystery the next time someone reads a missed burst in the field.
    straddle = series("straddle", seed=303)
    mid = pd.Timestamp(START + timedelta(days=DAYS - 4, hours=12))
    window = ((straddle["ts"] >= mid)
              & (straddle["ts"] < mid + pd.Timedelta(hours=24)))
    rng = np.random.RandomState(9)
    straddle.loc[window, "value"] += rng.randn(int(window.sum())) * 0.6
    straddling = detect_noise_burst("straddle", straddle)
    whole_day = series("wholeday", burst_date=START + timedelta(days=DAYS - 4),
                       seed=303)
    check("a burst inside one calendar day is caught",
          len(detect_noise_burst("wholeday", whole_day)) >= 1)
    check("one straddling midnight may not be, as documented",
          isinstance(straddling, list),
          f"({len(straddling)} found; each half carries half the burst diluted "
          f"with half a day of quiet. Accepted: a failing transducer stays "
          f"noisy and is caught on its first whole day)")

    print("\nthe baselines are stored, and stored per batch")
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT COUNT(*) FROM das2_sensor_profile")).scalar()
        sensors = conn.execute(
            text("SELECT COUNT(DISTINCT sensor_key) "
                 "FROM das2_sensor_profile")).scalar()
    check("every sensor has a stored profile", sensors == 12, f"({sensors})")
    check("with a bucket row per time of day", rows >= 12 * 48,
          f"({rows} rows; 48 half-hourly visits map onto 96 buckets)")
    loaded = load_baselines(engine)
    check("and they load back usable", sum(1 for b in loaded.values() if b.usable) == 12,
          f"({sum(1 for b in loaded.values() if b.usable)} of 12 usable)")

    print("\nthe hourly run notices for itself when they are stale")
    age = baseline_age_hours(engine)
    check("the age is readable now the job has run",
          age is not None and age < 1.0, f"({age})")

    # Fresh: must not rebuild. The job costs minutes on the real fleet, and an
    # hourly run that rebuilt every time would spend most of its hour doing it.
    calls = {"n": 0}
    import das2.cli as cli
    real = cli.build_profiles
    cli.build_profiles = lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1),
                                          (None, 0))[1]
    _refresh_baselines_if_stale(config, engine)
    check("a fresh baseline is left alone", calls["n"] == 0,
          f"({age:.2f}h old, under the {config.baseline.max_age_hours}h "
          f"refresh age)")

    # Stale: must rebuild.
    with engine.begin() as conn:
        conn.execute(text("UPDATE das2_sensor_profile SET updated_at = :old"),
                     {"old": datetime.now() - timedelta(hours=40)})
    _refresh_baselines_if_stale(config, engine)
    check("a 40-hour-old baseline is rebuilt", calls["n"] == 1,
          f"(over the {config.baseline.max_age_hours}h refresh age)")

    config.baseline.auto_refresh = False
    _refresh_baselines_if_stale(config, engine)
    check("and an installation with its own cron can turn it off",
          calls["n"] == 1, "auto_refresh=False skips it entirely")
    config.baseline.auto_refresh = True

    # Never fatal: a failure here costs the L2 layer one run, not the alert.
    def explode(*a, **k):
        raise RuntimeError("simulated profile failure")
    cli.build_profiles = explode
    _refresh_baselines_if_stale(config, engine)
    check("a failed rebuild does not take the run down", True,
          "logged at WARNING; the run carries on with whatever is stored")
    cli.build_profiles = real

    print("\nand a stored baseline actually makes the L2 layer fire")
    # This is the assertion that would have caught the original defect. With no
    # stored baseline, `score_window` returns [] for every sensor on every run
    # and the system is silent about exactly the degradation it is now scoped to
    # detect -- with nothing in the output to say so.
    quiet = loaded["quiet0"]
    window_ts = pd.Series([NOW + timedelta(minutes=i * 2) for i in range(120)])
    normal = np.array([
        4.0 + 0.8 * np.sin(2 * np.pi * ((t.hour + t.minute / 60.0) / 24.0))
        for t in window_ts])
    check("a window at its own normal raises nothing",
          score_window(window_ts, normal, quiet, unit="bar") == [],
          "the false-positive guard: a baseline that flags normal is useless")

    departed = normal + 1.6          # twice the daily swing, held for 4 hours
    signals = score_window(window_ts, departed, quiet, unit="bar")
    check("a sustained departure from it IS raised",
          any(s.type is AnomalyType.RESIDUAL_OUTLIER for s in signals),
          f"({[s.type.value for s in signals]})")
    check("with the deviation in engineering units an engineer can check",
          any(s.unit == "bar" and s.magnitude for s in signals),
          f"({[(s.magnitude, s.unit) for s in signals]})")
    check("and the same departure scored against NO baseline is silent",
          score_window(window_ts, departed, None, unit="bar") == [],
          "which is what every run on the client's deployment was doing")

    print(f"\n{passed} passed, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
