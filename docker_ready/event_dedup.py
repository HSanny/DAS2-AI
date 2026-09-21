"""
event_dedup.py
==============

Cross-run event identity for the anomaly pipeline.

The problem
-----------
The detector runs on a schedule over a 72-hour window, so consecutive runs
overlap almost completely -- at 4 runs/day, by 66 of 72 hours. `alert_bot`
sends every history row that has a Plot_Path and no AlertTriggered, and nothing
anywhere relates a row to the row the previous run wrote for the same ongoing
condition. So one sensor fault lasting three days is re-detected by every run
that can still see it and generates up to **12 separate Telegram alerts**.

`cluster_suppression` does not help here: it de-duplicates across *sensors*
within a single run (fan-out from one panel), never across *runs*.

The approach
------------
Identity by *window overlap*, not by start time.

The obvious key -- (sensor, bucketed start time) -- does not survive contact
with this detector. `robust_z` uses a centred rolling window, so each timestamp
is re-scored against different future context on every run, and the reported
First_Anomaly_Time for one physical event drifts between runs. Bucketing a
drifting timestamp produces a different key each run, which is the bug rather
than the fix.

Two detections are instead treated as the same event when they are on the same
sensor and their [first, last] anomaly windows overlap, allowing a tolerance
gap so a brief detection dropout does not split one event in two. The first
detection mints a new event id; later detections inherit it.

This is deliberately independent of the database so it can be tested directly.
`db_writer` supplies prior rows and persists the decision.
"""

from __future__ import annotations

import os
import uuid

import pandas as pd


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        return int(raw) if raw is not None and raw.strip() != "" else default
    except (TypeError, ValueError):
        return default


# Two detections on one sensor separated by less than this are the same event.
# Sized against the run cadence: it must comfortably exceed the gap between
# runs, or an event quiet at the moment one run happens to sample it would be
# re-announced by the next.
EVENT_CONTINUITY_GAP_MIN = _env_int("EVENT_CONTINUITY_GAP_MIN", 180)

# How far back to consider prior rows. Bounds the query; anything older than
# this is treated as a genuinely new event even on the same sensor.
EVENT_LOOKBACK_HOURS = _env_int("EVENT_LOOKBACK_HOURS", 168)  # 7 days

SENSOR_KEYS = ["Equipment", "Description"]


def new_event_id() -> str:
    return uuid.uuid4().hex[:16]


def _windows_continuous(a_start, a_end, b_start, b_end, gap_min: int) -> bool:
    """
    True when [a_start, a_end] and [b_start, b_end] overlap, or are separated
    by less than `gap_min` minutes.

    Rows with unusable timestamps never match -- guessing identity from a
    missing time would merge unrelated events, which is worse than sending a
    duplicate alert.
    """
    if any(pd.isna(x) for x in (a_start, a_end, b_start, b_end)):
        return False
    gap = pd.Timedelta(minutes=gap_min)
    return (pd.Timestamp(a_start) - gap) <= pd.Timestamp(b_end) and \
           (pd.Timestamp(b_start) - gap) <= pd.Timestamp(a_end)


def assign_event_identity(
    new_rows: pd.DataFrame,
    prior_rows: pd.DataFrame,
    *,
    gap_min: int = EVENT_CONTINUITY_GAP_MIN,
    start_col: str = "First_Anomaly_Time",
    end_col: str = "Last_Anomaly_Time",
) -> pd.DataFrame:
    """
    Tag each new detection with an EventKey, and mark it as a continuation when
    it belongs to an event that has already been alerted.

    Parameters
    ----------
    new_rows
        This run's summary rows. Not mutated.
    prior_rows
        Previously persisted rows for the same sensors, with at least
        Equipment, Description, the two time columns, EventKey, and
        AlertTriggered. May be empty.

    Returns
    -------
    A copy of `new_rows` with three added columns:
      EventKey            stable id shared by every detection of one event
      DedupOfEventKey     the prior EventKey this continues, else ""
      AlertSuppressed     True when this event has already been alerted, so the
                          bot should stay quiet

    Note the asymmetry: a continuation of an event that was detected but NOT
    yet alerted (for example it was fan-out suppressed, or delivery failed)
    inherits the EventKey but is NOT suppressed. Otherwise a first alert that
    never actually went out would silence every later detection of it.
    """
    out = new_rows.copy()
    out["EventKey"] = ""
    out["DedupOfEventKey"] = ""
    out["AlertSuppressed"] = False

    if out.empty:
        return out

    for col in (start_col, end_col):
        if col not in out.columns:
            raise KeyError(f"assign_event_identity: required column '{col}' missing")
        out[col] = pd.to_datetime(out[col], errors="coerce")

    have_prior = (
        prior_rows is not None
        and not prior_rows.empty
        and all(c in prior_rows.columns for c in SENSOR_KEYS + [start_col, end_col])
    )
    if have_prior:
        prior = prior_rows.copy()
        for col in (start_col, end_col):
            prior[col] = pd.to_datetime(prior[col], errors="coerce")
        if "EventKey" not in prior.columns:
            prior["EventKey"] = ""
        if "AlertTriggered" not in prior.columns:
            prior["AlertTriggered"] = pd.NA
        # Most recent first, so a continuation attaches to the latest detection
        # of the event rather than an older one.
        prior = prior.sort_values(end_col, ascending=False)
        prior_by_sensor = {k: v for k, v in prior.groupby(SENSOR_KEYS, sort=False)}
    else:
        prior_by_sensor = {}

    # Detections minted during THIS run, so several rows for one sensor in one
    # batch also collapse onto a single event rather than each minting its own.
    minted: dict[tuple, list[dict]] = {}

    for idx, row in out.iterrows():
        sensor = tuple(row[k] for k in SENSOR_KEYS)
        start, end = row[start_col], row[end_col]

        matched_key = ""
        already_alerted = False

        for cand in minted.get(sensor, []):
            if _windows_continuous(start, end, cand["start"], cand["end"], gap_min):
                matched_key = cand["key"]
                already_alerted = cand["alerted"]
                break

        if not matched_key and sensor in prior_by_sensor:
            for _, cand in prior_by_sensor[sensor].iterrows():
                if _windows_continuous(start, end, cand[start_col], cand[end_col], gap_min):
                    matched_key = str(cand.get("EventKey") or "") or new_event_id()
                    # Ask whether the EVENT has ever been alerted, not whether
                    # this particular row was. Continuation rows are written
                    # with AlertTriggered NULL precisely because they were
                    # suppressed, so testing the matched row alone would see
                    # "never alerted" and re-alert on the very next run -- the
                    # bug this module exists to fix, reintroduced one level down.
                    chain = prior[prior["EventKey"] == matched_key]
                    already_alerted = bool(
                        chain["AlertTriggered"].fillna(0).astype(bool).any()
                    ) if not chain.empty else False
                    break

        if matched_key:
            out.at[idx, "EventKey"] = matched_key
            out.at[idx, "DedupOfEventKey"] = matched_key
            out.at[idx, "AlertSuppressed"] = already_alerted
        else:
            matched_key = new_event_id()
            out.at[idx, "EventKey"] = matched_key

        minted.setdefault(sensor, []).append({
            "key": matched_key,
            "start": start,
            "end": end,
            # A row we are about to alert counts as alerted for the rest of this
            # batch, so two overlapping detections in one run alert only once.
            "alerted": already_alerted or not out.at[idx, "AlertSuppressed"],
        })

    return out


def prior_rows_query(lookback_hours: int = EVENT_LOOKBACK_HOURS) -> str:
    """
    SQL for the candidate prior rows. Bounded by time so the scan stays small
    as the history table grows.
    """
    return f"""
        SELECT Equipment, [Description], First_Anomaly_Time, Last_Anomaly_Time,
               EventKey, AlertTriggered
          FROM dbo.abnormal_sensor_history
         WHERE Last_Anomaly_Time >= DATEADD(hour, -{int(lookback_hours)}, SYSUTCDATETIME())
    """
