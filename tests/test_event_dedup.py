"""
Tests for cross-run event identity (Phase 0.6).

The behaviour under test is the one that produces the loudest complaint about
the current system: a 72h detection window re-run every 6h means consecutive
runs overlap by 66 hours, so one ongoing fault is re-detected by ~12 runs and
alerts ~12 times.

Run:  python3 tests/test_event_dedup.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docker_ready"))
from event_dedup import assign_event_identity  # noqa: E402


def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
    if not cond:
        raise SystemExit(1)


def rows(*specs):
    """specs: (equipment, description, start, end)"""
    return pd.DataFrame(
        [{"Equipment": e, "Description": d,
          "First_Anomaly_Time": pd.Timestamp(s), "Last_Anomaly_Time": pd.Timestamp(t)}
         for e, d, s, t in specs]
    )


def prior(*specs):
    """specs: (equipment, description, start, end, event_key, alert_triggered)"""
    return pd.DataFrame(
        [{"Equipment": e, "Description": d,
          "First_Anomaly_Time": pd.Timestamp(s), "Last_Anomaly_Time": pd.Timestamp(t),
          "EventKey": k, "AlertTriggered": a}
         for e, d, s, t, k, a in specs]
    )


SENSOR = ("Flowrate", "BedokPS-Delivery-Flow")
OTHER = ("Pressure", "Kranji2PS-Mains-PRESSURE")


def main():
    print("the 12-alert scenario: one 3-day fault seen by 12 overlapping runs")
    # A fault starting 2026-03-01 06:00 and still running. Each run sees it,
    # but the reported start drifts because robust_z is centred.
    alerts_sent = 0
    prior_df = pd.DataFrame()
    for run in range(12):
        run_time = pd.Timestamp("2026-03-01 12:00") + pd.Timedelta(hours=6 * run)
        drift = pd.Timedelta(minutes=7 * (run % 3))  # start-time jitter between runs
        new = rows((*SENSOR, pd.Timestamp("2026-03-01 06:00") + drift, run_time))
        tagged = assign_event_identity(new, prior_df)
        if not bool(tagged.loc[0, "AlertSuppressed"]):
            alerts_sent += 1
        prior_df = pd.concat([prior_df, prior(
            (*SENSOR, tagged.loc[0, "First_Anomaly_Time"], tagged.loc[0, "Last_Anomaly_Time"],
             tagged.loc[0, "EventKey"], 1 if not tagged.loc[0, "AlertSuppressed"] else None)
        )], ignore_index=True)

    check("one ongoing fault alerts exactly once across 12 runs",
          alerts_sent == 1, f"(sent {alerts_sent}, was 12 before)")
    check("all 12 detections share one EventKey",
          prior_df["EventKey"].nunique() == 1, f"({prior_df['EventKey'].nunique()} keys)")

    print("\na genuinely new event still alerts")
    old = prior((*SENSOR, "2026-03-01 06:00", "2026-03-01 10:00", "evt-old", 1))
    fresh = rows((*SENSOR, "2026-03-05 02:00", "2026-03-05 04:00"))
    t = assign_event_identity(fresh, old)
    check("separated by days -> not suppressed", not bool(t.loc[0, "AlertSuppressed"]))
    check("separated by days -> new EventKey", t.loc[0, "EventKey"] != "evt-old")

    print("\nevent continuity tolerates a short detection dropout")
    old = prior((*SENSOR, "2026-03-01 06:00", "2026-03-01 10:00", "evt-1", 1))
    resumed = rows((*SENSOR, "2026-03-01 11:00", "2026-03-01 14:00"))  # 1h gap
    t = assign_event_identity(resumed, old)
    check("1h gap treated as the same event", t.loc[0, "EventKey"] == "evt-1")
    check("and therefore suppressed", bool(t.loc[0, "AlertSuppressed"]))

    print("\ndifferent sensors never share an event")
    old = prior((*SENSOR, "2026-03-01 06:00", "2026-03-01 10:00", "evt-1", 1))
    other = rows((*OTHER, "2026-03-01 07:00", "2026-03-01 09:00"))
    t = assign_event_identity(other, old)
    check("other sensor alerts independently", not bool(t.loc[0, "AlertSuppressed"]))
    check("other sensor gets its own key", t.loc[0, "EventKey"] != "evt-1")

    print("\nan event detected but never alerted is not silenced")
    # e.g. it was fan-out suppressed, or Telegram delivery failed.
    unsent = prior((*SENSOR, "2026-03-01 06:00", "2026-03-01 10:00", "evt-1", None))
    follow = rows((*SENSOR, "2026-03-01 08:00", "2026-03-01 12:00"))
    t = assign_event_identity(follow, unsent)
    check("inherits the event identity", t.loc[0, "EventKey"] == "evt-1")
    check("but still alerts, since the first one never went out",
          not bool(t.loc[0, "AlertSuppressed"]))

    print("\nduplicates within a single run collapse too")
    batch = rows(
        (*SENSOR, "2026-03-01 06:00", "2026-03-01 09:00"),
        (*SENSOR, "2026-03-01 08:30", "2026-03-01 11:00"),   # overlaps the first
        (*OTHER, "2026-03-01 06:00", "2026-03-01 09:00"),
    )
    t = assign_event_identity(batch, pd.DataFrame())
    check("overlapping rows in one batch share a key",
          t.loc[0, "EventKey"] == t.loc[1, "EventKey"])
    check("the second is suppressed", bool(t.loc[1, "AlertSuppressed"]))
    check("the first still alerts", not bool(t.loc[0, "AlertSuppressed"]))
    check("the unrelated sensor alerts", not bool(t.loc[2, "AlertSuppressed"]))

    print("\nunusable timestamps never merge unrelated events")
    bad = rows((*SENSOR, pd.NaT, pd.NaT))
    old = prior((*SENSOR, "2026-03-01 06:00", "2026-03-01 10:00", "evt-1", 1))
    t = assign_event_identity(bad, old)
    check("NaT window does not match anything", t.loc[0, "EventKey"] != "evt-1")
    check("NaT window is not suppressed", not bool(t.loc[0, "AlertSuppressed"]))

    print("\nedge cases")
    t = assign_event_identity(pd.DataFrame(columns=["Equipment", "Description",
                                                    "First_Anomaly_Time", "Last_Anomaly_Time"]),
                              pd.DataFrame())
    check("empty batch returns empty with the new columns",
          t.empty and {"EventKey", "AlertSuppressed"} <= set(t.columns))

    t = assign_event_identity(rows((*SENSOR, "2026-03-01 06:00", "2026-03-01 10:00")),
                              pd.DataFrame())
    check("no prior history -> alerts", not bool(t.loc[0, "AlertSuppressed"]))
    check("input frame is not mutated", "EventKey" not in rows(
        (*SENSOR, "2026-03-01 06:00", "2026-03-01 10:00")).columns)

    print("\nAll event-dedup tests passed.")


if __name__ == "__main__":
    main()
