"""
Tests for das2.io.ingest against REAL historian files (Phase 3).

tests/data/real/ holds genuine exports supplied by the client (HISTORY sampled
down to keep the repo small; HISTCURR and LongLat complete). Testing against
these rather than synthetic files is the point: the previous pipeline's readers
disagree with the live feed in four separate ways, and every one of those was
found by opening a real file, not by reading code.

Run:  python3 tests/test_ingest.py
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from das2.io.ingest import (  # noqa: E402
    DATETIME_FORMAT,
    IngestReport,
    build_sensor_table,
    filename_timestamp,
    load_all,
    make_sensor_key,
    parse_datetime_column,
    read_history_file,
    read_inventory,
    read_longlat,
)

DATA = REPO / "tests" / "data" / "real"
HISTORY = DATA / "hts_2026_09_HISTORY_2026Sep20-210000.csv"
HISTCURR = DATA / "hts_HISTCURR_2026Sep20-220001.csv"
LONGLAT = DATA / "LongLat.csv"


def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
    if not cond:
        raise SystemExit(1)


def main():
    print("real HISTORY: semicolon, US 12-hour timestamps")
    hist = read_history_file(HISTORY)
    check("rows parsed", len(hist) > 2000, f"({len(hist):,})")
    check("columns normalised", list(hist.columns) == ["sensor_key", "ts", "value"])
    check("timestamps all parsed", hist["ts"].notna().all())
    check("values numeric", pd.api.types.is_numeric_dtype(hist["value"]))
    check("timestamps land in the expected hour",
          hist["ts"].dt.date.nunique() == 1 and hist["ts"].min().hour >= 20,
          f"({hist['ts'].min()} -> {hist['ts'].max()})")

    print("\nthe US date format is handled explicitly")
    # 9/5 is ambiguous month/day; 9/20 is not. Both must read month-first.
    s = pd.Series(["9/20/2026 9:00:00 PM", "9/5/2026 1:30:00 AM", "12/31/2026 11:59:59 PM"])
    got = parse_datetime_column(s)
    check("9/20 -> 20 September", got[0] == datetime(2026, 9, 20, 21, 0, 0))
    check("9/5 -> 5 September, not 9 May", got[1] == datetime(2026, 9, 5, 1, 30, 0),
          "(month-first, pinned rather than inferred)")
    check("midnight boundary", got[2] == datetime(2026, 12, 31, 23, 59, 59))
    check("format constant is the US one", DATETIME_FORMAT == "%m/%d/%Y %I:%M:%S %p")
    mixed = parse_datetime_column(pd.Series(["9/20/2026 9:00:00 PM", "garbage", None]))
    check("unparseable rows become NaT rather than raising",
          mixed.notna().sum() == 1 and mixed.isna().sum() == 2)

    print("\nreal HISTCURR: nine semicolon columns including POINTTYPE")
    inv = read_inventory(HISTCURR)
    check("full inventory read", len(inv) > 13000, f"({len(inv):,} points)")
    check("POINTTYPE carried through", "pointtype" in inv.columns)
    check("POINTTYPE actually populated", inv["pointtype"].notna().sum() > 10000)
    check("analog/digital split from RawType",
          set(inv["signal_type"].unique()) == {"Analog", "Digital"})
    analog = (inv["signal_type"] == "Analog").sum()
    check("analog share matches the live fleet", 4500 < analog < 5600, f"({analog:,})")
    check("sensor keys unique", inv["sensor_key"].is_unique,
          "(no Hkey collisions -- verified on the real inventory)")

    print("\none unreadable HISTORY file does not take the run down")
    # The share is written hourly, so the newest file is routinely mid-write
    # and arrives as zero bytes. pandas answers that with "No columns to parse
    # from file", and one such file out of 10,334 aborted the client's entire
    # first run. A monitoring system must not go quiet because a file it does
    # not need yet is still being copied.
    import tempfile as _tf
    from das2.io.ingest import IngestReport, read_history_dir
    hdr = ("ROW_ID;IPADDRESS;DESCRIPTION;TAGNAME;RTUNUMBER;RAWTYPE;"
           "POINTTYPE;DATETIME;CURRVALUE")
    row = "1;10;Some-Sensor;T1;1010;1;22;9/22/2026 1:00:00 PM;4.2"
    with _tf.TemporaryDirectory() as td:
        d = Path(td)
        for h in ("120000", "130000"):
            (d / f"hts_2026_09_HISTORY_2026Sep22-{h}.csv").write_text(
                f"{hdr}\n{row}\n")
        (d / "hts_2026_09_HISTORY_2026Sep22-140000.csv").write_text("")
        rep = IngestReport()
        got = read_history_dir(d, report=rep)
        check("the good files are still read", len(got) == 2, f"({len(got)} rows)")
        check("the empty file is counted, not ignored",
              rep.history_files_skipped == 1 and rep.history_files == 2,
              "(missing data wearing a filename)")
        check("it appears in the run stats",
              rep.as_dict()["history_files_skipped"] == 1)
        check("and is excluded from the window, not treated as present",
              rep.window_end == datetime(2026, 9, 22, 13, 0, 0),
              f"({rep.window_end})")

    with _tf.TemporaryDirectory() as td:
        d = Path(td)
        for h in ("120000", "130000"):
            (d / f"hts_2026_09_HISTORY_2026Sep22-{h}.csv").write_text("")
        try:
            read_history_dir(d)
            broke = ""
        except ValueError as e:
            broke = str(e)
    check("but a wholly unreadable feed still fails loudly",
          "broken feed" in broke,
          "(silently analysing nothing is the failure this prevents)")

    print("\nHISTCURR is an HOURLY export, so the inventory path resolves")
    # The share has never held the old pipeline's single pre-merged
    # histcurr_fujitsu.csv. It exports hts_HISTCURR_2026Sep22-130001 every
    # hour, often with no .csv extension at all, and the deployment config
    # pointed at the dead name -- so the pre-flight reported "inventory file
    # exists: FAIL" while the inventory sat beside it under another name.
    import tempfile
    from das2.io.ingest import resolve_inventory_path
    with tempfile.TemporaryDirectory() as td:
        share = Path(td)
        # Real names from the client's share. Note: no extension.
        for stamp in ("2026Sep21-230001", "2026Sep22-000001", "2026Sep22-130001"):
            (share / f"hts_HISTCURR_{stamp}").write_text("x")
        check("a directory resolves to the newest snapshot",
              resolve_inventory_path(share).name == "hts_HISTCURR_2026Sep22-130001")
        check("a missing .csv extension is not required",
              not resolve_inventory_path(share).suffix,
              "(the real files carry none)")
        check("the stale v1 filename still resolves, with a warning",
              resolve_inventory_path(share / "histcurr_fujitsu.csv").name
              == "hts_HISTCURR_2026Sep22-130001",
              "(an upgrade must not break on a stale config value)")
        exact = share / "hts_HISTCURR_2026Sep22-000001"
        check("an explicit file is honoured, not overridden",
              resolve_inventory_path(exact) == exact)

    with tempfile.TemporaryDirectory() as td:
        share = Path(td)
        # The trap: "Sep" > "Dec" alphabetically, so sorted() picks September.
        for stamp in ("2026Apr02-000000", "2026Sep22-130000", "2026Dec01-120000"):
            (share / f"hts_HISTCURR_{stamp}.csv").write_text("x")
        check("ordered by parsed timestamp, not by filename",
              resolve_inventory_path(share).name == "hts_HISTCURR_2026Dec01-120000.csv",
              "(sorted() would pick Sep over Dec)")

    with tempfile.TemporaryDirectory() as td:
        try:
            resolve_inventory_path(Path(td) / "nothing.csv")
            missing = ""
        except FileNotFoundError as e:
            missing = str(e)
    check("an empty share fails loudly", "No HISTCURR inventory" in missing)
    check("and the error says what it was looking for", "hts_HISTCURR" in missing)

    print("\nthe old comma-separated format is rejected with a useful message")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        bad = Path(td) / "old.csv"
        bad.write_text("TAGNAME,IPADDRESS,ROW_ID,DESCRIPTION,RAWTYPE,RTUNUMBER\n"
                       "T1,1,2,Some-Sensor,1,9\n")
        try:
            read_inventory(bad)
            raised = ""
        except ValueError as e:
            raised = str(e)
    check("a 6-column comma file is refused", "missing" in raised.lower())
    check("the error names the expected format", "POINTTYPE" in raised,
          "(so the next person knows which file to supply)")

    print("\nreal LongLat: Latitude BEFORE Longitude, placeholders removed")
    ll = read_longlat(LONGLAT)
    check("sites loaded", len(ll) > 60, f"({len(ll)} sites)")
    check("read by name, not position -- lat/lon not swapped",
          ll["latitude"].between(1.15, 1.50).all()
          and ll["longitude"].between(103.6, 104.1).all(),
          "(swapped values would land in the Indian Ocean)")
    check("no placeholder sites survive",
          not ll["site"].str.lower().eq("unused").any(),
          "(7 LKeys share one fake coordinate; RTU 0 alone has >1,200 sensors)")
    known = dict(zip(ll["rtu_number"], ll["site"]))
    check("a known RTU resolves to its real site",
          known.get("506") == "Bedok PS", f"({known.get('506')})")
    check("the fake coordinate is gone",
          not ((ll["latitude"] == 1.2575396) & (ll["longitude"] == 103.7847767)).any())

    print("\nplaceholder exclusion is reported, not silent")
    rep = IngestReport()
    read_longlat(LONGLAT, report=rep)
    check("placeholders counted", rep.sites_unused >= 5, f"({rep.sites_unused})")
    check("and explained in the report", any("placeholder" in n for n in rep.notes))

    print("\nfull load: readings joined to sensors, placed on the map")
    readings, sensors, report = load_all(DATA, HISTCURR, LONGLAT)
    check("readings matched to inventory", report.readings_matched > 2000,
          f"({report.readings_matched:,})")
    check("no unmatched readings in this sample", report.readings_unmatched == 0)
    check("coordinate coverage measured, not assumed",
          report.coordinate_coverage_pct > 80,
          f"({report.coordinate_coverage_pct}% of {len(sensors):,} sensors)")
    check("unplaced sensors are counted", report.sensors_without_coords > 0,
          f"({report.sensors_without_coords:,} -- geo-clustering cannot see these)")

    print("\nthe Location column collision is fixed")
    # The old pipeline merged two frames that each had a 'Location' column, so
    # pandas produced Location_x/Location_y and every downstream lookup for
    # 'Location' silently returned "".
    check("site column exists and is populated",
          "site" in sensors.columns and sensors["site"].notna().sum() > 10000,
          f"({sensors['site'].notna().sum():,} sensors have a site)")
    check("no _x/_y suffix columns leaked from the merge",
          not any(c.endswith(("_x", "_y")) for c in sensors.columns),
          f"({[c for c in sensors.columns if c.endswith(('_x','_y'))]})")

    print("\nevery sensor gets a region -- the whole point of the exercise")
    check("region assigned to all rows", sensors["region"].notna().all())
    placed = sensors[sensors["region"] != "Unknown"]
    check("most sensors land in a real region",
          len(placed) / len(sensors) > 0.80,
          f"({100*len(placed)/len(sensors):.0f}%)")
    check("all five regions are represented",
          {"Central", "East", "North", "North-East", "West"} <= set(sensors["region"]),
          f"({sorted(sensors['region'].unique())})")
    check("placement source recorded for every sensor",
          set(sensors["placement_source"]) <= {"coordinates", "site-name", "none"})
    check("coordinates preferred over the name fallback",
          (sensors["placement_source"] == "coordinates").sum()
          > (sensors["placement_source"] == "site-name").sum())

    print("\nfilename timestamps")
    check("HISTORY filename parses",
          filename_timestamp(HISTORY) == datetime(2026, 9, 20, 21, 0, 0))
    check("a non-matching name returns None", filename_timestamp("nope.csv") is None)

    print("\nsensor keys")
    keys = make_sensor_key(pd.Series([1, 12]), pd.Series([23, 3]))
    check("a separator removes the concatenation ambiguity",
          keys.iloc[0] != keys.iloc[1], f"({keys.tolist()})")

    print("\nbuild_sensor_table without coordinates still works")
    bare = build_sensor_table(inv.head(200), None)
    check("no LongLat -> still classified and region-assigned",
          bare["equipment"].notna().all() and bare["region"].notna().all(),
          "(falls back to the site-name hint)")

    print("\nAll ingest tests passed.")


if __name__ == "__main__":
    main()
