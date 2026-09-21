"""
Tests for tools/make_fixtures.py (Phase 2).

The fixture is the ground truth everything later is measured against, so it has
to be checked itself. Two things matter:

  1. It is readable by the REAL Fujitsu parsing rules -- filename regex,
     timestamp format, separators, column names -- or later phases would be
     tested against a format the plant does not produce.

  2. It reproduces the awkward properties of the real feed. Generating clean,
     clock-aligned data would make every later phase look correct while hiding
     the exact problems that motivated the rewrite: report-by-exception timing,
     an 8x spread in reporting rate, duplicate and out-of-order timestamps,
     idle meters sitting at zero, and RTU-level coordinates.

Run:  python3 tests/test_fixtures.py
"""

import csv
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Verbatim from docker_ready/fujitsu_data_pipeline.py -- if the fixture stops
# matching these, it has stopped resembling the real export.
HISTORY_PATTERN = r"hts_\d{4}_\d{2}_HISTORY_.*\.csv$"
TS_REGEX = re.compile(r"(\d{4}[A-Za-z]{3}\d{2}-\d{6})")
TS_FORMAT = "%Y%b%d-%H%M%S"


def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
    if not cond:
        raise SystemExit(1)


def generate(out: Path, extra=()):
    cmd = [sys.executable, str(REPO / "tools" / "make_fixtures.py"),
           "--out", str(out), "--hours", "72", "--end", "2026-03-04T00:00:00",
           "--seed", "42", *extra]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        raise SystemExit(1)
    return json.loads((out / "manifest.json").read_text())


def read_history(out: Path):
    """Parse every HISTORY file the way the real pipeline does."""
    rows = []
    for f in sorted((out / "HISTORY").iterdir()):
        with open(f, encoding="utf-8") as fh:
            for rec in csv.DictReader(fh, delimiter=";"):
                rows.append(rec)
    return rows


def main():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        manifest = generate(out)

        print("readable by the real Fujitsu parsing rules")
        files = sorted((out / "HISTORY").iterdir())
        check("72 hourly files", len(files) == 72, f"({len(files)})")
        check("all match the v1 discovery regex",
              all(re.match(HISTORY_PATTERN, f.name) for f in files))
        parsed = [datetime.strptime(TS_REGEX.search(f.name).group(1), TS_FORMAT)
                  for f in files]
        check("all filename timestamps parse", len(parsed) == 72)
        check("one file per hour, contiguous",
              all((parsed[i + 1] - parsed[i]).total_seconds() == 3600
                  for i in range(len(parsed) - 1)))

        rows = read_history(out)
        check("HISTORY is ';'-separated with the real column names",
              set(rows[0].keys()) == {"IPADDRESS", "ROW_ID", "DATETIME", "CURRVALUE"},
              f"({sorted(rows[0].keys())})")
        check("readings present", len(rows) > 50_000, f"({len(rows):,})")

        with open(out / "HISTCURR" / "histcurr_fujitsu.csv", encoding="utf-8") as fh:
            inv = list(csv.DictReader(fh))     # comma-separated, unlike HISTORY
        check("HISTCURR uses the real inventory columns",
              {"TAGNAME", "IPADDRESS", "ROW_ID", "DESCRIPTION", "RAWTYPE",
               "RTUNUMBER"} <= set(inv[0].keys()))
        check("HISTCURR carries DATETIME, which v1 requires",
              "DATETIME" in inv[0],
              "(process_histcurr fails with KeyError('DATETIME') without it)")

        check("LongLat is also written where v1 looks for it",
              (out / "processed" / "LongLat.csv").exists(),
              "(data_preprocessing.py reads processed/LongLat.csv, not the root copy)")
        check("RawType marks analog vs digital as the real feed does",
              {r["RAWTYPE"] for r in inv} <= {"1", "3", "5"})

        with open(out / "LongLat.csv", encoding="utf-8") as fh:
            longlat = list(csv.DictReader(fh))
        check("LongLat keyed on LKey, as the RTUNumber join expects",
              set(longlat[0].keys()) == {"Longitude", "Latitude", "Location", "LKey"})

        # ------------------------------------------------------------------ #
        print("\nreproduces report-by-exception timing")
        seconds = Counter(datetime.fromisoformat(r["DATETIME"]).second for r in rows)
        check("timestamps are spread across the whole minute, not clock-aligned",
              len(seconds) > 50, f"({len(seconds)} distinct second-of-minute values)")
        check("no single second dominates",
              max(seconds.values()) / len(rows) < 0.10,
              f"(max share {100*max(seconds.values())/len(rows):.1f}%)")

        by_sensor = defaultdict(list)
        for r in rows:
            by_sensor[(r["IPADDRESS"], r["ROW_ID"])].append(
                datetime.fromisoformat(r["DATETIME"]))
        rates = {}
        for key, ts in by_sensor.items():
            ts.sort()
            gaps = [(b - a).total_seconds() for a, b in zip(ts, ts[1:]) if (b - a).total_seconds() > 0]
            if gaps:
                rates[key] = sorted(gaps)[len(gaps) // 2]
        spread = max(rates.values()) / min(rates.values())
        check("an 8x-plus spread in reporting rate, as in the real fleet",
              spread >= 5.0,
              f"({min(rates.values()):.0f}s to {max(rates.values()):.0f}s = {spread:.0f}x)")

        print("\nreproduces the timestamp defects that break naive dv/dt")
        dup = sum(1 for ts in by_sensor.values()
                  if len(ts) != len(set(ts)))
        check("duplicate timestamps present", dup > 0,
              f"({dup} sensors have one - dt==0 would give an infinite rate)")
        unsorted_sensors = 0
        for key, ts in by_sensor.items():
            raw = [datetime.fromisoformat(r["DATETIME"])
                   for r in rows if (r["IPADDRESS"], r["ROW_ID"]) == key]
            if any(b < a for a, b in zip(raw, raw[1:])):
                unsorted_sensors += 1
        check("out-of-order timestamps present", unsorted_sensors > 0,
              f"({unsorted_sensors} sensors - dt<0)")

        print("\nreproduces the shapes that broke v1")
        idle_row = next(r for r in inv if "THOMSON" in r["DESCRIPTION"])
        vals = [float(r["CURRVALUE"]) for r in rows
                if r["IPADDRESS"] == idle_row["IPADDRESS"]
                and r["ROW_ID"] == idle_row["ROW_ID"]]
        negative = sum(1 for v in vals if v < 0)
        check("idle flowmeter sits at zero and reads negative about half the time",
              0.25 < negative / len(vals) < 0.75,
              f"({100*negative/len(vals):.0f}% negative - a raw `value<0` test "
              f"would flag all of them)")

        rtus = {r["RTUNUMBER"] for r in inv}
        check("dirty RTU numbers present", "-1" in rtus,
              "(coordinate coverage must be exercised, not assumed)")
        lkeys = {r["LKey"] for r in longlat}
        check("the dirty RTU has no coordinates", "-1" not in lkeys)

        per_rtu = defaultdict(set)
        for r in inv:
            per_rtu[r["RTUNUMBER"]].add(r["DESCRIPTION"])
        shared = [k for k, v in per_rtu.items() if len(v) > 1]
        check("several sensors share one RTU", len(shared) >= 3,
              "(so they will share identical coordinates - site-level resolution)")

        # ------------------------------------------------------------------ #
        print("\nmanifest is usable ground truth")
        faults = Counter(i["fault"] for i in manifest["injections"])
        for expected in ("FLATLINE", "STALE", "RANGE_VIOLATION", "SPIKE", "DRIFT",
                         "QUANTISATION_COLLAPSE", "NOISE_BURST", "SHORT_CYCLING"):
            check(f"{expected} injected", faults.get(expected, 0) >= 1)
        check("REGIONAL_EVENT spans 4 sensors", faults.get("REGIONAL_EVENT") == 4)
        check("TELEMETRY_FANOUT spans 3 sensors", faults.get("TELEMETRY_FANOUT") == 3)

        regional = [i for i in manifest["injections"] if i["fault"] == "REGIONAL_EVENT"]
        check("regional event covers 3 distinct sites",
              len({i["site"] for i in regional}) == 3,
              "(name clustering cannot link these - the case v1 is blind to)")
        check("regional event covers 2 equipment types",
              len({i["equipment"] for i in regional}) == 2)
        check("regional event members overlap in time",
              len({i["start"] for i in regional}) == 1)

        fanout = [i for i in manifest["injections"] if i["fault"] == "TELEMETRY_FANOUT"]
        check("fan-out is confined to ONE site",
              len({i["site"] for i in fanout}) == 1,
              "(opposite verdict to the regional event: suppress, do not dispatch)")

        check("clean controls exist", manifest["counts"]["controls"] >= 4,
              "(a detector firing on these is producing false positives)")
        control_keys = {s["description"] for s in manifest["sensors"] if s["is_control"]}
        injected_keys = {i["description"] for i in manifest["injections"]}
        check("controls and injected sensors are disjoint",
              not (control_keys & injected_keys))

        check("rain gauge present for the weather path",
              any(s["equipment"] == "Rainfall" for s in manifest["sensors"]))
        check("digital pump present", any(s["signal_type"] == "Digital"
                                          for s in manifest["sensors"]))
        check("expected regions recorded for the placement test",
              {s["expected_region"] for s in manifest["sensors"]} >=
              {"East", "North", "West", "Central", "North-East"})

        print("\ndeterministic")
        with tempfile.TemporaryDirectory() as td2:
            m2 = generate(Path(td2))
            check("same seed reproduces the same reading count",
                  m2["counts"]["readings"] == manifest["counts"]["readings"])
            check("same seed reproduces the same injections",
                  [i["description"] for i in m2["injections"]] ==
                  [i["description"] for i in manifest["injections"]])

        print("\n--clean produces a false-positive control corpus")
        with tempfile.TemporaryDirectory() as td3:
            clean = generate(Path(td3), extra=("--clean",))
            check("nothing is injected", clean["counts"]["injections"] == 0,
                  "(measures the false-alarm rate against a real null)")
            check("but the fleet is unchanged",
                  clean["counts"]["sensors"] == manifest["counts"]["sensors"])

    print("\nAll fixture tests passed.")


if __name__ == "__main__":
    main()
