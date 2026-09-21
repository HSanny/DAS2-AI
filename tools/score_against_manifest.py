#!/usr/bin/env python3
"""
score_against_manifest.py
=========================

Score a detector's output against the fixture manifest's ground truth.

This is the measurement the whole rewrite is accountable to. Without labelled
data there is no way to say whether a change helped, and "agreement with the
previous detector" is worthless here because the incumbent is a fixed-rate
top-10 ranker rather than a detector -- it emits about ten sensors per run
whether the network is healthy or on fire, so agreeing with it measures
agreement with a rank cut.

Injected faults invert that. The fixture states what is wrong before anything
runs, so recall and false-alarm rate are both directly computable.

Usage
-----
    # v1 output
    python3 tools/score_against_manifest.py \\
        --manifest fixtures/manifest.json \\
        --detected  run/output_csv/abnormal_sensor.csv \\
        --inventory run/processed/dim.csv

    # das2 output (same flags; any CSV with a Description column works)
    python3 tools/score_against_manifest.py -m fixtures/manifest.json -d out/incidents.csv

Exit code is 0 always: this reports, it does not gate. Ship gates belong in the
shadow harness, where they can be compared between detector versions.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

#: Equipment classes the v1 detector drops before analysis. A fault on a sensor
#: in one of these was never even looked at, which is a different failure from
#: looking and missing -- and the distinction matters, because the fix is a
#: better classifier rather than a better detector.
V1_SKIPPED_EQUIPMENT = {"Others", "Digital Signal"}


def load_detected(path: Path) -> set[str]:
    """Sensor descriptions a detector reported. Tolerates either schema."""
    if not path.exists():
        return set()
    out: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            for key in ("Description", "description", "sensor", "sensor_key"):
                if row.get(key):
                    out.add(row[key])
                    break
    return out


def load_inventory(path: Path | None) -> dict[str, str]:
    """description -> equipment class, for reporting what was never analysed."""
    if not path or not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        return {r["Description"]: r.get("Equipment", "?")
                for r in csv.DictReader(fh) if r.get("Description")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--manifest", required=True, type=Path)
    ap.add_argument("-d", "--detected", required=True, type=Path)
    ap.add_argument("-i", "--inventory", type=Path, default=None,
                    help="processed/dim.csv, to report sensors never analysed")
    ap.add_argument("--label", default="detector", help="name for the report header")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    detected = load_detected(args.detected)
    inventory = load_inventory(args.inventory)

    injected = {i["description"]: i for i in manifest["injections"]}
    controls = {s["description"] for s in manifest["sensors"] if s["is_control"]}

    print(f"\n=== {args.label} vs ground truth ===")
    print(f"fixture : {args.manifest}")
    print(f"output  : {args.detected}\n")

    header = f"{'INJECTED FAULT':<24} {'SENSOR':<42} {'CLASS':<16} {'SEEN':<6} DETECTED"
    print(header)
    print("-" * len(header))

    hits = 0
    never_analysed = 0
    by_fault: dict[str, list[bool]] = defaultdict(list)

    for desc, inj in sorted(injected.items(), key=lambda kv: (kv[1]["fault"], kv[0])):
        equipment = inventory.get(desc, "?")
        analysed = equipment not in V1_SKIPPED_EQUIPMENT if inventory else True
        found = desc in detected
        hits += found
        never_analysed += (not analysed)
        by_fault[inj["fault"]].append(found)
        print(f"{inj['fault']:<24} {desc[:42]:<42} {equipment[:16]:<16} "
              f"{'yes' if analysed else 'NO':<6} {'YES' if found else 'no'}")

    print("-" * len(header))
    total = len(injected)
    recall = 100.0 * hits / total if total else 0.0
    print(f"injected        : {total}")
    print(f"detected        : {hits}   ({recall:.0f}% recall)")
    print(f"missed          : {total - hits}")
    if inventory:
        print(f"never analysed  : {never_analysed}   "
              f"(classified into a skipped bucket - a classifier problem, "
              f"not a detector one)")

    print("\nrecall by fault type")
    for fault, results in sorted(by_fault.items()):
        got, n = sum(results), len(results)
        print(f"  {fault:<24} {got}/{n}")

    false_positives = detected & controls
    print(f"\nfalse positives on clean controls: {len(false_positives)}"
          f"   (of {len(controls)} controls)")
    for fp in sorted(false_positives):
        print(f"  {fp}")

    unexpected = detected - set(injected) - controls
    if unexpected:
        print(f"\nreported but neither injected nor a control: {len(unexpected)}")
        for u in sorted(unexpected):
            print(f"  {u}")

    print("\nA detector is only trustworthy when BOTH numbers are good: high "
          "recall\nwith many false positives just moves the noise, and zero "
          "false positives\nwith low recall is a detector that mostly says "
          "nothing.\n")


if __name__ == "__main__":
    main()
