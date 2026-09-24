#!/usr/bin/env python3
"""
"Would the statistics we already run have found this?"

The client's own words for what the report has to let him verify: *"then
verify oh there really is something that is not yet discovered by already
existing stats calculation he put in"*, that calculation being *"the median,
std and all that"*.

Answering it is easy to do dishonestly. A comparison against someone else's
method is an argument for replacing it, so the tests here are weighted towards
the ways this could flatter itself:

* **It must report when their check WINS.** A single large spike is 25σ and
  they find it. If this module only ever printed "your check missed it", the
  first time an operator noticed otherwise the whole page would stop being
  believed.
* **A crossing is not automatically a find.** "Beyond 3σ" describes ONE sample.
  A 900-sample window touches 3σ 91% of the time on pure noise, so calling
  every crossing a detection would credit their check with thousands of
  catches a run and understate the gap in the opposite direction. Every
  crossing is scored against what noise produces at that sample count.
* **A constant sensor is `undefined`, not `missed`.** σ is 0, the band has no
  width, and 57% of this fleet never moves. Reporting that as a miss would be
  counting a question the check cannot be asked as a question it got wrong.
* **The masking claim has to be falsifiable.** It is only made when removing
  the event's own samples from the baseline moves the sensor from inside the
  band to outside it. A score that merely rises is not masking.

Run:  python3 tests/test_conventional.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from das2.detect import conventional as cv                # noqa: E402

passed = failed = 0
N = 900
TS = pd.Series(pd.date_range("2026-09-20", periods=N, freq="120s"))


def check(label: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {label}" + (f"  ({detail})" if detail else ""))
    else:
        failed += 1
        print(f"  FAIL  {label}" + (f"  ({detail})" if detail else ""))


def noise(seed: int = 3, sd: float = 0.02) -> np.ndarray:
    return 3.0 + np.random.default_rng(seed).normal(0, sd, N)


def main() -> int:
    print("\nwhen their check works, this says so")
    spike = noise()
    spike[400] += 1.0
    hit = cv.evaluate("spike", TS, spike, spans=[(TS[400], TS[400])])
    check("a single large spike is found by mean ± 3σ",
          hit.verdict == "alarmed", f"{hit.peak_z:.1f}σ")
    check("and the report says it is their find, not ours",
          "your check finds too" in hit.why())
    check("the odds are stated in words, not in floating point",
          "e-" not in hit.why() and len(hit.why()) < 200,
          "p=7e-141 spelled out is a 140-digit number mid-sentence")

    print("\na crossing is not a find")
    clean = cv.evaluate("clean", TS, noise())
    check("pure noise crosses 3σ somewhere in 900 samples",
          clean.crossed, f"peak {clean.peak_z:.2f}σ")
    check("but it is reported as chance, not as a detection",
          clean.verdict == "chance" and not clean.alarmed,
          f"p={clean.p_value:.2f} — a coin toss")
    check("the probability is the standard multiple-comparison one",
          abs(cv.peak_p_value(3.0, 900) - 0.912) < 0.01,
          f"P(noise touches 3σ in 900 samples) = "
          f"{cv.peak_p_value(3.0, 900):.3f}")
    check("a single sample at 3σ would be a finding; the largest of 900 is not",
          cv.peak_p_value(3.0, 1) < 0.01 < cv.peak_p_value(3.0, 900),
          "this is the whole difference between a check that works and one "
          "that reports most of the fleet every run")
    check("and the maths stays honest at the extremes",
          0.0 < cv.peak_p_value(25.0, 900) < 1e-100,
          "the direct form underflows to exactly 0 and loses the difference "
          "between 'very unlikely' and 'impossible'")

    print("\nthe event hides in the σ that should reveal it")
    # The regional-event shape: a sustained step over a third of the window.
    step = noise()
    step[300:600] += 0.30
    held = cv.evaluate("step", TS, step, spans=[(TS[300], TS[599])])
    check("a sustained step does NOT reach 3σ", held.verdict == "missed",
          f"{held.peak_z:.2f}σ over the whole window")
    check("against the hours before it, the same sensor is far outside",
          held.peak_z_excluded is not None and held.peak_z_excluded > 10,
          f"{held.peak_z_excluded:.1f}σ")
    check("so it is reported as masked", held.masked,
          "σ is computed from the window that contains the event; a sustained "
          "excursion inflates its own denominator")
    check("the sentence gives both numbers, so it can be checked by hand",
          "1.8σ" in held.why() and "17.7σ" in held.why(), held.why())

    print("\nand the masking claim cannot be made loosely")
    quiet = noise()
    quiet[500:520] += 0.01
    weak = cv.evaluate("quiet", TS, quiet, spans=[(TS[500], TS[519])])
    check("a score that rises but stays inside the band is not masking",
          not weak.masked,
          "otherwise every sensor in every incident would carry the claim")

    print("\na sensor that never moves has no answer to give")
    flat = cv.evaluate("flat", TS, np.full(N, 3.0))
    check("σ of 0 is undefined, not passed", flat.verdict == "undefined")
    check("it is not counted as a miss either", not flat.alarmed
          and not flat.crossed,
          "57% of this fleet never moves; counting them as misses would be "
          "counting a question the check cannot be asked")
    check("the sentence explains the zero-width band",
          "zero-width" in flat.why())

    print("\ntoo little data is no answer at all")
    check("a handful of samples returns nothing",
          cv.evaluate("short", TS[:8], [1.0] * 8) is None,
          "a standard deviation from 8 points is noise about noise")

    print("\nper incident")
    members = [NS(sensor=NS(sensor_key=f"S{i}", description=f"Site{i}-Pressure",
                            equipment="Pressure", site=f"Site {i}"))
               for i in range(4)]
    incident = NS(cluster=NS(members=members, sites={f"Site {i}" for i in range(4)},
                             equipment_types={"Pressure", "Flowrate"},
                             start=TS[300], end=TS[599]))
    checks = {f"S{i}": cv.evaluate(f"S{i}", TS, step, spans=[(TS[300], TS[599])])
              for i in range(4)}
    panel = cv.for_incident(incident, checks)
    check("the headline answers the client's question first",
          panel["headline"].startswith("No sensor here reaches 3σ"),
          panel["headline"])
    check("every member is accounted for", panel["of"] == 4)
    check("and the reason we saw it names a mechanism",
          "moving together" in panel["found_because"],
          panel["found_because"])
    check("the sensors are listed worst-first, so the table can be truncated",
          [r["peak_z"] for r in panel["sensors"]]
          == sorted((r["peak_z"] for r in panel["sensors"]), reverse=True))

    hits = {f"S{i}": hit for i in range(4)}
    panel = cv.for_incident(incident, hits)
    check("and when their check would have caught it, that is the headline",
          panel["headline"].startswith("Your current check finds this too"),
          panel["headline"])

    print("\nacross the run")
    fleet = {f"N{i}": cv.evaluate(f"N{i}", TS, noise(seed=i)) for i in range(20)}
    summary = cv.summarise(fleet)
    check("healthy sensors cross the line in bulk",
          summary["crossed"] >= 15, f"{summary['crossed']} of 20 healthy")
    check("and are reported as explained by noise",
          summary["explained_by_noise"] >= summary["crossed"] - 2,
          f"{summary['explained_by_noise']} of {summary['crossed']} crossings "
          f"are what noise produces anyway")
    check("the expected sample count is carried next to the observed one",
          summary["expected_crossing_samples"] > 0
          and "crossing_samples" in summary,
          f"{summary['crossing_samples']} observed vs "
          f"{summary['expected_crossing_samples']} expected")

    print("\nit is a comparison, never a detector")
    check("it emits no Signal and imports no detector type",
          not hasattr(cv, "Signal")
          and "from das2.models" not in Path(cv.__file__).read_text(),
          "nothing downstream can consult it, by construction")
    check("and it holds no threshold anything else reads",
          not any(name.endswith("_THRESH") for name in dir(cv)))

    print(f"\n{passed} passed, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
