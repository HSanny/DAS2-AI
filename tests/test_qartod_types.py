#!/usr/bin/env python3
"""
The taxonomy: what we inherited, what we invented, and what we split.

Two changes are pinned here.

**Naming.** A finding an operator can look up in a published manual is worth
more than one that exists only in our source. IOOS QARTOD's *Manual for
Real-Time Quality Control of Water Level Data* is the operational standard for
exactly this estate, so where QARTOD has a name for a thing we now use it.
`DITHERING_DEAD` was an invented name for QARTOD Test 10, *Attenuated Signal*,
and our detector -- a rolling range below a ceiling -- was already the
`check_type="range"` form of that test. The rename cost nothing and bought a
citation.

Just as important is what the mapping does NOT claim. `QUANTISATION_COLLAPSE`,
`REVERSE_FLOW` and the rest have no QARTOD equivalent, and the table says so
with an explicit `None` rather than by omission, so nothing we made up can be
mistaken for something we inherited.

**INSTRUMENT_OFFSET.** A post-maintenance recalibration and a genuine
water-level change are the same signal. Every published taxonomy carries
offset as a sensor fault -- Ni et al. (2009) *ACM TOSN* 5(3), Sharma et al.
(2010) *ACM TOSN* 6(3), Leigh et al. (2019) *STOTEN* 664 -- and we had no way
to express it, so every recalibration bump was routed to "the water moved, go
look". The discriminator is ours and unvalidated; these tests pin its
behaviour and, more importantly, pin that it stays CONSERVATIVE -- when the
evidence is thin it must fall back to LEVEL_SHIFT, because over-calling an
instrument fault means ignoring a real event.

Run:  python3 tests/test_qartod_types.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from das2.detect.changepoint import (  # noqa: E402
    OFFSET_MIN_HOLD_S,
    detect_level_shift,
)
from das2.detect.fusion import TYPE_PRECEDENCE  # noqa: E402
from das2.models import (  # noqa: E402
    MAINTENANCE_TYPES,
    PROCESS_TYPES,
    QARTOD_TEST,
    SENSOR_HEALTH_TYPES,
    AnomalyType,
    QartodFlag,
    aggregate_flags,
    flag_for,
    qartod_label,
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


N, DT = 900, 120
TS = pd.Series(pd.date_range("2026-09-20", periods=N, freq=f"{DT}s"))


def series(seed: int = 3) -> np.ndarray:
    return 3.0 + np.random.default_rng(seed).normal(0, 0.004, N)


def types_of(values: np.ndarray) -> list[AnomalyType]:
    return [s.type for s in detect_level_shift(TS, values, unit="m")]


def main() -> int:
    print("\nnames follow QARTOD where QARTOD has a name")
    check("the invented DITHERING_DEAD is gone",
          not hasattr(AnomalyType, "DITHERING_DEAD"))
    check("it is QARTOD Test 10, Attenuated Signal",
          QARTOD_TEST[AnomalyType.ATTENUATED_SIGNAL] == (10, "Attenuated Signal"))
    for atype, test in [(AnomalyType.STALE, 1), (AnomalyType.RANGE_VIOLATION, 4),
                        (AnomalyType.SPIKE, 6), (AnomalyType.FLATLINE, 8)]:
        check(f"{atype.value} is QARTOD Test {test}",
              (QARTOD_TEST.get(atype) or (None,))[0] == test)
    check("a label carries the citation",
          qartod_label(AnomalyType.FLATLINE) == "FLATLINE (QARTOD 8 Flat Line)")

    print("\nand says plainly where it does not")
    ours = [t for t, v in QARTOD_TEST.items() if v is None]
    check("types with no standard equivalent are mapped to None explicitly",
          AnomalyType.QUANTISATION_COLLAPSE in ours
          and AnomalyType.INSTRUMENT_OFFSET in ours,
          f"{len(ours)} of ours")
    check("every type is accounted for, one way or the other",
          set(QARTOD_TEST) == set(AnomalyType),
          "a type missing from the table reads as 'no comment', which is "
          "how an invention becomes mistaken for a standard")
    check("a label for one of ours claims nothing",
          qartod_label(AnomalyType.QUANTISATION_COLLAPSE)
          == "QUANTISATION_COLLAPSE")

    print("\nold rows still read back")
    # das2_sensor_anomaly.dominant_type is a string column with months of data
    # in it. A rename without this turns every historical row into a ValueError.
    check("DITHERING_DEAD deserialises to ATTENUATED_SIGNAL",
          AnomalyType("DITHERING_DEAD") is AnomalyType.ATTENUATED_SIGNAL)
    check("and an unknown value still raises",
          _raises(lambda: AnomalyType("NOT_A_TYPE")),
          "silently inventing a type would be worse than failing")

    print("\nflags are a severity scale, separate from the reason")
    check("a fact about the channel FAILs",
          flag_for(AnomalyType.FLATLINE) is QartodFlag.FAIL)
    check("no data is MISSING, not FAIL",
          flag_for(AnomalyType.STALE) is QartodFlag.MISSING)
    check("an inference is only ever SUSPECT",
          flag_for(AnomalyType.LEVEL_SHIFT) is QartodFlag.SUSPECT
          and flag_for(AnomalyType.INSTRUMENT_OFFSET) is QartodFlag.SUSPECT,
          "QARTOD's FAIL means 'failed the primary criterion'; a statistical "
          "verdict is not that")
    check("aggregation follows QARTOD's precedence, not numeric order",
          aggregate_flags([QartodFlag.MISSING, QartodFlag.FAIL]) is QartodFlag.FAIL,
          "MISSING is 9 and FAIL is 4, but 'the data is wrong' beats "
          "'we have no data'")
    check("SUSPECT beats GOOD", aggregate_flags(
        [QartodFlag.GOOD, QartodFlag.SUSPECT]) is QartodFlag.SUSPECT)
    check("nothing at all is UNKNOWN", aggregate_flags([]) is QartodFlag.UNKNOWN)

    print("\na recalibration is not the water moving")
    base = series()

    recal = base.copy()
    recal[500:] += 0.35
    check("an instantaneous step that never returns is the instrument",
          types_of(recal) == [AnomalyType.INSTRUMENT_OFFSET],
          "water has mass; it does not change level between two samples")

    ramp = np.linspace(0, 0.35, 8)
    event = base.copy()
    event[500:508] += ramp
    event[508:640] += 0.35
    event[640:660] += np.linspace(0.35, 0, 20)
    check("one that ramps in and recedes is the water",
          types_of(event) == [AnomalyType.LEVEL_SHIFT])

    sustained = base.copy()
    sustained[500:508] += ramp
    sustained[508:] += 0.35
    check("ramping in is enough on its own to stay LEVEL_SHIFT",
          types_of(sustained) == [AnomalyType.LEVEL_SHIFT],
          "both tests are required; a rise with no recession is still a rise")

    print("\nand when the evidence is thin it says LEVEL_SHIFT")
    late = base.copy()
    late[880:] += 0.35
    check("a step near the window edge is not called an offset",
          types_of(late) == [AnomalyType.LEVEL_SHIFT],
          f"'never came back' over {(N - 880) * DT / 60:.0f} min is not "
          f"evidence; {OFFSET_MIN_HOLD_S / 3600:.0f} h is the minimum")

    print("\nthe consequences of the split")
    check("an offset is maintenance, not dispatch",
          AnomalyType.INSTRUMENT_OFFSET in MAINTENANCE_TYPES,
          "the response is 'check the calibration', same as DRIFT")
    check("it is NOT a process type",
          AnomalyType.INSTRUMENT_OFFSET not in PROCESS_TYPES,
          "so it cannot count towards a regional event")
    check("and NOT a sensor-health type either",
          AnomalyType.INSTRUMENT_OFFSET not in SENSOR_HEALTH_TYPES,
          "a maintenance sweep recalibrating twenty instruments must not be "
          "reported as the comms link having failed")
    check("LEVEL_SHIFT stays a process type",
          AnomalyType.LEVEL_SHIFT in PROCESS_TYPES)
    check("offset outranks level shift when both fit",
          TYPE_PRECEDENCE.index(AnomalyType.INSTRUMENT_OFFSET)
          < TYPE_PRECEDENCE.index(AnomalyType.LEVEL_SHIFT),
          "checking a calibration is cheaper and safer than sending a crew")
    check("every type has a precedence slot",
          set(TYPE_PRECEDENCE) == set(AnomalyType),
          "one missing sorts last by accident rather than by decision")

    print("\nthe offset verdict shows its working")
    sig = detect_level_shift(TS, recal, unit="m")[0]
    evidence = sig.detail.get("offset_evidence") or {}
    check("the transition length is reported", "transition_samples" in evidence,
          f"{evidence.get('transition_samples')} sample(s) between the levels")
    check("how long it held is reported", "held_h" in evidence,
          f"{evidence.get('held_h')} h")
    check("how close it came back is reported",
          "closest_return_fraction" in evidence,
          f"{evidence.get('closest_return_fraction')} of the step")
    check("a LEVEL_SHIFT carries no offset evidence",
          "offset_evidence" not in detect_level_shift(TS, event, unit="m")[0].detail)

    print(f"\n{passed} passed, {failed} failed.")
    return 1 if failed else 0


def _raises(fn) -> bool:
    try:
        fn()
    except Exception:                                      # noqa: BLE001
        return True
    return False


if __name__ == "__main__":
    sys.exit(main())
