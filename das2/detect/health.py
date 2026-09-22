"""
das2.detect.health
==================

Sensor-health detectors: is the *instrument* broken?

These run on the raw irregular stream, need no resampling, and — crucially —
need no quantile calibration. That last point is what gives the system a null
hypothesis for the first time. The detector being replaced had none: its
thresholds never bound, because a quantile always won, so it emitted about ten
sensors per run whether the network was healthy or on fire.

A range violation is not "the top 0.3% of points". It is a reading outside what
the instrument can physically produce. On a quiet night it fires zero times.

Every detector here is comparative where it has to be
-----------------------------------------------------
The temptation is absolute rules — "unchanged for 30 minutes", "silent for an
hour". Measured against the real feed, those are catastrophic:

* **57% of actively-reporting sensors never change value** in two hours, so an
  absolute flatline rule raises ~1,400 alerts per run.
* Reporting cadence spans 120 s to 4,798 s between sensors, so a fixed silence
  timeout pages for every slow-scanning sensor.
* An idle flowmeter sitting at zero with symmetric noise reads negative about
  half the time, so a raw `value < 0` range test flagged 1,079 points from one
  healthy meter in a smoke run.

So `FLATLINE` asks the sensor's own profile whether it is supposed to move,
`STALE` compares against that sensor's own historical cadence, and
`RANGE_VIOLATION` must clear the sensor's own measured noise.

Where a profile is missing or too thin, the detector **abstains** rather than
guessing. A detector that stays quiet when it cannot tell is worth far more than
one that guesses and has to be switched off.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from das2.detect.changepoint import detect_level_shift
from das2.detect.profile import SensorProfile, _resolution
from das2.timeutils import to_epoch_seconds
from das2.models import AnomalyType, Signal

DETECTOR = "health"


# --------------------------------------------------------------------------- #
# Tunables
# --------------------------------------------------------------------------- #
#: A flatline must exceed the sensor's own longest normal flat run by this
#: multiple before it counts. Generous on purpose: the cost of a false "your
#: sensor is frozen" is an engineer driving to a working instrument.
FLATLINE_MULTIPLE = 3.0

#: Absolute floor on a flatline, whatever the sensor's own history says. Below
#: this, a frozen value is more likely a quiet half-hour than a fault, and the
#: cost of being wrong is an engineer driving to a working instrument.
FLATLINE_FLOOR_S = 1800.0

#: Silence is judged against the sensor's own p99 reporting interval, times this.
#: A sensor that normally reports every 120 s with a p99 of 300 s is stale after
#: ~25 minutes; one that normally reports hourly is not.
STALE_MULTIPLE = 5.0
STALE_FLOOR_S = 1800.0

#: A range violation must clear this many robust sigma of the sensor's own noise.
RANGE_TOLERANCE_SIGMA = 6.0

#: A spike must move this many robust sigma of the sensor's own spread.
#: Measured headroom on the fixture fleet: the injected spike is ~350 sigma,
#: the worst step in pure sensor noise is ~5 sigma.
SPIKE_SIGMA = 8.0

#: Samples used to establish the pre-spike baseline, and to look for the
#: return. The return window bounds how long an excursion may last and still
#: count as a spike rather than a level shift.
SPIKE_BASELINE_N = 5
SPIKE_RETURN_N = 10

#: How close to baseline counts as "came back".
SPIKE_RETURN_FRACTION = 0.35

#: Reverse flow must be sustained to distinguish a real backflow from noise
#: around zero on an idle meter.
REVERSE_FLOW_MIN_S = 300.0

#: Quantisation collapse: the later part of the window must resolve steps this
#: many times coarser than the earlier part. A failing ADC or a transmitter
#: dropping bits shows up as the signal snapping to an increasingly coarse
#: grid, long before the value itself becomes implausible -- which is what
#: makes it worth detecting at all.
QUANT_COLLAPSE_FACTOR = 4.0

#: ...and the coarsened step must be this many robust sigma of the sensor's own
#: noise. Without it, a sensor that merely reports to three decimal places
#: registers as "quantised" and any rounding change looks like a collapse.
QUANT_COLLAPSE_SIGMA = 3.0

#: Each half of the window needs this many points before its resolution can be
#: estimated at all.
QUANT_MIN_POINTS = 60

#: Dithering-dead: a live analog input normally jitters by at least a count or
#: two of noise. A window whose entire range fits inside this many resolution
#: steps is reporting a number rather than measuring one -- the classic
#: symptom of an input stuck upstream of the ADC, which FLATLINE misses
#: because the value is not exactly constant.
DITHER_MAX_STEPS = 2.0

#: ...and it only counts if the sensor normally moves far more than that.
DITHER_FRACTION_OF_NORMAL = 0.1

#: Window over which dithering is judged. Long enough that a genuinely quiet
#: process period does not look dead.
DITHER_WINDOW_S = 21600.0        # 6 hours


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs as (start, end) inclusive index pairs."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    splits = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[splits + 1]]
    ends = np.r_[idx[splits], idx[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def _scale(values: np.ndarray, profile: SensorProfile | None) -> float:
    """
    Robust spread, floored so a quiet sensor is not scored against zero.

    Without the floor, any excursion on a constant sensor scores zero: a flat
    neighbourhood has MAD 0, and dividing by it yields no signal at all. That
    was a measured defect in the previous detector — 300 zeros plus a spike to
    250.0 scored exactly 0.000.
    """
    if profile is not None and np.isfinite(profile.mad) and profile.mad > 0:
        sigma = 1.4826 * profile.mad
    else:
        med = float(np.median(values))
        sigma = 1.4826 * float(np.median(np.abs(values - med)))
    resolution = profile.resolution if profile else 0.0
    return max(sigma, float(resolution), 1e-12)


# --------------------------------------------------------------------------- #
# Detectors
# --------------------------------------------------------------------------- #
def detect_flatline(ts: pd.Series, values: np.ndarray,
                    profile: SensorProfile | None) -> list[Signal]:
    """
    The sensor is still reporting, but the value has frozen.

    Abstains unless the profile says this sensor normally moves. That guard is
    the whole detector: without it this fires on 57% of the healthy fleet.
    """
    if profile is None or not profile.is_well_observed or profile.is_static:
        return []
    if len(values) < 3:
        return []

    seconds = to_epoch_seconds(ts)
    threshold = max(profile.expected_flat_seconds * FLATLINE_MULTIPLE,
                    FLATLINE_FLOOR_S)

    signals: list[Signal] = []
    start = 0
    for i in range(1, len(values) + 1):
        ended = i == len(values) or values[i] != values[start]
        if not ended:
            continue
        duration = seconds[i - 1] - seconds[start]
        # n_raw >= 2 by construction here: the sensor kept reporting through the
        # flat stretch, which is what separates a frozen value from a dropout.
        if duration >= threshold and (i - start) >= 2:
            signals.append(Signal(
                type=AnomalyType.FLATLINE,
                start=pd.Timestamp(ts.iloc[start]).to_pydatetime(),
                end=pd.Timestamp(ts.iloc[i - 1]).to_pydatetime(),
                detector=DETECTOR,
                magnitude=duration,
                unit="s",
                n_points=i - start,
                detail={
                    "value": float(values[start]),
                    "normal_flat_s": round(profile.expected_flat_seconds, 1),
                    "normal_change_rate_per_hour": profile.change_rate_per_hour,
                },
            ))
        start = i
    return signals


def detect_stale(ts: pd.Series, profile: SensorProfile | None,
                 window_end: datetime | None = None) -> list[Signal]:
    """
    The sensor has stopped reporting.

    Distinct from FLATLINE: this is an absence of rows, not a frozen value.
    Conflating the two loses the difference between "instrument stuck at a
    plausible reading" and "RTU offline", which need different responses.

    Judged against the sensor's own p99 interval, because cadence spans 120 s to
    4,798 s across this fleet and a single global timeout cannot serve both.
    """
    if profile is None or not profile.is_well_observed:
        return []
    if not np.isfinite(profile.p99_interval_s) or profile.p99_interval_s <= 0:
        return []
    if len(ts) == 0:
        return []

    threshold = max(profile.p99_interval_s * STALE_MULTIPLE, STALE_FLOOR_S)
    times = pd.to_datetime(pd.Series(ts)).sort_values().reset_index(drop=True)
    seconds = to_epoch_seconds(times)

    signals: list[Signal] = []
    gaps = np.diff(seconds)
    for i in np.flatnonzero(gaps >= threshold):
        signals.append(Signal(
            type=AnomalyType.STALE,
            start=pd.Timestamp(times.iloc[i]).to_pydatetime(),
            end=pd.Timestamp(times.iloc[i + 1]).to_pydatetime(),
            detector=DETECTOR,
            magnitude=float(gaps[i]),
            unit="s",
            n_points=0,
            detail={"normal_p99_interval_s": round(profile.p99_interval_s, 1)},
        ))

    # Still silent at the end of the window: the most urgent case, and one a
    # gap-based scan alone would miss because there is no closing row.
    if window_end is not None:
        trailing = (pd.Timestamp(window_end).timestamp() - seconds[-1])
        if trailing >= threshold:
            signals.append(Signal(
                type=AnomalyType.STALE,
                start=pd.Timestamp(times.iloc[-1]).to_pydatetime(),
                end=pd.Timestamp(window_end).to_pydatetime(),
                detector=DETECTOR,
                magnitude=float(trailing),
                unit="s",
                n_points=0,
                detail={"ongoing": True,
                        "normal_p99_interval_s": round(profile.p99_interval_s, 1)},
            ))
    return signals


def detect_range_violation(ts: pd.Series, values: np.ndarray,
                           range_min: float | None, range_max: float | None,
                           profile: SensorProfile | None,
                           unit: str = "") -> list[Signal]:
    """
    A reading outside what the instrument can physically produce.

    Needs no statistics at all, which is what makes it the backbone of the null
    hypothesis — but it does need a deadband. The bounds are fleet-wide, and an
    idle flowmeter at zero with symmetric noise reads negative about half the
    time: testing raw `value < 0` produced 1,079 flagged points and 542 events
    from one healthy meter.
    """
    if range_min is None and range_max is None:
        return []
    if len(values) == 0:
        return []

    tolerance = RANGE_TOLERANCE_SIGMA * _scale(values, profile)
    below = values < (range_min - tolerance) if range_min is not None else np.zeros(len(values), bool)
    above = values > (range_max + tolerance) if range_max is not None else np.zeros(len(values), bool)
    mask = below | above
    if not mask.any():
        return []

    signals: list[Signal] = []
    for s, e in _runs(mask):
        segment = values[s:e + 1]
        worst = segment[np.argmax(np.abs(segment - np.clip(
            segment, range_min if range_min is not None else -np.inf,
            range_max if range_max is not None else np.inf)))]
        bound = range_min if worst < (range_min or -np.inf) else range_max
        signals.append(Signal(
            type=AnomalyType.RANGE_VIOLATION,
            start=pd.Timestamp(ts.iloc[s]).to_pydatetime(),
            end=pd.Timestamp(ts.iloc[e]).to_pydatetime(),
            detector=DETECTOR,
            magnitude=float(abs(worst - bound)) if bound is not None else float(worst),
            unit=unit,
            n_points=e - s + 1,
            detail={"worst_value": float(worst), "bound": bound,
                    "deadband": round(tolerance, 6)},
        ))
    return signals


def detect_spike(ts: pd.Series, values: np.ndarray,
                 profile: SensorProfile | None, unit: str = "") -> list[Signal]:
    """
    A large, brief excursion that returns to where it came from.

    Two gates, both required, because either alone produces nonsense.

    **Size, measured against the sensor's own spread.** The first version of
    this detector thresholded the distribution of |dv/dt| at
    `median + 8 x 1.4826 x MAD` of that distribution. That has no null
    hypothesis: for ordinary Gaussian noise the MAD of |dv/dt| is comparable to
    its own median, so the threshold sits only a few multiples above typical,
    and over a 2,160-sample window plenty of pure noise clears it. Measured on
    the fixture fleet it produced **11 spikes for 1 injected fault**, every one
    of them a deviation of 0.005 to 0.07 engineering units -- while missing the
    real injected spike, a 34.8 V drop. It was reporting noise and ignoring the
    signal, which is the exact failure mode of the detector this replaces.

    Gating on step size relative to the sensor's *value* scale separates them
    cleanly: on the fixture the real spike is ~350 sigma while the worst noise
    step is ~5 sigma. That is three orders of magnitude of headroom, not a
    threshold balanced on a knife edge.

    **Return.** A spike goes and comes back; a step goes and stays. Without the
    return test this fires on every genuine level shift as well, which is a
    different fault with a different response -- and LEVEL_SHIFT is what the
    changepoint detector is for.

    Non-positive intervals are dropped rather than divided by: 7% of sensors
    have duplicate timestamps and 12% have out-of-order rows, and a naive dv/dt
    turns every one of those into an infinite-rate spike.
    """
    n = len(values)
    if n < 6:
        return []

    seconds = to_epoch_seconds(ts)
    dt = np.diff(seconds)
    dv = np.diff(values)
    scale = _scale(values, profile)

    step = np.zeros(n - 1, dtype=float)
    usable = np.isfinite(dv) & (dt > 0)
    step[usable] = np.abs(dv[usable]) / scale

    candidates = np.flatnonzero(step >= SPIKE_SIGMA)
    if candidates.size == 0:
        return []

    signals: list[Signal] = []
    consumed = -1
    for i in candidates:
        if i <= consumed:
            continue                     # the far edge of a spike already emitted

        # Where the sensor was sitting before the jump. A median over a short
        # lead-in rather than the single previous sample, so one noisy reading
        # cannot define the baseline the excursion is measured from.
        lo = max(0, i - SPIKE_BASELINE_N + 1)
        baseline = float(np.median(values[lo:i + 1]))
        excursion = float(values[i + 1] - baseline)
        if abs(excursion) < SPIKE_SIGMA * scale:
            continue

        # Does it come back? Checked over a bounded lookahead, so a spike is
        # distinguished from a step by behaviour rather than by assumption.
        hi = min(n, i + 2 + SPIKE_RETURN_N)
        after = values[i + 1:hi]
        back = np.flatnonzero(np.abs(after - baseline)
                              <= SPIKE_RETURN_FRACTION * abs(excursion))
        if back.size == 0:
            continue                     # a sustained move: a level shift, not a spike

        end_index = min(n - 1, i + 1 + int(back[0]))
        consumed = end_index
        peak_rate = float(np.max(step[i:max(i + 1, end_index)]) * scale
                          / max(1e-9, float(np.median(dt[dt > 0]))))

        signals.append(Signal(
            type=AnomalyType.SPIKE,
            start=pd.Timestamp(ts.iloc[i]).to_pydatetime(),
            end=pd.Timestamp(ts.iloc[end_index]).to_pydatetime(),
            detector=DETECTOR,
            # Reported in ENGINEERING UNITS, not as a rate. The rate form made
            # a 34.8 V collapse read as "0.29", which is not a number anyone
            # can sanity-check against an instrument.
            magnitude=round(excursion, 6),
            unit=unit,
            n_points=end_index - i + 1,
            detail={
                "baseline": round(baseline, 6),
                "peak": round(float(values[i + 1]), 6),
                "sigma": round(abs(excursion) / scale, 1),
                "scale": round(scale, 6),
                "peak_rate_per_s": round(peak_rate, 6),
                "returned_after_samples": int(back[0]) + 1,
            },
        ))
    return signals


def detect_reverse_flow(ts: pd.Series, values: np.ndarray,
                        profile: SensorProfile | None,
                        unit: str = "") -> list[Signal]:
    """
    Sustained negative flow — backflow through a failed non-return valve.

    A real, operationally important event. The previous pipeline discarded these
    as "invalid", which is backwards. Must be sustained and clear the noise
    deadband, so an idle meter dithering around zero does not qualify.
    """
    if len(values) < 3:
        return []
    tolerance = RANGE_TOLERANCE_SIGMA * _scale(values, profile)
    mask = values < -tolerance
    if not mask.any():
        return []

    seconds = to_epoch_seconds(ts)
    signals: list[Signal] = []
    for s, e in _runs(mask):
        duration = float(seconds[e] - seconds[s])
        if duration < REVERSE_FLOW_MIN_S:
            continue
        signals.append(Signal(
            type=AnomalyType.REVERSE_FLOW,
            start=pd.Timestamp(ts.iloc[s]).to_pydatetime(),
            end=pd.Timestamp(ts.iloc[e]).to_pydatetime(),
            detector=DETECTOR,
            magnitude=float(abs(np.min(values[s:e + 1]))),
            unit=unit,
            n_points=e - s + 1,
            detail={"duration_s": duration, "deadband": round(tolerance, 6)},
        ))
    return signals


# --------------------------------------------------------------------------- #
def detect_quantisation_collapse(ts: pd.Series, values: np.ndarray,
                                 profile: SensorProfile | None,
                                 unit: str = "") -> list[Signal]:
    """
    The sensor's resolution has degraded: it now moves in coarser steps.

    A failing ADC, a transmitter dropping bits, or a comms path silently
    truncating shows up first as the signal snapping to a coarser grid. The
    *value* stays plausible throughout, so nothing else here notices -- range,
    spike and flatline all see a healthy-looking number. That is precisely why
    it is worth detecting: it is a warning that arrives before the reading
    becomes wrong.

    Scanned in blocks rather than by splitting the window in half. The
    half-split was the first implementation and it failed on the realistic
    case: a collapse that begins midway and lasts 20 of 72 hours leaves the
    later half a mixture of coarse and fine steps, and a mixture reads as
    "not quantised" -- so the detector was blind to exactly the shape it was
    written for. Blocks also localise *when* it started, which the half-split
    could not.

    Each block is compared against the sensor's early behaviour, so the sensor
    is its own control and no fleet-wide notion of "normal resolution" is
    needed -- just as well, since resolution spans three orders of magnitude
    across this fleet.
    """
    n = len(values)
    if n < QUANT_MIN_POINTS * 3:
        return []

    block = max(QUANT_MIN_POINTS, n // 12)
    bounds = [(i, min(n, i + block)) for i in range(0, n, block)]
    bounds = [(a, b) for a, b in bounds if b - a >= QUANT_MIN_POINTS]
    if len(bounds) < 3:
        return []

    resolutions = [_resolution(values[a:b]) for a, b in bounds]

    # Reference: the earliest third of the window, before anything went wrong.
    # Noise is taken from the same early stretch and deliberately NOT from the
    # profile, which is built over the whole window and so already contains the
    # collapse -- on the fixture that inflated sigma nearly tenfold and hid the
    # fault. Same self-contamination that made FLATLINE unable to fire.
    reference_blocks = max(1, len(bounds) // 3)
    early_end = bounds[reference_blocks - 1][1]
    sigma = _scale(values[:early_end], None)
    early = [r for r in resolutions[:reference_blocks] if r > 0]
    reference = float(np.median(early)) if early else 0.0

    flagged = []
    for k in range(reference_blocks, len(bounds)):
        r = resolutions[k]
        if r <= 0:
            continue
        if r < QUANT_COLLAPSE_SIGMA * sigma:
            continue
        if reference > 0 and r < QUANT_COLLAPSE_FACTOR * reference:
            continue
        flagged.append(k)

    if not flagged:
        return []

    # Merge adjacent flagged blocks into one event.
    runs: list[list[int]] = [[flagged[0]]]
    for k in flagged[1:]:
        if k == runs[-1][-1] + 1:
            runs[-1].append(k)
        else:
            runs.append([k])

    signals: list[Signal] = []
    for run in runs:
        i = bounds[run[0]][0]
        j = bounds[run[-1]][1] - 1
        coarsest = max(resolutions[k] for k in run)
        signals.append(Signal(
            type=AnomalyType.QUANTISATION_COLLAPSE,
            start=pd.Timestamp(ts.iloc[i]).to_pydatetime(),
            end=pd.Timestamp(ts.iloc[j]).to_pydatetime(),
            detector=DETECTOR,
            magnitude=round(coarsest, 6),
            unit=unit,
            n_points=j - i + 1,
            detail={
                "resolution_before": round(reference, 6),
                "resolution_after": round(coarsest, 6),
                "ratio": (round(coarsest / reference, 1) if reference > 0 else None),
                "was_continuous_before": reference <= 0,
                "sigma": round(sigma, 6),
                "blocks_affected": len(run),
            },
        ))
    return signals


def detect_dithering_dead(ts: pd.Series, values: np.ndarray,
                          profile: SensorProfile | None,
                          unit: str = "") -> list[Signal]:
    """
    Alive, reporting, not quite constant -- and measuring nothing.

    A live analog input jitters by a count or two of noise simply because the
    world and the ADC are noisy. An input that has come adrift upstream of the
    converter often keeps reporting a *nearly* constant number: it wanders by
    one least-significant bit and no more. FLATLINE cannot see that, because
    the value is not exactly constant, and the reading stays perfectly
    plausible.

    So this asks a different question from FLATLINE: not "has it stopped
    changing?" but "has it stopped changing *enough to be measuring
    anything*?", judged against how much this sensor normally moves.

    Requires a non-zero range on purpose, so the two detectors are disjoint and
    an exactly frozen sensor is reported once, as a flatline.
    """
    if profile is None or not profile.is_well_observed or profile.is_static:
        return []
    if len(values) < 20:
        return []

    resolution = profile.resolution
    if resolution <= 0:
        return []                       # not a quantised sensor; nothing to count

    normal_spread = 1.4826 * profile.mad if np.isfinite(profile.mad) else 0.0
    if normal_spread <= 0:
        return []
    ceiling = min(DITHER_MAX_STEPS * resolution,
                  DITHER_FRACTION_OF_NORMAL * normal_spread)
    if ceiling <= 0:
        return []

    seconds = to_epoch_seconds(ts)
    n = len(values)
    signals: list[Signal] = []

    # Two-pointer scan with a running min/max, rather than re-measuring a
    # growing slice. The slice form was not just slower, it was wrong: the
    # sample that BROKE the stretch was still inside the slice being measured,
    # so every candidate's range included the jump that ended it and nothing
    # ever qualified.
    start = 0
    while start < n:
        lo = hi = float(values[start])
        end = start
        j = start + 1
        while j < n:
            value = float(values[j])
            new_lo, new_hi = min(lo, value), max(hi, value)
            if new_hi - new_lo > ceiling:
                break
            lo, hi, end = new_lo, new_hi, j
            j += 1

        duration = seconds[end] - seconds[start]
        held_span = hi - lo
        if (duration >= DITHER_WINDOW_S and held_span > 0
                and (end - start) >= 20):
            signals.append(Signal(
                type=AnomalyType.DITHERING_DEAD,
                start=pd.Timestamp(ts.iloc[start]).to_pydatetime(),
                end=pd.Timestamp(ts.iloc[end]).to_pydatetime(),
                detector=DETECTOR,
                magnitude=round(held_span, 6),
                unit=unit,
                n_points=end - start + 1,
                detail={
                    "range_observed": round(held_span, 6),
                    "resolution": round(resolution, 6),
                    "normal_spread": round(normal_spread, 6),
                    "duration_h": round(duration / 3600.0, 2),
                },
            ))
        start = end + 1
    return signals


def run_health_checks(ts: pd.Series, values: np.ndarray,
                      profile: SensorProfile | None = None,
                      *, equipment_kind: str = "measurement",
                      range_min: float | None = None,
                      range_max: float | None = None,
                      unit: str = "",
                      is_flow: bool = False,
                      window_end: datetime | None = None) -> list[Signal]:
    """
    Every health detector for one sensor.

    Counters (kWh, run hours) are skipped entirely: they only ever climb, so a
    flat counter means the plant is idle and a drop means a rollover. Running
    flatline or spike detection over them produces nothing but noise.

    Config points (setpoints, simulation values) are skipped because their value
    is an operator decision — there is nothing for an anomaly detector to say.
    """
    if equipment_kind in ("counter", "config"):
        return []
    if len(values) == 0:
        return []

    signals: list[Signal] = []
    signals += detect_flatline(ts, values, profile)
    # Level shifts are what make an AREA event visible: pressure falling across
    # a district is not a sensor fault, and without this the clustering layer
    # has nothing to cluster.
    signals += detect_level_shift(ts, values, profile, unit)
    signals += detect_stale(ts, profile, window_end=window_end)
    signals += detect_range_violation(ts, values, range_min, range_max, profile, unit)
    signals += detect_spike(ts, values, profile, unit)
    signals += detect_quantisation_collapse(ts, values, profile, unit)
    signals += detect_dithering_dead(ts, values, profile, unit)
    if is_flow:
        signals += detect_reverse_flow(ts, values, profile, unit)
    return signals
