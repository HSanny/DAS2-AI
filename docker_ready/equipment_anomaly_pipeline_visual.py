# equipment_anomaly_pipeline_visual.py

import os
os.environ["MPLBACKEND"] = "Agg"  # for headless servers
import re
import sys
import time
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ioff()
import matplotlib.dates as mdates
from datetime import datetime
from sklearn.ensemble import IsolationForest
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from cluster_suppression import annotate_suppression

# =========================
# Tunables
# =========================
TIME_WINDOW_HOURS = 72   # e.g. 72 or None

# Robust-Z
ROLL_WIN_Z   = 24
MAD_Z_THRESH = 4.5
MAD_Z_HARD   = 8.0

# Scale flooring for robust-Z (see robust_z / resolution_estimate).
# RZ_SATURATE also caps reported scores: without it, dividing by a near-zero
# floor yields |RZ| in the tens of thousands, which is meaningless as a
# magnitude and distorts every ranking that sorts on Peak_RZ.
RZ_SATURATE                 = 50.0
RZ_REL_SCALE_FLOOR          = 1e-6   # relative guard vs the local median level
RESOLUTION_MIN_ACTIVE_FRAC  = 0.05   # need >=5% non-zero diffs to trust a resolution estimate
QUANT_MIN_SAMPLES           = 10     # minimum non-zero diffs before testing for quantisation
QUANT_TOL                   = 0.05   # median |ratio - round(ratio)| below this => quantised

# Isolation Forest
ISO_BASE_CONTAM   = 0.01
ISO_N_ESTIMATORS  = 200
ISO_WIN           = 12
ISO_RANDOM_STATE  = 42

# =========================
# Rate-of-change channel (replaces the former sliding-DTW channel)
# =========================
# The old `dtw_sliding` compared v[i-w:i] against v[i-w+1:i+1] -- the SAME
# window shifted by one sample. Because DTW must match both endpoints and the
# interior then aligns at zero cost, its optimal path always costs exactly
#
#     |v[i-w] - v[i-w+1]|  +  |v[i-1] - v[i]|
#
# Verified against an exact DTW implementation: correlation 1.000000, maximum
# absolute difference 0.0. It was an identity, not an approximation -- so that
# channel never compared shapes with anything. It measured a first difference,
# plus the same difference echoed w samples later.
#
# That echo was a live bug: a transient at index k produced a phantom second
# peak at k+w (reproduced at indices 123/124 for a spike at 100 with w=24),
# which MERGE_GAP=3 and COOLDOWN=12 neither merge nor suppress, so it cast a
# spurious third vote at the wrong time.
#
# It also cost ~16.5k fastdtw() calls with a Python-lambda cost function on the
# highest-rate sensor, to compute what is one vectorised line.
#
# Replaced by an explicit, honest rate-of-change channel: |dv/dt| in
# engineering units per second. Same intent, no echo, no dependency, O(n).
ROC_K          = 6.0    # MAD multiplier for the rate threshold
MIN_SENSOR_POINTS = 48  # admission floor (previously implied by the DTW window)

# Per-sensor anomaly budget
TARGET_POINT_RATE = 0.003
MAX_POINT_RATE    = 0.01

# Event gating / smoothing
MIN_EVENT_LEN = 6
MERGE_GAP     = 3
COOLDOWN      = 12
Z_EVENT_MIN   = 5.5
REL_JUMP_MIN  = 0.10
ROC_EVENT_RATE= 0.20

# =========================
# Time-based windows
# =========================
# Every window and gate above is counted in SAMPLES, but this feed is
# report-by-exception and sensors report at wildly different rates. Measured
# over the same 72h window (abnormal_sensor_backup.csv):
#
#   PulauTekong-Dissolved-Oxygen   16482 points ->  15.7 s between reports
#   Kranji1PS-Total-Flow-Rate       2083 points -> 124.4 s between reports
#
# So ROLL_WIN_Z = 24 is a 6.3-minute baseline on one sensor and a 49.8-minute
# baseline on the other -- an 8x difference in what the detector actually does,
# from a single constant. MIN_EVENT_LEN is worse, because it is a HARD filter:
# a 2-minute glitch alerts on the fast sensor and is silently discarded on the
# slow one. Neither behaviour was chosen; both fall out of the sampling rate.
#
# These are the same gates expressed as durations, then converted per sensor
# using that sensor's own median inter-arrival time.
#
# The values are anchored to ~120 s, the median inter-arrival across the
# sensors in the shipped sample (9 of the 10 cluster between 90 s and 125 s).
# That is deliberate: it leaves behaviour UNCHANGED for the large majority of
# sensors and corrects only the fast outliers, so this fix does not quietly
# move alert volume for the whole fleet at the same time.
#
# Set USE_TIME_BASED_WINDOWS=0 to restore the old sample-counted behaviour --
# needed as the control arm when shadow-comparing detector versions.
USE_TIME_BASED_WINDOWS = os.getenv("USE_TIME_BASED_WINDOWS", "1").strip() == "1"

ROLL_WIN_Z_SEC      = int(os.getenv("ROLL_WIN_Z_SEC",      "2880"))  # 48 min
ISO_WIN_SEC         = int(os.getenv("ISO_WIN_SEC",         "1440"))  # 24 min
MIN_EVENT_SEC       = int(os.getenv("MIN_EVENT_SEC",        "720"))  # 12 min
MERGE_GAP_SEC       = int(os.getenv("MERGE_GAP_SEC",        "360"))  #  6 min
COOLDOWN_SEC        = int(os.getenv("COOLDOWN_SEC",        "1440"))  # 24 min
STEP_WINDOW_SEC     = int(os.getenv("STEP_WINDOW_SEC",     "2400"))  # 40 min
STEP_WINDOW_MIN_SEC = int(os.getenv("STEP_WINDOW_MIN_SEC",  "600"))  # 10 min

# Sample counts are clamped so a pathological rate cannot produce a window of
# 3 points (meaningless) or 50 000 (unusably slow).
WINDOW_MIN_SAMPLES = 3
WINDOW_MAX_SAMPLES = 500

# Sensor-level selection (for final outputs)
MAX_ABNORMAL_SENSORS        = 10  # hard cap on sensors to send out
REQUIRED_METHODS_PER_SENSOR = 2   # how many detection families must contribute anomalies

# =========================
# Equipment-category filter
# =========================
# The client doesn't maintain analysis programs for 'Others' (unclassified
# analog) or 'Digital Signal' (unclassified digital) sensors. We skip them
# entirely at the detector — they're still in the DB inventory (dim/data
# tables), just not analysed or alerted on.
#
# Env-overridable. Set SKIP_UNCATEGORIZED_EQUIPMENT="" to disable the filter
# entirely (e.g. for debugging). Set to a comma-separated list to customise.
SKIP_UNCATEGORIZED_EQUIPMENT = {
    s.strip() for s in os.getenv(
        "SKIP_UNCATEGORIZED_EQUIPMENT",
        "Others,Digital Signal"
    ).split(",") if s.strip()
}

# =========================
# Edge / step-change anomaly suppression
# =========================
# Equipment cycling on/off (pumps starting, valves opening, gates moving)
# produces sharp transitions that the detectors (Z, ISO, ROC) all flag
# simultaneously because the value changes fast — statistically anomalous,
# operationally normal. Per direct client instruction (May 2026): suppress
# any event where the value level BEFORE differs significantly from the
# value level AFTER, regardless of how noisy either side is.
#
# Rule: an event at rows [s, e] is suppressed iff
#   |median(before-window) - median(after-window)| / max(|before|, |after|)
#       >= STEP_MIN_LEVEL_DIFF_PCT
# Brief transients that return to baseline (e.g. a momentary voltage dip
# that recovers) are NOT suppressed because their before/after medians are
# essentially identical.
#
# Trade-off accepted by client: a real fault that produces a clean sustained
# level shift (e.g. calibration drift settling to a new value) will also be
# suppressed. Client has acknowledged this and prefers reduced alert volume.
#
# Per-equipment-class controlled via DETECTOR_PROFILE.suppress_steps.
STEP_WINDOW_SAMPLES        = 20    # samples each side of the event (preferred)
STEP_WINDOW_MIN            = 5     # minimum samples needed on each side to judge
STEP_MIN_LEVEL_DIFF_PCT    = 0.10  # before/after median must differ by ≥ this fraction

# Detection profiles per Equipment (which methods to use)
DETECTOR_PROFILE = {
    'Pressure':         dict(use_z=True,  use_iso=True,  use_roc=True,  suppress_steps=True),
    'Flowrate':         dict(use_z=True,  use_iso=True,  use_roc=True,  suppress_steps=True),
    'Conductivity':     dict(use_z=True,  use_iso=False, use_roc=True,  suppress_steps=True),
    'Voltage':          dict(use_z=True,  use_iso=True,  use_roc=False, suppress_steps=True),
    'Dissolved Oxygen': dict(use_z=True,  use_iso=True,  use_roc=False, suppress_steps=True),
    'Temperature':      dict(use_z=True,  use_iso=False, use_roc=False, suppress_steps=True),
    'LevelSensor':      dict(use_z=True,  use_iso=False, use_roc=False, suppress_steps=True),
}
DEFAULT_PROFILE = dict(use_z=True, use_iso=True, use_roc=True, suppress_steps=True)

# Plotting
now_time = datetime.now().strftime("%Y%m%d_%H%M")
PLOTS_DIR = (Path("output_plots") / now_time).as_posix()
PLOT_PER_SENSOR_TIMESERIES = True   # <- turn ON/OFF per-sensor time series plots
PLOT_PER_SENSOR_CLUSTERS   = False    # cluster plots for selected abnormal sensors

# Logging / progress
VERBOSE_LEVEL = 1          # 0=quiet, 1=progress, 2=debug-per-step
PRINT_EVERY   = 25         # print ETA every N sensors
LOG_FILE      = "logs/detector_errors.log"


# ================= Helpers =================
# Physical plausibility ranges per equipment class.
#
# These are fleet-wide and crude -- "Pressure 0..20" is applied to every
# pressure sensor in Singapore, while the shipped sample contains pressure
# sensors with medians of 0.0034 and 3.90, so the band is far too wide to be
# useful for either. Replacing them with the per-point HH/H/L/LL limits already
# commissioned in the SCADA is the single highest-value improvement available
# here, but those limits are not in the current feed.
EQUIPMENT_RANGES = {
    'Temperature':      (0.0, 60.0),
    'Flowrate':         (0.0, 2000.0),
    'Conductivity':     (0.0, 50000.0),
    'Pressure':         (0.0, 20.0),
    'Voltage':          (0.0, 500.0),
    'Dissolved Oxygen': (0.0, 20.0),
    'LevelSensor':      (0.0, 100.0),
}

# A range violation must clear the sensor's OWN noise before it counts.
#
# Without this, the bound is applied to raw values and an idle flowmeter
# sitting at zero with symmetric measurement noise reports negative readings
# roughly half the time. On the shipped sample, MRRS-THOMSON FLOWMETER has
# Median_Value 0.001302 -- it is idle almost always. Testing `value < 0`
# against it flags ~53% of its samples as physically impossible.
#
# That was harmless only because violations used to be discarded. Now that they
# alert (correctly -- see the combine step), the raw bound would turn one
# healthy idle meter into ~1000 anomalies per run. Measured in a smoke run
# before this deadband existed: 1079 flagged points and 542 events, from a
# sensor doing nothing wrong.
RANGE_TOLERANCE_SIGMA = float(os.getenv("RANGE_TOLERANCE_SIGMA", "6.0"))


def range_tolerance(values, resolution=0.0):
    """
    Deadband outside the physical range, from the sensor's own spread.

    Uses a robust sigma so the tolerance is not itself inflated by the
    excursions being tested for, and floors at the measurement resolution: a
    reading cannot meaningfully violate a bound by less than the instrument
    can resolve.
    """
    v = pd.Series(values).astype(float)
    med = v.median()
    mad = (v - med).abs().median()
    sigma = 1.4826 * float(mad) if pd.notna(mad) else 0.0
    return max(RANGE_TOLERANCE_SIGMA * sigma, float(resolution), 0.0)


def range_violation_mask(equipment, values, tolerance=0.0):
    """
    Boolean mask of readings outside the equipment's physical range by more
    than `tolerance`.

    Note that a genuinely negative flow beyond the deadband still flags. That
    is intended -- reverse flow through a failed non-return valve is a real and
    important event. It is currently reported as a range violation; typing it
    as its own anomaly class is Phase 4 work.
    """
    v = np.asarray(values, dtype=float)
    bounds = EQUIPMENT_RANGES.get(equipment)
    if bounds is None:
        return np.zeros(len(v), dtype=bool)
    lo, hi = bounds
    with np.errstate(invalid="ignore"):
        return (v < lo - tolerance) | (v > hi + tolerance)


def is_rule_invalid(equipment, value, tolerance=0.0):
    """Scalar form, kept for callers that check a single reading."""
    bounds = EQUIPMENT_RANGES.get(equipment)
    if bounds is None:
        return False
    lo, hi = bounds
    return bool(value < lo - tolerance or value > hi + tolerance)


def median_interval_seconds(timestamps) -> float:
    """
    Median seconds between consecutive reports for one sensor.

    Only positive intervals count: duplicate timestamps (dt == 0) and
    out-of-order rows (dt < 0) both occur in this feed and would drag the
    median towards zero, which would then inflate every derived window.

    Returns NaN when there is nothing usable to measure.
    """
    ts = pd.to_datetime(pd.Series(timestamps).reset_index(drop=True))
    dt = ts.diff().dt.total_seconds()
    dt = dt[dt > 0]
    return float(dt.median()) if len(dt) else float("nan")


def samples_for(seconds: float, median_dt: float, fallback: int) -> int:
    """
    Convert a duration into a sample count for one sensor.

    Falls back to the legacy sample-counted constant when the rate cannot be
    measured, so a sensor with unusable timestamps still gets analysed rather
    than silently skipped.
    """
    if not (median_dt and np.isfinite(median_dt) and median_dt > 0):
        return int(fallback)
    n = int(round(seconds / median_dt))
    return int(np.clip(n, WINDOW_MIN_SAMPLES, WINDOW_MAX_SAMPLES))


def compute_sensor_windows(timestamps) -> dict:
    """
    Per-sensor sample counts for every window and gate, derived from that
    sensor's own reporting rate. See the time-based windows block in Tunables.
    """
    if not USE_TIME_BASED_WINDOWS:
        return {
            "median_dt_s": float("nan"),
            "roll_win_z": ROLL_WIN_Z,
            "iso_win": ISO_WIN,
            "min_event_len": MIN_EVENT_LEN,
            "merge_gap": MERGE_GAP,
            "cooldown": COOLDOWN,
            "step_window": STEP_WINDOW_SAMPLES,
            "step_window_min": STEP_WINDOW_MIN,
        }
    dt = median_interval_seconds(timestamps)
    return {
        "median_dt_s": dt,
        "roll_win_z": samples_for(ROLL_WIN_Z_SEC, dt, ROLL_WIN_Z),
        "iso_win": samples_for(ISO_WIN_SEC, dt, ISO_WIN),
        "min_event_len": samples_for(MIN_EVENT_SEC, dt, MIN_EVENT_LEN),
        "merge_gap": samples_for(MERGE_GAP_SEC, dt, MERGE_GAP),
        "cooldown": samples_for(COOLDOWN_SEC, dt, COOLDOWN),
        "step_window": samples_for(STEP_WINDOW_SEC, dt, STEP_WINDOW_SAMPLES),
        "step_window_min": samples_for(STEP_WINDOW_MIN_SEC, dt, STEP_WINDOW_MIN),
    }


def resolution_estimate(values, min_active_frac=RESOLUTION_MIN_ACTIVE_FRAC):
    """
    Estimate a sensor's measurement resolution (quantisation step), or return
    0.0 when the signal is not quantised.

    A quantised signal changes only in whole multiples of its step, so the
    smallest non-zero difference is the step and every other difference is
    close to an integer multiple of it. That is an falsifiable test, and it is
    the test used here -- a low percentile of the differences on its own is NOT
    a resolution estimate. For continuous data (say N(0,1) noise) the 25th
    percentile of |diff| is around 0.45, which is a substantial fraction of the
    real spread; flooring the scale with it would suppress genuine outliers on
    perfectly healthy sensors.

    Returns 0.0 in two cases, both meaning "no usable resolution evidence":

      * The series is too static. If nearly every difference is zero, the few
        non-zero ones are far more likely to BE the anomaly than to reveal the
        resolution -- using them as a scale floor would divide the excursion by
        itself and hide it, the exact failure this exists to prevent.
      * The differences are not consistent with any single step, i.e. the
        signal is continuous.

    robust_z() treats 0.0 as "fall through to the saturation rule".
    """
    v = np.asarray(values, dtype=float)
    d = np.abs(np.diff(v))
    d = d[np.isfinite(d)]
    if d.size == 0:
        return 0.0
    nz = d[d > 0]
    # Too static to infer anything, or too few changes for the test to mean much.
    if nz.size < QUANT_MIN_SAMPLES or (nz.size / d.size) < min_active_frac:
        return 0.0

    # Try the smallest observed steps as candidates; the true step must divide
    # all the others. Several candidates are tried because a single float
    # artefact could otherwise make the smallest difference unusable.
    for candidate in (np.min(nz), np.percentile(nz, 1), np.percentile(nz, 5)):
        if not np.isfinite(candidate) or candidate <= 0:
            continue
        ratios = nz / candidate
        residual = np.abs(ratios - np.round(ratios))
        if np.median(residual) < QUANT_TOL:
            return float(candidate)
    return 0.0


def robust_z(series, window=ROLL_WIN_Z, resolution=None):
    """
    Robust z-score of each point against a local median, scaled by a FLOORED
    estimate of local spread.

    Why the floor matters
    ---------------------
    The previous implementation did `mad.replace(0, np.nan)` and then
    `rz.fillna(0.0)`, which meant any window whose MAD was zero scored exactly
    0 -- i.e. no anomaly. That silently blinded the detector to the cases that
    matter most on this fleet:

      * 300 zeros with a single spike to 250.0  -> max|RZ| was 0.000
      * a 0.1-quantised 414.8 V bus dropping to 380 V -> 2.965 (below 4.5)

    Both are invisible because the median filter sees a flat neighbourhood, so
    MAD is 0. Idle pumps make this common here: abnormal_sensor_backup.csv
    contains Flowrate sensors with Median_Value of 0.0 and 0.001302, and those
    are precisely the sensors where a spurious reading matters.

    The fix has two parts:
      1. Floor the scale at the sensor's measurement resolution (and a tiny
         relative floor), so a quantised signal is scored against its real
         resolution rather than a degenerate zero.
      2. Where the local baseline is *exactly* constant and no resolution can
         be inferred, any non-zero deviation is by definition maximally
         surprising, so it saturates instead of collapsing to zero.

    Scores are clipped to +/-RZ_SATURATE. Dividing by a near-zero floor
    otherwise produces values in the tens of thousands, which are meaningless
    as a magnitude and distort every downstream ranking that uses Peak_RZ.

    Behaviour on healthy, noisy sensors is unchanged: there MAD comfortably
    exceeds the floor, so the floor never binds. That is deliberate -- buying
    sensitivity on quiet sensors by raising the false-alarm rate on healthy
    ones would be no improvement at all.

    Two known defects deliberately NOT fixed here
    ---------------------------------------------
    1. Mis-scaling. The formula is `1.4826 * dev / MAD`, but 1.4826 * MAD is
       the estimator of sigma, so the constant belongs on the denominator. As
       written, every score is inflated by 1.4826^2 ~= 2.198x, which means the
       documented thresholds are really:

           MAD_Z_THRESH 4.5 -> 2.05 sigma
           Z_EVENT_MIN  5.5 -> 2.50 sigma
           MAD_Z_HARD   8.0 -> 3.64 sigma

       Correcting it would change which points are flagged and therefore alert
       volume, which is out of scope for a Phase 0 fix. Every threshold in this
       file was tuned against the inflated scale, so the scale and the
       thresholds must be corrected together -- that happens in Phase 4, where
       thresholds are re-derived from scratch.

    2. Non-causality. The rolling windows are `center=True`, so a point's score
       depends on data that arrived after it. With a 72h window re-run every
       6h, each timestamp is re-scored ~12 times against different future
       context, which is why the same event appears and disappears between
       runs. Replacing this with a causal baseline is also Phase 4 work.

    Returns (rz, med, scale) -- the third element is the floored MAD actually
    used, not the raw MAD.
    """
    s = pd.Series(series).astype(float)
    med = s.rolling(window, min_periods=1, center=True).median()
    mad = (s - med).abs().rolling(window, min_periods=1, center=True).median()

    if resolution is None:
        resolution = resolution_estimate(s.values)

    # Floor 1: the instrument cannot resolve finer than its quantisation step.
    # Floor 2: a tiny relative guard, so a large-magnitude signal is not scored
    #          against a scale smaller than its own floating-point granularity.
    floor = np.maximum(float(resolution), RZ_REL_SCALE_FLOOR * med.abs())
    scale = np.maximum(mad, floor)

    dev = s - med
    with np.errstate(divide="ignore", invalid="ignore"):
        # NOTE: `1.4826 *` belongs on the denominator (1.4826*MAD estimates
        # sigma), so this formula inflates every score by 1.4826^2 ~= 2.198x.
        # It is preserved verbatim here ON PURPOSE -- see the mis-scaling note
        # in the docstring.
        rz = 1.4826 * dev / scale.where(scale > 0)

    # Exactly-constant baseline with no resolution evidence: a zero deviation is
    # normal, anything else is maximally anomalous.
    degenerate = ~(scale > 0)
    rz = rz.mask(degenerate & (dev == 0), 0.0)
    rz = rz.mask(degenerate & (dev != 0), np.sign(dev) * RZ_SATURATE)

    rz = rz.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rz = rz.clip(-RZ_SATURATE, RZ_SATURATE)
    return rz, med, scale


def features_for_iso(values, win=ISO_WIN):
    v = pd.Series(values).astype(float)
    diff  = v.diff().fillna(0)
    rmean = v.rolling(win, min_periods=1).mean()
    rstd  = v.rolling(win, min_periods=1).std().fillna(0)
    slope = rmean.diff().fillna(0)
    X = np.column_stack([v.values, diff.values, rmean.values, rstd.values, slope.values])
    return np.nan_to_num(X)


def auto_contamination(values, floor=0.001, ceil=0.05):
    s = pd.Series(values).astype(float)
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        rz, _, _ = robust_z(s, window=min(len(s), 24))
        est = (np.abs(rz) > 3.5).mean() * 0.7
    else:
        lo = q1 - 3 * iqr
        hi = q3 + 3 * iqr
        est = ((s < lo) | (s > hi)).mean() * 0.7
    return float(np.clip(max(est, ISO_BASE_CONTAM * 0.5), floor, ceil))


def isolation_forest_detection(values, win=ISO_WIN):
    X = features_for_iso(values, win=win)
    cont = auto_contamination(values)
    clf = IsolationForest(
        n_estimators=ISO_N_ESTIMATORS,
        contamination=cont,
        random_state=ISO_RANDOM_STATE,
        max_samples='auto',
        bootstrap=False,
        n_jobs=-1
    )
    labels = clf.fit_predict(X)  # -1 outlier, 1 inlier
    return labels == -1, cont


def roc_sliding(series, timestamps):
    """
    Rate-of-change channel: |dv/dt| in engineering units per second.

    Replaces the former sliding-DTW channel, which was provably just a first
    difference plus an echo -- see the ROC block in Tunables for the derivation.

    Timestamp hazard
    ----------------
    This telemetry is report-by-exception with 1-second resolution, and
    duplicate timestamps do occur. A naive dv/dt would divide by zero on every
    duplicate and flag it as an infinite-rate spike, so non-positive intervals
    are dropped rather than divided by. Backwards intervals (out-of-order rows,
    or an RTU clock stepping at NTP sync) are treated the same way.

    Returns (rate, flags), both length n, with rate[0] = NaN.
    """
    v = np.asarray(series, dtype=float)
    n = len(v)
    rate = np.full(n, np.nan)
    if n < 2:
        return rate, np.zeros(n, dtype=bool)

    # Use .dt.total_seconds() rather than casting to int64: the int64
    # representation carries whatever unit pandas inferred (s / ms / us / ns),
    # so dividing by 1e9 silently mis-scales the rate by orders of magnitude.
    # reset_index guards against the caller's non-trivial index misaligning diff().
    ts = pd.to_datetime(pd.Series(timestamps).reset_index(drop=True))
    dv = np.diff(v)
    dt = ts.diff().dt.total_seconds().to_numpy()[1:]
    # Guard: non-positive intervals carry no rate information.
    dt = np.where(dt > 0, dt, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate[1:] = np.abs(dv / dt)
    rate[~np.isfinite(rate)] = np.nan

    rs = pd.Series(rate)
    med = rs.median(skipna=True)
    mad = (rs - med).abs().median(skipna=True)
    if pd.notna(med) and pd.notna(mad):
        thr = med + ROC_K * mad
        flags = (rs > thr).fillna(False).to_numpy()
    else:
        flags = np.zeros(n, dtype=bool)
    return rate, flags


def _runs(flags):
    idx = np.where(flags)[0]
    if len(idx) == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[splits + 1]]
    ends   = np.r_[idx[splits], idx[-1]]
    return list(zip(starts, ends))


def apply_run_length_filters(flags, min_len=3, merge_gap=2):
    f = np.asarray(flags, dtype=bool).copy()
    runs = _runs(f)
    for s, e in runs:
        if e - s + 1 < min_len:
            f[s:e + 1] = False
    runs = _runs(f)
    for (s1, e1), (s2, e2) in zip(runs, runs[1:]):
        if s2 - e1 - 1 <= merge_gap:
            f[e1 + 1:s2] = True
    return f


def apply_cooldown(flags, cooldown=6):
    f = np.asarray(flags, dtype=bool)
    out = f.copy()
    i, n = 0, len(f)
    while i < n:
        if out[i]:
            j, end = i + 1, i
            while j < n and out[j]:
                end = j
                j += 1
            out[end + 1:end + 1 + cooldown] = False
            i = end + cooldown + 1
        else:
            i += 1
    return out


def vote_and_smooth(z_flags, iso_flags, roc_flags, hard_overrides=None,
                    min_event_len=MIN_EVENT_LEN, merge_gap=MERGE_GAP,
                    cooldown=COOLDOWN):
    zf = np.asarray(z_flags, dtype=bool)
    if iso_flags is None:
        iso_flags = np.zeros_like(zf)
    if roc_flags is None:
        roc_flags = np.zeros_like(zf)
    if hard_overrides is None:
        hard_overrides = np.zeros_like(zf)
    votes = zf.astype(int) + iso_flags.astype(int) + roc_flags.astype(int)
    base = (votes >= 2) | np.asarray(hard_overrides, dtype=bool)
    filtered = apply_run_length_filters(base, min_len=min_event_len, merge_gap=merge_gap)
    cooled   = apply_cooldown(filtered, cooldown=cooldown)
    return cooled


def _is_step_transition(g, s, e,
                        window=STEP_WINDOW_SAMPLES,
                        min_window=STEP_WINDOW_MIN,
                        min_level_diff_pct=STEP_MIN_LEVEL_DIFF_PCT):
    """
    Return True iff the event at rows [s, e] is a step-shaped value change:
    the median value in a window BEFORE the event differs from the median
    value in a window AFTER the event by at least `min_level_diff_pct`
    (relative to the larger of the two medians).

    This deliberately does NOT require either side to be "steady" — per
    client direction (May/June 2026), ALL events where the value level
    shifts should be suppressed, including those during cycling pump
    operation where the running state is noisy.

    Brief transients that return to baseline (single spike up and back)
    naturally fail this test because their before/after medians are
    essentially the same — those are NOT suppressed.

    Window sizing: uses up to `window` samples on each side; if fewer are
    available (e.g. event near start/end of the series, like a daily
    counter resetting at the end of the pipeline window), uses whatever
    is available down to `min_window` samples. Only refuses to judge if
    there are fewer than `min_window` samples on either side.
    """
    n = len(g)
    samples_before = s              # available samples before the event
    samples_after  = n - e - 1      # available samples after the event
    if samples_before < min_window or samples_after < min_window:
        return False

    take_before = min(window, samples_before)
    take_after  = min(window, samples_after)

    before = g['CurrValue'].iloc[s - take_before : s].astype(float)
    after  = g['CurrValue'].iloc[e + 1 : e + 1 + take_after].astype(float)

    if before.isna().all() or after.isna().all():
        return False

    med_b = float(np.nanmedian(before))
    med_a = float(np.nanmedian(after))

    # Levels must differ by min_level_diff_pct relative to the larger of the two
    scale = max(abs(med_b), abs(med_a), 1e-6)
    level_diff_pct = abs(med_a - med_b) / scale
    return level_diff_pct >= min_level_diff_pct


def filter_events_by_impact(g, flags, suppress_steps=False,
                            min_event_len=MIN_EVENT_LEN,
                            step_window=STEP_WINDOW_SAMPLES,
                            step_window_min=STEP_WINDOW_MIN):
    flags = np.asarray(flags, dtype=bool)
    out = np.zeros_like(flags)
    runs = _runs(flags)
    if not runs:
        return flags
    med_abs = float(np.nanmedian(np.abs(g['CurrValue'])))
    med_abs = med_abs if med_abs > 1e-9 else 1.0
    for s, e in runs:
        # Edge / step-change suppression: only applied for equipment profiles
        # where suppress_steps=True (pumps/valves can cycle, so a clean step
        # transition is normal operation).
        if suppress_steps and _is_step_transition(g, s, e,
                                                  window=step_window,
                                                  min_window=step_window_min):
            continue  # skip — treat as a state transition, not a fault

        seg = g.iloc[s:e + 1]
        dur      = e - s + 1
        peak_rz  = float(np.nanmax(np.abs(seg.get('RZ', pd.Series([0])))))
        rel_jump = abs(seg['CurrValue'].iloc[-1] - seg['CurrValue'].iloc[0]) / med_abs
        roc_flag_frac = float(seg.get('ROC_Flag', pd.Series([False] * len(seg))).mean())
        keep = (
            (dur >= min_event_len) and
            ((peak_rz >= Z_EVENT_MIN) or
             (rel_jump >= REL_JUMP_MIN) or
             (roc_flag_frac >= ROC_EVENT_RATE))
        )
        if keep:
            out[s:e + 1] = True
    return out


def calibrate_flags_per_sensor(rz_abs, roc_rate, iso_flags, base_cont):
    if np.isfinite(rz_abs).any():
        qz = np.nanquantile(rz_abs, 1 - TARGET_POINT_RATE)
        BASE_MIN_Z = 2.5
        z_thr = max(BASE_MIN_Z, qz, MAD_Z_THRESH)
        z_flags = rz_abs > z_thr
    else:
        z_flags = np.zeros_like(rz_abs, dtype=bool)

    ds = np.asarray(roc_rate, dtype=float)
    if np.isfinite(ds).any():
        qd  = np.nanquantile(ds, 1 - TARGET_POINT_RATE)
        med = np.nanmedian(ds)
        mad = np.nanmedian(np.abs(ds - med))
        d_thr = max(qd, med + ROC_K * mad) if np.isfinite(med) and np.isfinite(mad) else qd
        d_flags = ds > d_thr
    else:
        d_flags = np.zeros_like(z_flags, dtype=bool)

    used_cont = float(
        min(
            max(base_cont if np.isfinite(base_cont) else ISO_BASE_CONTAM,
                TARGET_POINT_RATE / 2),
            MAX_POINT_RATE
        )
    )
    if iso_flags is None:
        i_flags = np.zeros_like(z_flags, dtype=bool)
    else:
        rate = float(np.nanmean(iso_flags))
        if rate > MAX_POINT_RATE:
            keep = int(np.ceil(MAX_POINT_RATE * len(iso_flags)))
            order = np.argsort(-rz_abs)
            sel = np.zeros_like(iso_flags, dtype=bool)
            sel[order[:keep]] = True
            i_flags = iso_flags & sel
        else:
            i_flags = iso_flags
    return z_flags, d_flags, i_flags, used_cont


def safe_plot_filename(equipment, description):
    safe = re.sub(r'[<>:"/\\|?*]', '_', f"{equipment}_{description}")[:140]
    return f"{safe}.png"


# ================= Outputs & plotting =================
def build_event_table(g, equipment, description):
    flags = g['Combined_Anomaly'].fillna(False).to_numpy(dtype=bool)
    rows = []
    for s, e in _runs(flags):
        seg = g.iloc[s:e + 1]
        rows.append({
            'Equipment': equipment,
            'Description': description,
            'Start_Time': seg['DateTime'].iloc[0],
            'End_Time':   seg['DateTime'].iloc[-1],
            'Duration_pts': len(seg),
            'First_Value':  float(seg['CurrValue'].iloc[0]),
            'Last_Value':   float(seg['CurrValue'].iloc[-1]),
            'Max_RZ':       float(seg['RZ'].abs().max(skipna=True)),
            'Z_Points':     int(seg['Z_Flag'].sum()),
            'ISO_Points':   int(seg['ISO_Flag'].sum()),
            'ROC_Points':   int(seg['ROC_Flag'].sum()),
            'Rule_Invalid_In_Event': bool(seg['Rule_Based_Invalid'].any()),
        })
    return pd.DataFrame(rows)


def build_sensor_summary(result_df: pd.DataFrame,
                         plot_paths=None,
                         output_dir=None) -> pd.DataFrame:
    """
    Per-sensor summary. Now also includes Longitude, Latitude, Location if present.
    """
    cols = [
        "Equipment", "Description",
        "Total_Points", "Anomaly_Points", "Anomaly_Pct",
        "Num_Events", "First_Anomaly_Time", "Last_Anomaly_Time",
        "Z_Points", "ISO_Points", "ROC_Points", "Rule_Invalid_Points",
        "Peak_RZ", "Max_ROC_Rate",
        "Mean_Value", "Median_Value",
        "Plot_Path",
        "Longitude", "Latitude", "Location",
    ]
    if result_df.empty:
        return pd.DataFrame(columns=cols)

    has_long = "Longitude" in result_df.columns
    has_lat  = "Latitude"  in result_df.columns
    has_loc  = "Location"  in result_df.columns

    rows = []
    for (equipment, description), grp in result_df.groupby(['Equipment', 'Description']):
        total = int(len(grp))
        anom_mask = grp['Combined_Anomaly'].fillna(False).to_numpy(dtype=bool)
        anom_pts = int(anom_mask.sum())
        if anom_pts == 0:
            continue

        key = (equipment, description)
        if plot_paths and key in plot_paths:
            plot_path_out = plot_paths[key]
        elif output_dir is not None:
            plot_path_out = (Path(output_dir) / safe_plot_filename(equipment, description)).as_posix()
        else:
            plot_path_out = ""

        runs = _runs(anom_mask)

        # Take coordinates/location from first row in the group (assumed constant per sensor)
        if has_long:
            longitude = float(grp['Longitude'].iloc[0])
        else:
            longitude = np.nan
        if has_lat:
            latitude = float(grp['Latitude'].iloc[0])
        else:
            latitude = np.nan
        if has_loc:
            location = str(grp['Location'].iloc[0])
        else:
            location = ""

        rows.append({
            "Equipment": equipment,
            "Description": description,
            "Total_Points": total,
            "Anomaly_Points": anom_pts,
            "Anomaly_Pct": round(100.0 * anom_pts / total, 2),
            "Num_Events": len(runs),
            "First_Anomaly_Time": grp.loc[anom_mask, 'DateTime'].min(),
            "Last_Anomaly_Time":  grp.loc[anom_mask, 'DateTime'].max(),
            "Z_Points":  int(grp.loc[anom_mask, 'Z_Flag'].sum()),
            "ISO_Points": int(grp.loc[anom_mask, 'ISO_Flag'].sum()),
            "ROC_Points": int(grp.loc[anom_mask, 'ROC_Flag'].sum()),
            # Physical range violations, reported separately: these are certain
            # faults, not statistical inferences, and an operator triaging the
            # alert needs to see that distinction immediately.
            "Rule_Invalid_Points": int(grp.loc[anom_mask, 'Rule_Based_Invalid'].sum()),
            "Peak_RZ":     float(np.nanmax(np.abs(grp.loc[anom_mask, 'RZ']))),
            "Max_ROC_Rate": float(np.nanmax(grp.loc[anom_mask, 'ROC_Rate'])),
            "Mean_Value":  float(np.nanmean(grp['CurrValue'])),
            "Median_Value": float(np.nanmedian(grp['CurrValue'])),
            "Plot_Path":   plot_path_out,
            "Longitude":   longitude,
            "Latitude":    latitude,
            "Location":    location,
        })

    summary = pd.DataFrame(rows, columns=cols)
    if summary.empty:
        return pd.DataFrame(columns=cols)

    return summary.sort_values(by='Anomaly_Points', ascending=False).reset_index(drop=True)

def build_sensor_locations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a table of unique sensors with their Location / Latitude / Longitude
    from the time-filtered dataframe `df`.
    """
    cols = ["Equipment", "Description", "Longitude", "Latitude", "Location"]
    has_long = "Longitude" in df.columns
    has_lat  = "Latitude"  in df.columns
    has_loc  = "Location"  in df.columns

    if not (has_long or has_lat or has_loc):
        # No geo info at all
        return pd.DataFrame(columns=cols)

    rows = []
    for (equipment, description), grp in df.groupby(["Equipment", "Description"]):
        # Take first row as representative of that sensor
        if has_long:
            try:
                longitude = float(grp["Longitude"].iloc[0])
            except Exception:
                longitude = np.nan
        else:
            longitude = np.nan

        if has_lat:
            try:
                latitude = float(grp["Latitude"].iloc[0])
            except Exception:
                latitude = np.nan
        else:
            latitude = np.nan

        if has_loc:
            location = str(grp["Location"].iloc[0])
        else:
            location = ""

        rows.append({
            "Equipment":  equipment,
            "Description": description,
            "Longitude":  longitude,
            "Latitude":   latitude,
            "Location":   location,
        })

    loc_df = pd.DataFrame(rows, columns=cols)
    return loc_df


def plot_sensor_location_map(all_locations: pd.DataFrame,
                             abnormal_summary: pd.DataFrame,
                             output_dir: str):
    """
    Plot all sensors as points in (Longitude, Latitude) space,
    and highlight abnormal sensors (from abnormal_summary) on top.
    Abnormal sensors are also labelled with Location + Description.
    """
    if all_locations.empty:
        print("[MAP] No sensor locations found, skip location map.", flush=True)
        return None

    # Clean up numeric coords
    loc = all_locations.copy()
    loc["Longitude"] = pd.to_numeric(loc["Longitude"], errors="coerce")
    loc["Latitude"]  = pd.to_numeric(loc["Latitude"], errors="coerce")
    loc = loc.dropna(subset=["Longitude", "Latitude"])
    if loc.empty:
        print("[MAP] All sensor coordinates are NaN, skip location map.", flush=True)
        return None

    plt.figure(figsize=(8, 8))

    # Base: all sensors
    plt.scatter(
        loc["Longitude"], loc["Latitude"],
        alpha=0.4, s=30,
        label="All sensors"
    )

    # Overlay abnormal sensors + labels
    if not abnormal_summary.empty:
        abnormal_keys = set(zip(abnormal_summary["Equipment"],
                                abnormal_summary["Description"]))
        mask_ab = loc.apply(
            lambda r: (r["Equipment"], r["Description"]) in abnormal_keys,
            axis=1
        )
        ab = loc[mask_ab]
        if not ab.empty:
            plt.scatter(
                ab["Longitude"], ab["Latitude"],
                s=80, marker="o",
                edgecolors="red", linewidths=1.5,
                label="Abnormal sensors"
            )

            # ---- NEW: label each abnormal sensor with Location + Description ----
            for _, row in ab.iterrows():
                label = f"{row.get('Location','')}\n{row.get('Description','')}"
                plt.annotate(
                    label,
                    (row["Longitude"], row["Latitude"]),
                    textcoords="offset points",
                    xytext=(3, 3),      # small offset so text doesn't sit exactly on the point
                    ha="left",
                    fontsize=8
                )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Sensor Locations (abnormal sensors highlighted & labelled)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    out_path = Path(output_dir) / "sensor_location_map.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    print(f"[PLOT] {out_path}", flush=True)
    return out_path.as_posix()



def cluster_anomalous_sensors(abnormal_summary: pd.DataFrame,
                              output_dir: str,
                              n_clusters: int = 3):
    """
    Cluster sensors (Equipment, Description) based on anomaly stats and
    plot a 2D scatter (Anomaly_Pct vs Peak_RZ) colored by cluster.
    """
    if abnormal_summary.empty:
        print("[CLUSTER] No abnormal sensors to cluster.", flush=True)
        return None

    n_clusters = min(n_clusters, len(abnormal_summary))
    if n_clusters <= 1:
        print("[CLUSTER] Only one abnormal sensor, skipping KMeans.", flush=True)
        return None

    feature_cols = [
        "Anomaly_Pct",
        "Peak_RZ",
        "Max_ROC_Rate",
        "Num_Events",
        "Z_Points",
        "ISO_Points",
        "ROC_Points",
    ]

    X = abnormal_summary[feature_cols].fillna(0.0).to_numpy(dtype=float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = km.fit_predict(X_scaled)

    abnormal_summary = abnormal_summary.copy()
    abnormal_summary["Cluster"] = clusters

    out_csv = Path("output_csv") / "abnormal_sensor_clusters.csv"
    abnormal_summary.to_csv(out_csv, index=False)
    print(f"[WRITE] {out_csv} (clustered sensors)", flush=True)

    plt.figure(figsize=(10, 6))
    for c in range(n_clusters):
        mask = abnormal_summary["Cluster"] == c
        plt.scatter(
            abnormal_summary.loc[mask, "Anomaly_Pct"],
            abnormal_summary.loc[mask, "Peak_RZ"],
            label=f"Cluster {c}", alpha=0.7
        )

    plt.xlabel("Anomaly_Pct")
    plt.ylabel("Peak_RZ")
    plt.title("Clusters of Abnormal Sensors")
    plt.grid(True, alpha=0.3)
    plt.legend()

    cluster_plot_path = Path(output_dir) / "sensor_clusters.png"
    plt.tight_layout()
    plt.savefig(cluster_plot_path)
    plt.close()

    print(f"[PLOT] {cluster_plot_path}", flush=True)
    return cluster_plot_path.as_posix()


def cluster_anomaly_points(abnormal_points: pd.DataFrame,
                           output_dir: str,
                           n_clusters: int = 3):
    """
    Cluster individual anomaly points based on RZ, ROC_Rate, and CurrValue.
    Produces a scatter of RZ vs ROC_Rate colored by cluster.
    """
    if abnormal_points.empty:
        print("[CLUSTER] No abnormal points to cluster.", flush=True)
        return None

    for col in ["RZ", "ROC_Rate", "CurrValue"]:
        if col not in abnormal_points.columns:
            print(f"[CLUSTER] Missing column '{col}' in abnormal_points, skip point clusters.", flush=True)
            return None

    ap = abnormal_points[["RZ", "ROC_Rate", "CurrValue"]].copy()
    ap = ap.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = ap.to_numpy(dtype=float)

    n_clusters = min(n_clusters, len(ap))
    if n_clusters <= 1:
        print("[CLUSTER] Not enough points for KMeans.", flush=True)
        return None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = km.fit_predict(X_scaled)

    abnormal_points = abnormal_points.copy()
    abnormal_points["PointCluster"] = clusters

    out_csv = Path("output_csv") / "abnormal_points_clusters.csv"
    abnormal_points.to_csv(out_csv, index=False)
    print(f"[WRITE] {out_csv} (clustered points)", flush=True)

    plt.figure(figsize=(10, 6))
    for c in range(n_clusters):
        mask = abnormal_points["PointCluster"] == c
        plt.scatter(
            abnormal_points.loc[mask, "RZ"],
            abnormal_points.loc[mask, "ROC_Rate"],
            label=f"Cluster {c}", alpha=0.7
        )

    plt.xlabel("RZ")
    plt.ylabel("ROC_Rate")
    plt.title("Clusters of Anomaly Points")
    plt.grid(True, alpha=0.3)
    plt.legend()

    cluster_plot_path = Path(output_dir) / "anomaly_point_clusters.png"
    plt.tight_layout()
    plt.savefig(cluster_plot_path)
    plt.close()

    print(f"[PLOT] {cluster_plot_path}", flush=True)
    return cluster_plot_path.as_posix()


def plot_sensor(g, equipment, description, output_dir):
    plt.figure(figsize=(14, 6))
    plt.plot(g['DateTime'], g['CurrValue'], label='CurrValue', alpha=0.65)

    zmask = g['Z_Flag'].fillna(False).to_numpy(dtype=bool)
    imask = g['ISO_Flag'].fillna(False).to_numpy(dtype=bool)
    dmask = g['ROC_Flag'].fillna(False).to_numpy(dtype=bool)

    if zmask.any():
        zpts = g.loc[zmask]
        plt.scatter(zpts['DateTime'], zpts['CurrValue'], marker='o', s=18, label='Robust-Z')
    if imask.any():
        ipts = g.loc[imask]
        plt.scatter(ipts['DateTime'], ipts['CurrValue'], marker='s', s=22, label='IsolationForest')
    if dmask.any():
        dpts = g.loc[dmask]
        plt.scatter(dpts['DateTime'], dpts['CurrValue'], marker='x', s=28, label='Rate-of-change')

    event_mask = g['Combined_Anomaly'].fillna(False).to_numpy(dtype=bool)
    runs = _runs(event_mask)
    for k, (s, e) in enumerate(runs):
        plt.axvspan(g['DateTime'].iloc[s], g['DateTime'].iloc[e],
                    alpha=0.15, label='Event' if k == 0 else None)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gcf().autofmt_xdate()
    plt.grid(True, alpha=0.3)
    plt.title(f"{equipment} | {description}")
    plt.xlabel("Time")
    plt.ylabel("CurrValue")
    plt.legend()

    filename = safe_plot_filename(equipment, description)
    path = Path(output_dir) / filename
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    posix_path = path.as_posix()
    if VERBOSE_LEVEL >= 2:
        print(f"      Saved plot → {posix_path}", flush=True)
    return posix_path


def plot_sensor_clusters(g, equipment, description, output_dir, n_clusters=3):
    """
    For a single sensor (one Equipment + Description group):

    - Build features [CurrValue, RZ, ROC_Rate] for each timestamp.
    - Cluster ONLY normal points (Combined_Anomaly == False) with KMeans.
    - Project all points (normal + abnormal) into 2D via PCA.
    - Plot:
        * normal points colored by cluster
        * abnormal points highlighted with a different marker/color
    """
    required_cols = ["CurrValue", "RZ", "ROC_Rate", "Combined_Anomaly"]
    for col in required_cols:
        if col not in g.columns:
            print(f"[CLUSTER-PLOT] Missing column '{col}' for {equipment} | {description}, skip.", flush=True)
            return ""

    feats = g[["CurrValue", "RZ", "ROC_Rate"]].copy()
    feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = feats.to_numpy(dtype=float)

    anom_mask   = g["Combined_Anomaly"].fillna(False).to_numpy(dtype=bool)
    normal_mask = ~anom_mask

    if normal_mask.sum() < 2:
        print(f"[CLUSTER-PLOT] Not enough normal points for {equipment} | {description}, skip.", flush=True)
        return ""

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)

    k = min(n_clusters, normal_mask.sum())
    if k <= 1:
        print(f"[CLUSTER-PLOT] Only one cluster possible for {equipment} | {description}, plotting without KMeans.", flush=True)
        clusters = np.zeros(len(g), dtype=int)
    else:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        km.fit(X_2d[normal_mask])
        clusters = km.predict(X_2d)

    plt.figure(figsize=(8, 6))

    for c in np.unique(clusters[normal_mask]):
        mask = (clusters == c) & normal_mask
        plt.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            s=20, alpha=0.7,
            label=f"Cluster {c} (normal)"
        )

    if anom_mask.any():
        plt.scatter(
            X_2d[anom_mask, 0], X_2d[anom_mask, 1],
            s=50, marker="x", color="red", linewidths=1.5,
            label="Anomaly points"
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"Clusters of normal points with anomalies highlighted\n{equipment} | {description}")
    plt.grid(True, alpha=0.3)
    plt.legend()

    base_name = safe_plot_filename(equipment, description).replace(".png", "_clusters.png")
    path = Path(output_dir) / base_name
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    posix_path = path.as_posix()
    if VERBOSE_LEVEL >= 2:
        print(f"      Saved cluster plot → {posix_path}", flush=True)
    return posix_path


# ================= Main with progress logging =================
def equipment_aware_anomaly_pipeline(filepath, output_dir=PLOTS_DIR):
    all_groups = []
    events_all = []
    abnormal_sensor_count = 0
    total_events_count = 0

    t0 = time.time()
    print(f"[INFO] Loading data from: {filepath}", flush=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("output_csv", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    with open(LOG_FILE, "w", encoding="utf-8") as lf:
        lf.write("detector errors\n")

    df = pd.read_csv(filepath, parse_dates=['DateTime']).sort_values('DateTime')
    if TIME_WINDOW_HOURS:
        end_ts = df['DateTime'].max()
        start_ts = end_ts - pd.Timedelta(hours=TIME_WINDOW_HOURS)
        df = df[df['DateTime'].between(start_ts, end_ts)]
        print(f"[INFO] Time-filtered to last {TIME_WINDOW_HOURS}h: {len(df)} rows ({start_ts} -> {end_ts})", flush=True)
    else:
        print(f"[INFO] Total rows: {len(df)}", flush=True)

    # Filter out unanalysed Equipment categories per client direction.
    # See SKIP_UNCATEGORIZED_EQUIPMENT in the Tunables section.
    if SKIP_UNCATEGORIZED_EQUIPMENT:
        before = len(df)
        skip_mask = df['Equipment'].isin(SKIP_UNCATEGORIZED_EQUIPMENT)
        skipped_sensors = df.loc[skip_mask, ['Equipment', 'Description']].drop_duplicates()
        df = df[~skip_mask]
        after = len(df)
        print(
            f"[INFO] Equipment filter: dropped {before - after} rows "
            f"({len(skipped_sensors)} distinct sensors) in categories "
            f"{sorted(SKIP_UNCATEGORIZED_EQUIPMENT)}",
            flush=True,
        )

    grouped = df.groupby(['Equipment', 'Description'])
    total_pairs = len(grouped)
    print(f"[INFO] Processing {total_pairs} (Equipment, Description) pairs...", flush=True)

    start_loop = time.time()
    for idx, ((equipment, description), g) in enumerate(grouped, start=1):
        sensor_start = time.time()
        try:
            g = g.sort_values('DateTime').reset_index(drop=True).copy()
            n_rows = len(g)
            # Windows are derived from THIS sensor's reporting rate, so the
            # same gate means the same duration on every sensor.
            win = compute_sensor_windows(g['DateTime'])
            if n_rows < max(MIN_SENSOR_POINTS, win['roll_win_z'] + 5):
                if VERBOSE_LEVEL >= 2:
                    print(f"[{idx}/{total_pairs}] Skip {equipment} | {description}: not enough points ({n_rows}).", flush=True)
                continue

            profile = DETECTOR_PROFILE.get(equipment, DEFAULT_PROFILE)
            use_z   = profile.get('use_z', True)
            use_iso = profile.get('use_iso', True)
            use_roc = profile.get('use_roc', True)

            if VERBOSE_LEVEL >= 1:
                print(f"[{idx}/{total_pairs}] {equipment} | {description} (rows={n_rows}, "
                      f"dt={win['median_dt_s']:.0f}s, z_win={win['roll_win_z']}) "
                      f"[use_z={use_z}, use_iso={use_iso}, use_roc={use_roc}]...", flush=True)

            values = g['CurrValue'].astype(float).values

            # Rule invalids
            # Deadband from this sensor's own noise -- see range_tolerance.
            sensor_resolution = resolution_estimate(values)
            range_tol = range_tolerance(values, sensor_resolution)
            g['Rule_Based_Invalid'] = pd.Series(
                range_violation_mask(equipment, values, tolerance=range_tol),
                index=g.index,
            )

            # Robust Z is always computed (used for diagnostics even if we don't vote with it)
            rz, _, _ = robust_z(values, window=win['roll_win_z'], resolution=sensor_resolution)
            g['RZ'] = rz

            # Isolation Forest (optional per profile)
            iso_flags = None
            iso_cont  = np.nan
            if use_iso and np.nanstd(values) > 1e-9:
                iso_flags, iso_cont = isolation_forest_detection(values, win=win['iso_win'])
            g['ISO_Flag'] = pd.Series(
                iso_flags if iso_flags is not None else np.zeros(n_rows, dtype=bool),
                index=g.index
            )
            g['ISO_Cont'] = iso_cont

            # Rate-of-change (optional per profile)
            if use_roc:
                roc_rate, roc_flags = roc_sliding(values, g['DateTime'])
            else:
                roc_rate  = np.full(n_rows, np.nan)
                roc_flags = np.zeros(n_rows, dtype=bool)
            g['ROC_Rate'] = roc_rate
            g['ROC_Flag'] = pd.Series(roc_flags, index=g.index)

            # Calibrate + vote
            rz_abs = np.abs(rz)
            zF, dF, iF, used_cont = calibrate_flags_per_sensor(rz_abs, roc_rate, iso_flags, iso_cont)

            # Enforce per-equipment profile on flags
            if not use_z:
                zF = np.zeros_like(zF, dtype=bool)
            if not use_roc:
                dF = np.zeros_like(dF, dtype=bool)
            if not use_iso:
                iF = np.zeros_like(iF, dtype=bool)

            g['Z_Flag']   = pd.Series(zF, index=g.index)
            g['ROC_Flag'] = pd.Series(dF, index=g.index)
            g['ISO_Flag'] = pd.Series(iF, index=g.index)
            g['ISO_Cont'] = used_cont

            voted = vote_and_smooth(
                zF, iF, dF,
                hard_overrides=(rz_abs > MAD_Z_HARD),
                min_event_len=win['min_event_len'],
                merge_gap=win['merge_gap'],
                cooldown=win['cooldown'],
            )

            # Physically impossible readings used to be SUBTRACTED here
            # (`voted & ~Rule_Based_Invalid`), so a pressure sensor reporting
            # -5 bar was silently dropped instead of alerted. That is backwards:
            # a reading outside the instrument's physical range is the single
            # most certain fault signal available -- it needs no statistical
            # inference at all.
            #
            # They are still excluded from the statistical channels (so a
            # garbage value cannot also inflate the vote), then OR-ed back in
            # AFTER event gating. The gating exists to filter weak statistical
            # evidence; a physical violation is not weak evidence, and a single
            # out-of-range sample is a real fault even though it is shorter
            # than MIN_EVENT_LEN.
            invalid = g['Rule_Based_Invalid'].fillna(False).to_numpy(dtype=bool)
            combined = np.asarray(voted, dtype=bool) & (~invalid)
            suppress_steps_flag = bool(profile.get('suppress_steps', False))
            gated = filter_events_by_impact(
                g, combined,
                suppress_steps=suppress_steps_flag,
                min_event_len=win['min_event_len'],
                step_window=win['step_window'],
                step_window_min=win['step_window_min'],
            )
            g['Combined_Anomaly'] = pd.Series(
                np.asarray(gated, dtype=bool) | invalid,
                index=g.index,
            )

            # Events for this sensor
            ev_tbl = build_event_table(g, equipment, description)
            if not ev_tbl.empty:
                events_all.append(ev_tbl)
                num_events = len(ev_tbl)
            else:
                num_events = 0

            comb_pts = int(g['Combined_Anomaly'].sum())
            z_pts    = int(g['Z_Flag'].sum())
            iso_pts  = int(g['ISO_Flag'].sum())
            roc_pts  = int(g['ROC_Flag'].sum())
            inv_pts  = int(g['Rule_Based_Invalid'].sum())

            if comb_pts > 0:
                abnormal_sensor_count += 1
                total_events_count += num_events

            sensor_time = time.time() - sensor_start
            if VERBOSE_LEVEL >= 1:
                print(f"   -> Z:{z_pts} ISO:{iso_pts} ROC:{roc_pts} INVALID:{inv_pts} "
                      f"COMB:{comb_pts} events:{num_events} "
                      f"time:{sensor_time:.2f}s", flush=True)

            if VERBOSE_LEVEL >= 1 and (idx % PRINT_EVERY == 0):
                avg = (time.time() - start_loop) / idx
                eta = avg * (total_pairs - idx)
                print(f"[PROGRESS] {idx}/{total_pairs} ({idx/total_pairs:.1%}), "
                      f"avg/sensor {avg:.2f}s, ETA ~ {eta/60:.1f} min", flush=True)

            all_groups.append(g)

        except Exception as e:
            err = f"[ERROR] ({idx}/{total_pairs}) {equipment} | {description}: {e}"
            print(err, flush=True)
            with open(LOG_FILE, "a", encoding="utf-8") as lf:
                lf.write(err + "\n" + traceback.format_exc() + "\n")

    # ======= Outputs with progress prints =======
    result_df = pd.concat(all_groups, ignore_index=True) if all_groups else pd.DataFrame()

    if not result_df.empty:
        orig_cols = [c for c in df.columns]
        add_cols  = [
            'RZ', 'Z_Flag',
            'ISO_Flag', 'ISO_Cont',
            'ROC_Rate', 'ROC_Flag',
            'Combined_Anomaly', 'Rule_Based_Invalid'
        ]
        add_cols  = [c for c in add_cols if c not in orig_cols]

        # Build provisional summary for ALL sensors with anomalies (no plots yet)
        provisional_summary = build_sensor_summary(result_df, plot_paths=None, output_dir=None)

        if provisional_summary.empty:
            print("[WARN] No sensors with anomalies found.", flush=True)
        else:
            # Compute how many detection families contributed to each sensor
            tmp = provisional_summary.copy()
            tmp["Methods_Used"] = (
                (tmp["Z_Points"]   > 0).astype(int) +
                (tmp["ISO_Points"] > 0).astype(int) +
                (tmp["ROC_Points"] > 0).astype(int)
            )

            # Filter: require at least REQUIRED_METHODS_PER_SENSOR and at least 1 event
            tmp = tmp[tmp["Methods_Used"] >= REQUIRED_METHODS_PER_SENSOR]
            tmp = tmp[tmp["Num_Events"] > 0]

            # Sort by severity and pick top MAX_ABNORMAL_SENSORS
            tmp = tmp.sort_values(
                by=["Methods_Used", "Anomaly_Pct", "Peak_RZ"],
                ascending=[False, False, False]
            )
            top_summary = tmp.head(MAX_ABNORMAL_SENSORS).copy()

            if top_summary.empty:
                print("[WARN] No sensors satisfy multi-method criteria. No abnormal_sensor.csv will be written.", flush=True)
            else:
                # Set of selected sensors
                top_keys = set(zip(top_summary["Equipment"], top_summary["Description"]))

                # =========================================================
                # REORDERED: decide suppression FIRST, then plot only the
                # sensors that will actually alert. Previously we plotted
                # every top sensor and only afterwards marked some as
                # Suppressed — wasting disk space and runtime on plots
                # that nobody would ever see. Now plots are generated
                # only for non-suppressed sensors. Suppressed rows keep
                # an empty Plot_Path (auditable via Suppression_Reason).
                # =========================================================

                # Step 1: build a provisional summary WITHOUT plots so we
                # have something annotate_suppression can work on. It needs
                # First_Anomaly_Time / Last_Anomaly_Time / Description.
                provisional = build_sensor_summary(result_df, plot_paths=None, output_dir=output_dir)
                key_idx_prov = list(zip(provisional["Equipment"], provisional["Description"]))
                mask_prov = [k in top_keys for k in key_idx_prov]
                provisional = provisional[mask_prov].reset_index(drop=True)

                # Step 2: annotate suppression on the provisional summary.
                provisional = annotate_suppression(provisional)
                n_suppressed = int(provisional["Suppressed"].sum())
                n_total = len(provisional)
                print(
                    f"[SUPPRESS] {n_suppressed}/{n_total} sensors flagged as fan-out "
                    f"(plots skipped for suppressed sensors)",
                    flush=True,
                )

                # Step 3: figure out which sensors actually need plots.
                suppressed_keys = set(
                    zip(
                        provisional.loc[provisional["Suppressed"], "Equipment"],
                        provisional.loc[provisional["Suppressed"], "Description"],
                    )
                )
                plot_keys = top_keys - suppressed_keys

                # Step 4: plot only the survivors.
                plot_paths = {}
                for (equipment, description) in plot_keys:
                    g_sensor = result_df[
                        (result_df["Equipment"] == equipment) &
                        (result_df["Description"] == description)
                    ].copy()

                    plot_path = ""
                    if PLOT_PER_SENSOR_TIMESERIES:
                        plot_path = plot_sensor(g_sensor, equipment, description, output_dir)
                    plot_paths[(equipment, description)] = plot_path

                # Suppressed sensors get an empty plot path (preserves CSV column).
                for key in suppressed_keys:
                    plot_paths[key] = ""

                # Step 5: rebuild the final summary with plot paths now filled in,
                # then re-apply suppression so the Suppressed/Suppression_Reason
                # columns carry through to the final CSV.
                final_summary = build_sensor_summary(result_df, plot_paths=plot_paths, output_dir=output_dir)
                key_idx = list(zip(final_summary["Equipment"], final_summary["Description"]))
                mask_final = [k in top_keys for k in key_idx]
                final_summary = final_summary[mask_final].reset_index(drop=True)
                final_summary = annotate_suppression(final_summary)

                # Per-point anomalies (restricted to selected sensors)
                abnormal_points = result_df[result_df['Combined_Anomaly']].copy()
                top_keys_df = pd.DataFrame(list(top_keys), columns=["Equipment", "Description"])
                abnormal_points = abnormal_points.merge(top_keys_df, on=["Equipment", "Description"], how="inner")
                abnormal_points = abnormal_points[orig_cols + add_cols]

                abnormal_points.to_csv("output_csv/abnormal_points.csv", index=False)
                print(f"[WRITE] output_csv/abnormal_points.csv  (rows={len(abnormal_points)})", flush=True)

                final_summary.to_csv("output_csv/abnormal_sensor.csv", index=False)
                print(f"[WRITE] output_csv/abnormal_sensor.csv  (sensors={len(final_summary)})", flush=True)

                # Events (restricted to selected sensors)
                if events_all:
                    events_df = pd.concat(events_all, ignore_index=True).sort_values('Start_Time')
                    events_df = events_df.merge(top_keys_df, on=["Equipment", "Description"], how="inner")
                    events_df.to_csv("output_csv/abnormal_events.csv", index=False)
                    print(f"[WRITE] output_csv/abnormal_events.csv (events={len(events_df)})", flush=True)

                # Clustering visualizations only on selected sensors/points
                                # Location-based “cluster” plot:
                # all sensors vs abnormal sensors highlighted
                all_locations = build_sensor_locations(df)
                plot_sensor_location_map(all_locations, final_summary, output_dir)

    else:
        print("[WARN] No groups processed into results dataframe.", flush=True)

    total_time = time.time() - t0
    print(f"[DONE] Abnormal sensors (any anomalies): {abnormal_sensor_count} | "
          f"Total events: {total_events_count} | Elapsed: {total_time:.2f}s", flush=True)


# ================= Entrypoint =================
if __name__ == '__main__':
    print(f"[START] Detector running (python {sys.version.split()[0]})", flush=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs("output_csv", exist_ok=True)
    equipment_aware_anomaly_pipeline("processed/data.csv", output_dir=PLOTS_DIR)