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
from fastdtw import fastdtw
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

# Isolation Forest
ISO_BASE_CONTAM   = 0.01
ISO_N_ESTIMATORS  = 200
ISO_WIN           = 12
ISO_RANDOM_STATE  = 42

# DTW
DTW_REF_WIN = 24
DTW_K       = 6.0

# Per-sensor anomaly budget
TARGET_POINT_RATE = 0.003
MAX_POINT_RATE    = 0.01

# Event gating / smoothing
MIN_EVENT_LEN = 6
MERGE_GAP     = 3
COOLDOWN      = 12
Z_EVENT_MIN   = 5.5
REL_JUMP_MIN  = 0.10
DTW_EVENT_RATE= 0.20

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
# produces sharp transitions that the detectors (Z, ISO, DTW) all flag
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
    'Pressure':         dict(use_z=True,  use_iso=True,  use_dtw=True,  suppress_steps=True),
    'Flowrate':         dict(use_z=True,  use_iso=True,  use_dtw=True,  suppress_steps=True),
    'Conductivity':     dict(use_z=True,  use_iso=False, use_dtw=True,  suppress_steps=True),
    'Voltage':          dict(use_z=True,  use_iso=True,  use_dtw=False, suppress_steps=True),
    'Dissolved Oxygen': dict(use_z=True,  use_iso=True,  use_dtw=False, suppress_steps=True),
    'Temperature':      dict(use_z=True,  use_iso=False, use_dtw=False, suppress_steps=True),
    'LevelSensor':      dict(use_z=True,  use_iso=False, use_dtw=False, suppress_steps=True),
}
DEFAULT_PROFILE = dict(use_z=True, use_iso=True, use_dtw=True, suppress_steps=True)

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
def is_rule_invalid(equipment, value):
    if equipment == 'Temperature':       return value < 0 or value > 60
    elif equipment == 'Flowrate':        return value < 0 or value > 2000
    elif equipment == 'Conductivity':    return value < 0 or value > 50000
    elif equipment == 'Pressure':        return value < 0 or value > 20
    elif equipment == 'Voltage':         return value < 0 or value > 500
    elif equipment == 'Dissolved Oxygen':return value < 0 or value > 20
    elif equipment == 'LevelSensor':     return value < 0 or value > 100
    return False


def robust_z(series, window=ROLL_WIN_Z):
    s = pd.Series(series).astype(float)
    med = s.rolling(window, min_periods=1, center=True).median()
    mad = (s - med).abs().rolling(window, min_periods=1, center=True).median()
    mad = mad.replace(0, np.nan)
    rz = 1.4826 * (s - med) / mad
    return rz.fillna(0.0), med, mad


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


def isolation_forest_detection(values):
    X = features_for_iso(values, win=ISO_WIN)
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


def dtw_sliding(series, ref_win=DTW_REF_WIN):
    v = np.asarray(series).astype(float)
    n = len(v)
    dists = np.full(n, np.nan)
    if n < ref_win * 2:
        return dists, np.zeros(n, dtype=bool)
    for i in range(ref_win, n):
        ref = v[i-ref_win:i]
        tgt = v[i-ref_win+1:i+1]
        dist, _ = fastdtw(ref, tgt, dist=lambda a, b: abs(a - b))
        dists[i] = dist
    ds = pd.Series(dists)
    med = ds.median(skipna=True)
    mad = (ds - med).abs().median(skipna=True)
    thr = med + DTW_K * mad if pd.notna(med) and pd.notna(mad) else np.nan
    flags = (ds > thr).fillna(False).values if pd.notna(thr) else np.zeros(n, dtype=bool)
    return dists, flags


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


def vote_and_smooth(z_flags, iso_flags, dtw_flags, hard_overrides=None):
    zf = np.asarray(z_flags, dtype=bool)
    if iso_flags is None:
        iso_flags = np.zeros_like(zf)
    if dtw_flags is None:
        dtw_flags = np.zeros_like(zf)
    if hard_overrides is None:
        hard_overrides = np.zeros_like(zf)
    votes = zf.astype(int) + iso_flags.astype(int) + dtw_flags.astype(int)
    base = (votes >= 2) | np.asarray(hard_overrides, dtype=bool)
    filtered = apply_run_length_filters(base, min_len=MIN_EVENT_LEN, merge_gap=MERGE_GAP)
    cooled   = apply_cooldown(filtered, cooldown=COOLDOWN)
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


def filter_events_by_impact(g, flags, suppress_steps=False):
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
        if suppress_steps and _is_step_transition(g, s, e):
            continue  # skip — treat as a state transition, not a fault

        seg = g.iloc[s:e + 1]
        dur      = e - s + 1
        peak_rz  = float(np.nanmax(np.abs(seg.get('RZ', pd.Series([0])))))
        rel_jump = abs(seg['CurrValue'].iloc[-1] - seg['CurrValue'].iloc[0]) / med_abs
        dtw_rate = float(seg.get('DTW_Flag', pd.Series([False] * len(seg))).mean())
        keep = (
            (dur >= MIN_EVENT_LEN) and
            ((peak_rz >= Z_EVENT_MIN) or
             (rel_jump >= REL_JUMP_MIN) or
             (dtw_rate >= DTW_EVENT_RATE))
        )
        if keep:
            out[s:e + 1] = True
    return out


def calibrate_flags_per_sensor(rz_abs, dtw_dist, iso_flags, base_cont):
    if np.isfinite(rz_abs).any():
        qz = np.nanquantile(rz_abs, 1 - TARGET_POINT_RATE)
        BASE_MIN_Z = 2.5
        z_thr = max(BASE_MIN_Z, qz, MAD_Z_THRESH)
        z_flags = rz_abs > z_thr
    else:
        z_flags = np.zeros_like(rz_abs, dtype=bool)

    ds = np.asarray(dtw_dist, dtype=float)
    if np.isfinite(ds).any():
        qd  = np.nanquantile(ds, 1 - TARGET_POINT_RATE)
        med = np.nanmedian(ds)
        mad = np.nanmedian(np.abs(ds - med))
        d_thr = max(qd, med + DTW_K * mad) if np.isfinite(med) and np.isfinite(mad) else qd
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
            'DTW_Points':   int(seg['DTW_Flag'].sum()),
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
        "Z_Points", "ISO_Points", "DTW_Points",
        "Peak_RZ", "Max_DTW_Dist",
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
            "DTW_Points": int(grp.loc[anom_mask, 'DTW_Flag'].sum()),
            "Peak_RZ":     float(np.nanmax(np.abs(grp.loc[anom_mask, 'RZ']))),
            "Max_DTW_Dist": float(np.nanmax(grp.loc[anom_mask, 'DTW_Dist'])),
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
        "Max_DTW_Dist",
        "Num_Events",
        "Z_Points",
        "ISO_Points",
        "DTW_Points",
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
    Cluster individual anomaly points based on RZ, DTW_Dist, and CurrValue.
    Produces a scatter of RZ vs DTW_Dist colored by cluster.
    """
    if abnormal_points.empty:
        print("[CLUSTER] No abnormal points to cluster.", flush=True)
        return None

    for col in ["RZ", "DTW_Dist", "CurrValue"]:
        if col not in abnormal_points.columns:
            print(f"[CLUSTER] Missing column '{col}' in abnormal_points, skip point clusters.", flush=True)
            return None

    ap = abnormal_points[["RZ", "DTW_Dist", "CurrValue"]].copy()
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
            abnormal_points.loc[mask, "DTW_Dist"],
            label=f"Cluster {c}", alpha=0.7
        )

    plt.xlabel("RZ")
    plt.ylabel("DTW_Dist")
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
    dmask = g['DTW_Flag'].fillna(False).to_numpy(dtype=bool)

    if zmask.any():
        zpts = g.loc[zmask]
        plt.scatter(zpts['DateTime'], zpts['CurrValue'], marker='o', s=18, label='Robust-Z')
    if imask.any():
        ipts = g.loc[imask]
        plt.scatter(ipts['DateTime'], ipts['CurrValue'], marker='s', s=22, label='IsolationForest')
    if dmask.any():
        dpts = g.loc[dmask]
        plt.scatter(dpts['DateTime'], dpts['CurrValue'], marker='x', s=28, label='DTW')

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

    - Build features [CurrValue, RZ, DTW_Dist] for each timestamp.
    - Cluster ONLY normal points (Combined_Anomaly == False) with KMeans.
    - Project all points (normal + abnormal) into 2D via PCA.
    - Plot:
        * normal points colored by cluster
        * abnormal points highlighted with a different marker/color
    """
    required_cols = ["CurrValue", "RZ", "DTW_Dist", "Combined_Anomaly"]
    for col in required_cols:
        if col not in g.columns:
            print(f"[CLUSTER-PLOT] Missing column '{col}' for {equipment} | {description}, skip.", flush=True)
            return ""

    feats = g[["CurrValue", "RZ", "DTW_Dist"]].copy()
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
            if n_rows < max(DTW_REF_WIN * 2, ROLL_WIN_Z + 5):
                if VERBOSE_LEVEL >= 2:
                    print(f"[{idx}/{total_pairs}] Skip {equipment} | {description}: not enough points ({n_rows}).", flush=True)
                continue

            profile = DETECTOR_PROFILE.get(equipment, DEFAULT_PROFILE)
            use_z   = profile.get('use_z', True)
            use_iso = profile.get('use_iso', True)
            use_dtw = profile.get('use_dtw', True)

            if VERBOSE_LEVEL >= 1:
                print(f"[{idx}/{total_pairs}] {equipment} | {description} (rows={n_rows}) "
                      f"[use_z={use_z}, use_iso={use_iso}, use_dtw={use_dtw}]...", flush=True)

            values = g['CurrValue'].astype(float).values

            # Rule invalids
            g['Rule_Based_Invalid'] = g['CurrValue'].apply(lambda v: is_rule_invalid(equipment, v))

            # Robust Z is always computed (used for diagnostics even if we don't vote with it)
            rz, _, _ = robust_z(values, window=min(ROLL_WIN_Z, max(5, len(values) // 10)))
            g['RZ'] = rz

            # Isolation Forest (optional per profile)
            iso_flags = None
            iso_cont  = np.nan
            if use_iso and np.nanstd(values) > 1e-9:
                iso_flags, iso_cont = isolation_forest_detection(values)
            g['ISO_Flag'] = pd.Series(
                iso_flags if iso_flags is not None else np.zeros(n_rows, dtype=bool),
                index=g.index
            )
            g['ISO_Cont'] = iso_cont

            # DTW (optional per profile)
            if use_dtw:
                dtw_dist, dtw_flags = dtw_sliding(values, ref_win=min(DTW_REF_WIN, max(6, len(values) // 12)))
            else:
                dtw_dist  = np.full(n_rows, np.nan)
                dtw_flags = np.zeros(n_rows, dtype=bool)
            g['DTW_Dist'] = dtw_dist
            g['DTW_Flag'] = pd.Series(dtw_flags, index=g.index)

            # Calibrate + vote
            rz_abs = np.abs(rz)
            zF, dF, iF, used_cont = calibrate_flags_per_sensor(rz_abs, dtw_dist, iso_flags, iso_cont)

            # Enforce per-equipment profile on flags
            if not use_z:
                zF = np.zeros_like(zF, dtype=bool)
            if not use_dtw:
                dF = np.zeros_like(dF, dtype=bool)
            if not use_iso:
                iF = np.zeros_like(iF, dtype=bool)

            g['Z_Flag']   = pd.Series(zF, index=g.index)
            g['DTW_Flag'] = pd.Series(dF, index=g.index)
            g['ISO_Flag'] = pd.Series(iF, index=g.index)
            g['ISO_Cont'] = used_cont

            voted = vote_and_smooth(zF, iF, dF, hard_overrides=(rz_abs > MAD_Z_HARD))
            combined = pd.Series(np.asarray(voted, dtype=bool), index=g.index) & (~g['Rule_Based_Invalid'])
            suppress_steps_flag = bool(profile.get('suppress_steps', False))
            g['Combined_Anomaly'] = pd.Series(
                filter_events_by_impact(g, combined.to_numpy(), suppress_steps=suppress_steps_flag),
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
            dtw_pts  = int(g['DTW_Flag'].sum())

            if comb_pts > 0:
                abnormal_sensor_count += 1
                total_events_count += num_events

            sensor_time = time.time() - sensor_start
            if VERBOSE_LEVEL >= 1:
                print(f"   -> Z:{z_pts} ISO:{iso_pts} DTW:{dtw_pts} COMB:{comb_pts} events:{num_events} "
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
            'DTW_Dist', 'DTW_Flag',
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
                (tmp["DTW_Points"] > 0).astype(int)
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