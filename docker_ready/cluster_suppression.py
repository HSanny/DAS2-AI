"""
cluster_suppression.py
======================

Cluster suppression for the Fujitsu/PUB anomaly pipeline.

Adapted from the DAS1 cluster-suppression spec, with two project-specific
adjustments:

  1. No anomaly_type field in this project — buckets are keyed on time only.
  2. Detection timestamps are not exactly aligned across sensors (each sensor's
     First_Anomaly_Time comes from its own sample stream). We use a tolerance
     window (default +/-5 minutes) instead of exact equality.

This module is name-based clustering only. It does NOT analyze value patterns.
If you ever want correlation-based clustering, that's an additional rule layered
on top — see the spec doc.

Behavior preserved from DAS1
----------------------------
  - Lone events (sole member of a time bucket) always alert.
  - Suppression is symmetric: if A clusters with B, both are suppressed.
  - Suppressed events are kept in the output for audit; only a flag is set.
  - The rule operates at OUTPUT TIME, not detection time — detection is
    untouched.

Hybrid name rule
----------------
Two Descriptions are "same equipment group" if EITHER:
  - difflib.SequenceMatcher.ratio() >= CLUSTER_SIMILARITY_THRESHOLD (0.85), OR
  - common-prefix length >= CLUSTER_PREFIX_THRESHOLD (15 chars).

Both checks complement each other. See the spec doc for the empirical pairs
table that justifies these thresholds.

Tuning
------
Override via env vars in production:
  - ENABLE_CLUSTER_SUPPRESSION   (1/0, default 1)
  - CLUSTER_SIMILARITY_THRESHOLD (float, default 0.85)
  - CLUSTER_PREFIX_THRESHOLD     (int, default 15)
  - CLUSTER_TIME_TOLERANCE_MIN   (int, minutes, default 5)
"""

from __future__ import annotations

import os
from difflib import SequenceMatcher
from typing import Iterable

import pandas as pd


# --------------------------------------------------------------------------- #
# Tunable constants (env-overridable, see module docstring)
# --------------------------------------------------------------------------- #
def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    try:
        return float(v) if v is not None and v.strip() != "" else default
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v is not None and v.strip() != "" else default
    except (TypeError, ValueError):
        return default


ENABLE_CLUSTER_SUPPRESSION = _env_int("ENABLE_CLUSTER_SUPPRESSION", 1) == 1
CLUSTER_SIMILARITY_THRESHOLD = _env_float("CLUSTER_SIMILARITY_THRESHOLD", 0.85)
CLUSTER_PREFIX_THRESHOLD = _env_int("CLUSTER_PREFIX_THRESHOLD", 15)
CLUSTER_TIME_TOLERANCE_MIN = _env_int("CLUSTER_TIME_TOLERANCE_MIN", 5)


# --------------------------------------------------------------------------- #
# Name-matching primitives
# --------------------------------------------------------------------------- #
def _common_prefix_length(a: str, b: str) -> int:
    """Number of characters two strings share from index 0."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def descriptions_are_clustered(
    desc_a: str,
    desc_b: str,
    similarity_threshold: float = CLUSTER_SIMILARITY_THRESHOLD,
    prefix_threshold: int = CLUSTER_PREFIX_THRESHOLD,
) -> bool:
    """
    OR-gate: similarity above threshold OR shared prefix above threshold.
    Either alone is sufficient to call two descriptions same-equipment-group.

    Empty / None strings never cluster.
    """
    if not desc_a or not desc_b:
        return False
    if desc_a == desc_b:
        # Identical names always cluster — short-circuit before similarity.
        return True
    if SequenceMatcher(None, desc_a, desc_b).ratio() >= similarity_threshold:
        return True
    if _common_prefix_length(desc_a, desc_b) >= prefix_threshold:
        return True
    return False


# --------------------------------------------------------------------------- #
# Time bucketing with tolerance
# --------------------------------------------------------------------------- #
def _time_bucket_key(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    tolerance_min: int,
) -> tuple:
    """
    Floor start/end to the nearest tolerance window. Two events that fall in
    the same bucket pair are CANDIDATES for clustering — we still do a
    fine-grained +/- check on the actual timestamps to handle events that
    floor to different buckets but are still within tolerance of each other.

    Returns a hashable key suitable for bucketing.
    """
    if pd.isna(start_time) or pd.isna(end_time):
        return (None, None)
    freq = f"{tolerance_min}min"
    s = pd.Timestamp(start_time).floor(freq)
    e = pd.Timestamp(end_time).floor(freq)
    return (s, e)


def _within_tolerance(
    t_a: pd.Timestamp,
    t_b: pd.Timestamp,
    tolerance_min: int,
) -> bool:
    if pd.isna(t_a) or pd.isna(t_b):
        return False
    return abs((pd.Timestamp(t_a) - pd.Timestamp(t_b)).total_seconds()) <= tolerance_min * 60


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def annotate_suppression(
    summary_df: pd.DataFrame,
    *,
    start_col: str = "First_Anomaly_Time",
    end_col: str = "Last_Anomaly_Time",
    description_col: str = "Description",
    id_col: str | None = None,
    similarity_threshold: float = CLUSTER_SIMILARITY_THRESHOLD,
    prefix_threshold: int = CLUSTER_PREFIX_THRESHOLD,
    tolerance_min: int = CLUSTER_TIME_TOLERANCE_MIN,
    enabled: bool = ENABLE_CLUSTER_SUPPRESSION,
) -> pd.DataFrame:
    """
    Return a copy of summary_df with two added columns:

      - Suppressed: bool. True if this row would be silenced by a Telegram
        dispatcher; False if it should alert.
      - Suppression_Reason: str. Empty for non-suppressed rows. For suppressed
        rows, the Description of one peer that triggered the cluster (useful
        for audit).

    Rules:
      1. Bucket rows by floored (start, end) at the tolerance granularity.
      2. Within each bucket, do pairwise checks: two rows cluster if their
         start AND end times are within tolerance AND their descriptions
         satisfy the hybrid name rule.
      3. Any row that clusters with at least one OTHER row is suppressed.
      4. Lone rows in a bucket alert.

    If `enabled=False`, the columns are still added but all rows have
    Suppressed=False — making it trivial to A/B compare.
    """
    out = summary_df.copy()
    out["Suppressed"] = False
    out["Suppression_Reason"] = ""

    if out.empty or not enabled:
        return out

    # Required columns must exist; if not, fail loud (better than silently mis-clustering).
    for col in (start_col, end_col, description_col):
        if col not in out.columns:
            raise KeyError(f"annotate_suppression: required column '{col}' missing from summary_df")

    # Ensure timestamp columns are datetime — coerce if needed.
    out[start_col] = pd.to_datetime(out[start_col], errors="coerce")
    out[end_col] = pd.to_datetime(out[end_col], errors="coerce")

    # Stable row identifier for bookkeeping.
    if id_col and id_col in out.columns:
        ids = out[id_col].tolist()
    else:
        ids = list(out.index)

    # Bucket rows. Rows in different buckets cannot cluster (no need to check
    # across buckets when tolerance == bucket size, because pairs >tolerance
    # apart can't satisfy the within-tolerance check anyway).
    buckets: dict[tuple, list[int]] = {}
    for pos_idx, (_, row) in enumerate(out.iterrows()):
        key = _time_bucket_key(row[start_col], row[end_col], tolerance_min)
        if key == (None, None):
            continue  # rows with no valid time can't cluster
        buckets.setdefault(key, []).append(pos_idx)

    # Also check the adjacent bucket on each axis, because two events near a
    # bucket boundary may floor differently but still be within tolerance.
    # We resolve this by checking neighbors: for each row, look at its bucket
    # AND the +/- 1 buckets on start and end axes.
    def _neighbor_keys(key: tuple) -> Iterable[tuple]:
        s, e = key
        if s is None or e is None:
            return []
        step = pd.Timedelta(minutes=tolerance_min)
        ds = (s - step, s, s + step)
        de = (e - step, e, e + step)
        return [(a, b) for a in ds for b in de]

    suppressed = [False] * len(out)
    reasons: list[str] = [""] * len(out)

    descriptions = out[description_col].fillna("").astype(str).tolist()
    start_times = out[start_col].tolist()
    end_times = out[end_col].tolist()

    # For each row, search its bucket and neighboring buckets for a peer.
    for pos_idx in range(len(out)):
        if pd.isna(start_times[pos_idx]) or pd.isna(end_times[pos_idx]):
            continue
        my_key = _time_bucket_key(start_times[pos_idx], end_times[pos_idx], tolerance_min)
        desc_a = descriptions[pos_idx]
        if not desc_a:
            continue

        candidate_positions: list[int] = []
        for nk in _neighbor_keys(my_key):
            candidate_positions.extend(buckets.get(nk, []))

        for other_pos in candidate_positions:
            if other_pos == pos_idx:
                continue
            if not _within_tolerance(start_times[pos_idx], start_times[other_pos], tolerance_min):
                continue
            if not _within_tolerance(end_times[pos_idx], end_times[other_pos], tolerance_min):
                continue
            desc_b = descriptions[other_pos]
            if not desc_b:
                continue
            if descriptions_are_clustered(
                desc_a, desc_b,
                similarity_threshold=similarity_threshold,
                prefix_threshold=prefix_threshold,
            ):
                suppressed[pos_idx] = True
                reasons[pos_idx] = f"clustered_with: {desc_b}"
                break  # one peer is enough

    out["Suppressed"] = suppressed
    out["Suppression_Reason"] = reasons
    return out


# --------------------------------------------------------------------------- #
# Self-test against the known-pairs table from the spec
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    """
    Run the spec's known-pairs table as a regression check.
    Raises AssertionError on regression.
    """
    cases = [
        # (desc_a, desc_b, should_cluster_by_name_rule)
        ("MARINABARRAGE-CG1 TOTAL FLOW", "MARINABARRAGE-CG7 TOTAL FLOW", True),
        ("SerangoonTG-SRGSR DAILY FLOW", "SerangoonTG-SRGSR TOTAL FLOWRATE", True),
        ("MARINABARRAGE-CG1 TOTAL FLOW", "MARINABARRAGE-DP1 FLOWRATE", False),
        ("MARINABARRAGE-CG1 TOTAL FLOW", "MARINABARRAGE-SG1 SLUICE 1 TOTAL FLOW", False),
        ("UpperSeletarPS-Dissolved-Oxygen", "BedokPS-Dissolved-Oxygen", False),
        ("MacRitchiePS-Compressor-1-Discharge-Flow",
         "MacRitchiePS-Compressor-2-Discharge-Flow", True),
    ]
    failures = []
    for a, b, expected in cases:
        got = descriptions_are_clustered(a, b)
        if got != expected:
            sim = SequenceMatcher(None, a, b).ratio()
            pre = _common_prefix_length(a, b)
            failures.append(
                f"  {a!r} vs {b!r}: expected={expected}, got={got} "
                f"(sim={sim:.3f}, prefix={pre})"
            )
    if failures:
        raise AssertionError(
            "cluster_suppression name-rule regression:\n" + "\n".join(failures)
        )
    print("[OK] cluster_suppression name-rule: all 6 known pairs correct.")


if __name__ == "__main__":
    _self_test()
