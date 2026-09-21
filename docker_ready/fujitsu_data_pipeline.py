import os
import re
import sys
from datetime import datetime, timedelta

# === Config ===
# history_folder    = r"F:\DDS_SHARE\HISTORY"
# histalmevt_folder = r"F:\DDS_SHARE\HISTALMEVT"
# history_folder    = r"C:\Users\wrms_vm\Documents\v6\analysis\Scripts\raw\HISTORY"
# histalmevt_folder = r"C:\Users\wrms_vm\Documents\v6\analysis\Scripts\raw\HISTALMEVT"
history_folder    = r"HISTORY"
histalmevt_folder = r"HISTALMEVT"

# Filename patterns (match just the filename, not full path)
history_pattern    = r"hts_\d{4}_\d{2}_HISTORY_.*\.csv$"
histalmevt_pattern = r"hts_\d{4}_\d{2}_HISTALMEVT_.*\.csv$"

# Timestamp pattern inside filenames, e.g. ..._2025Apr02-000000...
TS_REGEX = re.compile(r'(\d{4}[A-Za-z]{3}\d{2}-\d{6})')
TS_FORMAT = '%Y%b%d-%H%M%S'  # 2025Apr02-000000

# ---------------------------------------------------------------------------
# Wall-clock-anchored window config
# ---------------------------------------------------------------------------
# Instead of "find the newest file and take 72h backwards from there" (which
# silently processes stale data when the upstream stops feeding), we compute
# the expected hourly file timestamps for the last 72 hours from NOW, then
# check which ones actually exist on disk.
#
# Tunable via env (all optional):
#   FETCH_WINDOW_HOURS              - how many hours back to look (default 72)
#   FETCH_GRACE_MINUTES             - subtract this from "now" before computing
#                                     the window, to allow network-share sync
#                                     delay (default 30)
#   FETCH_MAX_MISSING_PERCENT       - abort if more than this % of expected
#                                     files are missing (default 25)
# ---------------------------------------------------------------------------
FETCH_WINDOW_HOURS        = int(os.getenv("FETCH_WINDOW_HOURS", "72"))
FETCH_GRACE_MINUTES       = int(os.getenv("FETCH_GRACE_MINUTES", "30"))
FETCH_MAX_MISSING_PERCENT = int(os.getenv("FETCH_MAX_MISSING_PERCENT", "25"))

def extract_datetime_from_filename(filename: str):
    """
    Extract datetime from filename using e.g. 2025Apr02-000000.
    Returns datetime or None.
    """
    m = TS_REGEX.search(filename)
    if not m:
        return None
    ts = m.group(1)
    try:
        return datetime.strptime(ts, TS_FORMAT)
    except ValueError:
        return None

def iter_matching_files(root_folder: str, filename_regex: str):
    """
    Yield (full_path, filename, dt) for files whose filename matches the regex
    and contains a parseable timestamp.
    Walks subfolders.
    """
    name_re = re.compile(filename_regex)
    for dirpath, _, files in os.walk(root_folder):
        for fn in files:
            if not fn.lower().endswith(".csv"):
                continue
            if not name_re.match(fn):
                continue
            dt = extract_datetime_from_filename(fn)
            if dt is None:
                continue
            yield os.path.join(dirpath, fn), fn, dt

def expected_hourly_timestamps(now: datetime, window_hours: int, grace_minutes: int) -> list[datetime]:
    """
    Compute the list of expected file timestamps for the last `window_hours`
    of hourly files, anchored to wall-clock now (minus a grace period for
    network-share sync delay).

    Returns timestamps floored to the hour, in ascending order.
    e.g. at 14:23 with window=3, grace=30: returns [11:00, 12:00, 13:00]
         (because now-grace = 13:53, floor to hour = 13:00, then back 2 hours)
    """
    anchor = now - timedelta(minutes=grace_minutes)
    end_hour = anchor.replace(minute=0, second=0, microsecond=0)
    timestamps = [end_hour - timedelta(hours=h) for h in range(window_hours)]
    timestamps.sort()
    return timestamps


def pick_wallclock_window(files_with_dt, now: datetime,
                          window_hours: int = FETCH_WINDOW_HOURS,
                          grace_minutes: int = FETCH_GRACE_MINUTES):
    """
    Wall-clock-anchored replacement for pick_latest_window.

    Computes the expected hourly file timestamps for the last `window_hours`
    from now (minus grace), then returns the subset of `files_with_dt` whose
    timestamps fall in that window. Also returns the list of expected
    timestamps that are MISSING from disk, for diagnostic purposes.

    Why this matters: the old behavior was "find the newest file and take
    72h backwards". If the upstream stops writing new files, the newest
    file is N days old and the pipeline silently processes ancient data
    forever. The new behavior anchors to real time, so a stalled upstream
    is detected immediately.

    Returns
    -------
    window_items : list of (path, filename, dt) sorted ascending by dt
    expected_timestamps : list of hourly datetimes expected in the window
    missing_timestamps : list of expected datetimes for which no file exists
    """
    expected = expected_hourly_timestamps(now, window_hours, grace_minutes)
    expected_set = set(expected)

    # Filter to files whose timestamp matches an expected hourly slot.
    # Floor each found file's timestamp to the hour for matching (in case the
    # upstream's HHMMSS isn't exactly :00:00).
    window_items = []
    found_hours = set()
    for path, fn, dt in files_with_dt:
        dt_hour = dt.replace(minute=0, second=0, microsecond=0)
        if dt_hour in expected_set:
            window_items.append((path, fn, dt))
            found_hours.add(dt_hour)

    window_items.sort(key=lambda x: x[2])
    missing = sorted(expected_set - found_hours)
    return window_items, expected, missing


# Kept for backward compatibility / debugging. Not used in the run block.
def pick_latest_window(files_with_dt, hours: int = 72):
    """
    DEPRECATED: legacy "newest-file-anchored" window. Kept so existing
    callers (if any) don't break, but the run block now uses
    pick_wallclock_window which is robust against stalled upstreams.
    """
    items = list(files_with_dt)
    if not items:
        return [], None, None
    latest_dt = max(dt for _, _, dt in items)
    start_dt = latest_dt - timedelta(hours=hours - 1)
    window_items = [(p, f, d) for (p, f, d) in items if start_dt <= d <= latest_dt]
    window_items.sort(key=lambda x: x[2])
    return window_items, start_dt, latest_dt

def safe_read_lines(path: str):
    """
    Read file with a couple of common fallbacks.
    """
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with open(path, "r", encoding=enc, errors="strict") as fh:
                return fh.readlines()
        except Exception:
            continue
    # Last resort with 'errors=replace' to avoid hard crashes
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.readlines()

def merge_latest_hours(root_folder: str,
                       filename_regex: str,
                       base_output_name: str,
                       hours: int = FETCH_WINDOW_HOURS,
                       output_dir: str = "raw",
                       now: datetime = None):
    """
    Wall-clock-anchored merge.

    Compute the expected hourly file timestamps for the last `hours` hours
    (from `now` minus grace), then merge whichever of those files actually
    exist on disk into a single CSV. Diagnostic info about missing files is
    printed and returned, so the caller can decide whether to abort the run.

    Returns
    -------
    ok : bool
        True if the merge succeeded AND missing-file ratio is within tolerance.
        False if too many expected files are missing (caller should abort).
    info : dict
        Keys: expected_count, found_count, missing_count, missing_percent,
              window_start, window_end, output_path, missing_timestamps
    """
    if now is None:
        now = datetime.now()

    files_iter = iter_matching_files(root_folder, filename_regex)
    window_items, expected, missing = pick_wallclock_window(
        files_iter, now=now, window_hours=hours, grace_minutes=FETCH_GRACE_MINUTES
    )

    expected_count = len(expected)
    found_count = len(window_items)
    missing_count = len(missing)
    missing_percent = (missing_count / expected_count * 100.0) if expected_count else 0.0
    window_start = expected[0] if expected else None
    window_end   = expected[-1] if expected else None

    info = {
        "expected_count": expected_count,
        "found_count": found_count,
        "missing_count": missing_count,
        "missing_percent": missing_percent,
        "window_start": window_start,
        "window_end": window_end,
        "output_path": None,
        "missing_timestamps": missing,
    }

    print(
        f"[{base_output_name}] Wall-clock window: {window_start} to {window_end} | "
        f"expected={expected_count}  found={found_count}  missing={missing_count} "
        f"({missing_percent:.1f}%)",
        flush=True,
    )

    if missing_count > 0:
        # Show first few missing slots so an operator can spot patterns
        # (e.g. "all from 6 hours ago onward" -> upstream stopped at that time).
        sample = ", ".join(m.strftime("%Y-%m-%d %H:%M") for m in missing[:5])
        suffix = "" if missing_count <= 5 else f" ... and {missing_count - 5} more"
        print(f"[{base_output_name}] Missing hourly slots (sample): {sample}{suffix}",
              flush=True)

    # Decide if the gap is tolerable.
    if expected_count > 0 and missing_percent > FETCH_MAX_MISSING_PERCENT:
        print(
            f"[{base_output_name}] [STALE-DATA] {missing_percent:.1f}% of expected "
            f"hourly files are missing (threshold: {FETCH_MAX_MISSING_PERCENT}%). "
            f"Upstream feed may be down, or the folder mount may be wrong. "
            f"Check: (1) upstream Fujitsu export still writing? "
            f"(2) Docker volume points at the right host folder?",
            flush=True,
        )
        return False, info

    if found_count == 0:
        print(f"[{base_output_name}] No files found in window. Nothing to merge.",
              flush=True)
        return False, info

    os.makedirs(output_dir, exist_ok=True)
    actual_start = window_items[0][2]
    actual_end   = window_items[-1][2]
    output_name = (
        f"{base_output_name}_"
        f"{actual_start.strftime('%Y%m%d-%H%M%S')}_to_"
        f"{actual_end.strftime('%Y%m%d-%H%M%S')}.csv"
    )
    output_path = os.path.join(output_dir, output_name)
    info["output_path"] = output_path

    print(f"[{base_output_name}] Merging {found_count} files into: {output_path}",
          flush=True)
    for _, fn, dt in window_items:
        print(f" - {dt}  {fn}")

    combined = []
    header_saved = False
    merged_count = 0

    for full_path, _, _ in window_items:
        try:
            lines = safe_read_lines(full_path)
            if not lines:
                continue
            if header_saved:
                combined.extend(lines[1:])
            else:
                combined.extend(lines)
                header_saved = True
            merged_count += 1
        except Exception as e:
            print(f"Failed to read {full_path}: {e}")

    if merged_count == 0:
        print(f"[{base_output_name}] Found files but failed to read them.")
        return False, info

    with open(output_path, "w", encoding="utf-8") as out:
        out.writelines(combined)

    print(f"[{base_output_name}] Done. Wrote {merged_count} files -> {output_path}\n")
    return True, info


# === Run (wall-clock-anchored window: detects stalled upstreams) ===
_now = datetime.now()
print(f"[wallclock] Anchoring window to now={_now.strftime('%Y-%m-%d %H:%M:%S')} "
      f"(grace={FETCH_GRACE_MINUTES}min, window={FETCH_WINDOW_HOURS}h)",
      flush=True)

ok_hist,  info_hist  = merge_latest_hours(
    history_folder,    history_pattern,    "history_fujitsu",
    hours=FETCH_WINDOW_HOURS, output_dir="raw", now=_now,
)
ok_alarm, info_alarm = merge_latest_hours(
    histalmevt_folder, histalmevt_pattern, "histalmevt_fujitsu",
    hours=FETCH_WINDOW_HOURS, output_dir="raw", now=_now,
)

if not (ok_hist and ok_alarm):
    print(
        "[ABORT] fujitsu_data_pipeline.py: one or more source feeds are stale "
        "or have too many missing files. Skipping merge of downstream-ready CSVs "
        "to avoid processing outdated data.",
        flush=True,
    )
    sys.exit(1)