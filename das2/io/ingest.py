"""
das2.io.ingest
==============

Read the Fujitsu historian export as it is actually written.

Formats, verified against real files (2026-09-20 samples)
---------------------------------------------------------
Everything here was checked against genuine exports rather than inferred from
the previous pipeline's code, which disagrees with the live feed in several
places.

``HISTORY/hts_YYYY_MM_HISTORY_<ts>.csv`` — hourly, **semicolon**::

    IPADDRESS;ROW_ID;DATETIME;CURRVALUE
    413120;16826219;9/20/2026 9:00:00 PM;37.5

``HISTCURR/hts_HISTCURR_<ts>.csv`` — hourly snapshots, **semicolon**, nine
columns including ``POINTTYPE``::

    ROW_ID;IPADDRESS;DESCRIPTION;TAGNAME;RTUNUMBER;RAWTYPE;POINTTYPE;DATETIME;CURRVALUE

Note this is **not** what the older loaders expect. They read a single
comma-separated ``HISTCURR/histcurr_fujitsu.csv`` (or ``histcurr.csv``) with six
columns in a different order, and would fail outright on the live file.

``LongLat.csv`` — **`Location,Latitude,Longitude,LKey`**, joined to
``RTUNUMBER`` on ``LKey``. Note Latitude precedes Longitude, the opposite of
what a "LongLat" name suggests; getting this backwards silently places every
sensor in the Indian Ocean.

Timestamps are US 12-hour: ``%m/%d/%Y %I:%M:%S %p``. The format is passed
explicitly. pandas does infer these correctly today — that was tested, including
the ambiguous ``9/5/2026`` case — but inference is per-call and month/day order
is only unambiguous for days above 12, so a silent locale-dependent flip is not
a risk worth carrying for a feed this size.
"""

from __future__ import annotations

import glob
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

log = logging.getLogger("das2.io.ingest")

#: Timestamps inside the CSVs.
DATETIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"

#: Timestamp embedded in a filename, e.g. hts_2026_09_HISTORY_2026Sep20-210000.csv
FILENAME_TS_RE = re.compile(r"(\d{4}[A-Za-z]{3}\d{2}-\d{6})")
FILENAME_TS_FORMAT = "%Y%b%d-%H%M%S"

#: RawType values that denote an analog point. Carried over from the production
#: pipeline, and consistent with the live inventory.
ANALOG_RAWTYPES = {1, 5}

#: Sanity bound on readings, matching the production pipeline.
VALUE_LIMIT = 1e9

#: LongLat rows whose Location is this are placeholders: seven LKeys (0, 1, 2047,
#: 6, 725, 726 ...) all share the single coordinate 1.2575396, 103.7847767. They
#: are NOT a location. RTU 0 alone carries over 1,200 sensors, so accepting these
#: would drop a thousand pins on a plausible-looking spot near Clementi and make
#: the regional map confidently wrong. Excluded, and counted as unplaced.
UNUSED_LOCATION_MARKERS = {"unused", "unassigned", "spare", "n/a", ""}


@dataclass
class IngestReport:
    """What a load actually produced. Logged every run rather than assumed."""

    history_files: int = 0
    #: Files present but unreadable -- empty, truncated or malformed. Counted
    #: rather than ignored: they are missing data wearing a filename, and a
    #: feed quietly losing half its files must not look like a quiet network.
    history_files_skipped: int = 0
    history_rows: int = 0
    history_rows_dropped: int = 0
    inventory_rows: int = 0
    readings_matched: int = 0
    readings_unmatched: int = 0
    sites_with_coords: int = 0
    sites_unused: int = 0
    sensors_with_coords: int = 0
    sensors_without_coords: int = 0
    window_start: datetime | None = None
    window_end: datetime | None = None
    missing_hours: list[datetime] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def coordinate_coverage_pct(self) -> float:
        total = self.sensors_with_coords + self.sensors_without_coords
        return round(100.0 * self.sensors_with_coords / total, 1) if total else 0.0

    def as_dict(self) -> dict:
        """
        Loggable form.

        `missing_hours` is reduced to a count and a range rather than the full
        list: a feed that has been down for a week produces 168 timestamps, and
        dumping them into every run's stats buries the numbers that matter
        under an unreadable wall of datetimes.
        """
        return {
            "history_files": self.history_files,
            "history_files_skipped": self.history_files_skipped,
            "history_rows": self.history_rows,
            "history_rows_dropped": self.history_rows_dropped,
            "inventory_rows": self.inventory_rows,
            "readings_matched": self.readings_matched,
            "readings_unmatched": self.readings_unmatched,
            "sensors_with_coords": self.sensors_with_coords,
            "sensors_without_coords": self.sensors_without_coords,
            "coordinate_coverage_pct": self.coordinate_coverage_pct,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "missing_hours": len(self.missing_hours),
            "missing_hours_range": (
                f"{self.missing_hours[0]:%Y-%m-%d %H:%M} .. "
                f"{self.missing_hours[-1]:%Y-%m-%d %H:%M}"
                if self.missing_hours else None),
        }

    def summary(self) -> str:
        lines = [
            f"HISTORY   : {self.history_files} files, {self.history_rows:,} rows "
            f"({self.history_rows_dropped:,} dropped)",
            f"inventory : {self.inventory_rows:,} points",
            f"matched   : {self.readings_matched:,} readings "
            f"({self.readings_unmatched:,} with no inventory entry)",
            f"coords    : {self.sensors_with_coords:,}/"
            f"{self.sensors_with_coords + self.sensors_without_coords:,} sensors "
            f"({self.coordinate_coverage_pct}%) from {self.sites_with_coords} sites"
            + (f", {self.sites_unused} placeholder sites excluded" if self.sites_unused else ""),
        ]
        if self.missing_hours:
            lines.append(f"missing   : {len(self.missing_hours)} expected hourly files")
        lines += self.notes
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def parse_datetime_column(series: pd.Series) -> pd.Series:
    """
    Parse the historian's US 12-hour timestamps.

    Falls back to inference for any row the explicit format rejects, so a single
    oddly-formatted line cannot discard a whole hour of readings.
    """
    parsed = pd.to_datetime(series, format=DATETIME_FORMAT, errors="coerce")
    unparsed = parsed.isna() & series.notna()
    if unparsed.any():
        parsed.loc[unparsed] = pd.to_datetime(
            series[unparsed], errors="coerce", format="mixed"
        )
    return parsed


def filename_timestamp(path: str | Path) -> datetime | None:
    m = FILENAME_TS_RE.search(Path(path).name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), FILENAME_TS_FORMAT)
    except ValueError:
        return None


def make_sensor_key(ip: pd.Series, row_id: pd.Series) -> pd.Series:
    """
    Stable per-sensor key.

    The production pipeline concatenates the two integers (``Hkey``). That is in
    principle ambiguous -- ip=1,row=23 and ip=12,row=3 both give "123" -- so it
    was checked against the live inventory: 13,982 points produce 13,982
    distinct keys, no collisions, because the real IDs are consistently wide.
    A separator is used here anyway, since it costs nothing and removes the
    failure mode entirely.
    """
    return ip.astype("int64").astype(str) + ":" + row_id.astype("int64").astype(str)


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #
def read_history_file(path: str | Path) -> pd.DataFrame:
    """One hourly HISTORY file -> sensor_key, ts, value."""
    df = pd.read_csv(path, sep=";", low_memory=False, on_bad_lines="skip")
    df.columns = [c.strip().upper() for c in df.columns]

    required = ["IPADDRESS", "ROW_ID", "DATETIME", "CURRVALUE"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path}: HISTORY missing {missing}. Found: {list(df.columns)}"
        )

    df = df.dropna(subset=required)
    df["value"] = pd.to_numeric(df["CURRVALUE"], errors="coerce")
    df["ts"] = parse_datetime_column(df["DATETIME"])
    df["ip_num"] = pd.to_numeric(df["IPADDRESS"], errors="coerce")
    df["row_num"] = pd.to_numeric(df["ROW_ID"], errors="coerce")

    df = df.dropna(subset=["value", "ts", "ip_num", "row_num"])
    # Same sanity bound as production: values beyond this are transport garbage.
    df = df[df["value"].abs() < VALUE_LIMIT]

    df["sensor_key"] = make_sensor_key(df["ip_num"], df["row_num"])
    return df[["sensor_key", "ts", "value"]].reset_index(drop=True)


def read_history_dir(directory: str | Path, *, pattern: str = "*HISTORY*.csv",
                     since: datetime | None = None,
                     report: IngestReport | None = None) -> pd.DataFrame:
    """
    Every hourly HISTORY file in a directory, optionally limited to a window.

    Reports how many expected hourly files are missing. A silently short feed
    looks identical to a quiet network, and processing a stale window as though
    it were current is worse than not running.
    """
    files = sorted(glob.glob(os.path.join(str(directory), pattern)))
    if since is not None:
        files = [f for f in files
                 if (ts := filename_timestamp(f)) is None or ts >= since]
    if not files:
        where = os.path.join(str(directory), pattern)
        if since is not None:
            raise FileNotFoundError(
                f"No HISTORY files at or after {since:%Y-%m-%d %H:%M} under "
                f"{where}. Files exist but all predate the window, so the feed "
                f"has stalled -- analysing older data as though it were current "
                f"is worse than not running."
                if glob.glob(where) else
                f"No HISTORY files matched {where}"
            )
        raise FileNotFoundError(f"No HISTORY files matched {where}")

    # One unreadable file must not take the monitoring system down.
    #
    # The share is written hourly, so the newest file is routinely being
    # written while this reads it, and arrives as zero bytes. pandas answers
    # that with `EmptyDataError: No columns to parse from file` -- and one such
    # file out of 10,334 aborted the entire run on the client's first attempt.
    # A monitoring system that stops because a file it does not need yet is
    # mid-copy is worse than useless: it goes quiet exactly when someone is
    # watching.
    #
    # Skipped files are COUNTED, logged, and excluded from the hourly stamps
    # below, so they register as missing data rather than vanishing. A file
    # present but unreadable is missing data wearing a filename, and a feed
    # quietly losing half its files must not look like a quiet network -- that
    # is the failure this module exists to make visible.
    frames, rows_in, skipped = [], 0, []
    for path in files:
        try:
            raw_rows = sum(1 for _ in open(path, encoding="utf-8",
                                           errors="replace")) - 1
            frame = read_history_file(path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError,
                ValueError, OSError) as exc:
            skipped.append(path)
            log.warning("skipping unreadable HISTORY file %s (%s: %s)",
                        os.path.basename(path), type(exc).__name__,
                        str(exc)[:80])
            continue
        rows_in += max(0, raw_rows)
        frames.append(frame)

    if not frames:
        raise ValueError(
            f"All {len(files)} HISTORY file(s) under {directory} were "
            f"unreadable. This is a broken feed, not a quiet network."
        )
    if skipped:
        log.warning("%d of %d HISTORY file(s) were unreadable and skipped",
                    len(skipped), len(files))

    out = pd.concat(frames, ignore_index=True)

    if report is not None:
        report.history_files = len(files) - len(skipped)
        report.history_files_skipped = len(skipped)
        report.history_rows = len(out)
        report.history_rows_dropped = max(0, rows_in - len(out))
        readable = [f for f in files if f not in set(skipped)]
        stamps = sorted(t for t in (filename_timestamp(f) for f in readable) if t)
        if stamps:
            report.window_start, report.window_end = stamps[0], stamps[-1]
            expected = set()
            cursor = stamps[0]
            while cursor <= stamps[-1]:
                expected.add(cursor)
                cursor += timedelta(hours=1)
            report.missing_hours = sorted(expected - set(stamps))
    return out


#: Matches any HISTCURR export, with or without a .csv extension. The real
#: share exports `hts_HISTCURR_2026Sep22-130000.csv` hourly; the extension is
#: not always visible, so it is not required here.
#:
#: Matched case-INSENSITIVELY, and by hand rather than with glob(), because
#: glob is case-sensitive on Linux while the CIFS share this reads is not.
#: The real files are `hts_HISTCURR_...` and the fixtures are
#: `histcurr_fujitsu.csv`; a case-sensitive pattern silently finds one and not
#: the other, which is the kind of difference that makes a system pass every
#: test and fail on the only machine that matters.
HISTCURR_TOKEN = "histcurr"
HISTCURR_GLOB = "*HISTCURR* (case-insensitive)"


def _histcurr_files(directory: Path) -> list[Path]:
    """Every HISTCURR export in `directory`, whatever its case."""
    try:
        entries = list(directory.iterdir())
    except OSError:
        return []
    return [p for p in entries
            if p.is_file() and HISTCURR_TOKEN in p.name.lower()]


def resolve_inventory_path(path: str | Path) -> Path:
    """
    The HISTCURR file to read, given a file OR a directory.

    The old pipeline consumed one pre-merged `HISTCURR/histcurr_fujitsu.csv`.
    The real share has never contained such a file: it exports an hourly
    snapshot, `hts_HISTCURR_2026Sep22-130000`, alongside the hourly HISTORY
    files. A deployment configured with the old name therefore fails its
    pre-flight with "inventory file exists: FAIL" and no indication that the
    inventory is sitting right next to it under a different name.

    HISTCURR is a snapshot of current values rather than an accumulating log,
    so "the newest one" is always the right answer and pointing at the
    directory is the more honest configuration. Three cases:

      * a file that exists          -> use it, unchanged
      * a directory                 -> the newest HISTCURR file inside it
      * a missing file whose parent holds HISTCURR files -> the newest of
        those, with a warning naming the setting to change

    The third exists so an upgrade does not break on a stale config value, and
    it warns rather than substituting quietly -- reading a different file from
    the one configured is exactly the kind of helpfulness that becomes a
    mystery six months later.
    """
    p = Path(path)
    if p.is_file():
        return p

    if p.is_dir():
        return _newest_histcurr(p)

    if p.parent.is_dir() and _histcurr_files(p.parent):
        chosen = _newest_histcurr(p.parent)
        log.warning(
            "%s does not exist; using %s instead. Point "
            "DAS2_INGEST_HISTCURR_PATH at the directory (%s) to silence this.",
            p, chosen.name, p.parent)
        return chosen

    raise FileNotFoundError(
        f"No HISTCURR inventory at {p}. Expected either that file, or a "
        f"directory containing {HISTCURR_GLOB} (the share exports one hourly, "
        f"e.g. hts_HISTCURR_2026Sep22-130000)."
    )


def _newest_histcurr(directory: Path) -> Path:
    """
    The most recent HISTCURR file in `directory`.

    Ordered by the timestamp in the filename, not by name: `2026Sep22` sorts
    lexically as Apr < Aug < Dec < Feb, so a plain `sorted()` would happily
    pick April's file in December. Modification time is the fallback for a
    file whose name carries no parseable stamp -- and it is only a fallback,
    because a re-copied share can reset mtime on every file at once.
    """
    candidates = _histcurr_files(directory)
    if not candidates:
        raise FileNotFoundError(
            f"No {HISTCURR_GLOB} files in {directory}. This directory is the "
            f"sensor inventory; without it there are no sensors to analyse."
        )
    return max(
        candidates,
        key=lambda f: (filename_timestamp(f) or datetime.min, f.stat().st_mtime),
    )


def read_inventory(path: str | Path, *, report: IngestReport | None = None) -> pd.DataFrame:
    """
    HISTCURR snapshot -> the sensor dimension.

    Semicolon-separated with nine columns. `POINTTYPE` is the SCADA's own
    engineering type code; it is carried through so classification can consult
    it, but it is not authoritative -- see io.classify.

    `path` may be the file or the directory holding the hourly exports.
    """
    path = resolve_inventory_path(path)
    df = pd.read_csv(path, sep=";", low_memory=False, on_bad_lines="skip",
                     encoding="utf-8", encoding_errors="replace")
    df.columns = [c.strip().upper() for c in df.columns]

    required = ["IPADDRESS", "ROW_ID", "DESCRIPTION", "RAWTYPE"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path}: HISTCURR missing {missing}. Found: {list(df.columns)}. "
            f"Expected the live 9-column semicolon format "
            f"(ROW_ID;IPADDRESS;DESCRIPTION;TAGNAME;RTUNUMBER;RAWTYPE;POINTTYPE;"
            f"DATETIME;CURRVALUE), not the older comma-separated 6-column file."
        )

    df = df.dropna(subset=required)
    df["ip_num"] = pd.to_numeric(df["IPADDRESS"], errors="coerce")
    df["row_num"] = pd.to_numeric(df["ROW_ID"], errors="coerce")
    df = df.dropna(subset=["ip_num", "row_num"])

    out = pd.DataFrame({
        "sensor_key": make_sensor_key(df["ip_num"], df["row_num"]),
        "description": df["DESCRIPTION"].astype(str).str.strip(),
        "tagname": df.get("TAGNAME", pd.Series(dtype=str)).astype(str).str.strip(),
        "rtu_number": df.get("RTUNUMBER", pd.Series(dtype=str)).astype(str).str.strip(),
        "rawtype": pd.to_numeric(df["RAWTYPE"], errors="coerce").fillna(-1).astype(int),
        "pointtype": pd.to_numeric(df.get("POINTTYPE"), errors="coerce"),
    })
    out["signal_type"] = out["rawtype"].apply(
        lambda x: "Analog" if x in ANALOG_RAWTYPES else "Digital")
    out = out.drop_duplicates("sensor_key").reset_index(drop=True)

    if report is not None:
        report.inventory_rows = len(out)
    return out


def read_longlat(path: str | Path, *, report: IngestReport | None = None) -> pd.DataFrame:
    """
    Site coordinates, keyed on LKey (which joins to RTUNUMBER).

    Column order is `Location,Latitude,Longitude,LKey` -- Latitude BEFORE
    Longitude, despite the filename. Read by name, never by position.

    Placeholder rows are dropped: seven LKeys share Location "Unused" at the
    single coordinate 1.2575396, 103.7847767. RTU 0 alone carries over 1,200
    sensors, so treating that as a real position would put a thousand pins on
    one plausible-looking spot and make the map confidently wrong. They are
    excluded and counted instead.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    required = ["Location", "Latitude", "Longitude", "LKey"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: LongLat missing {missing}. Found: {list(df.columns)}")

    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["LKey"] = df["LKey"].astype(str).str.strip()
    df["Location"] = df["Location"].astype(str).str.strip()

    placeholder = df["Location"].str.lower().isin(UNUSED_LOCATION_MARKERS)
    n_unused = int(placeholder.sum())
    df = df[~placeholder]
    df = df.dropna(subset=["Latitude", "Longitude"])

    out = df[["LKey", "Location", "Latitude", "Longitude"]].rename(columns={
        "LKey": "rtu_number", "Location": "site",
        "Latitude": "latitude", "Longitude": "longitude",
    }).drop_duplicates("rtu_number").reset_index(drop=True)

    if report is not None:
        report.sites_with_coords = len(out)
        report.sites_unused = n_unused
        if n_unused:
            report.notes.append(
                f"note      : {n_unused} LongLat rows are placeholders "
                f"('Unused') and were excluded; sensors on those RTUs have no "
                f"position and are omitted from the map."
            )
    return out


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def build_sensor_table(inventory: pd.DataFrame, longlat: pd.DataFrame | None = None,
                       *, report: IngestReport | None = None) -> pd.DataFrame:
    """
    Join the inventory to site coordinates and classify each point.

    Coordinates arrive per RTU, so every sensor on one RTU shares a position:
    resolution is site level, which is the right granularity for deciding where
    to send someone but cannot resolve within a site.

    The previous pipeline merged two frames that each carried a `Location`
    column, so pandas renamed them `Location_x`/`Location_y` and the downstream
    lookup for plain `Location` silently produced an empty string on every row.
    Here the site column arrives only from LongLat, under a distinct name.
    """
    from das2.io.classify import get_classifier
    from das2.spatial.regions import place, site_from_description

    out = inventory.copy()

    if longlat is not None and not longlat.empty:
        out = out.merge(longlat, on="rtu_number", how="left")
    else:
        out["site"] = pd.NA
        out["latitude"] = pd.NA
        out["longitude"] = pd.NA

    # Fall back to the description prefix when the RTU join found no site.
    out["site"] = out["site"].where(
        out["site"].notna(),
        out["description"].map(site_from_description),
    )

    clf = get_classifier()
    classifications = [
        clf.classify(desc, rawtype) for desc, rawtype
        in zip(out["description"], out["rawtype"])
    ]
    out["equipment"] = [c.equipment for c in classifications]
    out["alertable"] = [c.alertable for c in classifications]
    out["kind"] = [c.meta.kind for c in classifications]
    out["unit"] = [c.meta.unit for c in classifications]

    placements = [
        place(lat if pd.notna(lat) else None,
              lon if pd.notna(lon) else None,
              desc)
        for lat, lon, desc in zip(out["latitude"], out["longitude"], out["description"])
    ]
    out["region"] = [p.region.value for p in placements]
    out["planning_area"] = [p.planning_area for p in placements]
    out["placement_source"] = [p.source for p in placements]

    if report is not None:
        has_coords = out["latitude"].notna() & out["longitude"].notna()
        report.sensors_with_coords = int(has_coords.sum())
        report.sensors_without_coords = int((~has_coords).sum())
    return out.reset_index(drop=True)


def load_all(history_dir: str | Path, inventory_path: str | Path,
             longlat_path: str | Path | None = None,
             *, since: datetime | None = None
             ) -> tuple[pd.DataFrame, pd.DataFrame, IngestReport]:
    """
    Load a full window: (readings, sensors, report).

    Readings with no inventory entry are dropped but counted -- an unmatched key
    means a sensor is being recorded that nothing knows the name of, which is
    worth seeing rather than silently discarding.
    """
    report = IngestReport()
    readings = read_history_dir(history_dir, since=since, report=report)
    inventory = read_inventory(inventory_path, report=report)
    longlat = read_longlat(longlat_path, report=report) if longlat_path else None
    sensors = build_sensor_table(inventory, longlat, report=report)

    known = set(sensors["sensor_key"])
    matched = readings["sensor_key"].isin(known)
    report.readings_matched = int(matched.sum())
    report.readings_unmatched = int((~matched).sum())
    return readings[matched].reset_index(drop=True), sensors, report
