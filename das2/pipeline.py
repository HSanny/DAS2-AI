"""
das2.pipeline
=============

One analysis run, end to end: CSVs in, incidents out.

    ingest -> classify -> profile -> detect -> fuse -> cluster -> triage

The whole point of the ordering is that each stage narrows the question. The
detectors ask "is this number wrong?", fusion asks "what is wrong with this
sensor?", clustering asks "did several places go wrong together?", and triage
asks the only question the client actually cares about: *do we need to drive
there?*

Everything here is pure computation over a window. Nothing in this module talks
to a database, Telegram, or the filesystem beyond reading the input CSVs, so a
run can be executed and inspected offline -- which is how it is tested, and how
the client can dry-run it against real data before pointing it at production.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from das2.config import Config
from das2.detect.fusion import fuse_all, fusion_summary
from das2.detect.health import run_health_checks
from das2.detect.profile import build_profiles, profile_summary
from das2.incident.build import build_incidents, incident_summary
from das2.io.classify import get_classifier
from das2.io.ingest import load_all
from das2.models import Cluster, Incident, SensorAnomaly, SensorMeta
from das2.spatial.cluster import (
    ClusterParams,
    cluster_anomalies,
    cluster_summary,
    region_equipment_matrix,
    unclustered,
)
from das2.weather.provider import InternalRainGaugeProvider

log = logging.getLogger("das2.pipeline")


@dataclass
class RunResult:
    """Everything one run produced, for reporting, persistence and tests."""

    run_id: str
    started_at: datetime
    window_start: datetime | None = None
    window_end: datetime | None = None
    sensors: pd.DataFrame = field(default_factory=pd.DataFrame)
    readings: pd.DataFrame = field(default_factory=pd.DataFrame)
    anomalies: list[SensorAnomaly] = field(default_factory=list)
    clusters: list[Cluster] = field(default_factory=list)
    loose: list[SensorAnomaly] = field(default_factory=list)
    incidents: list[Incident] = field(default_factory=list)
    rainfall_by_region: dict[str, float] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0

    @property
    def alertable(self) -> list[Incident]:
        return [i for i in self.incidents if i.should_alert]

    @property
    def region_matrix(self) -> dict[str, dict[str, int]]:
        return region_equipment_matrix(self.anomalies)


def _sensor_meta(row: pd.Series) -> SensorMeta:
    """One inventory row -> SensorMeta, tolerating the columns that can be NA."""
    def opt(name, cast=None):
        value = row.get(name)
        if value is None or pd.isna(value):
            return None
        return cast(value) if cast else value

    return SensorMeta(
        sensor_key=str(row["sensor_key"]),
        description=str(row.get("description") or ""),
        equipment=str(row.get("equipment") or "UNCLASSIFIED"),
        signal_type=str(row.get("signal_type") or "Analog"),
        rtu_number=opt("rtu_number", str),
        site=opt("site", str),
        latitude=opt("latitude", float),
        longitude=opt("longitude", float),
        planning_area=opt("planning_area", str),
        region=opt("region", str),
        unit=opt("unit", str) or "",
        alertable=bool(row.get("alertable", True)),
    )


def run(config: Config, *, now: datetime | None = None,
        open_incidents: list[Incident] | None = None) -> RunResult:
    """
    Execute one analysis run.

    `open_incidents` carries the incidents that were still open at the end of
    the previous run. Passing them in is what gives an incident a stable id
    across runs, and therefore what stops one three-day fault from producing
    twelve Telegram messages. Omitting them is valid -- every incident is then
    new -- which is the right behaviour for a one-off dry run.
    """
    started = time.time()
    now = now or datetime.now()
    run_id = now.strftime("%Y%m%d-%H%M%S")
    log.info("run %s starting", run_id)

    result = RunResult(run_id=run_id, started_at=now)

    # --- ingest ------------------------------------------------------------- #
    readings, sensors, report = load_all(
        config.ingest.history_dir,
        config.ingest.inventory_path,
        config.ingest.longlat_path,
    )
    result.readings, result.sensors = readings, sensors
    result.stats["ingest"] = report.as_dict() if hasattr(report, "as_dict") else vars(report)
    log.info("ingest: %d readings over %d sensors", len(readings), len(sensors))

    if readings.empty:
        log.warning("no readings in window -- nothing to analyse")
        result.duration_s = time.time() - started
        return result

    result.window_start = readings["ts"].min()
    result.window_end = readings["ts"].max()

    classifier = get_classifier(config.ingest.equipment_rules_path or None)
    result.stats["coverage"] = classifier.coverage_report(
        zip(sensors["description"], sensors.get("rawtype", [None] * len(sensors)))
    )

    # --- profiles ----------------------------------------------------------- #
    # Built from the window itself. That is weaker than the daily job over 28
    # days -- see das2.detect.profile -- but it is what a standalone run has,
    # and the flatline detector correctly abstains when a profile is too thin
    # to trust rather than guessing.
    profiles = build_profiles(readings)
    result.stats["profiles"] = profile_summary(profiles)
    log.info("profiles: %s", result.stats["profiles"])

    # --- detect ------------------------------------------------------------- #
    meta_by_key = {str(r["sensor_key"]): _sensor_meta(r) for _, r in sensors.iterrows()}
    ranges = _range_lookup(classifier)

    signals_by_sensor: dict[str, tuple[SensorMeta, list]] = {}
    for key, group in readings.groupby("sensor_key", sort=False):
        meta = meta_by_key.get(str(key))
        if meta is None or not _analysable(meta, sensors, key):
            continue
        group = group.sort_values("ts")
        lo, hi = ranges.get(meta.equipment, (None, None))
        signals = run_health_checks(
            group["ts"], group["value"].to_numpy(dtype=float),
            profiles.get(str(key)),
            equipment_kind=_kind_of(sensors, key),
            range_min=lo, range_max=hi,
            unit=meta.unit or "",
            is_flow=meta.equipment == "Flowrate",
            window_end=result.window_end,
        )
        if signals:
            signals_by_sensor[str(key)] = (meta, signals)

    result.anomalies = fuse_all(signals_by_sensor,
                                window_start=result.window_start,
                                window_end=result.window_end)
    result.stats["detection"] = fusion_summary(result.anomalies)
    log.info("detection: %s", result.stats["detection"])

    # --- rain context -------------------------------------------------------- #
    # Sourced from the client's own 188 rain gauges, which v1 discarded as
    # unclassified. No external API, no internet dependency.
    if config.rain.enabled:
        provider = InternalRainGaugeProvider(readings, sensors)
        result.rainfall_by_region = provider.rainfall_by_region(
            result.window_start, result.window_end)
        log.info("rain: %s", result.rainfall_by_region)

    # --- cluster ------------------------------------------------------------- #
    params = ClusterParams(
        radius_m=config.spatial.cluster_radius_m,
        max_diameter_m=config.spatial.max_cluster_diameter_m,
        time_tolerance_min=config.spatial.time_overlap_tolerance_min,
        min_size=config.spatial.min_cluster_size,
    )
    # Only alertable classes may form an incident. Everything else is still
    # detected and still appears on the dashboard, which is how a newly
    # classified equipment type accrues evidence before it is allowed to page
    # anyone -- the alternative, with coverage now ~4x wider than v1, is an
    # alert volume that gets the system switched off in a week.
    pageable = [a for a in result.anomalies if a.sensor.alertable]
    result.clusters = cluster_anomalies(pageable, params)
    result.loose = unclustered(pageable, result.clusters)
    result.stats["clustering"] = cluster_summary(result.clusters, result.loose)
    log.info("clustering: %s", result.stats["clustering"])

    # --- incidents ------------------------------------------------------------ #
    candidates = build_incidents(result.clusters, now=now, loose=result.loose,
                                 rainfall=result.rainfall_by_region)

    if open_incidents:
        from das2.incident.build import reconcile
        new, updated, resolved = reconcile(
            candidates, open_incidents, now=now,
            jaccard_min=config.incident.member_jaccard_threshold,
            continuity_hours=config.incident.resolve_after_min / 60.0,
        )
        result.incidents = new + updated
        result.stats["lifecycle"] = {"new": len(new), "updated": len(updated),
                                     "resolved": len(resolved)}
    else:
        result.incidents = candidates
        result.stats["lifecycle"] = {"new": len(candidates), "updated": 0, "resolved": 0}

    result.incidents.sort(key=lambda i: -i.severity)
    result.stats["incidents"] = incident_summary(result.incidents)
    log.info("incidents: %s", result.stats["incidents"])

    result.duration_s = round(time.time() - started, 2)
    log.info("run %s finished in %.2fs", run_id, result.duration_s)
    return result


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _range_lookup(classifier) -> dict[str, tuple[float | None, float | None]]:
    """
    Per-equipment physical limits from the rule file.

    These are fleet-wide defaults and are the weakest input the system has --
    applying "Pressure 0-20" to every pressure sensor in Singapore is close to
    meaningless when real sensors have medians of 0.0034 and 3.90. The range
    detector deadbands them against each sensor's own noise for exactly that
    reason. Replacing them with the plant's commissioned HH/H/L/LL limits is
    the single highest-value data request open with the client.
    """
    return {name: (cls.range_min, cls.range_max)
            for name, cls in classifier.classes.items()}


def _kind_of(sensors: pd.DataFrame, key) -> str:
    row = sensors.loc[sensors["sensor_key"] == key, "kind"]
    return str(row.iloc[0]) if len(row) else "measurement"


def _analysable(meta: SensorMeta, sensors: pd.DataFrame, key) -> bool:
    kind = _kind_of(sensors, key)
    return kind not in ("config",) and meta.equipment != "UNCLASSIFIED"
