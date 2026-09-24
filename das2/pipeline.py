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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from das2.config import Config
from das2.detect import conventional
from das2.detect.asset import asset_summary, run_asset_checks
from das2.detect.baseline import baseline_summary, score_window
from das2.detect.digital import run_digital_checks, run_pump_flow_checks
from das2.detect.fusion import fuse_all, fusion_summary
from das2.detect.health import run_health_checks
from das2.detect.massbalance import balance_summary, find_groups, run_mass_balance
from das2.detect.profile import build_profiles, profile_summary
from das2.detect.selection import select
from das2.incident.build import build_incidents, incident_summary
from das2.io.classify import get_classifier
from das2.io.ingest import load_all
from das2.models import Cluster, Incident, SensorAnomaly, SensorMeta
from das2.profile.build import TimeOfDayBaseline
from das2.spatial.correlation import cluster_correlation, correlation_summary
from das2.spatial.cluster import (
    cluster_by_asset,
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
    #: Per-cluster rain evidence: how much, over which window, on whose
    #: gauges, and whether a lag was measurable. Keyed the same way as
    #: `rainfall_by_cluster` so an incident can find its own.
    rain_context: dict = field(default_factory=dict)
    #: Per sensor, what `mean ± 3σ` over the same window would have said.
    #: Reporting only -- nothing in detection, triage or selection reads it.
    conventional: dict = field(default_factory=dict)
    correlations: dict[str, float] = field(default_factory=dict)
    neighbour_results: list = field(default_factory=list)
    selected: list[Incident] = field(default_factory=list)
    held: list = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0

    @property
    def alertable(self) -> list[Incident]:
        """
        What will actually be sent.

        Once selection has run this is its output, not merely "everything whose
        class is alertable" -- otherwise the per-region budgets would be
        computed and then ignored by the sender.
        """
        if self.selected:
            return self.selected
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
        open_incidents: list[Incident] | None = None,
        baselines: dict[str, TimeOfDayBaseline] | None = None) -> RunResult:
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
    # The code identity goes in the FIRST line of every run. A container
    # running yesterday's image looks exactly like one running today's, and
    # finding that out by noticing an absent log line requires already knowing
    # which commit added it.
    from das2 import build_stamp
    log.info("run %s starting -- das2 source %s", run_id, build_stamp())

    result = RunResult(run_id=run_id, started_at=now)

    # --- ingest ------------------------------------------------------------- #
    # The window is a HARD limit on what is read, not a filter applied after.
    #
    # This was omitted, and `load_all` defaults to `since=None`, so every run
    # read every HISTORY file in the directory. On the fixture that is
    # invisible -- it holds exactly 72 hours -- but the client's share holds
    # 10,334 hourly files, about 431 days, which is roughly 1.1 BILLION rows
    # against the 7.9 million a 72-hour window wants: 144x more data than the
    # analysis asks for, read into pandas on every run.
    #
    # DAS2_INGEST_WINDOW_HOURS was therefore documented, configurable, and
    # completely inert -- the same defect DAS2_RUN_INTERVAL_MINUTES had.
    #
    # grace_minutes is subtracted as well because the share lags wall clock:
    # the file for the current hour may not have landed, and reaching exactly
    # `window_hours` back would silently analyse one hour less than asked.
    since = now - timedelta(hours=config.ingest.window_hours,
                            minutes=config.ingest.grace_minutes)
    log.info("window: %s -> %s (%d h)", since, now, config.ingest.window_hours)
    readings, sensors, report = load_all(
        config.ingest.history_dir,
        config.ingest.inventory_path,
        config.ingest.longlat_path,
        since=since,
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

    # Stored time-of-day baselines from the daily job, if it has run. Without
    # them the L2 residual layer abstains, which is stated rather than silent.
    baselines = baselines or {}
    result.stats["baselines"] = baseline_summary(baselines)
    if not baselines:
        log.info("no stored time-of-day baselines -- L2 residual scoring is "
                 "inactive until the daily profile job has run")

    # --- detect ------------------------------------------------------------- #
    meta_by_key = {str(r["sensor_key"]): _sensor_meta(r) for _, r in sensors.iterrows()}
    ranges = _range_lookup(classifier)

    signals_by_sensor: dict[str, tuple[SensorMeta, list]] = {}
    series: dict[str, tuple[pd.Series, Any]] = {}
    for key, group in readings.groupby("sensor_key", sort=False):
        meta = meta_by_key.get(str(key))
        if meta is None or not _analysable(meta, sensors, key):
            continue
        group = group.sort_values("ts")
        values = group["value"].to_numpy(dtype=float)
        series[str(key)] = (group["ts"], values)
        kind = _kind_of(sensors, key)
        lo, hi = ranges.get(meta.equipment, (None, None))

        if _is_digital(meta, kind):
            # Binary state signals get counting and consistency checks, not the
            # analog stack. v1 ran ~1,567 of these through robust-Z and DTW on
            # a 0/1 series, which can only produce silence or noise.
            signals = run_digital_checks(group["ts"], values, profiles.get(str(key)))
        else:
            signals = run_health_checks(
                group["ts"], values, profiles.get(str(key)),
                equipment_kind=kind,
                range_min=lo, range_max=hi,
                unit=meta.unit or "",
                is_flow=meta.equipment == "Flowrate",
                window_end=result.window_end,
            )
            # L2: score against the stored time-of-day baseline, when the daily
            # profile job has built one. Abstains otherwise rather than
            # inventing a baseline from this window.
            signals += score_window(
                group["ts"], values, baselines.get(str(key)),
                unit=meta.unit or "",
                resolution=(profiles[str(key)].resolution
                            if str(key) in profiles else 0.0),
            )
        if signals:
            signals_by_sensor[str(key)] = (meta, signals)

    # --- pump vs its own discharge flow -------------------------------------- #
    # The other cross-signal detector. Needs pump-to-flowmeter pairing, which
    # nothing in the feed declares, so it is recovered from the descriptions.
    for run_key, pump_signals in run_pump_flow_checks(sensors, series).items():
        meta = meta_by_key.get(run_key)
        if meta is None:
            continue
        existing = signals_by_sensor.get(run_key)
        if existing:
            existing[1].extend(pump_signals)
        else:
            signals_by_sensor[run_key] = (meta, pump_signals)

    # --- the machine, not the instrument and not the water ------------------ #
    # Cross-channel contradictions at one unit: running and not delivering,
    # energised while off, commanded on and drawing nothing. Emitted on the
    # run-state sensor, which is the point an operator recognises.
    for run_key, asset_signals in run_asset_checks(sensors, series).items():
        meta = meta_by_key.get(run_key)
        if meta is None:
            continue
        existing = signals_by_sensor.get(run_key)
        if existing:
            existing[1].extend(asset_signals)
        else:
            signals_by_sensor[run_key] = (meta, asset_signals)
    result.stats["assets"] = asset_summary(sensors)
    log.info("assets: %s", result.stats["assets"])

    # --- mass balance: the one genuinely multivariate detector -------------- #
    # Grouped per site from level + inflow + outflow. Abstains wherever the
    # group is not actually a closed system, which the fit quality decides.
    groups = find_groups(sensors)
    result.stats["mass_balance"] = balance_summary(groups)
    for level_key, mb_signals in run_mass_balance(sensors, series).items():
        meta = meta_by_key.get(level_key)
        if meta is None:
            continue
        existing = signals_by_sensor.get(level_key)
        if existing:
            existing[1].extend(mb_signals)
        else:
            signals_by_sensor[level_key] = (meta, mb_signals)

    result.anomalies = fuse_all(signals_by_sensor,
                                window_start=result.window_start,
                                window_end=result.window_end)
    result.stats["detection"] = fusion_summary(result.anomalies)
    log.info("detection: %s", result.stats["detection"])

    # --- what a median-and-sigma check would have said ----------------------- #
    # The client verifies this system by asking whether it found anything his
    # existing statistics did not. So run his check -- mean, sigma, 3 sigma, on
    # the same arrays -- and record the answer per sensor, INCLUDING where it
    # would have fired and we added nothing. Nothing downstream reads it; see
    # das2/detect/conventional.py for why that separation is not negotiable.
    spans_by_sensor: dict[str, list] = {}
    for anomaly in result.anomalies:
        spans_by_sensor.setdefault(anomaly.sensor.sensor_key, []).append(
            (anomaly.start, anomaly.end))
    result.conventional = conventional.evaluate_all(
        series, spans_by_sensor=spans_by_sensor)
    result.stats["conventional"] = conventional.summarise(result.conventional)
    log.info("median±3σ check: %s", result.stats["conventional"])

    # --- rain context -------------------------------------------------------- #
    # Sourced from the client's own 188 rain gauges, which v1 discarded as
    # unclassified. No external API, no internet dependency.
    provider = None
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
    # An anomaly may page if its equipment class is trusted to alert, OR if the
    # finding itself is strong enough not to need that -- see
    # ALWAYS_PAGEABLE_TYPES. Without the second clause every digital and
    # cross-signal detector is silenced, because Pump, Valve and DigitalStatus
    # all start `alertable: false`.
    from das2.models import ALWAYS_PAGEABLE_TYPES
    pageable = [a for a in result.anomalies
                if a.sensor.alertable or a.dominant_type in ALWAYS_PAGEABLE_TYPES]
    # Machines first, and taken OUT of the spatial pass. An asset finding knows
    # which unit it belongs to, so grouping it by coordinates can only lose
    # that -- and does: on the fixture a failed pump was swallowed by an
    # unrelated regional event 400 m away and reported as "investigate the
    # area".
    asset_clusters, consumed = cluster_by_asset(pageable)
    consumed_keys = {a.sensor.sensor_key for a in consumed}
    spatial = [a for a in pageable if a.sensor.sensor_key not in consumed_keys]

    result.clusters = cluster_anomalies(spatial, params) + asset_clusters
    result.loose = unclustered(spatial, result.clusters)
    result.stats["clustering"] = cluster_summary(result.clusters, result.loose)
    log.info("clustering: %s", result.stats["clustering"])

    # --- neighbour correlation ------------------------------------------------ #
    # The single most decision-relevant signal: neighbours moving together means
    # the water moved, neighbours flat means the instrument is lying. Computed
    # per cluster, because that is what triage asks about.
    all_meta = list(meta_by_key.values())
    for cluster in result.clusters:
        median_r, pairs = cluster_correlation(
            cluster, all_meta, series,
            radius_m=config.spatial.correlation_radius_m)
        if median_r is not None:
            result.correlations[",".join(sorted(cluster.sensor_keys))] = median_r
        result.neighbour_results.extend(pairs)
    for anomaly in result.loose:
        from das2.spatial.correlation import correlate_anomaly
        pairs = correlate_anomaly(anomaly, all_meta, series,
                                  radius_m=config.spatial.correlation_radius_m)
        values = [p.pearson_r for p in pairs if p.pearson_r is not None]
        if values:
            import numpy as _np
            result.correlations[anomaly.sensor.sensor_key] = round(
                float(_np.median(_np.abs(values))), 3)
        result.neighbour_results.extend(pairs)
    result.stats["correlation"] = correlation_summary(result.neighbour_results)
    log.info("correlation: %s", result.stats["correlation"])

    # --- incidents ------------------------------------------------------------ #
    # Rainfall per incident, over ITS OWN window and near ITS OWN centroid.
    # `rainfall_by_region` is a whole-run total kept for the dashboard header;
    # using it to classify would let rain at any hour excuse an event at any
    # other hour.
    # The window is shifted back by a LEARNED lag, not the incident's own.
    #
    # Rain that raises a level at 14:30 fell before 14:30. Integrating over the
    # incident's window -- minutes, for a level shift -- asked whether it was
    # raining at the instant the level moved, which is not how a catchment
    # behaves. `das2.weather.context` measures the lag per cluster by
    # cross-correlating rainfall against the members' rate of change, and
    # declines to answer when the window was dry or the sensors still.
    rainfall_by_cluster: dict[str, float] = {}
    rain_context: dict[str, "RainObservation"] = {}
    if config.rain.enabled and provider is not None and provider.available:
        from das2.weather.context import observe_cluster, summarise

        for cluster in result.clusters:
            key = ",".join(sorted(cluster.sensor_keys))
            observed = observe_cluster(provider, cluster, readings, since, now)
            rain_context[key] = observed
            if observed.mm is not None:
                rainfall_by_cluster[key] = observed.mm
        for anomaly in result.loose:
            # A single sensor has no catchment to learn a lag from, so it gets
            # the unshifted window and says so through its own observation.
            observed = provider.observe(anomaly.sensor.latitude,
                                        anomaly.sensor.longitude,
                                        anomaly.start, anomaly.end)
            rain_context[anomaly.sensor.sensor_key] = observed
            if observed.mm is not None:
                rainfall_by_cluster[anomaly.sensor.sensor_key] = observed.mm

        result.rain_context = rain_context
        result.stats["rain_context"] = summarise(rain_context)
        log.info("rain context: %s", result.stats["rain_context"])

    candidates = build_incidents(result.clusters, now=now, loose=result.loose,
                                 correlations=result.correlations,
                                 rainfall=rainfall_by_cluster)

    from das2.incident import signature
    signature_matches: list = []

    # Attach the rain PROVENANCE, not just the number. A millimetre figure on
    # its own cannot be checked: a reader disagreeing with a WEATHER_DRIVEN
    # verdict needs to see how many gauges, how far away, over which window,
    # and whether a lag was measurable at all.
    for candidate in candidates:
        key = (",".join(sorted(candidate.cluster.sensor_keys))
               if len(candidate.cluster.members) > 1
               else next(iter(candidate.cluster.sensor_keys), ""))
        observed = rain_context.get(key)
        context = "unknown"
        if observed is not None and observed.known:
            candidate.detail["rain_evidence"] = observed.describe()
            candidate.detail["rain_context"] = observed.context
            context = observed.context

        # What the moving parameters usually mean. DESCRIPTIVE: it is attached
        # to the incident and never consulted when deciding its class,
        # priority or recommendation. A wrong evidence class costs a wasted
        # trip; a wrong signature able to stop a dispatch costs a flood.
        found = signature.match(candidate, rain_context=context)
        if found is not None:
            candidate.detail[signature.DETAIL_KEY] = signature.as_detail(found)
        signature_matches.append(found)

        # And what their own median-and-sigma check makes of the same sensors,
        # so the report can be checked against the spreadsheet it is asking to
        # be trusted over.
        panel = conventional.for_incident(candidate, result.conventional)
        if panel:
            candidate.detail["conventional"] = panel

    result.stats["signatures"] = signature.summary(signature_matches)
    log.info("signatures: %s", result.stats["signatures"])

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

    # --- selection ------------------------------------------------------------- #
    # Per-region budgets, replacing v1's global top-10 rank cut. Nothing is
    # discarded: what is held back stays on the dashboard and in the database,
    # and the reason is recorded.
    from das2.models import Priority
    chosen = select(
        result.incidents,
        per_region=config.alert.max_alerts_per_region,
        global_cap=config.alert.max_incidents_per_run,
        min_priority=_priority(config.alert.min_priority),
    )
    result.selected, result.held = chosen.selected, chosen.held
    result.stats["selection"] = chosen.summary()
    log.info("selection: %s", result.stats["selection"])

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


def _is_digital(meta: SensorMeta, kind: str) -> bool:
    """
    Binary state signals, which need counting checks rather than statistics.

    Both the declared signal type and the equipment kind are consulted, because
    the classifier reaches these two ways: RawType marks the signal Digital, and
    the rule table marks pump/valve classes as `status`.
    """
    return (str(meta.signal_type).lower() == "digital"
            or kind == "status"
            or meta.equipment in ("Pump", "Valve", "DigitalStatus"))


def _priority(name: str):
    from das2.models import Priority
    try:
        return Priority(str(name).upper())
    except ValueError:
        return Priority.P3


def _analysable(meta: SensorMeta, sensors: pd.DataFrame, key) -> bool:
    kind = _kind_of(sensors, key)
    return kind not in ("config",) and meta.equipment != "UNCLASSIFIED"
