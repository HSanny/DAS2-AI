"""
das2.config
===========

Typed configuration, layered: dataclass defaults -> optional YAML file ->
environment variables.

Three rules this module exists to enforce.

**Durations are seconds, never sample counts.** The system being replaced
counted every window in samples, so `ROLL_WIN_Z = 24` meant a 6.3-minute
baseline on a sensor reporting every 15.7 s and a 49.8-minute baseline on one
reporting every 124 s -- an 8x difference in behaviour from a single constant,
chosen by nobody. Everything here that describes a length of time is named
`*_s` or `*_min` and is converted per sensor at the point of use.

**No cadence assumptions.** The analysis interval is whatever the scheduler
uses. Nothing here may assume 6-hourly (or hourly) runs; `window_hours` and
`continuity_gap_min` are independent knobs, and code that needs to know how far
apart two runs were must measure it, not infer it.

**Thresholds carry their provenance.** Several defaults are honest guesses, and
they are marked as such. `is_rule_invalid` in the old detector applied
"Pressure 0..20" to every pressure sensor in Singapore while the real fleet
contains pressure sensors with medians of 0.0034 and 3.90 -- a band useless for
both. Guessed values should be visibly guessed so they get replaced with the
plant's commissioned limits rather than quietly inherited forever.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# Env helpers
# --------------------------------------------------------------------------- #
def _env(name: str) -> str | None:
    raw = os.getenv(name)
    return raw if raw is not None and raw.strip() != "" else None


def _coerce(value: str, target_type: type) -> Any:
    if target_type is bool:
        return value.strip().lower() in ("1", "true", "yes", "on")
    if target_type is int:
        return int(float(value))          # tolerate "30.0" for an int field
    if target_type is float:
        return float(value)
    if target_type in (list, tuple, set):
        return [v.strip() for v in value.split(",") if v.strip()]
    return value


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
@dataclass
class IngestConfig:
    """Discovery and merging of the hourly Fujitsu CSV exports."""

    history_dir: str = "HISTORY"
    alarm_dir: str = "HISTALMEVT"
    histcurr_path: str = "HISTCURR/histcurr_fujitsu.csv"
    longlat_path: str = "LongLat.csv"
    raw_dir: str = "raw"

    #: Equipment classification rules. Empty means the packaged
    #: das2/data/equipment_rules.yaml, which is the normal case; pointing this
    #: at a copy lets the client retune classification without a rebuild.
    equipment_rules_path: str = ""

    @property
    def inventory_path(self) -> str:
        """
        The sensor inventory the run reads.

        Named separately from `histcurr_path` because "the inventory" is the
        concept the rest of the system depends on, while HISTCURR is merely the
        file Fujitsu happens to export it as today.
        """
        return self.histcurr_path

    #: How much history each analysis run looks at.
    window_hours: int = 72
    #: Allowance for the network share lagging behind wall clock.
    grace_minutes: int = 30
    #: Abort rather than analyse a stalled feed. Processing three-day-old data
    #: as though it were current is worse than not running at all.
    max_missing_percent: int = 25


@dataclass
class ResampleConfig:
    """
    Layer 0: the uniform grid that cross-sensor comparison requires.

    Needed because this telemetry is report-by-exception at 1-second
    resolution with no clock alignment -- 159,347 distinct timestamps over 71
    hours, spread evenly across every second-of-minute. Two sensors therefore
    share almost no timestamps at all, so correlating them is impossible until
    both are placed on a common grid.
    """

    grid_seconds: int = 60

    #: Step-hold, never mean. Under report-by-exception a value persists until
    #: the next report, so averaging a bucket fabricates readings the
    #: instrument never held and smears the step transitions that
    #: change-point detection depends on.
    method: str = "locf"

    #: Never interpolate across a gap longer than this; beyond it, the sensor's
    #: value is unknown rather than unchanged.
    max_hold_seconds: int = 3600


@dataclass
class HealthConfig:
    """
    Layer 1: sensor-health detectors, run on the RAW stream.

    These are physical facts needing no quantile calibration, which is what
    finally gives the system a null hypothesis -- the old detector had none and
    emitted ~10 sensors per run whether the network was healthy or on fire.
    """

    #: Unchanged for this long, while still reporting, is a stuck sensor.
    #: Gated on continuing to report: forward-filling a comms dropout onto the
    #: grid manufactures a perfect flatline, so FLATLINE and STALE must be
    #: distinguished by whether raw samples are still arriving.
    flatline_min_seconds: int = 1800

    #: Silence is judged against each sensor's OWN inter-arrival distribution,
    #: not a fleet-wide threshold: a stable report-by-exception sensor
    #: legitimately says nothing for long stretches, so a global timeout would
    #: page for every quiet healthy sensor.
    stale_quantile: float = 0.99
    stale_multiplier: float = 5.0
    stale_floor_seconds: int = 3600

    #: A range violation must clear the sensor's own noise before it counts.
    #: Without this, an idle flowmeter sitting at zero with symmetric noise
    #: reads negative about half the time: in a smoke run, one healthy meter
    #: produced 1079 flagged points and 542 events from `value < 0` alone.
    range_tolerance_sigma: float = 6.0

    #: Reverse flow is a real fault (backflow through a failed non-return
    #: valve), so it is typed rather than discarded -- but it must be sustained
    #: to distinguish it from noise around zero on an idle meter.
    reverse_flow_min_seconds: int = 300

    #: Resolution collapse: a sensor whose ADC degrades to a handful of
    #: distinct values. Not a flatline (variance > 0) and not a range
    #: violation, so nothing else catches it.
    quantisation_min_distinct_ratio: float = 0.01
    dithering_range_multiple: float = 3.0


@dataclass
class BaselineConfig:
    """
    Layer 2: expected value per time of day, from persisted history.

    A time-of-day profile rather than STL: a 72-hour window holds only three
    daily cycles, so STL would estimate each seasonal phase from ~3
    observations and absorb any real 24h-scale fault straight into the seasonal
    component. Many of these series are also duty-cycle square waves where a
    "seasonal component" is an artefact of which hours a pump happened to run.
    """

    profile_days: int = 28
    min_profile_days: int = 14
    bucket_minutes: int = 15
    split_weekend: bool = True

    #: Absorbs duty-cycle timing jitter. A pump starting 20 minutes late is
    #: normal operation but produces a large residual against an exact
    #: time-of-day expectation, so the residual is scored against the best
    #: match within this tolerance band.
    tolerance_minutes: int = 30

    #: Threshold on the robust residual score.
    #:
    #: NOTE: the incumbent's equivalent constants are not comparable. Its
    #: robust-Z applies the 1.4826 consistency constant to the NUMERATOR
    #: instead of scaling MAD, inflating every score by 1.4826^2 ~= 2.198x --
    #: so its `MAD_Z_THRESH = 4.5` is really 2.05 sigma. This value is in
    #: correctly-scaled sigma and must not be copied across from the old file.
    residual_sigma: float = 4.0

    #: Scale floor, mirroring the Phase 0 fix: without it, any excursion on a
    #: quiet or quantised sensor scores zero, because a flat neighbourhood has
    #: MAD 0. Verified: 300 zeros plus a spike to 250.0 scored exactly 0.000.
    rel_scale_floor: float = 1e-6


@dataclass
class ChangePointConfig:
    """CUSUM level-shift detection, replacing the before/after median heuristic."""

    threshold_sigma: float = 5.0
    drift_sigma: float = 0.5
    min_segment_seconds: int = 900


@dataclass
class SpatialConfig:
    """
    Clustering abnormal sensors by place -- the client's actual request.

    Distances are site-level: coordinates come from an RTU-number join, so
    every sensor on one RTU carries identical coordinates. A cluster therefore
    answers "which SITES went abnormal together", which is the right
    granularity for deciding where to send someone but cannot resolve position
    within a site.
    """

    #: Measured from the real LongLat.csv, not guessed -- see
    #: das2.spatial.cluster for the site-spacing and percolation figures. The
    #: earlier 2,000 m value reached only 68% of sites and failed to link the
    #: real 3.3-4.4 km spacing of an actual East-side event.
    cluster_radius_m: float = 5000.0
    #: A cluster wider than this is re-split: it is no longer one crew's job.
    max_cluster_diameter_m: float = 5000.0
    min_cluster_size: int = 2

    #: Only sensors whose anomaly windows overlap in time may join a cluster:
    #: co-located sensors failing a day apart are not one event.
    time_overlap_tolerance_min: int = 30

    #: Radius for pulling neighbouring sensors to correlate against. Wider than
    #: the cluster radius on purpose -- the question is whether the surrounding
    #: network moved, which needs context from outside the cluster.
    correlation_radius_m: float = 5000.0
    correlation_min_overlap_points: int = 30
    #: Above this, neighbours are judged to have moved together.
    correlation_strong: float = 0.6

    #: Beyond this from any known centroid, a coordinate is treated as
    #: unplaceable rather than forced into the nearest planning area.
    max_centroid_distance_m: float = 15000.0


@dataclass
class IncidentConfig:
    """Stateful incident lifecycle across runs."""

    #: Membership overlap needed to call a new cluster a continuation of an
    #: open incident. Membership drifts as an event spreads, so identity cannot
    #: require an exact match.
    member_jaccard_threshold: float = 0.4

    #: Independent of run cadence by design -- see the module docstring.
    continuity_gap_min: int = 180

    #: Clear an incident once nothing has been seen for this long.
    resolve_after_min: int = 720

    #: Re-announce an open incident only on a material change, or the
    #: per-run repetition this whole design exists to remove comes straight
    #: back.
    escalate_severity_delta: float = 20.0
    escalate_on_new_members: bool = True

    #: REGIONAL_EVENT requires corroboration from several sites and several
    #: equipment types; one panel misbehaving is fan-out, not a regional event.
    regional_min_sensors: int = 3
    regional_min_equipment_types: int = 2


@dataclass
class RainConfig:
    """
    Rain context from the client's OWN gauges.

    188 `*-Rainfall` sensors already exist in the feed and are currently
    discarded as unclassified. Using them needs no internet, no external API
    and no new dependency -- it is a matter of not throwing them away.
    """

    enabled: bool = True
    gauge_equipment: str = "Rainfall"
    search_radius_m: float = 10000.0
    #: Rain above this over the incident window is taken to explain a
    #: flow/level excursion, which turns a dispatch into a monitor.
    explains_mm: float = 5.0


@dataclass
class AlertConfig:
    """Incident-level alerting."""

    enabled: bool = True
    #: Per-region budget per run, replacing the old global top-10 rank cut.
    #: That cut emitted ten sensors whether the network was healthy or on fire,
    #: and would silently drop a genuine ten-site regional event.
    max_alerts_per_region: int = 5
    #: Absolute cap on messages per run, as a backstop against a bad run
    #: flooding the chat. Anything beyond it is summarised in one line and
    #: remains on the dashboard.
    max_incidents_per_run: int = 10
    min_priority: str = "P3"
    dashboard_base_url: str = ""
    feedback_buttons: bool = True

    #: Bot credentials. Set these via DAS2_ALERT_TELEGRAM_TOKEN and
    #: DAS2_ALERT_TELEGRAM_CHAT_ID rather than in a config file, so they never
    #: reach the repository.
    telegram_token: str = field(default="", repr=False)
    telegram_chat_id: str = ""


@dataclass
class ReportConfig:
    output_dir: str = "output_html"
    charts_dir: str = "output_plots"
    csv_dir: str = "output_csv"
    #: Interactive map tiles need a CDN. The page must still render its tables
    #: and the region-by-type matrix when that is unreachable, which is the
    #: normal state on an isolated operations network.
    map_tiles: bool = True

    @property
    def public_url(self) -> str:
        """Link included in alerts. Empty when the HTML is not served anywhere."""
        return self.dashboard_base_url if hasattr(self, "dashboard_base_url") else ""


@dataclass
class DatabaseConfig:
    enabled: bool = True
    url: str = ""                       # full SQLAlchemy URL; overrides the parts below
    host: str = ""
    port: int = 1433
    database: str = ""
    username: str = ""
    password: str = field(default="", repr=False)
    driver: str = "ODBC Driver 17 for SQL Server"
    schema: str = "dbo"

    @property
    def safe_url(self) -> str:
        """The URL with the password removed, for logs and the check command."""
        url = self.sqlalchemy_url()
        if self.password and self.password in url:
            url = url.replace(self.password, "***")
        from urllib.parse import quote_plus
        if self.password:
            url = url.replace(quote_plus(self.password), "***")
        return url

    def sqlalchemy_url(self) -> str:
        if self.url:
            return self.url
        from urllib.parse import quote_plus
        return (
            f"mssql+pyodbc://{quote_plus(self.username)}:{quote_plus(self.password)}"
            f"@{self.host}:{self.port}/{self.database}"
            f"?driver={quote_plus(self.driver)}&TrustServerCertificate=yes"
        )


@dataclass
class Config:
    """Root configuration."""

    ingest: IngestConfig = field(default_factory=IngestConfig)
    resample: ResampleConfig = field(default_factory=ResampleConfig)
    health: HealthConfig = field(default_factory=HealthConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    changepoint: ChangePointConfig = field(default_factory=ChangePointConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    incident: IncidentConfig = field(default_factory=IncidentConfig)
    rain: RainConfig = field(default_factory=RainConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)

    #: How often `das2 schedule` runs. Configurable by decision -- nothing in
    #: the code may assume a cadence, because every window and threshold is
    #: expressed in time rather than in runs or samples.
    run_interval_minutes: int = 60

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, path: str | Path | None = None, *, use_env: bool = True) -> "Config":
        """
        Build a Config from defaults, then an optional YAML/JSON file, then env.

        Later layers win. The file is optional so the package runs out of the
        box; env override exists so a container can be retuned without a
        rebuild.
        """
        cfg = cls()
        path = path or _env("DAS2_CONFIG")
        if path:
            cfg = cfg.merge(_read_config_file(Path(path)))
        if use_env:
            cfg.apply_env()
        return cfg

    def merge(self, data: dict[str, Any]) -> "Config":
        """Overlay a nested dict, ignoring unknown keys rather than failing."""
        for section_name, section_values in (data or {}).items():
            section = getattr(self, section_name, None)
            if section is None or not is_dataclass(section):
                continue
            if not isinstance(section_values, dict):
                continue
            valid = {f.name for f in fields(section)}
            for key, value in section_values.items():
                if key in valid:
                    setattr(section, key, value)
        return self

    def apply_env(self, prefix: str = "DAS2_") -> "Config":
        """
        Apply `DAS2_<SECTION>_<FIELD>` overrides, e.g. DAS2_SPATIAL_CLUSTER_RADIUS_M.

        A few widely-used credentials also accept the legacy DAS_* names the
        existing containers already set, so this package can be dropped into
        the current deployment without rewriting its environment.
        """
        for section_field in fields(self):
            section = getattr(self, section_field.name)
            if not is_dataclass(section):
                # A top-level scalar such as run_interval_minutes, overridden
                # as DAS2_RUN_INTERVAL_MINUTES. Walking only the sections
                # silently ignored these, so the documented
                # DAS2_RUN_INTERVAL_MINUTES had no effect at all and the
                # scheduler stayed on its default cadence whatever the
                # deployment asked for.
                raw = _env(f"{prefix}{section_field.name.upper()}")
                if raw is not None:
                    try:
                        setattr(self, section_field.name,
                                _coerce(raw, type(section)))
                    except (TypeError, ValueError):
                        pass
                continue
            for f in fields(section):
                env_name = f"{prefix}{section_field.name.upper()}_{f.name.upper()}"
                raw = _env(env_name)
                if raw is not None:
                    try:
                        setattr(section, f.name, _coerce(raw, f.type if isinstance(f.type, type) else type(getattr(section, f.name))))
                    except (TypeError, ValueError):
                        pass

        legacy = {
            ("database", "host"): "DAS_MSSQL_HOST",
            ("database", "database"): "DAS_MSSQL_DATABASE",
            ("database", "username"): "DAS_MSSQL_USERNAME",
            ("database", "password"): "DAS_MSSQL_PASSWORD",
        }
        for (section_name, field_name), env_name in legacy.items():
            raw = _env(env_name)
            if raw is not None:
                setattr(getattr(self, section_name), field_name, raw)
        port = _env("DAS_MSSQL_PORT")
        if port:
            try:
                self.database.port = int(port)
            except ValueError:
                pass
        return self

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["database"]["password"] = "***" if self.database.password else ""
        return data

    def __repr__(self) -> str:      # keep credentials out of logs and tracebacks
        return f"Config({json.dumps(self.to_dict(), indent=2, default=str)})"


def _read_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # optional dependency
        except ImportError as exc:
            raise RuntimeError(
                f"{path} is YAML but PyYAML is not installed. "
                f"Install pyyaml, or use a .json config file instead."
            ) from exc
        return yaml.safe_load(text) or {}
    return json.loads(text)


def load_config(path: str | Path | None = None) -> Config:
    """
    Load configuration: defaults, then the file, then environment overrides.

    The module-level entry point everything else uses, so there is exactly one
    place where "where does configuration come from" is answered.
    """
    return Config.load(path)
