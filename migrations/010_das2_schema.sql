-- 010_das2_schema.sql
-- ---------------------------------------------------------------------------
-- Schema for the das2 rewrite.
--
-- ADDITIVE. Nothing here drops or alters the v1 tables (dim, data, inst,
-- linktable, dateDim, alarmevent, abnormal_sensor_history). Both systems must
-- run side by side through the shadow comparison, and the v1 tables stay the
-- source of truth until the new detector clears its ship gates.
--
-- Design notes
-- ------------
-- * `sensor` + `reading` collapse the v1 star schema (inst / linktable /
--   dateDim). That indirection existed for a BI tool; a `dateDim` view can be
--   re-exposed if anything still needs it. Joining `inst` to `linktable` on a
--   string-concatenated Hkey+timestamp key was also a correctness hazard --
--   duplicate timestamps for one sensor collapse in drop_duplicates() and then
--   fan out at the merge.
--
-- * Anomalies carry a TYPE, not just a score. v1 ranked sensors by Peak_RZ,
--   which is not comparable across sensors: a 1.8% voltage excursion scored
--   109.7 while a full-scale flow event scored 306.9, because the first
--   sensor's MAD was pinned at its 0.1 V quantisation step. Severity here is
--   stored in engineering units and as a fraction of instrument span.
--
-- * `incident` is the unit of alerting, and it persists across runs. v1
--   alerted per sensor per run over a 72h window re-run every 6h, so one
--   three-day fault produced up to twelve Telegram messages.
--
-- * No alarm_event table: SCADA alarm events are out of scope by decision.
--
-- Idempotent. Portable between SQL Server and SQLite (used by the tests), so
-- it avoids T-SQL-only syntax: no GO batches, no IF EXISTS blocks, no
-- NVARCHAR/DATETIMEOFFSET/BIT. Text and numeric affinities behave sensibly on
-- both; SQL Server maps TEXT-ish columns via VARCHAR below.
-- ---------------------------------------------------------------------------

-- Sensor inventory -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS das2_sensor (
    sensor_key        VARCHAR(200) NOT NULL PRIMARY KEY,
    description       VARCHAR(400) NOT NULL,
    equipment         VARCHAR(64)  NOT NULL,
    signal_type       VARCHAR(16),            -- Analog | Digital
    -- measurement | counter | config | status. Drives which detectors run:
    -- a flat kWh counter means the pump is off, not that the sensor is stuck,
    -- and a setpoint's value is an operator decision with nothing to detect.
    kind              VARCHAR(16),
    rtu_number        VARCHAR(32),
    site              VARCHAR(128),           -- prefix parsed from description
    latitude          DOUBLE PRECISION,       -- RTU-level: all sensors on one
    longitude         DOUBLE PRECISION,       -- RTU share these coordinates
    planning_area     VARCHAR(64),            -- indicative only, see below
    region            VARCHAR(32),            -- the level clustering trusts
    placement_source  VARCHAR(16),            -- coordinates | site-name | none
    unit              VARCHAR(32),
    span_min          DOUBLE PRECISION,
    span_max          DOUBLE PRECISION,
    alertable         INTEGER DEFAULT 1,      -- new classes start at 0
    first_seen        TIMESTAMP,
    last_seen         TIMESTAMP
);
-- planning_area is stored but NOT authoritative: nearest-centroid assignment is
-- ambiguous near boundaries (BedokPS resolves to Paya Lebar, not Bedok -- both
-- East). Cluster and alert on region; show planning_area as a label only.
CREATE INDEX IF NOT EXISTS ix_das2_sensor_region ON das2_sensor (region);
CREATE INDEX IF NOT EXISTS ix_das2_sensor_equipment ON das2_sensor (equipment);
CREATE INDEX IF NOT EXISTS ix_das2_sensor_site ON das2_sensor (site);

-- Time series ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS das2_reading (
    sensor_key   VARCHAR(200) NOT NULL,
    ts           TIMESTAMP    NOT NULL,
    value        DOUBLE PRECISION,
    PRIMARY KEY (sensor_key, ts)
);
CREATE INDEX IF NOT EXISTS ix_das2_reading_ts ON das2_reading (ts);

-- Persisted time-of-day profile ----------------------------------------------
-- Built by the daily job over >=14 days (28 preferred). The hourly run scores
-- against this rather than deriving a baseline from its own window: a 72h
-- window holds only three daily cycles, too few to separate seasonality from a
-- genuine 24h-scale fault.
CREATE TABLE IF NOT EXISTS das2_sensor_profile (
    sensor_key     VARCHAR(200) NOT NULL,
    bucket_of_day  INTEGER      NOT NULL,   -- e.g. 0..95 for 15-minute buckets
    is_weekend     INTEGER      NOT NULL,
    median_value   DOUBLE PRECISION,
    mad_value      DOUBLE PRECISION,
    n_samples      INTEGER,
    updated_at     TIMESTAMP,
    PRIMARY KEY (sensor_key, bucket_of_day, is_weekend)
);

-- Runs -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS das2_detection_run (
    run_id            VARCHAR(64) NOT NULL PRIMARY KEY,
    started_at        TIMESTAMP,
    finished_at       TIMESTAMP,
    window_start      TIMESTAMP,
    window_end        TIMESTAMP,
    sensors_analysed  INTEGER,
    sensors_skipped   INTEGER,
    anomalies_found   INTEGER,
    incidents_open    INTEGER,
    detector_version  VARCHAR(32),   -- lets v1 and v2 share this table in shadow mode
    status            VARCHAR(32),
    notes             VARCHAR(1000)
);

-- Per-sensor findings --------------------------------------------------------
CREATE TABLE IF NOT EXISTS das2_sensor_anomaly (
    anomaly_id       VARCHAR(64)  NOT NULL PRIMARY KEY,
    run_id           VARCHAR(64)  NOT NULL,
    sensor_key       VARCHAR(200) NOT NULL,
    start_ts         TIMESTAMP,
    end_ts           TIMESTAMP,
    dominant_type    VARCHAR(48),            -- FLATLINE, LEVEL_SHIFT, ...
    -- Physical severity: checkable by an engineer, comparable across equipment.
    deviation        DOUBLE PRECISION,
    deviation_unit   VARCHAR(32),
    span_fraction    DOUBLE PRECISION,
    duration_s       DOUBLE PRECISION,
    window_fraction  DOUBLE PRECISION,
    severity_score   DOUBLE PRECISION,       -- derived ordering only
    signals_json     TEXT,                   -- full evidence, kept for audit
    plot_path        VARCHAR(500)
);
CREATE INDEX IF NOT EXISTS ix_das2_anomaly_run ON das2_sensor_anomaly (run_id);
CREATE INDEX IF NOT EXISTS ix_das2_anomaly_sensor ON das2_sensor_anomaly (sensor_key, start_ts);
CREATE INDEX IF NOT EXISTS ix_das2_anomaly_type ON das2_sensor_anomaly (dominant_type);

-- Incidents ------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS das2_incident (
    incident_id       VARCHAR(64) NOT NULL PRIMARY KEY,
    incident_class    VARCHAR(48),           -- REGIONAL_EVENT, SENSOR_FAULT, ...
    status            VARCHAR(16),           -- OPEN | UPDATED | RESOLVED
    severity          DOUBLE PRECISION,
    priority          VARCHAR(4),            -- P1..P4
    region            VARCHAR(32),
    centroid_lat      DOUBLE PRECISION,
    centroid_lon      DOUBLE PRECISION,
    radius_m          DOUBLE PRECISION,
    sensor_count      INTEGER,
    site_count        INTEGER,
    equipment_types   VARCHAR(400),
    -- The single most decision-relevant field: neighbours moving together means
    -- the water moved; neighbours flat means the instrument is lying.
    neighbour_correlation DOUBLE PRECISION,
    rainfall_mm       DOUBLE PRECISION,      -- from the client's own gauges
    narrative         TEXT,
    recommendation    VARCHAR(500),
    opened_at         TIMESTAMP,
    last_seen_at      TIMESTAMP,
    resolved_at       TIMESTAMP,
    ack_state         VARCHAR(16) DEFAULT 'NONE',
    ack_by            VARCHAR(100),
    ack_at            TIMESTAMP,
    ack_note          VARCHAR(500)
);
CREATE INDEX IF NOT EXISTS ix_das2_incident_status ON das2_incident (status, last_seen_at);
CREATE INDEX IF NOT EXISTS ix_das2_incident_region ON das2_incident (region, status);

CREATE TABLE IF NOT EXISTS das2_incident_member (
    incident_id   VARCHAR(64)  NOT NULL,
    sensor_key    VARCHAR(200) NOT NULL,
    anomaly_id    VARCHAR(64),
    first_seen_at TIMESTAMP,
    last_seen_at  TIMESTAMP,
    contribution  DOUBLE PRECISION,
    PRIMARY KEY (incident_id, sensor_key)
);

-- Lifecycle audit trail. An operator asking "why did this page me at 3am?"
-- should get an answer from the data, not from reading the code.
CREATE TABLE IF NOT EXISTS das2_incident_event (
    event_id     VARCHAR(64) NOT NULL PRIMARY KEY,
    incident_id  VARCHAR(64) NOT NULL,
    ts           TIMESTAMP,
    event_type   VARCHAR(32),   -- opened|escalated|spread|updated|resolved|acknowledged
    detail       TEXT
);
CREATE INDEX IF NOT EXISTS ix_das2_incident_event ON das2_incident_event (incident_id, ts);

-- Neighbour correlation ------------------------------------------------------
CREATE TABLE IF NOT EXISTS das2_neighbour_correlation (
    run_id        VARCHAR(64)  NOT NULL,
    incident_id   VARCHAR(64)  NOT NULL,
    sensor_key    VARCHAR(200) NOT NULL,
    neighbour_key VARCHAR(200) NOT NULL,
    distance_m    DOUBLE PRECISION,
    pearson_r     DOUBLE PRECISION,
    spearman_r    DOUBLE PRECISION,
    n_points      INTEGER,
    PRIMARY KEY (run_id, incident_id, sensor_key, neighbour_key)
);

-- Rain context, from the client's own 188 rain gauges (no external API) -------
CREATE TABLE IF NOT EXISTS das2_rain_observation (
    sensor_key  VARCHAR(200) NOT NULL,
    window_start TIMESTAMP   NOT NULL,
    window_end   TIMESTAMP   NOT NULL,
    total_mm     DOUBLE PRECISION,
    peak_rate_mm_h DOUBLE PRECISION,
    PRIMARY KEY (sensor_key, window_start, window_end)
);

-- Alert delivery -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS das2_alert_delivery (
    delivery_id   VARCHAR(64) NOT NULL PRIMARY KEY,
    incident_id   VARCHAR(64) NOT NULL,
    channel       VARCHAR(32),
    sent_at       TIMESTAMP,
    message_id    VARCHAR(64),
    response_code INTEGER,
    payload_hash  VARCHAR(64),   -- suppresses resending an unchanged alert
    suppressed    INTEGER DEFAULT 0,
    suppress_reason VARCHAR(200)
);
CREATE INDEX IF NOT EXISTS ix_das2_delivery_incident ON das2_alert_delivery (incident_id, sent_at);

-- Operator feedback ----------------------------------------------------------
-- The only source of labelled ground truth. Phase 0 started collecting this on
-- the v1 alerts; the incident-level ack buttons feed the same well.
CREATE TABLE IF NOT EXISTS das2_feedback (
    feedback_id  VARCHAR(64) NOT NULL PRIMARY KEY,
    incident_id  VARCHAR(64),
    anomaly_id   VARCHAR(64),
    label        VARCHAR(16),    -- real | noise | unsure
    operator     VARCHAR(100),
    created_at   TIMESTAMP,
    note         VARCHAR(500)
);
CREATE INDEX IF NOT EXISTS ix_das2_feedback_label ON das2_feedback (label);

-- Useful checks once this is running:
--
--   -- alerts per incident; should be ~1, was up to 12 per fault under v1
--   SELECT incident_id, COUNT(*) FROM das2_alert_delivery GROUP BY incident_id;
--
--   -- coordinate coverage, since geo-clustering is useless without it
--   SELECT placement_source, COUNT(*) FROM das2_sensor GROUP BY placement_source;
--
--   -- classifier coverage, tracked rather than silently defaulted
--   SELECT equipment, COUNT(*) FROM das2_sensor GROUP BY equipment ORDER BY 2 DESC;
