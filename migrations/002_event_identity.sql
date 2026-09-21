-- 002_event_identity.sql
-- ---------------------------------------------------------------------------
-- Adds cross-run event identity to dbo.abnormal_sensor_history.
--
-- The problem this solves
-- -----------------------
-- The detector runs over a 72-hour window on a 6-hour cadence, so consecutive
-- runs overlap by 66 of 72 hours. alert_bot sends every history row that has a
-- Plot_Path and no AlertTriggered, and nothing related a row to the row the
-- previous run wrote for the SAME ongoing condition. One sensor fault lasting
-- three days was therefore re-detected by every run that could still see it and
-- produced up to 12 separate Telegram alerts.
--
-- cluster_suppression does not address this: it de-duplicates across SENSORS
-- within one run (panel fan-out), never across RUNS.
--
-- With these columns, the first detection of an event mints an EventKey; later
-- detections of the same event inherit it and are written with
-- AlertSuppressed = 1, which alert_bot now filters on.
--
-- Also adds Rule_Invalid_Points, which reports how many of a sensor's flagged
-- points were physical range violations. Those used to be SUBTRACTED from the
-- results (a pressure sensor reading -5 bar was discarded rather than alerted);
-- they are now reported, and counted separately because a physical violation is
-- a certain fault rather than a statistical inference.
--
-- Idempotent: safe to run repeatedly.
-- Target: SQL Server.
-- ---------------------------------------------------------------------------

IF NOT EXISTS (
    SELECT 1 FROM sys.columns
     WHERE object_id = OBJECT_ID(N'dbo.abnormal_sensor_history') AND name = N'EventKey'
)
BEGIN
    ALTER TABLE dbo.abnormal_sensor_history
        ADD EventKey NVARCHAR(64) NULL;   -- shared by every detection of one event
    PRINT 'Added column EventKey.';
END
ELSE
    PRINT 'Column EventKey already exists; skipping.';
GO

IF NOT EXISTS (
    SELECT 1 FROM sys.columns
     WHERE object_id = OBJECT_ID(N'dbo.abnormal_sensor_history') AND name = N'DedupOfEventKey'
)
BEGIN
    ALTER TABLE dbo.abnormal_sensor_history
        ADD DedupOfEventKey NVARCHAR(64) NULL;   -- set when this row continues an earlier event
    PRINT 'Added column DedupOfEventKey.';
END
ELSE
    PRINT 'Column DedupOfEventKey already exists; skipping.';
GO

IF NOT EXISTS (
    SELECT 1 FROM sys.columns
     WHERE object_id = OBJECT_ID(N'dbo.abnormal_sensor_history') AND name = N'AlertSuppressed'
)
BEGIN
    ALTER TABLE dbo.abnormal_sensor_history
        ADD AlertSuppressed BIT NULL;   -- 1 = continuation, deliberately not alerted
    PRINT 'Added column AlertSuppressed.';
END
ELSE
    PRINT 'Column AlertSuppressed already exists; skipping.';
GO

IF NOT EXISTS (
    SELECT 1 FROM sys.columns
     WHERE object_id = OBJECT_ID(N'dbo.abnormal_sensor_history') AND name = N'Rule_Invalid_Points'
)
BEGIN
    ALTER TABLE dbo.abnormal_sensor_history
        ADD Rule_Invalid_Points INT NULL;   -- points outside the physical range
    PRINT 'Added column Rule_Invalid_Points.';
END
ELSE
    PRINT 'Column Rule_Invalid_Points already exists; skipping.';
GO

-- Supports the per-run lookup of prior detections for a sensor
-- (see event_dedup.prior_rows_query), which filters on Last_Anomaly_Time and
-- then groups by sensor.
IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
     WHERE object_id = OBJECT_ID(N'dbo.abnormal_sensor_history')
       AND name = N'IX_abnormal_sensor_history_EventLookup'
)
BEGIN
    CREATE INDEX IX_abnormal_sensor_history_EventLookup
        ON dbo.abnormal_sensor_history (Last_Anomaly_Time)
        INCLUDE (Equipment, [Description], First_Anomaly_Time, EventKey, AlertTriggered);
    PRINT 'Added index IX_abnormal_sensor_history_EventLookup.';
END
ELSE
    PRINT 'Index IX_abnormal_sensor_history_EventLookup already exists; skipping.';
GO

-- Verifying the fix after a few days of running:
--
--   -- alerts per distinct event; should be ~1, was up to 12
--   SELECT EventKey, COUNT(*) AS detections,
--          SUM(CASE WHEN AlertTriggered = 1 THEN 1 ELSE 0 END) AS alerts_sent
--     FROM dbo.abnormal_sensor_history
--    WHERE EventKey IS NOT NULL
--    GROUP BY EventKey
--    ORDER BY detections DESC;
