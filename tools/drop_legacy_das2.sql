-- drop_legacy_das2.sql
-- ===========================================================================
-- Remove the PREVIOUS system's das2_* tables from anomaly_db, so the rewrite
-- can take the das2_* namespace cleanly.
--
-- DESTRUCTIVE AND IRREVERSIBLE. Run it only once, deliberately.
--
-- WHY THIS EXISTS
-- ---------------
-- The old system and this one both want the das2_* prefix, and one name
-- collides outright:
--
--     das2_feedback   old: 10 columns     new: 7 columns
--
-- `CREATE TABLE IF NOT EXISTS` silently skips a table that already exists, so
-- without this the new code would insert seven columns into the old
-- ten-column table and fail at runtime with an error pointing nowhere near the
-- cause. Two more pairs differ only by a plural -- das2_sensor/das2_sensors
-- and das2_reading/das2_readings -- which SQL distinguishes and people do not.
--
-- ORDER MATTERS
--     1. this script          (remove the old namespace)
--     2. das2-migrate         (create the new one)
--
-- Running migrate first would skip das2_feedback and leave the collision in
-- place.
--
-- OPTIONAL: KEEP A COPY OF THE LABELS FIRST
-- -----------------------------------------
-- das2_feedback holds operator labels -- the only ground truth either system
-- has, and the one input that cannot be regenerated from the raw data later.
-- Preserving it costs one statement and nothing else, so it is offered here
-- even though the decision to drop has been made. Uncomment to keep a copy
-- outside the das2_* namespace, where the new system will not touch it:
--
--     SELECT * INTO dbo.legacy_feedback_backup FROM dbo.das2_feedback;
--
-- The same one-liner works for any of the others.
-- ===========================================================================

DROP TABLE dbo.das2_ml_labels;
DROP TABLE dbo.das2_ml_shadow;
DROP TABLE dbo.das2_feedback;
DROP TABLE dbo.das2_kv_store;
DROP TABLE dbo.das2_sensor_filters;
DROP TABLE dbo.das2_anomaly_events;
DROP TABLE dbo.das2_baseline_profiles;
DROP TABLE dbo.das2_readings;
DROP TABLE dbo.das2_sensors;
DROP TABLE dbo.das2_alert_events;

-- Verify: this should return no rows before you run das2-migrate.
SELECT name, create_date FROM sys.tables WHERE name LIKE 'das2[_]%';
