-- reset_das2.sql
-- ===========================================================================
-- Wipe DAS2's own data for a clean start, WITHOUT touching anything else.
--
-- THIS IS DESTRUCTIVE AND IS NOT RUN AUTOMATICALLY. `das2 migrate` never
-- deletes anything -- every statement in migrations/ is CREATE TABLE IF NOT
-- EXISTS or ALTER TABLE ADD -- which means re-running migrate against tables
-- that already hold rows leaves those rows exactly where they are. If you want
-- an empty database, you have to say so, and this file is how.
--
-- WHAT THIS TOUCHES
--   Only the twelve das2_* tables. Nothing else in the database is referenced.
--
-- WHAT THIS DELIBERATELY DOES NOT TOUCH
--   The v1 tables: dim, data, inst, linktable, dateDim, alarmevent,
--   abnormal_sensor_history.
--
--   You do not need to drop them, and there is one concrete reason to keep
--   `dbo.data` in particular:
--
--       The daily profile job reads das2_reading first and FALLS BACK to
--       dbo.data. das2_reading starts empty, so on a fresh install the
--       fallback is the only thing that gives you 28 days of history on day
--       one. Without it, DRIFT, NOISE_BURST and the time-of-day baselines
--       produce nothing for the first four weeks while das2_reading fills.
--
--   Keeping the v1 tables also lets both systems run side by side, which is
--   what the shadow comparison needs.
--
-- HOW TO RUN
--   Paste into SSMS against the DAS2 database, or:
--       docker compose run --rm das2 python -c \
--         "from das2.config import load_config; from das2.io.store import make_engine, apply_migrations; \
--          e = make_engine(load_config().database.sqlalchemy_url()); \
--          apply_migrations(e, files=['../tools/reset_das2.sql'])"
--
--   Then re-create the schema:
--       docker compose run --rm das2-migrate
--
-- Portable between SQL Server and SQLite: no GO batches, no IF EXISTS blocks.
-- Dropping a table that is not there is harmless here, because the migration
-- runner treats a missing-object error as already-done.
-- ===========================================================================

-- Child tables first, so nothing is orphaned mid-run on a database that has
-- had foreign keys added by hand.
DROP TABLE das2_feedback;
DROP TABLE das2_alert_delivery;
DROP TABLE das2_rain_observation;
DROP TABLE das2_neighbour_correlation;
DROP TABLE das2_incident_event;
DROP TABLE das2_incident_member;
DROP TABLE das2_incident;
DROP TABLE das2_sensor_anomaly;
DROP TABLE das2_detection_run;
DROP TABLE das2_sensor_profile;
DROP TABLE das2_reading;
DROP TABLE das2_sensor;
