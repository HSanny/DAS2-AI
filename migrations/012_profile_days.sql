-- 012_profile_days.sql
-- ---------------------------------------------------------------------------
-- Record how much history each stored baseline rests on.
--
-- Without this the hourly run can load a baseline but cannot say whether it was
-- built from four weeks or from four days, and "usable: 28, median_days: 0" is
-- worse than no number at all -- it looks like a bug and hides the one fact
-- that decides how much to trust the L2 layer.
--
-- It also matters operationally: DRIFT needs 14 days and prefers 28, so an
-- engineer asking "why is nothing drifting?" should be able to answer it from
-- the table rather than from the logs.
--
-- Safe to re-run: applying twice raises "duplicate column", which the migration
-- runner treats as already-applied.
-- ---------------------------------------------------------------------------

ALTER TABLE das2_sensor_profile ADD days_observed INTEGER;
