-- 001_roc_columns.sql
-- ---------------------------------------------------------------------------
-- Adds the rate-of-change columns that replace the DTW columns in
-- dbo.abnormal_sensor_history.
--
-- Why the rename: the old sliding-DTW channel compared each window against
-- ITSELF shifted by one sample. Because DTW must match both endpoints and the
-- interior then aligns at zero cost, its distance was always exactly
--
--     |v[i-w] - v[i-w+1]| + |v[i-1] - v[i]|
--
-- (verified against an exact DTW implementation: correlation 1.000000, max
-- absolute difference 0.0). So "Max_DTW_Dist" never held a shape distance --
-- it held a first difference plus an echo of that difference w samples later.
-- Keeping the name would keep the false implication, and the column was also
-- used as a dedupe key in db_writer.py, so it had to be replaced rather than
-- reinterpreted.
--
-- This migration is ADDITIVE. The DTW columns are left in place and made
-- nullable so historical rows stay readable and comparable; new runs simply
-- stop populating them. Drop them only after the shadow comparison is over
-- and nobody is querying the old values.
--
-- Idempotent: safe to run repeatedly.
-- Target: SQL Server.
-- ---------------------------------------------------------------------------

IF NOT EXISTS (
    SELECT 1 FROM sys.columns
     WHERE object_id = OBJECT_ID(N'dbo.abnormal_sensor_history')
       AND name = N'ROC_Points'
)
BEGIN
    ALTER TABLE dbo.abnormal_sensor_history
        ADD ROC_Points INT NULL;   -- points flagged by the rate-of-change channel
    PRINT 'Added column ROC_Points.';
END
ELSE
    PRINT 'Column ROC_Points already exists; skipping.';
GO

IF NOT EXISTS (
    SELECT 1 FROM sys.columns
     WHERE object_id = OBJECT_ID(N'dbo.abnormal_sensor_history')
       AND name = N'Max_ROC_Rate'
)
BEGIN
    ALTER TABLE dbo.abnormal_sensor_history
        ADD Max_ROC_Rate FLOAT NULL;   -- peak |dv/dt|, engineering units per second
    PRINT 'Added column Max_ROC_Rate.';
END
ELSE
    PRINT 'Column Max_ROC_Rate already exists; skipping.';
GO

-- The retired DTW columns must be nullable, or appends from the new detector
-- (which no longer supplies them) would fail.
IF EXISTS (
    SELECT 1 FROM sys.columns
     WHERE object_id = OBJECT_ID(N'dbo.abnormal_sensor_history')
       AND name = N'DTW_Points' AND is_nullable = 0
)
BEGIN
    ALTER TABLE dbo.abnormal_sensor_history ALTER COLUMN DTW_Points INT NULL;
    PRINT 'Made DTW_Points nullable (retired column).';
END
GO

IF EXISTS (
    SELECT 1 FROM sys.columns
     WHERE object_id = OBJECT_ID(N'dbo.abnormal_sensor_history')
       AND name = N'Max_DTW_Dist' AND is_nullable = 0
)
BEGIN
    ALTER TABLE dbo.abnormal_sensor_history ALTER COLUMN Max_DTW_Dist FLOAT NULL;
    PRINT 'Made Max_DTW_Dist nullable (retired column).';
END
GO

-- After cutover, once no report or query still reads the old values:
--
--   ALTER TABLE dbo.abnormal_sensor_history DROP COLUMN DTW_Points;
--   ALTER TABLE dbo.abnormal_sensor_history DROP COLUMN Max_DTW_Dist;
