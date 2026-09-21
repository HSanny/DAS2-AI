-- 000_feedback_columns.sql
-- ---------------------------------------------------------------------------
-- Adds operator-feedback columns to dbo.abnormal_sensor_history.
--
-- Why: the detector has never had labelled ground truth, so its precision has
-- never been measured and no detector change can be justified with evidence.
-- Each alert already reaches a duty operator who knows whether it was real;
-- these columns capture that verdict from two taps on the Telegram alert.
--
-- Labels accumulate only as alerts are sent, so this should be applied as early
-- as possible -- unlabelled alerts cannot be labelled retrospectively.
--
-- Idempotent: safe to run repeatedly.
-- Target: SQL Server (matches the existing db_writer.py connection).
-- ---------------------------------------------------------------------------

IF NOT EXISTS (
    SELECT 1 FROM sys.columns
     WHERE object_id = OBJECT_ID(N'dbo.abnormal_sensor_history')
       AND name = N'FeedbackLabel'
)
BEGIN
    ALTER TABLE dbo.abnormal_sensor_history
        ADD FeedbackLabel NVARCHAR(16) NULL;   -- 'real' | 'noise' | 'unsure'
    PRINT 'Added column FeedbackLabel.';
END
ELSE
    PRINT 'Column FeedbackLabel already exists; skipping.';
GO

IF NOT EXISTS (
    SELECT 1 FROM sys.columns
     WHERE object_id = OBJECT_ID(N'dbo.abnormal_sensor_history')
       AND name = N'FeedbackBy'
)
BEGIN
    ALTER TABLE dbo.abnormal_sensor_history
        ADD FeedbackBy NVARCHAR(100) NULL;     -- Telegram username or display name
    PRINT 'Added column FeedbackBy.';
END
ELSE
    PRINT 'Column FeedbackBy already exists; skipping.';
GO

IF NOT EXISTS (
    SELECT 1 FROM sys.columns
     WHERE object_id = OBJECT_ID(N'dbo.abnormal_sensor_history')
       AND name = N'FeedbackAt'
)
BEGIN
    ALTER TABLE dbo.abnormal_sensor_history
        ADD FeedbackAt DATETIMEOFFSET NULL;    -- written as SGT (+08:00)
    PRINT 'Added column FeedbackAt.';
END
ELSE
    PRINT 'Column FeedbackAt already exists; skipping.';
GO

-- Supports "how many labels do we have so far, and of what kind?" without a
-- full scan of the history table as it grows.
IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
     WHERE object_id = OBJECT_ID(N'dbo.abnormal_sensor_history')
       AND name = N'IX_abnormal_sensor_history_FeedbackLabel'
)
BEGIN
    CREATE INDEX IX_abnormal_sensor_history_FeedbackLabel
        ON dbo.abnormal_sensor_history (FeedbackLabel)
        WHERE FeedbackLabel IS NOT NULL;
    PRINT 'Added index IX_abnormal_sensor_history_FeedbackLabel.';
END
ELSE
    PRINT 'Index IX_abnormal_sensor_history_FeedbackLabel already exists; skipping.';
GO

-- Label tally, for tracking progress towards a usable training/validation set.
-- Run periodically: at ~40 alerts/day, a few hundred labels accrue within weeks.
--
--   SELECT FeedbackLabel, COUNT(*) AS n
--     FROM dbo.abnormal_sensor_history
--    WHERE FeedbackLabel IS NOT NULL
--    GROUP BY FeedbackLabel;
