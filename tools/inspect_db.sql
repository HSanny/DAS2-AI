-- inspect_db.sql
-- ===========================================================================
-- READ ONLY. Shows what is in the database and what it costs, so a decision to
-- drop anything is made from numbers rather than from a guess about which
-- table "compiles up a lot".
--
-- Run this in SSMS against anomaly_db (or whichever database you are using)
-- BEFORE dropping anything.
--
-- NOT TESTED against SQL Server from here -- there is no SQL Server in the
-- development environment. The syntax is standard T-SQL against sys.* views;
-- if a query errors, say so and it will be corrected rather than guessed at.
-- ===========================================================================

-- 1. Every table, biggest first. This answers "what is actually taking the
--    space" -- which is usually one or two reading tables, not the dozen
--    small ones people assume.
SELECT
    s.name                                                  AS [schema],
    t.name                                                  AS [table],
    SUM(p.rows)                                             AS [row_count],
    CAST(SUM(a.total_pages) * 8.0 / 1024 AS DECIMAL(12, 1)) AS [size_mb]
FROM sys.tables t
JOIN sys.schemas s          ON s.schema_id = t.schema_id
JOIN sys.indexes i          ON i.object_id = t.object_id
JOIN sys.partitions p       ON p.object_id = t.object_id AND p.index_id = i.index_id
JOIN sys.allocation_units a ON a.container_id = p.partition_id
WHERE i.index_id IN (0, 1)          -- heap or clustered index only, no double count
GROUP BY s.name, t.name
ORDER BY SUM(a.total_pages) DESC;


-- 2. Anything already called das2_*, and WHEN it was created.
--    If these predate today, they belong to the existing system, not to this
--    one -- and their names will collide with the twelve tables
--    migrations/010_das2_schema.sql creates. `CREATE TABLE IF NOT EXISTS`
--    silently skips a table that already exists, so a collision means the new
--    code would write into a table with the wrong columns and fail in a way
--    that is hard to read.
SELECT
    t.name          AS [table],
    t.create_date,
    t.modify_date,
    (SELECT COUNT(*) FROM sys.columns c WHERE c.object_id = t.object_id) AS [columns]
FROM sys.tables t
WHERE t.name LIKE 'das2[_]%'
ORDER BY t.create_date;


-- 3. The columns of each das2_* table, to compare against what this system
--    expects. If an existing das2_sensor has nothing like sensor_key /
--    description / equipment / latitude / longitude, it is a different table
--    that happens to share a name.
SELECT
    t.name  AS [table],
    c.name  AS [column],
    ty.name AS [type],
    c.max_length,
    c.is_nullable
FROM sys.tables t
JOIN sys.columns c   ON c.object_id = t.object_id
JOIN sys.types ty    ON ty.user_type_id = c.user_type_id
WHERE t.name LIKE 'das2[_]%'
ORDER BY t.name, c.column_id;


-- 4. The v1 reading table, if it exists. This is usually the one that grows
--    without bound -- months x ~2,672 sensors x 120 s is on the order of 10^8
--    rows -- and it is ALSO the table the daily profile job falls back to for
--    28 days of history on day one. Check its span before deciding.
SELECT
    MIN([DateTime]) AS [oldest],
    MAX([DateTime]) AS [newest],
    COUNT(*)        AS [rows]
FROM dbo.data;
