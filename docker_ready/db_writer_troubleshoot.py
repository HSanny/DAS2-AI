"""
CSV → Staging → MERGE into final tables (SQL Server)
---------------------------------------------------
For each expected CSV in ./processed, this script:

1) Reads the CSV into a pandas DataFrame
2) Writes it to a staging table: <name>_staging  (REPLACE each run)
3) MERGEs new rows from staging into the final table (INSERT only)
4) TRUNCATEs the staging table

Notes:
- Uses SQLAlchemy + pyodbc to connect to SQL Server.
- MERGE statements are defined per table and set to "INSERT when NOT MATCHED".
- Assumes the destination tables (final) already exist and have the columns shown.

Safety:
- Consider loading credentials from a .env rather than hardcoding.
- Check your ODBC driver name (e.g., "ODBC Driver 17 for SQL Server") is installed.
"""

import os
import sys
import time
import urllib.parse
import pandas as pd
import logging
import pytz

from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from dotenv import load_dotenv

# ---------- Setup logging ----------
LOG_FILE = "logs/db_writer.log"
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),  # log to file
        logging.StreamHandler(sys.stdout)                           # still print to console
    ]
)

logger = logging.getLogger(__name__)

# ---------- 1) Load environment  ----------
# If using a .env file, this will populate process env vars (os.getenv).
# You currently hardcode credentials below — keep load_dotenv() so
# you can switch to env-based creds easily later.
load_dotenv()

# Folder containing the CSVs: processed/alarmevent.csv, processed/data.csv, etc.
csv_dir = Path("processed")

# ---------- 2) Database credentials  ----------
def getenv_required(name: str) -> str:
    v = os.getenv(name)
    if not v or not v.strip():
        raise RuntimeError(f"Missing required env: {name}")
    return v.strip()

def getenv_int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v and v.strip() else default
    except ValueError:
        return default
    
# Option A (recommended): read from environment variables
host =      getenv_required("DAS_MSSQL_HOST")
database =  getenv_required("DAS_MSSQL_DATABASE")
port =      getenv_int("DAS_MSSQL_PORT", "14330")
username =  getenv_required("DAS_MSSQL_USERNAME")
password =  getenv_required("DAS_MSSQL_PASSWORD")

driver_name = os.getenv("ODBC_DRIVER", "ODBC Driver 17 for SQL Server")

# Option B (current): hard-coded test environment credentials
# # ⚠️ Avoid committing this to source control.
# host = "192.168.1.216"
# database = "anomaly_db"
# port = 14430              # Non-default port; omit 'port' if 1433
# username = "flotech"
# password = "P@ssword1234"

# URL-encode only if you manually build connection strings.
# (Not necessary with URL.create, but safe to keep around.)
# encoded_password = urllib.parse.quote_plus(password)

# ---------- 3) Create SQLAlchemy Engine  ----------
# URL.create helps build a proper mssql+pyodbc connection URL.
# Ensure the matching ODBC driver is installed on the machine.
conn_str = URL.create(
    "mssql+pyodbc",
    username=username,
    password=password,
    host=host,
    database=database,
    port=port,  # Omit for default 1433
    query={"driver": driver_name, "TrustServerCertificate": "yes"},
)

# Engine manages DB connections. fast_executemany boosts bulk insert speed on pyodbc.
engine = create_engine(conn_str, fast_executemany=True, pool_pre_ping=True)

# ---------- 4) Expected CSV base names (and corresponding DB tables) ----------
file_list = [
    'alarmevent',
    'data',
    'dateDim',
    'dim',
    'inst',
    'linktable',
    'longlat'
]

# ---------- 5) MERGE statements per table ----------
# Each MERGE inserts only when NOT MATCHED; no UPDATE logic here.
# Ensure columns match exactly (names + types) with your DB schema.
merge_queries = {
    "alarmevent": """
        MERGE INTO dbo.alarmevent AS target
        USING (
            SELECT DISTINCT HkeyDateTime, [Status]
            FROM dbo.alarmevent_staging
        ) AS source
        ON  target.HkeyDateTime = source.HkeyDateTime
        AND target.[Status]     = source.[Status]
        WHEN NOT MATCHED BY TARGET THEN
            INSERT (HkeyDateTime, [Status])
            VALUES (source.HkeyDateTime, source.[Status]);
    """,
    "data": """
        MERGE INTO dbo.data AS target
        USING (
            SELECT DISTINCT Equipment, [Description], [DateTime], CurrValue
            FROM dbo.data_staging
        ) AS source
        ON  target.Equipment     = source.Equipment
        AND target.[Description] = source.[Description]
        AND target.[DateTime]    = source.[DateTime]
        WHEN NOT MATCHED BY TARGET THEN
            INSERT (Equipment, [Description], [DateTime], CurrValue, Latitude, Longitude, [Location])
            VALUES (source.Equipment, source.[Description], source.[DateTime], source.CurrValue, source.Latitude, source.Longitude, source.[Location]);
    """,
    "dateDim": """
        MERGE INTO dbo.dateDim AS target
        USING dbo.dateDim_staging AS source
        ON target.[DateTime] = source.[DateTime]
        WHEN NOT MATCHED THEN
            INSERT ([DateTime], [Date], [Hour], [Minute])
            VALUES (source.[DateTime], source.[Date], source.[Hour], source.[Minute]);
    """,
    "dim": """
        MERGE INTO dbo.dim AS target
        USING dbo.dim_staging AS source
        ON target.IPAddress = source.IPAddress
           AND target.RowId = source.RowId
           AND target.Tagname = source.Tagname
        WHEN NOT MATCHED THEN
            INSERT (IPAddress, RowId, Description, Tagname, RTUNumber, RawType, Location, Hkey, Signaltype, Equipment)
            VALUES (source.IPAddress, source.RowId, source.Description, source.Tagname, source.RTUNumber,
                    source.RawType, source.Location, source.Hkey, source.Signaltype, source.Equipment);
    """,
    "inst": """
        MERGE INTO dbo.inst AS target
        USING (
            SELECT DISTINCT HkeyDateTime, CurrValue
            FROM dbo.inst_staging
        ) AS source
        ON target.HkeyDateTime = source.HkeyDateTime
        WHEN NOT MATCHED BY TARGET THEN
            INSERT (HkeyDateTime, CurrValue)
            VALUES (source.HkeyDateTime, source.CurrValue);
    """,
    "linktable": """
        MERGE INTO dbo.linktable AS target
        USING (
            SELECT DISTINCT HkeyDateTime, [DateTime], Hkey
            FROM dbo.linktable_staging
        ) AS source
        ON  target.HkeyDateTime = source.HkeyDateTime
        AND target.[DateTime]   = source.[DateTime]
        AND target.Hkey         = source.Hkey
        WHEN NOT MATCHED BY TARGET THEN
            INSERT (HkeyDateTime, [DateTime], Hkey)
            VALUES (source.HkeyDateTime, source.[DateTime], source.Hkey);
    """,
    "longlat": """
        MERGE INTO dbo.longlat AS target
        USING dbo.longlat_staging AS source
        ON target.Location = source.Location
           AND target.Longitude = source.Longitude
           AND target.Latitude = source.Latitude
        WHEN NOT MATCHED THEN
            INSERT ([Location], Longitude, Latitude)
            VALUES (source.[Location], source.Longitude, source.Latitude);
    """
}

# ---------- 6) Main loop: process each CSV ----------
for file in file_list:
    logger.info(f'checking for {file}.csv ...')

    # Build the full path to the CSV, e.g., processed/data.csv
    csv_file = csv_dir / f'{file}.csv'
    logger.info(str(csv_file))

    # Skip if file is missing
    if not csv_file.exists():
        logger.info(f"{file}.csv file not found.")
        continue

    # Skip empty files to avoid creating empty staging tables
    if csv_file.stat().st_size == 0:
        logger.info(f"[{file}] File is empty.")
        continue

    try:
        logger.info(f"Found {file}.csv file.")
    
        # Read CSV to DataFrame.
        # 🛈 If files can be large, consider: pd.read_csv(..., chunksize=100_000) and loop inserts.
        working_df = pd.read_csv(csv_file,
                                 low_memory=False,               # read whole file before deciding dtype
                                 keep_default_na=False,
                                 na_values=[]
                                 )

        # Name of the staging table that will temporarily hold CSV data
        staging_table = f"{file}_staging"

        logger.info(f"[{file}] Loading to staging table: {staging_table}")

        # Write DataFrame to staging table.
        # if_exists='replace' drops/recreates the table each run (schema inferred from DataFrame).
        # 🛈 If you need explicit SQL types, consider 'dtype={'col': sqlalchemy.types.<Type>}'.
        with engine.begin() as conn:
            working_df.to_sql(
                staging_table,
                conn,                # pass a Connection, not the Engine
                if_exists='replace',
                schema="dbo",
                index=False,
                chunksize=50_000       
                # , chunksize=50_000  # optional for very large CSVs
            )

        logger.info(f"[{file}] Executing MERGE...")

        # continue

        # Retrieve and execute the corresponding MERGE statement
        merge_sql = merge_queries[file]
        with engine.begin() as conn:  # Transactional scope
            conn.execute(text(merge_sql))

        logger.info(f"[{file}] Truncating staging table...")

        # Clear out staging table for the next run (keeps table, removes rows)
        with engine.begin() as conn:
            conn.execute(text(f"TRUNCATE TABLE [dbo].[{staging_table}]"))

        logger.info(f"Data updated to {file} successfully.")

    except Exception as e:
        # Any exception (CSV read, to_sql, MERGE, TRUNCATE) gets logged and we move on
        logger.error(f"Error reading {file}.csv: {e}")
        continue

# ---------- 7) Append abnormal_sensor.csv to a history table ----------
try:
    summary_csv = Path("output_csv/abnormal_sensor.csv")
    if not summary_csv.exists() or summary_csv.stat().st_size == 0:
        logger.info("[abnormal_sensor_history] CSV not found or empty; skipping upload.")
    else:
        df_summary = pd.read_csv(summary_csv)

        # Parse datetimes if present
        for col in ("First_Anomaly_Time", "Last_Anomaly_Time"):
            if col in df_summary.columns:
                df_summary[col] = pd.to_datetime(df_summary[col], errors="coerce")

        # Add snapshot metadata
        # Singapore timezone
        sgt = pytz.timezone("Asia/Singapore")

        snapshot_sgt = pd.Timestamp.now(tz=sgt).to_pydatetime()
        # Strip timezone → naive datetime (valid for DATETIME2)
        snapshot_sgt_naive = snapshot_sgt.replace(tzinfo=None)

        df_summary["SnapshotAtSGT"] = snapshot_sgt_naive
        df_summary["SnapshotRunId"] = snapshot_sgt.strftime("%Y%m%d_%H%M")

        # Optional: dedupe helper
        hash_cols = [c for c in [
            "Equipment","Description","First_Anomaly_Time","Last_Anomaly_Time",
            "Anomaly_Points","Num_Events","Peak_RZ","Max_DTW_Dist","Plot_Path"
        ] if c in df_summary.columns]
        if hash_cols:
            df_summary["RecordHash"] = (
                pd.util.hash_pandas_object(df_summary[hash_cols], index=False)
                .astype("int64")
            )

        target_table = "abnormal_sensor_history"  # schema defaults to dbo unless specified

        logger.info(f"[abnormal_sensor_history] Appending {len(df_summary)} rows to [{target_table}] ...")
        # Use an explicit SQLAlchemy connection so pandas doesn't try the DB-API path
        with engine.begin() as conn:
            df_summary.to_sql(
                target_table,
                conn,                      # <- pass a Connection, not the Engine
                if_exists="append",
                schema="dbo",
                index=False,
                chunksize=50_000       
                # , schema="dbo"           # uncomment if you need to force schema
            )
        logger.info(f"[abnormal_sensor_history] Append complete.")
except Exception as e:
    logger.error(f"[abnormal_sensor_history] Failed to append to DB: {e}")


# ---------- 8) Optional: small pause or final log ----------
# (Useful if this runs in a scheduler/loop)
# time.sleep(1)
