import os
import sys
import time
import json
import math
import logging
from pathlib import Path
from datetime import datetime, timezone
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────────────────────
# Logging (same feel as your analysis scripts)
# ──────────────────────────────────────────────────────────────────────────────
LOG_FILE = "logs/alerts.log"
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Env & Config
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()  # reads ./.env (or parent) if present

BOT_TOKEN   = os.getenv("DAS_BOT_TOKEN", "").strip()
CHAT_ID     = os.getenv("DAS_CHAT_ID", "").strip()              # numeric chat ID or @channelusername
THREAD_ID   = os.getenv("DAS_THREAD_ID", "").strip() or None    # optional (Telegram forum threads)
SLEEP_LOOP  = int(os.getenv("DAS_SLEEP_INTERVAL", "10"))
logger.info(f"BOT_TOKEN = {BOT_TOKEN} and CHAT_ID = {CHAT_ID}")

# concurrency & throttling
WORKERS               = int(os.getenv("ALERT_WORKERS", "6"))                # parallel senders
PER_CHAT_RATE_PER_SEC = float(os.getenv("PER_CHAT_RATE_PER_SEC", "2"))     # msgs/sec
MIN_INTERVAL_SEC      = 1.0 / max(PER_CHAT_RATE_PER_SEC, 0.1)


# telegram endpoints
if not BOT_TOKEN or not CHAT_ID:
    logger.error("BOT_TOKEN or CHAT_ID missing in environment; exiting.")
    sys.exit(1)

TG_BASE        = f"https://api.telegram.org/bot{BOT_TOKEN}"
SEND_MSG_URL   = f"{TG_BASE}/sendMessage"
SEND_PHOTO_URL = f"{TG_BASE}/sendPhoto"

# DB connection (SQL Server via ODBC Driver 17)
MSSQL_HOST = os.getenv("DAS_MSSQL_HOST", "192.168.1.216")
MSSQL_PORT = int(os.getenv("DAS_MSSQL_PORT", "1433"))
MSSQL_DB   = os.getenv("DAS_MSSQL_DATABASE", "anomaly_db")
MSSQL_USER = os.getenv("DAS_MSSQL_USERNAME", "")
MSSQL_PASS = os.getenv("DAS_MSSQL_PASSWORD", "")

# logger.info(f"MSSQL_HOST = {MSSQL_HOST}")
# logger.info(f"MSSQL_PORT = {MSSQL_PORT}")
# logger.info(f"MSSQL_DB = {MSSQL_DB}")
# logger.info(f"MSSQL_USER = {MSSQL_USER}")
# logger.info(f"MSSQL_PASS = {MSSQL_PASS}")

conn_url = URL.create(
    "mssql+pyodbc",
    username=MSSQL_USER,
    password=MSSQL_PASS,
    host=MSSQL_HOST,
    port=MSSQL_PORT,
    database=MSSQL_DB,
    query={"driver": "ODBC Driver 17 for SQL Server", "TrustServerCertificate": "yes"},
)
engine = create_engine(conn_url, fast_executemany=True, pool_pre_ping=True)

# ──────────────────────────────────────────────────────────────────────────────
# Simple rate limiter (per chat)
# ──────────────────────────────────────────────────────────────────────────────
class RateLimiter:
    def __init__(self, min_interval_sec: float):
        self.min_interval = float(min_interval_sec)
        self._lock = Lock()
        self._last = 0.0

    def wait(self):
        with self._lock:
            now = time.perf_counter()
            delta = now - self._last
            if delta < self.min_interval:
                time.sleep(self.min_interval - delta)
            self._last = time.perf_counter()

rate_limiter = RateLimiter(MIN_INTERVAL_SEC)

# ──────────────────────────────────────────────────────────────────────────────
# DB helpers
# ──────────────────────────────────────────────────────────────────────────────
def fetch_unsent(limit=None):
    """Fetch rows that need alerts. Returns list of dicts."""
    base_sql = """
        SELECT Id, Equipment, [Description], Plot_Path, SnapshotRunId
          FROM dbo.abnormal_sensor_history
         WHERE AlertTriggered IS NULL
           AND Plot_Path IS NOT NULL
        ORDER BY Id ASC
    """
    sql = base_sql if not limit else f"{base_sql} OFFSET 0 ROWS FETCH NEXT {int(limit)} ROWS ONLY"
    with engine.begin() as conn:
        rows = conn.execute(text(sql)).mappings().all()
    return [dict(r) for r in rows]

def mark_success(row_id: int, message_id: int | None, http_code: int):
    """Mark success for one row, set SGT time, store Telegram IDs/codes."""
    with engine.begin() as conn:
        conn.execute(
            text("""
                UPDATE dbo.abnormal_sensor_history
                   SET AlertTriggered     = 1,
                       AlertTriggeredAt   = SWITCHOFFSET(SYSDATETIMEOFFSET(), '+08:00'), -- SGT
                       AlertMessageId     = :mid,
                       AlertResponseCode  = :code
                 WHERE Id = :id
            """),
            {"id": int(row_id), "mid": message_id, "code": int(http_code)},
        )

def mark_failure_code(row_id: int, http_code: int):
    """Store the HTTP code even on failure (helps diagnose)."""
    with engine.begin() as conn:
        conn.execute(
            text("""
                UPDATE dbo.abnormal_sensor_history
                   SET AlertResponseCode = :code
                 WHERE Id = :id
            """),
            {"id": int(row_id), "code": int(http_code)},
        )

# ──────────────────────────────────────────────────────────────────────────────
# Telegram senders
# ──────────────────────────────────────────────────────────────────────────────
def _extract_message_id(resp: requests.Response) -> int | None:
    try:
        data = resp.json()
        return data.get("result", {}).get("message_id")
    except Exception:
        return None

def send_photo(photo_path: Path, caption: str) -> tuple[int, int | None]:
    """
    Send a single photo with rate limiting & 429 handling.
    Returns (http_status, message_id or None).
    """
    rate_limiter.wait()
    payload = {"chat_id": CHAT_ID}
    if THREAD_ID:
        payload["message_thread_id"] = int(THREAD_ID)
    if caption:
        payload["caption"] = caption

    files = {"photo": open(photo_path, "rb")}
    try:
        resp = requests.post(SEND_PHOTO_URL, data=payload, files=files, timeout=40)
    except Exception as e:
        logger.error(f"send_photo exception for {photo_path}: {e}")
        try:
            files["photo"].close()
        except Exception:
            pass
        return 0, None
    finally:
        try:
            files["photo"].close()
        except Exception:
            pass

    if resp.status_code == 429:
        # backoff then single retry
        retry_after = 1
        try:
            retry_after = int(resp.json().get("parameters", {}).get("retry_after", 1))
        except Exception:
            pass
        retry_after = max(retry_after, 1)
        logger.warning(f"429 from Telegram. Sleeping {retry_after}s then retrying once…")
        time.sleep(retry_after + 0.1)
        rate_limiter.wait()
        with open(photo_path, "rb") as fh:
            resp = requests.post(SEND_PHOTO_URL, data=payload, files={"photo": fh}, timeout=30)

    return resp.status_code, _extract_message_id(resp)

def send_text(text_msg: str) -> tuple[int, int | None]:
    """Fallback: send a text message. Returns (http_status, message_id or None)."""
    rate_limiter.wait()
    payload = {"chat_id": CHAT_ID, "text": text_msg}
    if THREAD_ID:
        payload["message_thread_id"] = int(THREAD_ID)
    try:
        resp = requests.post(SEND_MSG_URL, data=payload, timeout=20)
        return resp.status_code, _extract_message_id(resp)
    except Exception as e:
        logger.error(f"send_text exception: {e}")
        return 0, None

# ──────────────────────────────────────────────────────────────────────────────
# Worker for a single row
# ──────────────────────────────────────────────────────────────────────────────
def process_row(row: dict) -> tuple[int, int]:
    """
    Send the image for one anomaly row and mark it.
    Returns (row_id, http_status).
    """
    row_id = int(row["Id"])
    equipment = str(row.get("Equipment") or "")
    desc = str(row.get("Description") or "")
    plot_path = str(row.get("Plot_Path") or "")
    run_id = str(row.get("SnapshotRunId") or "")

    caption = f"🚨 Anomaly\n• Equipment: {equipment}\n• Desc: {desc}\n• Run: {run_id}"

    p = Path(plot_path)
    if p.exists() and p.is_file():
        status, msg_id = send_photo(p, caption)
    else:
        logger.warning(f"Image missing for Id={row_id}: {p}. Sending text fallback.")
        status, msg_id = send_text(caption + "\n\n(Plot not found)")

    if status == 200:
        try:
            mark_success(row_id, msg_id, status)
            logger.info(f"✅ Sent Id={row_id} (HTTP={status}, MsgID={msg_id})")
        except Exception as e:
            logger.error(f"Failed to mark success for Id={row_id}: {e}")
    else:
        try:
            mark_failure_code(row_id, status)
        except Exception as e:
            logger.error(f"Failed to record failure code for Id={row_id}: {e}")
        logger.warning(f"Telegram HTTP {status} for Id={row_id}")

    return row_id, status

# ──────────────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────────────
def alert_once():
    rows = fetch_unsent()  # all unsent
    if not rows:
        logger.info("No new anomalies to alert.")
        return

    total = len(rows)
    logger.info(f"Found {total} anomalies to send. Using {WORKERS} workers at ~{PER_CHAT_RATE_PER_SEC}/sec.")

    # Each worker marks its row immediately after success (SGT timestamp inside SQL).
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(process_row, r) for r in rows]
        done = 0
        ok = 0
        for f in as_completed(futures):
            row_id, status = f.result()
            done += 1
            ok += 1 if status == 200 else 0
            if done % 10 == 0 or done == total:
                logger.info(f"Progress: {done}/{total} sent (OK={ok}).")

def main_loop():
    logger.info("🚀 Alert bot started")
    while True:
        try:
            alert_once()
        except Exception as e:
            logger.error(f"Fatal loop error: {e}", exc_info=True)
        time.sleep(SLEEP_LOOP)

if __name__ == "__main__":
    # Keep “future features” in place (commented) for later:
    # - templated messages per anomaly type
    # - Telegram feedback polling / inline buttons
    # - multi-topic routing via message_thread_id
    #
    # from threading import Thread
    # Thread(target=poll_feedback, daemon=True).start()  # future
    main_loop()