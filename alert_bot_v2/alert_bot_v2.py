import os
import sys
import time
import json
import math
import logging
from pathlib import Path
from datetime import datetime, timezone
from threading import Lock, Thread
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

# ──────────────────────────────────────────────────────────────────────────────
# Operator feedback (inline "Real / Noise" buttons)
# ──────────────────────────────────────────────────────────────────────────────
# Why this exists: the detector has never had labelled ground truth. Every alert
# we send is an opportunity to collect one label at effectively zero operator
# cost — two taps. Those labels are the only way to ever measure precision, to
# tune suppression, or to justify a detector change to the client.
#
# Labels accumulate at the rate alerts are sent, so this is deliberately the
# FIRST change made: every week it isn't running is a week of unlabelled alerts
# that cannot be recovered retrospectively.
#
# Mechanism: each alert carries an inline keyboard whose callback_data encodes
# the abnormal_sensor_history row id. A background thread long-polls getUpdates,
# writes the tap to the DB, and edits the message to show what was recorded.
FEEDBACK_ENABLED      = os.getenv("FEEDBACK_ENABLED", "1").strip() == "1"
FEEDBACK_POLL_TIMEOUT = int(os.getenv("FEEDBACK_POLL_TIMEOUT", "25"))  # getUpdates long-poll seconds
FEEDBACK_OFFSET_FILE  = os.getenv("FEEDBACK_OFFSET_FILE", "logs/telegram_offset.txt")

# callback_data is capped at 64 bytes by Telegram, so keep the encoding terse:
#   "fb:<label>:<row_id>"
FEEDBACK_CHOICES = [
    ("real",  "✅ Real"),
    ("noise", "🔕 Noise"),
    ("unsure", "🤷 Unsure"),
]
_FEEDBACK_LABELS = {k: v for k, v in FEEDBACK_CHOICES}


# telegram endpoints
if not BOT_TOKEN or not CHAT_ID:
    logger.error("BOT_TOKEN or CHAT_ID missing in environment; exiting.")
    sys.exit(1)

TG_BASE        = f"https://api.telegram.org/bot{BOT_TOKEN}"
SEND_MSG_URL   = f"{TG_BASE}/sendMessage"
SEND_PHOTO_URL = f"{TG_BASE}/sendPhoto"
GET_UPDATES_URL       = f"{TG_BASE}/getUpdates"
ANSWER_CALLBACK_URL   = f"{TG_BASE}/answerCallbackQuery"
EDIT_REPLY_MARKUP_URL = f"{TG_BASE}/editMessageReplyMarkup"

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
    """
    Fetch rows that need alerts. Returns list of dicts.

    AlertSuppressed marks a detection that continues an event already
    alerted on in an earlier run. The detector runs over a 72h window on a
    6h cadence, so consecutive runs overlap by 66 hours and one ongoing
    fault is re-detected by ~12 runs; without this filter each of those
    sent its own Telegram message. ISNULL keeps rows written before the
    column existed alertable.
    """
    base_sql = """
        SELECT Id, Equipment, [Description], Plot_Path, SnapshotRunId
          FROM dbo.abnormal_sensor_history
         WHERE AlertTriggered IS NULL
           AND Plot_Path IS NOT NULL
           AND ISNULL(AlertSuppressed, 0) = 0
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


def record_feedback(row_id: int, label: str, operator: str) -> bool:
    """
    Persist an operator's Real/Noise/Unsure verdict for one alerted row.

    Returns True if a row was updated. Requires the feedback columns from
    migrations/000_feedback_columns.sql; if they're missing we log the remedy
    rather than crashing the poller, because a missing column must not take
    the alerting path down with it.
    """
    try:
        with engine.begin() as conn:
            res = conn.execute(
                text("""
                    UPDATE dbo.abnormal_sensor_history
                       SET FeedbackLabel = :label,
                           FeedbackBy    = :operator,
                           FeedbackAt    = SWITCHOFFSET(SYSDATETIMEOFFSET(), '+08:00')
                     WHERE Id = :id
                """),
                {"id": int(row_id), "label": label, "operator": operator},
            )
        return (res.rowcount or 0) > 0
    except Exception as e:
        logger.error(
            f"Failed to record feedback for Id={row_id}: {e}. "
            f"If this mentions an invalid column name, run "
            f"migrations/000_feedback_columns.sql against the database."
        )
        return False

# ──────────────────────────────────────────────────────────────────────────────
# Telegram senders
# ──────────────────────────────────────────────────────────────────────────────
def _extract_message_id(resp: requests.Response) -> int | None:
    try:
        data = resp.json()
        return data.get("result", {}).get("message_id")
    except Exception:
        return None


def feedback_keyboard(row_id: int) -> str | None:
    """
    JSON inline keyboard offering the Real/Noise/Unsure verdict for this row.
    Returns None when feedback is disabled, so callers can pass it through
    unconditionally.
    """
    if not FEEDBACK_ENABLED:
        return None
    return json.dumps({
        "inline_keyboard": [[
            {"text": text_label, "callback_data": f"fb:{key}:{int(row_id)}"}
            for key, text_label in FEEDBACK_CHOICES
        ]]
    })


def send_photo(photo_path: Path, caption: str,
               reply_markup: str | None = None) -> tuple[int, int | None]:
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
    if reply_markup:
        payload["reply_markup"] = reply_markup

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

def send_text(text_msg: str, reply_markup: str | None = None) -> tuple[int, int | None]:
    """Fallback: send a text message. Returns (http_status, message_id or None)."""
    rate_limiter.wait()
    payload = {"chat_id": CHAT_ID, "text": text_msg}
    if THREAD_ID:
        payload["message_thread_id"] = int(THREAD_ID)
    if reply_markup:
        payload["reply_markup"] = reply_markup
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
    if FEEDBACK_ENABLED:
        caption += "\n\nWas this a real problem?"

    keyboard = feedback_keyboard(row_id)

    p = Path(plot_path)
    if p.exists() and p.is_file():
        status, msg_id = send_photo(p, caption, reply_markup=keyboard)
    else:
        logger.warning(f"Image missing for Id={row_id}: {p}. Sending text fallback.")
        status, msg_id = send_text(caption + "\n\n(Plot not found)", reply_markup=keyboard)

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

# ──────────────────────────────────────────────────────────────────────────────
# Feedback poller
# ──────────────────────────────────────────────────────────────────────────────
def _load_offset() -> int:
    try:
        return int(Path(FEEDBACK_OFFSET_FILE).read_text().strip())
    except Exception:
        return 0


def _save_offset(offset: int) -> None:
    """
    Persist the getUpdates offset, atomically.

    The obvious `write_text` is not atomic: it truncates the file and then
    writes, so for a moment the offset file exists and is EMPTY. That window
    is small but it recurs, because this is called after every poll cycle --
    once a second against a responsive server, for as long as the bot runs.

    Two things fall into it. A reader that happens to look in that moment gets
    "", which is what made the feedback test fail about one run in eight with
    the correct value already on disk. Worse, a container killed in that
    window leaves an empty file behind, `_load_offset` reads 0, and the bot
    replays every acknowledgement Telegram still holds -- re-answering
    callbacks operators dealt with days ago.

    Writing to a sibling and renaming makes the replacement atomic on POSIX,
    so the file is only ever the old value or the new one.
    """
    try:
        p = Path(FEEDBACK_OFFSET_FILE)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(str(int(offset)))
        os.replace(tmp, p)
    except Exception as e:
        logger.warning(f"Could not persist getUpdates offset: {e}")


def _answer_callback(callback_id: str, note: str) -> None:
    """Acknowledge the tap so Telegram stops showing a spinner on the operator's client."""
    try:
        requests.post(
            ANSWER_CALLBACK_URL,
            data={"callback_query_id": callback_id, "text": note},
            timeout=10,
        )
    except Exception as e:
        logger.warning(f"answerCallbackQuery failed: {e}")


def _freeze_keyboard(message: dict, verdict_text: str) -> None:
    """
    Replace the keyboard with a single inert button showing what was recorded.
    Stops double-tapping and makes the thread self-documenting when scrolled back.
    """
    chat = (message.get("chat") or {}).get("id")
    msg_id = message.get("message_id")
    if chat is None or msg_id is None:
        return
    try:
        requests.post(
            EDIT_REPLY_MARKUP_URL,
            data={
                "chat_id": chat,
                "message_id": msg_id,
                "reply_markup": json.dumps(
                    {"inline_keyboard": [[{"text": verdict_text, "callback_data": "fb:done"}]]}
                ),
            },
            timeout=10,
        )
    except Exception as e:
        logger.warning(f"editMessageReplyMarkup failed: {e}")


def _handle_callback(cb: dict) -> None:
    data = str(cb.get("data") or "")
    cb_id = str(cb.get("id") or "")
    user = cb.get("from") or {}
    operator = (user.get("username")
                or " ".join(filter(None, [user.get("first_name"), user.get("last_name")]))
                or str(user.get("id") or "unknown"))[:100]

    parts = data.split(":")
    if len(parts) != 3 or parts[0] != "fb":
        # "fb:done" (an already-answered alert) and anything unrecognised land here.
        _answer_callback(cb_id, "Already recorded." if data == "fb:done" else "Unrecognised action.")
        return

    _, label, raw_id = parts
    if label not in _FEEDBACK_LABELS:
        _answer_callback(cb_id, "Unknown option.")
        return
    try:
        row_id = int(raw_id)
    except ValueError:
        _answer_callback(cb_id, "Bad alert reference.")
        return

    ok = record_feedback(row_id, label, operator)
    pretty = _FEEDBACK_LABELS[label]
    if ok:
        logger.info(f"📝 Feedback Id={row_id}: {label} (by {operator})")
        _answer_callback(cb_id, f"Recorded: {pretty}. Thank you.")
        _freeze_keyboard(cb.get("message") or {}, f"{pretty} — logged by {operator}")
    else:
        _answer_callback(cb_id, "Could not save — the alert row was not found.")


def feedback_poller() -> None:
    """
    Long-poll getUpdates for callback_query events. Runs in a daemon thread
    beside the alert loop.

    Note: getUpdates and webhooks are mutually exclusive. If a webhook is
    configured on this bot, Telegram answers 409 and we say so plainly rather
    than spinning silently.
    """
    offset = _load_offset()
    logger.info(f"👂 Feedback poller started (offset={offset}, timeout={FEEDBACK_POLL_TIMEOUT}s)")
    backoff = 1
    while True:
        try:
            resp = requests.get(
                GET_UPDATES_URL,
                params={
                    "offset": offset,
                    "timeout": FEEDBACK_POLL_TIMEOUT,
                    "allowed_updates": json.dumps(["callback_query"]),
                },
                timeout=FEEDBACK_POLL_TIMEOUT + 15,
            )
            if resp.status_code == 409:
                logger.error(
                    "getUpdates conflict (HTTP 409): a webhook is set on this bot, or another "
                    "instance is polling. Feedback buttons will not work until that is cleared "
                    "(call deleteWebhook, or set FEEDBACK_ENABLED=0). Retrying in 60s."
                )
                time.sleep(60)
                continue
            if resp.status_code != 200:
                logger.warning(f"getUpdates HTTP {resp.status_code}; backing off {backoff}s")
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue

            backoff = 1
            for upd in resp.json().get("result", []):
                offset = max(offset, int(upd.get("update_id", 0)) + 1)
                cb = upd.get("callback_query")
                if cb:
                    try:
                        _handle_callback(cb)
                    except Exception as e:
                        logger.error(f"Error handling callback: {e}", exc_info=True)
            _save_offset(offset)

        except requests.exceptions.ReadTimeout:
            continue  # normal for a long poll with no traffic
        except Exception as e:
            logger.error(f"Feedback poller error: {e}")
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)


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
    # - multi-topic routing via message_thread_id
    if FEEDBACK_ENABLED:
        Thread(target=feedback_poller, name="feedback-poller", daemon=True).start()
    else:
        logger.info("Feedback buttons disabled (FEEDBACK_ENABLED=0).")
    main_loop()