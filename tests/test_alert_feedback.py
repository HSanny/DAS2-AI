"""
Functional test for the operator-feedback path in alert_bot_v2.

Runs a stub Telegram API on localhost and drives the real callback-handling
code against it, so the test covers the actual request shapes the bot sends --
not a reimplementation of them.

The bot module builds a SQL Server engine and reads Telegram credentials at
import time, so both are stubbed before import: pyodbc is faked (it isn't
installable here and is never called, since we never open a connection), and
record_feedback is swapped for an in-memory recorder.

Run:  python3 tests/test_alert_feedback.py
"""

import json
import os
import sys
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "alert_bot_v2"))


# --------------------------------------------------------------------------- #
# Stub Telegram API
# --------------------------------------------------------------------------- #
class StubTelegram(BaseHTTPRequestHandler):
    calls: list = []          # (method, params) observed, shared across requests
    updates_to_serve: list = []

    def log_message(self, *a):        # keep test output clean
        pass

    def _reply(self, payload, code=200):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _method(self):
        return urlparse(self.path).path.rsplit("/", 1)[-1]

    def do_GET(self):
        method = self._method()
        params = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
        StubTelegram.calls.append((method, params))
        if method == "getUpdates":
            served, StubTelegram.updates_to_serve = StubTelegram.updates_to_serve, []
            return self._reply({"ok": True, "result": served})
        self._reply({"ok": True, "result": {}})

    def do_POST(self):
        method = self._method()
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode(errors="replace")
        params = {k: v[0] for k, v in parse_qs(raw).items()}
        StubTelegram.calls.append((method, params))
        self._reply({"ok": True, "result": {"message_id": 999}})


def start_stub():
    # Threading, because the real Telegram API serves concurrent requests and
    # a single-threaded stub makes the test's timing depend on the order the
    # poller happens to interleave its long-poll GET with the POSTs that
    # answer each callback.
    srv = ThreadingHTTPServer(("127.0.0.1", 0), StubTelegram)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


# --------------------------------------------------------------------------- #
# Import the bot with its external dependencies stubbed
# --------------------------------------------------------------------------- #
def _fake_pyodbc():
    """
    Minimal pyodbc stand-in. SQLAlchemy's mssql+pyodbc dialect inspects a few
    DBAPI attributes when the engine is constructed; it never connects here, so
    these are only needed to get past create_engine().
    """
    mod = types.ModuleType("pyodbc")
    mod.paramstyle = "qmark"
    mod.apilevel = "2.0"
    mod.threadsafety = 1
    mod.version = "5.2.0"

    class Error(Exception):
        pass

    for name in ("Error", "Warning", "InterfaceError", "DatabaseError",
                 "DataError", "OperationalError", "IntegrityError",
                 "InternalError", "ProgrammingError", "NotSupportedError"):
        setattr(mod, name, type(name, (Error,), {}))
    class Cursor:
        def nextset(self):
            raise NotImplementedError

    mod.Cursor = Cursor
    mod.Connection = type("Connection", (), {})
    mod.connect = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("test must not open a real DB connection")
    )
    return mod


def load_bot(port):
    sys.modules.setdefault("pyodbc", _fake_pyodbc())
    os.environ.update(
        DAS_BOT_TOKEN="TEST-TOKEN",
        DAS_CHAT_ID="-100123",
        DAS_MSSQL_USERNAME="u",
        DAS_MSSQL_PASSWORD="p",
        FEEDBACK_ENABLED="1",
        FEEDBACK_OFFSET_FILE=str(REPO / "tests" / ".tmp_offset"),
    )
    import alert_bot_v2 as bot

    base = f"http://127.0.0.1:{port}"
    bot.GET_UPDATES_URL = f"{base}/getUpdates"
    bot.ANSWER_CALLBACK_URL = f"{base}/answerCallbackQuery"
    bot.EDIT_REPLY_MARKUP_URL = f"{base}/editMessageReplyMarkup"

    recorded = []
    bot.record_feedback = lambda row_id, label, operator: (
        recorded.append((row_id, label, operator)) or True
    )
    return bot, recorded


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def check(label, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    if not cond:
        raise SystemExit(1)


def callback(data, row_msg_id=42, user="siti"):
    return {
        "id": "cb-1",
        "data": data,
        "from": {"id": 7, "username": user},
        "message": {"message_id": row_msg_id, "chat": {"id": -100123}},
    }


def main():
    srv, port = start_stub()
    bot, recorded = load_bot(port)

    print("keyboard")
    kb = json.loads(bot.feedback_keyboard(12345))
    btns = kb["inline_keyboard"][0]
    check("offers three verdicts", len(btns) == 3)
    check("encodes the row id", all(b["callback_data"].endswith(":12345") for b in btns))
    check("all callback_data <= 64 bytes",
          all(len(b["callback_data"].encode()) <= 64 for b in btns))

    print("\nvalid tap")
    StubTelegram.calls.clear()
    bot._handle_callback(callback("fb:real:12345"))
    check("verdict persisted", recorded == [(12345, "real", "siti")])
    methods = [m for m, _ in StubTelegram.calls]
    check("callback answered", "answerCallbackQuery" in methods)
    check("keyboard frozen", "editMessageReplyMarkup" in methods)
    frozen = next(p for m, p in StubTelegram.calls if m == "editMessageReplyMarkup")
    frozen_kb = json.loads(frozen["reply_markup"])["inline_keyboard"][0]
    check("frozen to one inert button", len(frozen_kb) == 1)
    check("frozen button names the operator", "siti" in frozen_kb[0]["text"])

    print("\nmalformed / hostile input is rejected without recording")
    for bad in ("fb:done", "garbage", "fb:real", "fb:bogus:1", "fb:real:notanint",
                "fb:real:1:2", "", "'; DROP TABLE x--"):
        recorded.clear()
        StubTelegram.calls.clear()
        bot._handle_callback(callback(bad))
        check(f"{bad!r} not recorded", recorded == [])
        check(f"{bad!r} still answered",
              "answerCallbackQuery" in [m for m, _ in StubTelegram.calls])

    print("\npoller mechanics")
    StubTelegram.calls.clear()
    recorded.clear()
    StubTelegram.updates_to_serve = [
        {"update_id": 500, "callback_query": callback("fb:noise:777")},
        {"update_id": 501, "callback_query": callback("fb:unsure:888")},
    ]
    offset_file = Path(os.environ["FEEDBACK_OFFSET_FILE"])
    offset_file.unlink(missing_ok=True)
    t = threading.Thread(target=bot.feedback_poller, daemon=True)
    t.start()
    # Wait for the value being asserted, not for the file to merely EXIST.
    #
    # The poller writes the offset independently of recording the feedback, so
    # "the file is there and two updates arrived" can be true while the file
    # still holds an earlier offset. The test then read 501 and failed, about
    # one run in two -- an intermittent failure in the suite, which is worse
    # than a consistent one because it teaches everyone to re-run rather than
    # look. Nothing was ever wrong with the poller.
    def offset() -> str:
        try:
            return offset_file.read_text().strip()
        except OSError:
            return ""

    # Budget generously -- this exits the moment the condition holds, so the
    # only cost is in the failing case. The poller backs off on any transport
    # hiccup (1s, then 2s, then 4s), and against a 10s budget two of those
    # were enough to time out and report a defect that was not there.
    #
    # 30s was not enough either, and the arithmetic says why: one long-poll
    # cycle is 25s and the three backoffs are another 7s, so a slow start
    # alone can reach 32s with nothing wrong. That left roughly one full-suite
    # run in twelve failing here. An intermittent failure is worse than a
    # consistent one -- it teaches everyone to re-run instead of look -- so
    # the budget is now well clear of the worst legitimate case rather than
    # just above the typical one.
    deadline = time.time() + 90.0
    while time.time() < deadline:
        if len(recorded) >= 2 and offset() == "502":
            break
        threading.Event().wait(0.05)

    check("both updates handled", sorted(r[0] for r in recorded) == [777, 888])
    check("labels preserved", {r[1] for r in recorded} == {"noise", "unsure"})
    check("offset advanced past last update_id", offset() == "502")
    getu = next(p for m, p in StubTelegram.calls if m == "getUpdates")
    check("restricts to callback_query",
          json.loads(getu["allowed_updates"]) == ["callback_query"])
    check("long-polls", int(getu["timeout"]) > 0)

    offset_file.unlink(missing_ok=True)
    srv.shutdown()
    print("\nAll feedback tests passed.")


if __name__ == "__main__":
    main()
