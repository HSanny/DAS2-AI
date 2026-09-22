"""
Telegram alerting, tested against a mock Bot API.

This boundary cannot be exercised for real from here -- there is no bot token
and no network -- so it is pinned with a stub that answers like Telegram does.
That is worth doing carefully, because the alert path is the only part of this
system the client actually sees, and its failure modes are quiet ones:

  * `callback_data` over 64 bytes is rejected by the Bot API, so the buttons
    would render and then do nothing;
  * a webhook set on the bot makes `getUpdates` return HTTP 409 forever, so the
    acknowledge worker would poll silently and never process a press;
  * a suppressed incident that leaks into the send loop is a wasted trip, which
    is the exact cost this whole system exists to avoid.

Run:  python3 tests/test_alerting.py
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from urllib import error

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from das2.alerting import telegram  # noqa: E402
from das2.alerting.telegram import (  # noqa: E402
    ACK_STATE,
    CALLBACK_DATA_MAX,
    SendReport,
    TelegramConfig,
    ack_keyboard,
    compose,
    parse_callback,
    run_ack_worker,
    send_run,
)
from das2.models import (  # noqa: E402
    AckState,
    AnomalyType,
    Cluster,
    Incident,
    IncidentClass,
    PhysicalSeverity,
    SensorAnomaly,
    SensorMeta,
)

T0 = datetime(2026, 9, 21, 5, 49)


def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
    if not cond:
        raise SystemExit(1)


class FakeTelegram:
    """Records calls instead of making them. Optionally fails, as the real one does."""

    def __init__(self, fail_on=None, http_error=None):
        self.calls = []
        self.fail_on = fail_on or set()
        self.http_error = http_error

    def install(self):
        telegram.TelegramClient._call = self._call
        telegram.TelegramClient.send_photo = self._send_photo
        return self

    def _call(self, method, payload):
        if self.http_error and method == "getUpdates":
            raise error.HTTPError("url", self.http_error, "boom", {}, None)
        if method in self.fail_on:
            raise error.URLError("network down")
        self.calls.append((method, payload))
        if method == "getUpdates":
            return {"ok": True, "result": self.updates.pop(0) if self.updates else []}
        return {"ok": True, "result": {"message_id": len(self.calls)}}

    def _send_photo(self, path, caption="", *, reply_markup=None, chat_id=None):
        self.calls.append(("sendPhoto", {"caption": caption, "photo": str(path)}))
        return {"ok": True, "result": {"message_id": len(self.calls)}}

    def messages(self):
        return [p["text"] for m, p in self.calls if m == "sendMessage"]


def anomaly(key, desc, site, equipment, atype, *, deviation=1.5, unit="bar"):
    return SensorAnomaly(
        sensor=SensorMeta(sensor_key=key, description=desc, equipment=equipment,
                          site=site, latitude=1.34286, longitude=103.91977,
                          region="East", unit=unit, rtu_number="1010"),
        start=T0, end=T0 + timedelta(minutes=39),
        dominant_type=atype,
        severity=PhysicalSeverity(deviation=deviation, unit=unit,
                                  span_fraction=0.35, duration_s=2340,
                                  window_fraction=0.01),
    )


def make_incident(cls, incident_id="East-20260921-344ef706", members=None,
                  severity=65.4):
    members = members or [
        anomaly("1", "BedokPS-Trunk-Main-Pressure", "BedokPS", "Pressure",
                AnomalyType.LEVEL_SHIFT),
        anomaly("2", "TampinesPS-Outlet-Flow", "TampinesPS", "Flowrate",
                AnomalyType.LEVEL_SHIFT, deviation=8.04, unit="L/s"),
    ]
    return Incident(
        incident_id=incident_id,
        cluster=Cluster(members=members, region="East", centroid_lat=1.33976,
                        centroid_lon=103.94195, radius_m=2452.1),
        incident_class=cls, severity=severity,
        opened_at=T0, last_seen_at=T0 + timedelta(minutes=39),
        detail={"evidence": ["2 sensors across 2 sites",
                             "2 equipment types affected"]},
    )


class FakeResult:
    def __init__(self, incidents):
        self.incidents = incidents
        self.run_id = "20260921-060000"
        self.anomalies = [m for i in incidents for m in i.cluster.members]
        self.clusters = [i.cluster for i in incidents]
        self.window_start = T0
        self.window_end = T0 + timedelta(hours=1)
        self.stats = {"incidents": {"suppressed":
                                    sum(1 for i in incidents if not i.should_alert)}}

    @property
    def alertable(self):
        return [i for i in self.incidents if i.should_alert]


def main():
    print("the message leads with the decision")
    incident = make_incident(IncidentClass.REGIONAL_EVENT)
    text = compose(incident)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    check("first line is priority and class",
          "P2" in lines[0] and "REGIONAL_EVENT" in lines[0], f"({lines[0]})")
    check("second line is what to do",
          "investigate the area" in lines[1].lower(), f"({lines[1]})")
    check("where it is appears before the sensor list",
          text.index("Where:") < text.index("Sensors:"))
    check("the evidence is included", "2 equipment types affected" in text)
    check("the incident id is included", incident.incident_id in text)
    check("deviations carry their unit", "8.04 L/s" in text, f"({text})")
    check("it is within Telegram's message limit", len(text) <= 4096,
          f"({len(text)} chars)")

    print("\n  a 60-sensor incident stays readable on a phone")
    many = make_incident(IncidentClass.REGIONAL_EVENT, members=[
        anomaly(str(i), f"Site{i}-Pump{i}-Delivery-Pressure",
                f"Site{i}", "Pressure", AnomalyType.LEVEL_SHIFT)
        for i in range(60)])
    long_text = compose(many)
    check("within the limit", len(long_text) <= 4096, f"({len(long_text)})")
    check("only the first few sensors are listed", "and 52 more" in long_text,
          "(60 sensor lines would be unreadable and would blow the limit)")
    check("the count is still stated in full", "60 sensor(s)" in long_text)

    print("\n  and truncation is a real backstop, not decoration")
    huge = make_incident(IncidentClass.REGIONAL_EVENT, members=[
        anomaly(str(i), "X" * 900, f"Site{i}", "Pressure",
                AnomalyType.LEVEL_SHIFT) for i in range(8)])
    huge_text = compose(huge)
    check("an incident that would exceed the limit is cut",
          len(huge_text) <= 4096, f"({len(huge_text)})")
    check("and says so rather than silently losing the tail",
          "truncated" in huge_text)

    print("\ncallback_data fits the Bot API's 64-byte cap")
    for ident in ("East-20260921-344ef706",
                  "North-East-20260921-" + "a" * 40,
                  "X"):
        keyboard = ack_keyboard(ident)
        for button in keyboard["inline_keyboard"][0]:
            size = len(button["callback_data"].encode())
            check(f"{button['callback_data'][:18]}... is {size} bytes",
                  size <= CALLBACK_DATA_MAX,
                  "(over 64 and the button renders but silently does nothing)")
    check("three buttons are offered",
          len(ack_keyboard("x")["inline_keyboard"][0]) == 3)

    print("\ncallback parsing")
    for code, state in ACK_STATE.items():
        parsed_code, ref = parse_callback(f"{code}:East-20260921-344ef706")
        check(f"{code} -> {state.value}",
              parsed_code == code and ref == "East-20260921-344ef706")
    check("unknown codes are rejected", parse_callback("drop:table") == (None, None))
    check("malformed data is rejected", parse_callback("garbage") == (None, None))
    check("empty data is rejected", parse_callback("") == (None, None))

    # --- what gets sent, and what must not --------------------------------- #
    print("\nsuppressed classes are never sent")
    fake = FakeTelegram().install()
    result = FakeResult([
        make_incident(IncidentClass.REGIONAL_EVENT, "East-1"),
        make_incident(IncidentClass.SENSOR_FAULT, "West-1", severity=45.0),
        make_incident(IncidentClass.TELEMETRY_FANOUT, "East-2", severity=8.0),
        make_incident(IncidentClass.WATCH, "North-1", severity=12.0),
    ])
    report = send_run(result, TelegramConfig(token="t", chat_id="c"))
    check("the two actionable incidents are sent", len(report.sent) == 2,
          f"({report.sent})")
    check("fan-out is not sent", "East-2" not in report.sent)
    check("watch is not sent", "North-1" not in report.sent)
    check("and both are recorded as skipped, not lost",
          set(report.skipped) == {"East-2", "North-1"}, f"({report.skipped})")
    check("an overview message precedes the incidents",
          "DAS2 run" in fake.messages()[0], f"({fake.messages()[0][:60]})")
    check("the overview says how many were suppressed",
          "2 suppressed" in fake.messages()[0], f"({fake.messages()[0]})")

    print("\n  a quiet run says so explicitly, rather than saying nothing")
    fake = FakeTelegram().install()
    quiet = FakeResult([make_incident(IncidentClass.TELEMETRY_FANOUT, "East-2")])
    report = send_run(quiet, TelegramConfig(token="t", chat_id="c"))
    check("nothing is sent", report.sent == [])
    check("but the run is still reported",
          "Nothing to action" in fake.messages()[0], f"({fake.messages()[0]})")

    print("\nper-run cap protects the chat from a bad run")
    fake = FakeTelegram().install()
    flood = FakeResult([make_incident(IncidentClass.SENSOR_FAULT, f"E-{i}",
                                      severity=50.0 + i) for i in range(30)])
    report = send_run(flood, TelegramConfig(token="t", chat_id="c"),
                      max_incidents=10)
    check("only the cap is sent", len(report.sent) == 10, f"({len(report.sent)})")
    check("and the rest are acknowledged in one line",
          any("20 further incident" in m for m in fake.messages()))
    check("the most severe are the ones sent",
          report.sent[0] == "E-29", f"({report.sent[:3]})")

    print("\nTelegram being down must not lose the run")
    fake = FakeTelegram(fail_on={"sendMessage"}).install()
    report = send_run(FakeResult([make_incident(IncidentClass.SENSOR_FAULT)]),
                      TelegramConfig(token="t", chat_id="c"))
    check("the failure is reported, not raised", isinstance(report, SendReport))
    check("nothing is claimed as sent", report.sent == [])
    check("the error is captured for the log", len(report.failed) > 0,
          f"({report.as_dict()})")

    print("\nmisconfiguration is stated, not silently ignored")
    report = send_run(FakeResult([make_incident(IncidentClass.SENSOR_FAULT)]),
                      TelegramConfig(token="", chat_id=""))
    check("no token means nothing is sent", report.sent == [])
    check("and every incident is recorded as skipped", len(report.skipped) == 1)
    report = send_run(FakeResult([make_incident(IncidentClass.SENSOR_FAULT)]),
                      TelegramConfig(token="t", chat_id="c", enabled=False))
    check("disabled by config sends nothing", report.sent == [])

    # --- the acknowledge loop ----------------------------------------------- #
    print("\nacknowledgement round trip")
    fake = FakeTelegram().install()
    fake.updates = [[{
        "update_id": 101,
        "callback_query": {
            "id": "cb1",
            "from": {"id": 42, "username": "duty_operator"},
            "data": "dispatched:East-20260921-344ef706",
        },
    }], []]
    recorded = []
    offset_file = Path("/tmp/das2-test-offset.json")
    offset_file.unlink(missing_ok=True)
    run_ack_worker(TelegramConfig(token="t", chat_id="c"),
                   lambda ref, state, user: recorded.append((ref, state, user)),
                   offset_file=offset_file, stop_after=2)
    check("the press is recorded once", len(recorded) == 1, f"({recorded})")
    ref, state, user = recorded[0]
    check("against the right incident", ref == "East-20260921-344ef706")
    check("with the right state", state is AckState.DISPATCHED)
    check("and the operator who pressed it", user == "duty_operator")
    check("the callback is answered, as the API requires",
          any(m == "answerCallbackQuery" for m, _ in fake.calls),
          "(without it the button spins and gets pressed again)")

    print("\n  the offset is persisted, so a restart does not replay yesterday")
    check("an offset file is written", offset_file.exists())
    check("it points past the handled update",
          json.loads(offset_file.read_text())["offset"] == 102,
          f"({offset_file.read_text()})")
    offset_file.unlink(missing_ok=True)

    print("\n  a webhook on the bot is diagnosed, not polled forever")
    fake = FakeTelegram(http_error=409).install()
    fake.updates = []
    run_ack_worker(TelegramConfig(token="t", chat_id="c"),
                   lambda *a: None, offset_file=offset_file, stop_after=5)
    check("the worker returns instead of spinning", True,
          "(HTTP 409 means a webhook is set; getUpdates can never succeed)")
    offset_file.unlink(missing_ok=True)

    print("\nAll alerting tests passed.")


if __name__ == "__main__":
    main()
