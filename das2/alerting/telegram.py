"""
das2.alerting.telegram
======================

Incident-level Telegram alerting, with operator acknowledgement.

What changes from v1
--------------------
v1 alerted **per sensor per run**. With a 72-hour window re-analysed every six
hours, consecutive runs overlapped by 66 hours, and `fetch_unsent` sent every
new row -- so one three-day condition produced up to twelve photos, each
carrying `Equipment / Desc / Run` and nothing else. No severity, no location,
no recommendation, and nothing saying whether the twelve were one problem.

Here the unit of alerting is the **incident**: one message per event per place,
announced once, updated only when the evidence materially changes, and closed
when it clears. The message leads with the decision -- drive there, or do not
-- because that is what the client asked the system to answer.

Acknowledgement without a webhook
---------------------------------
Buttons use `callback_query` consumed by a `getUpdates` long-poll worker, not a
webhook. That matters for deployment: a webhook would need a public HTTPS
endpoint and a certificate on a network that has neither. Long-polling works
from behind any firewall with outbound access only.

`callback_data` is capped at **64 bytes** by the Bot API, so it carries an
action code and a short incident reference, never the incident narrative.

Failure is never fatal to a run
-------------------------------
Every send is wrapped: a Telegram outage must not lose an analysis run or crash
the scheduler. Failures are logged and reported in the run summary, and the
incident stays unsent so the next run retries it.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib import error, parse, request

from das2.incident import signature
from das2.models import AckState, Incident, IncidentClass

log = logging.getLogger("das2.alerting.telegram")

API = "https://api.telegram.org/bot{token}/{method}"

#: Telegram hard limits.
CALLBACK_DATA_MAX = 64
#: Seconds the socket is allowed BEYOND the long-poll duration, for the
#: round trip and for Telegram's own slack in honouring the timeout. It only
#: has to be comfortably positive; the call returns as soon as Telegram
#: answers, so a generous margin costs nothing.
LONG_POLL_MARGIN_S = 15

MESSAGE_MAX = 4096
CAPTION_MAX = 1024

#: Acknowledgement buttons. The middle one is the one with operational value:
#: "dispatched" closes the loop between an alert and a truck actually moving,
#: which is the number the client will eventually want to report on.
ACK_BUTTONS: tuple[tuple[str, str], ...] = (
    ("ack", "✅ Acknowledge"),
    ("dispatched", "\U0001f69a Dispatched"),
    ("false", "\U0001f515 False alarm"),
)

ACK_STATE = {
    "ack": AckState.ACKNOWLEDGED,
    "dispatched": AckState.DISPATCHED,
    "false": AckState.FALSE_ALARM,
}

PRIORITY_ICON = {"P1": "\U0001f534", "P2": "\U0001f7e0",
                 "P3": "\U0001f7e1", "P4": "⚪"}


@dataclass
class TelegramConfig:
    token: str = ""
    chat_id: str = ""
    enabled: bool = True
    #: Socket timeout for ordinary calls -- sending a message or a photo.
    #: getUpdates does NOT use this; see LONG_POLL_MARGIN_S.
    timeout_s: int = 20

    @property
    def configured(self) -> bool:
        return bool(self.token and self.chat_id)


@dataclass
class SendReport:
    sent: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"sent": len(self.sent), "failed": len(self.failed),
                "skipped": len(self.skipped),
                "errors": [e for _, e in self.failed][:5]}


# --------------------------------------------------------------------------- #
# Message composition
# --------------------------------------------------------------------------- #
def _fmt_duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    if minutes < 90:
        return f"{minutes} min"
    return f"{minutes / 60:.1f} h"


def compose(incident: Incident, *, dashboard_url: str | None = None) -> str:
    """
    The alert body.

    Ordered so the first two lines are enough to decide whether to act, because
    that is how an alert on a phone is actually read. The evidence follows for
    anyone who wants to check the reasoning, and the member list is last.
    """
    c = incident.cluster
    icon = PRIORITY_ICON.get(incident.priority.value, "")
    sites = ", ".join(sorted(c.sites)) or "unnamed site"

    lines = [
        f"{icon} <b>{incident.priority.value} · {incident.incident_class.value}</b>",
        f"<b>{incident.recommendation}</b>",
        "",
        f"<b>Where:</b> {c.region or 'unplaced'} — {sites}",
        f"<b>Scale:</b> {len(c.members)} sensor(s) at {len(c.sites)} site(s)"
        + (f", spread {c.radius_m/1000:.1f} km" if c.radius_m > 100 else ""),
    ]
    if c.start and c.end:
        lines.append(f"<b>When:</b> {c.start:%d %b %H:%M} → {c.end:%d %b %H:%M}"
                     f" ({_fmt_duration((c.end - c.start).total_seconds())})")
    if incident.rainfall_mm is not None:
        lines.append(f"<b>Rain nearby:</b> {incident.rainfall_mm:.1f} mm")
    if incident.neighbour_correlation is not None:
        lines.append(f"<b>Neighbours:</b> r={incident.neighbour_correlation:.2f}")

    # The reading goes BELOW the recommendation, never above it. The class and
    # the action on lines one and two were computed without any of this; a
    # sentence about what the parameters usually mean must not be the first
    # thing read, because "looks like stormwater" is exactly the phrase someone
    # would stop reading after.
    sig = signature.attached(incident)
    if sig:
        lines += ["", f"<b>{_esc(sig.get('headline', ''))}</b>"]
        if sig.get("reads_as"):
            lines.append(_esc(sig["reads_as"]))
        if sig.get("would_change_it"):
            lines.append(f"<i>Would change this reading: "
                         f"{_esc(sig['would_change_it'])}</i>")
        if sig.get("caveat"):
            lines.append(f"<i>({_esc(sig['caveat'])})</i>")

    evidence = incident.detail.get("evidence") or []
    if evidence:
        lines += ["", "<b>Why:</b>"] + [f"• {_esc(e)}" for e in evidence[:5]]

    lines += ["", "<b>Sensors:</b>"]
    for m in c.members[:8]:
        deviation = (f" — {m.severity.deviation:g} {m.severity.unit}"
                     if m.severity.deviation and m.severity.unit else "")
        lines.append(f"• {_esc(m.sensor.description or m.sensor.sensor_key)} "
                     f"[{m.dominant_type.value}]{deviation}")
    if len(c.members) > 8:
        lines.append(f"• …and {len(c.members) - 8} more")

    lines += ["", f"<code>{incident.incident_id}</code>"]
    if dashboard_url:
        lines.append(f'<a href="{_esc(dashboard_url)}">Open the full dashboard</a>')

    text = "\n".join(lines)
    return text if len(text) <= MESSAGE_MAX else text[:MESSAGE_MAX - 20] + "\n…(truncated)"


def _esc(s: Any) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def ack_keyboard(incident_id: str) -> dict[str, Any]:
    """
    Inline acknowledge buttons.

    `callback_data` is capped at 64 bytes by the Bot API, so the incident id is
    truncated to fit rather than being sent whole and silently rejected. The
    truncated form stays unique in practice because the id ends in a hash.
    """
    room = CALLBACK_DATA_MAX - len("dispatched") - 2
    short = incident_id[-room:] if len(incident_id) > room else incident_id
    return {"inline_keyboard": [[
        {"text": label, "callback_data": f"{code}:{short}"}
        for code, label in ACK_BUTTONS
    ]]}


# --------------------------------------------------------------------------- #
# Transport
# --------------------------------------------------------------------------- #
class TelegramClient:
    """Thin Bot API client over stdlib urllib -- no third-party dependency."""

    def __init__(self, config: TelegramConfig):
        self.config = config

    def _call(self, method: str, payload: dict[str, Any], *,
              timeout_s: int | None = None) -> dict[str, Any]:
        url = API.format(token=self.config.token, method=method)
        data = parse.urlencode(
            {k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
             for k, v in payload.items() if v is not None}).encode()
        req = request.Request(url, data=data,
                              headers={"Content-Type":
                                       "application/x-www-form-urlencoded"})
        with request.urlopen(req,
                             timeout=timeout_s or self.config.timeout_s) as resp:
            return json.loads(resp.read().decode())

    def send_message(self, text: str, *, reply_markup=None,
                     chat_id: str | None = None) -> dict[str, Any]:
        return self._call("sendMessage", {
            "chat_id": chat_id or self.config.chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
            "reply_markup": reply_markup,
        })

    def send_photo(self, photo_path: str | Path, caption: str = "", *,
                   reply_markup=None, chat_id: str | None = None) -> dict[str, Any]:
        return self._upload("sendPhoto", "photo", "image/png", photo_path,
                            caption=caption, reply_markup=reply_markup,
                            chat_id=chat_id)

    def send_document(self, doc_path: str | Path, caption: str = "", *,
                      reply_markup=None,
                      chat_id: str | None = None) -> dict[str, Any]:
        """
        Upload a file Telegram will show as an attachment rather than inline.

        This is how the run report arrives: one notification carrying the whole
        picture, which can be opened, kept and forwarded to whoever actually
        drives out -- none of which a stream of chat messages allows.
        """
        return self._upload("sendDocument", "document", "application/pdf",
                            doc_path, caption=caption,
                            reply_markup=reply_markup, chat_id=chat_id)

    def send_document_group(self, docs: list[tuple[str | Path, str]],
                            caption: str = "", *,
                            chat_id: str | None = None) -> dict[str, Any]:
        """
        Several attachments as ONE notification.

        The run delivers a PDF and an interactive HTML page, and sending them
        separately would put the notification count back up -- which is the
        complaint the single-report shape exists to answer. `sendMediaGroup`
        posts them as one album: one buzz, two files.

        Telegram refuses to mix a photo with documents in a group, so the map
        stays its own message. The caption goes on the FIRST item, which is
        where Telegram shows an album's caption.
        """
        files: list[tuple[str, str, str, bytes]] = []
        media: list[dict[str, Any]] = []
        for n, (path, mime) in enumerate(docs):
            path = Path(path)
            field = f"file{n}"
            files.append((field, path.name, mime, path.read_bytes()))
            item: dict[str, Any] = {"type": "document",
                                    "media": f"attach://{field}"}
            if n == 0 and caption:
                item["caption"] = caption[:CAPTION_MAX]
                item["parse_mode"] = "HTML"
            media.append(item)
        return self._post_multipart(
            "sendMediaGroup",
            {"chat_id": chat_id or self.config.chat_id,
             "media": json.dumps(media)},
            files)

    def _upload(self, method: str, field: str, mime: str,
                file_path: str | Path, *, caption: str = "",
                reply_markup=None, chat_id: str | None = None) -> dict[str, Any]:
        """One file, with a caption. The ordinary case."""
        path = Path(file_path)
        fields = {
            "chat_id": chat_id or self.config.chat_id,
            "caption": caption[:CAPTION_MAX],
            "parse_mode": "HTML",
        }
        if reply_markup:
            fields["reply_markup"] = json.dumps(reply_markup)
        return self._post_multipart(
            method, fields, [(field, path.name, mime, path.read_bytes())])

    def _post_multipart(self, method: str, fields: dict[str, Any],
                        files: list[tuple[str, str, str, bytes]]) -> dict[str, Any]:
        """
        multipart/form-data, built by hand.

        Deliberately not `requests`: the alert path has no dependency that
        could fail to install in the container, so a message can still go out
        when the rest of the stack is unhappy.
        """
        boundary = f"----das2{int(time.time()*1000)}"
        body = bytearray()
        for key, value in fields.items():
            body += (f"--{boundary}\r\n"
                     f'Content-Disposition: form-data; name="{key}"\r\n\r\n'
                     f"{value}\r\n").encode()
        for field, filename, mime, blob in files:
            body += (f"--{boundary}\r\n"
                     f'Content-Disposition: form-data; name="{field}"; '
                     f'filename="{filename}"\r\n'
                     f"Content-Type: {mime}\r\n\r\n").encode()
            body += blob + b"\r\n"
        body += f"--{boundary}--\r\n".encode()

        req = request.Request(
            API.format(token=self.config.token, method=method),
            data=bytes(body),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        # A PDF is far larger than a PNG and the upload is the slow part, so
        # the socket gets longer than an ordinary call would need.
        with request.urlopen(req, timeout=self.config.timeout_s * 3) as resp:
            return json.loads(resp.read().decode())

    def answer_callback(self, callback_id: str, text: str = "") -> dict[str, Any]:
        """
        Required by the Bot API within seconds of a button press.

        Without it the button spins on the operator's phone and they press it
        again, so this is not optional politeness.
        """
        return self._call("answerCallbackQuery",
                          {"callback_query_id": callback_id, "text": text[:200]})

    def get_updates(self, offset: int | None = None,
                    timeout: int = 25) -> list[dict[str, Any]]:
        # The socket must outlive the long poll, or every call dies waiting.
        #
        # getUpdates is a LONG POLL: Telegram is asked to hold the connection
        # open for `timeout` seconds and answer early only if an update
        # arrives. With the socket timeout at 20 s and the poll at 25 s, the
        # client hung up five seconds before Telegram was due to reply --
        # every call, on every network. The acknowledge worker could never
        # succeed, and it announced this as
        #
        #     WARNING getUpdates network error: The read operation timed out
        #
        # every 25 seconds, which reads like a flaky link rather than a
        # guaranteed failure. It went unrecognised for exactly that reason.
        result = self._call("getUpdates", {
            "offset": offset, "timeout": timeout,
            "allowed_updates": ["callback_query"],
        }, timeout_s=timeout + LONG_POLL_MARGIN_S)
        return result.get("result", [])


# --------------------------------------------------------------------------- #
# Sending a run
# --------------------------------------------------------------------------- #
def send_report(result, config: TelegramConfig, *, report_pdf: Path,
                map_png: Path | None = None,
                dashboard_html: Path | None = None,
                p1_detail_messages: bool = False,
                dashboard_url: str | None = None) -> SendReport:
    """
    The cluster map, then the analysis, then the page to dig into it.

    The shape the client asked for: *"1 cluster image on actual singapore map,
    and 1 pdf explanation on the case"*, and in that order for a reason -- the
    map renders inline in the chat, so a phone shows where the trouble is on
    the lock screen, while a PDF is an attachment that has to be opened. The
    glance first, the reasoning behind it second.

    `dashboard_html` is the third: *"can be interactive collective report show
    on the web ... he will click into the collective or interactive report to
    view, then verify"*. It rides along as a second document in the SAME
    media group as the PDF, so the run still costs two notifications rather
    than three -- the volume complaint that produced this shape in the first
    place applies to the fix for it as well.

    Telegram will not mix a photo and documents in one media group, which is
    why the map stays a separate message rather than all three arriving
    together.

    The map's caption carries enough to act on without opening anything: how
    many need a decision, where, and how many were deliberately held back. The
    document carries the reasoning, the tables and the caveats.

    The map is also page three of the document, so a PDF forwarded on its own
    still stands up. That duplication is deliberate: the two artefacts travel
    separately once someone shares one of them.

    `p1_detail_messages` adds one button-carrying message per P1 on top. It is
    off by default and exists because a document cannot carry an inline
    keyboard per incident, and those buttons are the only source of labels this
    system has for learning what a false alarm looks like. Five extra messages
    on the client's run, not a hundred and ninety-eight.
    """
    report = SendReport()
    if not config.enabled:
        log.info("telegram disabled by config")
        report.skipped = [i.incident_id for i in result.incidents]
        return report
    if not config.configured:
        log.warning("telegram token/chat_id missing -- nothing sent")
        report.skipped = [i.incident_id for i in result.incidents]
        return report

    client = TelegramClient(config)
    alertable = sorted(result.alertable, key=lambda i: -i.severity)
    detail = [i for i in alertable if i.priority.value == "P1"] \
        if p1_detail_messages else []
    report.skipped = [i.incident_id for i in result.incidents
                      if i not in alertable]

    caption = _report_caption(result, alertable)
    if map_png is not None and Path(map_png).exists():
        try:
            client.send_photo(map_png, caption)
            caption = ""          # said once; the document needs no repeat
        except (error.URLError, error.HTTPError, OSError) as exc:
            # Not fatal. The document carries the same map on page three, so a
            # failed photo costs the glance, not the delivery -- and falling
            # over here would lose the analysis to save the picture of it.
            log.error("telegram map upload failed (report continues): %s", exc)
            report.failed.append(("<map>", str(exc)))
            caption = _report_caption(result, alertable)

    body = caption or _document_caption(result)
    interactive = (Path(dashboard_html)
                   if dashboard_html and Path(dashboard_html).exists() else None)
    if interactive is not None:
        body = (body + "\n\n" + _interactive_note())[:CAPTION_MAX]
    try:
        if interactive is not None:
            try:
                client.send_document_group(
                    [(report_pdf, "application/pdf"),
                     (interactive, "text/html")], body)
            except (error.URLError, error.HTTPError, OSError) as exc:
                # An album is a nicety; the report is not. Two notifications
                # beat none, so a group that will not post falls back rather
                # than losing the run's only delivery to save a buzz.
                log.warning("media group failed, sending separately: %s", exc)
                client.send_document(report_pdf, body)
                client.send_document(interactive, "")
        else:
            client.send_document(report_pdf, body)
        # The document IS the delivery. Every alertable incident was in it, so
        # marking only the P1s as sent would misreport the run and would make
        # the next run re-announce everything it had already reported.
        report.sent = [i.incident_id for i in alertable]
    except (error.URLError, error.HTTPError, OSError) as exc:
        log.error("telegram report upload failed: %s", exc)
        report.failed.append(("<report>", str(exc)))
        return report

    for incident in detail:
        try:
            client.send_message(compose(incident, dashboard_url=dashboard_url),
                                reply_markup=ack_keyboard(incident.incident_id))
        except (error.URLError, error.HTTPError, OSError) as exc:
            log.error("telegram P1 detail failed for %s: %s",
                      incident.incident_id, exc)
            report.failed.append((incident.incident_id, str(exc)))
    return report


def _document_caption(result) -> str:
    """
    The one line under the attachment when the map already carried the numbers.

    Short on purpose: a caption repeated verbatim under two consecutive
    messages reads as a duplicate send, and a reader who has decided the first
    one was noise skips the second.
    """
    from das2.report import narrative

    return (f"<b>Run {result.run_id}</b> — full analysis\n"
            f"{_esc(narrative.summary_line(result))}")


def _interactive_note() -> str:
    """
    What the second attachment is for, in one line.

    Telegram will not render an HTML attachment inline, so without this the
    file reads as a duplicate of the PDF and never gets opened. It says what
    it does and what it costs to open.
    """
    return ("📊 The .html is the interactive version — open it in a browser to "
            "filter by region, see each sensor's trace, and check what your "
            "own median ± 3σ would have made of the same data.")


def _report_caption(result, alertable: list) -> str:
    """
    What the notification says before anyone opens the attachment.

    Written so a reader on a locked phone can decide whether to open it at all.
    The two numbers that matter are how many need a decision and how many were
    held back; the second is there because a count with no denominator invites
    the question "and what did it not tell me?".
    """
    counts: dict[str, int] = {}
    for incident in alertable:
        counts[incident.priority.value] = counts.get(incident.priority.value, 0) + 1
    urgent = counts.get("P1", 0) + counts.get("P2", 0)

    regions: dict[str, int] = {}
    for incident in alertable:
        if incident.priority.value in ("P1", "P2") and incident.cluster.region:
            name = str(getattr(incident.cluster.region, "value",
                               incident.cluster.region))
            regions[name] = regions.get(name, 0) + 1
    where = ", ".join(f"{r} {n}" for r, n in
                      sorted(regions.items(), key=lambda kv: -kv[1])[:4])

    held = result.stats.get("selection", {}).get("held", 0)
    lines = [f"<b>Run {result.run_id}</b>"]
    if urgent:
        lines.append(f"{PRIORITY_ICON['P1']} <b>{urgent}</b> need a decision "
                     f"now ({where})" if where else
                     f"{PRIORITY_ICON['P1']} <b>{urgent}</b> need a decision now")
    else:
        lines.append("No P1 or P2 this run.")

    # One reading, for the most severe incident that HAS one, named so nobody
    # reads it as a verdict on the whole run. A caption is read on a lock
    # screen, so it carries the headline and the hedge and nothing else; the
    # sentence behind it, and what would falsify it, are in the report.
    for incident in sorted(alertable, key=lambda i: -i.severity):
        reading = signature.attached(incident)
        if not reading:
            continue
        where = str(getattr(incident.cluster.region, "value",
                            incident.cluster.region) or "unplaced")
        lines.append(f"{_esc(where)} {incident.priority.value} — "
                     f"{_esc(reading.get('headline', ''))}"
                     + (f" ({_esc(reading['caveat'])})"
                        if reading.get("caveat") else ""))
        break
    lines.append(f"{len(result.incidents)} open incident(s) · "
                 f"{held} held back, with reasons inside")
    lines.append("")
    lines.append("Full report attached: where, what, what to act on, "
                 "and what was deliberately not sent.")
    return "\n".join(lines)[:CAPTION_MAX]


def send_run(result, config: TelegramConfig, *,
             charts: dict[str, Path] | None = None,
             dashboard_url: str | None = None,
             max_incidents: int = 10) -> SendReport:
    """
    Alert one run's incidents, one message each.

    The original mode, kept because the per-incident acknowledge buttons ride
    on these messages and they are the only label source the system has. See
    `send_report` for the document mode, which is now the default.

    Suppressed classes are never sent -- that is the entire point of computing
    them -- and the count of what was suppressed rides along in the summary, so
    silence is visibly *deliberate* rather than indistinguishable from a
    crashed job.
    """
    report = SendReport()
    if not config.enabled:
        log.info("telegram disabled by config")
        report.skipped = [i.incident_id for i in result.incidents]
        return report
    if not config.configured:
        log.warning("telegram token/chat_id missing -- nothing sent")
        report.skipped = [i.incident_id for i in result.incidents]
        return report

    client = TelegramClient(config)
    alertable = sorted(result.alertable, key=lambda i: -i.severity)
    report.skipped = [i.incident_id for i in result.incidents
                      if i not in alertable]

    # Overview first, so a reader has the shape of the run before the detail.
    try:
        client.send_message(_run_header(result, len(alertable)))
        if charts and charts.get("matrix"):
            client.send_photo(charts["matrix"],
                              "Anomalies by region and by type")
        if charts and charts.get("map") and alertable:
            client.send_photo(charts["map"], "Where the clusters are")
    except (error.URLError, error.HTTPError, OSError) as exc:
        log.error("telegram overview failed: %s", exc)
        report.failed.append(("<overview>", str(exc)))

    for incident in alertable[:max_incidents]:
        try:
            client.send_message(compose(incident, dashboard_url=dashboard_url),
                                reply_markup=ack_keyboard(incident.incident_id))
            report.sent.append(incident.incident_id)
        except (error.URLError, error.HTTPError, OSError) as exc:
            log.error("telegram send failed for %s: %s", incident.incident_id, exc)
            report.failed.append((incident.incident_id, str(exc)))

    # Everything past the cap, as a digest rather than a pointer.
    #
    # This used to read "See the dashboard for the full list", which is a
    # dangling reference: the dashboard is written to disk but nothing serves
    # it, and Telegram is the only channel anyone actually reads. An alert
    # that says "the rest is somewhere you cannot go" is worse than no line at
    # all, because it implies the information was delivered.
    #
    # Full messages are capped at ten so one bad run cannot flood the chat;
    # the digest is one line each, so the remainder stays visible without
    # becoming a hundred notifications.
    for chunk in _digest(alertable[max_incidents:]):
        try:
            client.send_message(chunk)
        except (error.URLError, error.HTTPError, OSError) as exc:
            log.error("telegram digest failed: %s", exc)
            break

    # What was withheld, and why. Silence has to be visibly deliberate: a run
    # that suppressed 151 incidents and one that crashed look identical from
    # the chat unless the suppression is stated.
    try:
        summary = _suppression_summary(result)
        if summary:
            client.send_message(summary)
    except (error.URLError, error.HTTPError, OSError):
        pass

    log.info("telegram: %s", report.as_dict())
    return report


def _digest(incidents: list) -> list[str]:
    """
    One line per incident, split into messages under the 4096-byte limit.

    Enough to recognise an incident and ask about it -- priority, class,
    region, sites, scale -- without the evidence and sensor list a full alert
    carries. Splitting on whole lines matters: Telegram rejects the message
    outright when it is too long, so a single over-long digest would deliver
    nothing rather than deliver less.
    """
    if not incidents:
        return []

    lines = [f"<b>{len(incidents)} further incident(s)</b>, one line each:"]
    for inc in incidents:
        sites = ", ".join(sorted(inc.cluster.sites)) or "unnamed site"
        if len(sites) > 60:
            sites = sites[:57] + "…"
        lines.append(
            f"{PRIORITY_ICON.get(inc.priority.value, '')} "
            f"{inc.priority.value} {_esc(inc.incident_class.value)} · "
            f"{_esc(inc.cluster.region or 'unplaced')} · "
            f"{len(inc.cluster.members)} sensor(s) · {_esc(sites)}")

    chunks, current = [], []
    size = 0
    for line in lines:
        # +1 for the newline that joins it.
        if size + len(line) + 1 > MESSAGE_MAX and current:
            chunks.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


def _suppression_summary(result) -> str:
    """
    What the run decided NOT to send, by class.

    Suppression is the product here, not a side effect -- v1 sent every
    sensor every run and the complaint was noise. But an unexplained silence
    is indistinguishable from a dead job, so the withheld count and its
    reasons are stated rather than assumed.
    """
    held = [i for i in result.incidents if i not in result.alertable]
    if not held:
        return ""
    by_class: dict[str, int] = {}
    for inc in held:
        key = inc.incident_class.value
        by_class[key] = by_class.get(key, 0) + 1
    rows = "\n".join(f"• {_esc(k)}: {v}"
                     for k, v in sorted(by_class.items(), key=lambda kv: -kv[1]))
    return (f"<b>{len(held)} incident(s) withheld</b> — deliberate, not a "
            f"quiet run:\n{rows}")


def _run_header(result, alertable: int) -> str:
    kpi = result.stats.get("incidents", {})
    suppressed = kpi.get("suppressed", 0)
    window = ""
    if result.window_start and result.window_end:
        window = (f"{result.window_start:%d %b %H:%M} → "
                  f"{result.window_end:%d %b %H:%M}")
    if alertable == 0:
        return (f"✅ <b>DAS2 run {result.run_id}</b>\n{window}\n\n"
                f"Nothing to action. "
                f"{len(result.anomalies)} sensor anomaly(ies) seen, "
                f"{suppressed} suppressed as telemetry noise or weak evidence.")
    return (f"<b>DAS2 run {result.run_id}</b>\n{window}\n\n"
            f"<b>{alertable}</b> incident(s) to action, "
            f"{suppressed} suppressed.\n"
            f"{len(result.anomalies)} sensor anomaly(ies) across "
            f"{len(result.clusters)} cluster(s).")


# --------------------------------------------------------------------------- #
# Acknowledgement worker
# --------------------------------------------------------------------------- #
def parse_callback(data: str) -> tuple[str | None, str | None]:
    """`ack:REGION-20260920-ab12cd34` -> ('ack', 'REGION-20260920-ab12cd34')."""
    if not data or ":" not in data:
        return None, None
    code, _, ref = data.partition(":")
    return (code, ref) if code in ACK_STATE else (None, None)


def run_ack_worker(config: TelegramConfig, on_ack, *,
                   offset_file: str | Path = "logs/telegram_offset.json",
                   poll_timeout: int = 25, stop_after: int | None = None) -> None:
    """
    Long-poll for button presses and apply them via `on_ack(ref, state, user)`.

    The offset is persisted so a restart does not replay every acknowledgement
    already handled -- Telegram retains updates for 24 hours, and replaying them
    would mark incidents acknowledged that nobody touched this shift.
    """
    offset_path = Path(offset_file)
    offset_path.parent.mkdir(parents=True, exist_ok=True)
    offset: int | None = None
    if offset_path.exists():
        try:
            offset = json.loads(offset_path.read_text()).get("offset")
        except (ValueError, OSError):
            offset = None

    client = TelegramClient(config)
    cycles = 0
    log.info("ack worker started (offset=%s)", offset)

    while stop_after is None or cycles < stop_after:
        cycles += 1
        try:
            updates = client.get_updates(offset=offset, timeout=poll_timeout)
        except error.HTTPError as exc:
            if exc.code == 409:
                # A webhook is set on this bot, so getUpdates is refused. Say so
                # explicitly: the symptom is otherwise a silent no-op forever.
                log.error("HTTP 409 -- a webhook is set on this bot. "
                          "Delete it (deleteWebhook) to use long-polling.")
                return
            log.error("getUpdates failed: %s", exc)
            time.sleep(5)
            continue
        except (error.URLError, OSError) as exc:
            log.warning("getUpdates network error: %s", exc)
            time.sleep(5)
            continue

        for update in updates:
            offset = update["update_id"] + 1
            query = update.get("callback_query")
            if not query:
                continue
            code, ref = parse_callback(query.get("data", ""))
            if not code:
                continue
            user = (query.get("from") or {}).get("username") \
                or str((query.get("from") or {}).get("id", "unknown"))
            try:
                on_ack(ref, ACK_STATE[code], user)
                client.answer_callback(query["id"],
                                       f"Recorded: {ACK_STATE[code].value}")
                log.info("ack %s -> %s by %s", ref, ACK_STATE[code].value, user)
            except Exception as exc:                   # noqa: BLE001
                log.exception("failed to record ack for %s", ref)
                try:
                    client.answer_callback(query["id"], "Could not record that")
                except (error.URLError, error.HTTPError, OSError):
                    pass

        if offset is not None:
            # Written atomically. `write_text` truncates and then writes, so
            # the file is momentarily EMPTY -- and this runs after every poll
            # cycle, so that window recurs for as long as the worker lives. A
            # container killed inside it leaves an empty file, the next start
            # reads no offset, and every acknowledgement Telegram still holds
            # is replayed. The same race made the feedback test fail about one
            # run in eight with the right value already on disk.
            try:
                import os

                tmp = offset_path.with_name(offset_path.name + ".tmp")
                tmp.write_text(json.dumps({"offset": offset}))
                os.replace(tmp, offset_path)
            except OSError:
                log.warning("could not persist telegram offset")
