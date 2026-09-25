"""
das2.report.pdf
===============

One PDF per run: the whole picture, in one Telegram message.

Why this replaces a stream of messages
--------------------------------------
The system was sending a run header, two photos, up to ten incident messages
and a digest -- fourteen notifications an hour, every hour, for ever. The
client's verdict was blunt and correct: *"the alert messages are way too many"*.

Volume is not the only problem with that shape. Fourteen separate messages
cannot be read as one thing: the reader cannot see that six of the ten are the
same Bedok event, cannot tell 198 incidents from 12, and has no way back to
last hour's picture once the chat has scrolled. A document can carry all of
that, is one notification, and can be forwarded to whoever actually drives out.

What this deliberately gives up
-------------------------------
Per-incident acknowledge buttons. Telegram attaches an inline keyboard to a
MESSAGE, and one keyboard cannot acknowledge 198 incidents separately, so a
single document means no per-incident feedback -- which is the only source of
labels the system has for learning what a false alarm looks like. Setting
`alert.p1_detail_messages` brings back one button-carrying message per P1 (five
on the client's run, not a hundred and ninety-eight) as a middle ground.

Built on matplotlib's PdfPages
------------------------------
Not reportlab or weasyprint. Every chart in this report is matplotlib already,
the dependency is installed and proven in this container, and the alert path is
the last place to add a library that can fail to build. The cost is that layout
is done in figure coordinates by hand rather than with a flow engine, which is
why the page helpers below exist.
"""

from __future__ import annotations

import logging
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages   # noqa: E402

from das2.incident import parameters, signature        # noqa: E402
from das2.report import charts, narrative, theme       # noqa: E402

log = logging.getLogger("das2.report.pdf")

theme.install()

#: A4 landscape. Landscape because the map and every table in here are wider
#: than they are tall, and because the report is read on a phone held sideways
#: as often as it is printed.
PAGE_SIZE = (11.69, 8.27)

#: Margins in figure fractions.
L, R = 0.045, 0.955
TOP, BOTTOM = 0.93, 0.06

MAX_TABLE_ROWS = 18          # per page, at SIZE_BODY with comfortable leading

#: Characters that fit the text column at each size. Matplotlib clips silently
#: at the figure edge, so anything drawn as a single `fig.text` has to be
#: wrapped to a measured width or it loses its tail without an error.
FOOTER_WRAP = 185            # SIZE_TINY
CONTEXT_WRAP = 155           # SIZE_SMALL


# --------------------------------------------------------------------------- #
# Page furniture
# --------------------------------------------------------------------------- #
def _page(pdf: PdfPages, title: str, subtitle: str = "", *,
          footer: str = "") -> tuple[Any, float]:
    """
    Start a page and return (figure, y of the first free line).

    Every page carries the same header so a reader who lands in the middle of
    the document knows what they are looking at without scrolling back.
    """
    fig = plt.figure(figsize=PAGE_SIZE)
    fig.patch.set_facecolor(theme.PAGE)
    fig.text(L, 0.955, title, fontsize=theme.SIZE_TITLE,
             fontweight=theme.WEIGHT_BOLD, color=theme.INK, va="center")
    y = 0.915
    if subtitle:
        fig.text(L, y, subtitle, fontsize=theme.SIZE_SMALL,
                 color=theme.INK_MUTED, va="center", wrap=True)
        y -= 0.030
    if footer:
        # Wrapped, and stacked UPWARDS from the bottom margin. Drawn as one
        # `fig.text` it simply ran off the right edge of the paper: matplotlib
        # clips at the figure boundary without complaining, so a footnote that
        # grew by a sentence lost its last clause silently -- which on this
        # page is the sentence saying a reading never decided anything.
        lines = textwrap.wrap(footer, FOOTER_WRAP)
        for i, line in enumerate(reversed(lines)):
            fig.text(L, 0.025 + i * 0.016, line, fontsize=theme.SIZE_TINY,
                     color=theme.INK_MUTED, va="center")
    return fig, y


def _close(pdf: PdfPages, fig) -> None:
    pdf.savefig(fig, facecolor=theme.PAGE)
    plt.close(fig)


def _stat_tiles(fig, tiles: Sequence[tuple[str, str, str]], *,
                top: float, height: float = 0.17) -> None:
    """
    A row of headline numbers.

    A stat tile, not a chart: a single current value is not a bar chart with
    one bar. Each tile is (value, label, note) and the value is the largest
    thing on the page, because the whole point of a summary is that one number
    answers the question before anything else is read.
    """
    if not tiles:
        return
    gap = 0.012
    width = (R - L - gap * (len(tiles) - 1)) / len(tiles)
    for i, (value, label, note) in enumerate(tiles):
        x = L + i * (width + gap)
        fig.patches.append(plt.Rectangle(
            (x, top - height), width, height, transform=fig.transFigure,
            facecolor=theme.SURFACE, edgecolor=theme.HAIRLINE, linewidth=0.8,
            zorder=0))
        fig.text(x + 0.016, top - 0.052, value, fontsize=theme.SIZE_HERO * 0.62,
                 fontweight=theme.WEIGHT_BOLD, color=theme.INK, va="center")
        fig.text(x + 0.016, top - 0.098, label, fontsize=theme.SIZE_BODY,
                 color=theme.INK_SECONDARY, va="center")
        if note:
            fig.text(x + 0.016, top - 0.128, note, fontsize=theme.SIZE_TINY,
                     color=theme.INK_MUTED, va="center")


def _hbar(ax, labels: Sequence[str], values: Sequence[float], *,
          title: str, subtitle: str = "",
          colors: Sequence[str] | None = None,
          value_fmt: str = "{:,.0f}") -> None:
    """
    Horizontal bars, sorted by the caller, one colour unless told otherwise.

    One colour is the default on purpose: incident classes and anomaly types
    have no natural order, so shading them darker-where-bigger would encode the
    bar length twice and spend the only free channel on information the bar
    already carries. `colors` is passed only where the categories genuinely ARE
    a status scale -- priority -- and there the label travels with the colour.
    """
    y = range(len(labels))
    ax.barh(list(y), list(values), height=0.62,
            color=list(colors) if colors else theme.BAR, zorder=2)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=theme.SIZE_SMALL,
                       color=theme.INK_SECONDARY)
    ax.invert_yaxis()
    top = max(values) if len(values) and max(values) else 1
    ax.set_xlim(0, top * 1.18)
    ax.set_xticks([])
    for i, v in zip(y, values):
        ax.text(v + top * 0.015, i, value_fmt.format(v), va="center",
                fontsize=theme.SIZE_SMALL, color=theme.INK_SECONDARY)
    theme.apply(ax, title=title, subtitle=subtitle)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)


def _fit(text: str, width_fraction: float) -> str:
    """Trim a cell to the width it is drawn in, marking that it was trimmed."""
    budget = max(4, int(CONTEXT_WRAP * width_fraction))
    if len(text) <= budget:
        return text
    return text[:budget - 1].rstrip() + "…"


def _table(fig, rect: tuple[float, float, float, float],
           columns: Sequence[tuple[str, float, str]],
           rows: Sequence[Sequence[Any]], *,
           row_colors: Sequence[str] | None = None) -> None:
    """
    A plain table, drawn in figure coordinates.

    `columns` is (heading, relative width, alignment). Matplotlib's own
    `ax.table` is avoided: it sizes cells by content and cannot be told to keep
    a column at a fixed width, so a long site name silently squeezes the
    priority column to nothing -- which is the one column that must never move.
    """
    x0, y0, width, height = rect
    total = sum(c[1] for c in columns) or 1
    edges, acc = [], x0
    for _, w, _ in columns:
        edges.append(acc)
        acc += width * w / total

    line_h = min(0.036, height / max(len(rows) + 1.6, 1))
    y = y0 + height - line_h

    for (heading, _, align), ex, (_, w, _) in zip(columns, edges, columns):
        cell = ex + (width * w / total) / 2 if align == "center" else (
            ex + width * w / total - 0.010 if align == "right" else ex)
        fig.text(cell, y, heading.upper(), fontsize=theme.SIZE_TINY,
                 color=theme.INK_MUTED, fontweight=theme.WEIGHT_BOLD,
                 ha={"left": "left", "right": "right",
                     "center": "center"}[align], va="center")
    y -= line_h * 0.45
    fig.lines.append(plt.Line2D([x0, x0 + width], [y, y],
                                transform=fig.transFigure,
                                color=theme.AXIS, linewidth=0.9))
    y -= line_h * 0.75

    for r, row in enumerate(rows):
        for value, ex, (_, w, align) in zip(row, edges, columns):
            cell = ex + (width * w / total) / 2 if align == "center" else (
                ex + width * w / total - 0.010 if align == "right" else ex)
            color = theme.INK
            weight = "normal"
            if row_colors and r < len(row_colors) and align == "center":
                color, weight = row_colors[r], theme.WEIGHT_BOLD
            # Truncate to the column, visibly. Matplotlib draws the whole
            # string and lets the page edge cut it off without a word, so a
            # cell that outgrew its column lost its tail in silence -- which
            # on this report cost the last clause of a sentence about what the
            # numbers mean.
            value = _fit(str(value), width * w / total)
            fig.text(cell, y, str(value), fontsize=theme.SIZE_SMALL,
                     color=color, fontweight=weight,
                     ha={"left": "left", "right": "right",
                         "center": "center"}[align], va="center")
        y -= line_h * 0.72
        if r < len(rows) - 1:
            fig.lines.append(plt.Line2D(
                [x0, x0 + width], [y + line_h * 0.30, y + line_h * 0.30],
                transform=fig.transFigure, color=theme.GRID, linewidth=0.6))
        y -= line_h * 0.28


def _chunks(items: Sequence, size: int) -> Iterable[Sequence]:
    for start in range(0, max(len(items), 1), size):
        yield items[start:start + size]


def _region_of(incident) -> str:
    """
    The region's NAME.

    `str()` on the enum gives `Region.NORTH_EAST`, which is what the first
    version of this table printed in a column eight characters wide. The
    repr of an implementation detail is not a place.
    """
    region = incident.cluster.region
    if region is None:
        return "—"
    return str(getattr(region, "value", region))


def _type_label(name: str) -> str:
    """
    `'Flat Line  Q8'` -- the finding, and the standard test it is.

    The citation is the entire point of adopting QARTOD's names. An operator
    who wants to know what "Attenuated Signal" means can look up Test 10 in a
    published manual; one who reads `DITHERING_DEAD` has only us to ask. Types
    that are ours carry no suffix, so the page never implies a standard we do
    not have.
    """
    from das2.models import QARTOD_TEST, AnomalyType

    pretty = name.replace("_", " ").title()
    try:
        test = QARTOD_TEST.get(AnomalyType(name))
    except ValueError:
        return pretty
    return f"{pretty}  Q{test[0]}" if test else pretty


def _sites_of(incident) -> str:
    sites = sorted(incident.cluster.sites)
    if not sites:
        return str(incident.cluster.region or "unplaced")
    text = ", ".join(sites[:2])
    return text + (f" +{len(sites) - 2}" if len(sites) > 2 else "")


# --------------------------------------------------------------------------- #
# The pages
# --------------------------------------------------------------------------- #
def _cover(pdf: PdfPages, result, stamp: str) -> None:
    """
    The headline numbers, then the run explained in sentences.

    The prose is here rather than at the back because it is the part the
    client asked for -- "an explanation on the case or this round of
    analysis" -- and because on most runs it is the only part that gets read.
    Every other page in this document answers *what*; this one answers
    *so what*.
    """
    incidents = list(result.incidents)
    by_priority: dict[str, int] = {}
    for incident in incidents:
        p = incident.priority.value
        by_priority[p] = by_priority.get(p, 0) + 1
    urgent = by_priority.get("P1", 0) + by_priority.get("P2", 0)

    window = ""
    if result.window_start and result.window_end:
        window = (f"{result.window_start:%Y-%m-%d %H:%M} to "
                  f"{result.window_end:%Y-%m-%d %H:%M}")

    fig, y = _page(
        pdf, "Water network — this round of analysis",
        f"Run {result.run_id} · window {window} · "
        f"analysed in {result.duration_s:.0f}s · {stamp}",
        footer="Generated by DAS2-AI. Positions are site-level (per RTU); "
               "severity is 0-100 from member count, equipment diversity, "
               "spatial spread, duration and neighbour correlation.")

    lifecycle = result.stats.get("lifecycle", {})
    selection = result.stats.get("selection", {})
    _stat_tiles(fig, [
        (str(urgent), "need a decision now", "P1 and P2 incidents"),
        (str(len(incidents)), "open incidents", "all priorities"),
        (str(lifecycle.get("new", 0)), "new this run",
         f"{lifecycle.get('updated', 0)} carried over, "
         f"{lifecycle.get('resolved', 0)} closed"),
        (str(selection.get("held", 0)), "held back",
         "suppressed or below threshold — see inside"),
    ], top=y - 0.01)

    # The explanation. Wrapped by hand because matplotlib has no flow layout,
    # and continued onto a second page rather than truncated.
    #
    # Truncation was the first version's behaviour and it cut exactly the
    # wrong thing: the caveats paragraph sits last, so a page that overflowed
    # dropped "24 of the window's hourly files were missing" while keeping the
    # counts that number invalidates. A report that silently discards its own
    # limitations is worse than one that runs to two pages.
    y_text = y - 0.235
    fig.text(L, y_text, "What this run found",
             fontsize=theme.SIZE_HEADING, fontweight=theme.WEIGHT_BOLD,
             color=theme.INK, va="center")
    y_text -= 0.042

    lines: list[tuple[str, bool]] = []          # (text, is_paragraph_start)
    for para in narrative.paragraphs(result):
        for n, line in enumerate(textwrap.wrap(para, width=118)):
            lines.append((line, n == 0))

    line_h, para_gap = 0.0245, 0.012
    for text, starts in lines:
        if starts:
            y_text -= para_gap
        if y_text < BOTTOM + 0.03:
            _close(pdf, fig)
            fig, y = _page(pdf, "What this run found (continued)", "")
            y_text = y - 0.02
        fig.text(L, y_text, text, fontsize=theme.SIZE_BODY,
                 color=theme.INK_SECONDARY, va="center")
        y_text -= line_h
    _close(pdf, fig)


def _act_first(pdf: PdfPages, result) -> None:
    """The P1 and P2 list, on its own page, first after the explanation."""
    urgent = sorted([i for i in result.incidents
                     if i.priority.value in ("P1", "P2")],
                    key=lambda i: -i.severity)
    fig, y = _page(
        pdf, "Act first",
        f"{len(urgent)} incident(s) at P1 or P2, most severe first"
        if urgent else
        "Nothing at P1 or P2 this run.")
    if not urgent:
        fig.text(L, y - 0.06,
                 "The network is quiet at the priorities that mean dispatch. "
                 "The pages that follow record what was seen and what was "
                 "deliberately not sent.",
                 fontsize=theme.SIZE_BODY, color=theme.INK_SECONDARY,
                 va="center")
        _close(pdf, fig)
        return

    page = urgent[:MAX_TABLE_ROWS]
    _table(fig, (L, BOTTOM, R - L, y - BOTTOM - 0.02),
           [("", 0.04, "center"), ("class", 0.15, "left"),
            ("region", 0.09, "left"), ("sites", 0.20, "left"),
            ("what is moving", 0.26, "left"),
            ("sensors", 0.06, "right"), ("sev", 0.05, "right"),
            ("what to do", 0.19, "left")],
           [(i.priority.value, i.incident_class.value, _region_of(i),
             _sites_of(i), parameters.inline_summary(i),
             len(i.cluster.members), f"{i.severity:.0f}",
             _short_action(i)) for i in page],
           row_colors=[theme.PRIORITY_COLOR[i.priority.value] for i in page])
    if len(urgent) > MAX_TABLE_ROWS:
        fig.text(L, BOTTOM - 0.02,
                 f"{len(urgent) - MAX_TABLE_ROWS} more at P1/P2 — "
                 f"all of them are in 'What to act on'.",
                 fontsize=theme.SIZE_TINY, color=theme.INK_MUTED)
    _close(pdf, fig)


def _short_action(incident) -> str:
    """The recommendation, trimmed to something that fits a table cell."""
    text = incident.recommendation
    for long, short in (
            ("Multiple sites affected together - investigate the area, "
             "not one sensor.", "Investigate the area"),
            ("Instrument fault with no corroboration from neighbours - "
             "dispatch a technician.", "Dispatch a technician"),
            ("The machine, not the instrument - mechanical callout.",
             "Mechanical callout"),
    ):
        if text.startswith(long[:40]):
            return short
    text = text.split(" - ")[-1].rstrip(".")
    return text[:40] + ("…" if len(text) > 40 else "")


def _anatomy_pages(pdf: PdfPages, result, *, limit: int = 6) -> None:
    """
    What the most severe incidents are actually made of.

    The page the client asked for: "in a region, which kind of sensors or what
    parameter are triggering the event". The tables elsewhere answer it in a
    column; this answers it in full -- every parameter in the incident, how
    many sensors, which way they moved, how far, and what QARTOD makes of it.

    Two incidents to a page, because the block has to stay readable on a phone
    and a page of eight would be a spreadsheet.
    """
    incidents = sorted(
        [i for i in result.incidents if i.priority.value in ("P1", "P2")],
        key=lambda i: -i.severity)[:limit]
    if not incidents:
        return

    for start in range(0, len(incidents), 2):
        chunk = incidents[start:start + 2]
        page_no = start // 2 + 1
        pages = (len(incidents) + 1) // 2
        fig, y = _page(
            pdf, "What each one is made of",
            "Every parameter in the incident, which way it moved and how far"
            + (f" · page {page_no} of {pages}" if pages > 1 else ""),
            footer="Direction comes from the signed deviation of each sensor's "
                   "dominant finding. A parameter whose sensors disagree is "
                   "reported as mixed rather than resolved by majority. A "
                   "reading describes the incident; it never changed how it "
                   "was classified or what was recommended.")

        top = y - 0.03
        # Share the page rather than letting the first block take what it
        # likes. A twelve-parameter incident on top of a four-parameter one used
        # to run the second block off the bottom of the page, which is a silent
        # loss of exactly the detail this page exists to show. The last block
        # gets everything still unspent, so a short one above it donates its
        # slack instead of leaving a third of the paper blank.
        for n, incident in enumerate(chunk):
            top = _anatomy_block(fig, incident, top,
                                 budget=(top - BOTTOM) / (len(chunk) - n))
        _close(pdf, fig)


def _anatomy_block(fig, incident, top: float, *, budget: float = 0.42) -> float:
    """One incident's parameter table. Returns the y to continue from."""
    groups = parameters.breakdown(incident)
    started = top

    fig.text(L, top, f"{incident.priority.value}  {incident.incident_class.value}"
                     f"   ·   {_region_of(incident)}   ·   {_sites_of(incident)}",
             fontsize=theme.SIZE_HEADING, fontweight=theme.WEIGHT_BOLD,
             color=theme.PRIORITY_COLOR[incident.priority.value], va="center")
    top -= 0.030

    when = ""
    if incident.cluster.start and incident.cluster.end:
        when = (f"{incident.cluster.start:%d %b %H:%M}"
                f" → {incident.cluster.end:%d %b %H:%M}   ·   ")
    context = [f"{len(incident.cluster.members)} sensors",
               *([f"{incident.cluster.episodes} separate bursts"]
                 if getattr(incident.cluster, "episodes", 1) > 1 else []),
               f"{len(incident.cluster.sites)} sites",
               f"severity {incident.severity:.0f}"]
    if incident.rainfall_mm is not None:
        context.append(f"rain {incident.rainfall_mm:.1f} mm")
    evidence = (incident.detail or {}).get("rain_evidence")
    if evidence:
        context.append(evidence)
    if incident.neighbour_correlation is not None:
        context.append(f"neighbours r={incident.neighbour_correlation:.2f}")
    wrapped = textwrap.wrap(when + "   ·   ".join(context), CONTEXT_WRAP)
    for i, line in enumerate(wrapped[:2]):
        fig.text(L, top, line + ("…" if i == 1 and len(wrapped) > 2 else ""),
                 fontsize=theme.SIZE_SMALL, color=theme.INK_MUTED, va="center")
        top -= 0.022
    top -= 0.012

    top = _signature_lines(fig, incident, top)
    top = _conventional_line(fig, incident, top)

    channels = _asset_channels(incident)
    if channels:
        # A machine's anatomy is its channels, not a parameter breakdown.
        # Grouped by equipment class this incident reads "Digital Status 1, no
        # direction, —", which names the least interesting of the three
        # channels involved and gives an operator nothing to act on. What they
        # need is the contradiction: 41 A became 12 A while 24 L/s became zero.
        height = min(0.30, max(0.06, budget - (started - top) - 0.055),
                     0.045 + 0.028 * len(channels))
        _table(fig, (L, top - height, R - L, height),
               [("channel", 0.22, "left"), ("normal", 0.16, "right"),
                ("observed", 0.16, "right"), ("what it says", 0.46, "left")],
               channels)
        return top - height - 0.055

    rows = [(g.display, g.count, len(g.sites), g.direction_text(),
             g.move_text() or "—",
             ", ".join(t.value.replace("_", " ").title()
                       for t in g.behaviours[:2]),
             g.flag.name)
            for g in groups]

    # What is left of this block's share of the page, once the heading, the
    # context line and any reading have taken theirs.
    room = max(0.06, budget - (started - top) - 0.055)
    height = min(0.30, room, 0.045 + 0.028 * len(rows))
    if 0.045 + 0.028 * len(rows) > height:
        keep = max(1, int((height - 0.045) / 0.028))
        hidden = len(rows) - keep
        rows = rows[:keep]
        height = 0.045 + 0.028 * len(rows)
        rows.append((f"+{hidden} more parameter(s)", "", "", "see the region "
                     "page for the full breakdown", "", "", ""))
    _table(fig, (L, top - height, R - L, height),
           [("parameter", 0.18, "left"), ("sensors", 0.07, "right"),
            ("sites", 0.06, "right"), ("direction", 0.20, "left"),
            ("typical move", 0.13, "right"),
            ("behaviour", 0.26, "left"), ("qartod", 0.10, "left")],
           rows)
    return top - height - 0.055


def _fmt(value: Any, unit: str) -> str:
    try:
        number = f"{float(value):,.4g}"
    except (TypeError, ValueError):
        return "—"
    return f"{number} {unit}".strip()


def _asset_channels(incident) -> list[tuple[str, str, str, str]]:
    """
    A failed machine's channels, and what each one is saying.

    Empty for every other class, which is how the caller decides whether to
    draw this table or the parameter breakdown.
    """
    from das2.models import ASSET_TYPES

    rows: list[tuple[str, str, str, str]] = []
    for member in incident.cluster.members:
        # `getattr`, because a member arriving without its signals must cost
        # this one table and not the whole report. The report is the delivery;
        # an AttributeError here loses the run to save a detail of it.
        for signal in getattr(member, "signals", ()) or ():
            if signal.type not in ASSET_TYPES or not signal.detail:
                continue
            d = signal.detail
            duty_unit = str(d.get("duty_unit") or "")
            if "observed_output" in d:
                rows.append((
                    f"{d.get('output_kind', 'output')}",
                    _fmt(d.get("normal_output"), str(d.get("output_unit") or "")),
                    _fmt(d.get("observed_output"), str(d.get("output_unit") or "")),
                    "gone while the machine was running"))
            if "observed_duty" in d:
                running = d.get("running_duty")
                observed = d.get("observed_duty")
                says = "unchanged"
                try:
                    if float(observed) < float(running):
                        says = "below its own running normal"
                    elif float(observed) > float(running):
                        says = "above its own running normal"
                except (TypeError, ValueError):
                    pass
                if signal.type.value == "ASSET_ENERGISED_WHEN_OFF":
                    says = "drawn while the control says off"
                elif signal.type.value == "ASSET_NOT_ENERGISED_WHEN_ON":
                    says = "at its OFF level while the control says running"
                rows.append((f"motor {d.get('duty_kind', 'duty')}",
                             _fmt(running, duty_unit),
                             _fmt(observed, duty_unit), says))
            rows.append(("run status",
                         "running" if signal.type.value != "ASSET_ENERGISED_WHEN_OFF"
                         else "off",
                         "unchanged", "reported correctly throughout"))
    return rows[:6]


def _conventional_line(fig, incident, top: float) -> float:
    """
    What the client's own median-and-sigma check makes of the same sensors.

    One line here, the full table in the interactive report. This is the
    sentence he asked to be able to check -- *"verify oh there really is
    something that is not yet discovered by already existing stats
    calculation"* -- so it is printed whichever way it comes out, including
    the runs where their check would have caught it and this one added
    nothing.
    """
    panel = (incident.detail or {}).get("conventional") or {}
    if not panel:
        return top

    fig.text(L, top, panel.get("headline", ""), fontsize=theme.SIZE_SMALL,
             fontweight=theme.WEIGHT_BOLD, color=theme.INK_SECONDARY,
             va="center")
    top -= 0.022
    because = f"Found here because {panel.get('found_because', '')}"
    for line in textwrap.wrap(because, FOOTER_WRAP)[:2]:
        fig.text(L, top, line, fontsize=theme.SIZE_TINY,
                 color=theme.INK_MUTED, va="center")
        top -= 0.017
    return top - 0.008


def _signature_lines(fig, incident, top: float) -> float:
    """
    What the moving parameters usually mean, if anything is known to mean it.

    Printed under the evidence and above the numbers, never in place of the
    recommendation: an operator reading "reads as stormwater response" must
    still see the class and the action that were computed WITHOUT it. The
    falsifier rides on the same block for the same reason -- a hedged sentence
    with nothing to check it against is a horoscope, and the line below it is
    how a PUB engineer corrects the rule.

    The signature's own `action` is deliberately NOT rendered. Several read
    like instructions -- STORMWATER_RESPONSE's is *"Log it. This is the trip
    not worth making"* -- and printing that beside a computed "investigate the
    area" lets an unvalidated rule countermand a decision in the reader's head,
    which is the exact failure the layer was built to avoid. It is carried in
    the detail and on the dashboard for whoever is reviewing the rules; it
    reaches an operator's alert when a rule reaches `confirmed`, and not
    before.
    """
    sig = signature.attached(incident)
    if not sig:
        return top

    headline = sig.get("headline") or ""
    caveat = sig.get("caveat") or ""
    fig.text(L, top, headline, fontsize=theme.SIZE_BODY,
             fontweight=theme.WEIGHT_BOLD, color=theme.INK_SECONDARY,
             va="center")
    if caveat:
        fig.text(R, top, caveat, fontsize=theme.SIZE_TINY,
                 color=theme.INK_MUTED, va="center", ha="right")
    top -= 0.026

    for line in textwrap.wrap(sig.get("reads_as") or "", 140)[:2]:
        fig.text(L, top, line, fontsize=theme.SIZE_SMALL,
                 color=theme.INK_SECONDARY, va="center")
        top -= 0.021

    changes = sig.get("would_change_it") or ""
    if changes:
        wrapped = textwrap.wrap(f"Would change this reading: {changes}", 150)
        fig.text(L, top, wrapped[0] + ("…" if len(wrapped) > 1 else ""),
                 fontsize=theme.SIZE_TINY, color=theme.INK_MUTED, va="center")
        top -= 0.024
    return top


def _map_page(pdf: PdfPages, result) -> None:
    fig, y = _page(
        pdf, "Where",
        "One marker per place, sized by how many incidents are there and "
        "coloured by the worst of them. Shading ranks the five regions by "
        "total severity.")
    # Size the axes to the map's own aspect rather than to the space left over.
    # `set_aspect("equal")` shrinks whichever dimension is too long and leaves
    # the slack as blank paper -- which on the first version was a fifth of the
    # page below the island, on the one page where the island is the point.
    lon0, lon1, lat0, lat1 = charts.ISLAND_VIEW
    data_aspect = (lon1 - lon0) / (lat1 - lat0)
    width = R - L
    height = (width * PAGE_SIZE[0] / data_aspect) / PAGE_SIZE[1]
    available = y - BOTTOM - 0.02
    if height > available:                       # too tall: fit the height
        height = available
        width = (height * PAGE_SIZE[1] * data_aspect) / PAGE_SIZE[0]
    left = L + (R - L - width) / 2
    ax = fig.add_axes([left, y - height - 0.02, width, height])
    charts.draw_map(ax, result.incidents, scale=1.5, label_limit=12,
                    view=charts.ISLAND_VIEW)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    _close(pdf, fig)


def _breakdown(pdf: PdfPages, result) -> None:
    fig, y = _page(
        pdf, "What, and where it is concentrated",
        "Priority is a status scale and carries its label everywhere. "
        "The other three are counts, so every bar is the same colour and "
        "length alone is the measure.")

    stats = result.stats
    by_priority: dict[str, int] = {}
    by_class: dict[str, int] = {}
    for incident in result.incidents:
        by_priority[incident.priority.value] = \
            by_priority.get(incident.priority.value, 0) + 1
        name = incident.incident_class.value
        by_class[name] = by_class.get(name, 0) + 1

    top = y - 0.04
    h = (top - BOTTOM - 0.10) / 2
    w = (R - L - 0.075) / 2

    ax = fig.add_axes([L + 0.055, top - h, w - 0.055, h - 0.055])
    priorities = [p for p in theme.PRIORITY_ORDER if by_priority.get(p)]
    _hbar(ax, [f"{p}  {theme.PRIORITY_MEANING[p]}" for p in priorities],
          [by_priority[p] for p in priorities],
          title="By priority",
          colors=[theme.PRIORITY_COLOR[p] for p in priorities])

    ax = fig.add_axes([L + w + 0.13, top - h, w - 0.13, h - 0.055])
    classes = sorted(by_class.items(), key=lambda kv: -kv[1])[:8]
    _hbar(ax, [k.replace("_", " ").title() for k, _ in classes],
          [v for _, v in classes],
          title="By what the evidence says it is",
          subtitle="the verdict that decides whether anyone drives out")

    detection = stats.get("detection", {}).get("by_type", {})
    ax = fig.add_axes([L + 0.10, BOTTOM + 0.035, w - 0.10, h - 0.055])
    types = sorted(detection.items(), key=lambda kv: -kv[1])[:8]
    _hbar(ax, [_type_label(k) for k, _ in types],
          [v for _, v in types],
          title="Anomalies by type",
          subtitle=f"{stats.get('detection', {}).get('anomalies', 0):,} "
                   f"across {stats.get('detection', {}).get('sensors', 0):,} "
                   f"sensors · Q<n> cites the IOOS QARTOD test")

    ax = fig.add_axes([L + w + 0.13, BOTTOM + 0.035, w - 0.13, h - 0.055])
    _matrix(ax, result.region_matrix)
    _close(pdf, fig)


def _region_parameter_page(pdf: PdfPages, result) -> None:
    """
    Region by parameter, with direction. The whole-run view.

    The existing matrix says "the East has 31 level findings". This says "31,
    and 28 of them rising", which is the difference between a count and a
    direction of travel -- and direction is what makes a combination mean
    something.

    Diverging, because the quantity has a true centre: zero net movement. That
    is the one case where a diverging scale is right and a sequential one
    would lie, by rendering "20 up, 20 down" and "no findings at all" as the
    same colour. Blue for rising, red for falling, neutral grey between --
    warm against cool, so the poles read as opposite.
    """
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    matrix = parameters.region_parameter_matrix(result.anomalies)
    fig, y = _page(
        pdf, "By region and by parameter",
        "Net direction: sensors rising minus sensors falling. The number in "
        "each cell is the total findings, so a pale cell with a large number "
        "is a parameter pulling both ways at once.",
        footer="Findings with no direction — a stale or flatlined sensor has "
               "not moved either way — are counted in the total but not in "
               "the net.")

    regions = sorted(matrix)
    totals: dict[str, int] = {}
    for row in matrix.values():
        for parameter, cell in row.items():
            totals[parameter] = totals.get(parameter, 0) + sum(cell.values())
    params = [p for p, _ in sorted(totals.items(), key=lambda kv: -kv[1])][:12]

    if not regions or not params:
        fig.text(L, y - 0.08, "No placed findings this run.",
                 fontsize=theme.SIZE_BODY, color=theme.INK_SECONDARY)
        _close(pdf, fig)
        return

    net = np.array([[parameters.net_direction(
        matrix.get(r, {}).get(p, {})) for p in params] for r in regions],
        dtype=float)
    count = np.array([[sum(matrix.get(r, {}).get(p, {}).values())
                       for p in params] for r in regions], dtype=float)

    ax = fig.add_axes([L + 0.07, y - 0.60, R - L - 0.10, 0.52])
    limit = max(1.0, float(np.abs(net).max()))
    cmap = LinearSegmentedColormap.from_list(
        "das2div", ["#b2182b", "#e8a598", theme.PAGE, "#9ec5f4", "#184f95"])
    ax.imshow(net, cmap=cmap, aspect="auto",
              norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit))

    ax.set_xticks(range(len(params)))
    ax.set_xticklabels([parameters.display_name(p)[:16] for p in params],
                       rotation=35, ha="right", fontsize=theme.SIZE_SMALL)
    ax.set_yticks(range(len(regions)))
    ax.set_yticklabels(regions, fontsize=theme.SIZE_SMALL)

    for r in range(net.shape[0]):
        for c in range(net.shape[1]):
            if not count[r, c]:
                continue
            strong = abs(net[r, c]) > limit * 0.55
            ax.text(c, r, f"{int(count[r, c])}", ha="center", va="center",
                    fontsize=theme.SIZE_SMALL,
                    color="white" if strong else theme.INK_SECONDARY)
    theme.apply(ax)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)

    # A legend in words, because a diverging ramp with no anchor is a puzzle.
    for i, (label, colour) in enumerate([
            ("▲ net rising", "#184f95"),
            ("  balanced / no direction", theme.INK_MUTED),
            ("▼ net falling", "#b2182b")]):
        fig.text(L + 0.07 + i * 0.21, BOTTOM + 0.055, label,
                 fontsize=theme.SIZE_SMALL, color=colour,
                 fontweight=theme.WEIGHT_BOLD, va="center")
    _close(pdf, fig)


def _matrix(ax, matrix: dict[str, dict[str, int]], *, max_types: int = 8) -> None:
    """Region by equipment type, as a heatmap. Sequential, one hue."""
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    regions = sorted(matrix)
    totals: dict[str, int] = {}
    for row in matrix.values():
        for equip, n in row.items():
            totals[equip] = totals.get(equip, 0) + n
    types = [t for t, _ in sorted(totals.items(), key=lambda kv: -kv[1])][:max_types]

    if not regions or not types:
        ax.text(0.5, 0.5, "No anomalies to place", ha="center", va="center",
                transform=ax.transAxes, fontsize=theme.SIZE_BODY,
                color=theme.INK_MUTED)
        ax.set_xticks([])
        ax.set_yticks([])
        theme.apply(ax, title="By region and by type")
        return

    grid = np.array([[matrix.get(r, {}).get(t, 0) for t in types]
                     for r in regions], dtype=float)
    cmap = LinearSegmentedColormap.from_list("das2seq", theme.SEQUENTIAL)
    ax.imshow(grid, cmap=cmap, aspect="auto",
              vmin=0, vmax=max(grid.max(), 1))
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels([parameters.display_name(t)[:13] for t in types],
                       rotation=35, ha="right", fontsize=theme.SIZE_TINY)
    ax.set_yticks(range(len(regions)))
    ax.set_yticklabels(regions, fontsize=theme.SIZE_TINY)
    # Counts in every cell: a heatmap alone is indicative, and a reader
    # deciding where to send someone needs the number.
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c]:
                dark = grid[r, c] > grid.max() * 0.55
                ax.text(c, r, f"{int(grid[r, c])}", ha="center", va="center",
                        fontsize=theme.SIZE_TINY,
                        color="white" if dark else theme.INK_SECONDARY)
    theme.apply(ax, title="By region and by type",
                subtitle="the client's original question, as a grid")
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)


def _action_pages(pdf: PdfPages, result) -> None:
    incidents = sorted(result.alertable, key=lambda i: -i.severity)
    if not incidents:
        fig, y = _page(pdf, "What to act on",
                       "Nothing cleared the alerting threshold this run.")
        fig.text(L, y - 0.08,
                 "Every incident this run was either suppressed as telemetry "
                 "fan-out, held below the priority threshold, or judged too "
                 "weakly evidenced to act on. The next page says which.",
                 fontsize=theme.SIZE_BODY, color=theme.INK_SECONDARY)
        _close(pdf, fig)
        return

    pages = list(_chunks(incidents, MAX_TABLE_ROWS))
    for n, page in enumerate(pages, start=1):
        fig, y = _page(
            pdf, "What to act on",
            f"{len(incidents)} incident(s), most severe first"
            + (f" · page {n} of {len(pages)}" if len(pages) > 1 else ""),
            footer="Rain is integrated over each incident's own window, not "
                   "the run's. 'r' is the correlation with neighbouring "
                   "sensors: high means the water moved, low means the "
                   "instrument did.")
        _table(fig, (L, BOTTOM, R - L, y - BOTTOM - 0.02),
               [("", 0.04, "center"), ("incident", 0.09, "left"),
                ("class", 0.13, "left"), ("region", 0.09, "left"),
                ("sites", 0.17, "left"), ("what is moving", 0.22, "left"),
                ("sensors", 0.06, "right"),
                ("sev", 0.05, "right"), ("r", 0.05, "right"),
                ("rain", 0.06, "right"), ("what to do", 0.12, "left")],
               [(i.priority.value, i.incident_id[-10:],
                 i.incident_class.value, _region_of(i),
                 _sites_of(i), parameters.inline_summary(i, limit=2),
                 len(i.cluster.members), f"{i.severity:.0f}",
                 "—" if i.neighbour_correlation is None
                 else f"{i.neighbour_correlation:.2f}",
                 "—" if i.rainfall_mm is None else f"{i.rainfall_mm:.1f}",
                 _short_action(i)) for i in page],
               row_colors=[theme.PRIORITY_COLOR[i.priority.value] for i in page])
        _close(pdf, fig)


def _held_page(pdf: PdfPages, result) -> None:
    """
    What did NOT get sent, and why.

    The most important page for trust, and the one a stream of alert messages
    could never carry. A system that shows only what it decided to tell you is
    indistinguishable from one that missed everything else.
    """
    selection = result.stats.get("selection", {})
    reasons = selection.get("held_reasons", {})
    fig, y = _page(
        pdf, "What was held back, and why",
        f"{selection.get('held', 0)} incident(s) were deliberately not "
        f"alerted. Silence in this system is a decision with a reason, not "
        f"an absence of one.")

    if reasons:
        ax = fig.add_axes([L + 0.30, y - 0.40, R - L - 0.33, 0.36])
        items = sorted(reasons.items(), key=lambda kv: -kv[1])[:8]
        _hbar(ax, [k[:48] for k, _ in items], [v for _, v in items],
              title="")

    held = [i for i, _ in getattr(result, "held", [])]
    worst = sorted(held, key=lambda i: -i.severity)[:MAX_TABLE_ROWS - 6]
    if worst:
        reason_of = {id(i): r for i, r in result.held}
        _table(fig, (L, BOTTOM, R - L, y - 0.44),
               [("", 0.04, "center"), ("class", 0.16, "left"),
                ("region", 0.11, "left"), ("sites", 0.24, "left"),
                ("sev", 0.05, "right"), ("held because", 0.40, "left")],
               [(i.priority.value, i.incident_class.value,
                 _region_of(i), _sites_of(i),
                 f"{i.severity:.0f}", reason_of.get(id(i), ""))
                for i in worst],
               row_colors=[theme.PRIORITY_COLOR[i.priority.value] for i in worst])
    _close(pdf, fig)


def _quality_page(pdf: PdfPages, result) -> None:
    """
    What the run could and could not see.

    Every number on the earlier pages is conditional on this one. A run that
    read 48 of 72 hourly files will under-report, and a reader who does not
    know that will read the gap as good news.
    """
    stats = result.stats
    ingest = stats.get("ingest", {})
    coverage = stats.get("coverage", {})
    baselines = stats.get("baselines", {})

    fig, y = _page(
        pdf, "Data quality and coverage",
        "What this run could see. Every count in this report is bounded by "
        "these numbers.")

    rows: list[tuple[str, str, str]] = [
        ("Hourly files read", f"{ingest.get('history_files', 0)}",
         f"{ingest.get('history_files_skipped', 0)} unreadable and skipped"),
        ("Readings ingested", f"{ingest.get('history_rows', 0):,}",
         f"{ingest.get('history_rows_dropped', 0):,} dropped as unparseable"),
        ("Sensors in the inventory", f"{ingest.get('inventory_rows', 0):,}", ""),
        ("Coordinates resolved",
         f"{ingest.get('coordinate_coverage_pct', 0)}%",
         f"{ingest.get('sensors_without_coords', 0):,} sensors cannot be "
         f"placed on the map or clustered by location"),
        ("Analog sensors classified",
         f"{coverage.get('analog_coverage_pct', 0)}%",
         f"{coverage.get('unclassified', 0):,} remain UNCLASSIFIED and are "
         f"analysed but never alerted"),
        ("Time-of-day baselines", f"{baselines.get('usable', 0):,} sensor(s)",
         "the L2 residual layer abstains without these; they come from the "
         "daily profile job"),
    ]
    missing = ingest.get("missing_hours", 0)
    if missing:
        rows.append(("Missing hours in the window", f"{missing}",
                     f"{ingest.get('missing_hours_range', '')} — sensors "
                     f"silent across a gap read as STALE, so that count is "
                     f"inflated while this persists"))

    # The fleet-level answer to "would my own statistics have found this?".
    # It belongs beside the coverage numbers because it is the same kind of
    # claim: what this run could and could not have seen, and by what method.
    cvn = stats.get("conventional") or {}
    if cvn.get("sensors"):
        rows.append((
            "Your median ± 3σ, fleet-wide",
            f"{cvn.get('crossed', 0):,} of {cvn['sensors']:,}",
            f"cross the band; noise explains "
            f"{cvn.get('explained_by_noise', 0):,}, leaving "
            f"{cvn.get('would_alarm', 0):,} to act on. "
            f"{cvn.get('masked', 0):,} masked by their own event"))

    _table(fig, (L, y - 0.46, R - L, 0.44),
           [("measure", 0.26, "left"), ("value", 0.12, "left"),
            ("what it means for this report", 0.62, "left")],
           rows)

    rain = getattr(result, "rainfall_by_region", {}) or {}
    if rain:
        ax = fig.add_axes([L, BOTTOM + 0.02, (R - L) * 0.42, 0.24])
        items = sorted(rain.items(), key=lambda kv: -kv[1])
        _hbar(ax, [k for k, _ in items], [v for _, v in items],
              title="Rainfall by region, this window",
              subtitle="mm, from PUB's own gauges — no external forecast",
              value_fmt="{:,.1f}")
    _close(pdf, fig)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def write(result, out_path: str | Path, *, stamp: str = "") -> Path:
    """
    Render the whole run as one PDF and return its path.

    Page order follows what a duty operator does, not what the pipeline did:
    the decision first, then where, then the evidence, then everything that
    was deliberately left out.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_path) as pdf:
        _cover(pdf, result, stamp)
        _act_first(pdf, result)
        _anatomy_pages(pdf, result)
        _map_page(pdf, result)
        _breakdown(pdf, result)
        _region_parameter_page(pdf, result)
        _action_pages(pdf, result)
        _held_page(pdf, result)
        _quality_page(pdf, result)

        info = pdf.infodict()
        info["Title"] = f"DAS2-AI run {result.run_id}"
        info["Subject"] = "Water sensor anomaly summary"
        info["Creator"] = "DAS2-AI"
        info["CreationDate"] = datetime.now()

    log.info("report written: %s (%.0f KB)", out_path,
             out_path.stat().st_size / 1024)
    return out_path
