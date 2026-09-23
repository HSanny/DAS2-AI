"""
das2.report.theme
=================

One palette, one set of type sizes, for every chart and every page of the PDF.

This exists because the report is read on a phone at 3 a.m. by someone deciding
whether to send a crew out, and a document whose colours mean different things
on different pages costs them time they do not have. Defining it once also
makes the colour decisions reviewable in one file rather than scattered across
six plotting functions.

The rules it follows, and why they are not negotiable here
---------------------------------------------------------
* **Priority is a STATUS scale, never a categorical one.** P1-P4 is an ordered
  judgement about urgency, so it gets the reserved status colours and those
  colours are used for nothing else. A status colour is never the only channel:
  every use in this report carries the label `P1`/`P2`/... beside it, because
  two of the four sit below 3:1 contrast on a light page by design, and because
  roughly 1 man in 12 cannot separate the red from the orange.
* **Magnitude is one hue, light to dark.** The region shading and the
  region-by-type matrix are sequential blue. Never a rainbow: a multi-hue ramp
  invents an ordering the reader has to learn, and gets it wrong under any form
  of colour blindness.
* **Nominal categories get ONE colour, not a ramp.** Incident classes and
  anomaly types have no natural order, so every bar is the same blue and length
  alone carries the value. Colouring them darker-where-bigger would double-encode
  the bar length and burn the only channel left for anything else.
* **Saturated ink is reserved for the small marks that matter** -- the incident
  markers -- and everything structural (grid, axes, coastline, labels) recedes.
  A page where everything shouts says nothing.

Values are the validated defaults from the visualisation reference: the status
set is measured against this page colour, and the categorical order clears the
colour-blind separation gates. Substituting PUB's own brand palette means
replacing the values in this file and nothing else.
"""

from __future__ import annotations

# --- surfaces and ink ------------------------------------------------------ #
PAGE = "#f9f9f7"          # the paper
SURFACE = "#fcfcfb"       # a chart's own panel, very slightly lighter
INK = "#0b0b0b"           # primary text
INK_SECONDARY = "#52514e"  # supporting text
INK_MUTED = "#898781"     # axis labels, captions, anything structural
GRID = "#e1e0d9"          # hairline gridlines
AXIS = "#c3c2b7"          # baselines and axis rules
HAIRLINE = "#dfe3e8"      # table rules, card borders

# --- status: priority, and nothing else ------------------------------------ #
#: P4 is deliberately NOT a status colour. It means "logged, no action", and
#: giving it a warning hue would put 149 of this run's 198 incidents in warning
#: ink -- which is how a report teaches its reader to ignore colour entirely.
PRIORITY_COLOR: dict[str, str] = {
    "P1": "#d03b3b",      # critical
    "P2": "#ec835a",      # serious
    "P3": "#fab219",      # warning
    "P4": "#898781",      # muted: present, not urgent
}
PRIORITY_ORDER = ("P1", "P2", "P3", "P4")

#: What each priority is for, in the words the report uses. The colour never
#: travels without one of these or the label itself.
PRIORITY_MEANING: dict[str, str] = {
    "P1": "act now",
    "P2": "act today",
    "P3": "schedule",
    "P4": "log only",
}

# --- sequential: magnitude -------------------------------------------------- #
#: One hue, light to dark. Used for the region shading on the map and for the
#: region-by-type matrix. Index 0 means "nothing here" and is allowed to recede
#: almost to the page.
SEQUENTIAL = ("#f2f5f7", "#cde2fb", "#9ec5f4", "#6da7ec",
              "#3987e5", "#256abf", "#184f95")

#: The map's region shading is the same hue held to the light end of the ramp.
#: It is a CONTEXT layer sitting under thirty-odd saturated priority markers,
#: and the full sequential range beneath them leaves a red P1 dot on a strong
#: blue field with nothing between the two. Five steps is enough to rank five
#: regions, which is all this layer has to do.
CHOROPLETH = ("#f4f6f8", "#e7eff9", "#d6e6f7", "#c0d9f3", "#a6c9ee")

#: The single colour every nominal bar is drawn in.
BAR = "#2a78d6"
#: For the one bar in a nominal chart worth pointing at, when there is one.
BAR_EMPHASIS = "#184f95"

# --- map-specific ----------------------------------------------------------- #
SEA = "#dce9f2"
NEIGHBOUR_LAND = "#dfe2e5"
NEIGHBOUR_EDGE = "#c8cdd2"
COAST_EDGE = "#5f6b76"

# --- type ------------------------------------------------------------------- #
#: DejaVu Sans is what matplotlib ships and what the container has. It carries
#: `normal` and `bold` only -- asking for weight 600 makes matplotlib warn on
#: every single text object and silently use 700 anyway, which is four warnings
#: a run in the log for no gain.
FONT_FAMILY = "DejaVu Sans"
WEIGHT_BOLD = "bold"

SIZE_HERO = 54
SIZE_TITLE = 20
SIZE_HEADING = 14
SIZE_BODY = 10
SIZE_SMALL = 8.5
SIZE_TINY = 7.5


def apply(ax, *, title: str = "", subtitle: str = "", grid_axis: str = "") -> None:
    """
    The chrome every chart in this report shares.

    `grid_axis` is "x", "y" or "" -- gridlines only along the axis the reader
    measures against. A grid on both axes of a bar chart is decoration.
    """
    if title:
        ax.set_title(title, fontsize=SIZE_HEADING, fontweight=WEIGHT_BOLD,
                     loc="left", color=INK, pad=22 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=SIZE_SMALL,
                color=INK_MUTED, va="bottom")
    # Transparent, not a panel. The stat tiles on the cover are the only
    # cards in this document; giving every chart its own box as well produces
    # a page of rectangles of five different widths and no hierarchy.
    ax.set_facecolor("none")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=SIZE_SMALL, length=0)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)


def install() -> None:
    """Set the matplotlib defaults this report assumes."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": FONT_FAMILY,
        "figure.facecolor": PAGE,
        "savefig.facecolor": PAGE,
        "axes.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": AXIS,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "pdf.fonttype": 42,          # embed TrueType, so the text stays text
    })
