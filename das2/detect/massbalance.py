"""
das2.detect.massbalance
=======================

Water in, water out, and the level in between must agree.

    dLevel/dt * Area  ~=  Q_in - Q_out

This is the only detector here that is genuinely multivariate, and the only one
that can say *which* of several instruments is lying. Everything else reports
that a number looks wrong; this reports that three numbers cannot all be true
at once, which is a physical fact rather than a statistical opinion.

Why this replaced peer comparison
---------------------------------
The plan originally called for cross-sectional peer comparison — score each
sensor against other sensors of the same type. Three measured facts killed it:

  * **Coordinates are RTU-level**, so "nearby peers" collapses to "same site,
    or kilometres away". There is no middle distance to compare across.
  * **Peer groups are tiny.** Dissolved Oxygen: 19 sensors across 12 sites.
    Conductivity: 86 across 39. A MAD over a handful of sensors has exactly
    the small-sample instability that made the incumbent's robust-Z unreliable
    in the first place.
  * **Peers share no physical driver.** Dissolved oxygen at Pulau Tekong,
    Bedok and Upper Seletar are three separate bodies of water. The residual
    between them is noise divided by noise.

It also contradicted a decision already regression-tested in
`cluster_suppression`, which asserts that `UpperSeletarPS-Dissolved-Oxygen` and
`BedokPS-Dissolved-Oxygen` must *not* be grouped.

Mass balance has none of those problems: the three signals are physically
coupled by plumbing, the relationship is a conservation law rather than a
correlation, and it works at n=3.

What it cannot do, and says so
------------------------------
The tank area is almost never known. Rather than guess it — a wrong area turns
every filling cycle into a violation — the area is **fitted from the data**:
over a quiet stretch the relationship between `dLevel/dt` and `Qin - Qout`
gives the effective area directly. If the fit is poor, the group is not
actually a closed system (an unmetered overflow, a second inlet, a level sensor
on a different tank) and the detector abstains rather than reporting nonsense.

That abstention is the point. A mass-balance detector that fires on every
plumbing arrangement it does not understand would be worse than none.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from das2.models import AnomalyType, Signal
from das2.timeutils import to_epoch_seconds

DETECTOR = "massbalance"

#: Equipment classes that can play each role in the balance.
LEVEL_CLASSES = frozenset({"Level", "LevelSensor"})
FLOW_CLASSES = frozenset({"Flowrate"})

#: Words in a description that mark a flow as entering or leaving. Inflow and
#: outflow cannot be told apart by equipment class -- both are `Flowrate` -- so
#: the direction has to come from the name, which is the only place it exists.
INLET_WORDS = ("inlet", "influent", "intake", "inflow", "raw-water",
               "rawwater", "feed", "supply", "in-flow")
OUTLET_WORDS = ("outlet", "effluent", "discharge", "delivery", "outflow",
                "deliver", "out-flow", "export")

#: Grid the three series are resampled onto before differencing. Long enough
#: that level-sensor quantisation does not dominate dLevel/dt, short enough to
#: see a real imbalance develop.
GRID_S = 900.0                    # 15 minutes

#: Minimum aligned points before a balance can be fitted or judged.
MIN_POINTS = 40

#: The fitted area must explain at least this much of the variance, or the
#: group is not a closed system and the detector abstains.
MIN_FIT_R2 = 0.5

#: A violation must exceed this many robust sigma of the residual's own spread.
VIOLATION_SIGMA = 5.0

#: ...and persist this long. A single bad sample is a spike, not an imbalance.
VIOLATION_MIN_S = 1800.0


@dataclass
class BalanceGroup:
    """One tank or reservoir: a level, and the flows in and out of it."""

    site: str
    level_key: str = ""
    inlet_keys: list[str] = field(default_factory=list)
    outlet_keys: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """A balance needs a level and at least one metered flow."""
        return bool(self.level_key) and bool(self.inlet_keys or self.outlet_keys)

    @property
    def member_keys(self) -> list[str]:
        return [self.level_key, *self.inlet_keys, *self.outlet_keys]


def _direction(description: str) -> str | None:
    text = (description or "").lower()
    if any(word in text for word in INLET_WORDS):
        return "in"
    if any(word in text for word in OUTLET_WORDS):
        return "out"
    return None


def find_groups(sensors: pd.DataFrame) -> list[BalanceGroup]:
    """
    Assemble balance groups from the inventory, one per site.

    Grouped by the same site prefix the name-based suppression rule already
    uses, so the two agree about what "one place" means. A site with several
    tanks will be mixed together here, which the fit quality check is what
    ultimately catches: mixed tanks do not balance, the fit fails, and the
    group is dropped.
    """
    groups: dict[str, BalanceGroup] = {}
    for _, row in sensors.iterrows():
        site = row.get("site")
        if site is None or pd.isna(site) or not str(site):
            continue
        site = str(site)
        equipment = str(row.get("equipment") or "")
        key = str(row["sensor_key"])
        group = groups.setdefault(site, BalanceGroup(site=site))

        if equipment in LEVEL_CLASSES and not group.level_key:
            group.level_key = key
        elif equipment in FLOW_CLASSES:
            direction = _direction(str(row.get("description") or ""))
            if direction == "in":
                group.inlet_keys.append(key)
            elif direction == "out":
                group.outlet_keys.append(key)

    return [g for g in groups.values() if g.usable]


def _on_grid(ts: pd.Series, values: np.ndarray,
             grid: np.ndarray) -> np.ndarray:
    """
    Values sampled onto a common grid by last-observation-carried-forward.

    LOCF rather than interpolation or bucket means, because this feed re-reports
    a held value on every scan: a reading persists until the next one, and
    averaging would invent numbers the instrument never produced.
    """
    seconds = to_epoch_seconds(ts)
    order = np.argsort(seconds)
    seconds, values = seconds[order], np.asarray(values, dtype=float)[order]
    idx = np.searchsorted(seconds, grid, side="right") - 1
    out = np.full(grid.shape, np.nan)
    valid = idx >= 0
    out[valid] = values[idx[valid]]
    return out


def evaluate_group(group: BalanceGroup,
                   series: dict[str, tuple[pd.Series, np.ndarray]],
                   *, unit: str = "") -> list[Signal]:
    """
    Check one group's balance, returning a signal per sustained violation.

    The signal is attached to the **level** sensor, because that is the series
    the imbalance is expressed against — but `detail` names every member, so
    triage and the dashboard can show the operator all three readings and let
    them decide which one to disbelieve. Naming a culprit here would be a
    guess; showing the contradiction is the finding.
    """
    if group.level_key not in series:
        return []

    members = [k for k in group.member_keys if k in series]
    if len(members) < 2:
        return []

    # Common grid across the overlap of every member.
    starts, ends = [], []
    for key in members:
        seconds = to_epoch_seconds(series[key][0])
        seconds = seconds[np.isfinite(seconds)]
        if seconds.size == 0:
            return []
        starts.append(seconds.min())
        ends.append(seconds.max())
    t0, t1 = max(starts), min(ends)
    if t1 - t0 < MIN_POINTS * GRID_S:
        return []
    grid = np.arange(t0, t1 + GRID_S, GRID_S)

    level = _on_grid(*series[group.level_key], grid)
    inflow = np.zeros_like(grid)
    outflow = np.zeros_like(grid)
    for key in group.inlet_keys:
        if key in series:
            inflow = inflow + np.nan_to_num(_on_grid(*series[key], grid))
    for key in group.outlet_keys:
        if key in series:
            outflow = outflow + np.nan_to_num(_on_grid(*series[key], grid))

    net_flow = inflow - outflow
    d_level = np.gradient(level, GRID_S)

    finite = np.isfinite(d_level) & np.isfinite(net_flow)
    if finite.sum() < MIN_POINTS:
        return []

    # Fit the effective area: net_flow ~= area * dLevel/dt. Through the origin,
    # because zero net flow must mean a level that is not moving -- an intercept
    # would absorb a real steady leak into the fitted geometry and hide exactly
    # what this detector exists to find.
    x = d_level[finite]
    y = net_flow[finite]
    denominator = float(np.dot(x, x))
    if denominator <= 0:
        return []
    area = float(np.dot(x, y) / denominator)
    if not np.isfinite(area) or area == 0:
        return []

    predicted = area * x
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    if r2 < MIN_FIT_R2:
        # Not a closed system as configured: an unmetered overflow, a second
        # inlet, or a level sensor on a different tank. Abstain rather than
        # report a violation that is really a modelling failure.
        return []

    residual = np.full(grid.shape, np.nan)
    residual[finite] = y - predicted
    spread = 1.4826 * float(np.nanmedian(np.abs(residual - np.nanmedian(residual))))
    if not np.isfinite(spread) or spread <= 0:
        return []

    bad = np.isfinite(residual) & (np.abs(residual) >= VIOLATION_SIGMA * spread)
    if not bad.any():
        return []

    idx = np.flatnonzero(bad)
    splits = np.flatnonzero(np.diff(idx) > 1)
    starts_i = np.r_[idx[0], idx[splits + 1]]
    ends_i = np.r_[idx[splits], idx[-1]]

    level_ts = series[group.level_key][0]
    epoch = pd.Timestamp("1970-01-01")
    signals: list[Signal] = []
    for i, j in zip(starts_i.tolist(), ends_i.tolist()):
        duration = grid[j] - grid[i]
        if duration < VIOLATION_MIN_S:
            continue
        peak = float(np.nanmax(np.abs(residual[i:j + 1])))
        signals.append(Signal(
            type=AnomalyType.MASS_BALANCE_VIOLATION,
            start=(epoch + pd.Timedelta(seconds=float(grid[i]))).to_pydatetime(),
            end=(epoch + pd.Timedelta(seconds=float(grid[j]))).to_pydatetime(),
            detector=DETECTOR,
            magnitude=round(peak, 6),
            unit=unit,
            n_points=j - i + 1,
            detail={
                "site": group.site,
                "level_sensor": group.level_key,
                "inlets": group.inlet_keys,
                "outlets": group.outlet_keys,
                "fitted_area": round(area, 4),
                "fit_r2": round(r2, 3),
                "residual_sigma": round(spread, 6),
                "peak_sigma": round(peak / spread, 1),
                "verdict": "level, inflow and outflow cannot all be correct; "
                           "check all three instruments",
            },
        ))
    return signals


def run_mass_balance(sensors: pd.DataFrame,
                     series: dict[str, tuple[pd.Series, np.ndarray]]
                     ) -> dict[str, list[Signal]]:
    """
    Every balance group in the fleet. Returns {level_sensor_key: [Signal, ...]}.

    Keyed on the level sensor so fusion can attach the finding to a real
    sensor, while `detail` carries the whole group.
    """
    out: dict[str, list[Signal]] = {}
    for group in find_groups(sensors):
        signals = evaluate_group(group, series)
        if signals:
            out.setdefault(group.level_key, []).extend(signals)
    return out


def balance_summary(groups: list[BalanceGroup]) -> dict[str, object]:
    """What the fleet's plumbing looks like, for the run log."""
    return {
        "groups": len(groups),
        "with_both_directions": sum(1 for g in groups
                                    if g.inlet_keys and g.outlet_keys),
        "sites": sorted({g.site for g in groups})[:10],
    }
