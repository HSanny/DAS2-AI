"""
das2.spatial.correlation
========================

Did the neighbours move too?

This is the single most decision-relevant question the system asks, and it is
the one that most directly answers what the client actually wanted to know
before sending anyone out:

    neighbours moved with it  ->  the WATER moved. Real event, possibly
                                  operational, possibly nothing wrong at all.
    neighbours sat still      ->  the INSTRUMENT is lying. Dispatch a
                                  technician; nothing is happening in the pipe.

No single-sensor detector can distinguish those two. A pressure transducer
reading 30% low looks exactly like a district that really has lost 30% of its
pressure — until you look at the sensor next to it.

Why this is correlation against a baseline, not correlation outright
--------------------------------------------------------------------
Two pressure sensors in one network correlate strongly at all times, because
they share a daily demand cycle. Reporting "r = 0.8 during the incident" would
therefore say almost nothing: 0.8 might be dramatically *less* than their
normal 0.97, which would be evidence of a fault rather than against one.

So correlation is measured over the incident window **and** over a quiet
baseline window before it, and what is reported is the pair. A neighbour that
normally tracks at 0.95 and tracked at 0.2 through the incident is strong
evidence the suspect sensor broke; one that tracked at 0.9 throughout is strong
evidence the event is real.

Alignment, and why it needs care
--------------------------------
Sensors report on independent scans and share almost no exact timestamps, so
raw series cannot be correlated at all — the intersection is usually empty.
They are put on a common grid by last-observation-carried-forward, which is the
correct reading of this feed: a scanned value persists until the next report,
and bucket averaging would fabricate values the instrument never held.

Spearman is reported alongside Pearson because a level shift is monotone but
not linear, and Pearson alone understates it.

Honest limit
------------
Coordinates are RTU-level, so "neighbours within R" means *other sites* within
R, plus every other sensor at the same site at distance zero. Same-site
neighbours are excluded by default: they share an RTU, a power supply and a
panel, so their agreement is evidence about the telemetry rather than about the
water, and including them would quietly turn this into a fan-out detector.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from das2.models import Cluster, SensorAnomaly, SensorMeta
from das2.spatial.regions import haversine_m
from das2.timeutils import to_epoch_seconds

#: How far to look for corroborating sensors. Wider than the cluster radius on
#: purpose: the question is whether the surrounding network reacted, which
#: needs context from outside the cluster itself.
DEFAULT_RADIUS_M = 5000.0

#: Grid the series are aligned onto before correlating.
DEFAULT_GRID_S = 300.0

#: Below this many overlapping points a correlation is not worth reporting.
MIN_OVERLAP_POINTS = 12

#: Quiet window before the incident, used as the "how do these normally relate"
#: control. Six hours gives enough points without reaching back into a
#: different operating regime.
BASELINE_WINDOW_S = 21600.0

#: Minimum span to correlate over, regardless of how long the incident itself
#: was flagged for. Detector windows are often short -- a level shift is
#: reported over the 30 minutes it takes to establish, which on a 5-minute grid
#: is six points, below the overlap minimum -- so every pair was rejected and
#: the whole layer silently reported "no correlation" on a fleet that was
#: plainly moving together. The window is widened symmetrically around the
#: incident, so the comparison still centres on the event.
MIN_CORRELATION_SPAN_S = 10800.0        # 3 hours

#: A neighbour whose values never move over the window carries no information
#: about anything -- correlation with a constant is undefined, not zero.
MIN_NEIGHBOUR_VARIATION = 1e-9

#: Same-site sensors are excluded by default: shared RTU, shared panel, shared
#: power. Their agreement is evidence about the telemetry, not the water.
EXCLUDE_SAME_SITE = True


@dataclass(frozen=True)
class NeighbourResult:
    """One neighbour's verdict on one suspect sensor."""

    sensor_key: str
    neighbour_key: str
    neighbour_description: str
    distance_m: float
    pearson_r: float | None
    spearman_r: float | None
    baseline_r: float | None
    n_points: int

    @property
    def moved_together(self) -> bool:
        return self.pearson_r is not None and abs(self.pearson_r) >= 0.6

    @property
    def decoupled(self) -> bool:
        """
        Normally tracks this sensor, and stopped during the incident.

        The most diagnostic shape available: it says the relationship broke,
        which a shared process change cannot explain but a broken instrument
        can.
        """
        # Judged RELATIVE to the baseline, not against a fixed number. A pair
        # that normally tracks at 0.97 and tracked at 0.35 through the incident
        # has plainly come apart, but 0.35 clears any absolute threshold low
        # enough to be safe on a weakly-coupled pair. Measured on the synthetic
        # broken-instrument case, a fixed `< 0.3` missed it at 0.33.
        return (self.baseline_r is not None and self.pearson_r is not None
                and abs(self.baseline_r) >= 0.6
                and abs(self.pearson_r) < 0.5 * abs(self.baseline_r))


def _align(ts: pd.Series, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """LOCF onto a shared grid. See the module docstring for why not averaging."""
    seconds = to_epoch_seconds(ts)
    finite = np.isfinite(seconds)
    seconds, values = seconds[finite], np.asarray(values, dtype=float)[finite]
    if seconds.size == 0:
        return np.full(grid.shape, np.nan)
    order = np.argsort(seconds)
    seconds, values = seconds[order], values[order]
    idx = np.searchsorted(seconds, grid, side="right") - 1
    out = np.full(grid.shape, np.nan)
    valid = idx >= 0
    out[valid] = values[idx[valid]]
    return out


def _correlate(a: np.ndarray, b: np.ndarray) -> tuple[float | None, float | None, int]:
    """Pearson and Spearman over the finite overlap."""
    both = np.isfinite(a) & np.isfinite(b)
    n = int(both.sum())
    if n < MIN_OVERLAP_POINTS:
        return None, None, n
    x, y = a[both], b[both]
    if np.std(x) < MIN_NEIGHBOUR_VARIATION or np.std(y) < MIN_NEIGHBOUR_VARIATION:
        # One of them never moved. Correlation with a constant is undefined,
        # and returning 0.0 would be read as "did not react", which is a much
        # stronger claim than the data supports.
        return None, None, n
    pearson = float(np.corrcoef(x, y)[0, 1])
    # Spearman is Pearson on the ranks, which needs no scipy.
    rank_x = pd.Series(x).rank().to_numpy()
    rank_y = pd.Series(y).rank().to_numpy()
    spearman = float(np.corrcoef(rank_x, rank_y)[0, 1])
    return (pearson if np.isfinite(pearson) else None,
            spearman if np.isfinite(spearman) else None, n)


def neighbours_of(sensor: SensorMeta, sensors: list[SensorMeta], *,
                  radius_m: float = DEFAULT_RADIUS_M,
                  exclude_same_site: bool = EXCLUDE_SAME_SITE
                  ) -> list[tuple[SensorMeta, float]]:
    """Other sensors within radius, nearest first."""
    if not sensor.has_coords:
        return []
    out = []
    for other in sensors:
        if other.sensor_key == sensor.sensor_key or not other.has_coords:
            continue
        distance = haversine_m(sensor.latitude, sensor.longitude,
                               other.latitude, other.longitude)
        if distance > radius_m:
            continue
        if exclude_same_site and distance < 50.0:
            continue
        out.append((other, distance))
    out.sort(key=lambda pair: pair[1])
    return out


def correlate_anomaly(anomaly: SensorAnomaly, sensors: list[SensorMeta],
                      series: dict[str, tuple[pd.Series, np.ndarray]], *,
                      radius_m: float = DEFAULT_RADIUS_M,
                      grid_s: float = DEFAULT_GRID_S,
                      max_neighbours: int = 8) -> list[NeighbourResult]:
    """
    Correlate one abnormal sensor against its nearest neighbours.

    Limited to the nearest few: a sensor with forty neighbours does not produce
    a better answer from all forty, and the operator reading the result has to
    be able to scan it.
    """
    if anomaly.sensor.sensor_key not in series:
        return []

    incident_start = anomaly.start.timestamp()
    incident_end = anomaly.end.timestamp()

    # Widen a short detector window to something correlatable, keeping it
    # centred on the event. See MIN_CORRELATION_SPAN_S.
    span = incident_end - incident_start
    if span < MIN_CORRELATION_SPAN_S:
        pad = (MIN_CORRELATION_SPAN_S - span) / 2.0
        incident_start -= pad
        incident_end += pad

    incident_grid = np.arange(incident_start, incident_end + grid_s, grid_s)
    baseline_grid = np.arange(incident_start - BASELINE_WINDOW_S,
                              incident_start, grid_s)

    subject_ts, subject_values = series[anomaly.sensor.sensor_key]
    subject_incident = _align(subject_ts, subject_values, incident_grid)
    subject_baseline = _align(subject_ts, subject_values, baseline_grid)

    results: list[NeighbourResult] = []
    for other, distance in neighbours_of(anomaly.sensor, sensors,
                                         radius_m=radius_m)[:max_neighbours]:
        if other.sensor_key not in series:
            continue
        other_ts, other_values = series[other.sensor_key]
        pearson, spearman, n = _correlate(
            subject_incident, _align(other_ts, other_values, incident_grid))
        baseline_r, _, _ = _correlate(
            subject_baseline, _align(other_ts, other_values, baseline_grid))
        if pearson is None and baseline_r is None:
            continue
        results.append(NeighbourResult(
            sensor_key=anomaly.sensor.sensor_key,
            neighbour_key=other.sensor_key,
            neighbour_description=other.description,
            distance_m=round(distance, 1),
            pearson_r=None if pearson is None else round(pearson, 3),
            spearman_r=None if spearman is None else round(spearman, 3),
            baseline_r=None if baseline_r is None else round(baseline_r, 3),
            n_points=n,
        ))
    return results


def cluster_correlation(cluster: Cluster, sensors: list[SensorMeta],
                        series: dict[str, tuple[pd.Series, np.ndarray]], *,
                        radius_m: float = DEFAULT_RADIUS_M,
                        grid_s: float = DEFAULT_GRID_S
                        ) -> tuple[float | None, list[NeighbourResult]]:
    """
    One number for the whole cluster, plus the evidence behind it.

    The summary is the **median** of the neighbour correlations, not the
    maximum. A maximum would be driven by whichever single neighbour happened
    to agree, and with eight neighbours something usually does; the median says
    what the surrounding network did as a whole, which is the question triage
    is actually asking.

    Returns `(None, [])` when no neighbour could be correlated at all. None
    means "unknown", and triage must treat it as unknown rather than as "did
    not react" -- absence of evidence about the neighbours is not evidence that
    the sensor is broken.
    """
    all_results: list[NeighbourResult] = []
    for member in cluster.members:
        all_results.extend(correlate_anomaly(member, sensors, series,
                                             radius_m=radius_m, grid_s=grid_s))

    values = [r.pearson_r for r in all_results if r.pearson_r is not None]
    if not values:
        return None, all_results
    return round(float(np.median(np.abs(values))), 3), all_results


def correlation_summary(results: list[NeighbourResult]) -> dict[str, object]:
    """Counts for the run log and the dashboard."""
    return {
        "pairs": len(results),
        "moved_together": sum(1 for r in results if r.moved_together),
        "decoupled": sum(1 for r in results if r.decoupled),
        "median_r": (round(float(np.median([abs(r.pearson_r) for r in results
                                            if r.pearson_r is not None])), 3)
                     if any(r.pearson_r is not None for r in results) else None),
    }
