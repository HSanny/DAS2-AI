"""
das2.weather.context
====================

Put one cluster's rainfall in the window the water actually came from.

This is the join between `lag` (how long this place takes to answer the rain)
and `provider` (how much fell, weighted across gauges). It exists as its own
module because the pipeline should ask one question -- "what was the rain
context for this cluster?" -- and not have to know that answering it means
gridding two series, correlating them across twenty-five lags, taking a
consensus and then shifting an integration window.

Cost, and why this is affordable
--------------------------------
Learning a lag for every sensor against every gauge would be ~1,600 x 54
pairs a run. It is done per CLUSTER instead, against the gauges near that
cluster, for the members that can actually respond to rain -- a few hundred
correlations a run rather than tens of thousands.

That is also the more honest unit. The lag is a property of a catchment, not
of a reading, and a cluster is the closest thing this system has to a
catchment. Properly it belongs in the daily profile job, learned once over
weeks and persisted alongside the baselines; doing it per run is the version
that works before that job exists, and the interface will not change when it
moves.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from das2.weather.lag import RainLag, consensus, learn_lag
from das2.weather.provider import RainObservation

log = logging.getLogger("das2.weather.context")

#: Equipment classes whose level or rate can respond to rainfall at all.
#: A motor winding temperature does not answer the rain, and correlating it
#: against a gauge would produce a lag from pure coincidence.
RESPONSIVE_EQUIPMENT = frozenset({
    "CanalLevel", "Level", "Flowrate", "Pressure",
})

#: How many members to correlate. The consensus is a median, so a handful of
#: pairs settles it; correlating forty adds cost and no information.
MAX_MEMBERS = 6

#: How many gauges to try per member, nearest first.
MAX_GAUGES = 3

#: Extra window ahead of the lag, so a storm that fell over half an hour is
#: caught rather than just the instant `lag` seconds before the response. A
#: catchment integrates; so should the window measuring what fell on it.
DEFAULT_SPREAD_S = 1800.0


def _series(readings: pd.DataFrame, key: str, start, end):
    rows = readings[(readings["sensor_key"].astype(str) == str(key))
                    & (readings["ts"] >= start) & (readings["ts"] <= end)]
    if rows.empty:
        return None, None
    rows = rows.sort_values("ts")
    return rows["ts"], rows["value"].to_numpy(dtype=float)


def learn_cluster_lag(provider, cluster, readings: pd.DataFrame,
                      window_start, window_end) -> RainLag | None:
    """
    How long this cluster takes to respond to rain, or `None` if unmeasurable.

    Learned over the FULL analysis window, not the incident's window. The
    incident window is minutes long and contains the response but usually not
    the rain that caused it; the full window contains both, and whatever else
    happened over three days, which is what makes the correlation meaningful.
    """
    if provider is None or not getattr(provider, "available", False):
        return None

    members = [m for m in cluster.members
               if m.sensor.equipment in RESPONSIVE_EQUIPMENT][:MAX_MEMBERS]
    if not members:
        return None

    gauges = provider.nearby_gauges(cluster.centroid_lat, cluster.centroid_lon)
    if not gauges:
        return None

    results: list[RainLag] = []
    for member in members:
        level_ts, level_values = _series(
            readings, member.sensor.sensor_key, window_start, window_end)
        if level_ts is None or len(level_ts) < 8:
            continue
        for gauge_key, _distance in gauges[:MAX_GAUGES]:
            # Increments, from the provider, which knows whether this gauge
            # reports a running total or a tipping bucket. Doing the
            # conversion here was the first version's bug: `diff()` is right
            # for one convention and destroys the signal in the other.
            gauge_ts, mm = provider.gauge_increments(
                gauge_key, window_start, window_end)
            if gauge_ts is None or len(gauge_ts) < 8:
                continue
            results.append(learn_lag(level_ts, level_values, gauge_ts, mm,
                                     window_start, window_end,
                                     gauge_key=gauge_key))
    if not results:
        return None
    return consensus(results)


def observe_cluster(provider, cluster, readings: pd.DataFrame,
                    window_start, window_end, *,
                    spread_s: float = DEFAULT_SPREAD_S) -> RainObservation:
    """
    The rain context for one cluster: learn the lag, then integrate over it.

    Returns a `RainObservation` even when nothing is known -- its `context` is
    then `unknown`, which is a different statement from `dry` and one that
    triage has to be able to tell apart.
    """
    if provider is None or not getattr(provider, "available", False):
        return RainObservation()

    lag = learn_cluster_lag(provider, cluster, readings,
                            window_start, window_end)
    return provider.observe(cluster.centroid_lat, cluster.centroid_lon,
                            cluster.start, cluster.end,
                            lag=lag, spread_s=spread_s)


def summarise(observations: dict[str, RainObservation]) -> dict:
    """Run-level counters, for the log line and the report."""
    known = [o for o in observations.values() if o.known]
    lagged = [o for o in known if o.lag and o.lag.confident]
    contexts: dict[str, int] = {}
    for o in observations.values():
        contexts[o.context] = contexts.get(o.context, 0) + 1
    out = {
        "clusters": len(observations),
        "with_rainfall": len(known),
        "lag_measured": len(lagged),
        "context": contexts,
    }
    if lagged:
        out["median_lag_min"] = round(
            float(np.median([o.lag.minutes for o in lagged])), 1)
    return out
