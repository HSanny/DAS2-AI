"""
das2.weather.provider
=====================

Rain context, sourced from the client's own gauges.

This was going to be an external weather API integration, and it turned out not
to need one. The inventory contains **188 sensors matching `*-Rainfall`**
(`Pandan1PS-Rainfall`, `BedokPS-Rainfall`, ...), all of which v1 dropped into
its unclassified bucket and discarded before detection. The rain context the
client asked for is therefore not an integration at all -- it is un-dropping
data they already collect, on their own network, with no internet dependency,
no API key, and no third-party availability to depend on.

Why it matters operationally
----------------------------
A flow or level excursion during a downpour is the most common wasted trip in a
water network: the numbers really did move, nothing is broken, and somebody
drives out anyway. Attaching rainfall to the incident lets triage say
`WEATHER_DRIVEN -- monitor only, do not dispatch`, with the millimetres quoted
so the operator can disagree.

What it deliberately does not do
--------------------------------
Rain is only allowed to explain types that rain can physically cause. A
transmitter that has stopped reporting is not explained by weather, however
hard it is raining, so `triage.RAIN_EXPLICABLE` gates this and a STALE sensor
in a storm is still a broken sensor.

Gauges are totalised, not averaged
----------------------------------
Tipping-bucket gauges report an increment per tip, so the quantity of interest
over a window is a **sum**, not a mean. Averaging would report a light drizzle
for a cloudburst. Where the gauge instead reports a running daily total, the
increase across the window is the right measure; both are handled below by
taking whichever interpretation the data supports.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

import numpy as np
import pandas as pd

from das2.spatial.regions import haversine_m

#: Equipment class the classifier assigns to rain gauges.
RAINFALL_EQUIPMENT = "Rainfall"

#: A gauge further than this from an incident says nothing useful about it.
#: Singapore convective rain cells are commonly 2-5 km across, so a gauge 10 km
#: away being wet is not evidence that it rained on the sensor in question.
DEFAULT_GAUGE_RADIUS_M = 5000.0

#: Above this, a single reading is a running total rather than an increment --
#: no tipping bucket reports 50 mm in one 5-minute scan.
CUMULATIVE_HINT_MM = 50.0


class WeatherProvider(Protocol):
    """
    The interface triage depends on.

    Kept minimal and abstract so a public forecast API can be added later
    without touching anything else. Forecast is the one thing internal gauges
    genuinely cannot supply, and is the only reason to want an external source.
    """

    def rainfall_mm(self, lat: float | None, lon: float | None,
                    start: datetime, end: datetime) -> float | None:
        ...


class NullWeatherProvider:
    """No rain data. Returns None, which triage reads as 'unknown', not 'dry'."""

    def rainfall_mm(self, lat, lon, start, end) -> float | None:
        return None

    def rainfall_by_region(self, start, end) -> dict[str, float]:
        return {}


class InternalRainGaugeProvider:
    """
    Rainfall from the 188 `*-Rainfall` sensors in the client's own inventory.

    Constructed per run from the same readings frame the detectors use, so it
    adds no I/O and cannot disagree with the rest of the run about what
    happened.
    """

    def __init__(self, readings: pd.DataFrame, sensors: pd.DataFrame, *,
                 radius_m: float = DEFAULT_GAUGE_RADIUS_M):
        self.radius_m = radius_m
        gauges = sensors[sensors["equipment"] == RAINFALL_EQUIPMENT]
        self.gauges = gauges.reset_index(drop=True)
        keys = set(gauges["sensor_key"].astype(str))
        self.readings = readings[readings["sensor_key"].astype(str).isin(keys)]

    @property
    def available(self) -> bool:
        return not self.gauges.empty and not self.readings.empty

    def _gauge_total(self, key: str, start: datetime, end: datetime) -> float | None:
        """
        Millimetres at one gauge over the window.

        Handles both reporting conventions. A tipping bucket reports an
        increment per scan, so the window total is the sum; a gauge reporting a
        running total would give an absurd sum, so the rise across the window
        is used instead. The distinction is made from the data rather than
        configured, because nothing in the feed declares which kind a gauge is.
        """
        rows = self.readings[self.readings["sensor_key"].astype(str) == str(key)]
        rows = rows[(rows["ts"] >= start) & (rows["ts"] <= end)]
        if rows.empty:
            return None
        values = rows.sort_values("ts")["value"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return None

        if float(np.max(values)) >= CUMULATIVE_HINT_MM:
            rise = float(values[-1] - values[0])
            return max(0.0, rise)
        return float(np.sum(values[values > 0]))

    def rainfall_mm(self, lat: float | None, lon: float | None,
                    start: datetime, end: datetime) -> float | None:
        """
        Rainfall near one point, as the maximum over gauges within radius.

        Maximum rather than mean: the question triage asks is "did rain fall on
        this thing?", and one wet gauge 2 km away answers yes. Averaging it
        against three dry gauges further out would answer no to a question
        nobody asked.
        """
        if lat is None or lon is None or not self.available:
            return None

        totals = []
        for _, gauge in self.gauges.iterrows():
            glat, glon = gauge.get("latitude"), gauge.get("longitude")
            if glat is None or glon is None or pd.isna(glat) or pd.isna(glon):
                continue
            if haversine_m(lat, lon, float(glat), float(glon)) > self.radius_m:
                continue
            total = self._gauge_total(str(gauge["sensor_key"]), start, end)
            if total is not None:
                totals.append(total)

        return round(max(totals), 2) if totals else None

    def rainfall_by_region(self, start: datetime, end: datetime) -> dict[str, float]:
        """
        Wettest gauge per region over the window.

        Region rather than point, because that is the level the placement is
        trustworthy at, and because an incident's members can span several
        sites whose individual rain totals would only disagree.
        """
        if not self.available:
            return {}
        out: dict[str, float] = {}
        for _, gauge in self.gauges.iterrows():
            region = gauge.get("region")
            if region is None or pd.isna(region):
                continue
            total = self._gauge_total(str(gauge["sensor_key"]), start, end)
            if total is None:
                continue
            region = str(region)
            out[region] = max(out.get(region, 0.0), round(total, 2))
        return out
