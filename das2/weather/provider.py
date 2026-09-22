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

import logging
from datetime import datetime
from typing import Protocol

import numpy as np
import pandas as pd

from das2.spatial.regions import haversine_m

log = logging.getLogger("das2.weather")

#: Equipment class the classifier assigns to rain gauges.
RAINFALL_EQUIPMENT = "Rainfall"

#: A gauge further than this from an incident says nothing useful about it.
#: Singapore convective rain cells are commonly 2-5 km across, so a gauge 10 km
#: away being wet is not evidence that it rained on the sensor in question.
DEFAULT_GAUGE_RADIUS_M = 5000.0

#: Above this, a single reading is a running total rather than an increment --
#: no tipping bucket reports 50 mm in one 5-minute scan.
#:
#: Kept for reference. It is NOT the test any more: deciding the convention
#: from a magnitude was wrong, and wrong in the dangerous direction. See
#: _gauge_total.
CUMULATIVE_HINT_MM = 50.0

#: Above this over one analysis window, the number is not rainfall.
#:
#: Singapore receives about 2,300 mm in a YEAR and its heaviest recorded day is
#: around 500 mm. A 72-hour total beyond this is arithmetic, not weather, so it
#: is reported as unknown rather than passed on. The first real run produced
#: 26,118 mm -- twenty-six metres -- and nothing downstream noticed, because
#: nothing downstream was checking.
MAX_PLAUSIBLE_WINDOW_MM = 1000.0

#: How close to monotonic a series must be to count as a running total. Not 1.0
#: because a counter that resets, or a single out-of-order scan, should not
#: disqualify it.
MONOTONIC_SHARE = 0.99

#: A tipping bucket that is not tipping reports zero. If most of the window is
#: zeros, the non-zero readings are increments and summing them is right.
ZERO_SHARE_INCREMENTS = 0.5

#: Below this, a reading is the gauge's noise floor, not a tip.
#:
#: Standard tipping buckets resolve 0.1, 0.2 or 0.254 mm, so nothing real
#: arrives under 0.05. Testing against exact zero instead was wrong twice over:
#: a gauge idling at 0.01 counted as never dry, so the series read as neither
#: convention and rainfall came back unknown -- and had it been summed, 0.01
#: across 2,160 scans is 21.6 mm of rain invented out of noise, which is the
#: 26-metre defect again at a size small enough to be believed.
GAUGE_NOISE_FLOOR_MM = 0.05


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

        if values.size < 2:
            return None

        # Decide the convention from the SHAPE of the series, not from how big
        # the numbers are.
        #
        # The magnitude test this replaces -- "any reading over 50 mm means a
        # running total, otherwise sum the readings" -- produced 26,118 mm over
        # 72 hours on the client's real gauges. Twenty-six metres, against a
        # Singapore ANNUAL average near 2,300 mm. The gauges sit at a constant
        # value of about 10, which is under 50, so every scan was added to the
        # total: 10 x 2,160 readings.
        #
        # The damage was not the number. RAIN_EXPLAINS_MM is 2.0, so a
        # permanent 21,600 mm held the rain gate wide open, and any incident
        # whose types were all rain-explicable was classified WEATHER_DRIVEN
        # and withheld from dispatch -- in a country where it rains most days,
        # silently, on the strength of arithmetic.
        diffs = np.diff(values)

        # Constant: nothing accumulated, whichever convention this gauge uses.
        # A running total that has not moved means no rain; a tipping bucket
        # reporting the same non-zero figure every scan for three days is not
        # reporting rain either.
        if np.allclose(diffs, 0.0):
            return 0.0

        # Running total: non-decreasing apart from resets and the odd
        # out-of-order scan. Summing the positive steps handles a midnight
        # reset, which last-minus-first silently turns negative.
        if float(np.mean(diffs >= -1e-9)) >= MONOTONIC_SHARE:
            total = float(np.sum(diffs[diffs > 0]))
        # Tipping bucket: mostly dry, with increments when it tips. "Dry" means
        # below the gauge's resolution, not exactly zero -- a real instrument
        # idles at a few hundredths, and only values above the floor are summed
        # so that idling cannot accumulate into rain.
        elif float(np.mean(np.abs(values) <= GAUGE_NOISE_FLOOR_MM)) >= ZERO_SHARE_INCREMENTS:
            total = float(np.sum(values[values > GAUGE_NOISE_FLOOR_MM]))
        else:
            # Neither shape. A gauge wandering around a non-zero value is
            # reporting something this code cannot read as depth -- an
            # intensity, a raw count, a fault. Unknown is the honest answer,
            # and triage treats it as "no information" rather than "dry", so
            # the rain rule simply does not fire.
            log.debug("gauge %s fits neither convention; rainfall unknown", key)
            return None

        if total > MAX_PLAUSIBLE_WINDOW_MM:
            log.warning(
                "gauge %s totals %.0f mm over the window, which is not "
                "weather -- reporting rainfall as unknown", key, total)
            return None
        return max(0.0, total)

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
