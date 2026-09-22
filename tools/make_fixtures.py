#!/usr/bin/env python3
"""
make_fixtures.py
================

Generate synthetic Fujitsu-format telemetry with KNOWN injected faults, plus a
manifest recording exactly what was injected where.

Why this exists
---------------
There is no labelled ground truth anywhere in this system, and no access from
here to the real Fujitsu share, so every later phase would otherwise be tested
against assertions I wrote to match whatever the code already did. Injected
faults invert that: the fixture states what is wrong before any detector runs,
so a detector can genuinely fail its test.

It also doubles as the client's own test fixture, and as the fault-injection
corpus the shadow-mode harness needs to produce detectability curves
("a flatline >= 25 min is caught 95% of the time; below 12 min it is not").

Faithful to the real feed, including the awkward parts
------------------------------------------------------
Generating clean, clock-aligned data would make the later phases look correct
while hiding the exact problems that matter here. So the output reproduces:

* **Report-by-exception timing.** `processed/dateDim.csv` holds 159,347
  distinct timestamps over 71 hours at 1-second resolution, spread essentially
  uniformly across every second-of-minute and minute-of-hour. Timestamps here
  are jittered accordingly -- nothing lands on a neat boundary.
* **8x spread in reporting rate.** The real fleet ranges from one report per
  ~16 s (PulauTekong DO) to one per ~124 s (Kranji1PS flow), which is what made
  sample-counted windows meaningless.
* **Duplicate and out-of-order timestamps**, which occur in the real feed and
  turn a naive dv/dt into an infinite-rate spike.
* **Idle equipment sitting at zero with symmetric noise** -- the shape that
  made a raw `value < 0` range test flag ~53% of one healthy meter's samples.
* **RTU-level coordinates**: every sensor on one RTU shares identical lat/lon,
  because that is what the `RTUNumber -> LKey` join produces.
* **Dirty RTU numbers** (0, 1, -1) alongside real ones, so coordinate-coverage
  handling is exercised rather than assumed.

Usage
-----
    python3 tools/make_fixtures.py --out fixtures/ --hours 72
    python3 tools/make_fixtures.py --out fixtures/ --seed 7 --clean

Writes:
    <out>/HISTORY/hts_YYYY_MM_HISTORY_<ts>.csv    hourly, ';'-separated
    <out>/HISTCURR/histcurr_fujitsu.csv           sensor inventory, ','-separated
    <out>/HISTALMEVT/hts_..._HISTALMEVT_...csv    kept for v1 compatibility only
    <out>/LongLat.csv                             RTU coordinates
    <out>/manifest.json                           ground truth
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

# --------------------------------------------------------------------------- #
# Fleet definition
# --------------------------------------------------------------------------- #


@dataclass
class SiteSpec:
    name: str
    rtu: str
    lat: float
    lon: float
    region: str


#: Real PUB site names and coordinates, so region assignment is exercised
#: against the same values the production join would produce.
SITES: tuple[SiteSpec, ...] = (
    SiteSpec("BedokPS", "1010", 1.34286, 103.91977, "East"),
    SiteSpec("BedokPond4", "1011", 1.31697, 103.93443, "East"),
    SiteSpec("TampinesPS", "1012", 1.34960, 103.95680, "East"),
    SiteSpec("Kranji1PS", "1020", 1.41567, 103.72871, "North"),
    SiteSpec("Kranji2PS", "1021", 1.42483, 103.74441, "North"),
    SiteSpec("PandanTG", "1030", 1.31214, 103.74723, "West"),
    SiteSpec("Pandan1PS", "1031", 1.31500, 103.75100, "West"),
    SiteSpec("MRRS", "1040", 1.34235, 103.83571, "Central"),
    SiteSpec("PulauTekong", "1050", 1.40396, 104.05330, "North-East"),
    # Deliberately dirty RTU: the join will not resolve coordinates for this
    # one, so the site-name fallback and the coverage reporting get exercised.
    SiteSpec("UnknownSite", "-1", 0.0, 0.0, "Unknown"),
)

#: RawType drives Analog/Digital in the real pipeline: 1 and 5 are analog.
RAWTYPE_ANALOG = 1
RAWTYPE_DIGITAL = 3


@dataclass
class SensorSpec:
    """One synthetic sensor, and the fault (if any) injected into it."""

    description: str
    equipment: str
    site: str
    rtu: str
    ip: int
    row_id: int
    dt_s: float                      # nominal seconds between reports
    base: float
    noise: float
    unit: str = ""
    rawtype: int = RAWTYPE_ANALOG
    diurnal_amp: float = 0.0         # amplitude of the daily cycle
    fault: str | None = None         # matches an AnomalyType name, or None
    fault_start_frac: float = 0.5    # where in the window the fault begins
    fault_duration_h: float = 2.0
    fault_detail: dict = field(default_factory=dict)

    @property
    def sensor_key(self) -> str:
        return f"{self.ip}{self.row_id}"


def build_fleet() -> list[SensorSpec]:
    """
    The synthetic fleet: clean controls, one sensor per injected fault type, a
    multi-site regional event, and a single-RTU fan-out group.
    """
    s: list[SensorSpec] = []
    ip = 3232235800  # arbitrary but stable base for IPADDRESS
    row = 100

    def add(**kw) -> SensorSpec:
        nonlocal ip, row
        row += 1
        spec = SensorSpec(ip=ip + int(kw["rtu"]) if kw["rtu"].isdigit() else ip,
                          row_id=row, **kw)
        s.append(spec)
        return spec

    # --- clean controls, at both ends of the real sampling-rate spread ------
    # Any detector that fires on these is producing false positives.
    add(description="BedokPS-Pump1-Delivery-Pressure", equipment="Pressure",
        site="BedokPS", rtu="1010", dt_s=120, base=3.90, noise=0.05, unit="bar",
        diurnal_amp=0.25)
    add(description="PulauTekong-Dissolved-Oxygen", equipment="Dissolved Oxygen",
        site="PulauTekong", rtu="1050", dt_s=16, base=5.30, noise=0.08, unit="mg/L",
        diurnal_amp=0.40)
    add(description="Kranji2PS-Mains-Delivery-PRESSURE", equipment="Pressure",
        site="Kranji2PS", rtu="1021", dt_s=124, base=4.10, noise=0.06, unit="bar",
        diurnal_amp=0.30)
    # Idle flowmeter, straddling zero: the shape that made a raw `value < 0`
    # range test flag ~53% of a healthy meter's samples, producing 1079 points
    # and 542 events in a smoke run. The real MRRS-THOMSON meter has
    # Median_Value 0.001302 -- it is idle almost always. Must stay silent.
    add(description="MRRS-THOMSON FLOWMETER READING", equipment="Flowrate",
        site="MRRS", rtu="1040", dt_s=104, base=0.0, noise=0.002, unit="L/s")

    # --- one sensor per injected sensor-health fault ------------------------
    add(description="Kranji1PS-Total-Flow-Rate", equipment="Flowrate",
        site="Kranji1PS", rtu="1020", dt_s=124, base=45.0, noise=1.2, unit="L/s",
        diurnal_amp=8.0, fault="FLATLINE", fault_start_frac=0.40, fault_duration_h=3.0)
    add(description="TampinesPS-Inlet-Pressure", equipment="Pressure",
        site="TampinesPS", rtu="1012", dt_s=120, base=3.60, noise=0.05, unit="bar",
        fault="STALE", fault_start_frac=0.55, fault_duration_h=4.0)
    add(description="BedokPond4-PS-Delivery-Flow-Rate", equipment="Flowrate",
        site="BedokPond4", rtu="1011", dt_s=106, base=30.0, noise=0.9, unit="L/s",
        fault="RANGE_VIOLATION", fault_start_frac=0.60, fault_duration_h=0.2,
        fault_detail={"value": -25.0})
    add(description="PandanTG-LT-Voltage1", equipment="Voltage",
        site="PandanTG", rtu="1030", dt_s=122, base=414.8, noise=0.05, unit="V",
        fault="SPIKE", fault_start_frac=0.45, fault_duration_h=0.05,
        fault_detail={"value": 380.0, "quantise": 0.1})
    add(description="Pandan1PS-Motor-Winding-Temperature", equipment="Temperature",
        site="Pandan1PS", rtu="1031", dt_s=120, base=42.0, noise=0.30, unit="C",
        fault="DRIFT", fault_start_frac=0.10, fault_duration_h=60.0,
        fault_detail={"total_change": 6.0})
    add(description="BedokPS-Pump2-Discharge-Pressure", equipment="Pressure",
        site="BedokPS", rtu="1010", dt_s=120, base=3.75, noise=0.04, unit="bar",
        fault="QUANTISATION_COLLAPSE", fault_start_frac=0.50, fault_duration_h=20.0,
        fault_detail={"step": 0.5})
    add(description="Kranji2PS-Raw-Water-Conductivity", equipment="Conductivity",
        site="Kranji2PS", rtu="1021", dt_s=120, base=250.0, noise=3.0, unit="uS/cm",
        fault="NOISE_BURST", fault_start_frac=0.55, fault_duration_h=5.0,
        fault_detail={"multiplier": 9.0})

    # --- REGIONAL EVENT ------------------------------------------------------
    # Four sensors, three different EAST sites, two equipment types, all
    # stepping together. This is the case v1 is structurally blind to: name
    # clustering sees four unrelated names, so nothing links them.
    for desc, site, rtu, equip, base, noise, unit in [
        ("BedokPS-Trunk-Main-Pressure", "BedokPS", "1010", "Pressure", 4.20, 0.05, "bar"),
        ("BedokPond4-Outlet-Flow", "BedokPond4", "1011", "Flowrate", 22.0, 0.8, "L/s"),
        ("TampinesPS-Trunk-Pressure", "TampinesPS", "1012", "Pressure", 4.05, 0.05, "bar"),
        ("TampinesPS-Outlet-Flow", "TampinesPS", "1012", "Flowrate", 18.0, 0.7, "L/s"),
    ]:
        add(description=desc, equipment=equip, site=site, rtu=rtu, dt_s=120,
            base=base, noise=noise, unit=unit, diurnal_amp=base * 0.06,
            fault="REGIONAL_EVENT", fault_start_frac=0.70, fault_duration_h=2.5,
            fault_detail={"relative_step": -0.35})

    # --- TELEMETRY FAN-OUT ---------------------------------------------------
    # Three sensors on ONE RTU with near-identical names, all going at once.
    # The opposite verdict to the regional event above: suppress, do not
    # dispatch. This is what cluster_suppression's name rule already catches.
    for i in (1, 2, 3):
        add(description=f"PandanTG-LT-Current{i}", equipment="Current",
            site="PandanTG", rtu="1030", dt_s=122, base=120.0 + i, noise=0.5,
            unit="A", fault="TELEMETRY_FANOUT", fault_start_frac=0.80,
            fault_duration_h=1.0, fault_detail={"relative_step": 0.5})

    # --- a real tank: level + metered inflow + metered outflow ---------------
    # Mass balance is the only genuinely multivariate detector, and without a
    # site carrying all three signals there is nothing for it to check. The
    # outflow meter under-reads for six hours, which is the fault that makes
    # the level, the inflow and the outflow mutually inconsistent -- and which
    # no single-sensor detector can see, because every one of the three
    # readings stays entirely plausible on its own.
    add(description="Kranji1PS-Service-Reservoir-Level", equipment="LevelSensor",
        site="Kranji1PS", rtu="1020", dt_s=120, base=3.0, noise=0.002, unit="m",
        fault="TANK_LEVEL", fault_start_frac=0.30, fault_duration_h=6.0)
    # Inlet steady, outlet following demand. They must NOT share a profile: if
    # inflow and outflow track each other the net flow is ~0, the level barely
    # moves, and there is no relationship for mass balance to fit at all --
    # which is exactly what the first version of this fixture did, producing a
    # fit r2 of -0.06 and a detector that correctly abstained on data carrying
    # no information.
    add(description="Kranji1PS-Reservoir-Inlet-Flow", equipment="Flowrate",
        site="Kranji1PS", rtu="1020", dt_s=120, base=0.50, noise=0.008,
        unit="m3/s", fault="TANK_INLET", fault_start_frac=0.30,
        fault_duration_h=6.0)
    add(description="Kranji1PS-Reservoir-Outlet-Flow", equipment="Flowrate",
        site="Kranji1PS", rtu="1020", dt_s=120, base=0.50, noise=0.008,
        unit="m3/s", diurnal_amp=0.18, fault="TANK_OUTLET",
        fault_start_frac=0.30, fault_duration_h=6.0,
        fault_detail={"under_read": 0.7})

    # --- rain gauges ---------------------------------------------------------
    # 188 of these exist in the real feed and are currently discarded as
    # unclassified. One rains during the regional event, which is what lets
    # triage separate WEATHER_DRIVEN from a genuine fault.
    add(description="BedokPS-Rainfall", equipment="Rainfall", site="BedokPS",
        rtu="1010", dt_s=300, base=0.0, noise=0.01, unit="mm",
        fault="RAINFALL", fault_start_frac=0.70, fault_duration_h=2.5,
        fault_detail={"peak_mm_per_interval": 1.8})
    add(description="Pandan1PS-Rainfall", equipment="Rainfall", site="Pandan1PS",
        rtu="1031", dt_s=300, base=0.0, noise=0.01, unit="mm")

    # --- digital pump: ~1,567 of these run through the analog stack today ----
    add(description="BedokPS-Pump3-Run-Status", equipment="Pump", site="BedokPS",
        rtu="1010", dt_s=60, base=0.0, noise=0.0, unit="", rawtype=RAWTYPE_DIGITAL,
        fault="SHORT_CYCLING", fault_start_frac=0.35, fault_duration_h=3.0,
        fault_detail={"period_s": 180})

    # --- a pump and the meter on its own discharge --------------------------
    # RUN_STATE_INCONSISTENT is the second cross-signal detector, and it needs
    # a pump paired with the flowmeter it feeds. Nothing in the feed declares
    # that relationship, so the pairing is recovered from the descriptions --
    # which means the fixture has to carry a pair whose names actually match,
    # or the detector is never exercised at all.
    add(description="TampinesPS-Pump1-Run-Status", equipment="Pump",
        site="TampinesPS", rtu="1012", dt_s=60, base=0.0, noise=0.0, unit="",
        rawtype=RAWTYPE_DIGITAL, fault="PUMP_RUN", fault_start_frac=0.55,
        fault_duration_h=4.0)
    add(description="TampinesPS-Pump1-Discharge-Flow", equipment="Flowrate",
        site="TampinesPS", rtu="1012", dt_s=120, base=0.0, noise=0.05,
        unit="L/s", fault="PUMP_FLOW", fault_start_frac=0.55,
        fault_duration_h=4.0)

    # --- unplaceable sensor: dirty RTU, so no coordinates from the join ------
    add(description="UnknownSite-Mystery-Level", equipment="LevelSensor",
        site="UnknownSite", rtu="-1", dt_s=120, base=55.0, noise=0.8, unit="%")

    return s


# --------------------------------------------------------------------------- #
# Series generation
# --------------------------------------------------------------------------- #
def _diurnal(ts: datetime, amp: float) -> float:
    """Daily cycle. Water demand is strongly diurnal; a detector that ignores
    it will flag every morning peak."""
    if amp == 0:
        return 0.0
    hours = ts.hour + ts.minute / 60.0
    return amp * math.sin(2 * math.pi * (hours - 6.0) / 24.0)


def generate_series(spec: SensorSpec, start: datetime, end: datetime,
                    rng: random.Random, clean: bool = False) -> list[tuple[datetime, float]]:
    """
    Build one sensor's readings, applying its injected fault.

    Timestamps are jittered rather than placed on a grid: the real feed is
    report-by-exception at 1-second resolution with no clock alignment, and
    testing against neatly spaced data would hide exactly the problems that
    matter.
    """
    total_s = (end - start).total_seconds()
    fault_start = start + timedelta(seconds=total_s * spec.fault_start_frac)
    fault_end = fault_start + timedelta(hours=spec.fault_duration_h)
    detail = spec.fault_detail
    fault = None if clean else spec.fault

    points: list[tuple[datetime, float]] = []
    t = start
    stale_until: datetime | None = None
    flat_value: float | None = None
    pump_state = 0.0

    while t < end:
        in_fault = fault is not None and fault_start <= t < fault_end

        # --- STALE: the sensor stops reporting entirely --------------------
        if fault == "STALE" and in_fault:
            stale_until = fault_end
        if stale_until and t < stale_until:
            t += timedelta(seconds=spec.dt_s)
            continue

        # --- value ----------------------------------------------------------
        if spec.rawtype == RAWTYPE_DIGITAL:
            if fault == "SHORT_CYCLING" and in_fault:
                period = detail.get("period_s", 180)
                pump_state = 1.0 if int((t - fault_start).total_seconds() // (period / 2)) % 2 else 0.0
            else:
                # Normal duty: a long, slow on/off cycle.
                pump_state = 1.0 if int((t - start).total_seconds() // 5400) % 2 else 0.0
            value = pump_state
        else:
            value = spec.base + _diurnal(t, spec.diurnal_amp)
            noise = spec.noise

            if fault == "NOISE_BURST" and in_fault:
                noise *= detail.get("multiplier", 8.0)
            if fault == "DRIFT" and t >= fault_start:
                progress = min(1.0, (t - fault_start).total_seconds() /
                               max(1.0, (fault_end - fault_start).total_seconds()))
                value += detail.get("total_change", 5.0) * progress
            if fault in ("REGIONAL_EVENT", "TELEMETRY_FANOUT") and in_fault:
                value += spec.base * detail.get("relative_step", -0.3)
            if fault == "RAINFALL" and in_fault:
                # Rain arrives in bursts, not at a constant rate.
                value += detail.get("peak_mm_per_interval", 1.5) * rng.uniform(0.2, 1.0)

            value += rng.gauss(0.0, noise)

            if fault == "FLATLINE" and in_fault:
                if flat_value is None:
                    flat_value = value
                value = flat_value
            elif fault != "FLATLINE":
                flat_value = None

            if fault == "RANGE_VIOLATION" and in_fault:
                value = detail.get("value", -25.0)
            if fault == "SPIKE" and in_fault:
                value = detail.get("value", 0.0)

            # Quantisation: normally fine, collapsing to a coarse step during
            # the fault. Not a flatline (variance > 0) and not out of range, so
            # nothing else detects it.
            step = detail.get("quantise")
            if fault == "QUANTISATION_COLLAPSE" and in_fault:
                step = detail.get("step", 0.5)
            if step:
                value = round(value / step) * step

        points.append((t, float(value)))

        # --- advance, with report-by-exception jitter -----------------------
        jitter = rng.uniform(-0.25, 0.25) * spec.dt_s
        t += timedelta(seconds=max(1.0, spec.dt_s + jitter))

    if not clean and points:
        _inject_timestamp_defects(points, rng)
    return points


#: The pump and the meter on its discharge, and the fault between them.
PUMP_RUN = "TampinesPS-Pump1-Run-Status"
PUMP_FLOW = "TampinesPS-Pump1-Discharge-Flow"
PUMP_CYCLE_S = 4 * 3600.0          # 4 h on, 4 h off
PUMP_RUNNING_FLOW = 24.0           # L/s when it is actually pumping
PUMP_FAULT_START_FRAC = 0.55
PUMP_FAULT_HOURS = 4.0


def couple_pump(series: dict[str, list[tuple[datetime, float]]],
                start: datetime, end: datetime, *, clean: bool = False) -> None:
    """
    Make the pump's run state and its discharge flow consistent, then break it.

    Generated independently these two have no relationship, and the detector
    would either see contradictions everywhere or nothing at all. Here the flow
    follows the run state exactly -- pumping when the pump says it is running,
    zero when it is not -- so the pair is consistent by construction.

    Then, for four hours, the pump goes on reporting RUNNING while the meter
    reads nothing. That contradiction is invisible to every single-sensor
    detector: a pump that says it is running is unremarkable, and a flowmeter
    reading zero is unremarkable. Only the two together are impossible.
    """
    if PUMP_RUN not in series or PUMP_FLOW not in series:
        return

    total_s = (end - start).total_seconds()
    fault_start = start + timedelta(seconds=total_s * PUMP_FAULT_START_FRAC)
    fault_end = fault_start + timedelta(hours=PUMP_FAULT_HOURS)

    def running(when: datetime) -> bool:
        elapsed = (when - start).total_seconds()
        return (elapsed % (2 * PUMP_CYCLE_S)) < PUMP_CYCLE_S

    series[PUMP_RUN] = [(t, 1.0 if running(t) else 0.0)
                        for t, _ in series[PUMP_RUN]]

    rebuilt = []
    for t, value in series[PUMP_FLOW]:
        noise = value - round(value)          # keep the generator's jitter
        if not running(t):
            rebuilt.append((t, max(0.0, abs(noise))))
        elif not clean and fault_start <= t < fault_end:
            # The pump insists it is running; the meter says otherwise.
            rebuilt.append((t, max(0.0, abs(noise))))
        else:
            rebuilt.append((t, PUMP_RUNNING_FLOW + noise))
    series[PUMP_FLOW] = rebuilt


#: The tank's three signals, and the geometry that ties them together.
TANK_LEVEL = "Kranji1PS-Service-Reservoir-Level"
TANK_INLET = "Kranji1PS-Reservoir-Inlet-Flow"
TANK_OUTLET = "Kranji1PS-Reservoir-Outlet-Flow"
TANK_AREA_M2 = 500.0
TANK_UNDER_READ = 0.7
TANK_FAULT_START_FRAC = 0.30
TANK_FAULT_HOURS = 6.0


def couple_tank(series: dict[str, list[tuple[datetime, float]]],
                start: datetime, *, clean: bool = False) -> None:
    """
    Make the reservoir's three signals physically consistent, then break one.

    The per-sensor generator cannot do this: it builds each series
    independently, so a level, an inflow and an outflow generated separately
    have no relationship at all and mass balance would either see violations
    everywhere or refuse to fit. Here the level is INTEGRATED from the metered
    flows, so `dLevel/dt * Area == Qin - Qout` holds exactly, up to noise.

    Then the outflow meter is made to under-read by 30% for six hours. That is
    the fault worth testing, because it is invisible to every single-sensor
    detector: the level is plausible, the inflow is plausible, and the outflow
    is plausible. Only the three together are impossible, which is the entire
    reason mass balance exists.
    """
    if not all(k in series for k in (TANK_LEVEL, TANK_INLET, TANK_OUTLET)):
        return

    inlet = series[TANK_INLET]
    outlet = series[TANK_OUTLET]
    level = series[TANK_LEVEL]
    if not (inlet and outlet and level):
        return

    if not clean:
        # The outlet METER under-reads; the water itself is unaffected, which
        # is exactly why the books stop balancing.
        fault_start = start + timedelta(
            seconds=(outlet[-1][0] - start).total_seconds() * TANK_FAULT_START_FRAC)
        fault_end = fault_start + timedelta(hours=TANK_FAULT_HOURS)
        true_outlet = [(t, v) for t, v in outlet]
        series[TANK_OUTLET] = [
            (t, v * TANK_UNDER_READ if fault_start <= t < fault_end else v)
            for t, v in outlet
        ]
    else:
        true_outlet = [(t, v) for t, v in outlet]

    # Integrate the TRUE flows to get the level the water actually reaches.
    def at(points, when):
        best = points[0][1]
        for t, v in points:
            if t > when:
                break
            best = v
        return best

    # Integrate over SORTED time. `generate_series` has already injected the
    # duplicate and out-of-order timestamps the real feed contains, and
    # integrating straight over those gives dt <= 0 and a level that wanders
    # off into nonsense -- destroying the very relationship this tank exists to
    # provide. The defects stay in the emitted series; they are simply not
    # allowed to corrupt the physics.
    inlet = sorted(inlet, key=lambda row: row[0])
    true_outlet = sorted(true_outlet, key=lambda row: row[0])
    ordered = sorted(level, key=lambda row: row[0])

    height = ordered[0][1]
    previous = ordered[0][0]
    rebuilt = [(previous, height)]
    for t, _ in ordered[1:]:
        dt = (t - previous).total_seconds()
        if dt <= 0:
            rebuilt.append((t, height))
            continue
        net = at(inlet, t) - at(true_outlet, t)
        height += net * dt / TANK_AREA_M2
        rebuilt.append((t, height))
        previous = t
    series[TANK_LEVEL] = rebuilt


def _inject_timestamp_defects(points: list[tuple[datetime, float]],
                              rng: random.Random) -> None:
    """
    Add the timestamp defects the real feed contains.

    A duplicate timestamp gives dt == 0, which turns a naive dv/dt into an
    infinite-rate spike; an out-of-order row gives dt < 0. Both must be
    survived, so both are present in the fixture.
    """
    n = len(points)
    if n < 20:
        return
    i = rng.randrange(5, n - 5)
    points.insert(i + 1, (points[i][0], points[i][1] + 0.01))   # duplicate timestamp
    j = rng.randrange(5, n - 5)
    if j + 2 < len(points):
        points[j], points[j + 1] = points[j + 1], points[j]      # out of order


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #
def write_history(out: Path, fleet: list[SensorSpec],
                  series: dict[str, list[tuple[datetime, float]]],
                  start: datetime, end: datetime) -> int:
    """
    Write hourly `;`-separated HISTORY files, matching the real export.

    Filenames follow `hts_YYYY_MM_HISTORY_<2026Mar01-000000>.csv`, the pattern
    the ingest discovery regex expects.
    """
    hist_dir = out / "HISTORY"
    hist_dir.mkdir(parents=True, exist_ok=True)

    by_hour: dict[datetime, list[str]] = {}
    for spec in fleet:
        for ts, value in series[spec.description]:
            hour = ts.replace(minute=0, second=0, microsecond=0)
            by_hour.setdefault(hour, []).append(
                f"{spec.ip};{spec.row_id};{ts.strftime('%Y-%m-%d %H:%M:%S')};{value:.6f}"
            )

    hour = start.replace(minute=0, second=0, microsecond=0)
    written = 0
    while hour < end:
        rows = by_hour.get(hour, [])
        name = f"hts_{hour:%Y_%m}_HISTORY_{hour:%Y%b%d-%H%M%S}.csv"
        with open(hist_dir / name, "w", encoding="utf-8") as fh:
            fh.write("IPADDRESS;ROW_ID;DATETIME;CURRVALUE\n")
            fh.write("\n".join(rows))
            if rows:
                fh.write("\n")
        written += 1
        hour += timedelta(hours=1)
    return written


def write_histcurr(out: Path, fleet: list[SensorSpec], snapshot: datetime,
                   series: dict[str, list[tuple[datetime, float]]]) -> None:
    """
    Sensor inventory, in the format the LIVE feed actually uses.

    This was originally written as the older comma-separated 6-column file that
    the committed v1 code expects. The real hourly export the client supplied is
    neither: it is **semicolon-separated with 9 columns**, carries an extra
    POINTTYPE, orders the columns differently, and writes DATETIME as US
    12-hour `9/20/2026 9:00:00 PM`.

        committed code expects: TAGNAME,IPADDRESS,ROW_ID,DESCRIPTION,RAWTYPE,RTUNUMBER
        the live feed sends:    ROW_ID;IPADDRESS;DESCRIPTION;TAGNAME;RTUNUMBER;
                                RAWTYPE;POINTTYPE;DATETIME;CURRVALUE

    A fixture in the old shape would have let the ingest pass its tests and then
    fail on the client's first real file, so it is written in the live format
    here even though that makes it disagree with the committed v1 reader.

    DATETIME and CURRVALUE are present because this is a *current value*
    snapshot, not a pure inventory: v1's `process_histcurr` parses DATETIME and
    drops rows with any NA before selecting columns, so omitting it fails the
    whole stage with `KeyError: 'DATETIME'`.
    """
    d = out / "HISTCURR"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "histcurr_fujitsu.csv", "w", encoding="utf-8") as fh:
        fh.write("ROW_ID;IPADDRESS;DESCRIPTION;TAGNAME;RTUNUMBER;RAWTYPE;"
                 "POINTTYPE;DATETIME;CURRVALUE\n")
        for spec in fleet:
            tag = f"S606-{spec.site.upper()[:8]}-{spec.row_id}"
            points = series.get(spec.description) or []
            last_ts, last_val = points[-1] if points else (snapshot, spec.base)
            # POINTTYPE 0 is the real feed's largest and least informative
            # bucket (18% pure), so using it keeps the fixture honest about how
            # little the code may lean on that column.
            fh.write(f"{spec.row_id};{spec.ip};{spec.description};{tag};"
                     f"{spec.rtu};{spec.rawtype};0;"
                     f"{last_ts:%-m/%-d/%Y %-I:%M:%S %p};{last_val:.6f}\n")


def write_longlat(out: Path) -> None:
    """
    RTU coordinates, written to BOTH the fixture root and `processed/`.

    v1's data_preprocessing.py reads `processed/LongLat.csv`, not the root copy
    -- discovered by running the real pipeline against this fixture. Writing
    both keeps the shadow comparison able to drive v1 while das2 reads the
    root path configured in IngestConfig.

    Keyed on RTU, which is why every sensor on one RTU ends up with identical
    coordinates in the real system: spatial resolution is site level, not
    sensor level. The dirty RTU (-1) is deliberately absent, so the
    coordinate-coverage path is exercised rather than assumed.
    """
    rows = ["Longitude,Latitude,Location,LKey"]
    rows += [f"{s.lon},{s.lat},{s.name},{s.rtu}" for s in SITES if s.rtu != "-1"]
    body = "\n".join(rows) + "\n"

    (out / "LongLat.csv").write_text(body, encoding="utf-8")
    processed = out / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    (processed / "LongLat.csv").write_text(body, encoding="utf-8")


def write_histalmevt(out: Path, start: datetime, end: datetime,
                     rng: random.Random) -> int:
    """
    Hourly alarm files, for v1 compatibility during the shadow comparison only.

    SCADA alarm events are out of scope for the new detection, classification
    and alerting, so nothing in das2 reads these. They exist because the v1
    fetch stage aborts the whole run when more than 25% of the expected hourly
    files are missing from EITHER feed -- so the shadow harness cannot run v1
    against the fixture without them.
    """
    d = out / "HISTALMEVT"
    d.mkdir(parents=True, exist_ok=True)

    hour = start.replace(minute=0, second=0, microsecond=0)
    written = 0
    while hour < end:
        name = f"hts_{hour:%Y_%m}_HISTALMEVT_{hour:%Y%b%d-%H%M%S}.csv"
        with open(d / name, "w", encoding="utf-8") as fh:
            fh.write("ALARMSETTIME;ALARMTEXT\n")
            # A couple of routine events per hour; content is irrelevant here.
            for _ in range(rng.randint(1, 3)):
                site = rng.choice(SITES).name
                ts = hour + timedelta(seconds=rng.randrange(3600))
                status = rng.choice(["Normal", "Alarm", "Failed", "Open", "Close"])
                fh.write(f"{ts:%Y-%m-%d %H:%M:%S};{site}-PUMP-1 {status}\n")
        written += 1
        hour += timedelta(hours=1)
    return written


def build_manifest(fleet: list[SensorSpec],
                   series: dict[str, list[tuple[datetime, float]]],
                   start: datetime, end: datetime, seed: int, clean: bool) -> dict:
    """
    Ground truth: what was injected, where, and when.

    This is what later tests assert against, so a detector can fail rather than
    merely agree with itself.
    """
    total_s = (end - start).total_seconds()
    injections = []
    for spec in fleet:
        if not spec.fault or clean:
            continue
        f_start = start + timedelta(seconds=total_s * spec.fault_start_frac)
        injections.append({
            "sensor_key": spec.sensor_key,
            "description": spec.description,
            "equipment": spec.equipment,
            "site": spec.site,
            "fault": spec.fault,
            "start": f_start.isoformat(),
            "end": (f_start + timedelta(hours=spec.fault_duration_h)).isoformat(),
            "duration_h": spec.fault_duration_h,
            "detail": spec.fault_detail,
        })

    sensors = []
    for spec in fleet:
        pts = series[spec.description]
        site = next((s for s in SITES if s.name == spec.site), None)
        sensors.append({
            "sensor_key": spec.sensor_key,
            "description": spec.description,
            "equipment": spec.equipment,
            "site": spec.site,
            "rtu": spec.rtu,
            "signal_type": "Digital" if spec.rawtype == RAWTYPE_DIGITAL else "Analog",
            "nominal_dt_s": spec.dt_s,
            "unit": spec.unit,
            "n_points": len(pts),
            "expected_region": site.region if site else "Unknown",
            "has_coordinates": bool(site and site.rtu != "-1"),
            "is_control": spec.fault is None,
        })

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "clean": clean,
        "window": {"start": start.isoformat(), "end": end.isoformat(),
                   "hours": round(total_s / 3600.0, 2)},
        "counts": {
            "sensors": len(fleet),
            "controls": sum(1 for s in fleet if s.fault is None),
            "injections": len(injections),
            "readings": sum(len(v) for v in series.values()),
        },
        "sensors": sensors,
        "injections": injections,
    }


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="fixtures", help="output directory")
    ap.add_argument("--hours", type=int, default=72, help="window length")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (deterministic)")
    ap.add_argument("--end", default=None,
                    help="window end, ISO format (default: now, floored to the hour)")
    ap.add_argument("--clean", action="store_true",
                    help="inject nothing: the false-positive control corpus")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    end = (datetime.fromisoformat(args.end) if args.end
           else datetime.now().replace(minute=0, second=0, microsecond=0))
    start = end - timedelta(hours=args.hours)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fleet = build_fleet()
    series = {s.description: generate_series(s, start, end, rng, clean=args.clean)
              for s in fleet}
    couple_tank(series, start, clean=args.clean)
    couple_pump(series, start, end, clean=args.clean)

    n_files = write_history(out, fleet, series, start, end)
    write_histcurr(out, fleet, end, series)
    write_longlat(out)
    write_histalmevt(out, start, end, rng)

    manifest = build_manifest(fleet, series, start, end, args.seed, args.clean)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    c = manifest["counts"]
    print(f"[fixtures] {out}/")
    print(f"  window     : {start:%Y-%m-%d %H:%M} -> {end:%Y-%m-%d %H:%M} ({args.hours}h)")
    print(f"  hourly CSVs: {n_files}")
    print(f"  sensors    : {c['sensors']}  ({c['controls']} clean controls)")
    print(f"  readings   : {c['readings']:,}")
    print(f"  injected   : {c['injections']} faults"
          f"{' (CLEAN corpus: none)' if args.clean else ''}")
    if not args.clean:
        by_fault: dict[str, int] = {}
        for inj in manifest["injections"]:
            by_fault[inj["fault"]] = by_fault.get(inj["fault"], 0) + 1
        for fault, n in sorted(by_fault.items()):
            print(f"    {fault:24} x{n}")


if __name__ == "__main__":
    main()
