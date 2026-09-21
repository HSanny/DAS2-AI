"""
das2.io.classify
================

Map a sensor Description to an equipment class, driven by an ordered rule table.

The problem being fixed
-----------------------
v1's classifier was a six-keyword if/elif chain::

    if 'dissolved' in desc: return 'Dissolved Oxygen'
    elif 'cond' in desc:    return 'Conductivity'
    ...
    return 'Others'

Measured against the real inventory (`docker_ready/processed/dim.csv`, 13,567
sensors), that leaves **3,847 of 5,204 analog sensors -- 74% -- in 'Others'**,
which the detector then drops before analysis via SKIP_UNCATEGORIZED_EQUIPMENT.
The bottleneck on coverage was this mapping, not the detectors. Among the
discarded: 98 vibration sensors (the classic pump-failure precursor) and the
rain gauges that make weather-aware triage possible without any external API.

Three things this module does that the chain could not
------------------------------------------------------
1. **Ordering is explicit and meaningful.** First match wins, and the rule file
   documents which orderings are load-bearing -- notably that setpoints must be
   matched before every measurement class.

2. **Configuration is distinguished from measurement.** 86 analog points are
   alarm setpoints and 8 are simulation values. Classifying
   "...-Current-low-alarm-setpoint" as Current would anomaly-detect a number
   that only changes when an engineer edits it: a guaranteed false page. These
   get `kind: config` and never alert.

3. **Counters are distinguished from process values.** kWh meters and run-hour
   counters only climb. A flat counter means the plant is idle (normal) and a
   drop is a rollover (not a fault), so flatline and spike detection over them
   produce pure noise.

Coverage is a tracked metric, not a silent default: `coverage_report` counts
what landed in UNCLASSIFIED so it can be printed every run and driven down over
time, rather than disappearing into an 'Others' bucket nobody looks at.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

DEFAULT_RULES_PATH = Path(__file__).resolve().parent.parent / "data" / "equipment_rules.yaml"

#: RawType values the Fujitsu export uses for analog points. Everything else is
#: digital. Carried over verbatim from v1's process_histcurr.
ANALOG_RAWTYPES = {1, 5}

UNCLASSIFIED = "UNCLASSIFIED"


@dataclass(frozen=True)
class EquipmentClass:
    """Per-class metadata: how to treat sensors of this kind."""

    name: str
    kind: str = "measurement"      # measurement | counter | config | status | unknown
    unit: str = ""
    alertable: bool = False
    range_min: float | None = None
    range_max: float | None = None

    @property
    def is_measurement(self) -> bool:
        return self.kind == "measurement"

    @property
    def is_counter(self) -> bool:
        """Monotonic. Flatline/spike detection over these is meaningless."""
        return self.kind == "counter"

    @property
    def is_config(self) -> bool:
        """A setpoint or simulation value: changes by human action, never a fault."""
        return self.kind == "config"

    @property
    def has_range(self) -> bool:
        return self.range_min is not None and self.range_max is not None


@dataclass(frozen=True)
class Classification:
    equipment: str
    signal_type: str               # Analog | Digital
    meta: EquipmentClass
    matched_pattern: str | None = None

    @property
    def is_classified(self) -> bool:
        return self.equipment != UNCLASSIFIED

    @property
    def alertable(self) -> bool:
        return self.meta.alertable

    @property
    def analysable(self) -> bool:
        """
        Whether the detector stack should look at this at all.

        Config points are excluded outright: a setpoint's value is an operator
        decision, so there is nothing for an anomaly detector to say about it.
        Everything else is analysed even when not alertable, so it appears on
        the dashboard and accrues evidence before being promoted.
        """
        return not self.meta.is_config and self.is_classified


class EquipmentClassifier:
    """Ordered-rule classifier. Load once, reuse; patterns are pre-compiled."""

    def __init__(self, rules: list[dict[str, Any]],
                 classes: dict[str, EquipmentClass],
                 analog_fallback: str = UNCLASSIFIED,
                 digital_fallback: str = "DigitalStatus"):
        # (equipment, compiled pattern, raw pattern, allowed signal types|None)
        self._rules: list[tuple[str, re.Pattern[str], str, frozenset[str] | None]] = []
        for rule in rules:
            equipment = rule["equipment"]
            # A rule may restrict itself to Analog or Digital points. SCADA
            # naming is overloaded: "DO" is Dissolved Oxygen on an analog point
            # and Digital Output on a digital one, so without this guard three
            # dozen output commands get filed as water-quality measurements.
            allowed = rule.get("signal_types")
            allowed_set = frozenset(allowed) if allowed else None
            for pattern in rule.get("patterns", []):
                self._rules.append(
                    (equipment, re.compile(pattern, re.IGNORECASE), pattern, allowed_set))
        self._classes = classes
        self._analog_fallback = analog_fallback
        self._digital_fallback = digital_fallback

    # -- construction ----------------------------------------------------- #
    @classmethod
    def load(cls, path: str | Path | None = None) -> "EquipmentClassifier":
        path = Path(path) if path else DEFAULT_RULES_PATH
        data = _read_rule_file(path)

        classes: dict[str, EquipmentClass] = {}
        for name, meta in (data.get("classes") or {}).items():
            meta = meta or {}
            rng = meta.get("range") or [None, None]
            classes[name] = EquipmentClass(
                name=name,
                kind=meta.get("kind", "measurement"),
                unit=meta.get("unit", "") or "",
                alertable=bool(meta.get("alertable", False)),
                range_min=rng[0],
                range_max=rng[1],
            )
        classes.setdefault(UNCLASSIFIED, EquipmentClass(UNCLASSIFIED, kind="unknown"))

        defaults = data.get("defaults") or {}
        return cls(
            rules=data.get("rules") or [],
            classes=classes,
            analog_fallback=defaults.get("analog_fallback", UNCLASSIFIED),
            digital_fallback=defaults.get("digital_fallback", "DigitalStatus"),
        )

    # -- classification ---------------------------------------------------- #
    def classify(self, description: str, rawtype: int | str | None = None,
                 signal_type: str | None = None) -> Classification:
        """
        Classify one sensor. First matching rule wins, so order is significant.

        `signal_type` may be given directly, or derived from the Fujitsu
        RawType (1 and 5 are analog) exactly as v1 did.
        """
        if signal_type is None:
            signal_type = "Analog" if _as_int(rawtype) in ANALOG_RAWTYPES else "Digital"

        desc = (description or "").strip()
        if desc:
            for equipment, pattern, raw, allowed in self._rules:
                if allowed is not None and signal_type not in allowed:
                    continue
                if pattern.search(desc):
                    return Classification(equipment, signal_type,
                                          self._class(equipment), raw)

        fallback = (self._analog_fallback if signal_type == "Analog"
                    else self._digital_fallback)
        return Classification(fallback, signal_type, self._class(fallback), None)

    def _class(self, name: str) -> EquipmentClass:
        return self._classes.get(name) or EquipmentClass(name, kind="unknown")

    @property
    def classes(self) -> dict[str, EquipmentClass]:
        return dict(self._classes)

    # -- reporting --------------------------------------------------------- #
    def coverage_report(self, sensors: Iterable[tuple[str, Any]]) -> dict[str, Any]:
        """
        Classify a whole inventory and summarise coverage.

        Intended to be logged every run. v1's equivalent number was invisible:
        74% of analog sensors fell into 'Others' and were dropped without
        anything saying so.

        `sensors` yields (description, rawtype) pairs.
        """
        by_equipment: dict[str, int] = {}
        analog = analog_classified = 0
        total = 0
        unclassified_examples: list[str] = []

        for description, rawtype in sensors:
            total += 1
            result = self.classify(description, rawtype)
            by_equipment[result.equipment] = by_equipment.get(result.equipment, 0) + 1
            if result.signal_type == "Analog":
                analog += 1
                if result.is_classified:
                    analog_classified += 1
                elif len(unclassified_examples) < 10:
                    unclassified_examples.append(description)

        analysable = sum(
            n for name, n in by_equipment.items()
            if not self._class(name).is_config and name != UNCLASSIFIED
        )
        alertable = sum(n for name, n in by_equipment.items()
                        if self._class(name).alertable)

        return {
            "total": total,
            "analog": analog,
            "analog_classified": analog_classified,
            "analog_coverage_pct": round(100.0 * analog_classified / analog, 1) if analog else 0.0,
            "analysable": analysable,
            "alertable": alertable,
            "unclassified": by_equipment.get(UNCLASSIFIED, 0),
            "by_equipment": dict(sorted(by_equipment.items(),
                                        key=lambda kv: -kv[1])),
            "unclassified_examples": unclassified_examples,
        }


def _as_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return -1


def _read_rule_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"equipment rule file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:      # pragma: no cover - depends on env
            raise RuntimeError(
                f"{path} is YAML but PyYAML is not installed. "
                f"Install pyyaml, or supply a .json rule file instead."
            ) from exc
        return yaml.safe_load(text) or {}
    return json.loads(text)


@lru_cache(maxsize=4)
def get_classifier(path: str | None = None) -> EquipmentClassifier:
    """Shared classifier instance; rule compilation happens once."""
    return EquipmentClassifier.load(path)
