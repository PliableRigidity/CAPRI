"""The competition arena's DESIGN geometry: what the software already knows, so the operator never types it.

Single source: ``config/arena_geometry.yaml`` (read from the drawing ``Arena Floor Technical Drawing.pdf``,
rev. 1). The simulator's ``config/sim.yaml`` carries the same numbers; ``tests/test_arena_design.py`` pins the
two together so they cannot drift apart silently. The simulator's training config itself is deliberately not
rewired to read this file (a frozen evaluation setting), only compared with it.

What is here is the SIMULATOR FRAME: origin at the plain corner, pits centred on the corners (0, H) and
(W, 0), flipper chamfer at (W, H) with both legs ``flipper_leg_m`` long. Metres.

Measured differences ("advanced overrides") are kept separate from the design: ``Overrides`` holds only
what the operator changed, each with the reason, and ``EffectiveDesign`` records for every number whether it
comes from the design or from an override. Nothing is changed silently: the 1130 vs 1150 mm question is
answered by the design (1150) unless an inset is entered on purpose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DESIGN_PATH = Path(__file__).resolve().parents[2] / "config" / "arena_geometry.yaml"

# Plausible ranges for a MEASURED override (anything outside is refused as a typing slip).
MIN_SIDE_M, MAX_SIDE_M = 0.30, 3.00
MIN_RADIUS_M = 0.03
MIN_LEG_M = 0.05
MAX_INSET_M = 0.10
FIELDS = ("width_m", "height_m", "pit_radius_m", "flipper_leg_m", "inset_per_side_m")


class DesignError(ValueError):
    """The design file or an override is unusable; the message says what to do."""


@dataclass(frozen=True)
class ArenaDesign:
    width_m: float
    height_m: float
    pit_radius_m: float
    flipper_leg_m: float
    source_file: str = ""
    source_sha256: str = ""
    revision: str = ""
    created_by: str = ""
    created: str = ""
    wall_thickness_m: float = 0.020
    wall_height_m: float = 0.100

    def pit_centres(self) -> list[tuple[float, float]]:
        return [(0.0, self.height_m), (self.width_m, 0.0)]

    def flipper_vertices(self) -> list[tuple[float, float]]:
        w, h, leg = self.width_m, self.height_m, self.flipper_leg_m
        return [(w - leg, h), (w, h), (w, h - leg)]

    def provenance(self) -> str:
        return f"{self.source_file} rev {self.revision} ({self.created_by}, {self.created})"


def load_design(path: Path | str | None = None) -> ArenaDesign:
    """Read the canonical design. Raises ``DesignError`` for a missing or damaged file (never guesses)."""
    target = Path(path) if path else DESIGN_PATH
    try:
        data = yaml.safe_load(target.read_text(encoding="utf-8"))
        floor, pits, flipper, meta = data["floor"], data["pits"], data["flipper"], data.get("design") or {}
        sim = data.get("simulator_only") or {}
        design = ArenaDesign(float(floor["width_m"]), float(floor["height_m"]), float(pits["radius_m"]), float(flipper["leg_m"]),
                             str(meta.get("source_file", "")), str(meta.get("source_sha256", "")), str(meta.get("revision", "")),
                             str(meta.get("created_by", "")), str(meta.get("created", "")),
                             float(sim.get("wall_thickness_m", 0.020)), float(sim.get("wall_height_m", 0.100)))
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as error:
        raise DesignError(f"the arena design file {target} is missing or damaged ({type(error).__name__}: {error}); "
                          "restore config/arena_geometry.yaml from the repository") from None
    problem = _check(design.width_m, design.height_m, design.pit_radius_m, design.flipper_leg_m)
    if problem:
        raise DesignError(f"the arena design file {target} is inconsistent: {problem}")
    return design


def _check(width: float, height: float, radius: float, leg: float, inset: float = 0.0) -> str:
    for name, value in (("width", width), ("height", height), ("pit radius", radius), ("flipper leg", leg), ("inset", inset)):
        if not (isinstance(value, (int, float)) and math.isfinite(value)):
            return f"{name} is not a number"
    if inset < 0 or inset > MAX_INSET_M:
        return f"inset {inset * 1000:.0f} mm is outside 0 to {MAX_INSET_M * 1000:.0f} mm"
    width, height = width - 2 * inset, height - 2 * inset
    if not (MIN_SIDE_M <= width <= MAX_SIDE_M and MIN_SIDE_M <= height <= MAX_SIDE_M):
        return f"arena {width:.3f} x {height:.3f} m is outside the plausible {MIN_SIDE_M:g} to {MAX_SIDE_M:g} m"
    if radius < MIN_RADIUS_M or 2 * radius >= min(width, height):
        return f"pit radius {radius * 1000:.0f} mm does not fit two opposite corner pits in a {width:.3f} x {height:.3f} m arena"
    if leg < MIN_LEG_M or leg >= min(width, height) - radius:
        return f"flipper leg {leg * 1000:.0f} mm is implausible for this arena"
    return ""


@dataclass(frozen=True)
class Overrides:
    """Measured differences from the design. Only the fields the operator changed are set (others are None)."""
    width_m: float | None = None
    height_m: float | None = None
    pit_radius_m: float | None = None
    flipper_leg_m: float | None = None
    inset_per_side_m: float | None = None
    reason: str = ""

    @property
    def any(self) -> bool:
        return any(getattr(self, name) is not None for name in FIELDS)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {name: getattr(self, name) for name in FIELDS if getattr(self, name) is not None}
        if out and self.reason:
            out["reason"] = self.reason
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Overrides":
        data = data or {}
        try:
            return cls(**{name: (None if data.get(name) is None else float(data[name])) for name in FIELDS}, reason=str(data.get("reason", "")))
        except (TypeError, ValueError):
            raise DesignError("the saved advanced overrides are damaged") from None


@dataclass(frozen=True)
class EffectiveDesign:
    """The design with any overrides applied, and where each number came from."""
    design: ArenaDesign
    overrides: Overrides = field(default_factory=Overrides)

    def _value(self, name: str) -> float:
        value = getattr(self.overrides, name)
        return getattr(self.design, name) if value is None else value

    @property
    def inset_m(self) -> float:
        return 0.0 if self.overrides.inset_per_side_m is None else self.overrides.inset_per_side_m

    @property
    def width_m(self) -> float:
        return self._value("width_m") - 2 * self.inset_m

    @property
    def height_m(self) -> float:
        return self._value("height_m") - 2 * self.inset_m

    @property
    def pit_radius_m(self) -> float:
        return self._value("pit_radius_m")

    @property
    def flipper_leg_m(self) -> float:
        return self._value("flipper_leg_m")

    def problem(self) -> str:
        return _check(self._value("width_m"), self._value("height_m"), self.pit_radius_m, self.flipper_leg_m, self.inset_m)

    def with_overrides(self, overrides: Overrides) -> "EffectiveDesign":
        """A new EffectiveDesign; raises ``DesignError`` when the overrides are implausible."""
        candidate = EffectiveDesign(self.design, overrides)
        problem = candidate.problem()
        if problem:
            raise DesignError(problem)
        return candidate

    def source_of(self, name: str) -> str:
        if name in ("width_m", "height_m") and self.overrides.inset_per_side_m:
            return "design with inset override"
        return "override" if getattr(self.overrides, name) is not None else "design"

    def pit_centres(self) -> list[tuple[float, float]]:
        return [(0.0, self.height_m), (self.width_m, 0.0)]

    def lines(self) -> list[str]:
        """Plain statements for the ARENA tab: what the software will use and where it came from."""
        out = [f"Arena {self.width_m * 1000:.0f} x {self.height_m * 1000:.0f} mm ({self.source_of('width_m')}), "
               f"pit radius {self.pit_radius_m * 1000:.0f} mm ({self.source_of('pit_radius_m')}), "
               f"flipper legs {self.flipper_leg_m * 1000:.0f} mm ({self.source_of('flipper_leg_m')})",
               f"Design: {self.design.provenance()}"]
        if self.overrides.any:
            out.append("ADVANCED OVERRIDE in use: " + ", ".join(f"{k} = {v}" for k, v in self.overrides.to_dict().items() if k != "reason")
                       + (f" ({self.overrides.reason})" if self.overrides.reason else ""))
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"used": True, "source_file": self.design.source_file, "source_sha256": self.design.source_sha256,
                "revision": self.design.revision, "design": {"width_m": self.design.width_m, "height_m": self.design.height_m,
                                                             "pit_radius_m": self.design.pit_radius_m, "flipper_leg_m": self.design.flipper_leg_m},
                "overrides": self.overrides.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any], design: ArenaDesign | None = None) -> "EffectiveDesign":
        """Rebuild from a saved arena file. The CURRENT design file is used for the base numbers; when it no
        longer matches what the file recorded, the saved numbers win (a saved arena never changes by itself)."""
        base = design or load_design()
        saved = data.get("design") or {}
        try:
            snapshot = ArenaDesign(float(saved["width_m"]), float(saved["height_m"]), float(saved["pit_radius_m"]), float(saved["flipper_leg_m"]),
                                   str(data.get("source_file", base.source_file)), str(data.get("source_sha256", base.source_sha256)),
                                   str(data.get("revision", base.revision)), base.created_by, base.created)
        except (KeyError, TypeError, ValueError):
            snapshot = base
        return cls(snapshot, Overrides.from_dict(data.get("overrides")))


def design_for_profile(profile: str, path: Path | str | None = None) -> EffectiveDesign | None:
    """The design applies to the real competition arena (LIVE) only. HOME TEST is a patch of any size."""
    from control_center.tools.overlay_core import LIVE_PROFILE
    return EffectiveDesign(load_design(path)) if profile == LIVE_PROFILE else None
