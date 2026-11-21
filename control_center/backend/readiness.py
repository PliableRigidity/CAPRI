"""One vocabulary for "can I rely on this yet?" across the Command Centre.

Every screen that is a step of the competition-day workflow answers ``readiness()`` with one of these.
The guided setup page only displays them; it never decides them, so a step cannot be marked Ready by
anything except the code that actually checked it.

  NOT_CONFIGURED  nothing has been set up yet, or the needed input does not exist
  READY           checked, and nothing to warn about
  WARNING         usable, but something is weak, stale, unmeasured or still a placeholder
  FAILED          checked and not usable; ``fix`` says what to do
  BLOCKED         cannot be used (or attempted) until a prerequisite is met: e.g. autonomous drive before the
                  physical safety checks have passed. Never a software fault: ``detail`` names the missing prerequisite
"""

from __future__ import annotations

from dataclasses import dataclass

NOT_CONFIGURED, READY, WARNING, FAILED, BLOCKED = "Not configured", "Ready", "Warning", "Failed", "Blocked"
LEVELS = (NOT_CONFIGURED, READY, WARNING, FAILED, BLOCKED)

# Worst first, for combining several checks into one.
_SEVERITY = {FAILED: 4, BLOCKED: 3, NOT_CONFIGURED: 2, WARNING: 1, READY: 0}


@dataclass(frozen=True)
class Readiness:
    level: str
    headline: str
    detail: str = ""
    fix: str = ""            # what the operator should do next, in plain words

    def __post_init__(self) -> None:
        if self.level not in LEVELS:
            raise ValueError(f"unknown readiness level {self.level!r}; use one of {LEVELS}")

    @property
    def ok(self) -> bool:
        return self.level == READY

    @property
    def usable(self) -> bool:
        """Ready or Warning: the step may be relied on, with the warning in view."""
        return self.level in (READY, WARNING)

    def to_dict(self) -> dict[str, str]:
        return {"level": self.level, "headline": self.headline, "detail": self.detail, "fix": self.fix}


def worst(items: list[Readiness]) -> str:
    """The most serious level among ``items`` (READY for an empty list)."""
    return max((item.level for item in items), key=_SEVERITY.__getitem__, default=READY)


# Colours used by every chip, so the same word always looks the same.
CHIP_COLOURS = {NOT_CONFIGURED: "#7b8794", READY: "#3fa66b", WARNING: "#d9a441", FAILED: "#d34e59", BLOCKED: "#9b6fd6"}
