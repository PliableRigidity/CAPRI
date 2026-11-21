"""The two working profiles, HOME TEST and LIVE, and how a profile is loaded.

A profile is one place's calibration: which overlay config describes it, which camera corner/landmark
clicks belong to it, and which physical arena (if any) was saved for it. Nothing is shared between
profiles, so a home test can never leave A4-sheet numbers in the competition arena or the other way round.

``load_profile`` never raises for an ordinary problem: it returns what it could load and a ``problem``
the screens show, so a damaged file produces an explanation and not a crash.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from control_center.tools.arena_setup import (ArenaSetup, ArenaSetupError, guard_profile_and_config, load_arena_setup,
                                              profile_arena_path, with_arena_setup)
from control_center.tools.overlay_core import (LIVE_PROFILE, REPO_ROOT, ClickSet, OverlayConfig, load_clicks, load_config,
                                               profile_points_path)

log = logging.getLogger(__name__)
HOME_TEST_PROFILE = "home_test"


@dataclass(frozen=True)
class ProfileDef:
    name: str
    label: str
    config_path: Path
    description: str


PROFILE_DEFS: dict[str, ProfileDef] = {
    LIVE_PROFILE: ProfileDef(LIVE_PROFILE, "LIVE (competition arena)", REPO_ROOT / "config" / "arena_overlay.yaml",
                             "The real arena and your competition camera position."),
    HOME_TEST_PROFILE: ProfileDef(HOME_TEST_PROFILE, "HOME TEST (A4 sheet / table patch)", REPO_ROOT / "config" / "home_test_overlay.yaml",
                                  "A test patch at home. Never used for the competition arena."),
}
PROFILE_ORDER = (HOME_TEST_PROFILE, LIVE_PROFILE)


def profile_label(name: str) -> str:
    return PROFILE_DEFS[name].label if name in PROFILE_DEFS else name


@dataclass
class LoadedProfile:
    name: str
    config: OverlayConfig
    clicks: ClickSet | None = None
    arena: ArenaSetup | None = None            # the saved physical arena, when this profile has one
    reason: str = ""                           # why clicks is None ("" when it is loaded)
    problem: str = ""                          # something is wrong with the saved files (shown as FAILED)
    arena_path: Path | None = None
    clicks_path: Path | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def arena_saved(self) -> bool:
        return self.arena is not None

    @property
    def corners_saved(self) -> bool:
        from control_center.tools.arena_setup import CORNER_PREFIX
        return self.clicks is not None and any(name.startswith(CORNER_PREFIX) for name in self.clicks.points)


def load_profile(name: str, *, config_path: Path | str | None = None, clicks_path: Path | str | None = None,
                 arena_path: Path | str | None = None) -> LoadedProfile:
    """Everything the Command Centre needs for profile ``name``, from that profile's own files only."""
    if name not in PROFILE_DEFS and config_path is None:
        raise ValueError(f"unknown profile {name!r}; known: {', '.join(PROFILE_DEFS)}")
    definition = PROFILE_DEFS.get(name)
    config = load_config(config_path or definition.config_path)
    clicks_file = Path(clicks_path) if clicks_path else profile_points_path(name)
    arena_file = Path(arena_path) if arena_path else profile_arena_path(name)
    result = LoadedProfile(name, config, arena_path=arena_file, clicks_path=clicks_file)
    try:
        guard_profile_and_config(name, config)
    except ArenaSetupError as error:
        result.problem = str(error)
        return result
    if not clicks_file.exists():
        result.reason = f"no calibration saved yet for profile '{name}'"
    else:
        try:
            loaded = load_clicks(clicks_file)
        except (OSError, ValueError, KeyError) as error:
            result.reason = f"calibration file unreadable: {error}"
            result.problem = result.reason
            loaded = None
        if loaded is not None and loaded.historical:
            result.reason = "the saved points are a HISTORICAL footage profile and are not used for a live camera"
            loaded = None
        result.clicks = loaded
    try:
        result.arena = load_arena_setup(name, arena_file)
        result.config = with_arena_setup(config, name, result.clicks, arena_file)
    except ArenaSetupError as error:
        # An inconsistent pair must not drive the overlay: keep the config as it was and show the reason.
        result.problem = str(error)
        result.clicks = None if "different arena size" in str(error) else result.clicks
        if result.clicks is None and not result.reason:
            result.reason = str(error)
    return result
