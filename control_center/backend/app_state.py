"""What the Command Centre remembers between runs: the profile in use, the last tab, per-profile progress.

One small JSON file (``config/command_centre_state.json``). Progress is stored under the profile it belongs
to and read back only for that profile, so HOME TEST progress never shows up in LIVE. A file that is missing
or damaged is replaced by defaults (and the damage is reported), never by a crash. With no path the state
lives in memory only (mock mode and tests), so nothing is written.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)
STATE_VERSION = 1
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "command_centre_state.json"
# The competition-day sequence. (Older files used "control" and "monitoring"; they are read as "safety" and "operate".)
STEP_IDS = ("camera", "arena", "tracking", "opponent", "connections", "safety", "operate")
LEGACY_STEPS = {"control": "safety", "monitoring": "operate"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AppState:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else None
        self.data: dict[str, Any] = {"version": STATE_VERSION, "profile": None, "last_tab": None, "camera": {}, "profiles": {}}
        self.load_problem = ""
        self.load()

    # ---- file
    def load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or int(data.get("version", 0)) != STATE_VERSION:
                raise ValueError("unknown format or version")
            for key in ("camera", "profiles"):
                if not isinstance(data.get(key), dict):
                    raise ValueError(f"'{key}' is not an object")
            self.data = data
        except (OSError, ValueError) as error:
            self.load_problem = f"saved Command Centre state ignored ({error}); starting from defaults"
            log.warning(self.load_problem)

    def save(self) -> bool:
        if self.path is None:
            return False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_name(self.path.name + ".tmp")
            temp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
            os.replace(temp, self.path)
            return True
        except OSError as error:
            log.error("could not save Command Centre state: %s", error)
            return False

    # ---- global
    @property
    def profile(self) -> str | None:
        return self.data.get("profile")

    def set_profile(self, name: str) -> None:
        self.data["profile"] = name
        self.save()

    @property
    def last_tab(self) -> str | None:
        return self.data.get("last_tab")

    def set_last_tab(self, tab: str) -> None:
        if self.data.get("last_tab") != tab:
            self.data["last_tab"] = tab
            self.save()

    def camera_settings(self) -> dict[str, Any]:
        return dict(self.data.get("camera") or {})

    def set_camera_settings(self, settings: dict[str, Any]) -> None:
        self.data["camera"] = dict(settings)
        self.save()

    # ---- per profile
    def _profile(self, name: str) -> dict[str, Any]:
        return self.data["profiles"].setdefault(name, {"wizard_step": STEP_IDS[0], "steps": {}, "arena_verified": None})

    def wizard_step(self, profile: str) -> str:
        step = self._profile(profile).get("wizard_step", STEP_IDS[0])
        step = LEGACY_STEPS.get(step, step)
        return step if step in STEP_IDS else STEP_IDS[0]

    def set_wizard_step(self, profile: str, step: str) -> None:
        if step not in STEP_IDS:
            raise ValueError(f"unknown step {step!r}")
        self._profile(profile)["wizard_step"] = step
        self.save()

    def record_step(self, profile: str, step: str, level: str, headline: str) -> None:
        """The last readiness seen for a step: for the guided page to show on the next start (clearly as 'last time')."""
        entry = {"level": level, "headline": headline, "at": _now()}
        steps = self._profile(profile)["steps"]
        if steps.get(step, {}).get("level") != level or steps.get(step, {}).get("headline") != headline:
            steps[step] = entry
            self.save()

    def last_step(self, profile: str, step: str) -> dict[str, str] | None:
        return self._profile(profile)["steps"].get(step)

    def arena_verified(self, profile: str) -> dict[str, Any] | None:
        return self._profile(profile).get("arena_verified")

    def set_arena_verified(self, profile: str, saved_at: str, clicks_saved_at: str) -> None:
        self._profile(profile)["arena_verified"] = {"arena_saved_at": saved_at, "clicks_saved_at": clicks_saved_at, "at": _now()}
        self.save()

    def arena_draft(self, profile: str) -> dict[str, Any] | None:
        return self._profile(profile).get("arena_draft")

    def set_arena_draft(self, profile: str, draft: dict[str, Any] | None) -> None:
        """An unfinished arena set-up, so a restart does not lose the clicks (None clears it)."""
        if self._profile(profile).get("arena_draft") != draft:
            self._profile(profile)["arena_draft"] = draft
            self.save()

    def clear_arena_verified(self, profile: str) -> None:
        if self._profile(profile).get("arena_verified") is not None:
            self._profile(profile)["arena_verified"] = None
            self.save()
