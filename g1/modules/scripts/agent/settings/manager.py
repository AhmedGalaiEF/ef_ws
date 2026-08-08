"""Persistent settings store with session-only overrides (spec section 14).

``/settings set <key> <value>`` persists by default; a future
``/settings set --session <key> <value>`` (or programmatic
``persist=False``) keeps a change for this process only, per spec
section 14's "support session-only versus persistent settings".
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .models import AgentSettings, SkillMode

DEFAULT_SETTINGS_PATH = Path(
    os.environ.get("G1_AGENT_SETTINGS", "~/.g1_agent/settings.json")
).expanduser()


class InvalidSettingError(ValueError):
    pass


def _atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


class SettingsManager:
    def __init__(self, path: Path | str = DEFAULT_SETTINGS_PATH) -> None:
        self.path = Path(path).expanduser()
        self._persistent = self._load()
        self._session_overrides: dict[str, Any] = {}
        self._session_skill_mode_overrides: dict[str, SkillMode] = {}

    def _load(self) -> AgentSettings:
        if not self.path.exists():
            return AgentSettings()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return AgentSettings.model_validate(raw)
        except Exception:
            # Corrupt/unreadable settings fall back to defaults rather than
            # crashing startup -- consistent with checkpoint.py's stance.
            return AgentSettings()

    def _save(self) -> None:
        _atomic_write_text(self.path, self._persistent.model_dump_json(indent=2))

    def effective(self) -> AgentSettings:
        """Persistent settings with session-only overrides layered on top."""
        merged = self._persistent.model_copy(deep=True)
        for dotted_key, value in self._session_overrides.items():
            merged.set_path(dotted_key, value)
        for skill_name, mode in self._session_skill_mode_overrides.items():
            merged.set_skill_mode(skill_name, mode)
        return merged

    def get(self, dotted_key: str) -> Any:
        try:
            return self.effective().get_path(dotted_key)
        except KeyError as exc:
            raise InvalidSettingError(str(exc)) from exc

    def set(self, dotted_key: str, value: Any, *, persist: bool = True) -> None:
        # Validate on a scratch copy first so a bad value never touches
        # persistent or session state.
        probe = self._persistent.model_copy(deep=True)
        try:
            probe.set_path(dotted_key, value)
        except Exception as exc:
            raise InvalidSettingError(f"invalid value for {dotted_key!r}: {exc}") from exc

        if persist:
            self._persistent.set_path(dotted_key, value)
            self._session_overrides.pop(dotted_key, None)
            self._save()
        else:
            self._session_overrides[dotted_key] = value

    def as_flat_dict(self) -> dict[str, Any]:
        return self.effective().as_flat_dict()

    # -- per-skill execution mode (spec: "auto" / "confirm" / "disabled") --

    def get_skill_mode(self, skill_name: str) -> SkillMode:
        return self.effective().get_skill_mode(skill_name)

    def set_skill_mode(self, skill_name: str, mode: SkillMode | str, *, persist: bool = True) -> None:
        resolved = SkillMode(mode)  # raises ValueError on an invalid mode string
        if persist:
            self._persistent.set_skill_mode(skill_name, resolved)
            self._session_skill_mode_overrides.pop(skill_name, None)
            self._save()
        else:
            self._session_skill_mode_overrides[skill_name] = resolved
