from __future__ import annotations

import sys
from pathlib import Path

import pytest

pydantic = pytest.importorskip("pydantic")
if not pydantic.VERSION.startswith("2"):
    pytest.skip(f"agent package requires pydantic>=2 (found {pydantic.VERSION})", allow_module_level=True)

AGENT_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "g1" / "modules" / "scripts"
if str(AGENT_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_SCRIPTS_DIR))

from agent.settings.manager import InvalidSettingError, SettingsManager  # noqa: E402
from agent.settings.models import AgentSettings, SkillMode  # noqa: E402


def test_voice_and_gesture_announcements_default_off() -> None:
    """Voice responses and HL hand gestures must be off by default; the agent still
    handles every intent regardless -- these two toggles only gate the accompaniment."""
    settings = AgentSettings()
    assert settings.announcements.audio_enabled is False
    assert settings.announcements.gesture_enabled is False


def test_default_skill_mode_is_confirm() -> None:
    settings = AgentSettings()
    assert settings.get_skill_mode("reach_forward") == SkillMode.CONFIRM


def test_set_path_coerces_cli_string_to_bool() -> None:
    settings = AgentSettings()
    settings.set_path("motion.allow_arm_motion", "false")
    assert settings.motion.allow_arm_motion is False
    settings.set_path("motion.allow_arm_motion", "true")
    assert settings.motion.allow_arm_motion is True


def test_set_path_rejects_unknown_key() -> None:
    settings = AgentSettings()
    with pytest.raises(KeyError):
        settings.set_path("motion.does_not_exist", True)
    with pytest.raises(KeyError):
        settings.set_path("no_such_section.field", True)


def test_skill_mode_override_and_clear() -> None:
    settings = AgentSettings()
    settings.set_skill_mode("reach_forward", "auto")
    assert settings.get_skill_mode("reach_forward") == SkillMode.AUTO
    assert settings.get_skill_mode("grab") == SkillMode.CONFIRM  # unaffected

    settings.clear_skill_mode("reach_forward")
    assert settings.get_skill_mode("reach_forward") == SkillMode.CONFIRM


def test_skill_mode_rejects_invalid_value() -> None:
    settings = AgentSettings()
    with pytest.raises(ValueError):
        settings.set_skill_mode("reach_forward", "sometimes")


def test_settings_manager_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    manager = SettingsManager(path)
    manager.set("motion.allow_arm_motion", False)

    reloaded = SettingsManager(path)
    assert reloaded.get("motion.allow_arm_motion") is False


def test_settings_manager_session_only_does_not_persist(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    manager = SettingsManager(path)
    manager.set("motion.allow_arm_motion", False, persist=False)
    assert manager.get("motion.allow_arm_motion") is False

    reloaded = SettingsManager(path)
    assert reloaded.get("motion.allow_arm_motion") is True  # default, session change was never written


def test_settings_manager_rejects_invalid_value_without_side_effects(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    manager = SettingsManager(path)
    with pytest.raises(InvalidSettingError):
        manager.set("motion.allow_arm_motion", "not-a-bool-ish-value-either")
    # A rejected value must not have been written to disk.
    assert not path.exists()


def test_settings_manager_skill_mode_session_vs_persist(tmp_path: Path) -> None:
    manager = SettingsManager(tmp_path / "settings.json")
    manager.set_skill_mode("grab", "auto", persist=False)
    assert manager.get_skill_mode("grab") == SkillMode.AUTO

    reloaded = SettingsManager(tmp_path / "settings.json")
    assert reloaded.get_skill_mode("grab") == SkillMode.CONFIRM

    manager.set_skill_mode("grab", "disabled", persist=True)
    reloaded_again = SettingsManager(tmp_path / "settings.json")
    assert reloaded_again.get_skill_mode("grab") == SkillMode.DISABLED


def test_corrupt_settings_file_falls_back_to_defaults(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text("not json at all", encoding="utf-8")
    manager = SettingsManager(path)
    assert manager.get("motion.allow_arm_motion") is True
