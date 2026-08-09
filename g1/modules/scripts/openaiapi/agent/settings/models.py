"""Typed settings (spec section 14, 15, 16, 17).

Generalizes ``scene_executor.py``'s ad hoc ``ctx.toggles = {"voice":
True, "nav": True, "ll": True}`` dict into the spec's full, typed
namespace, while keeping the same spirit: a small set of booleans that
gate what the runtime is willing to do, checked deterministically, never
left to the model's discretion.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Union

from pydantic import BaseModel, ConfigDict, Field


class AudioSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_enabled: bool = True
    asr_enabled: bool = True
    audio_to_state_enabled: bool = True


class AnnouncementSettings(BaseModel):
    """Speech and gesture announcements default OFF.

    The cognitive agent still handles every intent (conversation,
    query_capability, move_arm, ...) regardless of these two toggles --
    they only gate whether a decision is *also* spoken out loud / played
    as a high-level arm gesture before/around execution (spec section 16).
    Defaulting both off means a freshly deployed agent starts silent and
    still-armed-but-not-gesturing until an operator opts in via
    ``/settings set announcements.audio_enabled true`` and/or
    ``/settings set announcements.gesture_enabled true``.
    """

    model_config = ConfigDict(extra="forbid")

    audio_enabled: bool = False
    gesture_enabled: bool = False
    announce_intent_before_action: bool = True
    announce_denials: bool = True


class MotionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allow_arm_motion: bool = True
    allow_arm_sdk: bool = True
    allow_low_cmd: bool = False


class SkillMode(str, Enum):
    """Per-skill execution policy (distinct from capability/physical gating).

    ``agent/capabilities.py`` decides whether a skill is physically and
    permissibly *possible*; this decides whether a human must explicitly
    approve *this particular invocation* before it runs, generalizing
    ``llm_client/cli.py``'s ``--confirm-tools`` y/N prompt and
    ``scene_executor.py``'s per-step "Press Enter to run" gate into a
    persisted, per-skill setting instead of an all-or-nothing CLI flag.
    """

    AUTO = "auto"
    CONFIRM = "confirm"
    DISABLED = "disabled"


class SkillsSettings(BaseModel):
    """Default execution mode plus per-skill-name overrides.

    A capability decision that itself demands approval (e.g. a /low_cmd
    fallback, spec section 18) can never be weakened to ``auto`` by this
    setting -- see ``agent/skills.py: resolve_and_maybe_invoke``'s
    "approval floor" logic. Setting a skill to ``auto`` only removes the
    *convenience* confirmation prompt; it never bypasses a safety-driven
    approval requirement.
    """

    model_config = ConfigDict(extra="forbid")

    default_mode: SkillMode = SkillMode.CONFIRM
    overrides: Dict[str, SkillMode] = Field(default_factory=dict)

    def mode_for(self, skill_name: str) -> SkillMode:
        return self.overrides.get(skill_name, self.default_mode)


class AgentSettings(BaseModel):
    """The full, typed settings tree exposed under ``/settings``."""

    model_config = ConfigDict(extra="forbid")

    audio: AudioSettings = Field(default_factory=AudioSettings)
    announcements: AnnouncementSettings = Field(default_factory=AnnouncementSettings)
    motion: MotionSettings = Field(default_factory=MotionSettings)
    skills: SkillsSettings = Field(default_factory=SkillsSettings)

    def get_skill_mode(self, skill_name: str) -> SkillMode:
        return self.skills.mode_for(skill_name)

    def set_skill_mode(self, skill_name: str, mode: Union[SkillMode, str]) -> None:
        """Set a per-skill override (``/settings skill <name> <auto|confirm|disabled>``).

        Kept as a dedicated method rather than forced through
        ``set_path``'s two-level ``<section>.<field>`` contract, since a
        skill name is a dynamic dict key, not a fixed model field.
        """
        resolved = SkillMode(mode)
        overrides = dict(self.skills.overrides)
        overrides[skill_name] = resolved
        self.skills = SkillsSettings.model_validate(
            {"default_mode": self.skills.default_mode, "overrides": overrides}
        )

    def clear_skill_mode(self, skill_name: str) -> None:
        """Remove a per-skill override, falling back to ``skills.default_mode``."""
        overrides = dict(self.skills.overrides)
        overrides.pop(skill_name, None)
        self.skills = SkillsSettings.model_validate(
            {"default_mode": self.skills.default_mode, "overrides": overrides}
        )

    def get_path(self, dotted_key: str) -> Any:
        node: Any = self
        for part in dotted_key.split("."):
            if not hasattr(node, part):
                raise KeyError(dotted_key)
            node = getattr(node, part)
        return node

    def set_path(self, dotted_key: str, value: Any) -> None:
        """Set ``dotted_key`` (e.g. ``motion.allow_arm_motion``) to ``value``.

        Rebuilds the leaf sub-model via ``model_validate`` (not
        ``model_copy(update=...)``, which skips validation) so the leaf's
        own pydantic validation -- type coercion (e.g. the CLI's string
        ``"false"`` into a real ``bool``), and ``extra="forbid"`` -- runs on
        the new value. An invalid value raises and nothing is left
        partially applied.
        """
        parts = dotted_key.split(".")
        if len(parts) != 2:
            raise KeyError(f"expected '<section>.<field>', got {dotted_key!r}")
        section_name, field_name = parts
        if not hasattr(self, section_name):
            raise KeyError(f"unknown settings section {section_name!r}")
        section = getattr(self, section_name)
        section_cls = type(section)
        if field_name not in section_cls.model_fields:
            raise KeyError(f"unknown setting {dotted_key!r}")
        data = section.model_dump()
        data[field_name] = value
        updated = section_cls.model_validate(data)
        setattr(self, section_name, updated)

    def as_flat_dict(self) -> Dict[str, Any]:
        flat: Dict[str, Any] = {}
        for section_name in type(self).model_fields:
            section = getattr(self, section_name)
            for field_name in type(section).model_fields:
                flat[f"{section_name}.{field_name}"] = getattr(section, field_name)
        return flat
