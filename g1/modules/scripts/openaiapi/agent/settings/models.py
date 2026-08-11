"""Typed settings (spec section 14, 15, 16, 17).

Generalizes ``scene_executor.py``'s ad hoc ``ctx.toggles = {"voice":
True, "nav": True, "ll": True}`` dict into the spec's full, typed
namespace, while keeping the same spirit: a small set of booleans that
gate what the runtime is willing to do, checked deterministically, never
left to the model's discretion.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, Union

from pydantic import BaseModel, ConfigDict, Field


class AudioSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_enabled: bool = True
    asr_enabled: bool = True
    audio_to_state_enabled: bool = True


class AsrSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    confidence_threshold: float = Field(default=0.72, ge=0.0, le=1.0)
    partial_display: bool = True
    silence_timeout_ms: int = Field(default=1200, ge=100, le=10000)
    wake_word_enabled: bool = False


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
    tts_language: str = ""
    tts_voice_model: str = ""
    tts_speaker: int = -1
    announce_intent_before_action: bool = True
    announce_denials: bool = True


class MotionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allow_arm_motion: bool = True
    allow_arm_sdk: bool = True
    allow_low_cmd: bool = False
    allow_locomotion_mode_change: bool = True


class ExpressiveMotionKindSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    cooldown_s: float = Field(default=5.0, ge=0.0, le=3600.0)


class ExpressiveMotionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    thinking: ExpressiveMotionKindSettings = Field(
        default_factory=lambda: ExpressiveMotionKindSettings(cooldown_s=8.0)
    )
    explain: ExpressiveMotionKindSettings = Field(
        default_factory=lambda: ExpressiveMotionKindSettings(cooldown_s=5.0)
    )
    thanking: ExpressiveMotionKindSettings = Field(
        default_factory=lambda: ExpressiveMotionKindSettings(cooldown_s=5.0)
    )
    explain_minimum_speech_chars: int = Field(default=80, ge=1, le=10000)
    motion_directory: str = str(Path("~/EF/ef_ws/g1/saved_motions/csv").expanduser())


class LlctlSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    allow_joint_control: bool = True
    allow_ik_control: bool = True
    require_explicit_enable_each_session: bool = True
    session_timeout_s: float = Field(default=60.0, ge=5.0, le=3600.0)


class CommandLanguage(str, Enum):
    EN = "en"
    DE = "de"
    BOTH = "both"


class ReplyLanguage(str, Enum):
    AUTO = "auto"
    EN = "en"
    DE = "de"


class InterfaceSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command_language: CommandLanguage = CommandLanguage.EN
    reply_language: ReplyLanguage = ReplyLanguage.AUTO


class VisionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rgbd_enabled: bool = False
    rgbd_host: str = "0.0.0.0"
    rgbd_port: int = 5555
    rgbd_topic: str = ""
    rgbd_timeout_s: float = 2.0
    openai_model: str = "gpt-4o-mini"


class ResponseSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_chars: int = 700
    memory_max_entries: int = 3


class CognitionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    periodic_enabled: bool = True
    periodic_interval_s: float = Field(default=10.0, ge=1.0, le=3600.0)
    attention_enabled: bool = True
    background_enabled: bool = True
    cli_priority: int = Field(default=1, ge=0, le=6)


class LearningSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    empirical_memory_enabled: bool = True
    procedural_learning_enabled: bool = True
    automatic_level_max: int = Field(default=1, ge=0, le=5)
    minimum_support_for_candidate: int = Field(default=3, ge=1, le=1000)
    minimum_support_for_procedure: int = Field(default=10, ge=1, le=5000)


class ActiveLearningSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    allow_autonomous_questions: bool = True
    cooldown_s: float = Field(default=120.0, ge=0.0, le=86400.0)
    minimum_confidence_gap: float = Field(default=0.20, ge=0.0, le=1.0)
    maximum_pending_questions: int = Field(default=1, ge=0, le=10)
    allow_during_active_scenario: bool = False
    allow_during_idle: bool = True
    allow_during_task_execution: bool = False
    duplicate_suppression_s: float = Field(default=3600.0, ge=0.0, le=604800.0)
    unanswered_timeout_s: float = Field(default=300.0, ge=1.0, le=86400.0)
    store_rejected_questions: bool = False


class RgbColorSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    r: int = Field(default=55, ge=0, le=255)
    g: int = Field(default=20, ge=0, le=255)
    b: int = Field(default=75, ge=0, le=255)


class HeadlightThinkingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    color: RgbColorSettings = Field(default_factory=RgbColorSettings)
    intensity: int = Field(default=100, ge=0, le=100)


class HeadlightSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cognitive_indicators_enabled: bool = True
    thinking: HeadlightThinkingSettings = Field(default_factory=HeadlightThinkingSettings)
    restore_previous_state: bool = True


class MemorySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    working_event_max: int = Field(default=1000, ge=10, le=100000)
    hot_episode_limit: int = Field(default=5000, ge=10, le=1000000)
    hot_disk_limit_mb: int = Field(default=500, ge=1, le=100000)
    cold_archive_limit_gb: int = Field(default=10, ge=1, le=1000)
    routine_retention_days: int = Field(default=14, ge=1, le=3650)
    significant_retention_days: int = Field(default=180, ge=1, le=3650)
    consolidation_interval_min: int = Field(default=60, ge=1, le=10080)


class MonitorSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    refresh_hz: float = Field(default=3.0, ge=0.5, le=10.0)
    event_buffer_size: int = Field(default=1000, ge=10, le=100000)


class DiskSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    warning_free_pct: float = Field(default=20.0, ge=1.0, le=95.0)
    critical_free_pct: float = Field(default=10.0, ge=1.0, le=90.0)


class ResetSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    require_confirmation: bool = True
    create_backup: bool = True
    full_preserve_settings: bool = True
    preserve_audit_log: bool = True


class TacitSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ui_enabled: bool = True
    show_confidence: bool = True
    show_evidence_counts: bool = True
    show_performance_metrics: bool = True


class ToolCategoryToggle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True


class ToolsSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    max_calls_per_turn: int = Field(default=12, ge=0, le=100)
    max_parallel_read_calls: int = Field(default=4, ge=1, le=32)
    observation: ToolCategoryToggle = Field(default_factory=ToolCategoryToggle)
    memory: ToolCategoryToggle = Field(default_factory=ToolCategoryToggle)
    knowledge: ToolCategoryToggle = Field(default_factory=ToolCategoryToggle)
    diagnostics: ToolCategoryToggle = Field(default_factory=ToolCategoryToggle)
    actions: ToolCategoryToggle = Field(default_factory=ToolCategoryToggle)


class McpReconnectSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    initial_backoff_s: float = Field(default=1.0, ge=0.1, le=300.0)
    max_backoff_s: float = Field(default=30.0, ge=1.0, le=3600.0)


class McpServerSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True


class McpServersSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory: McpServerSettings = Field(default_factory=McpServerSettings)
    documentation: McpServerSettings = Field(default_factory=McpServerSettings)
    diagnostics: McpServerSettings = Field(default_factory=McpServerSettings)


class McpSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    reconnect: McpReconnectSettings = Field(default_factory=McpReconnectSettings)
    servers: McpServersSettings = Field(default_factory=McpServersSettings)


class SelfModelSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    robot_id: str = ""
    update_from_skill_outcomes: bool = True
    reset_learned_components_on_reset_learned: bool = True


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
    asr: AsrSettings = Field(default_factory=AsrSettings)
    announcements: AnnouncementSettings = Field(default_factory=AnnouncementSettings)
    expressive_motion: ExpressiveMotionSettings = Field(default_factory=ExpressiveMotionSettings)
    interface: InterfaceSettings = Field(default_factory=InterfaceSettings)
    motion: MotionSettings = Field(default_factory=MotionSettings)
    llctl: LlctlSettings = Field(default_factory=LlctlSettings)
    vision: VisionSettings = Field(default_factory=VisionSettings)
    response: ResponseSettings = Field(default_factory=ResponseSettings)
    cognition: CognitionSettings = Field(default_factory=CognitionSettings)
    learning: LearningSettings = Field(default_factory=LearningSettings)
    active_learning: ActiveLearningSettings = Field(default_factory=ActiveLearningSettings)
    headlight: HeadlightSettings = Field(default_factory=HeadlightSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    monitor: MonitorSettings = Field(default_factory=MonitorSettings)
    disk: DiskSettings = Field(default_factory=DiskSettings)
    reset: ResetSettings = Field(default_factory=ResetSettings)
    tacit: TacitSettings = Field(default_factory=TacitSettings)
    tools: ToolsSettings = Field(default_factory=ToolsSettings)
    mcp: McpSettings = Field(default_factory=McpSettings)
    self_model: SelfModelSettings = Field(default_factory=SelfModelSettings)
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
        if len(parts) < 2:
            raise KeyError(f"expected '<section>.<field>', got {dotted_key!r}")

        def _updated_model(model: BaseModel, tail: list[str]) -> BaseModel:
            field_name = tail[0]
            model_cls = type(model)
            if field_name not in model_cls.model_fields:
                raise KeyError(f"unknown setting {dotted_key!r}")
            data = model.model_dump()
            if len(tail) == 1:
                data[field_name] = value
            else:
                child = getattr(model, field_name)
                if not isinstance(child, BaseModel):
                    raise KeyError(f"setting {dotted_key!r} crosses a non-nested value at {field_name!r}")
                data[field_name] = _updated_model(child, tail[1:]).model_dump()
            return model_cls.model_validate(data)

        section_name = parts[0]
        if section_name not in type(self).model_fields:
            raise KeyError(f"unknown settings section {section_name!r}")
        section = getattr(self, section_name)
        if not isinstance(section, BaseModel):
            raise KeyError(f"settings section {section_name!r} is not editable")
        setattr(self, section_name, _updated_model(section, parts[1:]))

    def as_flat_dict(self) -> Dict[str, Any]:
        flat: Dict[str, Any] = {}
        def _flatten(prefix: str, node: Any) -> None:
            if isinstance(node, BaseModel):
                for field_name in type(node).model_fields:
                    _flatten(f"{prefix}.{field_name}" if prefix else field_name, getattr(node, field_name))
            else:
                flat[prefix] = node

        _flatten("", self)
        return flat
