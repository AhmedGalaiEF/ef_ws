"""Cheap deterministic attention decisions before planner/model calls."""
from __future__ import annotations

from enum import IntEnum
from typing import Any, List, Optional
import time

from pydantic import BaseModel, ConfigDict

from .semantic_state import SemanticState


class AttentionPriority(IntEnum):
    P0_SAFETY = 0
    P1_USER = 1
    P2_URGENT_INTERNAL = 2
    P3_TASK = 3
    P4_SCENARIO = 4
    P5_BACKGROUND = 5
    P6_MAINTENANCE = 6


class AttentionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    priority: int
    action: str
    reason_code: str
    event_summary: str


class AttentionManager:
    def __init__(self) -> None:
        self._last_semantic: Optional[SemanticState] = None
        self._last_maintenance_at = time.time()

    def decide(
        self,
        *,
        event_type: str,
        semantic_state: SemanticState,
        semantic_changes: Optional[List[str]] = None,
        settings: Optional[Any] = None,
        event_summary: str = "",
        self_model: Optional[Any] = None,
    ) -> AttentionDecision:
        changes = list(semantic_changes or [])
        lowered_event = str(event_type).lower()
        enabled = bool(getattr(getattr(settings, "cognition", object()), "attention_enabled", True))
        if not enabled:
            return AttentionDecision(
                priority=AttentionPriority.P5_BACKGROUND,
                action="cognition_now",
                reason_code="attention_disabled",
                event_summary=event_summary or lowered_event,
            )
        if lowered_event in {"user_message", "asr_message", "chat", "audio"}:
            return AttentionDecision(
                priority=AttentionPriority.P1_USER,
                action="cognition_now",
                reason_code="user_input",
                event_summary=event_summary or lowered_event,
            )
        if lowered_event in {"skill_failed", "task_failed"}:
            return AttentionDecision(
                priority=AttentionPriority.P3_TASK,
                action="cognition_now",
                reason_code="task_failed",
                event_summary=event_summary or "task failed",
            )
        if semantic_state.battery == "critical":
            return AttentionDecision(
                priority=AttentionPriority.P0_SAFETY,
                action="cognition_now",
                reason_code="battery_critical",
                event_summary="battery critical",
            )
        if semantic_state.battery == "low":
            return AttentionDecision(
                priority=AttentionPriority.P2_URGENT_INTERNAL,
                action="cognition_background",
                reason_code="battery_low",
                event_summary="battery low",
            )
        self_boost = 0
        if self_model is not None:
            try:
                self_boost = int(self_model.attention_relevance_boost(semantic_changes=changes))
            except Exception:
                self_boost = 0
        if self_boost > 0 and any(change.startswith(("thermal:", "balance:", "faults:")) for change in changes):
            return AttentionDecision(
                priority=max(AttentionPriority.P2_URGENT_INTERNAL, AttentionPriority.P4_SCENARIO - self_boost),
                action="cognition_background",
                reason_code="self_model_relevance",
                event_summary=", ".join(changes[:4]),
            )
        if any(change.startswith(("battery:", "posture:", "balance:", "arm_state:", "faults:")) for change in changes):
            return AttentionDecision(
                priority=AttentionPriority.P4_SCENARIO,
                action="record",
                reason_code="semantic_change",
                event_summary=", ".join(changes[:4]),
            )
        if lowered_event == "cognitive_tick":
            now = time.time()
            interval_min = float(getattr(getattr(settings, "memory", object()), "consolidation_interval_min", 60))
            if now - self._last_maintenance_at >= max(60.0, interval_min * 60.0):
                self._last_maintenance_at = now
                return AttentionDecision(
                    priority=AttentionPriority.P6_MAINTENANCE,
                    action="cognition_background",
                    reason_code="maintenance_due",
                    event_summary="maintenance due",
                )
            return AttentionDecision(
                priority=AttentionPriority.P5_BACKGROUND,
                action="ignore",
                reason_code="no_meaningful_change",
                event_summary="periodic tick ignored",
            )
        return AttentionDecision(
            priority=AttentionPriority.P5_BACKGROUND,
            action="record",
            reason_code="record_only",
            event_summary=event_summary or lowered_event,
        )
