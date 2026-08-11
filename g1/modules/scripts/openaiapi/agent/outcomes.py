"""Skill postcondition/outcome evaluation.

The runtime evaluates physical truth from skill results and observed
semantic state. The planner may receive the summary later, but it does
not get to declare success from text alone.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, ConfigDict, Field

from .semantic_state import SemanticState


class SkillOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill_id: str
    invocation_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    command_accepted: bool
    execution_completed: bool
    goal_reached: bool
    safe: bool
    failure_type: Optional[str] = None
    anomalies: List[str] = Field(default_factory=list)
    expected_semantic_state: Dict[str, Any] = Field(default_factory=dict)
    observed_semantic_state: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, float] = Field(default_factory=dict)
    started_at: datetime
    completed_at: datetime


class OutcomeEvaluator:
    def evaluate(
        self,
        *,
        skill_id: str,
        invocation_id: str,
        invocation_outcome: Any,
        before: SemanticState,
        after: SemanticState,
        started_at: datetime,
        completed_at: datetime,
    ) -> SkillOutcome:
        status = str(getattr(invocation_outcome, "status", "unknown"))
        result = getattr(invocation_outcome, "result", None)
        message = str(getattr(result, "message", "") if result is not None else "")
        ok = bool(getattr(result, "ok", False)) if result is not None else False
        command_accepted = status == "executed"
        execution_completed = result is not None and status == "executed"
        safe = "safety" not in message.lower() and "unsafe" not in after.thermal
        anomalies: List[str] = []
        failure_type: Optional[str] = None
        if status == "denied":
            failure_type = "denied"
        elif status != "executed":
            failure_type = "not_executed"
        elif not ok:
            failure_type = "goal_not_reached"
        if "safety" in message.lower():
            failure_type = "safety_rejection"
            anomalies.append("safety_rejection")
        if after.balance in {"degraded", "recovery"} and before.balance == "stable":
            anomalies.append("balance_degraded")
        duration_s = max(0.0, (completed_at - started_at).total_seconds())
        return SkillOutcome(
            skill_id=skill_id,
            invocation_id=invocation_id,
            command_accepted=command_accepted,
            execution_completed=execution_completed,
            goal_reached=bool(ok),
            safe=safe,
            failure_type=failure_type,
            anomalies=anomalies,
            expected_semantic_state={"skill": skill_id, "goal": "skill_result_ok"},
            observed_semantic_state=after.model_dump(),
            metrics={"duration_s": duration_s},
            started_at=started_at,
            completed_at=completed_at,
        )


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
