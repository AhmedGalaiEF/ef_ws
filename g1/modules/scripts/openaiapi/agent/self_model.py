"""Persistent functional self-model for robot-specific learned identity.

The OpenAI model remains general. This module stores durable, auditable
robot-specific abstractions that can influence planning, attention,
skill selection and diagnostics without fine-tuning model weights.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
import uuid

from pydantic import BaseModel, ConfigDict, Field

from .monitor import MonitorEventBus
from .outcomes import SkillOutcome


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class HealthEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = "unknown"
    confidence: float = 0.0
    evidence_refs: List[str] = Field(default_factory=list)
    updated_at: Optional[str] = None


class CalibrationEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = "unknown"
    confidence: float = 0.0
    bias: Optional[float] = None
    evidence_refs: List[str] = Field(default_factory=list)
    updated_at: Optional[str] = None


class LearnedConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: "constraint_" + uuid.uuid4().hex[:10])
    description: str
    condition: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.5
    status: str = "candidate"
    evidence_refs: List[str] = Field(default_factory=list)
    updated_at: str = Field(default_factory=utc_iso)


class BodyAsymmetry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str
    confidence: float = 0.5
    status: str = "candidate"
    evidence_refs: List[str] = Field(default_factory=list)


class BodySelfModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hardware_revision: Optional[str] = None
    actuator_health: Dict[str, HealthEstimate] = Field(default_factory=dict)
    calibration_state: Dict[str, CalibrationEstimate] = Field(default_factory=dict)
    learned_constraints: List[LearnedConstraint] = Field(default_factory=list)
    wear_estimates: Dict[str, float] = Field(default_factory=dict)
    asymmetries: List[BodyAsymmetry] = Field(default_factory=list)
    confidence: float = 0.0


class CapabilityEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability: str
    available_backend: Optional[str] = None
    success_probability: Optional[float] = None
    confidence: float = 0.0
    applicable_conditions: Dict[str, Any] = Field(default_factory=dict)
    failure_modes: List[str] = Field(default_factory=list)
    last_validated_at: Optional[str] = None
    status: str = "candidate"


class CapabilitySelfModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    estimates: Dict[str, CapabilityEstimate] = Field(default_factory=dict)


class ContextPerformance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    context: Dict[str, Any] = Field(default_factory=dict)
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    success_rate: Optional[float] = None
    confidence: float = 0.0


class SkillIdentityRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill: str
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    success_rate: Optional[float] = None
    confidence: float = 0.0
    context_models: List[ContextPerformance] = Field(default_factory=list)
    active_procedures: List[str] = Field(default_factory=list)
    common_failure_modes: List[str] = Field(default_factory=list)
    last_used_at: Optional[str] = None


class SkillSelfModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: Dict[str, SkillIdentityRecord] = Field(default_factory=dict)


class EnergySelfModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    calibrated: bool = False
    observations: int = 0
    mean_prediction_error_pct: Optional[float] = None
    default_task_cost_pct: float = 1.0
    task_cost_pct: Dict[str, float] = Field(default_factory=dict)
    confidence: float = 0.0
    updated_at: Optional[str] = None


class LearnedPreference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: "pref_" + uuid.uuid4().hex[:10])
    domain: str
    condition: Dict[str, Any] = Field(default_factory=dict)
    preferred_option: str
    confidence: float = 0.5
    source: str = "robot_experience"
    evidence_refs: List[str] = Field(default_factory=list)
    status: str = "candidate"
    updated_at: str = Field(default_factory=utc_iso)


class PreferenceModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preferences: List[LearnedPreference] = Field(default_factory=list)


class RelationshipRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: "rel_" + uuid.uuid4().hex[:10])
    label: str
    preferred_name: Optional[str] = None
    interaction_preferences: Dict[str, Any] = Field(default_factory=dict)
    explicitly_taught_facts: List[str] = Field(default_factory=list)
    trust: str = "operator-provided"
    last_interaction_at: Optional[str] = None


class RelationshipModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: List[RelationshipRecord] = Field(default_factory=list)


class Commitment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: "commit_" + uuid.uuid4().hex[:10])
    description: str
    priority: int = Field(default=3, ge=0, le=6)
    created_at: str = Field(default_factory=utc_iso)
    deadline: Optional[str] = None
    state: Literal["active", "completed", "cancelled", "expired"] = "active"


class CommitmentModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    commitments: List[Commitment] = Field(default_factory=list)


class SelfPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success_probability: Optional[float] = None
    energy_cost: Optional[float] = None
    expected_duration_s: Optional[float] = None
    risk_score: Optional[float] = None
    likely_failure_modes: List[str] = Field(default_factory=list)
    confidence: float = 0.0


class SelfModelUpdateProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain: Literal["body", "capability", "skill", "energy", "preference", "relationship", "commitment"]
    proposed_change: Dict[str, Any] = Field(default_factory=dict)
    evidence_refs: List[str] = Field(default_factory=list)
    confidence: float = 0.5
    reason_summary: str


class SelfModelChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int
    timestamp: str = Field(default_factory=utc_iso)
    domains_changed: List[str] = Field(default_factory=list)
    evidence_refs: List[str] = Field(default_factory=list)
    reason: str
    previous_version: int


class SelfModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    robot_id: str
    platform: str = "Unitree G1"
    body: BodySelfModel = Field(default_factory=BodySelfModel)
    capabilities: CapabilitySelfModel = Field(default_factory=CapabilitySelfModel)
    energy: EnergySelfModel = Field(default_factory=EnergySelfModel)
    skills: SkillSelfModel = Field(default_factory=SkillSelfModel)
    preferences: PreferenceModel = Field(default_factory=PreferenceModel)
    relationships: RelationshipModel = Field(default_factory=RelationshipModel)
    commitments: CommitmentModel = Field(default_factory=CommitmentModel)
    autobiography_ref: Optional[str] = None
    created_at: str = Field(default_factory=utc_iso)
    updated_at: str = Field(default_factory=utc_iso)
    history: List[SelfModelChange] = Field(default_factory=list)


class SelfModelStore:
    def __init__(
        self,
        *,
        base_dir: Union[Path, str],
        robot_id: str = "g1_local",
        platform: str = "Unitree G1",
        monitor: Optional[MonitorEventBus] = None,
    ) -> None:
        safe_robot_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(robot_id).strip()) or "g1_local"
        self.robot_id = safe_robot_id
        self.platform = platform
        self.base_dir = Path(base_dir).expanduser() / "self_model" / self.robot_id
        self.path = self.base_dir / "self_model.json"
        self.monitor = monitor
        self._model = self._load_or_create()

    @classmethod
    def from_memory_base(
        cls,
        *,
        base_dir: Union[Path, str],
        robot_id: Optional[str] = None,
        monitor: Optional[MonitorEventBus] = None,
    ) -> "SelfModelStore":
        resolved_id = robot_id or os.environ.get("G1_ROBOT_ID") or os.environ.get("HOSTNAME") or "g1_local"
        return cls(base_dir=base_dir, robot_id=resolved_id, monitor=monitor)

    @property
    def model(self) -> SelfModel:
        return self._model

    def reload(self) -> SelfModel:
        self._model = self._load_or_create()
        return self._model

    def baseline(self) -> SelfModel:
        return SelfModel(robot_id=self.robot_id, platform=self.platform)

    def reset_to_baseline(self, *, reason: str = "reset") -> SelfModel:
        self._model = self.baseline()
        self._save(self._model)
        self._emit("self_model_reset", reason)
        return self._model

    def summary(self) -> Dict[str, Any]:
        model = self.model
        skill_confidence = {}
        ranked_records = sorted(
            model.skills.records.items(),
            key=lambda item: (item[1].attempts, item[1].confidence),
            reverse=True,
        )
        for skill, record in ranked_records[:8]:
            if record.success_rate is not None:
                skill_confidence[skill] = {
                    "success_rate": record.success_rate,
                    "confidence": record.confidence,
                    "attempts": record.attempts,
                }
        return {
            "robot_id": model.robot_id,
            "platform": model.platform,
            "version": model.version,
            "updated_at": model.updated_at,
            "overall_confidence": self.overall_confidence(),
            "current_high_level_condition": self.current_condition(),
            "notable_body_facts": [
                item.description for item in model.body.learned_constraints
                if item.status in {"active", "validated"} and item.confidence >= 0.5
            ][:5],
            "current_commitments": [
                item.description for item in model.commitments.commitments if item.state == "active"
            ][:5],
            "important_learned_preferences": [
                item.preferred_option for item in model.preferences.preferences
                if item.status in {"active", "validated"} and item.confidence >= 0.5
            ][:5],
            "skill_confidence": skill_confidence,
            "energy": {
                "calibrated": model.energy.calibrated,
                "mean_prediction_error_pct": model.energy.mean_prediction_error_pct,
                "confidence": model.energy.confidence,
            },
        }

    def current_condition(self) -> str:
        if any(c.status in {"active", "validated"} for c in self.model.body.learned_constraints):
            return "learned_constraints_present"
        return "nominal"

    def overall_confidence(self) -> float:
        values: List[float] = [self.model.body.confidence, self.model.energy.confidence]
        values.extend(record.confidence for record in self.model.skills.records.values())
        values.extend(pref.confidence for pref in self.model.preferences.preferences)
        meaningful = [value for value in values if value > 0.0]
        return round(sum(meaningful) / len(meaningful), 3) if meaningful else 0.0

    def predict(self, *, candidate_action: str, state: Optional[Any] = None) -> SelfPrediction:
        record = self.model.skills.records.get(candidate_action)
        success = record.success_rate if record is not None else None
        confidence = record.confidence if record is not None else 0.0
        failure_modes = list(record.common_failure_modes[:5]) if record is not None else []
        energy = self.estimate_energy_cost(candidate_action, state=state)
        risk_score = None
        if success is not None:
            risk_score = round(1.0 - success, 3)
        return SelfPrediction(
            success_probability=success,
            energy_cost=energy,
            likely_failure_modes=failure_modes,
            confidence=confidence,
            risk_score=risk_score,
        )

    def estimate_energy_cost(self, task: str, *, state: Optional[Any] = None) -> float:
        del state
        return float(self.model.energy.task_cost_pct.get(task, self.model.energy.default_task_cost_pct))

    def estimate_reserve_after(self, task: str, *, state: Any) -> Optional[float]:
        pct = getattr(state, "battery_pct", None)
        if pct is None:
            return None
        return max(0.0, float(pct) - self.estimate_energy_cost(task, state=state))

    def update_from_skill_outcome(
        self,
        outcome: SkillOutcome,
        *,
        episode_id: Optional[str] = None,
        before: Optional[Any] = None,
        after: Optional[Any] = None,
    ) -> SelfModel:
        del before, after
        model = self.model.model_copy(deep=True)
        record = model.skills.records.get(outcome.skill_id) or SkillIdentityRecord(skill=outcome.skill_id)
        record.attempts += 1
        if outcome.goal_reached:
            record.successes += 1
        else:
            record.failures += 1
            if outcome.failure_type and outcome.failure_type not in record.common_failure_modes:
                record.common_failure_modes.append(outcome.failure_type)
        record.success_rate = round(record.successes / max(1, record.attempts), 3)
        record.confidence = round(min(0.99, record.attempts / (record.attempts + 5.0)), 3)
        record.last_used_at = utc_iso()
        model.skills.records[outcome.skill_id] = record
        model.capabilities.estimates[outcome.skill_id] = CapabilityEstimate(
            capability=outcome.skill_id,
            available_backend="skill_registry",
            success_probability=record.success_rate,
            confidence=record.confidence,
            failure_modes=list(record.common_failure_modes),
            last_validated_at=record.last_used_at,
            status="active" if record.attempts >= 3 else "candidate",
        )
        energy_value = outcome.metrics.get("energy_cost_pct") or abs(outcome.metrics.get("battery_delta_pct", 0.0))
        if energy_value:
            self._update_energy_in_model(model, outcome.skill_id, float(energy_value))
        evidence = [episode_id] if episode_id else [outcome.invocation_id]
        self._accept_update(
            model,
            domains=["skill", "capability"] + (["energy"] if energy_value else []),
            evidence_refs=evidence,
            reason=f"skill outcome recorded for {outcome.skill_id}",
        )
        return self.model

    def calibrate_energy(self, *, task: str, observed_cost_pct: float, evidence_ref: str = "") -> SelfModel:
        model = self.model.model_copy(deep=True)
        self._update_energy_in_model(model, task, float(observed_cost_pct))
        self._accept_update(
            model,
            domains=["energy"],
            evidence_refs=[evidence_ref] if evidence_ref else [],
            reason=f"energy observation for {task}",
        )
        return self.model

    def apply_procedural_adaptation(self, adaptation: Any, *, evidence_ref: str = "") -> SelfModel:
        skill = str(getattr(adaptation, "skill", "") or "")
        if not skill:
            return self.model
        model = self.model.model_copy(deep=True)
        record = model.skills.records.get(skill) or SkillIdentityRecord(skill=skill)
        parameters = dict(getattr(adaptation, "recommended_parameters", {}) or {})
        pre_pose = parameters.get("pre_pose")
        procedure_id = f"{skill}:pre_pose:{pre_pose}" if pre_pose else f"{skill}:procedure"
        if procedure_id not in record.active_procedures:
            record.active_procedures.append(procedure_id)
        record.confidence = max(record.confidence, float(getattr(adaptation, "confidence", 0.5) or 0.5))
        model.skills.records[skill] = record
        preference = LearnedPreference(
            domain=f"skill:{skill}",
            condition=dict(getattr(adaptation, "condition", {}) or {}),
            preferred_option=procedure_id,
            confidence=float(getattr(adaptation, "confidence", 0.5) or 0.5),
            source="procedural_tacit_memory",
            evidence_refs=list(getattr(adaptation, "derived_from", []) or []) + ([evidence_ref] if evidence_ref else []),
            status="active",
        )
        if not any(pref.domain == preference.domain and pref.preferred_option == preference.preferred_option for pref in model.preferences.preferences):
            model.preferences.preferences.append(preference)
        self._accept_update(
            model,
            domains=["skill", "preference"],
            evidence_refs=preference.evidence_refs,
            reason=f"procedural adaptation activated for {skill}",
        )
        return self.model

    def learned_skill_kwargs(self, skill_name: str) -> Dict[str, Any]:
        record = self.model.skills.records.get(skill_name)
        if record is None:
            return {}
        for procedure in record.active_procedures:
            if ":pre_pose:" in procedure:
                return {"learned_pre_pose": procedure.rsplit(":pre_pose:", 1)[1]}
        return {}

    def add_body_constraint(
        self,
        *,
        description: str,
        condition: Optional[Dict[str, Any]] = None,
        confidence: float = 0.5,
        status: str = "active",
        evidence_refs: Optional[List[str]] = None,
    ) -> SelfModel:
        model = self.model.model_copy(deep=True)
        model.body.learned_constraints.append(
            LearnedConstraint(
                description=description,
                condition=dict(condition or {}),
                confidence=max(0.0, min(1.0, float(confidence))),
                status=status,
                evidence_refs=list(evidence_refs or []),
            )
        )
        model.body.confidence = max(model.body.confidence, max(0.0, min(1.0, float(confidence))))
        self._accept_update(model, domains=["body"], evidence_refs=list(evidence_refs or []), reason=description)
        return self.model

    def attention_relevance_boost(self, *, semantic_changes: List[str]) -> int:
        if not semantic_changes:
            return 0
        boost = 0
        for change in semantic_changes:
            lowered = change.lower()
            for constraint in self.model.body.learned_constraints:
                haystack = (constraint.description + " " + json.dumps(constraint.condition, default=str)).lower()
                if constraint.status in {"active", "validated"} and any(token in haystack for token in ("thermal", "temperature", "knee")) and "thermal" in lowered:
                    boost = max(boost, 2)
        return boost

    def invalidate(self, target: str, *, reason: str = "hardware invalidation") -> SelfModel:
        lowered = str(target).lower()
        model = self.model.model_copy(deep=True)
        for constraint in model.body.learned_constraints:
            if lowered in (constraint.description + json.dumps(constraint.condition, default=str)).lower():
                constraint.status = "deprecated"
                constraint.confidence = min(constraint.confidence, 0.2)
                constraint.updated_at = utc_iso()
        for skill, record in model.skills.records.items():
            if lowered in skill.lower() or lowered in json.dumps(record.model_dump(), default=str).lower():
                record.confidence = min(record.confidence, 0.2)
                record.context_models = []
        self._accept_update(model, domains=["body", "skill"], evidence_refs=[], reason=f"{reason}: {target}")
        return self.model

    def _load_or_create(self) -> SelfModel:
        if self.path.exists():
            try:
                return SelfModel.model_validate(json.loads(self.path.read_text(encoding="utf-8") or "{}"))
            except Exception:
                pass
        model = self.baseline()
        self._save(model)
        return model

    def _save(self, model: SelfModel) -> None:
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(model.model_dump(), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
            tmp.replace(self.path)
        except OSError:
            fallback = Path("/tmp/g1_agent_self_model") / self.robot_id
            if self.base_dir == fallback:
                raise
            self.base_dir = fallback
            self.path = self.base_dir / "self_model.json"
            self.base_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(model.model_dump(), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
            tmp.replace(self.path)

    def _accept_update(self, model: SelfModel, *, domains: List[str], evidence_refs: List[str], reason: str) -> None:
        previous = self.model.version
        model.version = previous + 1
        model.updated_at = utc_iso()
        model.history.append(
            SelfModelChange(
                version=model.version,
                previous_version=previous,
                domains_changed=sorted(set(domains)),
                evidence_refs=[ref for ref in evidence_refs if ref],
                reason=reason,
            )
        )
        model.history = model.history[-200:]
        self._model = model
        self._save(model)
        self._emit("self_model_updated", reason, references=evidence_refs, metadata={"version": model.version, "domains": domains})

    @staticmethod
    def _update_energy_in_model(model: SelfModel, task: str, observed_cost_pct: float) -> None:
        energy = model.energy
        previous = float(energy.task_cost_pct.get(task, energy.default_task_cost_pct))
        count = max(0, int(energy.observations))
        blended = ((previous * count) + observed_cost_pct) / (count + 1)
        error = abs(previous - observed_cost_pct)
        if energy.mean_prediction_error_pct is None:
            mean_error = error
        else:
            mean_error = ((energy.mean_prediction_error_pct * count) + error) / (count + 1)
        energy.task_cost_pct[task] = round(blended, 3)
        energy.observations = count + 1
        energy.mean_prediction_error_pct = round(mean_error, 3)
        energy.calibrated = energy.observations >= 1
        energy.confidence = round(min(0.95, energy.observations / (energy.observations + 5.0)), 3)
        energy.updated_at = utc_iso()

    def _emit(self, event: str, summary: str, *, references: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        if self.monitor is not None:
            self.monitor.emit("self", event, summary, references=references or [], metadata=metadata or {})
