"""Typed planner I/O and shared runtime models.

These are the structures that cross the boundary into and out of the
language model (spec sections 2, 3, 4, 6, 13). Everything the model can
*do* is a named field on ``PlannerDecision`` -- there is no field, at any
level, that could hold a raw ``q``/``dq``/``kp``/``kd``/``tau`` value, a
servo packet, or an arbitrary trajectory. See ``agent/skills.py`` for how
``requested_skills`` is resolved into an actual, validated action.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    AGENT_FIRST_BOOT = "agent_first_boot"
    AGENT_RESTART = "agent_restart"
    AGENT_WAKE = "agent_wake"
    USER_MESSAGE = "user_message"
    ASR_MESSAGE = "asr_message"
    SEMANTIC_EVENT = "semantic_event"
    SCENARIO_TRANSITION = "scenario_transition"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    ANOMALY = "anomaly"
    COGNITIVE_TICK = "cognitive_tick"


class LifecycleState(str, Enum):
    FIRST_BOOT = "first_boot"
    AWAKE = "awake"
    MAINTENANCE = "maintenance"
    PRE_SLEEP = "pre_sleep"
    SLEEPING = "sleeping"
    WAKING = "waking"
    RESTART_RECOVERY = "restart_recovery"


class IntentType(str, Enum):
    CONVERSATION = "conversation"
    QUERY_CAPABILITY = "query_capability"
    QUERY_STATE = "query_state"
    MOVE_ARM = "move_arm"
    EXECUTE_TASK = "execute_task"
    REQUEST_CHARGE = "request_charge"
    MAINTENANCE = "maintenance"
    REQUEST_SLEEP = "request_sleep"
    NO_ACTION = "no_action"


# ---------------------------------------------------------------------------
# Live state / scenario / capability structures
# ---------------------------------------------------------------------------

class RobotStateSnapshot(BaseModel):
    """Semantic robot state -- never raw /lowstate telemetry (spec section 22)."""

    model_config = ConfigDict(extra="allow")

    timestamp: float
    posture: str = "unknown"
    battery_pct: Optional[float] = None
    charging: Optional[bool] = None
    stability: str = "unknown"
    active_faults: List[str] = Field(default_factory=list)
    arm_control_state: str = "unknown"
    right_hand_state: Optional[str] = None
    left_hand_state: Optional[str] = None
    lowstate: Optional[Dict[str, Any]] = None
    battery: Optional[Dict[str, Any]] = None
    sensor_stale: Dict[str, Any] = Field(default_factory=dict)
    sensor_timestamps: Dict[str, Any] = Field(default_factory=dict)
    stale_sensor_topics: List[str] = Field(default_factory=list)
    source: str = "unavailable"  # e.g. "sdk_client.Robot" or "mock"


class ScenarioState(BaseModel):
    name: Optional[str] = None
    phase: Optional[str] = None
    objectives: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    available_transitions: List[str] = Field(default_factory=list)


class CapabilityStatus(BaseModel):
    """One entry of the capability summary handed to the planner.

    This is a *summary for the model to read*, not a grant of permission --
    the actual authorization happens in ``agent/capabilities.py`` at
    execution time, deterministically, regardless of what the model
    believes is true.
    """

    available: bool
    reason: str = ""


class KnowledgeRef(BaseModel):
    """One retrieved knowledge chunk, carrying explicit provenance (spec section 21)."""

    source_type: str  # "implementation" | "documentary" | "episodic" | "semantic" |
                       # "procedural" | "autobiographical" | "live_state"
    source: str
    text: str
    line_range: Optional[str] = None
    trust: str = "medium"  # "authoritative" | "medium" | "low"
    note: str = ""


class MemoryProposal(BaseModel):
    """A proposed memory write the model may suggest; never applied directly (spec section 9)."""

    kind: str  # "episodic" | "semantic" | "procedural" | "autobiographical"
    content: Dict[str, Any]
    confidence: float = 0.5
    derived_from: List[str] = Field(default_factory=list)


class MaintenanceProposal(BaseModel):
    kind: str
    description: str
    details: Dict[str, Any] = Field(default_factory=dict)


class IntentAnnouncement(BaseModel):
    speech: Optional[str] = None
    gesture: Optional[str] = None


# ---------------------------------------------------------------------------
# Planner input / output
# ---------------------------------------------------------------------------

class PlannerInput(BaseModel):
    """Everything the planner receives for one cognitive turn (spec sections 2, 31)."""

    model_config = ConfigDict(extra="forbid")

    event: EventType
    timestamp: float
    previous_cognitive_timestamp: Optional[float] = None
    elapsed_since_last_cognition_s: Optional[float] = None

    input_source: Optional[str] = None  # "chat" | "audio" | "system"
    user_text: Optional[str] = None  # preserved verbatim, never rewritten

    robot_state: RobotStateSnapshot
    lifecycle_state: LifecycleState

    active_scenario: Optional[ScenarioState] = None
    current_task: Optional[str] = None
    current_skill: Optional[str] = None

    available_skills: List[str] = Field(default_factory=list)
    capability_summary: Dict[str, CapabilityStatus] = Field(default_factory=dict)
    settings: Dict[str, Any] = Field(default_factory=dict)

    documentary_rag: List[KnowledgeRef] = Field(default_factory=list)
    sdk_wrapper_knowledge: List[KnowledgeRef] = Field(default_factory=list)
    episodic_memory: List[KnowledgeRef] = Field(default_factory=list)
    semantic_memory: List[KnowledgeRef] = Field(default_factory=list)
    procedural_memory: List[KnowledgeRef] = Field(default_factory=list)
    autobiography_summary: Optional[str] = None

    available_tools: List[str] = Field(default_factory=list)

    # Only populated for agent_first_boot / agent_restart / agent_wake.
    runtime: Dict[str, Any] = Field(default_factory=dict)


class PlannerDecision(BaseModel):
    """Structured output of one cognitive turn (spec section 2)."""

    model_config = ConfigDict(extra="forbid")

    intent: IntentType
    target: Optional[str] = None
    response_text: Optional[str] = None
    requested_skills: List[str] = Field(default_factory=list)
    intent_announcement: Optional[IntentAnnouncement] = None
    memory_proposal: Optional[MemoryProposal] = None
    maintenance_proposal: Optional[MaintenanceProposal] = None
    next_tick_s: Optional[float] = None


# ---------------------------------------------------------------------------
# Runtime checkpoint (spec section 6) -- lifecycle continuity only.
# ---------------------------------------------------------------------------

class RuntimeCheckpoint(BaseModel):
    version: int = 1
    last_cognitive_timestamp: Optional[float] = None
    lifecycle_state: LifecycleState = LifecycleState.FIRST_BOOT
    last_event_type: Optional[EventType] = None
    last_decision: Optional[IntentType] = None
    active_scenario: Optional[str] = None
    scenario_phase: Optional[str] = None
    active_task: Optional[str] = None
    active_skill: Optional[str] = None
    last_robot_state_summary: Dict[str, Any] = Field(default_factory=dict)
    sleep_reason: Optional[str] = None
    sleep_timestamp: Optional[float] = None
    wake_timestamp: Optional[float] = None
