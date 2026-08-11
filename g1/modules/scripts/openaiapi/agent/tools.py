"""Canonical cognitive tool registry.

The registry normalizes local Python tools and future MCP-backed tools
behind one typed interface. It is intentionally not a low-level robot
control surface: physical actions are routed through the existing
SkillRegistry + CapabilityResolver + outcome evaluator path.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .models import RobotStateSnapshot
from .monitor import MonitorEventBus
from .settings.models import AgentSettings


class ToolCategory(str, Enum):
    OBSERVATION = "observation"
    KNOWLEDGE = "knowledge"
    MEMORY = "memory"
    DIAGNOSTIC = "diagnostic"
    CONFIGURATION = "configuration"
    LEARNING = "learning"
    ACTION = "action"


class ToolAvailability(str, Enum):
    AVAILABLE = "available"
    DISABLED_BY_SETTING = "disabled_by_setting"
    UNAVAILABLE_BACKEND = "unavailable_backend"
    BLOCKED_BY_STATE = "blocked_by_state"
    BLOCKED_BY_SAFETY = "blocked_by_safety"
    OPERATOR_ONLY = "operator_only"


class ToolErrorCode(str, Enum):
    DISABLED = "DISABLED"
    UNAVAILABLE = "UNAVAILABLE"
    TIMEOUT = "TIMEOUT"
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    BLOCKED_BY_STATE = "BLOCKED_BY_STATE"
    BLOCKED_BY_SAFETY = "BLOCKED_BY_SAFETY"
    NOT_FOUND = "NOT_FOUND"
    STALE_STATE = "STALE_STATE"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    tool: str
    category: ToolCategory
    source_type: str = "tool"
    content: Any = None
    summary: str = ""
    provenance: Dict[str, Any] = Field(default_factory=dict)
    error_code: Optional[ToolErrorCode] = None
    message: str = ""
    retryable: bool = False
    duration_ms: float = 0.0
    result_ref: Optional[str] = None


class ToolMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    category: ToolCategory
    read_only: bool = True
    risk_level: str = "low"
    required_capabilities: List[str] = Field(default_factory=list)
    required_settings: List[str] = Field(default_factory=list)
    timeout_s: float = 5.0
    profiles: List[str] = Field(default_factory=lambda: ["social", "diagnostic", "maintenance"])
    operator_only: bool = False


class EmptyArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QueryArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    limit: int = Field(default=5, ge=1, le=10)


class OptionalLimitArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    limit: int = Field(default=5, ge=1, le=10)


class JointArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    joint: str = ""
    side: str = ""


class SourceExcerptArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file: str
    start_line: int = Field(default=1, ge=1)
    line_count: int = Field(default=40, ge=1, le=120)


class SettingPathArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = ""


class SkillActionArgs(BaseModel):
    model_config = ConfigDict(extra="allow")


class EmpiricalMemoryArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str
    supporting_episodes: List[str] = Field(default_factory=list)
    contradicting_episodes: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    applicable_context: Dict[str, Any] = Field(default_factory=dict)
    risk_level: int = Field(default=1, ge=0, le=5)
    tool_evidence: List[Dict[str, Any]] = Field(default_factory=list)


class ProceduralAdaptationArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill: str
    condition: Dict[str, Any] = Field(default_factory=dict)
    recommended_parameters: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    derived_from: List[str] = Field(default_factory=list)
    tool_evidence: List[Dict[str, Any]] = Field(default_factory=list)


class MemoryContradictionArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    episode_id: str


class CandidateActionArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str


@dataclass
class ToolContext:
    agent: Any
    settings: AgentSettings
    robot_state: RobotStateSnapshot
    profile: str = "social"
    event: str = ""


@dataclass
class ToolDefinition:
    metadata: ToolMetadata
    handler: Callable[[BaseModel, ToolContext], ToolResult]
    input_model: Type[BaseModel] = EmptyArgs
    availability: Callable[[ToolContext], ToolAvailability] = lambda _ctx: ToolAvailability.AVAILABLE

    def schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.metadata.name,
                "description": self.metadata.description,
                "parameters": self.input_model.model_json_schema(),
            },
        }


@dataclass
class ToolTurnSession:
    registry: "ToolRegistry"
    context: ToolContext
    max_calls: int
    calls: int = 0
    seen: Set[str] = field(default_factory=set)

    def invoke(self, name: str, **kwargs: Any) -> str:
        if self.calls >= self.max_calls:
            result = self.registry._error_result(
                name,
                ToolCategory.DIAGNOSTIC,
                ToolErrorCode.BUDGET_EXCEEDED,
                f"tool call budget exhausted ({self.calls}/{self.max_calls})",
            )
            return result.model_dump_json()
        signature = json.dumps({"name": name, "args": kwargs}, sort_keys=True, default=str)
        if signature in self.seen:
            result = self.registry._error_result(
                name,
                ToolCategory.DIAGNOSTIC,
                ToolErrorCode.BUDGET_EXCEEDED,
                "repeated equivalent tool call rejected",
            )
            return result.model_dump_json()
        self.seen.add(signature)
        self.calls += 1
        return self.registry.invoke(name, kwargs, self.context).model_dump_json()


class ToolRegistry:
    def __init__(self, *, monitor: Optional[MonitorEventBus] = None, audit_path: Optional[Union[Path, str]] = None) -> None:
        self._tools: Dict[str, ToolDefinition] = {}
        self.monitor = monitor
        self.audit_path = Path(audit_path).expanduser() if audit_path is not None else None
        self.mcp_health: Dict[str, Dict[str, Any]] = {
            "memory": {"enabled": True, "connected": False, "available_tools": 0, "last_error": "no MCP server configured"},
            "documentation": {"enabled": True, "connected": False, "available_tools": 0, "last_error": "no MCP server configured"},
            "diagnostics": {"enabled": True, "connected": False, "available_tools": 0, "last_error": "no MCP server configured"},
        }

    def register(self, definition: ToolDefinition) -> None:
        self._tools[definition.metadata.name] = definition

    def names(self) -> List[str]:
        return sorted(self._tools)

    def definition(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def session(self, context: ToolContext) -> ToolTurnSession:
        return ToolTurnSession(
            registry=self,
            context=context,
            max_calls=max(0, int(context.settings.tools.max_calls_per_turn)),
        )

    def available_for(self, context: ToolContext, *, include_unavailable: bool = False) -> List[Dict[str, Any]]:
        rows = []
        for name in sorted(self._tools):
            definition = self._tools[name]
            availability = self.availability(name, context)
            if availability == ToolAvailability.AVAILABLE or include_unavailable:
                rows.append(
                    {
                        **definition.metadata.model_dump(),
                        "availability": availability.value,
                    }
                )
        return rows

    def schemas_for(self, context: ToolContext) -> List[Dict[str, Any]]:
        if int(context.settings.tools.max_calls_per_turn) <= 0:
            return []
        return [
            self._tools[row["name"]].schema()
            for row in self.available_for(context)
            if row["availability"] == ToolAvailability.AVAILABLE
        ]

    def callables_for(self, session: ToolTurnSession) -> Dict[str, Callable[..., str]]:
        if session.max_calls <= 0:
            return {}
        return {
            row["name"]: (lambda _name=row["name"], **kwargs: session.invoke(_name, **kwargs))
            for row in self.available_for(session.context)
            if row["availability"] == ToolAvailability.AVAILABLE
        }

    def availability(self, name: str, context: ToolContext) -> ToolAvailability:
        definition = self._tools.get(name)
        if definition is None:
            return ToolAvailability.UNAVAILABLE_BACKEND
        settings = context.settings
        if not settings.tools.enabled:
            return ToolAvailability.DISABLED_BY_SETTING
        if definition.metadata.operator_only:
            return ToolAvailability.OPERATOR_ONLY
        category = definition.metadata.category
        if category == ToolCategory.OBSERVATION and not settings.tools.observation.enabled:
            return ToolAvailability.DISABLED_BY_SETTING
        if category == ToolCategory.MEMORY and not settings.tools.memory.enabled:
            return ToolAvailability.DISABLED_BY_SETTING
        if category == ToolCategory.KNOWLEDGE and not settings.tools.knowledge.enabled:
            return ToolAvailability.DISABLED_BY_SETTING
        if category == ToolCategory.DIAGNOSTIC and not settings.tools.diagnostics.enabled:
            return ToolAvailability.DISABLED_BY_SETTING
        if category == ToolCategory.ACTION and not settings.tools.actions.enabled:
            return ToolAvailability.DISABLED_BY_SETTING
        if category == ToolCategory.LEARNING and not settings.learning.enabled:
            return ToolAvailability.DISABLED_BY_SETTING
        if context.profile and definition.metadata.profiles and context.profile not in definition.metadata.profiles:
            return ToolAvailability.DISABLED_BY_SETTING
        return definition.availability(context)

    def invoke(self, name: str, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        started = time.time()
        definition = self._tools.get(name)
        if definition is None:
            return self._error_result(name, ToolCategory.DIAGNOSTIC, ToolErrorCode.NOT_FOUND, f"unknown tool {name!r}")
        availability = self.availability(name, context)
        if availability != ToolAvailability.AVAILABLE:
            return self._error_result(
                name,
                definition.metadata.category,
                ToolErrorCode.DISABLED if availability == ToolAvailability.DISABLED_BY_SETTING else ToolErrorCode.UNAVAILABLE,
                f"tool {name} is {availability.value}",
            )
        self._emit("tool_call_started", name, {"args": self._redact_args(args), "category": definition.metadata.category.value})
        try:
            parsed = definition.input_model.model_validate(args)
        except ValidationError as exc:
            result = self._error_result(name, definition.metadata.category, ToolErrorCode.INVALID_ARGUMENT, str(exc))
        else:
            try:
                result = definition.handler(parsed, context)
            except Exception as exc:
                result = self._error_result(name, definition.metadata.category, ToolErrorCode.INTERNAL_ERROR, str(exc), retryable=True)
        result.duration_ms = (time.time() - started) * 1000.0
        result.result_ref = result.result_ref or f"tool_result_{int(started * 1000)}_{name}"
        self._emit(
            "tool_call_completed",
            f"{name} ok={result.ok} {result.summary or result.message}",
            {"duration_ms": result.duration_ms, "result_ref": result.result_ref},
        )
        self._append_audit(name=name, args=args, result=result, context=context)
        return result

    def snapshot(self, context: ToolContext) -> Dict[str, Any]:
        rows = self.available_for(context, include_unavailable=True)
        mcp_health = {}
        servers = {
            "memory": context.settings.mcp.servers.memory.enabled,
            "documentation": context.settings.mcp.servers.documentation.enabled,
            "diagnostics": context.settings.mcp.servers.diagnostics.enabled,
        }
        for name, health in self.mcp_health.items():
            enabled = bool(context.settings.mcp.enabled and servers.get(name, True))
            mcp_health[name] = {**health, "enabled": enabled}
            if not enabled:
                mcp_health[name]["connected"] = False
                mcp_health[name]["last_error"] = "disabled by settings"
        return {
            "available_tools": sum(1 for row in rows if row["availability"] == ToolAvailability.AVAILABLE.value),
            "read_only": sum(1 for row in rows if row["availability"] == ToolAvailability.AVAILABLE.value and row["read_only"]),
            "action": sum(1 for row in rows if row["availability"] == ToolAvailability.AVAILABLE.value and row["category"] == ToolCategory.ACTION.value),
            "tools": rows,
            "mcp": mcp_health,
            "max_calls_per_turn": context.settings.tools.max_calls_per_turn,
        }

    def _append_audit(self, *, name: str, args: Dict[str, Any], result: ToolResult, context: ToolContext) -> None:
        if self.audit_path is None:
            return
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": time.time(),
            "event": context.event,
            "tool": name,
            "category": result.category.value,
            "arguments": self._redact_args(args),
            "ok": result.ok,
            "error_code": None if result.error_code is None else result.error_code.value,
            "duration_ms": result.duration_ms,
            "read_only": self._tools[name].metadata.read_only if name in self._tools else True,
            "summary": result.summary or result.message,
        }
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        self._trim_audit(max_lines=2000)

    def _trim_audit(self, *, max_lines: int) -> None:
        if self.audit_path is None or not self.audit_path.exists():
            return
        lines = self.audit_path.read_text(encoding="utf-8").splitlines()
        if len(lines) <= max_lines:
            return
        self.audit_path.write_text("\n".join(lines[-max_lines:]) + "\n", encoding="utf-8")

    def _emit(self, event: str, summary: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        if self.monitor is not None:
            self.monitor.emit("tool", event, summary, metadata=metadata or {})

    @staticmethod
    def _redact_args(args: Dict[str, Any]) -> Dict[str, Any]:
        redacted = {}
        for key, value in args.items():
            lowered = str(key).lower()
            if any(token in lowered for token in ("key", "token", "secret", "password", "authorization")):
                redacted[key] = "[redacted]"
            else:
                redacted[key] = value
        return redacted

    @staticmethod
    def _error_result(
        name: str,
        category: ToolCategory,
        code: ToolErrorCode,
        message: str,
        *,
        retryable: bool = False,
    ) -> ToolResult:
        return ToolResult(
            ok=False,
            tool=name,
            category=category,
            error_code=code,
            message=message,
            retryable=retryable,
        )


def _ok(name: str, category: ToolCategory, content: Any, *, summary: str, source_type: str, provenance: Optional[Dict[str, Any]] = None) -> ToolResult:
    return ToolResult(
        ok=True,
        tool=name,
        category=category,
        content=content,
        summary=summary,
        source_type=source_type,
        provenance=provenance or {},
    )


def _semantic_robot_state(state: RobotStateSnapshot) -> Dict[str, Any]:
    data = state.model_dump()
    lowstate = data.pop("lowstate", None) or {}
    if lowstate:
        data["lowstate_summary"] = {
            "timestamp": lowstate.get("timestamp"),
            "joint_count": lowstate.get("joint_count"),
            "imu": lowstate.get("imu"),
            "source": lowstate.get("source"),
        }
    return data


def _read_source_excerpt(path: Path, *, start_line: int, line_count: int, allowed_roots: List[Path]) -> Dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not any(str(resolved).startswith(str(root.resolve())) for root in allowed_roots):
        raise PermissionError(f"{resolved} is outside allowed source roots")
    if any(part in {".env", ".ssh"} for part in resolved.parts):
        raise PermissionError("refusing to read sensitive path")
    lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(1, start_line)
    end = min(len(lines), start + line_count - 1)
    return {
        "file": str(resolved),
        "line_range": f"{start}-{end}",
        "text": "\n".join(f"{idx}: {lines[idx - 1]}" for idx in range(start, end + 1)),
    }


def build_default_tool_registry(agent: Any) -> ToolRegistry:
    registry = ToolRegistry(
        monitor=agent.monitor_bus,
        audit_path=Path(agent.learning.base_dir) / "tool_audit.jsonl",
    )

    def register(
        name: str,
        description: str,
        category: ToolCategory,
        input_model: Type[BaseModel],
        handler: Callable[[BaseModel, ToolContext], ToolResult],
        *,
        read_only: bool = True,
        risk_level: str = "low",
        profiles: Optional[List[str]] = None,
        operator_only: bool = False,
        required_settings: Optional[List[str]] = None,
        availability: Callable[[ToolContext], ToolAvailability] = lambda _ctx: ToolAvailability.AVAILABLE,
    ) -> None:
        registry.register(
            ToolDefinition(
                metadata=ToolMetadata(
                    name=name,
                    description=description,
                    category=category,
                    read_only=read_only,
                    risk_level=risk_level,
                    profiles=profiles or ["social", "diagnostic", "maintenance", "navigation", "manipulation"],
                    operator_only=operator_only,
                    required_settings=required_settings or [],
                ),
                input_model=input_model,
                handler=handler,
                availability=availability,
            )
        )

    register(
        "get_robot_state",
        "Return current semantic robot state without raw high-frequency joint arrays.",
        ToolCategory.OBSERVATION,
        EmptyArgs,
        lambda _args, ctx: _ok(
            "get_robot_state",
            ToolCategory.OBSERVATION,
            _semantic_robot_state(ctx.robot_state),
            summary=f"posture={ctx.robot_state.posture} stability={ctx.robot_state.stability}",
            source_type="robot_state",
            provenance={"observed_at": ctx.robot_state.timestamp, "source": ctx.robot_state.source},
        ),
    )

    register(
        "get_battery_state",
        "Return current battery and charging summary with freshness.",
        ToolCategory.OBSERVATION,
        EmptyArgs,
        lambda _args, ctx: _ok(
            "get_battery_state",
            ToolCategory.OBSERVATION,
            {"battery_pct": ctx.robot_state.battery_pct, "charging": ctx.robot_state.charging, "battery": ctx.robot_state.battery},
            summary="battery unavailable" if ctx.robot_state.battery_pct is None else f"battery={ctx.robot_state.battery_pct:.0f}%",
            source_type="robot_state",
            provenance={"observed_at": ctx.robot_state.timestamp, "age_ms": max(0.0, (time.time() - ctx.robot_state.timestamp) * 1000.0)},
        ),
    )

    register(
        "get_arm_state",
        "Return semantic arm commandability and relevant faults.",
        ToolCategory.OBSERVATION,
        EmptyArgs,
        lambda _args, ctx: _ok(
            "get_arm_state",
            ToolCategory.OBSERVATION,
            {
                "arm_control_state": ctx.robot_state.arm_control_state,
                "active_faults": [f for f in ctx.robot_state.active_faults if "arm" in f or "lowstate" in f],
                "stale_sensor_topics": ctx.robot_state.stale_sensor_topics,
            },
            summary=f"arm_control_state={ctx.robot_state.arm_control_state}",
            source_type="robot_state",
            provenance={"observed_at": ctx.robot_state.timestamp},
        ),
    )

    def _joint_summary(args: BaseModel, ctx: ToolContext) -> ToolResult:
        parsed = args  # JointArgs
        lowstate = ctx.robot_state.lowstate or {}
        q = lowstate.get("joint_positions") or []
        dq = lowstate.get("joint_velocities") or []
        tau = lowstate.get("joint_torques") or []
        content = {
            "joint_count": lowstate.get("joint_count", len(q)),
            "requested_joint": getattr(parsed, "joint", ""),
            "side": getattr(parsed, "side", ""),
            "position_range": [min(q), max(q)] if q else None,
            "velocity_abs_max": max((abs(float(v)) for v in dq), default=None),
            "torque_abs_max": max((abs(float(v)) for v in tau), default=None),
            "raw_values_included": False,
        }
        return _ok(
            "get_joint_summary",
            ToolCategory.OBSERVATION,
            content,
            summary=f"joint_count={content['joint_count']} raw_values_included=false",
            source_type="robot_state",
            provenance={"source": lowstate.get("source", "/lowstate"), "observed_at": lowstate.get("timestamp")},
        )

    register("get_joint_summary", "Summarize joint telemetry without raw arrays.", ToolCategory.OBSERVATION, JointArgs, _joint_summary, profiles=["diagnostic", "manipulation"])

    def _joint_state(args: BaseModel, ctx: ToolContext) -> ToolResult:
        parsed: JointArgs = args  # type: ignore[assignment]
        lowstate = ctx.robot_state.lowstate or {}
        q = lowstate.get("joint_positions") or []
        dq = lowstate.get("joint_velocities") or []
        tau = lowstate.get("joint_torques") or []
        if not q:
            return registry._error_result("get_joint_state", ToolCategory.OBSERVATION, ToolErrorCode.STALE_STATE, "no lowstate joint data available")
        content = {
            "joint": parsed.joint or "index_0",
            "index": 0,
            "position": q[0],
            "velocity": dq[0] if dq else None,
            "torque": tau[0] if tau else None,
            "note": "Name-to-index mapping is not available in this state snapshot; returned index 0 unless runtime adds mapping.",
        }
        return _ok("get_joint_state", ToolCategory.OBSERVATION, content, summary=f"{content['joint']} position={content['position']}", source_type="robot_state", provenance={"source": lowstate.get("source", "/lowstate")})

    register("get_joint_state", "Return one requested joint's current state; bounded to a single joint.", ToolCategory.OBSERVATION, JointArgs, _joint_state, profiles=["diagnostic", "manipulation"])

    register(
        "get_navigation_status",
        "Return current navigation/SLAM status summary.",
        ToolCategory.OBSERVATION,
        EmptyArgs,
        lambda _args, ctx: _ok("get_navigation_status", ToolCategory.OBSERVATION, ctx.agent.navigation.snapshot().as_dict(), summary="navigation snapshot", source_type="robot_state", provenance={"source": "NavigationAdapter"}),
        profiles=["navigation", "diagnostic"],
    )

    register(
        "get_vision_observation",
        "Return latest semantic vision observation, if any.",
        ToolCategory.OBSERVATION,
        EmptyArgs,
        lambda _args, ctx: _ok("get_vision_observation", ToolCategory.OBSERVATION, ctx.agent.visual_observations.snapshot(), summary="vision observation snapshot", source_type="vision_model", provenance={"model": ctx.settings.vision.openai_model}),
    )

    register("get_asr_status", "Return ASR/microphone runtime status.", ToolCategory.OBSERVATION, EmptyArgs, lambda _args, ctx: _ok("get_asr_status", ToolCategory.OBSERVATION, ctx.agent.asr_snapshot(), summary="asr snapshot", source_type="runtime_state"))
    register("get_activity_state", "Return explicit agent activity/headlight state.", ToolCategory.OBSERVATION, EmptyArgs, lambda _args, ctx: _ok("get_activity_state", ToolCategory.OBSERVATION, ctx.agent.activity.snapshot(), summary="activity snapshot", source_type="runtime_state"))
    register("get_fault_summary", "Return active/stale fault summary.", ToolCategory.DIAGNOSTIC, EmptyArgs, lambda _args, ctx: _ok("get_fault_summary", ToolCategory.DIAGNOSTIC, {"active_faults": ctx.robot_state.active_faults, "stale_sensor_topics": ctx.robot_state.stale_sensor_topics, "sensor_timestamps": ctx.robot_state.sensor_timestamps}, summary=f"{len(ctx.robot_state.active_faults)} active faults", source_type="robot_state"), profiles=["diagnostic", "maintenance", "manipulation"])

    def _search_docs(args: BaseModel, ctx: ToolContext) -> ToolResult:
        parsed: QueryArgs = args  # type: ignore[assignment]
        refs = ctx.agent.document_rag.search(parsed.query, top_k=parsed.limit) if ctx.agent.document_rag is not None else []
        return _ok("search_official_docs", ToolCategory.KNOWLEDGE, [ref.model_dump() for ref in refs], summary=f"{len(refs)} documentary refs", source_type="documentary_rag", provenance={"source_type": "UNITREE_OFFICIAL"})

    register("search_official_docs", "Search official/static documentary RAG.", ToolCategory.KNOWLEDGE, QueryArgs, _search_docs)

    def _inspect_sdk(args: BaseModel, ctx: ToolContext) -> ToolResult:
        parsed: QueryArgs = args  # type: ignore[assignment]
        refs = ctx.agent.sdk_knowledge.search(parsed.query, top_k=parsed.limit) if ctx.agent.sdk_knowledge is not None else []
        return _ok("inspect_sdk_wrapper", ToolCategory.KNOWLEDGE, [ref.model_dump() for ref in refs], summary=f"{len(refs)} sdk_wrapper refs", source_type="implementation_source", provenance={"source_type": "SDK_WRAPPER", "authoritative_physical_truth": False})

    register("inspect_sdk_wrapper", "Inspect sdk_wrapper_v3 implementation snippets. Important but potentially fallible.", ToolCategory.KNOWLEDGE, QueryArgs, _inspect_sdk, profiles=["diagnostic", "maintenance", "manipulation"])

    allowed_roots = [
        Path(__file__).resolve().parents[1],
        Path(__file__).resolve().parents[4] / "dev",
        Path(__file__).resolve().parents[3],
    ]

    def _search_source(args: BaseModel, ctx: ToolContext) -> ToolResult:
        parsed: QueryArgs = args  # type: ignore[assignment]
        hits = []
        terms = [term.lower() for term in parsed.query.split() if term]
        for root in allowed_roots:
            if not root.exists():
                continue
            for path in root.rglob("*.py"):
                if len(hits) >= parsed.limit:
                    break
                if any(part.startswith(".") or part in {".env", ".ssh", "__pycache__"} for part in path.parts):
                    continue
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                lower = text.lower()
                if all(term in lower for term in terms):
                    hits.append({"file": str(path), "source_type": "implementation_source"})
            if len(hits) >= parsed.limit:
                break
        return _ok("search_source_code", ToolCategory.KNOWLEDGE, hits, summary=f"{len(hits)} source hits", source_type="implementation_source")

    register("search_source_code", "Search allowlisted project/source roots for code files.", ToolCategory.KNOWLEDGE, QueryArgs, _search_source, profiles=["diagnostic", "maintenance"])

    def _read_excerpt(args: BaseModel, ctx: ToolContext) -> ToolResult:
        parsed: SourceExcerptArgs = args  # type: ignore[assignment]
        content = _read_source_excerpt(Path(parsed.file), start_line=parsed.start_line, line_count=parsed.line_count, allowed_roots=allowed_roots)
        return _ok("read_source_excerpt", ToolCategory.KNOWLEDGE, content, summary=content["line_range"], source_type="implementation_source", provenance={"sensitive_redaction": "path allowlist enforced"})

    register("read_source_excerpt", "Read a bounded excerpt from allowlisted source files.", ToolCategory.KNOWLEDGE, SourceExcerptArgs, _read_excerpt, profiles=["diagnostic", "maintenance"])

    register("search_episodic_memory", "Search bounded episodic memory.", ToolCategory.MEMORY, QueryArgs, lambda args, ctx: _ok("search_episodic_memory", ToolCategory.MEMORY, [e.__dict__ for e in ctx.agent.memory.episodic.search(args.query, top_k=args.limit)], summary="episodic search", source_type="episodic_memory"))
    register("search_semantic_memory", "Search semantic memory.", ToolCategory.MEMORY, QueryArgs, lambda args, ctx: _ok("search_semantic_memory", ToolCategory.MEMORY, [s.__dict__ for s in ctx.agent.memory.semantic.search(args.query, top_k=args.limit)], summary="semantic search", source_type="semantic_memory"))
    register("search_procedural_memory", "Search procedural/tacit memory.", ToolCategory.MEMORY, QueryArgs, lambda args, ctx: _ok("search_procedural_memory", ToolCategory.MEMORY, [p.__dict__ for p in ctx.agent.memory.procedural.all()[:args.limit]], summary="procedural search", source_type="procedural_memory"))
    register("search_similar_failures", "Search recent failure episodes for a skill/query.", ToolCategory.MEMORY, QueryArgs, lambda args, ctx: _ok("search_similar_failures", ToolCategory.MEMORY, [e.__dict__ for e in ctx.agent.memory.episodic.search(args.query + ' failure', top_k=args.limit)], summary="failure search", source_type="episodic_memory"), profiles=["diagnostic", "maintenance", "manipulation"])

    register("get_settings", "Return redacted effective settings.", ToolCategory.CONFIGURATION, EmptyArgs, lambda _args, ctx: _ok("get_settings", ToolCategory.CONFIGURATION, {k: ("[redacted]" if any(t in k.lower() for t in ("key", "token", "secret")) else v) for k, v in ctx.settings.as_flat_dict().items()}, summary="settings snapshot", source_type="configuration"))
    register("get_setting", "Return one setting path.", ToolCategory.CONFIGURATION, SettingPathArgs, lambda args, ctx: _ok("get_setting", ToolCategory.CONFIGURATION, {"path": args.path, "value": ctx.agent.settings.get(args.path) if args.path else None}, summary=args.path or "setting", source_type="configuration"))
    register("get_capabilities", "Return grounded capability summary.", ToolCategory.CONFIGURATION, EmptyArgs, lambda _args, ctx: _ok("get_capabilities", ToolCategory.CONFIGURATION, ctx.agent._capability_tool_summary(ctx.robot_state), summary="capabilities", source_type="capability_resolver"))

    register("get_self_summary", "Return compact persistent functional self-model summary.", ToolCategory.OBSERVATION, EmptyArgs, lambda _args, ctx: _ok("get_self_summary", ToolCategory.OBSERVATION, ctx.agent.self_model.summary(), summary="self-model summary", source_type="self_model"), profiles=["social", "diagnostic", "maintenance", "manipulation", "navigation"])
    register("get_body_model", "Return learned body self-model abstractions, not raw telemetry.", ToolCategory.OBSERVATION, EmptyArgs, lambda _args, ctx: _ok("get_body_model", ToolCategory.OBSERVATION, ctx.agent.self_model.model.body.model_dump(), summary="body self-model", source_type="self_model"), profiles=["diagnostic", "maintenance"])
    register("get_skill_reliability", "Return learned skill reliability records.", ToolCategory.OBSERVATION, QueryArgs, lambda args, ctx: _ok("get_skill_reliability", ToolCategory.OBSERVATION, dict(list({name: record.model_dump() for name, record in ctx.agent.self_model.model.skills.records.items() if not args.query or args.query.lower() in name.lower()}.items())[:args.limit]), summary="skill reliability", source_type="self_model"), profiles=["social", "diagnostic", "maintenance", "manipulation"])
    register("get_energy_estimate", "Estimate robot-specific energy cost for a task/action.", ToolCategory.OBSERVATION, QueryArgs, lambda args, ctx: _ok("get_energy_estimate", ToolCategory.OBSERVATION, {"task": args.query, "estimated_cost_pct": ctx.agent.self_model.estimate_energy_cost(args.query, state=ctx.robot_state), "reserve_after_pct": ctx.agent.self_model.estimate_reserve_after(args.query, state=ctx.robot_state)}, summary=f"energy estimate for {args.query}", source_type="self_model"), profiles=["social", "diagnostic", "maintenance", "navigation"])
    register("get_capability_estimate", "Return learned capability estimate from the persistent self-model.", ToolCategory.OBSERVATION, QueryArgs, lambda args, ctx: _ok("get_capability_estimate", ToolCategory.OBSERVATION, dict(list({name: est.model_dump() for name, est in ctx.agent.self_model.model.capabilities.estimates.items() if not args.query or args.query.lower() in name.lower()}.items())[:args.limit]), summary="learned capability estimate", source_type="self_model"), profiles=["social", "diagnostic", "maintenance", "manipulation"])
    register("evaluate_candidate_action", "Predict success/energy/risk for one candidate action from self-model statistics.", ToolCategory.OBSERVATION, CandidateActionArgs, lambda args, ctx: _ok("evaluate_candidate_action", ToolCategory.OBSERVATION, ctx.agent.self_model.predict(candidate_action=args.action, state=ctx.robot_state).model_dump(), summary=f"prediction for {args.action}", source_type="self_model"), profiles=["social", "diagnostic", "maintenance", "manipulation", "navigation"])
    register("get_active_commitments", "Return active persistent commitments.", ToolCategory.OBSERVATION, EmptyArgs, lambda _args, ctx: _ok("get_active_commitments", ToolCategory.OBSERVATION, [item.model_dump() for item in ctx.agent.self_model.model.commitments.commitments if item.state == "active"], summary="active commitments", source_type="self_model"))
    register("get_learned_preferences", "Return learned preferences from functional self-model.", ToolCategory.OBSERVATION, EmptyArgs, lambda _args, ctx: _ok("get_learned_preferences", ToolCategory.OBSERVATION, [item.model_dump() for item in ctx.agent.self_model.model.preferences.preferences], summary="learned preferences", source_type="self_model"))

    register("get_topic_health", "Return semantic ROS/DDS topic health from navigation/state adapters.", ToolCategory.DIAGNOSTIC, EmptyArgs, lambda _args, ctx: _ok("get_topic_health", ToolCategory.DIAGNOSTIC, ctx.agent.navigation.snapshot().as_dict().get("topics", {}), summary="topic health", source_type="diagnostic"))
    register("get_recent_errors", "Return recent monitor error/failure events.", ToolCategory.DIAGNOSTIC, OptionalLimitArgs, lambda args, ctx: _ok("get_recent_errors", ToolCategory.DIAGNOSTIC, [e.model_dump() for e in ctx.agent.monitor_bus.recent(100) if "error" in e.event or "failed" in e.event][-args.limit:], summary="recent errors", source_type="monitor"))
    register("get_disk_status", "Return disk and memory store status.", ToolCategory.DIAGNOSTIC, EmptyArgs, lambda _args, ctx: _ok("get_disk_status", ToolCategory.DIAGNOSTIC, ctx.agent.learning.disk_stats(settings=ctx.settings), summary="disk status", source_type="diagnostic"))
    register("get_memory_store_status", "Return memory store counts/quotas.", ToolCategory.DIAGNOSTIC, EmptyArgs, lambda _args, ctx: _ok("get_memory_store_status", ToolCategory.DIAGNOSTIC, ctx.agent.learning.memory_stats(settings=ctx.settings), summary="memory status", source_type="diagnostic"))

    def _empirical(args: BaseModel, ctx: ToolContext) -> ToolResult:
        parsed: EmpiricalMemoryArgs = args  # type: ignore[assignment]
        claim = ctx.agent.learning.propose_empirical_memory(
            claim=parsed.claim,
            supporting_episodes=parsed.supporting_episodes,
            contradicting_episodes=parsed.contradicting_episodes,
            confidence=parsed.confidence,
            applicable_context={**parsed.applicable_context, "tool_evidence": parsed.tool_evidence},
            risk_level=parsed.risk_level,
        )
        return _ok("propose_empirical_memory", ToolCategory.LEARNING, claim.model_dump(), summary="candidate empirical memory accepted", source_type="learning_manager")

    register("propose_empirical_memory", "Propose empirical learned memory for validator.", ToolCategory.LEARNING, EmpiricalMemoryArgs, _empirical, read_only=False, profiles=["diagnostic", "maintenance", "manipulation"])

    def _procedure(args: BaseModel, ctx: ToolContext) -> ToolResult:
        parsed: ProceduralAdaptationArgs = args  # type: ignore[assignment]
        adaptation = ctx.agent.learning.propose_procedural_adaptation(
            skill=parsed.skill,
            condition={**parsed.condition, "tool_evidence": parsed.tool_evidence},
            recommended_parameters=parsed.recommended_parameters,
            confidence=parsed.confidence,
            derived_from=parsed.derived_from,
            settings=ctx.settings,
        )
        return _ok("propose_procedural_adaptation", ToolCategory.LEARNING, adaptation.__dict__, summary="procedural proposal submitted", source_type="learning_manager")

    register("propose_procedural_adaptation", "Propose procedural adaptation; validator decides persistence/promotion.", ToolCategory.LEARNING, ProceduralAdaptationArgs, _procedure, read_only=False, profiles=["diagnostic", "maintenance", "manipulation"])
    def _contradiction(args: BaseModel, ctx: ToolContext) -> ToolResult:
        parsed: MemoryContradictionArgs = args  # type: ignore[assignment]
        updated = ctx.agent.learning.report_memory_contradiction(parsed.claim_id, parsed.episode_id)
        return _ok(
            "report_memory_contradiction",
            ToolCategory.LEARNING,
            None if updated is None else updated.model_dump(),
            summary="contradiction reported" if updated is not None else "claim not found",
            source_type="learning_manager",
        )

    register("report_memory_contradiction", "Report contradiction evidence for a learned claim.", ToolCategory.LEARNING, MemoryContradictionArgs, _contradiction, read_only=False)

    def _skill_availability(skill_name: str) -> Callable[[ToolContext], ToolAvailability]:
        def _availability(ctx: ToolContext) -> ToolAvailability:
            if skill_name not in ctx.agent.skills.skills:
                return ToolAvailability.UNAVAILABLE_BACKEND
            policy = ctx.agent.resolver.resolve_skill(skill_name, settings=ctx.settings, robot_state=ctx.robot_state)
            if not policy.allowed:
                return ToolAvailability.BLOCKED_BY_STATE if policy.risk != "high" else ToolAvailability.BLOCKED_BY_SAFETY
            return ToolAvailability.AVAILABLE
        return _availability

    def _action_handler(skill_name: str) -> Callable[[BaseModel, ToolContext], ToolResult]:
        def _handler(args: BaseModel, ctx: ToolContext) -> ToolResult:
            from .skills import invoke_with_capability_check

            policy, skill_result = invoke_with_capability_check(
                ctx.agent.skills,
                ctx.agent.resolver,
                skill_name,
                settings=ctx.settings,
                robot_state=ctx.robot_state,
                **args.model_dump(),
            )
            content = {
                "status": "executed" if skill_result is not None else "denied",
                "policy": policy.__dict__,
                "result": None if skill_result is None else skill_result.__dict__,
            }
            ok = bool(skill_result and skill_result.ok)
            return ToolResult(
                ok=ok,
                tool=skill_name,
                category=ToolCategory.ACTION,
                content=content,
                summary=f"{skill_name} {content['status']}",
                source_type="validated_skill",
                provenance={"skill_registry": ctx.agent.skills.backend_label, "capability_decision": policy.reason},
                error_code=None if ok else ToolErrorCode.BLOCKED_BY_STATE,
                message=policy.reason if not ok else "",
            )
        return _handler

    for skill_name in ("announce", "wave", "face_wave", "high_wave", "thinking_motion", "explain_motion", "thanking_motion", "walk_mode", "run_mode", "request_sleep"):
        register(
            skill_name,
            f"Validated high-level action skill: {skill_name}.",
            ToolCategory.ACTION,
            SkillActionArgs,
            _action_handler(skill_name),
            read_only=False,
            risk_level="medium" if skill_name in {"walk_mode", "run_mode"} else "low",
            profiles=["social", "manipulation", "maintenance"],
            availability=_skill_availability(skill_name),
        )

    register("llctl_joint_control", "Operator-only low-level joint control surface; never exposed to autonomous planner.", ToolCategory.ACTION, SkillActionArgs, lambda _args, _ctx: registry._error_result("llctl_joint_control", ToolCategory.ACTION, ToolErrorCode.BLOCKED_BY_SAFETY, "operator-only"), read_only=False, risk_level="high", operator_only=True)
    register("llctl_ik_control", "Operator-only IK/end-effector control surface; never exposed to autonomous planner.", ToolCategory.ACTION, SkillActionArgs, lambda _args, _ctx: registry._error_result("llctl_ik_control", ToolCategory.ACTION, ToolErrorCode.BLOCKED_BY_SAFETY, "operator-only"), read_only=False, risk_level="high", operator_only=True)
    return registry
