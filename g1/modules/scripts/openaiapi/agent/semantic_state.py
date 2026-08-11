"""Canonical semantic robot-state abstraction.

High-frequency values such as joint arrays stay in diagnostics. This
module creates the compact cognitive state used by planner context,
memory episodes, attention decisions and the monitor.
"""
from __future__ import annotations

from typing import Any, List, Tuple
import time

from pydantic import BaseModel, ConfigDict, Field


class SemanticState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: float = Field(default_factory=time.time)
    battery: str = "unavailable"
    posture: str = "unknown"
    balance: str = "unknown"
    arm_state: str = "unknown"
    task: str = "idle"
    interaction: str = "alone"
    thermal: str = "unknown"
    lifecycle: str = "awake"
    active_faults: List[str] = Field(default_factory=list)
    source: str = "semantic_state"

    def summary(self) -> str:
        faults = ",".join(self.active_faults[:4]) if self.active_faults else "none"
        return (
            f"posture={self.posture}; balance={self.balance}; battery={self.battery}; "
            f"arm={self.arm_state}; task={self.task}; interaction={self.interaction}; faults={faults}"
        )


class SemanticStateTracker:
    def __init__(self) -> None:
        self.current = SemanticState()

    def update(
        self,
        robot_state: Any,
        *,
        lifecycle: str = "awake",
        interaction: str = "alone",
        task: str = "idle",
    ) -> Tuple[SemanticState, List[str]]:
        previous = self.current
        state = SemanticState(
            battery=self._battery_bucket(robot_state),
            posture=self._posture_bucket(robot_state),
            balance=self._balance_bucket(robot_state),
            arm_state=self._arm_bucket(robot_state),
            task=task,
            interaction=interaction,
            thermal=self._thermal_bucket(robot_state),
            lifecycle=lifecycle,
            active_faults=list(getattr(robot_state, "active_faults", []) or []),
        )
        changes = self._changes(previous, state)
        self.current = state
        return state, changes

    @staticmethod
    def _battery_bucket(robot_state: Any) -> str:
        charging = getattr(robot_state, "charging", None)
        if charging:
            return "charging"
        pct = getattr(robot_state, "battery_pct", None)
        if pct is None:
            return "unavailable"
        try:
            value = float(pct)
        except Exception:
            return "unavailable"
        if value < 10:
            return "critical"
        if value < 25:
            return "low"
        return "normal"

    @staticmethod
    def _posture_bucket(robot_state: Any) -> str:
        posture = str(getattr(robot_state, "posture", "") or "unknown").lower()
        if any(term in posture for term in ("stand", "walking", "walk")):
            return "standing" if "walk" not in posture else "walking"
        if "sit" in posture:
            return "sitting"
        if "damp" in posture or "zero" in posture:
            return "released"
        return posture or "unknown"

    @staticmethod
    def _balance_bucket(robot_state: Any) -> str:
        stability = str(getattr(robot_state, "stability", "") or "unknown").lower()
        if stability in {"stable", "nominal", "ok"}:
            return "stable"
        if "recover" in stability:
            return "recovery"
        if stability and stability != "unknown":
            return "degraded"
        return "unknown"

    @staticmethod
    def _arm_bucket(robot_state: Any) -> str:
        faults = set(getattr(robot_state, "active_faults", []) or [])
        arm_faults = {fault for fault in faults if fault in {"lowstate", "arm_sdk", "arm_control"}}
        state = str(getattr(robot_state, "arm_control_state", "") or "unknown").lower()
        if arm_faults:
            return "faulted"
        if "release" in state or "damp" in state or "zero" in state:
            return "released"
        if state in {"unknown", ""}:
            return "unknown"
        return "commandable"

    @staticmethod
    def _thermal_bucket(robot_state: Any) -> str:
        lowstate = getattr(robot_state, "lowstate", None) or {}
        temps = lowstate.get("motor_temperatures") or lowstate.get("motorTemp") or []
        try:
            max_temp = max(float(temp) for temp in temps) if temps else None
        except Exception:
            max_temp = None
        if max_temp is None:
            return "unknown"
        if max_temp >= 75:
            return "unsafe"
        if max_temp >= 60:
            return "elevated"
        return "nominal"

    @staticmethod
    def _changes(previous: SemanticState, current: SemanticState) -> List[str]:
        changes: List[str] = []
        fields = ("battery", "posture", "balance", "arm_state", "task", "interaction", "thermal", "lifecycle")
        for field_name in fields:
            before = getattr(previous, field_name)
            after = getattr(current, field_name)
            if before != after:
                changes.append(f"{field_name}:{before}->{after}")
        old_faults = set(previous.active_faults)
        new_faults = set(current.active_faults)
        if old_faults != new_faults:
            changes.append(f"faults:+{sorted(new_faults - old_faults)} -{sorted(old_faults - new_faults)}")
        return changes
