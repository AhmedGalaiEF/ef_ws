"""Observable agent activity and thinking-headlight management."""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .expressive_motion import ExpressiveMotionController
from .monitor import MonitorEventBus


THINKING_ACTIVITIES = {"thinking", "retrieving"}


@dataclass
class HeadlightRuntimeState:
    current_indicator: str = "none"
    color: Optional[Tuple[int, int, int]] = None
    previous_state: Optional[Dict[str, Any]] = None
    restore_count: int = 0
    last_error: str = ""
    operator_override: bool = False

    def snapshot(self) -> Dict[str, Any]:
        return {
            "current_indicator": self.current_indicator,
            "color": self.color,
            "previous_state": self.previous_state,
            "restore_count": self.restore_count,
            "last_error": self.last_error,
            "operator_override": self.operator_override,
        }


class HeadlightStateManager:
    """Best-effort headlight indicator with reference-counted restoration.

    The SDK exposes write-only ``Robot.headlight(color, intensity, duration)``;
    there is no read-back API. We therefore restore to the last state this
    manager knows about, or turn the light off when no prior managed state
    exists.
    """

    def __init__(self, *, robot: Any = None, monitor: Optional[MonitorEventBus] = None) -> None:
        self.robot = robot
        self.monitor = monitor
        self.state = HeadlightRuntimeState()
        self._depth = 0
        self._managed_state: Optional[Dict[str, Any]] = None

    def begin_thinking(self, settings: Any, *, activity: str) -> None:
        if not self._enabled(settings):
            return
        if self.state.operator_override:
            self._emit("headlight_skipped", "operator override active", activity=activity)
            return
        self._depth += 1
        if self._depth > 1:
            return
        self.state.previous_state = self._managed_state.copy() if self._managed_state else None
        rgb = (
            int(settings.headlight.thinking.color.r),
            int(settings.headlight.thinking.color.g),
            int(settings.headlight.thinking.color.b),
        )
        intensity = int(settings.headlight.thinking.intensity)
        try:
            self._set_headlight(rgb, intensity=intensity, duration=None)
            self._managed_state = {"color": rgb, "intensity": intensity}
            self.state.current_indicator = "dark purple"
            self.state.color = rgb
            self.state.last_error = ""
            self._emit("headlight_indicator_started", "dark purple thinking indicator", activity=activity, color=rgb)
        except Exception as exc:
            self.state.last_error = str(exc)
            self._emit("headlight_indicator_failed", str(exc), activity=activity)

    def end_thinking(self, settings: Any, *, activity: str) -> None:
        if self._depth <= 0:
            return
        self._depth -= 1
        if self._depth > 0:
            return
        if not self._enabled(settings):
            return
        if not settings.headlight.restore_previous_state:
            return
        previous = self.state.previous_state
        try:
            if previous:
                self._set_headlight(previous["color"], intensity=int(previous.get("intensity", 100)), duration=None)
                self._managed_state = dict(previous)
                self.state.current_indicator = "restored"
                self.state.color = tuple(previous["color"])
            else:
                self._set_headlight((0, 0, 0), intensity=100, duration=None)
                self._managed_state = None
                self.state.current_indicator = "none"
                self.state.color = None
            self.state.restore_count += 1
            self._emit("headlight_indicator_restored", "restored previous headlight state", activity=activity)
        except Exception as exc:
            self.state.last_error = str(exc)
            self._emit("headlight_restore_failed", str(exc), activity=activity)
        finally:
            self.state.previous_state = None

    def set_operator_override(self, enabled: bool) -> None:
        self.state.operator_override = bool(enabled)

    def snapshot(self) -> Dict[str, Any]:
        data = self.state.snapshot()
        data["active_depth"] = self._depth
        return data

    def _enabled(self, settings: Any) -> bool:
        return bool(
            self.robot is not None
            and hasattr(self.robot, "headlight")
            and settings.headlight.cognitive_indicators_enabled
            and settings.headlight.thinking.enabled
        )

    def _set_headlight(self, color: Tuple[int, int, int], *, intensity: int, duration: Any) -> None:
        if self.robot is None or not hasattr(self.robot, "headlight"):
            return
        code = self.robot.headlight(color=color, intensity=intensity, duration=duration)
        if int(code or 0) != 0:
            raise RuntimeError(f"Robot.headlight returned code {code}")

    def _emit(self, event: str, summary: str, **metadata: Any) -> None:
        if self.monitor is not None:
            self.monitor.emit("activity", event, summary, metadata=metadata)


@dataclass
class ActivityRuntimeState:
    current_activity: str = "idle"
    since: float = field(default_factory=time.time)
    stack: List[str] = field(default_factory=list)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "current_activity": self.current_activity,
            "since": self.since,
            "age_s": max(0.0, time.time() - self.since),
            "stack": list(self.stack),
        }


class ActivityManager:
    def __init__(
        self,
        *,
        robot: Any = None,
        monitor: Optional[MonitorEventBus] = None,
        expressive_motion: Optional[ExpressiveMotionController] = None,
    ) -> None:
        self.monitor = monitor
        self.expressive_motion = expressive_motion
        self.headlight = HeadlightStateManager(robot=robot, monitor=monitor)
        self.state = ActivityRuntimeState()

    @contextmanager
    def activity(self, name: str, *, settings: Any, reason: str = "") -> Iterator[None]:
        previous = self.state.current_activity
        self.state.stack.append(name)
        self.state.current_activity = name
        self.state.since = time.time()
        self._emit("activity_started", f"{name} started", activity=name, reason=reason)
        if name in THINKING_ACTIVITIES:
            self.headlight.begin_thinking(settings, activity=name)
            if name == "thinking" and self.expressive_motion is not None:
                self.expressive_motion.run_background("thinking", settings=settings, reason=reason or "activity_started")
        try:
            yield
        finally:
            if name in THINKING_ACTIVITIES:
                self.headlight.end_thinking(settings, activity=name)
            if self.state.stack:
                self.state.stack.pop()
            self.state.current_activity = self.state.stack[-1] if self.state.stack else previous
            if not self.state.stack and self.state.current_activity == previous:
                self.state.current_activity = "idle" if previous in THINKING_ACTIVITIES else previous
            self.state.since = time.time()
            self._emit("activity_completed", f"{name} completed", activity=name, reason=reason)

    def set(self, name: str, *, reason: str = "") -> None:
        self.state.current_activity = name
        self.state.since = time.time()
        self._emit("activity_changed", f"activity={name}", activity=name, reason=reason)

    def snapshot(self) -> Dict[str, Any]:
        data = self.state.snapshot()
        data["headlight"] = self.headlight.snapshot()
        return data

    def _emit(self, event: str, summary: str, **metadata: Any) -> None:
        if self.monitor is not None:
            self.monitor.emit("activity", event, summary, metadata=metadata)
