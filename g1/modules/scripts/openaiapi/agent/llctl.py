"""Operator-only low-level control adapter.

This wraps the newest dashboard implementation instead of exposing raw
controller packets to the cognitive planner.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


DASHBOARD_PATH = Path("/home/unitree/EF/ef_ws/g1/dev/dashboards/joint_control_dashboard.py")


@dataclass
class LlctlRuntimeState:
    session_enabled: bool = False
    enabled_at: Optional[float] = None
    last_activity_at: Optional[float] = None
    backend_available: bool = False
    backend_error: str = ""
    dashboard_path: str = str(DASHBOARD_PATH)
    control_backend: str = "joint_control_dashboard.RobotLink"

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        return {
            "session_enabled": self.session_enabled,
            "backend_available": self.backend_available,
            "backend_error": self.backend_error,
            "dashboard_path": self.dashboard_path,
            "control_backend": self.control_backend,
            "idle_s": None if self.last_activity_at is None else max(0.0, now - self.last_activity_at),
        }


class LlctlAdapter:
    def __init__(self) -> None:
        self.state = LlctlRuntimeState()
        self._robot_link: Any = None
        self._dashboard_module: Any = None

    def ensure_backend(self) -> bool:
        if self._dashboard_module is not None:
            self.state.backend_available = True
            return True
        try:
            dashboards_dir = DASHBOARD_PATH.parent
            if str(dashboards_dir) not in sys.path:
                sys.path.insert(0, str(dashboards_dir))
            import joint_control_dashboard as dashboard

            self._dashboard_module = dashboard
            self.state.backend_available = True
            self.state.backend_error = ""
            return True
        except Exception as exc:
            self.state.backend_available = False
            self.state.backend_error = str(exc)
            return False

    def enable_session(self, settings: Any) -> str:
        if not settings.llctl.enabled:
            return "llctl.enabled=false"
        if not self.ensure_backend():
            return f"llctl backend unavailable: {self.state.backend_error}"
        self.state.session_enabled = True
        self.state.enabled_at = time.time()
        self.state.last_activity_at = self.state.enabled_at
        return "llctl session enabled"

    def disable_session(self) -> str:
        self.state.session_enabled = False
        self.state.last_activity_at = time.time()
        return "llctl session disabled"

    def check_session(self, settings: Any) -> tuple[bool, str]:
        if not settings.llctl.enabled:
            return False, "llctl.enabled=false"
        if settings.llctl.require_explicit_enable_each_session and not self.state.session_enabled:
            return False, "run /llctl enable before manual control commands"
        if self.state.last_activity_at is not None:
            idle = time.time() - self.state.last_activity_at
            if idle > float(settings.llctl.session_timeout_s):
                self.state.session_enabled = False
                return False, "llctl session timed out"
        return True, "allowed"

    def snapshot(self, settings: Any = None) -> dict[str, Any]:
        self.ensure_backend()
        snap = self.state.snapshot()
        if settings is not None:
            allowed, reason = self.check_session(settings)
            snap["manual_commands_allowed"] = allowed
            snap["permission_reason"] = reason
            snap["allow_joint_control"] = bool(settings.llctl.allow_joint_control)
            snap["allow_ik_control"] = bool(settings.llctl.allow_ik_control)
        if self._dashboard_module is not None:
            snap["dashboard_features"] = [
                "low-level joint control",
                "IK end-effector control",
                "dashboard safety validation",
            ]
        return snap

    def validate_joint_command(self, settings: Any, *, joint: str, q: float) -> tuple[bool, str]:
        allowed, reason = self.check_session(settings)
        if not allowed:
            return False, reason
        if not settings.llctl.allow_joint_control:
            return False, "llctl.allow_joint_control=false"
        if not self.ensure_backend():
            return False, self.state.backend_error
        joint_name = str(joint).strip()
        if not joint_name:
            return False, "joint name is empty"
        if not -6.3 <= float(q) <= 6.3:
            return False, "joint target outside conservative [-6.3, +6.3] rad range"
        self.state.last_activity_at = time.time()
        return True, "validated by llctl front-end; dashboard controller still performs final safety checks"

    def validate_ee_command(self, settings: Any, *, side: str, x: float, y: float, z: float) -> tuple[bool, str]:
        allowed, reason = self.check_session(settings)
        if not allowed:
            return False, reason
        if not settings.llctl.allow_ik_control:
            return False, "llctl.allow_ik_control=false"
        if str(side).lower() not in {"left", "right"}:
            return False, "side must be left or right"
        if not (-1.0 <= float(x) <= 1.0 and -1.0 <= float(y) <= 1.0 and -0.5 <= float(z) <= 1.5):
            return False, "EE target outside conservative workspace preview bounds"
        self.state.last_activity_at = time.time()
        return True, "validated by llctl front-end; dashboard IK/safety path remains final authority"
