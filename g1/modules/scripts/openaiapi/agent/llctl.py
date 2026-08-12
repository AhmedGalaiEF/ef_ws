"""Operator-only low-level control adapter.

This wraps the newest dashboard implementation instead of exposing raw
controller packets to the cognitive planner.
"""
from __future__ import annotations

import sys
import math
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
    selected_joint: Optional[int] = None
    selected_backend: str = "arm_sdk"
    last_command: str = ""
    last_result: str = ""

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        return {
            "session_enabled": self.session_enabled,
            "backend_available": self.backend_available,
            "backend_error": self.backend_error,
            "dashboard_path": self.dashboard_path,
            "control_backend": self.control_backend,
            "selected_joint": self.selected_joint,
            "selected_backend": self.selected_backend,
            "last_command": self.last_command,
            "last_result": self.last_result,
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
            self._robot_link = getattr(dashboard, "ROBOT_LINK", None)
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
        if self._robot_link is not None:
            try:
                self._robot_link.connect()
            except Exception as exc:
                self.state.backend_error = str(exc)
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
            link_snapshot = self._link_snapshot()
            snap.update(link_snapshot)
            snap["dashboard_features"] = [
                "low-level joint control",
                "IK end-effector control",
                "dashboard safety validation",
            ]
        return snap

    def _link_snapshot(self) -> dict[str, Any]:
        if self._robot_link is None:
            return {}
        try:
            snap = self._robot_link.snapshot()
        except Exception as exc:
            self.state.backend_error = str(exc)
            return {}
        connected = bool(snap.get("connected"))
        selected = self.state.selected_joint
        defaults = None
        if selected is not None:
            try:
                raw = self._robot_link.joint_modal_defaults(int(selected))
                spec = raw.get("spec")
                defaults = {
                    "id": getattr(spec, "id", selected),
                    "name": getattr(spec, "name", str(selected)),
                    "group": getattr(spec, "group", ""),
                    "q_min": getattr(spec, "q_min", None),
                    "q_max": getattr(spec, "q_max", None),
                    "sensed_q": raw.get("sensed_q"),
                    "q": raw.get("q"),
                    "dq": raw.get("dq"),
                    "kp": raw.get("kp"),
                    "kd": raw.get("kd"),
                    "tau": raw.get("tau"),
                    "ramp_s": raw.get("ramp_s"),
                    "locked": raw.get("locked"),
                }
            except Exception as exc:
                defaults = {"error": str(exc)}
        return {
            "connected": connected,
            "dev_mode": bool(snap.get("dev_mode")),
            "arm_engaged": bool(snap.get("arm_engaged")),
            "arm_weight": snap.get("arm_weight"),
            "service_row": snap.get("service_row"),
            "selected_joint_defaults": defaults,
        }

    def _joint_id(self, joint: str) -> int:
        self.ensure_backend()
        text = str(joint).strip()
        if not text:
            raise ValueError("joint name/id is empty")
        try:
            return int(text)
        except ValueError:
            pass
        module = self._dashboard_module
        table = getattr(module, "JOINT_TABLE", []) if module is not None else []
        normalized = text.lower().replace("-", "_").replace(" ", "_")
        for spec in table:
            if str(getattr(spec, "name", "")).lower() == normalized:
                return int(getattr(spec, "id"))
        raise ValueError(f"unknown joint {joint!r}")

    def select_joint(self, settings: Any, *, joint: str) -> str:
        allowed, reason = self.check_session(settings)
        if not allowed:
            return reason
        try:
            joint_id = self._joint_id(joint)
        except Exception as exc:
            return str(exc)
        self.state.selected_joint = joint_id
        self.state.last_activity_at = time.time()
        self.state.last_command = f"select_joint {joint_id}"
        self.state.last_result = f"selected joint {joint_id}"
        return self.state.last_result

    def set_backend(self, settings: Any, *, backend: str) -> str:
        allowed, reason = self.check_session(settings)
        if not allowed:
            return reason
        if not self.ensure_backend() or self._robot_link is None:
            return self.state.backend_error or "llctl backend unavailable"
        backend = str(backend).strip().lower()
        if backend not in {"arm_sdk", "lowcmd"}:
            return "backend must be arm_sdk or lowcmd"
        try:
            snap = self._robot_link.snapshot()
            dev_mode = bool(snap.get("dev_mode"))
            if backend == "lowcmd" and not dev_mode:
                ok, message = self._robot_link.toggle_dev_mode()
                if not ok:
                    return message
            elif backend == "arm_sdk" and dev_mode:
                ok, message = self._robot_link.toggle_dev_mode()
                if not ok:
                    return message
        except Exception as exc:
            return f"backend switch failed: {exc}"
        self.state.selected_backend = backend
        self.state.last_activity_at = time.time()
        self.state.last_command = f"backend {backend}"
        self.state.last_result = f"backend set to {backend}"
        return self.state.last_result

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
        try:
            target = float(q)
        except (TypeError, ValueError):
            return False, "joint target must be a finite number"
        if not math.isfinite(target):
            return False, "joint target must be a finite number"
        if not -6.3 <= target <= 6.3:
            return False, "joint target outside conservative [-6.3, +6.3] rad range"
        self.state.last_activity_at = time.time()
        return True, "validated by llctl front-end; dashboard controller still performs final safety checks"

    def command_joint(
        self,
        settings: Any,
        *,
        joint: str,
        q: float,
        dq: float = 0.0,
        kp: float = 30.0,
        kd: float = 1.5,
        tau: float = 0.0,
        ramp_s: float = 0.6,
        backend: str = "arm_sdk",
    ) -> str:
        allowed, reason = self.check_session(settings)
        if not allowed:
            return reason
        if not settings.llctl.allow_joint_control:
            return "llctl.allow_joint_control=false"
        if not self.ensure_backend() or self._robot_link is None:
            return self.state.backend_error or "llctl backend unavailable"
        try:
            joint_id = self._joint_id(joint)
        except Exception as exc:
            return str(exc)
        backend_result = self.set_backend(settings, backend=backend)
        if "failed" in backend_result.lower() or "must be" in backend_result.lower() or "unavailable" in backend_result.lower():
            return backend_result
        try:
            ok, message = self._robot_link.set_joint_target(
                int(joint_id),
                float(q),
                float(dq),
                float(kp),
                float(kd),
                float(tau),
                float(ramp_s),
            )
        except Exception as exc:
            ok, message = False, f"joint command failed: {exc}"
        self.state.selected_joint = int(joint_id)
        self.state.last_activity_at = time.time()
        self.state.last_command = (
            f"joint {joint_id} q={q} dq={dq} kp={kp} kd={kd} tau={tau} ramp={ramp_s} backend={backend}"
        )
        self.state.last_result = message
        return ("ok: " if ok else "failed: ") + message

    def validate_ee_command(self, settings: Any, *, side: str, x: float, y: float, z: float) -> tuple[bool, str]:
        allowed, reason = self.check_session(settings)
        if not allowed:
            return False, reason
        if not settings.llctl.allow_ik_control:
            return False, "llctl.allow_ik_control=false"
        if str(side).lower() not in {"left", "right"}:
            return False, "side must be left or right"
        try:
            target = tuple(float(value) for value in (x, y, z))
        except (TypeError, ValueError):
            return False, "EE target must contain finite numbers"
        if not all(math.isfinite(value) for value in target):
            return False, "EE target must contain finite numbers"
        if not (-1.0 <= target[0] <= 1.0 and -1.0 <= target[1] <= 1.0 and -0.5 <= target[2] <= 1.5):
            return False, "EE target outside conservative workspace preview bounds"
        self.state.last_activity_at = time.time()
        return True, "validated by llctl front-end; dashboard IK/safety path remains final authority"

    def command_ee_delta(
        self,
        settings: Any,
        *,
        side: str,
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 0.0,
        droll: float = 0.0,
        dpitch: float = 0.0,
        dyaw: float = 0.0,
    ) -> str:
        allowed, reason = self.check_session(settings)
        if not allowed:
            return reason
        if not settings.llctl.allow_ik_control:
            return "llctl.allow_ik_control=false"
        if not self.ensure_backend() or self._robot_link is None:
            return self.state.backend_error or "llctl backend unavailable"
        side = str(side).lower()
        if side not in {"left", "right"}:
            return "side must be left or right"
        try:
            pose = self._robot_link.ee_pose_snapshot(side)
            x = float(pose["x"]) + float(dx)
            y = float(pose["y"]) + float(dy)
            z = float(pose["z"]) + float(dz)
            roll = float(pose["roll"]) + float(droll)
            pitch = float(pose["pitch"]) + float(dpitch)
            yaw = float(pose["yaw"]) + float(dyaw)
            valid, validation = self.validate_ee_command(settings, side=side, x=x, y=y, z=z)
            if not valid:
                return validation
            ok, message, _info = self._robot_link.set_arm_ee_target(side, x, y, z, roll, pitch, yaw)
        except Exception as exc:
            ok, message = False, f"EE command failed: {exc}"
        self.state.last_activity_at = time.time()
        self.state.last_command = f"ee {side} dx={dx} dy={dy} dz={dz} droll={droll} dpitch={dpitch} dyaw={dyaw}"
        self.state.last_result = message
        return ("ok: " if ok else "failed: ") + message

    def command_ee_target(
        self,
        settings: Any,
        *,
        side: str,
        x: float,
        y: float,
        z: float,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
    ) -> str:
        allowed, reason = self.check_session(settings)
        if not allowed:
            return reason
        if not settings.llctl.allow_ik_control:
            return "llctl.allow_ik_control=false"
        if not self.ensure_backend() or self._robot_link is None:
            return self.state.backend_error or "llctl backend unavailable"
        side = str(side).lower()
        if side not in {"left", "right"}:
            return "side must be left or right"
        valid, validation = self.validate_ee_command(settings, side=side, x=x, y=y, z=z)
        if not valid:
            return validation
        try:
            ok, message, _info = self._robot_link.set_arm_ee_target(
                side,
                float(x),
                float(y),
                float(z),
                float(roll),
                float(pitch),
                float(yaw),
            )
        except Exception as exc:
            ok, message = False, f"EE command failed: {exc}"
        self.state.last_activity_at = time.time()
        self.state.last_command = f"ee_target {side} x={x} y={y} z={z} roll={roll} pitch={pitch} yaw={yaw}"
        self.state.last_result = message
        return ("ok: " if ok else "failed: ") + message

    def release_arms(self, settings: Any) -> str:
        allowed, reason = self.check_session(settings)
        if not allowed:
            return reason
        if not self.ensure_backend() or self._robot_link is None:
            return self.state.backend_error or "llctl backend unavailable"
        try:
            ok, message = self._robot_link.release_arms()
        except Exception as exc:
            ok, message = False, f"release_arms failed: {exc}"
        self.state.last_activity_at = time.time()
        self.state.last_command = "release_arms"
        self.state.last_result = message
        return ("ok: " if ok else "failed: ") + message
