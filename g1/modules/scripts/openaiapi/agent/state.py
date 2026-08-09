"""RobotState snapshot builder (spec section 22).

Builds a *semantic* ``RobotStateSnapshot`` from ``sdk_client.Robot``'s
already-semantic accessors (``get_robot_state()``, ``get_fsm()``,
``sensors_stale()``) -- never from raw ``/lowstate`` telemetry directly,
and never at telemetry frequency. ``sdk_client.Robot`` is only imported by
callers that construct a ``SdkClientRobotStateSource``; this module itself
has no hard dependency on the Unitree SDK, so it (and everything that
consumes it) can be imported and unit-tested without hardware.

One real, documented gap in the live SDK wrapper (not invented here):
``sdk_client.Robot`` exposes no released/commandable boolean for the arms.
Rather than fabricate that hardware semantic, it stays honestly
``"unknown"`` unless a caller supplies a real value --
``arm_control_state_hint`` lets ``agent/capabilities.py`` feed in the
runtime's own best-known state, tracked from which arm skills it has
actually dispatched.
"""
from __future__ import annotations

import time
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Optional

from .models import RobotStateSnapshot


class RobotStateSource:
    """Anything that can produce one semantic snapshot on demand."""

    def read(self) -> RobotStateSnapshot:  # pragma: no cover - interface
        raise NotImplementedError


class MockRobotStateSource(RobotStateSource):
    """No hardware: a fixed, clearly-labeled snapshot. Used by tests and the offline CLI."""

    def __init__(self, **overrides: Any) -> None:
        self._overrides = overrides

    def read(self) -> RobotStateSnapshot:
        payload: dict[str, Any] = dict(
            timestamp=time.time(),
            posture="unknown",
            battery_pct=None,
            charging=None,
            stability="unknown",
            active_faults=[],
            arm_control_state="unknown",
            source="mock",
        )
        payload.update(self._overrides)
        return RobotStateSnapshot(**payload)


class SdkClientRobotStateSource(RobotStateSource):
    """Adapts ``g1/modules/sdk_client.py``'s ``Robot`` into a semantic snapshot."""

    def __init__(
        self,
        robot: Any,
        *,
        arm_control_state_hint: Optional[Callable[[], str]] = None,
    ) -> None:
        self._robot = robot
        self._arm_control_state_hint = arm_control_state_hint

    def read(self) -> RobotStateSnapshot:
        now = time.time()
        try:
            raw = self._robot.get_robot_state()
        except Exception as exc:  # defensive: hardware/DDS errors are not our concern here
            return RobotStateSnapshot(
                timestamp=now,
                posture="unknown",
                stability="unknown",
                active_faults=[f"state_read_error: {exc}"],
                arm_control_state="unknown",
                source="sdk_client.Robot (read failed)",
            )

        lowstate = self._read_lowstate_summary()
        battery = self._read_battery_summary()
        mode = raw.get("mode")
        is_moving = bool(raw.get("is_moving"))
        stale = raw.get("sensor_stale") or {}
        active_faults = [name for name, is_stale in stale.items() if is_stale]
        if lowstate is None and "lowstate" not in active_faults:
            active_faults.append("lowstate")

        posture = "moving" if is_moving else "standing"
        if mode is not None:
            try:
                if int(mode) in (0, 1):  # zero_torque / damp, per sdk_wrapper_v3.FSM_IDS
                    posture = "damped_or_zero_torque"
            except (TypeError, ValueError):
                pass

        arm_control_state = "unknown"
        if self._arm_control_state_hint is not None:
            try:
                arm_control_state = self._arm_control_state_hint()
            except Exception:
                arm_control_state = "unknown"

        return RobotStateSnapshot(
            timestamp=now,
            posture=posture,
            battery_pct=battery.get("soc") if battery else None,
            charging=battery.get("charging") if battery else None,
            stability="stable" if not active_faults else "degraded",
            active_faults=active_faults,
            arm_control_state=arm_control_state,
            lowstate=lowstate,
            battery=battery,
            source="sdk_client.Robot",
        )

    def _read_lowstate_summary(self) -> dict[str, Any] | None:
        """Attach semantic lowstate telemetry to every cognition snapshot.

        The planner can inspect joint/IMU state, but still cannot emit raw
        servo commands because PlannerDecision has no such field.
        """
        try:
            snap = self._robot.get_low_state_snapshot()
        except Exception:
            return None
        if snap is None:
            return None
        if is_dataclass(snap):
            data = asdict(snap)
        else:
            data = {
                "stamp": getattr(snap, "stamp", None),
                "joint_positions": getattr(snap, "joint_positions", []),
                "joint_velocities": getattr(snap, "joint_velocities", []),
                "joint_torques": getattr(snap, "joint_torques", []),
                "imu_rpy": getattr(snap, "imu_rpy", None),
                "imu_gyro": getattr(snap, "imu_gyro", None),
                "imu_acc": getattr(snap, "imu_acc", None),
            }
        return {
            "timestamp": data.get("stamp"),
            "joint_count": len(data.get("joint_positions") or []),
            "joint_positions": data.get("joint_positions") or [],
            "joint_velocities": data.get("joint_velocities") or [],
            "joint_torques": data.get("joint_torques") or [],
            "imu": {
                "rpy": data.get("imu_rpy"),
                "gyro": data.get("imu_gyro"),
                "acc": data.get("imu_acc"),
            },
            "source": "rt/lowstate",
        }

    def _read_battery_summary(self) -> dict[str, Any] | None:
        try:
            msg = self._robot.get_low_state_msg()
        except Exception:
            return None
        if msg is None:
            return None
        bms = getattr(msg, "bms_state", None)
        if bms is None:
            return None

        def _int_attr(name: str) -> int | None:
            try:
                value = getattr(bms, name)
            except Exception:
                return None
            try:
                return int(value)
            except Exception:
                return None

        current_ma = _int_attr("current")
        return {
            "soc": _int_attr("soc"),
            "soh": _int_attr("soh"),
            "status": _int_attr("status"),
            "current_ma": current_ma,
            "cycle": _int_attr("cycle"),
            "charging": None if current_ma is None else current_ma > 0,
            "source": "rt/lowstate.bms_state",
        }


def build_robot_state(source: RobotStateSource) -> RobotStateSnapshot:
    return source.read()
