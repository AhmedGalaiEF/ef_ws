"""RobotState snapshot builder (spec section 22).

Builds a *semantic* ``RobotStateSnapshot`` from ``sdk_client.Robot``'s
already-semantic accessors (``get_robot_state()``, ``get_fsm()``,
``sensors_stale()``) -- never from raw ``/lowstate`` telemetry directly,
and never at telemetry frequency. ``sdk_client.Robot`` is only imported by
callers that construct a ``SdkClientRobotStateSource``; this module itself
has no hard dependency on the Unitree SDK, so it (and everything that
consumes it) can be imported and unit-tested without hardware.

Two real, documented gaps in the live SDK wrapper (not invented here):
``sdk_client.Robot`` currently exposes no battery/BMS reading at all
(unlike ``sdk_wrapper_v3.py``'s ``G1.get_battery()``), and no
released/commandable boolean for the arms. Rather than fabricate hardware
semantics that don't exist, both fields stay honestly ``None``/``"unknown"``
unless a caller supplies a real value -- ``arm_control_state_hint`` lets
``agent/capabilities.py`` feed in the runtime's own best-known state,
tracked from which arm skills it has actually dispatched.
"""
from __future__ import annotations

import time
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

        mode = raw.get("mode")
        is_moving = bool(raw.get("is_moving"))
        stale = raw.get("sensor_stale") or {}
        active_faults = [name for name, is_stale in stale.items() if is_stale]

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
            battery_pct=None,  # TODO: sdk_client.Robot exposes no battery/BMS reading yet.
            charging=None,
            stability="stable" if not active_faults else "degraded",
            active_faults=active_faults,
            arm_control_state=arm_control_state,
            source="sdk_client.Robot",
        )


def build_robot_state(source: RobotStateSource) -> RobotStateSnapshot:
    return source.read()
