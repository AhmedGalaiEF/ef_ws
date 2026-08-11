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

import threading
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Optional

from .models import RobotStateSnapshot


BMS_TOPICS = (
    "rt/lf/bmsstate",
    "rt/lf/agvbmsstate",
    "rt/bmsstate",
    "rt/agvbmsstate",
)


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
        self._bms_lock = threading.Lock()
        self._latest_bms: tuple[Any, str, float] | None = None
        self._bms_subscribers: list[Any] = []
        self._bms_subscribe_error: str | None = None
        self._bms_started = False
        self._state_read_lock = threading.Lock()
        self._state_read_thread: threading.Thread | None = None
        self._state_read_result: dict[str, Any] | None = None
        self._state_read_error: BaseException | None = None

    def read(self) -> RobotStateSnapshot:
        now = time.time()
        try:
            raw = self._read_raw_state_with_timeout(timeout_s=1.0)
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
        sensor_timestamps = raw.get("sensor_timestamps") or {}
        stale_topics = [name for name, is_stale in stale.items() if is_stale]
        active_faults = self._active_faults_from_stale(stale, lowstate=lowstate, raw=raw)
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
            sensor_stale=stale,
            sensor_timestamps=sensor_timestamps,
            stale_sensor_topics=stale_topics,
            source="sdk_client.Robot",
        )

    def _read_raw_state_with_timeout(self, *, timeout_s: float) -> dict[str, Any]:
        """Read sdk_client.Robot.get_robot_state without letting SDK RPCs freeze the CLI.

        In low-command developer mode the locomotion RPC used by
        sdk_client.Robot.get_fsm() can block inside the Unitree client. The
        agent only needs a best-effort semantic snapshot, so a timeout becomes
        a degraded state_read_timeout instead of a wedged REPL.
        """
        with self._state_read_lock:
            thread = self._state_read_thread
            if thread is not None and thread.is_alive():
                raise TimeoutError("previous sdk_client.Robot.get_robot_state() call is still pending")
            self._state_read_result = None
            self._state_read_error = None

            def _worker() -> None:
                try:
                    result = self._robot.get_robot_state()
                except BaseException as exc:  # stored and re-raised on caller thread
                    self._state_read_error = exc
                else:
                    self._state_read_result = result

            thread = threading.Thread(target=_worker, name="g1-agent-state-read", daemon=True)
            self._state_read_thread = thread
            thread.start()

        thread.join(timeout=max(0.05, float(timeout_s)))
        if thread.is_alive():
            raise TimeoutError("sdk_client.Robot.get_robot_state() timed out")
        if self._state_read_error is not None:
            raise self._state_read_error
        return dict(self._state_read_result or {})

    @staticmethod
    def _active_faults_from_stale(
        stale: dict[str, Any],
        *,
        lowstate: dict[str, Any] | None,
        raw: dict[str, Any],
    ) -> list[str]:
        """Convert watched-topic staleness into active faults.

        The SDK watches several alternative lidar/SLAM topic names. If one
        usable cloud or pose source is fresh, stale aliases remain available
        as diagnostics but do not degrade the robot state.
        """
        has_fresh_lidar_cloud = any(
            name.startswith("lidar_cloud") and not is_stale for name, is_stale in stale.items()
        )
        has_pose = raw.get("position") is not None or raw.get("odom_pose") is not None or raw.get("slam_pose") is not None
        faults: list[str] = []
        for name, is_stale in stale.items():
            if not is_stale:
                continue
            if name.startswith("lidar_cloud") and has_fresh_lidar_cloud:
                continue
            if name in {"lidar_map", "odom", "slam_odom"} and has_pose:
                continue
            if name == "lowstate" and lowstate is not None:
                continue
            if name in {"left_hand_state", "right_hand_state"}:
                continue
            faults.append(name)
        return faults

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
            msg = None
        lowstate_summary = self._battery_from_lowstate_msg(msg) if msg is not None else None
        if lowstate_summary and lowstate_summary.get("soc") is not None:
            return lowstate_summary

        bms_summary = self._battery_from_bms_topics()
        if bms_summary is not None and bms_summary.get("soc") is not None:
            if lowstate_summary:
                bms_summary["lowstate_power_v"] = lowstate_summary.get("power_v")
                bms_summary["lowstate_power_a"] = lowstate_summary.get("power_a")
            return bms_summary
        if lowstate_summary is not None:
            if bms_summary and bms_summary.get("error"):
                lowstate_summary["bms_error"] = bms_summary.get("error")
            return lowstate_summary
        return bms_summary

    def _battery_from_lowstate_msg(self, msg: Any) -> dict[str, Any] | None:
        bms = self._read_attr(msg, "bms_state")
        power_v = self._float_attr(msg, "power_v")
        power_a = self._float_attr(msg, "power_a")
        if bms is None and power_v is None and power_a is None:
            return None

        current_ma = self._int_attr(bms, "current") if bms is not None else None
        current_a = power_a if power_a is not None else (None if current_ma is None else current_ma / 1000.0)
        soc = self._int_attr(bms, "soc") if bms is not None else None
        return {
            "soc": soc,
            "soh": self._int_attr(bms, "soh") if bms is not None else None,
            "status": self._int_attr(bms, "status") if bms is not None else None,
            "current_ma": current_ma,
            "current_a": current_a,
            "cycle": self._int_attr(bms, "cycle") if bms is not None else None,
            "power_v": power_v,
            "power_a": power_a,
            "charging": None if current_a is None else current_a > 0.05,
            "source": "rt/lowstate.bms_state" if bms is not None else "rt/lowstate.power",
            "available_fields": self._public_fields(bms if bms is not None else msg),
        }

    def _battery_from_bms_topics(self) -> dict[str, Any] | None:
        self._ensure_bms_subscribers()
        with self._bms_lock:
            latest = self._latest_bms
            subscribe_error = self._bms_subscribe_error
        if latest is None:
            if subscribe_error:
                return {"soc": None, "charging": None, "source": "bms_topics", "error": subscribe_error}
            return None
        msg, topic, stamp = latest
        current_ma = self._int_attr(msg, "current")
        current_a = None if current_ma is None else current_ma / 1000.0
        return {
            "soc": self._int_attr(msg, "soc"),
            "soh": self._int_attr(msg, "soh"),
            "status": self._int_attr(msg, "status"),
            "current_ma": current_ma,
            "current_a": current_a,
            "cycle": self._int_attr(msg, "cycle"),
            "charging": None if current_a is None else current_a > 0.05,
            "source": topic,
            "age_s": max(0.0, time.time() - stamp),
            "available_fields": self._public_fields(msg),
        }

    def _ensure_bms_subscribers(self) -> None:
        if self._bms_started:
            return
        self._bms_started = True
        try:
            from unitree_sdk2py.core.channel import ChannelSubscriber
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import BmsState_ as GoBmsState
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import BmsState_ as HgBmsState
        except Exception as exc:
            self._bms_subscribe_error = f"BMS subscriber unavailable: {exc}"
            return
        for topic in BMS_TOPICS:
            for msg_type in (HgBmsState, GoBmsState):
                try:
                    sub = ChannelSubscriber(topic, msg_type)
                    sub.Init(self._make_bms_callback(topic), 10)
                    self._bms_subscribers.append(sub)
                except Exception as exc:
                    self._bms_subscribe_error = f"{topic}: {exc}"

    def _make_bms_callback(self, topic: str) -> Callable[[Any], None]:
        def _callback(msg: Any) -> None:
            with self._bms_lock:
                self._latest_bms = (msg, topic, time.time())

        return _callback

    @staticmethod
    def _read_attr(obj: Any, name: str) -> Any | None:
        if obj is None or not hasattr(obj, name):
            return None
        try:
            return getattr(obj, name)
        except Exception:
            return None

    @classmethod
    def _int_attr(cls, obj: Any, name: str) -> int | None:
        try:
            value = cls._read_attr(obj, name)
            return None if value is None else int(value)
        except Exception:
            return None

    @classmethod
    def _float_attr(cls, obj: Any, name: str) -> float | None:
        try:
            value = cls._read_attr(obj, name)
            return None if value is None else float(value)
        except Exception:
            return None

    @staticmethod
    def _public_fields(obj: Any) -> list[str]:
        if obj is None:
            return []
        return [name for name in dir(obj) if not name.startswith("_")][:80]


def build_robot_state(source: RobotStateSource) -> RobotStateSnapshot:
    return source.read()
