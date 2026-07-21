"""
Step 9 — Send low-level arm command
=====================================
Wraps the G1 Robot SDK arm publisher (_ArmSdkPublisher) to send
joint-space commands with trajectory interpolation and safety gating.

The arm SDK expects the 30-DOF joint array published to ``rt/arm_sdk``
with PD gains.  We build that from the 7-DOF arm solution produced by
the IK solver.

Usage:
    executor = ArmExecutor(robot, arm="right")
    executor.execute(q_arm_desired, duration_s=2.0)
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# SDK imports from parent modules directory
try:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "modules"))
    from sdk_client import Robot, LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS
except ImportError:
    Robot = None  # type: ignore
    LEFT_ARM_JOINTS = list(range(15, 22))
    RIGHT_ARM_JOINTS = list(range(22, 29))

from .arm_fk import ArmFK
from .reachability_checker import ReachabilityChecker
from .obstacle_checker import Obstacles, check_swept_path


# ---------------------------------------------------------------------------
# Default PD gains (from sdk_client._ArmSdkPublisher defaults)
# ---------------------------------------------------------------------------
_DEFAULT_KP: Dict[int, float] = {}   # use arm SDK defaults
_DEFAULT_KD: Dict[int, float] = {}

_KP_ARM = 60.0   # position gain for arm joints
_KD_ARM = 2.0    # damping gain for arm joints
_BODY_JOINT_INDEX_BY_LABEL = {
    "left_arm.shoulder_pitch": 15,
    "left_arm.shoulder_roll": 16,
    "left_arm.shoulder_yaw": 17,
    "left_arm.elbow": 18,
    "left_arm.wrist_roll": 19,
    "left_arm.wrist_pitch": 20,
    "left_arm.wrist_yaw": 21,
    "right_arm.shoulder_pitch": 22,
    "right_arm.shoulder_roll": 23,
    "right_arm.shoulder_yaw": 24,
    "right_arm.elbow": 25,
    "right_arm.wrist_roll": 26,
    "right_arm.wrist_pitch": 27,
    "right_arm.wrist_yaw": 28,
}


class ArmExecutor:
    """
    Step 9: Execute an arm joint target using the Robot SDK.

    Args:
        robot:       Robot instance from sdk_client
        arm:         "left" | "right"
        kp:          proportional gain for all arm joints
        kd:          derivative gain for all arm joints
        rate_hz:     command rate during trajectory interpolation
        safety_gate: if True, refuse to send commands that fail reachability check
    """

    def __init__(
        self,
        robot,
        arm: str = "right",
        kp: float = _KP_ARM,
        kd: float = _KD_ARM,
        rate_hz: float = 50.0,
        safety_gate: bool = True,
        max_reach_m: float = 0.42,
        max_joint_step_rad: float = 0.1,
        max_joint_speed_rad_s: float = 0.15,
    ) -> None:
        self.robot = robot
        self.arm = arm
        self.kp = kp
        self.kd = kd
        self.rate_hz = rate_hz
        self.safety_gate = safety_gate
        self.max_joint_step_rad = max(0.0, float(max_joint_step_rad))
        self.max_joint_speed_rad_s = max(0.0, float(max_joint_speed_rad_s))
        self._joint_indices = LEFT_ARM_JOINTS if arm == "left" else RIGHT_ARM_JOINTS
        # URDF-exact (not the legacy DH approximation) — used for the swept-path
        # obstacle check below, where elbow/wrist accuracy matters.
        self._fk = ArmFK(arm=arm, backend="urdf")
        self._checker = ReachabilityChecker(arm=arm, max_reach_m=max_reach_m)
        self._last_command_q: Optional[np.ndarray] = None
        self._continuous_thread: Optional[threading.Thread] = None
        self._continuous_stop: Optional[threading.Event] = None
        self._target_queue: Optional["queue.Queue[np.ndarray]"] = None

    # ------------------------------------------------------------------
    def validate(
        self,
        q_arm_desired: np.ndarray,
        *,
        T_base_desired: Optional[np.ndarray] = None,
        q_arm_start: Optional[np.ndarray] = None,
        obstacles: Optional[Obstacles] = None,
    ) -> Dict:
        """Run the safety/obstacle gate without sending anything.

        Shared by execute() (one-shot burst moves) and the continuous
        streaming path (submit_target()), so both reject the same way.
        """
        if not self.safety_gate:
            return {"safe": True}
        result = self._checker.check(q_arm_desired, T_base_desired)
        if not result.safe:
            return {"safe": False, "reason": "safety_gate", "violations": result.reasons}
        if obstacles is not None and q_arm_start is not None:
            sweep = check_swept_path(self._fk, q_arm_start, q_arm_desired, obstacles)
            if not sweep.safe:
                return {"safe": False, "reason": "obstacle_gate", "violations": sweep.reasons}
        return {"safe": True}

    # ------------------------------------------------------------------
    def start_continuous(self, rate_hz: Optional[float] = None) -> None:
        """Start a background thread that republishes the latest submitted
        target at a fixed rate, ramping smoothly toward it every tick.

        Bursty execute() calls only publish while actively interpolating a
        single move, then go silent until the next call — if the caller's
        next command is delayed (perception/IK/detection taking a moment),
        rt/arm_sdk goes quiet and the low-level controller can reclaim the
        arm, producing a sudden drop followed by a snap back once publishing
        resumes. This loop never stops publishing once started: with no new
        target queued it just keeps re-sending (and holding) the last one,
        so compute-side latency no longer creates a gap on the wire.
        """
        if self._continuous_thread is not None:
            return
        seed = self._last_command_q
        if seed is None:
            seed = self._read_current_arm_q()
        if seed is None:
            raise RuntimeError("cannot start continuous execution: joint state unavailable")
        self._last_command_q = seed.copy()
        self._target_queue = queue.Queue(maxsize=1)
        self._continuous_stop = threading.Event()
        rate = float(rate_hz) if rate_hz else self.rate_hz
        thread = threading.Thread(
            target=self._continuous_loop,
            args=(self._target_queue, self._continuous_stop, max(1.0, rate)),
            daemon=True,
        )
        self._continuous_thread = thread
        thread.start()

    def stop_continuous(self) -> None:
        """Stop the continuous publish thread, if running."""
        thread = self._continuous_thread
        stop_event = self._continuous_stop
        self._continuous_thread = None
        self._continuous_stop = None
        self._target_queue = None
        if stop_event is not None:
            stop_event.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

    def submit_target(self, q_arm_desired: np.ndarray) -> None:
        """Queue a new target for the continuous execution thread.

        Non-blocking; replaces any not-yet-applied pending target so the
        thread always ramps toward the newest command rather than working
        through a backlog of stale intermediate ones.
        """
        target_queue = self._target_queue
        if target_queue is None:
            raise RuntimeError("start_continuous() was not called")
        q = np.asarray(q_arm_desired, dtype=np.float64).copy()
        try:
            target_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            target_queue.put_nowait(q)
        except queue.Full:
            pass

    def _continuous_loop(
        self,
        target_queue: "queue.Queue[np.ndarray]",
        stop_event: threading.Event,
        rate_hz: float,
    ) -> None:
        dt = 1.0 / rate_hz
        q_target = self._last_command_q.copy()
        while not stop_event.is_set():
            t0 = time.time()
            try:
                while True:
                    q_target = target_queue.get_nowait()
            except queue.Empty:
                pass
            q_cmd, _ = self._cap_command_step(q_target, dt)
            self._send_command(q_cmd)
            remaining = dt - (time.time() - t0)
            if remaining > 0:
                time.sleep(remaining)

    # ------------------------------------------------------------------
    def execute(
        self,
        q_arm_desired: np.ndarray,
        duration_s: float = 2.0,
        q_arm_start: Optional[np.ndarray] = None,
        T_base_desired: Optional[np.ndarray] = None,
        stop_event=None,
        obstacles: Optional[Obstacles] = None,
    ) -> Dict:
        """
        Interpolate from current arm pose to q_arm_desired and send commands.

        Args:
            q_arm_desired: 7-element target joint angles (radians)
            duration_s:    total move duration
            q_arm_start:   override start configuration (default: read from robot)
            T_base_desired: optional target pose for safety check context
            obstacles:      optional Obstacles (table plane, opposite-arm
                             proxy) — when given, the whole interpolated path
                             from q_arm_start to q_arm_desired is checked,
                             not just the endpoint.

        Returns:
            dict with "success", "duration_s", "steps", "final_q"
        """
        # Get start configuration first — needed by both the swept-path
        # obstacle check and the interpolation below.
        if q_arm_start is None:
            q_arm_start = self._read_current_arm_q()
            if q_arm_start is None:
                return {
                    "success": False,
                    "reason": "joint_state_unavailable",
                    "duration_s": 0.0,
                    "steps": 0,
                }
        q_arm_start = np.asarray(q_arm_start, dtype=np.float64).copy()
        q_arm_desired = np.asarray(q_arm_desired, dtype=np.float64).copy()

        # Safety check
        validation = self.validate(
            q_arm_desired,
            T_base_desired=T_base_desired,
            q_arm_start=q_arm_start,
            obstacles=obstacles,
        )
        if not validation.get("safe"):
            return {
                "success": False,
                "reason": validation.get("reason"),
                "violations": validation.get("violations"),
                "duration_s": 0.0,
                "steps": 0,
            }

        if self._last_command_q is None:
            self._last_command_q = q_arm_start.copy()

        requested_duration_s = max(0.0, float(duration_s))
        duration_s = self._duration_for_speed_limit(
            q_arm_start,
            q_arm_desired,
            requested_duration_s,
        )
        steps = max(1, int(duration_s * self.rate_hz))
        dt = duration_s / steps
        final_q = q_arm_start.copy()
        capped_samples = 0

        for i in range(steps):
            if stop_event is not None and stop_event.is_set():
                return {
                    "success": False,
                    "reason": "stopped",
                    "duration_s": duration_s,
                    "steps": i,
                    "final_q": final_q,
                }
            alpha = _smooth_step((i + 1) / steps)
            q_cmd = (1 - alpha) * q_arm_start + alpha * q_arm_desired
            q_cmd, was_capped = self._cap_command_step(q_cmd, dt)
            if was_capped:
                capped_samples += 1
            self._send_command(q_cmd)
            final_q = q_cmd
            time.sleep(dt)

        return {
            "success": True,
            "duration_s": duration_s,
            "requested_duration_s": requested_duration_s,
            "steps": steps,
            "final_q": final_q,
            "final_error_rad": float(np.max(np.abs(q_arm_desired - final_q))) if final_q.size else 0.0,
            "capped_samples": capped_samples,
        }

    # ------------------------------------------------------------------
    def execute_cartesian(
        self,
        waypoints: List[np.ndarray],
        duration_per_wp_s: float = 1.0,
    ) -> Dict:
        """
        Execute a sequence of 7-DOF joint waypoints (e.g., pre-grasp sequence).

        Each element of waypoints is a 7-element joint angle vector.
        """
        results = []
        for wp in waypoints:
            result = self.execute(wp, duration_s=duration_per_wp_s)
            results.append(result)
            if not result["success"]:
                return {"success": False, "waypoint_results": results}
        return {"success": True, "waypoint_results": results}

    # ------------------------------------------------------------------
    def stop(self) -> None:
        """Hold current position by re-sending current joint state."""
        q_cur = self._read_current_arm_q()
        if q_cur is None:
            return
        self._send_command(q_cur)

    # ------------------------------------------------------------------
    def _read_current_arm_q(self) -> Optional[np.ndarray]:
        """Read current arm joint angles from the robot."""
        try:
            js = self.robot.get_joint_states()
            if not js:
                return None
            joints = js.get("joints", {})
            q = np.zeros(30)
            seen_indices = set()
            for name, data in joints.items():
                idx = _joint_entry_index(name, data)
                if 0 <= idx < 30:
                    q[idx] = data.get("position", 0.0)
                    seen_indices.add(int(idx))
            if not all(int(idx) in seen_indices for idx in self._joint_indices):
                return None
            return q[self._joint_indices]
        except Exception:
            return None

    # ------------------------------------------------------------------
    def _cap_command_step(self, q_arm: np.ndarray, dt: float) -> Tuple[np.ndarray, bool]:
        """Hard-cap every low-level published sample against the previous one."""
        q_arm = np.asarray(q_arm, dtype=np.float64)
        caps = []
        if self.max_joint_step_rad > 0.0:
            caps.append(self.max_joint_step_rad)
        if self.max_joint_speed_rad_s > 0.0 and dt > 0.0:
            caps.append(self.max_joint_speed_rad_s * float(dt))
        if not caps:
            self._last_command_q = q_arm.copy()
            return q_arm, False
        cap = min(caps)
        if self._last_command_q is None:
            self._last_command_q = q_arm.copy()
            return q_arm, False
        delta = q_arm - self._last_command_q
        max_abs_delta = float(np.max(np.abs(delta))) if delta.size else 0.0
        if max_abs_delta <= cap:
            self._last_command_q = q_arm.copy()
            return q_arm, False
        scale = cap / max_abs_delta
        capped = self._last_command_q + delta * scale
        self._last_command_q = capped.copy()
        return capped, True

    # ------------------------------------------------------------------
    def _duration_for_speed_limit(
        self,
        q_start: np.ndarray,
        q_desired: np.ndarray,
        requested_duration_s: float,
    ) -> float:
        """Stretch command duration so slow speed caps still reach the target."""
        duration_s = max(0.02, float(requested_duration_s))
        delta = np.asarray(q_desired, dtype=np.float64) - np.asarray(q_start, dtype=np.float64)
        max_delta = float(np.max(np.abs(delta))) if delta.size else 0.0
        if max_delta <= 1e-9:
            return duration_s
        # Smooth-step reaches about 1.5x average velocity at mid-trajectory.
        if self.max_joint_speed_rad_s > 0.0:
            duration_s = max(duration_s, 2.0 * max_delta / self.max_joint_speed_rad_s)
        if self.max_joint_step_rad > 0.0 and self.rate_hz > 0.0:
            duration_s = max(duration_s, 2.0 * max_delta / (self.max_joint_step_rad * self.rate_hz))
        return duration_s

    # ------------------------------------------------------------------
    def _send_command(self, q_arm: np.ndarray) -> None:
        """
        Build the 30-DOF joint targets and publish via rt/arm_sdk.

        Only the arm joints for this side are set; the other 22 joints
        remain at whatever the loco controller holds.
        """
        # Build per-joint kp/kd overrides for arm joints only
        kp_by_joint = {idx: self.kp for idx in self._joint_indices}
        kd_by_joint = {idx: self.kd for idx in self._joint_indices}

        targets = {
            int(joint_idx): float(q_arm[i])
            for i, joint_idx in enumerate(self._joint_indices)
        }

        try:
            if hasattr(self.robot, "_get_arm_sdk"):
                arm_pub = self.robot._get_arm_sdk()
            else:
                arm_pub = self.robot._arm_pub
            arm_pub.publish_targets(
                joint_targets=targets,
                kp=self.kp,
                kd=self.kd,
                kp_by_joint=kp_by_joint,
                kd_by_joint=kd_by_joint,
            )
        except AttributeError:
            # Fallback: use move_upper_body_joint for each joint sequentially
            # (less coordinated but functional)
            for i, joint_idx in enumerate(self._joint_indices):
                try:
                    self.robot.move_upper_body_joint(
                        joint_index=joint_idx,
                        target=float(q_arm[i]),
                        max_speed_rad_s=1.0,
                        timeout=0.1,
                    )
                except Exception:
                    pass


# ---------------------------------------------------------------------------
def _smooth_step(t: float) -> float:
    """Smooth-step ease: 3t²-2t³ (zero velocity at endpoints)."""
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def _joint_entry_index(label: str, data: Dict) -> int:
    if "index" in data:
        try:
            return int(data.get("index", -1))
        except Exception:
            return -1
    return int(_BODY_JOINT_INDEX_BY_LABEL.get(str(label), -1))
