"""
Step 10 — Continuous TF feedback tracking
==========================================
Ties all pipeline steps together in a closed-loop controller that
continuously:
    1. Grabs a fresh RGB-D frame
    2. Detects the target pose
    3. Broadcasts the TF frame
    4. Looks up T_base_target via TF
    5. Computes desired hand pose
    6. Solves IK
    7. Checks reachability
    8. Sends the arm command

The loop runs at a configurable rate and terminates when the hand
converges to within tolerance of the target, or on timeout.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .target_detector import TargetDetector, DetectionResult
from .detected_pose_publisher import DetectedPosePublisher
from .arm_fk import ArmFK
from .tf_utils import TFUtils
from .grasp_planner import GraspPlanner
from .arm_ik import ArmIK
from .reachability_checker import ReachabilityChecker
from .arm_executor import ArmExecutor


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@dataclass
class LoopStatus:
    running: bool = False
    converged: bool = False
    iteration: int = 0
    last_error_pos_m: float = float("inf")
    last_error_rot_rad: float = float("inf")
    total_elapsed_s: float = 0.0
    ik_failures: int = 0
    detection_failures: int = 0
    safety_rejections: int = 0
    log: List[str] = field(default_factory=list)

    def record(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        entry = f"[{ts}] {msg}"
        self.log.append(entry)
        if len(self.log) > 200:
            del self.log[:-200]
        print(entry)

    def to_dict(self) -> Dict:
        def finite_or_none(value: float) -> Optional[float]:
            return value if math.isfinite(value) else None

        return {
            "running": self.running,
            "converged": self.converged,
            "iteration": self.iteration,
            "last_error_pos_m": finite_or_none(self.last_error_pos_m),
            "last_error_rot_rad": finite_or_none(self.last_error_rot_rad),
            "total_elapsed_s": finite_or_none(self.total_elapsed_s),
            "ik_failures": self.ik_failures,
            "detection_failures": self.detection_failures,
            "safety_rejections": self.safety_rejections,
            "log": list(self.log[-80:]),
        }


# ---------------------------------------------------------------------------
# Main tracking loop
# ---------------------------------------------------------------------------

class TrackingLoop:
    """
    Step 10: Closed-loop hand-to-target tracking controller.

    Args:
        robot:               Robot instance (sdk_client.Robot)
        detector:            TargetDetector instance
        pose_publisher:      DetectedPosePublisher instance
        tf_utils:            TFUtils instance
        fk:                  ArmFK instance
        grasp_planner:       GraspPlanner instance
        ik:                  ArmIK instance
        checker:             ReachabilityChecker instance
        executor:            ArmExecutor instance
        arm:                 "left" | "right"
        rate_hz:             control loop rate
        convergence_pos_m:   position tolerance to declare convergence
        convergence_rot_rad: rotation tolerance to declare convergence
        timeout_s:           maximum time before giving up (0 = run forever)
        on_converge:         optional callback when hand reaches target
    """

    def __init__(
        self,
        robot,
        detector: TargetDetector,
        pose_publisher: DetectedPosePublisher,
        tf_utils: TFUtils,
        fk: ArmFK,
        grasp_planner: GraspPlanner,
        ik: ArmIK,
        checker: ReachabilityChecker,
        executor: ArmExecutor,
        arm: str = "right",
        rate_hz: float = 10.0,
        convergence_pos_m: float = 0.015,
        convergence_rot_rad: float = 0.05,
        timeout_s: float = 30.0,
        on_converge: Optional[Callable[[], None]] = None,
    ) -> None:
        self.robot = robot
        self.detector = detector
        self.pose_publisher = pose_publisher
        self.tf_utils = tf_utils
        self.fk = fk
        self.grasp_planner = grasp_planner
        self.ik = ik
        self.checker = checker
        self.executor = executor
        self.arm = arm
        self.rate_hz = rate_hz
        self.convergence_pos_m = convergence_pos_m
        self.convergence_rot_rad = convergence_rot_rad
        self.timeout_s = timeout_s
        self.on_converge = on_converge

        self._stop_event = threading.Event()
        self._status = LoopStatus()
        self._thread: Optional[threading.Thread] = None

        self._joint_indices = (
            list(range(15, 22)) if arm == "left" else list(range(22, 29))
        )

    # ------------------------------------------------------------------
    def start(self, blocking: bool = True) -> LoopStatus:
        """Start the tracking loop. If blocking=False, runs in a thread."""
        self._stop_event.clear()
        self._status = LoopStatus(running=True)

        if blocking:
            self._run()
        else:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self._status

    def stop(self) -> None:
        """Signal the loop to stop cleanly."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    @property
    def status(self) -> LoopStatus:
        return self._status

    # ------------------------------------------------------------------
    def _run(self) -> None:
        dt = 1.0 / self.rate_hz
        start_t = time.time()
        q_arm_prev: Optional[np.ndarray] = None

        while not self._stop_event.is_set():
            t0 = time.time()
            self._status.iteration += 1
            elapsed = t0 - start_t
            self._status.total_elapsed_s = elapsed

            if self.timeout_s > 0 and elapsed > self.timeout_s:
                self._status.record(
                    f"[Step 10] Timeout after {elapsed:.1f}s — stopping."
                )
                break

            # ── Step 1-2: get frame + detect ──────────────────────────
            try:
                frame = self.robot.get_rgbd(timeout=0.5)
                rgb_bgr = frame.get("rgb_bgr")
                depth_m = frame.get("depth_m")
                if rgb_bgr is None or depth_m is None:
                    self._status.detection_failures += 1
                    self._status.record("[Step 2] No RGBD frame — skipping.")
                    self._sleep(dt, t0)
                    continue
            except Exception as exc:
                self._status.detection_failures += 1
                self._status.record(f"[Step 2] get_rgbd error: {exc}")
                self._sleep(dt, t0)
                continue

            detection: Optional[DetectionResult] = self.detector.detect(
                rgb_bgr,
                depth_m,
            )
            if detection is None:
                self._status.detection_failures += 1
                self._status.record("[Step 2] No detection — skipping.")
                self._sleep(dt, t0)
                continue

            # ── Step 3: broadcast TF ──────────────────────────────────
            self.pose_publisher.update(detection)

            # ── Step 4: get current hand pose (FK) ────────────────────
            q_full = self._read_q_full()
            T_base_hand = self.fk.compute(q_full)
            q_arm_cur = q_full[self._joint_indices]

            # ── Step 5: look up target in base frame ──────────────────
            T_base_target = self.tf_utils.base_to_target(timeout_s=0.05)
            if T_base_target is None:
                self._status.record("[Step 5] TF lookup failed — skipping.")
                self._sleep(dt, t0)
                continue

            # ── Step 6: desired hand pose ─────────────────────────────
            T_base_desired = self.grasp_planner.compute(T_base_target)

            # ── Pre-IK workspace check ────────────────────────────────
            reach_check = self.checker.check_target_reachable(T_base_desired)
            if not reach_check.safe:
                self._status.safety_rejections += 1
                self._status.record(
                    f"[Step 8] Workspace fail: {reach_check.reasons}"
                )
                if getattr(self.detector, "method", "") == "fixed":
                    self._status.record(
                        "[Step 8] Fixed selected target is unreachable — stopping grab."
                    )
                    break
                self._sleep(dt, t0)
                continue

            # ── Check if already converged ────────────────────────────
            err_pos, err_rot = self._pose_error_scalars(
                T_base_desired,
                T_base_hand,
            )
            self._status.last_error_pos_m = err_pos
            self._status.last_error_rot_rad = err_rot

            if (
                err_pos < self.convergence_pos_m
                and err_rot < self.convergence_rot_rad
            ):
                self._status.converged = True
                self._status.record(
                    f"[Step 10] Converged! pos_err={err_pos:.4f}m  "
                    f"rot_err={err_rot:.4f}rad  iter={self._status.iteration}"
                )
                if self.on_converge:
                    self.on_converge()
                break

            # ── Step 7: IK ───────────────────────────────────────────
            q_arm_desired, ik_info = self.ik.solve(
                T_base_desired,
                q_init=q_arm_cur,
            )
            if q_arm_desired is None:
                self._status.ik_failures += 1
                p_des = T_base_desired[:3, 3]
                self._status.record(
                    f"[Step 7] IK failed: pos_err={ik_info['error_pos_m']:.4f} "
                    f"target=({p_des[0]:+.3f},{p_des[1]:+.3f},{p_des[2]:+.3f})"
                )
                self._sleep(dt, t0)
                continue
            if ik_info.get("fallback"):
                self._status.record(
                    f"[Step 7] IK fallback={ik_info['fallback']} "
                    f"pos_err={ik_info['error_pos_m']:.4f}"
                )

            # ── Step 8: safety check ─────────────────────────────────
            safety = self.checker.check(q_arm_desired, T_base_desired)
            if not safety.safe:
                self._status.safety_rejections += 1
                self._status.record(f"[Step 8] Safety fail: {safety.reasons}")
                self._sleep(dt, t0)
                continue

            # ── Step 9: send command ─────────────────────────────────
            move_duration = min(
                dt * 1.5,
                0.3,
            )  # short smooth step each iteration
            self.executor.execute(
                q_arm_desired,
                duration_s=move_duration,
                q_arm_start=q_arm_cur,
                T_base_desired=T_base_desired,
            )
            q_arm_prev = q_arm_desired

            self._status.record(
                f"[Step 10] it={self._status.iteration:4d}  "
                f"pos_err={err_pos:.4f}m  rot_err={err_rot:.3f}rad  "
                f"ik_iter={ik_info['iterations']}"
            )

            self._sleep(dt, t0)

        self._status.running = False

    # ------------------------------------------------------------------
    def _read_q_full(self) -> np.ndarray:
        try:
            js = self.robot.get_joint_states()
            q = np.zeros(30)
            for name, data in js.get("joints", {}).items():
                idx = data.get("index", -1)
                if 0 <= idx < 30:
                    q[idx] = data.get("position", 0.0)
            return q
        except Exception:
            return np.zeros(30)

    @staticmethod
    def _pose_error_scalars(
        T_des: np.ndarray,
        T_cur: np.ndarray,
    ) -> tuple[float, float]:
        pos_err = float(np.linalg.norm(T_des[:3, 3] - T_cur[:3, 3]))
        R_err = T_des[:3, :3] @ T_cur[:3, :3].T
        rot_err = float(
            np.linalg.norm(
                np.array(
                    [
                        R_err[2, 1] - R_err[1, 2],
                        R_err[0, 2] - R_err[2, 0],
                        R_err[1, 0] - R_err[0, 1],
                    ]
                )
            )
            * 0.5
        )
        return pos_err, rot_err

    @staticmethod
    def _sleep(dt: float, t_start: float) -> None:
        elapsed = time.time() - t_start
        remaining = dt - elapsed
        if remaining > 0:
            time.sleep(remaining)
