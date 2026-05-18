"""
Whole Body Controller (WBC) for the Unitree G1.

Reads IMU roll/pitch + gyro rates, then:
  1. Commands waist roll/pitch to counteract body tilt.
  2. Compensates shoulder pitch/roll to keep arms world-aligned despite
     waist motion.
  3. Holds all other upper-body joints at their captured neutral pose.

All per-step joint increments are hard-clamped to MAX_JOINT_STEP (0.05 rad).

Typical usage::

    from modules.sdk_client import Robot
    from WBC import WBController, WBCConfig

    robot = Robot(iface="eth0")
    robot.unrelease_arms()                 # hand arm_sdk authority to us

    cfg = WBCConfig(roll_kp=0.6, pitch_kp=0.5, rate_hz=100)
    wbc = WBController(robot, cfg)
    wbc.start()

    wbc.set_loco_cmd(vx=0.3, vy=0.0, vyaw=0.0)   # walk forward
    time.sleep(5.0)
    wbc.set_loco_cmd(0, 0, 0)

    wbc.stop()
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Joint indices (sdk_client.py BODY_JOINT_NAME_BY_INDEX) ──────────────────
WAIST_YAW   = 12
WAIST_ROLL  = 13
WAIST_PITCH = 14

L_SHOULDER_PITCH = 15
L_SHOULDER_ROLL  = 16
L_SHOULDER_YAW   = 17
L_ELBOW          = 18
L_WRIST_ROLL     = 19
L_WRIST_PITCH    = 20
L_WRIST_YAW      = 21

R_SHOULDER_PITCH = 22
R_SHOULDER_ROLL  = 23
R_SHOULDER_YAW   = 24
R_ELBOW          = 25
R_WRIST_ROLL     = 26
R_WRIST_PITCH    = 27
R_WRIST_YAW      = 28

UPPER_BODY_JOINTS = [
    WAIST_YAW, WAIST_ROLL, WAIST_PITCH,
    L_SHOULDER_PITCH, L_SHOULDER_ROLL, L_SHOULDER_YAW,
    L_ELBOW, L_WRIST_ROLL, L_WRIST_PITCH, L_WRIST_YAW,
    R_SHOULDER_PITCH, R_SHOULDER_ROLL, R_SHOULDER_YAW,
    R_ELBOW, R_WRIST_ROLL, R_WRIST_PITCH, R_WRIST_YAW,
]

# ── Safety ───────────────────────────────────────────────────────────────────
# Hard limit: maximum joint change allowed per control step
MAX_JOINT_STEP = 0.05  # rad

# Conservative per-joint position limits [min, max] in radians
JOINT_LIMITS: Dict[int, Tuple[float, float]] = {
    WAIST_YAW:        (-0.52,  0.52),
    WAIST_ROLL:       (-0.40,  0.40),
    WAIST_PITCH:      (-0.50,  0.50),
    L_SHOULDER_PITCH: (-1.57,  1.57),
    L_SHOULDER_ROLL:  (-0.20,  2.60),
    L_SHOULDER_YAW:   (-1.57,  1.57),
    L_ELBOW:          (-1.57,  0.05),
    L_WRIST_ROLL:     (-1.57,  1.57),
    L_WRIST_PITCH:    (-1.57,  1.57),
    L_WRIST_YAW:      (-1.57,  1.57),
    R_SHOULDER_PITCH: (-1.57,  1.57),
    R_SHOULDER_ROLL:  (-2.60,  0.20),
    R_SHOULDER_YAW:   (-1.57,  1.57),
    R_ELBOW:          (-1.57,  0.05),
    R_WRIST_ROLL:     (-1.57,  1.57),
    R_WRIST_PITCH:    (-1.57,  1.57),
    R_WRIST_YAW:      (-1.57,  1.57),
}

# ── arm_sdk servo gains ───────────────────────────────────────────────────────
WAIST_KP = 480.0
WAIST_KD = 12.0
ARM_KP   = 30.0
ARM_KD   = 1.5

_KP_BY_JOINT: Dict[int, float] = {j: WAIST_KP for j in [WAIST_YAW, WAIST_ROLL, WAIST_PITCH]}
_KD_BY_JOINT: Dict[int, float] = {j: WAIST_KD for j in [WAIST_YAW, WAIST_ROLL, WAIST_PITCH]}


# ── Configuration ────────────────────────────────────────────────────────────
@dataclass
class WBCConfig:
    """Tunable parameters for the whole-body balance controller."""

    # Waist roll PD gains  (IMU roll → waist roll command)
    roll_kp:  float = 0.55
    roll_kd:  float = 0.08

    # Waist pitch PD gains  (IMU pitch → waist pitch command)
    pitch_kp: float = 0.45
    pitch_kd: float = 0.06

    # Fraction of waist delta fed back to shoulders (1.0 = full cancellation)
    arm_compensation: float = 1.0

    # Control loop frequency
    rate_hz: float = 100.0

    # Dead-band: tilt errors smaller than this are treated as zero (rad)
    tilt_deadband: float = 0.012

    # Waist command magnitude clamp before absolute joint limiting (rad)
    waist_roll_limit:  float = 0.35
    waist_pitch_limit: float = 0.45

    # Static feedforward offsets (rad) added on top of the PD output.
    # Positive pitch_offset tilts the waist backward — use this to counteract
    # a forward-heavy load (box held out in front).  Tune via set_load() or
    # set directly for quick manual adjustment.
    pitch_offset: float = 0.0
    roll_offset:  float = 0.0


# ── Controller ───────────────────────────────────────────────────────────────
class WBController:
    """
    Whole-body balance controller for the G1.

    Runs a daemon thread at `cfg.rate_hz` that:
      - Reads IMU (roll, pitch, gyro) via robot.get_imu().
      - Reads joint positions via robot.get_low_state_snapshot()  (fast path).
      - Applies PD control → waist roll/pitch corrections.
      - Compensates shoulder pitch/roll so the arms stay world-aligned.
      - Holds all other upper-body joints at the captured neutral pose.
      - Forwards loco_move(vx, vy, vyaw) each tick when a non-zero command
        has been set via set_loco_cmd().

    Thread safety: set_loco_cmd(), set_neutral_pose(), and cfg can be
    updated from any thread while the controller is running.
    """

    def __init__(self, robot, cfg: Optional[WBCConfig] = None) -> None:
        self._robot = robot
        self._cfg   = cfg or WBCConfig()

        self._lock  = threading.Lock()
        self._loco_cmd: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._neutral:  Optional[Dict[int, float]] = None   # {joint_idx: angle}

        self._thread:  Optional[threading.Thread] = None
        self._running: bool = False
        self._output_enabled: bool = False
        self._arms_released: bool = True

        # Last diagnostics (read from any thread)
        self.last_imu_roll:       float = 0.0
        self.last_imu_pitch:      float = 0.0
        self.last_waist_roll_cmd: float = 0.0
        self.last_waist_pitch_cmd: float = 0.0
        self.last_odom: Optional[Tuple[float, float, float]] = None

    # ── Public API ───────────────────────────────────────────────────────────

    def start(self) -> None:
        """Capture the current arm pose as neutral and start the control loop."""
        if self._running:
            return
        self._capture_neutral()
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="WBC", daemon=True)
        self._thread.start()
        logger.info("WBC started at %.0f Hz", self._cfg.rate_hz)

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the control loop, halt locomotion, and release arm authority."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        # Explicit locomotion stop — the SDK watchdog may not fire immediately
        for _ in range(3):
            try:
                self._robot.stop()
            except Exception:
                break
            time.sleep(0.05)
        try:
            self._robot.release_arms()
            self._arms_released = True
        except Exception:
            pass
        logger.info("WBC stopped")

    def set_output_enabled(self, enabled: bool) -> None:
        """
        Enable or disable low-level upper-body output.

        When disabled, the WBC keeps running for locomotion forwarding and state
        updates but does not publish any arm/waist joint command packet.
        """
        enabled = bool(enabled)
        with self._lock:
            previous = self._output_enabled
            self._output_enabled = enabled

        if enabled == previous:
            return

        if enabled:
            try:
                self._robot.unrelease_arms()
                self._arms_released = False
            except Exception as exc:
                logger.warning("WBC: failed to unrelease arms: %s", exc)
            self._capture_neutral()
            logger.info("WBC low-level output enabled")
            return

        try:
            self._robot.release_arms()
            self._arms_released = True
        except Exception as exc:
            logger.warning("WBC: failed to release arms: %s", exc)
        logger.info("WBC low-level output disabled")

    def set_loco_cmd(self, vx: float, vy: float, vyaw: float) -> None:
        """Set the locomotion command forwarded to loco_move() each tick."""
        with self._lock:
            self._loco_cmd = (float(vx), float(vy), float(vyaw))

    def set_load(self, mass_kg: float, moment_arm_m: float = 0.4) -> None:
        """
        Set a feedforward waist-pitch offset to compensate a held load.

        mass_kg        — mass of the held object (kg)
        moment_arm_m   — horizontal distance from waist to load CoM (m);
                         default 0.4 m ≈ arms extended forward

        Derivation: the load creates a forward torque τ = M·g·r.
        The WBC counters this by pitching the waist backward.  The empirical
        scale factor (~0.05 rad / N·m) was tuned for the G1 with WAIST_KP=480.
        Adjust LOAD_SCALE if the robot still leans forward under load.
        """
        LOAD_SCALE = 0.05   # rad per N·m — increase if under-correcting
        torque = mass_kg * 9.81 * moment_arm_m
        offset = torque * LOAD_SCALE
        with self._lock:
            self._cfg.pitch_offset = offset
        logger.info(
            "Load %.1f kg @ %.2f m arm → τ=%.1f N·m → pitch_offset=+%.4f rad (waist back)",
            mass_kg, moment_arm_m, torque, offset,
        )

    def set_neutral_pose(self, targets: Optional[Dict[int, float]] = None) -> None:
        """
        Update the neutral joint pose the WBC holds.

        Pass None to re-capture from the current robot state.
        """
        if targets is None:
            self._capture_neutral()
        else:
            with self._lock:
                self._neutral = dict(targets)

    @property
    def cfg(self) -> WBCConfig:
        return self._cfg

    @cfg.setter
    def cfg(self, value: WBCConfig) -> None:
        with self._lock:
            self._cfg = value

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _capture_neutral(self) -> None:
        """Read the current joint positions and store as the neutral pose."""
        try:
            # _read_joint_positions_or_raise returns {int_idx: float}
            neutral = self._robot._read_joint_positions_or_raise(
                UPPER_BODY_JOINTS, timeout=3.0
            )
        except Exception as exc:
            logger.warning("WBC: neutral capture failed: %s", exc)
            return
        with self._lock:
            self._neutral = neutral
        logger.debug("WBC: neutral captured: %s", neutral)

    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    @staticmethod
    def _deadband(v: float, band: float) -> float:
        """Zero out |v| < band, then subtract the band from the remainder."""
        if abs(v) < band:
            return 0.0
        return v - math.copysign(band, v)

    # ── Control loop ─────────────────────────────────────────────────────────

    def _loop(self) -> None:
        dt = 1.0 / self._cfg.rate_hz
        while self._running:
            t0 = time.monotonic()
            try:
                self._tick()
            except Exception as exc:
                logger.warning("WBC tick error: %s", exc)
            elapsed = time.monotonic() - t0
            wait = dt - elapsed
            if wait > 0:
                time.sleep(wait)

    def _tick(self) -> None:
        with self._lock:
            cfg     = self._cfg
            loco    = self._loco_cmd
            neutral = dict(self._neutral) if self._neutral else {}
            output_enabled = self._output_enabled

        if not neutral:
            return

        # ── 1. Read IMU ──────────────────────────────────────────────────────
        imu = self._robot.get_imu()
        if imu is None:
            return

        roll    = float(imu.rpy[0])   if imu.rpy  else 0.0
        pitch   = float(imu.rpy[1])   if imu.rpy  else 0.0
        gyro_x  = float(imu.gyro[0])  if imu.gyro else 0.0
        gyro_y  = float(imu.gyro[1])  if imu.gyro else 0.0

        self.last_imu_roll  = roll
        self.last_imu_pitch = pitch

        # Odometry (informational; available for future extensions)
        pos = self._robot.get_position()
        if pos is not None:
            self.last_odom = (float(pos[0]), float(pos[1]), float(pos[2]))

        # ── 2. Read current joint positions (fast low-state snapshot) ────────
        snapshot = self._robot.get_low_state_snapshot()
        if snapshot is None:
            return
        # joint_positions is a list[float] indexed by joint index (0-29)
        cur: Dict[int, float] = {
            j: float(snapshot.joint_positions[j]) for j in UPPER_BODY_JOINTS
        }

        # ── 3. PD corrections for waist roll and pitch ───────────────────────
        #   Error = current tilt (positive roll = robot leaning left)
        #   Correction opposes the tilt: negative feedback
        roll_err   = self._deadband(roll,  cfg.tilt_deadband)
        pitch_err  = self._deadband(pitch, cfg.tilt_deadband)

        waist_roll_delta  = -(cfg.roll_kp  * roll_err  + cfg.roll_kd  * gyro_x)
        waist_pitch_delta = -(cfg.pitch_kp * pitch_err + cfg.pitch_kd * gyro_y)

        # Clamp correction magnitude (soft limit before joint clamping)
        waist_roll_delta  = self._clamp(waist_roll_delta,  -cfg.waist_roll_limit,  cfg.waist_roll_limit)
        waist_pitch_delta = self._clamp(waist_pitch_delta, -cfg.waist_pitch_limit, cfg.waist_pitch_limit)

        self.last_waist_roll_cmd  = waist_roll_delta
        self.last_waist_pitch_cmd = waist_pitch_delta

        # ── 4. Build desired target dict ─────────────────────────────────────
        targets: Dict[int, float] = dict(neutral)   # start from neutral

        # Total waist commands: PD feedback + static load feedforward offset
        total_roll_cmd  = waist_roll_delta  + cfg.roll_offset
        total_pitch_cmd = waist_pitch_delta + cfg.pitch_offset

        targets[WAIST_YAW]   = neutral.get(WAIST_YAW,   0.0)               # yaw stays neutral
        targets[WAIST_ROLL]  = neutral.get(WAIST_ROLL,  0.0) + total_roll_cmd
        targets[WAIST_PITCH] = neutral.get(WAIST_PITCH, 0.0) + total_pitch_cmd

        # Arm compensation: shoulder pitch/roll counter-rotate to cancel
        # the world-space orientation change introduced by waist motion.
        # The compensation includes the static load offset so the hands stay
        # in the same world-space pose even when the waist is biased backward.
        c = cfg.arm_compensation
        targets[L_SHOULDER_PITCH] = neutral.get(L_SHOULDER_PITCH, 0.0) - c * total_pitch_cmd
        targets[L_SHOULDER_ROLL]  = neutral.get(L_SHOULDER_ROLL,  0.0) - c * total_roll_cmd
        targets[R_SHOULDER_PITCH] = neutral.get(R_SHOULDER_PITCH, 0.0) - c * total_pitch_cmd
        targets[R_SHOULDER_ROLL]  = neutral.get(R_SHOULDER_ROLL,  0.0) - c * total_roll_cmd

        # All other joints already set to neutral above.

        # ── 5. Enforce per-step increment limit: max 0.05 rad ────────────────
        for j in UPPER_BODY_JOINTS:
            desired = targets.get(j, cur[j])
            delta   = desired - cur[j]
            clamped = self._clamp(delta, -MAX_JOINT_STEP, MAX_JOINT_STEP)
            targets[j] = cur[j] + clamped

        # ── 6. Enforce absolute joint position limits ─────────────────────────
        for j, (lo, hi) in JOINT_LIMITS.items():
            if j in targets:
                targets[j] = self._clamp(targets[j], lo, hi)

        # ── 7. Publish via arm_sdk only when low-level output is enabled ─────
        if output_enabled:
            self._robot._get_arm_sdk().publish_targets(
                targets,
                kp=ARM_KP,
                kd=ARM_KD,
                kp_by_joint=_KP_BY_JOINT,
                kd_by_joint=_KD_BY_JOINT,
            )

        # ── 8. Forward locomotion command ─────────────────────────────────────
        # Always call loco_move so a (0,0,0) command explicitly stops the robot
        # rather than relying on the SDK watchdog timeout.
        self._robot.loco_move(*loco)
