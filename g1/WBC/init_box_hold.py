#!/usr/bin/env python3
"""
Box-hold initialization for the G1.

Sequence
--------
1. Acquire arm_sdk authority (unrelease_arms).
2. Extend right arm forward, then extend left arm forward.
3. Wait --wait seconds (default 60 s) so a box can be placed between
   the hands.
4. Slowly squeeze both shoulder-yaw joints inward to clamp the box from
   the sides — both arms move simultaneously.
5. Hold the squeeze pose.  If --start-wbc is given, hand off to the
   WBC which maintains balance while walking.

Squeeze geometry (shoulder YAW)
--------------------------------
When both arms are extended forward, shoulder yaw sweeps the hands
horizontally:
  left  yaw negative → hand moves inward (right, toward box centre)
  right yaw positive → hand moves inward (left, toward box centre)

  left  target = current_yaw - squeeze_delta
  right target = current_yaw + squeeze_delta

Usage
-----
  python3 init_box_hold.py
  python3 init_box_hold.py --wait 60 --squeeze-delta 0.25 --load-mass 2.0 --start-wbc
  python3 init_box_hold.py --vx 0.3 --duration 15 --load-mass 1.5 --start-wbc
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
MODULES_DIR = os.path.join(ROOT_DIR, "modules")
for _p in (ROOT_DIR, MODULES_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sdk_client import Robot, WAIST_JOINTS, UPPER_BODY_JOINTS  # noqa: E402
from sdk_client import WAIST_HOLD_KP, WAIST_HOLD_KD            # noqa: E402
from WBC.wbc import L_SHOULDER_YAW, R_SHOULDER_YAW             # noqa: E402  (17, 24)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("init_box_hold")

SQUEEZE_KP = 30.0
SQUEEZE_KD = 1.5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extend arms and grip a box from the sides.")
    p.add_argument("--iface", default="eth0", help="Network interface")
    p.add_argument("--domain-id", type=int, default=0)

    g = p.add_argument_group("extension")
    g.add_argument("--extend-duration", type=float, default=4.0,
                   help="Seconds for each arm-extension motion")

    g = p.add_argument_group("grip")
    g.add_argument("--wait", type=float, default=60.0,
                   help="Seconds to wait after extending for box placement (default 60)")
    g.add_argument("--squeeze-delta", type=float, default=0.20,
                   help="Shoulder-yaw inward travel (rad) — larger = tighter grip")
    g.add_argument("--squeeze-speed", type=float, default=0.15,
                   help="Max shoulder-yaw speed during squeeze (rad/s)")
    g.add_argument("--squeeze-rate", type=float, default=50.0,
                   help="Command rate during squeeze motion (Hz)")

    g = p.add_argument_group("WBC / walking")
    g.add_argument("--start-wbc", action="store_true",
                   help="Start WBC after gripping and hold the pose")
    g.add_argument("--load-mass", type=float, default=0.0,
                   help="Mass of held box (kg) — applies feedforward waist-pitch offset")
    g.add_argument("--load-arm", type=float, default=0.4,
                   help="Horizontal distance from waist to box CoM (m)")
    g.add_argument("--vx", type=float, default=0.0,
                   help="Forward walk speed once WBC is active (m/s)")
    g.add_argument("--vy", type=float, default=0.0)
    g.add_argument("--vyaw", type=float, default=0.0)
    g.add_argument("--duration", type=float, default=0.0,
                   help="Walk duration (s); 0 = hold indefinitely until Ctrl-C")
    g.add_argument("--roll-kp", type=float, default=0.55)
    g.add_argument("--roll-kd", type=float, default=0.08)
    g.add_argument("--pitch-kp", type=float, default=0.45)
    g.add_argument("--pitch-kd", type=float, default=0.06)
    g.add_argument("--wbc-rate", type=float, default=100.0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Squeeze helper
# ---------------------------------------------------------------------------

def squeeze_box(
    robot: Robot,
    *,
    squeeze_delta: float,
    duration_s: float,
    rate_hz: float,
) -> dict:
    """
    Interpolate both shoulder-yaw joints inward simultaneously.

      left  yaw: current - squeeze_delta   (sweeps hand toward box centre)
      right yaw: current + squeeze_delta   (sweeps hand toward box centre)

    All other upper-body joints are held at their current positions.
    """
    positions = robot._read_joint_positions_or_raise(UPPER_BODY_JOINTS, timeout=3.0)

    start_l = positions[L_SHOULDER_YAW]
    start_r = positions[R_SHOULDER_YAW]
    target_l = start_l - abs(squeeze_delta)
    target_r = start_r + abs(squeeze_delta)

    # Clamp to safe shoulder-yaw range (±1.57 rad)
    target_l = max(-1.57, min(1.57, target_l))
    target_r = max(-1.57, min(1.57, target_r))

    log.info(
        "Squeezing (yaw): L %.3f → %.3f   R %.3f → %.3f  over %.1f s",
        start_l, target_l, start_r, target_r, duration_s,
    )

    waist_gains = {j: float(WAIST_HOLD_KP) for j in WAIST_JOINTS}
    waist_damping = {j: float(WAIST_HOLD_KD) for j in WAIST_JOINTS}

    dt = 1.0 / max(1.0, rate_hz)
    steps = max(1, int(duration_s * rate_hz))

    for i in range(1, steps + 1):
        alpha = i / steps
        targets = dict(positions)
        targets[L_SHOULDER_YAW] = start_l + (target_l - start_l) * alpha
        targets[R_SHOULDER_YAW] = start_r + (target_r - start_r) * alpha

        robot._get_arm_sdk().publish_targets(
            targets,
            kp=SQUEEZE_KP,
            kd=SQUEEZE_KD,
            kp_by_joint=waist_gains,
            kd_by_joint=waist_damping,
        )
        time.sleep(dt)

    return {
        "L_shoulder_yaw_start": start_l,
        "L_shoulder_yaw_target": target_l,
        "R_shoulder_yaw_start": start_r,
        "R_shoulder_yaw_target": target_r,
    }


def hold_squeeze(robot: Robot, rate_hz: float = 20.0) -> None:
    """Re-publish the current joint positions at low rate to keep the squeeze active."""
    positions = robot._read_joint_positions_or_raise(UPPER_BODY_JOINTS, timeout=3.0)
    waist_gains = {j: float(WAIST_HOLD_KP) for j in WAIST_JOINTS}
    waist_damping = {j: float(WAIST_HOLD_KD) for j in WAIST_JOINTS}
    dt = 1.0 / max(1.0, rate_hz)
    while True:
        robot._get_arm_sdk().publish_targets(
            positions,
            kp=SQUEEZE_KP,
            kd=SQUEEZE_KD,
            kp_by_joint=waist_gains,
            kd_by_joint=waist_damping,
        )
        time.sleep(dt)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    robot = Robot(iface=args.iface, domain_id=args.domain_id)
    robot.wait_for_sport_state(timeout=5.0)
    robot.wait_for_low_state(timeout=5.0)

    # ── Step 1: acquire arm authority ────────────────────────────────────────
    log.info("Acquiring arm_sdk authority …")
    robot.unrelease_arms()
    time.sleep(0.5)

    # ── Step 2: extend arms forward ───────────────────────────────────────────
    log.info("Extending right arm forward …")
    robot.extend_arm_forward(arm="right", duration_s=args.extend_duration)

    log.info("Extending left arm forward …")
    robot.extend_arm_forward(arm="left", duration_s=args.extend_duration)

    # ── Step 3: wait for box placement ────────────────────────────────────────
    if args.wait > 0:
        log.info("Waiting %.0f s — place box between the hands now …", args.wait)
        time.sleep(args.wait)

    # ── Step 4: squeeze (shoulder yaw) ────────────────────────────────────────
    squeeze_duration = max(1.0, abs(args.squeeze_delta) / max(0.01, args.squeeze_speed))
    result = squeeze_box(
        robot,
        squeeze_delta=args.squeeze_delta,
        duration_s=squeeze_duration,
        rate_hz=args.squeeze_rate,
    )
    log.info("Squeeze complete: %s", result)

    # ── Step 5: hold / WBC ────────────────────────────────────────────────────
    if not args.start_wbc:
        log.info("Holding squeeze pose (Ctrl-C to release) …")

        def _release(sig, frame) -> None:
            log.info("Releasing arms …")
            try:
                robot.release_arms()
            except Exception:
                pass
            sys.exit(0)

        signal.signal(signal.SIGINT, _release)
        signal.signal(signal.SIGTERM, _release)
        hold_squeeze(robot, rate_hz=20.0)
        return

    # ── WBC path ──────────────────────────────────────────────────────────────
    from WBC import WBController, WBCConfig  # noqa: PLC0415

    cfg = WBCConfig(
        roll_kp=args.roll_kp,
        roll_kd=args.roll_kd,
        pitch_kp=args.pitch_kp,
        pitch_kd=args.pitch_kd,
        rate_hz=args.wbc_rate,
    )
    wbc = WBController(robot, cfg)

    if args.load_mass > 0:
        wbc.set_load(args.load_mass, args.load_arm)

    def _shutdown(sig, frame) -> None:
        log.info("Signal received — stopping …")
        wbc.set_loco_cmd(0, 0, 0)
        time.sleep(0.3)
        wbc.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # WBC captures the squeezed pose as its neutral → holds the grip while balancing
    wbc.start()
    log.info("WBC active — neutral pose locked to squeeze grip.")

    if args.vx != 0.0 or args.vy != 0.0 or args.vyaw != 0.0:
        log.info("Walking: vx=%.2f vy=%.2f vyaw=%.2f …", args.vx, args.vy, args.vyaw)
        wbc.set_loco_cmd(args.vx, args.vy, args.vyaw)

    if args.duration > 0:
        t_end = time.monotonic() + args.duration
        while time.monotonic() < t_end:
            imu = robot.get_imu()
            pos = robot.get_position()
            if imu:
                log.info(
                    "IMU roll=%.3f  pitch=%.3f | "
                    "waist r=%.3f  p=%.3f (ff+%.3f) | "
                    "pos=(%.2f, %.2f)",
                    imu.rpy[0] if imu.rpy else 0.0,
                    imu.rpy[1] if imu.rpy else 0.0,
                    wbc.last_waist_roll_cmd,
                    wbc.last_waist_pitch_cmd,
                    wbc.cfg.pitch_offset,
                    pos[0] if pos else 0.0,
                    pos[1] if pos else 0.0,
                )
            time.sleep(0.5)
        wbc.set_loco_cmd(0, 0, 0)
        time.sleep(0.3)
        wbc.stop()
        log.info("Done.")
    else:
        log.info("Holding grip + WBC active.  Ctrl-C to release.")
        while True:
            time.sleep(1.0)


if __name__ == "__main__":
    main()
