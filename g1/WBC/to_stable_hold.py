#!/usr/bin/env python3
"""
Move smoothly into the saved stable arm pose and keep holding it.

The stable pose below was copied from ik_pose_cli_v2.py joint targets:

  LEFT : +0.312  +0.221  +0.105  -0.684  -0.368  +0.164  +0.000
  RIGHT: +0.323  -0.207  -0.080  -0.688  +0.328  +0.140  +0.000

Waist joints are captured from live feedback at startup and held there, because
the saved TUI snapshot did not include explicit waist targets.
"""

from __future__ import annotations

import argparse
import logging
import math
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

from sdk_client import (  # noqa: E402
    LEFT_ARM_JOINTS,
    RIGHT_ARM_JOINTS,
    UPPER_BODY_JOINTS,
    WAIST_HOLD_KD,
    WAIST_HOLD_KP,
    WAIST_JOINTS,
    Robot,
)

log = logging.getLogger("to_stable_hold")

ARM_KP = 30.0
ARM_KD = 1.5

STABLE_LEFT = {
    LEFT_ARM_JOINTS[0]: 0.312,
    LEFT_ARM_JOINTS[1]: 0.221,
    LEFT_ARM_JOINTS[2]: 0.105,
    LEFT_ARM_JOINTS[3]: -0.684,
    LEFT_ARM_JOINTS[4]: -0.368,
    LEFT_ARM_JOINTS[5]: 0.164,
    LEFT_ARM_JOINTS[6]: 0.000,
}

STABLE_RIGHT = {
    RIGHT_ARM_JOINTS[0]: 0.323,
    RIGHT_ARM_JOINTS[1]: -0.207,
    RIGHT_ARM_JOINTS[2]: -0.080,
    RIGHT_ARM_JOINTS[3]: -0.688,
    RIGHT_ARM_JOINTS[4]: 0.328,
    RIGHT_ARM_JOINTS[5]: 0.140,
    RIGHT_ARM_JOINTS[6]: 0.000,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ramp the G1 upper body from its current pose to the saved stable hold pose."
    )
    p.add_argument("--iface", default="eth0", help="DDS network interface")
    p.add_argument("--domain-id", type=int, default=0)
    p.add_argument("--rate-hz", type=float, default=50.0, help="Command publish rate")
    p.add_argument(
        "--speed-rad-s",
        type=float,
        default=0.2,
        help="Per-joint ramp speed limit in rad/s",
    )
    p.add_argument(
        "--max-step-rad",
        type=float,
        default=0.2,
        help="Hard maximum joint position increment per command packet",
    )
    p.add_argument("--kp", type=float, default=ARM_KP, help="Arm position gain")
    p.add_argument("--kd", type=float, default=ARM_KD, help="Arm damping gain")
    p.add_argument("--waist-kp", type=float, default=WAIST_HOLD_KP)
    p.add_argument("--waist-kd", type=float, default=WAIST_HOLD_KD)
    p.add_argument(
        "--hold-rate-hz",
        type=float,
        default=20.0,
        help="Publish rate after the target pose is reached",
    )
    p.add_argument(
        "--no-release-on-exit",
        action="store_true",
        help="Leave arm_sdk authority active on Ctrl-C instead of fading it out",
    )
    p.add_argument(
        "--tolerance-rad",
        type=float,
        default=0.002,
        help="Stop ramping when every target is within this error",
    )
    return p.parse_args()


def _max_abs_error(current: dict[int, float], target: dict[int, float]) -> float:
    return max(abs(float(target[j]) - float(current[j])) for j in target)


def _step_toward(
    current: dict[int, float],
    target: dict[int, float],
    *,
    max_delta: float,
) -> dict[int, float]:
    stepped = dict(current)
    limit = max(1e-6, float(max_delta))
    for joint_index, target_q in target.items():
        cur_q = float(current[joint_index])
        delta = float(target_q) - cur_q
        if abs(delta) <= limit:
            stepped[joint_index] = float(target_q)
        else:
            stepped[joint_index] = cur_q + math.copysign(limit, delta)
    return stepped


def _publish(
    robot: Robot,
    targets: dict[int, float],
    *,
    kp: float,
    kd: float,
    waist_kp: float,
    waist_kd: float,
) -> None:
    waist_gains = {joint_index: float(waist_kp) for joint_index in WAIST_JOINTS}
    waist_damping = {joint_index: float(waist_kd) for joint_index in WAIST_JOINTS}
    robot._get_arm_sdk().publish_targets(
        targets,
        kp=kp,
        kd=kd,
        kp_by_joint=waist_gains,
        kd_by_joint=waist_damping,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    rate_hz = max(1.0, float(args.rate_hz))
    hold_rate_hz = max(1.0, float(args.hold_rate_hz))
    speed_rad_s = max(0.001, float(args.speed_rad_s))
    max_step_rad = max(0.001, float(args.max_step_rad))
    per_tick_delta = min(max_step_rad, speed_rad_s / rate_hz)

    robot = Robot(iface=args.iface, domain_id=args.domain_id)
    robot.wait_for_low_state(timeout=5.0)

    log.info("Reading live upper-body pose")
    start = robot._read_joint_positions_or_raise(UPPER_BODY_JOINTS, timeout=3.0)

    target = dict(start)
    target.update(STABLE_LEFT)
    target.update(STABLE_RIGHT)

    log.info("Acquiring arm_sdk authority while holding current pose")
    robot.unrelease_arms(
        duration_s=1.0,
        command_rate_hz=rate_hz,
        kp=args.kp,
        kd=args.kd,
        waist_kp=args.waist_kp,
        waist_kd=args.waist_kd,
    )

    running = True
    released = False

    def _handle_stop(signum, frame) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    current = dict(start)
    dt = 1.0 / rate_hz
    log.info(
        "Ramping to stable pose at %.3f rad/s, %.4f rad max per packet",
        speed_rad_s,
        per_tick_delta,
    )

    try:
        while running and _max_abs_error(current, target) > float(args.tolerance_rad):
            tick_start = time.monotonic()
            current = _step_toward(current, target, max_delta=per_tick_delta)
            _publish(
                robot,
                current,
                kp=args.kp,
                kd=args.kd,
                waist_kp=args.waist_kp,
                waist_kd=args.waist_kd,
            )
            sleep_s = dt - (time.monotonic() - tick_start)
            if sleep_s > 0.0:
                time.sleep(sleep_s)

        if running:
            log.info("Stable pose reached; holding until Ctrl-C")
        hold_dt = 1.0 / hold_rate_hz
        while running:
            _publish(
                robot,
                target,
                kp=args.kp,
                kd=args.kd,
                waist_kp=args.waist_kp,
                waist_kd=args.waist_kd,
            )
            time.sleep(hold_dt)
    finally:
        if not args.no_release_on_exit:
            log.info("Releasing arm_sdk authority")
            robot.release_arms(
                duration_s=1.0,
                command_rate_hz=rate_hz,
                kp=args.kp,
                kd=args.kd,
                waist_kp=args.waist_kp,
                waist_kd=args.waist_kd,
            )
            released = True
        if released:
            log.info("Released")


if __name__ == "__main__":
    main()
