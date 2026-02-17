#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

from dev.ef_client import Robot


def _move(
    robot: Robot,
    *,
    arm: str,
    joint: str,
    deg: float,
    duration: float,
    hold: float,
    cmd_hz: float,
    kp: float,
    kd: float,
    easing: str,
    max_joint_speed: float,
) -> None:
    rc = robot.rotate_joint(
        joint_name=joint,
        angle_deg=float(deg),
        arm=arm,
        duration=float(duration),
        hold=float(hold),
        cmd_hz=float(cmd_hz),
        kp=float(kp),
        kd=float(kd),
        easing=str(easing),
        max_joint_speed=float(max_joint_speed),
    )
    if int(rc) != 0:
        raise RuntimeError(f"rotate_joint failed: arm={arm} joint={joint} deg={deg} rc={rc}")


def main() -> None:
    p = argparse.ArgumentParser(description="Move both arms to a punch-ready guard position.")
    p.add_argument("--iface", default="enp1s0", help="DDS network interface")
    p.add_argument("--domain-id", type=int, default=0, help="DDS domain id")
    p.add_argument("--no-safety-boot", action="store_true", help="skip hanger safety boot in Robot()")
    p.add_argument("--duration", type=float, default=1.1, help="seconds per joint move")
    p.add_argument("--hold", type=float, default=0.10, help="seconds hold after each move")
    p.add_argument("--cmd-hz", type=float, default=50.0, help="command publish rate")
    p.add_argument("--kp", type=float, default=40.0, help="joint kp")
    p.add_argument("--kd", type=float, default=1.0, help="joint kd")
    p.add_argument("--easing", choices=["linear", "smooth"], default="smooth", help="interpolation easing")
    p.add_argument("--max-joint-speed", type=float, default=0.9, help="rad/s speed cap for anti-snap motion")
    p.add_argument(
        "--range-scale",
        type=float,
        default=1.0,
        help="scale punch posture angles (>1.0 larger range, <1.0 smaller range)",
    )
    args = p.parse_args()

    robot = Robot(
        iface=args.iface,
        domain_id=int(args.domain_id),
        safety_boot=not bool(args.no_safety_boot),
    )

    # Keep base stable before arm motion.
    robot.stop()
    robot.balanced_stand(0)
    time.sleep(0.3)

    s = max(0.5, min(2.0, float(args.range_scale)))
    shoulder_pitch_deg = 35.0 * s
    elbow_deg = 105.0 * s
    shoulder_yaw_deg = 18.0 * s

    # Step 1: bring both upper arms forward.
    _move(robot, arm="left", joint="shoulder_pitch", deg=shoulder_pitch_deg, duration=args.duration, hold=args.hold, cmd_hz=args.cmd_hz, kp=args.kp, kd=args.kd, easing=args.easing, max_joint_speed=args.max_joint_speed)
    _move(robot, arm="right", joint="shoulder_pitch", deg=shoulder_pitch_deg, duration=args.duration, hold=args.hold, cmd_hz=args.cmd_hz, kp=args.kp, kd=args.kd, easing=args.easing, max_joint_speed=args.max_joint_speed)

    # Step 2: bend elbows into a guard/punch-ready posture.
    _move(robot, arm="left", joint="elbow", deg=elbow_deg, duration=args.duration, hold=args.hold, cmd_hz=args.cmd_hz, kp=args.kp, kd=args.kd, easing=args.easing, max_joint_speed=args.max_joint_speed)
    _move(robot, arm="right", joint="elbow", deg=elbow_deg, duration=args.duration, hold=args.hold, cmd_hz=args.cmd_hz, kp=args.kp, kd=args.kd, easing=args.easing, max_joint_speed=args.max_joint_speed)

    # Step 3: a small inward yaw to square the fists forward.
    _move(robot, arm="left", joint="shoulder_yaw", deg=shoulder_yaw_deg, duration=0.9, hold=0.05, cmd_hz=args.cmd_hz, kp=args.kp, kd=args.kd, easing=args.easing, max_joint_speed=args.max_joint_speed)
    _move(robot, arm="right", joint="shoulder_yaw", deg=-shoulder_yaw_deg, duration=0.9, hold=0.05, cmd_hz=args.cmd_hz, kp=args.kp, kd=args.kd, easing=args.easing, max_joint_speed=args.max_joint_speed)

    print("Punch-ready arm position applied.")


if __name__ == "__main__":
    main()
