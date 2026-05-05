#!/usr/bin/env python3
"""
WBC demo: walk forward while the whole-body controller keeps the G1 balanced.

Usage:
    python3 demo.py [--iface eth0] [--domain-id 0] [--vx 0.3] [--duration 10]
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
MODULES_DIR  = os.path.join(ROOT_DIR, "modules")
for _p in (ROOT_DIR, MODULES_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sdk_client import Robot  # modules/sdk_client.py
from WBC import WBController, WBCConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("wbc_demo")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk with whole-body balance control.")
    p.add_argument("--iface",     default="eth0", help="Network interface")
    p.add_argument("--domain-id", type=int, default=0)
    p.add_argument("--vx",        type=float, default=0.3,  help="Forward speed (m/s)")
    p.add_argument("--vy",        type=float, default=0.0,  help="Lateral speed (m/s)")
    p.add_argument("--vyaw",      type=float, default=0.0,  help="Yaw rate (rad/s)")
    p.add_argument("--duration",  type=float, default=10.0, help="Walk duration (s)")
    p.add_argument("--roll-kp",   type=float, default=0.55)
    p.add_argument("--roll-kd",   type=float, default=0.08)
    p.add_argument("--pitch-kp",  type=float, default=0.45)
    p.add_argument("--pitch-kd",  type=float, default=0.06)
    p.add_argument("--rate-hz",   type=float, default=100.0, help="WBC control rate")
    p.add_argument("--load-mass", type=float, default=0.0,
                   help="Mass of held object in kg (adds feedforward waist-pitch offset)")
    p.add_argument("--load-arm",  type=float, default=0.4,
                   help="Horizontal distance from waist to load CoM (m); default 0.4")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    robot = Robot(iface=args.iface, domain_id=args.domain_id)
    robot.wait_for_sport_state(timeout=5.0)
    robot.wait_for_low_state(timeout=5.0)

    log.info("Entering balanced stand …")
    robot.unrelease_arms()
    robot.extend_arm_forward(arm='right')
    robot.extend_arm_forward(arm='left')
    time.sleep(1.0)

    cfg = WBCConfig(
        roll_kp=args.roll_kp,
        roll_kd=args.roll_kd,
        pitch_kp=args.pitch_kp,
        pitch_kd=args.pitch_kd,
        rate_hz=args.rate_hz,
    )

    wbc = WBController(robot, cfg)

    if args.load_mass > 0:
        wbc.set_load(args.load_mass, args.load_arm)

    def _shutdown(sig, frame) -> None:
        log.info("Signal received — stopping WBC …")
        wbc.set_loco_cmd(0, 0, 0)
        time.sleep(0.3)
        wbc.stop()      # sends robot.stop() internally
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    wbc.start()
    log.info("WBC running.  Walking for %.1f s …", args.duration)
    wbc.set_loco_cmd(args.vx, args.vy, args.vyaw)

    t_end = time.monotonic() + args.duration
    while time.monotonic() < t_end:
        imu = robot.get_imu()
        pos = robot.get_position()
        if imu:
            roll  = imu.rpy[0] if imu.rpy else 0.0
            pitch = imu.rpy[1] if imu.rpy else 0.0
            log.info(
                "IMU roll=%.3f  pitch=%.3f | "
                "waist r=%.3f p=%.3f (ff+%.3f) | "
                "pos=(%.2f, %.2f)",
                roll, pitch,
                wbc.last_waist_roll_cmd,
                wbc.last_waist_pitch_cmd,
                wbc.cfg.pitch_offset,
                pos[0] if pos else 0.0,
                pos[1] if pos else 0.0,
            )
        time.sleep(0.5)

    log.info("Walk complete — stopping …")
    wbc.set_loco_cmd(0, 0, 0)
    time.sleep(0.3)
    wbc.stop()      # sends robot.stop() internally
    log.info("Done.")


if __name__ == "__main__":
    main()
