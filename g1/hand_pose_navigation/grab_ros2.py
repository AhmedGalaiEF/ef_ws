#!/usr/bin/env python3
"""
Grab approach B — ROS 2 / TF.

Same job as grab_direct.py (drive the wrist to a pre-computed target pose,
then close the Dex3 hand) but routed through HandPoseNavNode's ROS 2 TF tree
(camera_tf_publisher + detected_pose_publisher + tf2) instead of the direct
in-process transform. Requires rclpy/tf2_ros to be installed and importable
alongside unitree_sdk2py in the same process — that combination is not set
up in every environment (see direct_nav.py's docstring); use grab_direct.py
if this fails to import.

Target JSON contract: identical to grab_direct.py.

Usage:
    python3 grab_ros2.py --target-json /tmp/grab_target.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
_G1_DIR = os.path.abspath(os.path.join(_DIR, ".."))
if _G1_DIR not in sys.path:
    sys.path.insert(0, _G1_DIR)
_MODULES = os.path.join(_G1_DIR, "modules")
if _MODULES not in sys.path:
    sys.path.insert(0, _MODULES)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grab a selected object (ROS 2/TF backend).")
    p.add_argument("--target-json", required=True)
    p.add_argument("--ik-solver", choices=("dls", "scipy", "pin"), default="dls")
    p.add_argument("--rate-hz", type=float, default=10.0)
    p.add_argument("--timeout-s", type=float, default=25.0)
    p.add_argument("--iface", default="eth0")
    p.add_argument("--domain-id", type=int, default=0)
    p.add_argument("--no-close-hand", action="store_true")
    p.add_argument("--hand-hold-s", type=float, default=0.6)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    with open(args.target_json, "r", encoding="utf-8") as f:
        target = json.load(f)

    try:
        import rclpy
    except ImportError as exc:
        print(
            f"[grab_ros2] rclpy is not installed ({exc}). "
            "Use grab_direct.py, or install ROS 2 + this package first."
        )
        return 2

    from hand_pose_navigation.hand_pose_nav_node import HandPoseNavNode
    from hand_pose_navigation.target_detector import DetectionResult

    arm = target.get("arm", "right")
    T_camera_object = np.array(target["T_camera_object"], dtype=np.float64)
    fixed_result = DetectionResult(
        T_camera_object=T_camera_object,
        confidence=float(target.get("confidence", 1.0)),
        method="fixed",
    )

    config = {
        "arm": arm,
        "detection_method": "fixed",
        "standoff_m": float(target.get("standoff_m", 0.08)),
        "rate_hz": args.rate_hz,
        "timeout_s": args.timeout_s,
        "ik_solver": args.ik_solver,
        "iface": args.iface,
        "domain_id": args.domain_id,
    }

    label = target.get("label", "<object>")
    print(f"[grab_ros2] arm={arm} label={label!r} source={target.get('source')}")

    rclpy.init(args=None)
    node = HandPoseNavNode(config=config, fixed_result=fixed_result)

    ok = False
    try:
        deadline = time.time() + args.timeout_s + 2.0
        while time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
            status = node.status_snapshot()
            if status.get("converged"):
                ok = True
                break
            if not status.get("running", True):
                break
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if not ok:
        print("[grab_ros2] Did not converge within timeout — not closing hand.")
        return 1

    print("[grab_ros2] Converged.")
    if not args.no_close_hand:
        _close_hand(arm, args.iface, args.domain_id, args.hand_hold_s)
    print("[grab_ros2] Done.")
    return 0


def _close_hand(arm: str, iface: str, domain_id: int, hold_s: float) -> None:
    try:
        from sdk_hand import Dex3HandController
    except Exception as exc:
        print(f"[grab_ros2] Could not import Dex3HandController: {exc}")
        return
    try:
        hand = Dex3HandController(hand=arm, iface=iface, domain_id=domain_id)
        hand.close(hold_s=hold_s)
        print(f"[grab_ros2] Closed {arm} Dex3 hand.")
    except Exception as exc:
        print(f"[grab_ros2] Hand close failed: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
