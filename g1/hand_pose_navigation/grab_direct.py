#!/usr/bin/env python3
"""
Grab approach A — direct (no ROS 2).

Drives DirectHandPoseNav's existing tracking loop (perception -> grasp
planner -> IK -> reachability check -> arm executor) toward a single
pre-computed target pose, then closes the Dex3 hand once the wrist
converges. This is the "approach" side of the recognition-layer Grab
button: object selection and pose estimation happen once in
recognition_app.py; this script just executes the move.

Target JSON contract (written by recognition_app.py):
{
  "arm": "right" | "left",
  "T_camera_object": [[..4x4 row-major..]],
  "camera_extrinsic": {"x":..,"y":..,"z":..,"roll":..,"pitch":..,"yaw":..},
  "standoff_m": 0.08,
  "label": "red mug",
  "source": "aruco" | "vision"
}

Usage:
    python3 grab_direct.py --target-json /tmp/grab_target.json
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

from hand_pose_navigation.direct_nav import DirectHandPoseNav, _make_transform
from hand_pose_navigation.grasp_planner import GraspPlanner
from hand_pose_navigation.target_detector import DetectionResult


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grab a selected object (direct backend).")
    p.add_argument("--target-json", required=True)
    p.add_argument("--ik-solver", choices=("dls", "scipy", "pin"), default="dls")
    p.add_argument("--rate-hz", type=float, default=10.0)
    p.add_argument("--timeout-s", type=float, default=25.0)
    p.add_argument("--iface", default="eth0")
    p.add_argument("--domain-id", type=int, default=0)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--no-close-hand", action="store_true")
    p.add_argument("--hand-hold-s", type=float, default=0.6)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    with open(args.target_json, "r", encoding="utf-8") as f:
        target = json.load(f)

    arm = target.get("arm", "right")
    T_camera_object = np.array(target["T_camera_object"], dtype=np.float64)
    cam = target.get("camera_extrinsic", {})
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
        "mock": args.mock,
        "camera_x": float(cam.get("x", 0.0)),
        "camera_y": float(cam.get("y", 0.0)),
        "camera_z": float(cam.get("z", 0.0)),
        "camera_roll": float(cam.get("roll", 0.0)),
        "camera_pitch": float(cam.get("pitch", 0.0)),
        "camera_yaw": float(cam.get("yaw", 0.0)),
    }

    label = target.get("label", "<object>")
    print(f"[grab_direct] arm={arm} label={label!r} source={target.get('source')}")
    T_base_camera = _make_transform(
        xyz=(
            config["camera_x"],
            config["camera_y"],
            config["camera_z"],
        ),
        rpy=(
            config["camera_roll"],
            config["camera_pitch"],
            config["camera_yaw"],
        ),
    )
    T_base_object = T_base_camera @ T_camera_object
    T_base_desired = GraspPlanner(
        arm=arm,
        standoff_m=config["standoff_m"],
    ).compute(T_base_object)
    print(
        "[grab_direct] object_base_xyz="
        f"({T_base_object[0, 3]:+.3f}, {T_base_object[1, 3]:+.3f}, {T_base_object[2, 3]:+.3f}) m "
        "desired_wrist_xyz="
        f"({T_base_desired[0, 3]:+.3f}, {T_base_desired[1, 3]:+.3f}, {T_base_desired[2, 3]:+.3f}) m "
        f"standoff={config['standoff_m']:.3f} m"
    )
    nav = DirectHandPoseNav(config, fixed_result=fixed_result)

    ok = False
    try:
        deadline = time.time() + args.timeout_s + 2.0
        while time.time() < deadline:
            status = nav.status_snapshot()
            if status.get("converged"):
                ok = True
                break
            if not status.get("running", True):
                break
            time.sleep(0.2)
    finally:
        nav.shutdown()

    if not ok:
        print("[grab_direct] Did not converge within timeout — not closing hand.")
        return 1

    print(f"[grab_direct] Converged. pos_err<=tol, rot_err<=tol.")
    if not args.no_close_hand:
        _close_hand(arm, args.iface, args.domain_id, args.hand_hold_s)
    print("[grab_direct] Done.")
    return 0


def _close_hand(arm: str, iface: str, domain_id: int, hold_s: float) -> None:
    try:
        from sdk_hand import Dex3HandController
    except Exception as exc:
        print(f"[grab_direct] Could not import Dex3HandController: {exc}")
        return
    try:
        hand = Dex3HandController(hand=arm, iface=iface, domain_id=domain_id)
        hand.close(hold_s=hold_s)
        print(f"[grab_direct] Closed {arm} Dex3 hand.")
    except Exception as exc:
        print(f"[grab_direct] Hand close failed: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
