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
from hand_pose_navigation.reachability_checker import ReachabilityChecker
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
    p.add_argument(
        "--auto-step-base",
        action="store_true",
        help="If the target is just outside arm reach, step the robot forward before arm IK.",
    )
    p.add_argument("--max-base-step-m", type=float, default=0.30)
    p.add_argument("--base-step-speed", type=float, default=0.05)
    p.add_argument("--reach-margin-m", type=float, default=0.04)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    with open(args.target_json, "r", encoding="utf-8") as f:
        target = json.load(f)

    arm = target.get("arm", "right")
    source = str(target.get("source") or "")
    T_camera_object = np.array(target["T_camera_object"], dtype=np.float64)
    cam = target.get("camera_extrinsic", {})
    fixed_result = DetectionResult(
        T_camera_object=T_camera_object,
        confidence=float(target.get("confidence", 1.0)),
        method="fixed",
    )

    requested_standoff = float(target.get("standoff_m", 0.08))
    config = {
        "arm": arm,
        "detection_method": "fixed",
        "standoff_m": requested_standoff,
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
        "ik_tol_pos_m": 0.035 if source == "vision" else 0.003,
        "ik_tol_rot_rad": 3.14 if source == "vision" else 0.01,
        "convergence_pos_m": 0.035 if source == "vision" else 0.015,
        "convergence_rot_rad": 3.14 if source == "vision" else 0.05,
    }
    shared_robot = None
    if not args.mock:
        try:
            shared_robot = _make_shared_robot(args.iface, args.domain_id)
        except Exception as exc:
            print(f"[grab_direct] Unitree DDS/Robot init failed: {exc}")
            return 1

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
    checker = ReachabilityChecker(arm=arm)
    standoff_m, T_base_desired, reach_dist, excess_m = _choose_grasp(
        arm=arm,
        T_base_object=T_base_object,
        requested_standoff_m=requested_standoff,
        checker=checker,
    )
    config["standoff_m"] = standoff_m

    base_step_m = 0.0
    if excess_m > 0.0:
        needed_m = excess_m + max(0.0, float(args.reach_margin_m))
        base_step_m = min(max(0.0, needed_m), max(0.0, float(args.max_base_step_m)))
        print(
            "[grab_direct] target outside arm workspace: "
            f"reach_dist={reach_dist:.3f} m max={checker.max_reach_m:.3f} m "
            f"excess={excess_m:.3f} m"
        )
        if not args.auto_step_base:
            print(
                "[grab_direct] Move the object/robot closer, or rerun with "
                f"--auto-step-base to step forward up to {args.max_base_step_m:.2f} m."
            )
            return 1
        if base_step_m <= 0.0:
            print("[grab_direct] Auto-step requested, but allowed step distance is zero.")
            return 1

        try:
            _step_base_forward(
                step_m=base_step_m,
                speed_m_s=max(0.01, float(args.base_step_speed)),
                iface=args.iface,
                domain_id=args.domain_id,
                mock=args.mock,
                robot=shared_robot,
            )
        except Exception as exc:
            print(f"[grab_direct] Auto-step failed: {exc}")
            print("[grab_direct] Move the object/robot closer manually, then press Grab again.")
            return 1

        T_base_object = T_base_object.copy()
        T_base_object[0, 3] -= base_step_m
        T_camera_object = np.linalg.inv(T_base_camera) @ T_base_object
        fixed_result = DetectionResult(
            T_camera_object=T_camera_object,
            confidence=float(target.get("confidence", 1.0)),
            method="fixed",
        )
        standoff_m, T_base_desired, reach_dist, excess_m = _choose_grasp(
            arm=arm,
            T_base_object=T_base_object,
            requested_standoff_m=requested_standoff,
            checker=checker,
        )
        config["standoff_m"] = standoff_m
        if excess_m > 0.0:
            print(
                "[grab_direct] Still outside workspace after base step: "
                f"reach_dist={reach_dist:.3f} m max={checker.max_reach_m:.3f} m."
            )
            return 1

    print(
        "[grab_direct] object_base_xyz="
        f"({T_base_object[0, 3]:+.3f}, {T_base_object[1, 3]:+.3f}, {T_base_object[2, 3]:+.3f}) m "
        "desired_wrist_xyz="
        f"({T_base_desired[0, 3]:+.3f}, {T_base_desired[1, 3]:+.3f}, {T_base_desired[2, 3]:+.3f}) m "
        f"standoff={config['standoff_m']:.3f} m base_step={base_step_m:.3f} m"
    )
    nav = DirectHandPoseNav(config, fixed_result=fixed_result, robot=shared_robot)

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


def _choose_grasp(
    arm: str,
    T_base_object: np.ndarray,
    requested_standoff_m: float,
    checker: ReachabilityChecker,
) -> tuple[float, np.ndarray, float, float]:
    candidates = [
        requested_standoff_m,
        min(requested_standoff_m, 0.06),
        min(requested_standoff_m, 0.04),
        min(requested_standoff_m, 0.02),
        0.0,
    ]
    unique_candidates = []
    for value in candidates:
        value = max(0.0, float(value))
        if value not in unique_candidates:
            unique_candidates.append(value)

    best = None
    for standoff_m in unique_candidates:
        T_desired = GraspPlanner(arm=arm, standoff_m=standoff_m).compute(T_base_object)
        reach_dist = _target_reach_distance(arm, T_desired)
        excess_m = max(0.0, reach_dist - checker.max_reach_m)
        if best is None or excess_m < best[3]:
            best = (standoff_m, T_desired, reach_dist, excess_m)
        if checker.check_target_reachable(T_desired).safe:
            return standoff_m, T_desired, reach_dist, 0.0

    assert best is not None
    return best


def _target_reach_distance(arm: str, T_base_desired: np.ndarray) -> float:
    shoulder_y = 0.10 if arm == "left" else -0.10
    shoulder = np.array([0.0, shoulder_y, 0.292], dtype=np.float64)
    return float(np.linalg.norm(T_base_desired[:3, 3] - shoulder))


def _step_base_forward(
    step_m: float,
    speed_m_s: float,
    iface: str,
    domain_id: int,
    mock: bool,
    robot=None,
) -> None:
    duration_s = abs(float(step_m)) / max(0.01, abs(float(speed_m_s)))
    print(
        f"[grab_direct] stepping base forward {step_m:.3f} m "
        f"at {speed_m_s:.3f} m/s for {duration_s:.2f} s"
    )
    if mock:
        return
    if robot is not None:
        robot.move_for(duration=duration_s, vx=speed_m_s, vy=0.0, vyaw=0.0)
        return
    try:
        from sdk_boot import create_loco_client
    except Exception as exc:
        raise RuntimeError(f"Cannot auto-step base; sdk_boot import failed: {exc}") from exc

    try:
        loco = create_loco_client(domain_id=domain_id, iface=iface, timeout=2.0)
    except Exception as exc:
        raise RuntimeError(
            "Cannot auto-step base; failed to create Unitree locomotion DDS client. "
            f"iface={iface!r} domain_id={domain_id}. Original error: {exc!r}"
        ) from exc

    try:
        if not hasattr(loco, "Move"):
            raise AttributeError("Current locomotion client does not support Move().")
        loco.Move(speed_m_s, 0.0, 0.0, continous_move=True)
        time.sleep(duration_s)
    finally:
        if hasattr(loco, "StopMove"):
            loco.StopMove()
        elif hasattr(loco, "Move"):
            loco.Move(0.0, 0.0, 0.0, continous_move=False)


def _make_shared_robot(iface: str, domain_id: int):
    try:
        from dds_env import ensure_channel_factory_initialized
        ensure_channel_factory_initialized(int(domain_id), iface)
    except Exception as exc:
        raise RuntimeError(
            f"ChannelFactoryInitialize failed for iface={iface!r} domain_id={domain_id}: {exc}"
        ) from exc
    try:
        from sdk_client import Robot
        return Robot(
            iface=iface,
            domain_id=domain_id,
            auto_start_sensors=True,
        )
    except Exception as exc:
        raise RuntimeError(f"Robot construction failed after DDS init: {exc!r}") from exc


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
