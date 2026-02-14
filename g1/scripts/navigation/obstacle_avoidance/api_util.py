#!/usr/bin/env python3
"""
api_util.py
===========

Utility to exercise slam_operate APIs and print request/response payloads.
Matches the C++ keyDemo.cpp interface list.
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Optional

try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
except ImportError as exc:
    raise SystemExit(
        "unitree_sdk2py is not installed. Install it with:\n"
        "  pip install -e <path-to-unitree_sdk2_python>"
    ) from exc

from slam_map import SlamInfoSubscriber
from slam_service import SlamOperateClient, SlamResponse


def _print_resp(label: str, req: dict, resp: SlamResponse) -> None:
    print(f"\n[{label}]")
    print("request:", json.dumps(req, indent=2))
    print(f"response: code={resp.code} raw={resp.raw}")


def _print_info(sub: SlamInfoSubscriber, seconds: float = 0.5, include_robot_data: bool = False) -> None:
    t0 = time.time()
    while time.time() - t0 < seconds:
        info = sub.get_info()
        key = sub.get_key()
        if info:
            if include_robot_data:
                print("slam_info:", info)
            else:
                try:
                    payload = json.loads(info)
                    if payload.get("type") != "robot_data":
                        print("slam_info:", info)
                except Exception:
                    print("slam_info:", info)
        if key:
            print("slam_key:", key)
        time.sleep(0.05)


def _wait_for_pos_info(sub: SlamInfoSubscriber, timeout: float = 5.0) -> Optional[dict]:
    t0 = time.time()
    while time.time() - t0 < timeout:
        info = sub.get_info()
        if info:
            try:
                payload = json.loads(info)
                if payload.get("type") == "pos_info":
                    return payload
            except Exception:
                pass
        time.sleep(0.05)
    return None


def _wait_for_task_result(sub: SlamInfoSubscriber, timeout: float = 5.0) -> Optional[dict]:
    t0 = time.time()
    while time.time() - t0 < timeout:
        key = sub.get_key()
        if key:
            try:
                payload = json.loads(key)
                if payload.get("type") == "task_result":
                    return payload
            except Exception:
                pass
        time.sleep(0.05)
    return None


def main() -> None:
    p = argparse.ArgumentParser(
        description="Print request/response for slam_operate APIs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--iface", default="eth0", help="Network interface for DDS")
    p.add_argument("--domain-id", type=int, default=0, help="DDS domain id")
    p.add_argument("--save-path", default="/home/unitree/test1.pcd", help="PCD save path on robot")
    p.add_argument("--load-path", default="/home/unitree/test1.pcd", help="PCD load path on robot")
    p.add_argument("--goal-x", type=float, default=1.0, help="pose_nav target x")
    p.add_argument("--goal-y", type=float, default=0.0, help="pose_nav target y")
    p.add_argument("--goal-z", type=float, default=0.0, help="pose_nav target z")
    p.add_argument("--goal-qx", type=float, default=0.0, help="pose_nav target qx")
    p.add_argument("--goal-qy", type=float, default=0.0, help="pose_nav target qy")
    p.add_argument("--goal-qz", type=float, default=0.0, help="pose_nav target qz")
    p.add_argument("--goal-qw", type=float, default=1.0, help="pose_nav target qw")
    p.add_argument("--slam-info-topic", default="rt/slam_info", help="SLAM info topic")
    p.add_argument("--slam-key-topic", default="rt/slam_key_info", help="SLAM key info topic")
    p.add_argument("--pause", action="store_true", help="Call pause navigation (1201)")
    p.add_argument("--resume", action="store_true", help="Call resume navigation (1202)")
    p.add_argument("--wait-pos-info", action="store_true", help="Wait for slam_info pos_info and print it")
    p.add_argument("--wait-task-result", action="store_true", help="Wait for slam_key task_result and print it")
    p.add_argument("--include-robot-data", action="store_true", help="Print robot_data messages too")
    p.add_argument("--only", choices=[
        "start_mapping",
        "end_mapping",
        "init_pose",
        "pose_nav",
        "pause",
        "resume",
        "close_slam",
    ], default=None, help="Run only a single API")
    args = p.parse_args()

    ChannelFactoryInitialize(args.domain_id, args.iface)

    info_sub = SlamInfoSubscriber(args.slam_info_topic, args.slam_key_topic)
    info_sub.start()

    client = SlamOperateClient()
    client.Init()
    client.SetTimeout(10.0)

    def do_start_mapping() -> None:
        req = {"data": {"slam_type": "indoor"}}
        resp = client.start_mapping("indoor")
        _print_resp("start_mapping (1801)", req, resp)
        _print_info(info_sub, include_robot_data=args.include_robot_data)
        if args.wait_pos_info:
            payload = _wait_for_pos_info(info_sub)
            if payload:
                print("pos_info:", json.dumps(payload, indent=2))
            else:
                print("pos_info: timeout")

    def do_end_mapping() -> None:
        req = {"data": {"address": args.save_path}}
        resp = client.end_mapping(args.save_path)
        _print_resp("end_mapping (1802)", req, resp)
        _print_info(info_sub, include_robot_data=args.include_robot_data)

    def do_init_pose() -> None:
        req = {
            "data": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "q_x": 0.0,
                "q_y": 0.0,
                "q_z": 0.0,
                "q_w": 1.0,
                "address": args.load_path,
            }
        }
        resp = client.init_pose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, args.load_path)
        _print_resp("init_pose (1804)", req, resp)
        _print_info(info_sub, include_robot_data=args.include_robot_data)
        if args.wait_pos_info:
            payload = _wait_for_pos_info(info_sub)
            if payload:
                print("pos_info:", json.dumps(payload, indent=2))
            else:
                print("pos_info: timeout")

    def do_pose_nav() -> None:
        req = {
            "data": {
                "targetPose": {
                    "x": args.goal_x,
                    "y": args.goal_y,
                    "z": args.goal_z,
                    "q_x": args.goal_qx,
                    "q_y": args.goal_qy,
                    "q_z": args.goal_qz,
                    "q_w": args.goal_qw,
                },
                "mode": 1,
            }
        }
        resp = client.pose_nav(
            args.goal_x,
            args.goal_y,
            args.goal_z,
            args.goal_qx,
            args.goal_qy,
            args.goal_qz,
            args.goal_qw,
            mode=1,
        )
        _print_resp("pose_nav (1102)", req, resp)
        _print_info(info_sub, seconds=1.0, include_robot_data=args.include_robot_data)
        if args.wait_task_result:
            payload = _wait_for_task_result(info_sub, timeout=10.0)
            if payload:
                print("task_result:", json.dumps(payload, indent=2))
            else:
                print("task_result: timeout")

    def do_pause() -> None:
        req = {"data": {}}
        resp = client.pause_nav()
        _print_resp("pause_nav (1201)", req, resp)
        _print_info(info_sub, include_robot_data=args.include_robot_data)

    def do_resume() -> None:
        req = {"data": {}}
        resp = client.resume_nav()
        _print_resp("resume_nav (1202)", req, resp)
        _print_info(info_sub, include_robot_data=args.include_robot_data)

    def do_close_slam() -> None:
        req = {"data": {}}
        resp = client.close_slam()
        _print_resp("close_slam (1901)", req, resp)
        _print_info(info_sub, include_robot_data=args.include_robot_data)

    if args.only:
        {
            "start_mapping": do_start_mapping,
            "end_mapping": do_end_mapping,
            "init_pose": do_init_pose,
            "pose_nav": do_pose_nav,
            "pause": do_pause,
            "resume": do_resume,
            "close_slam": do_close_slam,
        }[args.only]()
        return

    # Default: run through the full list in the same order as keyDemo
    do_start_mapping()
    do_end_mapping()
    do_init_pose()
    do_pose_nav()
    if args.pause:
        do_pause()
    if args.resume:
        do_resume()
    do_close_slam()


if __name__ == "__main__":
    main()
