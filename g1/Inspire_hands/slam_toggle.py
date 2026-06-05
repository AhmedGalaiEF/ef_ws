#!/usr/bin/env python3
"""
Control Unitree G1 SLAM operations from the command line.

Examples:
  ./slam_toggle.py status --iface eth0 --domain-id 0
  ./slam_toggle.py start --iface eth0 --domain-id 0
  ./slam_toggle.py stop --iface eth0 --domain-id 0
  ./slam_toggle.py save /home/unitree/map.pcd --iface eth0 --domain-id 0
  ./slam_toggle.py load /home/unitree/map.pcd --iface eth0 --domain-id 0
  ./slam_toggle.py nav --x 1.0 --y 0.0 --yaw 0.0 --iface eth0 --domain-id 0
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parent
MODULES = ROOT / "modules"
if str(MODULES) not in sys.path:
    sys.path.insert(0, str(MODULES))


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Control Unitree G1 /api/slam_operate actions."
    )
    parser.add_argument(
        "action",
        choices=(
            "status",
            "start",
            "on",
            "stop",
            "off",
            "close",
            "save",
            "end",
            "load",
            "init",
            "nav",
            "pause",
            "resume",
        ),
        help="SLAM action. 'on' is start, 'off' is stop, 'end' is save, 'init' is load.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="Map path for save/end or load/init, for example /home/unitree/map.pcd.",
    )
    parser.add_argument(
        "--iface",
        help="Network interface for Unitree DDS, for example eth0 or enx....",
    )
    parser.add_argument(
        "--domain-id",
        type=int,
        default=0,
        help="DDS domain id passed to ChannelFactoryInitialize. Default: 0.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="SLAM RPC timeout in seconds. Default: 10.0.",
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=1.0,
        help="Seconds to wait for status/task output after an action. Default: 1.0.",
    )
    parser.add_argument(
        "--slam-type",
        default="indoor",
        help="SLAM mapping type for start/on. Default: indoor.",
    )
    parser.add_argument("--x", type=float, default=0.0, help="X pose/goal in meters.")
    parser.add_argument("--y", type=float, default=0.0, help="Y pose/goal in meters.")
    parser.add_argument("--z", type=float, default=0.0, help="Z pose in meters. Default: 0.")
    parser.add_argument("--yaw", type=float, default=0.0, help="Yaw in radians for load/nav.")
    parser.add_argument("--qx", type=float, help="Quaternion x for load/init.")
    parser.add_argument("--qy", type=float, help="Quaternion y for load/init.")
    parser.add_argument("--qz", type=float, help="Quaternion z for load/init.")
    parser.add_argument("--qw", type=float, help="Quaternion w for load/init.")
    parser.add_argument("--mode", type=int, default=1, help="Navigation mode for nav. Default: 1.")
    parser.add_argument(
        "--info-topic",
        default="rt/slam_info",
        help="SLAM info topic. Default: rt/slam_info.",
    )
    parser.add_argument(
        "--key-topic",
        default="rt/slam_key_info",
        help="SLAM key info topic. Default: rt/slam_key_info.",
    )
    return parser.parse_args(argv)


def init_channel(args: argparse.Namespace) -> None:
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    except ImportError as exc:
        raise SystemExit(
            "error: unitree_sdk2py is not importable. Activate the Unitree SDK environment first."
        ) from exc

    if args.iface:
        ChannelFactoryInitialize(args.domain_id, args.iface)
    else:
        ChannelFactoryInitialize(args.domain_id)


def make_client(args: argparse.Namespace) -> Any:
    try:
        from sdk_slam import SlamOperateClient
    except ImportError as exc:
        raise SystemExit(
            "error: cannot import modules/sdk_slam.py or its dependencies."
        ) from exc

    client = SlamOperateClient()
    client.Init()
    client.SetTimeout(args.timeout)
    return client


def make_info_subscriber(args: argparse.Namespace) -> Any:
    try:
        from sdk_slam import SlamInfoSubscriber
    except ImportError as exc:
        raise SystemExit(
            "error: cannot import modules/sdk_slam.py or its dependencies."
        ) from exc

    sub = SlamInfoSubscriber(args.info_topic, args.key_topic)
    sub.start()
    return sub


def quaternion_from_args(args: argparse.Namespace) -> tuple[float, float, float, float]:
    values = (args.qx, args.qy, args.qz, args.qw)
    if any(value is not None for value in values):
        if not all(value is not None for value in values):
            raise SystemExit("error: pass all of --qx --qy --qz --qw, or pass only --yaw")
        return (float(args.qx), float(args.qy), float(args.qz), float(args.qw))
    qz = math.sin(float(args.yaw) * 0.5)
    qw = math.cos(float(args.yaw) * 0.5)
    return (0.0, 0.0, qz, qw)


def require_path(args: argparse.Namespace) -> str:
    if not args.path:
        raise SystemExit(f"error: action '{args.action}' requires PATH")
    return args.path


def format_jsonish(raw: Any) -> str:
    if raw is None:
        return "None"
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except Exception:
            return raw
        return json.dumps(parsed, sort_keys=True)
    try:
        return json.dumps(raw, sort_keys=True)
    except TypeError:
        return str(raw)


def print_response(label: str, response: Any) -> int:
    code = int(response.code)
    print(f"{label}: code={code} raw={format_jsonish(response.raw)}")
    return code


def print_status(args: argparse.Namespace, wait: float) -> None:
    sub = make_info_subscriber(args)
    deadline = time.time() + max(0.0, wait)
    info = sub.get_info()
    key = sub.get_key()
    pose = sub.get_pose()
    while time.time() < deadline and info is None and key is None and pose is None:
        time.sleep(0.05)
        info = sub.get_info()
        key = sub.get_key()
        pose = sub.get_pose()

    print(f"info: {info if info is not None else '<none>'}")
    print(f"key: {key if key is not None else '<none>'}")
    if pose is None:
        print("pose: <none>")
    else:
        print(f"pose: x={pose[0]:.3f} y={pose[1]:.3f} yaw={pose[2]:.3f}")


def run(args: argparse.Namespace) -> int:
    action = {
        "on": "start",
        "off": "stop",
        "close": "stop",
        "end": "save",
        "init": "load",
    }.get(args.action, args.action)

    init_channel(args)

    if action == "status":
        print_status(args, args.wait)
        return 0

    client = make_client(args)

    if action == "start":
        code = print_response("start_mapping", client.start_mapping(args.slam_type))
    elif action == "stop":
        code = print_response("close_slam", client.close_slam())
    elif action == "save":
        code = print_response("end_mapping", client.end_mapping(require_path(args)))
    elif action == "load":
        qx, qy, qz, qw = quaternion_from_args(args)
        code = print_response(
            "init_pose",
            client.init_pose(
                float(args.x),
                float(args.y),
                float(args.z),
                qx,
                qy,
                qz,
                qw,
                require_path(args),
            ),
        )
    elif action == "nav":
        qx, qy, qz, qw = quaternion_from_args(args)
        code = print_response(
            "pose_nav",
            client.pose_nav(
                float(args.x),
                float(args.y),
                float(args.z),
                qx,
                qy,
                qz,
                qw,
                mode=int(args.mode),
            ),
        )
    elif action == "pause":
        code = print_response("pause_nav", client.pause_nav())
    elif action == "resume":
        code = print_response("resume_nav", client.resume_nav())
    else:
        raise SystemExit(f"error: unsupported action {action!r}")

    if args.wait > 0:
        print_status(args, args.wait)
    return code


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
