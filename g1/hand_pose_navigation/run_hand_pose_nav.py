#!/usr/bin/env python3
"""
Direct runner for the hand_pose_navigation stack.

This avoids ROS package discovery and colcon packaging. It still uses rclpy
internally because the pipeline relies on ROS 2 TF.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
G1_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if G1_DIR not in sys.path:
    sys.path.insert(0, G1_DIR)
MODULES_DIR = os.path.join(G1_DIR, "modules")
if MODULES_DIR not in sys.path:
    sys.path.insert(0, MODULES_DIR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hand_pose_navigation directly from Python.",
    )
    parser.add_argument("--arm", choices=("left", "right"), default="right")
    parser.add_argument(
        "--detection-method",
        choices=("aruco", "color", "center"),
        default="aruco",
    )
    parser.add_argument("--aruco-id", type=int, default=0)
    parser.add_argument("--marker-size-m", type=float, default=0.05)
    parser.add_argument("--standoff-m", type=float, default=0.08)
    parser.add_argument("--rate-hz", type=float, default=10.0)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument(
        "--ik-solver",
        choices=("dls", "scipy", "pin"),
        default="dls",
    )
    parser.add_argument("--iface", default="eth0")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument(
        "--use-ros-tf",
        action="store_true",
        help=(
            "Use the ROS 2 TF-based node path. Default uses direct "
            "in-process TF."
        ),
    )
    parser.add_argument(
        "--ros-rmw",
        default="",
        help=(
            "Optional ROS 2 RMW override. Empty default preserves the shell "
            "environment."
        ),
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help=(
            "Use the built-in mock robot instead of connecting to sdk_client."
        ),
    )
    parser.add_argument("--web-host", default="0.0.0.0")
    parser.add_argument("--web-port", type=int, default=8088)
    parser.add_argument("--camera-x", type=float, default=0.0)
    parser.add_argument("--camera-y", type=float, default=0.0)
    parser.add_argument("--camera-z", type=float, default=0.0)
    parser.add_argument("--camera-roll", type=float, default=0.0)
    parser.add_argument("--camera-pitch", type=float, default=0.0)
    parser.add_argument("--camera-yaw", type=float, default=0.0)
    parser.add_argument(
        "--no-web",
        action="store_true",
        help="Run the controller without the status webpage.",
    )
    parser.add_argument(
        "--no-sdk-preinit",
        action="store_true",
        help=(
            "Let HandPoseNavNode construct sdk_client.Robot after ROS nodes "
            "start."
        ),
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Dict:
    return {
        "arm": args.arm,
        "detection_method": args.detection_method,
        "aruco_id": args.aruco_id,
        "marker_size_m": args.marker_size_m,
        "standoff_m": args.standoff_m,
        "rate_hz": args.rate_hz,
        "timeout_s": args.timeout_s,
        "ik_solver": args.ik_solver,
        "iface": args.iface,
        "domain_id": args.domain_id,
        "mock": args.mock,
        "camera_x": args.camera_x,
        "camera_y": args.camera_y,
        "camera_z": args.camera_z,
        "camera_roll": args.camera_roll,
        "camera_pitch": args.camera_pitch,
        "camera_yaw": args.camera_yaw,
    }


def main() -> int:
    args = parse_args()
    os.makedirs("/tmp/hand_pose_nav_ros_log", exist_ok=True)
    os.environ.setdefault("ROS_LOG_DIR", "/tmp/hand_pose_nav_ros_log")

    if not args.use_ros_tf:
        from hand_pose_navigation.direct_nav import DirectHandPoseNav
        from hand_pose_navigation.web_status import start_status_server

        nav = DirectHandPoseNav(config_from_args(args))
        server = None
        import time

        try:
            if not args.no_web:
                server, _thread = start_status_server(
                    nav.status_snapshot,
                    host=args.web_host,
                    port=args.web_port,
                )
                print(
                    f"Status webpage: http://{args.web_host}:{args.web_port}/"
                )

            if args.no_web:
                while nav.status_snapshot().get("running", False):
                    time.sleep(0.2)
            else:
                while True:
                    time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        finally:
            if server is not None:
                server.shutdown()
                server.server_close()
            nav.shutdown()
        return 0

    if args.ros_rmw:
        os.environ["RMW_IMPLEMENTATION"] = args.ros_rmw
    if args.mock:
        os.environ["ROS_LOCALHOST_ONLY"] = "1"
        cyclone_uri = "/tmp/hand_pose_nav_cyclonedds.xml"
        with open(cyclone_uri, "w", encoding="utf-8") as f:
            f.write(
                "<CycloneDDS><Domain id=\"any\"><General>"
                "<NetworkInterfaceAddress>lo</NetworkInterfaceAddress>"
                "<AllowMulticast>false</AllowMulticast>"
                "</General></Domain></CycloneDDS>"
            )
        os.environ["CYCLONEDDS_URI"] = cyclone_uri

    preconnected_robot = None
    sdk_error = ""
    if not args.mock and not args.no_sdk_preinit:
        try:
            from sdk_client import Robot

            preconnected_robot = Robot(
                iface=args.iface,
                domain_id=args.domain_id,
                auto_start_sensors=True,
            )
            print("[SDK] Robot pre-initialized before ROS node startup.")
        except Exception as exc:
            sdk_error = repr(exc)
            print(f"[SDK] Robot pre-initialization failed: {sdk_error}")

    import rclpy
    from hand_pose_navigation.hand_pose_nav_node import HandPoseNavNode
    from hand_pose_navigation.web_status import start_status_server

    rclpy.init(args=None)
    config = config_from_args(args)
    if sdk_error:
        config["sdk_preinit_error"] = sdk_error
    node = HandPoseNavNode(config=config, robot=preconnected_robot)
    server = None

    try:
        if not args.no_web:
            server, _thread = start_status_server(
                node.status_snapshot,
                host=args.web_host,
                port=args.web_port,
            )
            print(f"Status webpage: http://{args.web_host}:{args.web_port}/")

        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
