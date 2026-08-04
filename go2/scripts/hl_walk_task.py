from __future__ import annotations

import argparse
import math
import time
from typing import Any

last_imu_yaw: float | None = None
last_sport_pos: list[float] | None = None
last_odom_pos: list[float] | None = None


def _wrap_angle(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def _yaw_cb(msg: Any) -> None:
    global last_imu_yaw
    last_imu_yaw = float(msg.imu_state.rpy[2])


def _sport_cb(msg: Any) -> None:
    global last_sport_pos
    last_sport_pos = [float(v) for v in msg.position]


def _odom_cb(msg: Any) -> None:
    global last_odom_pos
    last_odom_pos = [
        float(msg.pose.pose.position.x),
        float(msg.pose.pose.position.y),
    ]


def _get_pose_xy() -> tuple[float, float] | None:
    if last_odom_pos is not None:
        return last_odom_pos[0], last_odom_pos[1]
    if last_sport_pos is not None and len(last_sport_pos) >= 2:
        return last_sport_pos[0], last_sport_pos[1]
    return None


def _wait_for_pose(timeout: float) -> bool:
    deadline = time.monotonic() + max(0.0, float(timeout))
    while time.monotonic() < deadline:
        if last_imu_yaw is not None and _get_pose_xy() is not None:
            return True
        time.sleep(0.05)
    return last_imu_yaw is not None and _get_pose_xy() is not None


def _turn_to_delta(client: Any, delta_yaw: float, yaw_rate: float, tick: float = 0.05, timeout: float = 8.0) -> None:
    if last_imu_yaw is None:
        raise RuntimeError("IMU yaw not available")
    start = last_imu_yaw
    target = abs(delta_yaw)
    end_time = time.monotonic() + max(0.0, float(timeout))
    reached = False
    try:
        while time.monotonic() < end_time:
            if last_imu_yaw is None:
                time.sleep(tick)
                continue
            progress = abs(_wrap_angle(last_imu_yaw - start))
            if progress >= target:
                reached = True
                break
            client.Move(0.0, 0.0, yaw_rate)
            time.sleep(tick)
    finally:
        client.StopMove()
    if not reached:
        raise TimeoutError(f"Turn timed out after {timeout:.1f}s before reaching {math.degrees(target):.1f} degrees")


def _walk_distance(client: Any, speed: float, distance: float, tick: float = 0.1, timeout: float = 20.0) -> None:
    start = _get_pose_xy()
    if start is None:
        raise RuntimeError("No position source available (odom/sportstate)")
    end_time = time.monotonic() + max(0.0, float(timeout))
    reached = False
    try:
        while time.monotonic() < end_time:
            pos = _get_pose_xy()
            if pos is None:
                time.sleep(tick)
                continue
            dx = pos[0] - start[0]
            dy = pos[1] - start[1]
            if math.hypot(dx, dy) >= distance:
                reached = True
                break
            client.Move(speed, 0.0, 0.0)
            time.sleep(tick)
    finally:
        client.StopMove()
    if not reached:
        raise TimeoutError(f"Walk timed out after {timeout:.1f}s before reaching {distance:.2f} m")


def main() -> int:
    parser = argparse.ArgumentParser(description="Go2 HL task: forward, turn, side, turn, return.")
    parser.add_argument("--iface", default="enp1s0")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--speed", type=_positive_float, default=0.3, help="forward speed (m/s)")
    parser.add_argument("--forward-dist", type=_positive_float, default=1.0, help="forward distance (m)")
    parser.add_argument("--side-dist", type=_positive_float, default=0.5, help="side leg distance (m)")
    parser.add_argument("--turn-dir", choices=["right", "left"], default="right")
    parser.add_argument("--turn-angle-deg", type=_positive_float, default=90.0)
    parser.add_argument("--yaw-rate", type=_positive_float, default=0.5, help="yaw rate (rad/s)")
    parser.add_argument("--free-walk", action="store_true", help="enable free walk before task")
    parser.add_argument("--pose-wait", type=_positive_float, default=2.0, help="seconds to wait for IMU and pose data")
    parser.add_argument("--yes", action="store_true", help="confirm that this script may move the robot")
    args = parser.parse_args()

    if not args.yes:
        print("This script moves the robot. Re-run with --yes to confirm.")
        return 2

    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, SportModeState_
        from unitree_sdk2py.go2.sport.sport_client import SportClient
    except ImportError as exc:
        raise SystemExit(
            "unitree_sdk2py is not installed. Install it with:\n"
            "  pip install -e <path-to-unitree_sdk2_python>"
        ) from exc

    ChannelFactoryInitialize(args.domain_id, args.iface)

    low_sub = ChannelSubscriber("rt/lowstate", LowState_)
    low_sub.Init(_yaw_cb, 10)
    sport_sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
    sport_sub.Init(_sport_cb, 10)
    odom_sub = ChannelSubscriber("rt/odom", Odometry_)
    odom_sub.Init(_odom_cb, 10)

    client = SportClient()
    client.SetTimeout(5.0)
    client.Init()

    if args.free_walk:
        client.FreeWalk()
        time.sleep(0.2)

    yaw_sign = -1.0 if args.turn_dir == "right" else 1.0
    yaw_rate = yaw_sign * abs(args.yaw_rate)
    turn_duration = math.radians(args.turn_angle_deg) / max(1e-3, abs(args.yaw_rate))

    try:
        if not _wait_for_pose(args.pose_wait):
            raise RuntimeError("Timed out waiting for IMU and odom/sportstate data.")
        _walk_distance(client, args.speed, args.forward_dist)
        _turn_to_delta(client, math.radians(args.turn_angle_deg), yaw_rate, timeout=turn_duration + 3.0)
        _walk_distance(client, args.speed, args.side_dist)
        _turn_to_delta(client, math.radians(args.turn_angle_deg), yaw_rate, timeout=turn_duration + 3.0)
        _walk_distance(client, args.speed, args.forward_dist)
    finally:
        client.StopMove()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
