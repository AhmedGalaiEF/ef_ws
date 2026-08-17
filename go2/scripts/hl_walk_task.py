from __future__ import annotations

import argparse
import math
import time
from typing import Any

try:
    from .hl_motion_helpers import turn_to_delta, wait_for_pose, walk_distance, wrap_angle
except ImportError:  # Direct execution: python go2/scripts/hl_walk_task.py
    from hl_motion_helpers import turn_to_delta, wait_for_pose, walk_distance, wrap_angle

last_imu_yaw: float | None = None
last_sport_pos: list[float] | None = None
last_odom_pos: list[float] | None = None


def _wrap_angle(a):
    return wrap_angle(a)


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite value > 0")
    return parsed


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _yaw_cb(msg: Any) -> None:
    global last_imu_yaw
    yaw = _finite_float(msg.imu_state.rpy[2])
    if yaw is not None:
        last_imu_yaw = yaw


def _sport_cb(msg: Any) -> None:
    global last_sport_pos
    position = [_finite_float(v) for v in msg.position]
    if position and all(v is not None for v in position):
        last_sport_pos = [float(v) for v in position]


def _odom_cb(msg: Any) -> None:
    global last_odom_pos
    position = [
        _finite_float(msg.pose.pose.position.x),
        _finite_float(msg.pose.pose.position.y),
    ]
    if all(v is not None for v in position):
        last_odom_pos = [float(v) for v in position]


def _get_pose_xy() -> tuple[float, float] | None:
    if last_odom_pos is not None:
        return last_odom_pos[0], last_odom_pos[1]
    if last_sport_pos is not None and len(last_sport_pos) >= 2:
        return last_sport_pos[0], last_sport_pos[1]
    return None


def _wait_for_pose(timeout: float) -> bool:
    return wait_for_pose(lambda: last_imu_yaw is not None and _get_pose_xy() is not None, timeout)


def _turn_to_delta(client: Any, delta_yaw: float, yaw_rate: float, tick: float = 0.05, timeout: float = 8.0) -> None:
    turn_to_delta(client, lambda: last_imu_yaw, delta_yaw, yaw_rate, tick=tick, timeout=timeout)


def _walk_distance(client: Any, speed: float, distance: float, tick: float = 0.1, timeout: float = 20.0) -> None:
    walk_distance(client, _get_pose_xy, speed, distance, tick=tick, timeout=timeout)


def main() -> int:
    parser = argparse.ArgumentParser(description="Go2 HL task: forward, turn, side, turn, return.")
    parser.add_argument("--iface", default="enp1s0")
    parser.add_argument("--domain-id", type=_nonnegative_int, default=0)
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
