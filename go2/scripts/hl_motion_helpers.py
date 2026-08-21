"""Safety-conscious motion helpers shared by the GO2 high-level tasks."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any


def _finite(value: float, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be a finite number")
    return parsed


def wrap_angle(angle: float) -> float:
    """Normalize an angle to the interval [-pi, pi]."""
    return (_finite(angle, "angle") + math.pi) % (2.0 * math.pi) - math.pi


def wait_for_pose(is_ready: Callable[[], bool], timeout: float, poll: float = 0.05) -> bool:
    """Wait for fresh pose inputs using a monotonic deadline."""
    deadline = time.monotonic() + max(0.0, _finite(timeout, "timeout"))
    poll = max(0.01, _finite(poll, "poll"))
    while time.monotonic() < deadline:
        if is_ready():
            return True
        time.sleep(poll)
    return bool(is_ready())


def turn_to_delta(
    client: Any,
    get_yaw: Callable[[], float | None],
    delta_yaw: float,
    yaw_rate: float,
    *,
    tick: float = 0.05,
    timeout: float = 8.0,
) -> None:
    """Turn until the measured yaw delta is reached or fail closed."""
    start = get_yaw()
    if start is None or not math.isfinite(start):
        raise RuntimeError("IMU yaw not available")

    target = abs(_finite(delta_yaw, "delta_yaw"))
    yaw_rate = _finite(yaw_rate, "yaw_rate")
    tick = max(0.01, _finite(tick, "tick"))
    deadline = time.monotonic() + max(0.0, _finite(timeout, "timeout"))
    reached = target == 0.0
    try:
        while not reached and time.monotonic() < deadline:
            current = get_yaw()
            if current is None or not math.isfinite(current):
                time.sleep(tick)
                continue
            progress = abs(wrap_angle(current - start))
            if progress >= target:
                reached = True
                break
            client.Move(0.0, 0.0, yaw_rate)
            time.sleep(tick)
    finally:
        client.StopMove()

    if not reached:
        raise TimeoutError(
            f"Turn timed out after {timeout:.1f}s before reaching {math.degrees(target):.1f} degrees"
        )


def walk_distance(
    client: Any,
    get_position: Callable[[], tuple[float, float] | None],
    speed: float,
    distance: float,
    *,
    tick: float = 0.1,
    timeout: float = 20.0,
) -> None:
    """Walk until measured displacement is reached or fail closed."""
    start = get_position()
    if start is None:
        raise RuntimeError("No position source available (odom/sportstate)")

    speed = _finite(speed, "speed")
    target = max(0.0, _finite(distance, "distance"))
    tick = max(0.01, _finite(tick, "tick"))
    deadline = time.monotonic() + max(0.0, _finite(timeout, "timeout"))
    reached = target == 0.0
    try:
        while not reached and time.monotonic() < deadline:
            position = get_position()
            if position is None:
                time.sleep(tick)
                continue
            dx = position[0] - start[0]
            dy = position[1] - start[1]
            if math.hypot(dx, dy) >= target:
                reached = True
                break
            client.Move(speed, 0.0, 0.0)
            time.sleep(tick)
    finally:
        client.StopMove()

    if not reached:
        raise TimeoutError(f"Walk timed out after {timeout:.1f}s before reaching {target:.2f} m")
