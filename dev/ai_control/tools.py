from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

from ai_control.robot_backend import RobotBackend

# Mirrors sdk_lib.HL_ARM_ACTIONS keys (minus "release arm", which fires
# automatically after a timed gesture). Kept as a plain constant here so this
# module doesn't have to import sdk_lib (and its hard Unitree SDK dependency)
# just to build a prompt.
GESTURE_NAMES = (
    "two-hand kiss",
    "left kiss",
    "right kiss",
    "hands up",
    "clap",
    "high five",
    "hug",
    "heart",
    "right heart",
    "reject",
    "right hand up",
    "x-ray",
    "face wave",
    "high wave",
    "shake hand",
)

# Keep model-proposed motion deliberately conservative.  These limits apply at
# the final dispatch boundary, so they still protect the real backend when a
# caller bypasses the model's prompt/schema and invokes ``dispatch`` directly.
MAX_LINEAR_SPEED_MPS = 1.0
MAX_ANGULAR_SPEED_RAD_S = 2.0
MAX_MOVE_DURATION_S = 30.0
MAX_NAVIGATION_DISTANCE_M = 50.0
VALID_HANDS = frozenset({"left", "right", "both"})


@dataclass
class ToolSpec:
    name: str
    category: str
    description: str
    args: dict[str, str]  # arg_name -> human-readable type/description
    handler: Callable[[RobotBackend, dict[str, Any]], str]
    requires_confirmation: bool = True


def _finite_number(args: dict[str, Any], name: str, default: float | None = None) -> float:
    value = args.get(name, default)
    if value is None:
        raise ValueError(f"missing numeric argument {name!r}")
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number, not a boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _bounded_number(
    args: dict[str, Any],
    name: str,
    *,
    default: float | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    number = _finite_number(args, name, default)
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return number


def _hand(args: dict[str, Any]) -> str:
    hand = str(args.get("hand", "right")).strip().lower()
    if hand not in VALID_HANDS:
        raise ValueError("hand must be 'left', 'right', or 'both'")
    return hand


def _text(args: dict[str, Any], name: str, *, max_length: int = 500) -> str:
    text = str(args.get(name, "")).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    if len(text) > max_length:
        raise ValueError(f"{name} is too long (maximum {max_length} characters)")
    return text


def _move(backend: RobotBackend, args: dict[str, Any]) -> str:
    vx = _bounded_number(args, "vx", default=0.0, minimum=-MAX_LINEAR_SPEED_MPS, maximum=MAX_LINEAR_SPEED_MPS)
    vy = _bounded_number(args, "vy", default=0.0, minimum=-MAX_LINEAR_SPEED_MPS, maximum=MAX_LINEAR_SPEED_MPS)
    vyaw = _bounded_number(
        args,
        "vyaw",
        default=0.0,
        minimum=-MAX_ANGULAR_SPEED_RAD_S,
        maximum=MAX_ANGULAR_SPEED_RAD_S,
    )
    duration = _bounded_number(args, "duration", default=1.0, minimum=0.01, maximum=MAX_MOVE_DURATION_S)
    if abs(vyaw) > MAX_ANGULAR_SPEED_RAD_S:
        raise ValueError(f"vyaw must be between {-MAX_ANGULAR_SPEED_RAD_S} and {MAX_ANGULAR_SPEED_RAD_S}")
    return backend.move(
        vx=vx,
        vy=vy,
        vyaw=vyaw,
        duration=duration,
    )


def _navigate_to(backend: RobotBackend, args: dict[str, Any]) -> str:
    x = _bounded_number(args, "x", minimum=-MAX_NAVIGATION_DISTANCE_M, maximum=MAX_NAVIGATION_DISTANCE_M)
    y = _bounded_number(args, "y", minimum=-MAX_NAVIGATION_DISTANCE_M, maximum=MAX_NAVIGATION_DISTANCE_M)
    yaw = _finite_number(args, "yaw", 0.0)
    return backend.navigate_to(x=x, y=y, yaw=yaw)


def _stop(backend: RobotBackend, args: dict[str, Any]) -> str:
    return backend.stop()


def _hand_open(backend: RobotBackend, args: dict[str, Any]) -> str:
    return backend.hand_open(hand=_hand(args))


def _hand_close(backend: RobotBackend, args: dict[str, Any]) -> str:
    return backend.hand_close(hand=_hand(args))


def _gesture(backend: RobotBackend, args: dict[str, Any]) -> str:
    name = str(args["name"]).strip().lower()
    if name not in GESTURE_NAMES:
        raise ValueError(f"Unknown gesture {name!r}. Valid gestures: {', '.join(GESTURE_NAMES)}")
    return backend.gesture(name)


def _release_arms(backend: RobotBackend, args: dict[str, Any]) -> str:
    return backend.release_arms()


def _say(backend: RobotBackend, args: dict[str, Any]) -> str:
    return backend.say(text=_text(args, "text"))


def _navbot_command(backend: RobotBackend, args: dict[str, Any]) -> str:
    return backend.navbot_command(text=_text(args, "text"))


TOOL_SPECS: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in (
        ToolSpec(
            name="move",
            category="navigation",
            description="Drive at a body-frame velocity for a fixed duration, then stop.",
            args={"vx": "m/s forward", "vy": "m/s sideways (+left)", "vyaw": "rad/s turn", "duration": "seconds"},
            handler=_move,
        ),
        ToolSpec(
            name="navigate_to",
            category="navigation",
            description="Close-loop navigate to a pose relative to the robot's current odometry frame.",
            args={"x": "meters forward", "y": "meters left", "yaw": "radians, relative heading change"},
            handler=_navigate_to,
        ),
        ToolSpec(
            name="stop",
            category="navigation",
            description="Immediately stop all locomotion.",
            args={},
            handler=_stop,
        ),
        ToolSpec(
            name="hand_open",
            category="endeffector",
            description="Open the end-effector (Dex3 hand).",
            args={"hand": "'left', 'right', or 'both'"},
            handler=_hand_open,
        ),
        ToolSpec(
            name="hand_close",
            category="endeffector",
            description="Close the end-effector (Dex3 hand) into a grip pose.",
            args={"hand": "'left', 'right', or 'both'"},
            handler=_hand_close,
        ),
        ToolSpec(
            name="gesture",
            category="gesture",
            description="Play a predefined high-level arm gesture.",
            args={"name": f"one of: {', '.join(GESTURE_NAMES)}"},
            handler=_gesture,
        ),
        ToolSpec(
            name="release_arms",
            category="gesture",
            description="Release the arms back to their neutral controlled state.",
            args={},
            handler=_release_arms,
        ),
        ToolSpec(
            name="say",
            category="speaker",
            description="Speak text out loud through the robot's loudspeaker.",
            args={"text": "text to announce"},
            handler=_say,
        ),
        ToolSpec(
            name="navbot_command",
            category="slam_navigation",
            description=(
                "Forward a text command to nav_bot.py on /model_api/navbot_command. "
                "Use for SLAM mapping, relocation, named points, and named-point navigation."
            ),
            args={
                "text": (
                    "nav_bot.py command text, e.g. 'start mapping', "
                    "'save current point as kitchen', 'go to kitchen'"
                )
            },
            handler=_navbot_command,
        ),
    )
}


def prompt_block() -> str:
    """Renders the tool registry as text for the thinker model's system prompt."""
    lines = []
    for spec in TOOL_SPECS.values():
        arg_desc = ", ".join(f"{name} ({desc})" for name, desc in spec.args.items()) or "no arguments"
        lines.append(f"- {spec.name} [{spec.category}]: {spec.description} Args: {arg_desc}.")
    return "\n".join(lines)


def dispatch(name: str, args: dict[str, Any], backend: RobotBackend) -> str:
    spec = TOOL_SPECS.get(name)
    if spec is None:
        raise KeyError(f"Unknown tool {name!r}. Known tools: {', '.join(TOOL_SPECS)}")
    return spec.handler(backend, args)
