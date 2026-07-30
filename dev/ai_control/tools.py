from __future__ import annotations

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


@dataclass
class ToolSpec:
    name: str
    category: str
    description: str
    args: dict[str, str]  # arg_name -> human-readable type/description
    handler: Callable[[RobotBackend, dict[str, Any]], str]
    requires_confirmation: bool = True


def _move(backend: RobotBackend, args: dict[str, Any]) -> str:
    return backend.move(
        vx=float(args.get("vx", 0.0)),
        vy=float(args.get("vy", 0.0)),
        vyaw=float(args.get("vyaw", 0.0)),
        duration=float(args.get("duration", 1.0)),
    )


def _navigate_to(backend: RobotBackend, args: dict[str, Any]) -> str:
    return backend.navigate_to(x=float(args["x"]), y=float(args["y"]), yaw=float(args.get("yaw", 0.0)))


def _stop(backend: RobotBackend, args: dict[str, Any]) -> str:
    return backend.stop()


def _hand_open(backend: RobotBackend, args: dict[str, Any]) -> str:
    return backend.hand_open(hand=str(args.get("hand", "right")))


def _hand_close(backend: RobotBackend, args: dict[str, Any]) -> str:
    return backend.hand_close(hand=str(args.get("hand", "right")))


def _gesture(backend: RobotBackend, args: dict[str, Any]) -> str:
    name = str(args["name"]).strip().lower()
    if name not in GESTURE_NAMES:
        raise ValueError(f"Unknown gesture {name!r}. Valid gestures: {', '.join(GESTURE_NAMES)}")
    return backend.gesture(name)


def _say(backend: RobotBackend, args: dict[str, Any]) -> str:
    return backend.say(text=str(args["text"]))


def _navbot_command(backend: RobotBackend, args: dict[str, Any]) -> str:
    return backend.navbot_command(text=str(args["text"]))


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
            args={"text": "nav_bot.py command text, e.g. 'start mapping', 'save current point as kitchen', 'go to kitchen'"},
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
