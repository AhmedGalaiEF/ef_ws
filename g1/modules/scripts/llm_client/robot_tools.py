"""LLM-callable robot tools: locomotion, reaching, and gripping.

Build the (tools, tool_schemas) pair to pass into ``send_chat_with_tool_usage``::

    from llm_client import send_chat_with_tool_usage
    from llm_client.robot_tools import build_robot_tools
    from sdk_client import Robot

    robot = Robot("enp1s0")
    tools, schemas = build_robot_tools(robot)

    content, _ = send_chat_with_tool_usage(
        "gpt-oss-120b",
        [{"role": "user", "content": "Move forward half a meter and grab whatever is in front of you."}],
        tools=tools,
        tool_schemas=schemas,
    )
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Tuple

# Dex3 default open / closed finger pose (7 joints):
#   thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1
# Approximate values; tune per-hand if a grab feels too loose/tight.
_DEX3_OPEN = [0.0] * 7
_DEX3_CLOSED = [1.0, 1.4, 1.4, 1.5, 1.4, 1.5, 1.4]


def build_robot_tools(robot: Any) -> Tuple[Dict[str, Callable[..., Any]], List[Dict[str, Any]]]:
    """Return ``(tools, tool_schemas)`` bound to a Robot instance.

    ``tools`` is a name → callable map, ``tool_schemas`` is a list of OpenAI
    function-calling JSON schemas. Both go straight into
    ``send_chat_with_tool_usage(..., tools=tools, tool_schemas=schemas)``.
    """

    # ---- locomotion -------------------------------------------------------
    def move(
        direction: str,
        distance_m: float = 0.5,
        yaw_rad: float = 0.0,
        speed_mps: float = 0.3,
    ) -> str:
        """Drive the base in a cardinal direction with optional yaw."""
        d = str(direction).strip().lower()
        if d == "stop":
            robot.stop_moving()
            return "stopped"
        if d not in {"forward", "backward", "left", "right"}:
            return f"error: unknown direction '{direction}'"
        speed = max(0.05, min(0.6, float(speed_mps)))
        distance = max(0.0, float(distance_m))
        duration = distance / speed if speed > 0 else 0.0
        vx = vy = 0.0
        if d == "forward":
            vx = speed
        elif d == "backward":
            vx = -speed
        elif d == "left":
            vy = speed
        elif d == "right":
            vy = -speed
        vyaw = float(yaw_rad) / duration if duration > 0.05 else 0.0
        robot.move_for(duration, vx=vx, vy=vy, vyaw=vyaw)
        return (
            f"moved {d} {distance:.2f} m at {speed:.2f} m/s with yaw {yaw_rad:+.2f} rad "
            f"over {duration:.2f} s"
        )

    # ---- arm reach --------------------------------------------------------
    def reach_forward(
        height_m: float = 0.0,
        length_m: float = 0.4,
        arm: str = "right",
        duration_s: float = 3.0,
    ) -> str:
        """Reach the arm forward using the saved pose trajectory.

        ``height_m`` and ``length_m`` are accepted for compatibility with the
        tool schema but no longer tune the joint trajectory.
        """
        side = str(arm).strip().lower()
        if side not in {"left", "right"}:
            return f"error: arm must be 'left' or 'right', got '{arm}'"
        h = max(-0.30, min(0.30, float(height_m)))
        reach = max(0.10, min(0.60, float(length_m)))
        result = robot.extend_arm_forward(
            arm=side,
            duration_s=float(duration_s),
        )
        target = result.get("target_pose") if isinstance(result, dict) else None
        return (
            f"reached {side} arm using saved pose trajectory "
            f"(requested height={h:+.2f} m, length={reach:.2f} m); target_pose={target}"
        )

    # ---- gripping ---------------------------------------------------------
    def grab(
        hand: str = "right",
        resistance_threshold: float = 0.5,
        n_steps: int = 8,
        step_s: float = 0.15,
    ) -> str:
        """Slowly close the fingers; stop when finger torque exceeds threshold.

        ``resistance_threshold`` is in N·m on ``tau_est`` of any finger
        joint. Typical free-air closure stays well below 0.2; touching a
        firm object spikes toward 0.5–1.0.
        """
        side = str(hand).strip().lower()
        if side not in {"left", "right"}:
            return f"error: hand must be 'left' or 'right', got '{hand}'"
        n = max(1, int(n_steps))
        dt = max(0.05, float(step_s))
        thresh = max(0.0, float(resistance_threshold))

        robot.hand_open(side, hold_s=0.2)
        time.sleep(0.2)

        for step in range(1, n + 1):
            alpha = step / n
            targets = [
                (1.0 - alpha) * o + alpha * c
                for o, c in zip(_DEX3_OPEN, _DEX3_CLOSED)
            ]
            robot.hand_pose(targets, hand=side, hold_s=dt)
            time.sleep(0.05)
            msg, _ts = robot._get_hand_state_msg(side)
            if msg is None:
                continue
            _, _, taus = robot._extract_hand_joint_series(msg)
            valid = [abs(t) for t in taus if t is not None]
            tau_max = max(valid) if valid else 0.0
            if tau_max >= thresh:
                return (
                    f"resistance hit at {int(alpha * 100)}% closure "
                    f"(tau_max={tau_max:.2f} N·m, threshold={thresh:.2f}); held."
                )
        return "fully closed without exceeding resistance threshold"

    def release(hand: str = "right") -> str:
        """Open the fingers fully."""
        side = str(hand).strip().lower()
        if side not in {"left", "right"}:
            return f"error: hand must be 'left' or 'right', got '{hand}'"
        robot.hand_open(side, hold_s=0.4)
        return f"opened {side} hand"

    tools: Dict[str, Callable[..., Any]] = {
        "move": move,
        "reach_forward": reach_forward,
        "grab": grab,
        "release": release,
    }

    tool_schemas: List[Dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "move",
                "description": (
                    "Drive the robot base in one cardinal direction for a set "
                    "distance, optionally rotating around its yaw axis during "
                    "the motion. Use direction='stop' to halt immediately."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "direction": {
                            "type": "string",
                            "enum": ["forward", "backward", "left", "right", "stop"],
                            "description": "Cardinal motion direction (or 'stop').",
                        },
                        "distance_m": {
                            "type": "number",
                            "description": "Distance to travel in meters. Ignored when direction='stop'.",
                            "minimum": 0.0,
                            "maximum": 5.0,
                        },
                        "yaw_rad": {
                            "type": "number",
                            "description": "Total yaw rotation (radians) to perform DURING the motion. Positive = left turn.",
                        },
                        "speed_mps": {
                            "type": "number",
                            "description": "Linear speed in m/s, clamped to [0.05, 0.6].",
                            "minimum": 0.05,
                            "maximum": 0.6,
                        },
                    },
                    "required": ["direction"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "reach_forward",
                "description": (
                    "Extend the chosen arm forward using the saved pose trajectory. "
                    "Use this to position the gripper near an object before calling 'grab'."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "height_m": {
                            "type": "number",
                            "description": "Accepted for compatibility; the saved pose trajectory currently ignores it.",
                            "minimum": -0.30,
                            "maximum": 0.30,
                        },
                        "length_m": {
                            "type": "number",
                            "description": "Accepted for compatibility; the saved pose trajectory currently ignores it.",
                            "minimum": 0.10,
                            "maximum": 0.60,
                        },
                        "arm": {
                            "type": "string",
                            "enum": ["left", "right"],
                            "description": "Which arm to use.",
                        },
                        "duration_s": {
                            "type": "number",
                            "description": "Seconds for the reach motion (smoother = larger).",
                            "minimum": 0.5,
                            "maximum": 8.0,
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "grab",
                "description": (
                    "Slowly close the fingers around whatever is between them, "
                    "stopping early if any finger torque exceeds "
                    "resistance_threshold (N·m). Use this AFTER reach_forward "
                    "has positioned the hand near the target."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "hand": {
                            "type": "string",
                            "enum": ["left", "right"],
                            "description": "Which hand to close.",
                        },
                        "resistance_threshold": {
                            "type": "number",
                            "description": "Torque threshold in N·m. 0.3 = gentle, 0.5 = default, 0.8 = firmer grip.",
                            "minimum": 0.05,
                            "maximum": 2.0,
                        },
                        "n_steps": {
                            "type": "integer",
                            "description": "Number of incremental closure steps (more = finer-grained resistance check).",
                            "minimum": 2,
                            "maximum": 20,
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "release",
                "description": "Open the chosen hand fully to release the held object.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "hand": {
                            "type": "string",
                            "enum": ["left", "right"],
                            "description": "Which hand to open.",
                        },
                    },
                    "required": [],
                },
            },
        },
    ]

    return tools, tool_schemas
