from __future__ import annotations

from typing import Any


class RobotCommandExecutor:
    """Sample executor stub.

    Replace the body of `execute()` with calls into your local SDK wrapper,
    for example `sdk_client.Robot`.
    """

    def execute(self, action: str, parameters: dict[str, Any]) -> tuple[bool, str]:
        normalized = str(action).strip().lower()

        if normalized == "get_state":
            return True, "State request accepted."
        if normalized == "get_pose":
            return True, "Pose request accepted."
        if normalized == "stop":
            return True, "Stop command accepted."
        if normalized == "balanced_stand":
            return True, "Balanced stand command accepted."
        if normalized == "hand_open":
            return True, f"Hand open accepted with parameters={parameters}."
        if normalized == "hand_close":
            return True, f"Hand close accepted with parameters={parameters}."
        if normalized == "walk_distance":
            return True, f"Walk command accepted with parameters={parameters}."
        if normalized == "turn_angle":
            return True, f"Turn command accepted with parameters={parameters}."
        if normalized == "start_slam":
            return True, f"SLAM start accepted with parameters={parameters}."
        if normalized == "stop_slam":
            return True, f"SLAM stop accepted with parameters={parameters}."

        return False, f"Unhandled action: {action}"
