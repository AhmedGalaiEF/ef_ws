from __future__ import annotations

import os
import time
from typing import Protocol


class RobotBackend(Protocol):
    """Minimal actuation surface the tool layer dispatches onto.

    Both backends implement the same interface so `tools.dispatch` never has
    to know whether it is talking to real hardware or a stub.
    """

    def move(self, vx: float, vy: float, vyaw: float, duration: float) -> str: ...
    def navigate_to(self, x: float, y: float, yaw: float) -> str: ...
    def stop(self) -> str: ...
    def hand_open(self, hand: str) -> str: ...
    def hand_close(self, hand: str) -> str: ...
    def gesture(self, name: str) -> str: ...
    def say(self, text: str) -> str: ...
    def navbot_command(self, text: str) -> str: ...
    def capture_frame(self) -> bytes | None: ...


class MockRobotBackend:
    """No hardware attached -- logs the action that *would* run.

    This is the default backend so the CLI is usable for prompt/routing/tool
    development without a robot on the network.
    """

    def __init__(self) -> None:
        self.log: list[str] = []

    def _record(self, message: str) -> str:
        self.log.append(message)
        print(f"    [MOCK ROBOT] {message}")
        return message

    def move(self, vx: float, vy: float, vyaw: float, duration: float) -> str:
        return self._record(f"move_for(duration={duration}, vx={vx}, vy={vy}, vyaw={vyaw})")

    def navigate_to(self, x: float, y: float, yaw: float) -> str:
        return self._record(f"move_with_odom(x={x}, y={y}, yaw={yaw})")

    def stop(self) -> str:
        return self._record("stop()")

    def hand_open(self, hand: str) -> str:
        return self._record(f"hand_open(hand={hand!r})")

    def hand_close(self, hand: str) -> str:
        return self._record(f"hand_close(hand={hand!r})")

    def gesture(self, name: str) -> str:
        return self._record(f"execute_arm_action(action={name!r})")

    def say(self, text: str) -> str:
        return self._record(f"say(text={text!r})")

    def navbot_command(self, text: str) -> str:
        return self._record(f"publish_navbot_command(text={text!r})")

    def capture_frame(self) -> bytes | None:
        self.log.append("capture_frame() -> no camera in mock mode")
        print("    [MOCK ROBOT] capture_frame() -> None (no camera attached)")
        return None


class RealRobotBackend:
    """Wraps `sdk_lib.G1`. Only imported/constructed when --robot is passed,
    since it requires the Unitree SDK2 Python stack to be installed."""

    def __init__(self, iface: str, domain_id: int) -> None:
        import sdk_lib  # local import: keeps the SDK dependency optional

        self._robot = sdk_lib.G1(iface=iface, domain_id=domain_id)
        self._domain_id = domain_id
        self._navbot_publisher = None

    def move(self, vx: float, vy: float, vyaw: float, duration: float) -> str:
        self._robot.move_for(duration, vx, vy, vyaw)
        return f"moved for {duration}s at vx={vx}, vy={vy}, vyaw={vyaw}"

    def navigate_to(self, x: float, y: float, yaw: float) -> str:
        result = self._robot.move_with_odom(x, y, yaw)
        return f"navigate_to result: {result}"

    def stop(self) -> str:
        self._robot.stop()
        return "stopped"

    def hand_open(self, hand: str) -> str:
        self._robot.hand_open(hand)
        return f"opened {hand} hand"

    def hand_close(self, hand: str) -> str:
        self._robot.hand_close(hand)
        return f"closed {hand} hand"

    def gesture(self, name: str) -> str:
        code = self._robot.execute_arm_action(name)
        return f"gesture {name!r} -> code {code}"

    def say(self, text: str) -> str:
        code = self._robot.say(text)
        return f"spoke -> code {code}"

    def navbot_command(self, text: str) -> str:
        if self._navbot_publisher is None:
            import rclpy
            from rclpy.node import Node
            from std_msgs.msg import String

            os.environ.setdefault("ROS_DOMAIN_ID", str(int(self._domain_id)))
            if not rclpy.ok():
                rclpy.init(args=None)
            node = Node("ai_control_navbot_command")
            publisher = node.create_publisher(String, "/model_api/navbot_command", 10)
            self._navbot_publisher = (node, publisher, String)
        node, publisher, string_msg = self._navbot_publisher
        deadline = time.time() + 1.0
        while publisher.get_subscription_count() == 0 and time.time() < deadline:
            import rclpy

            rclpy.spin_once(node, timeout_sec=0.05)
        for _ in range(3):
            publisher.publish(string_msg(data=text))
            time.sleep(0.05)
        node.get_logger().info(f"published nav bot command: {text!r}")
        return f"published nav bot command {text!r}"

    def capture_frame(self) -> bytes | None:
        return self._robot.get_camera_image_jpeg()
