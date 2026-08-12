from __future__ import annotations

import os
import subprocess
import sys
from textwrap import dedent
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
    def release_arms(self) -> str: ...
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

    def release_arms(self) -> str:
        return self._record("release_arm()")

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

    def __init__(self, iface: str, domain_id: int, navbot_command_topic: str = "/model_api/navbot_command") -> None:
        import sdk_lib  # local import: keeps the SDK dependency optional

        self._robot = sdk_lib.G1(iface=iface, domain_id=domain_id)
        self._domain_id = domain_id
        self._navbot_command_topic = str(navbot_command_topic)

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

    def release_arms(self) -> str:
        code = self._robot.release_arm()
        return f"released arms -> code {code}"

    def say(self, text: str) -> str:
        code = self._robot.say(text)
        return f"spoke -> code {code}"

    def navbot_command(self, text: str) -> str:
        code = r"""
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

topic = sys.argv[1]
text = sys.argv[2]

rclpy.init(args=None)
node = Node("ai_control_navbot_command_once")
try:
    publisher = node.create_publisher(String, topic, 10)
    deadline = time.monotonic() + 1.0
    subscribers = 0
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        subscribers = publisher.get_subscription_count()
        if subscribers:
            break
    msg = String()
    msg.data = text
    for _ in range(3):
        publisher.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.05)
        time.sleep(0.05)
    print(subscribers)
finally:
    node.destroy_node()
    rclpy.shutdown()
"""
        code = dedent(code)
        env = os.environ.copy()
        env["ROS_DOMAIN_ID"] = str(int(self._domain_id))
        env.setdefault("ROS_LOG_DIR", "/tmp/ros_log")
        try:
            result = subprocess.run(
                [sys.executable, "-c", code, self._navbot_command_topic, str(text)],
                check=False,
                capture_output=True,
                env=env,
                text=True,
                timeout=5.0,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"timed out publishing nav bot command on {self._navbot_command_topic}") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"failed to publish nav bot command on {self._navbot_command_topic}: {detail}")
        subscribers = (result.stdout or "0").strip().splitlines()[-1]
        return f"published nav bot command {text!r} on {self._navbot_command_topic} (subscribers={subscribers})"

    def capture_frame(self) -> bytes | None:
        return self._robot.get_camera_image_jpeg()
