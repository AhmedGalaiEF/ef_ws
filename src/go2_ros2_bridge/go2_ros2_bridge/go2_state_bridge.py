#!/usr/bin/env python3

from __future__ import annotations

import math
import threading
from typing import Any

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

try:
    import unitree_legged_const as go2_const
except ImportError:  # unitree_legged_const may not be available in all envs
    go2_const = None


class Go2StateBridge(Node):
    def __init__(self) -> None:
        super().__init__("go2_state_bridge")
        self.declare_parameter("interface", "")
        self.declare_parameter("publish_rate_hz", 50.0)

        self._interface = (
            self.get_parameter("interface").get_parameter_value().string_value
        )
        self._publish_rate_hz = (
            self.get_parameter("publish_rate_hz").get_parameter_value().double_value
        )
        if not math.isfinite(self._publish_rate_hz) or self._publish_rate_hz <= 0.0:
            self._publish_rate_hz = 50.0

        self._joint_map = self._build_joint_map()
        self._last_state = None
        self._state_lock = threading.Lock()

        self._joint_pub = self.create_publisher(JointState, "/joint_states", 10)

        if self._interface:
            self.get_logger().info(f"Initializing Unitree SDK on {self._interface}")
            ChannelFactoryInitialize(0, self._interface)
        else:
            self.get_logger().info("Initializing Unitree SDK with default interface")
            ChannelFactoryInitialize(0)

        self._subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self._subscriber.Init(self._on_lowstate, 10)

        self._timer = self.create_timer(
            1.0 / self._publish_rate_hz, self._publish_latest_state
        )

    def _build_joint_map(self) -> list[tuple[str, int]]:
        # Default Unitree motor index order for Go2 (FR, FL, RR, RL).
        default_map = [
            ("FR_hip_joint", 0),
            ("FR_thigh_joint", 1),
            ("FR_calf_joint", 2),
            ("FL_hip_joint", 3),
            ("FL_thigh_joint", 4),
            ("FL_calf_joint", 5),
            ("RR_hip_joint", 6),
            ("RR_thigh_joint", 7),
            ("RR_calf_joint", 8),
            ("RL_hip_joint", 9),
            ("RL_thigh_joint", 10),
            ("RL_calf_joint", 11),
        ]

        if go2_const is None or not hasattr(go2_const, "LegID"):
            self.get_logger().info(
                "unitree_legged_const not available; using default joint map"
            )
            return default_map

        leg = go2_const.LegID
        # Use SDK constants when available to match firmware ordering.
        return [
            ("FR_hip_joint", leg["FR_0"]),
            ("FR_thigh_joint", leg["FR_1"]),
            ("FR_calf_joint", leg["FR_2"]),
            ("FL_hip_joint", leg["FL_0"]),
            ("FL_thigh_joint", leg["FL_1"]),
            ("FL_calf_joint", leg["FL_2"]),
            ("RR_hip_joint", leg["RR_0"]),
            ("RR_thigh_joint", leg["RR_1"]),
            ("RR_calf_joint", leg["RR_2"]),
            ("RL_hip_joint", leg["RL_0"]),
            ("RL_thigh_joint", leg["RL_1"]),
            ("RL_calf_joint", leg["RL_2"]),
        ]

    def _on_lowstate(self, msg: LowState_) -> None:
        with self._state_lock:
            self._last_state = msg

    def _publish_latest_state(self) -> None:
        with self._state_lock:
            state = self._last_state
        if state is None:
            return

        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = [name for name, _ in self._joint_map]

        positions: list[float] = []
        velocities: list[float] = []
        efforts: list[float] = []
        for _, index in self._joint_map:
            try:
                motor = state.motor_state[index]
            except (AttributeError, IndexError, KeyError, TypeError):
                positions.append(0.0)
                velocities.append(0.0)
                efforts.append(0.0)
                continue

            positions.append(float(motor.q))
            velocities.append(float(getattr(motor, "dq", 0.0)))
            efforts.append(float(getattr(motor, "tau_est", 0.0)))

        joint_msg.position = positions
        if any(v != 0.0 for v in velocities):
            joint_msg.velocity = velocities
        if any(e != 0.0 for e in efforts):
            joint_msg.effort = efforts

        self._joint_pub.publish(joint_msg)


def main() -> None:
    rclpy.init()
    node: Go2StateBridge | None = None
    try:
        node = Go2StateBridge()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
