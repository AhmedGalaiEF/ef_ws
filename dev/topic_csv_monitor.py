#!/usr/bin/env python3
"""Monitor ROS 2 topics and append received messages to a CSV file."""

from __future__ import annotations

import argparse
import array
import csv
import json
import math
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosidl_runtime_py.utilities import get_message


DEFAULT_TOPICS = {
    "/SymState",
    "/api/action_store/request",
    "/api/action_store/response",
    "/api/arm/request",
    "/api/arm/response",
    "/api/audiohub/request",
    "/api/audiohub/response",
    "/api/bashrunner/request",
    "/api/bashrunner/response",
    "/api/basic_clearoip/request",
    "/api/basic_clearoip/response",
    "/api/basic_clearoip_lease/request",
    "/api/basic_clearoip_lease/response",
    "/api/basic_demarcate/request",
    "/api/basic_demarcate/response",
    "/api/basic_demarcate_lease/request",
    "/api/basic_demarcate_lease/response",
    "/api/basic_softlimit/request",
    "/api/basic_softlimit/response",
    "/api/basic_softlimit_lease/request",
    "/api/basic_softlimit_lease/response",
    "/api/basic_taumax/request",
    "/api/basic_taumax/response",
    "/api/basic_taumax_lease/request",
    "/api/basic_taumax_lease/response",
    "/api/config/request",
    "/api/config/response",
    "/api/dex3_msg_controller/request",
    "/api/dex3_msg_controller/response",
    "/api/gesture/request",
    "/api/gpt/request",
    "/api/gpt/response",
    "/api/motion_switcher/request",
    "/api/motion_switcher/response",
    "/api/rm_con/request",
    "/api/robot_state/request",
    "/api/robot_state/response",
    "/api/robot_type_service/request",
    "/api/robot_type_service/response",
    "/api/slam_operate/request",
    "/api/slam_operate/response",
    "/api/sport/request",
    "/api/sport/response",
    "/api/videohub/request",
    "/api/videohub/response",
    "/api/voice/request",
    "/api/voice/response",
    "/api/vui/request",
    "/api/vui/response",
    "/arm/action/state",
    "/arm_sdk",
    "/armsdk",
    "/audio_msg",
    "/audio_msg/filter",
    "/audiosender",
    "/collision_clouds",
    "/config_change_status",
    "/dex3/left/cmd",
    "/dex3/left/state",
    "/dex3/right/cmd",
    "/dex3/right/state",
    "/dog_imu_raw",
    "/dog_odom",
    "/ele_clouds",
    "/event/action_store",
    "/frontvideostream",
    "/gesture/result",
    "/global_map",
    "/gpt_cmd",
    "/gpt_state",
    "/gptflowfeedback",
    "/grid_clouds",
    "/gridmap",
    "/lf/agvalarmstate",
    "/lf/agvbmsstate",
    "/lf/battery_alarm",
    "/lf/bmsstate",
    "/lf/dex3/left/state",
    "/lf/dex3/right/state",
    "/lf/emergency_stop",
    "/lf/lowstate",
    "/lf/mainboardstate",
    "/lf/odommodestate",
    "/lf/secondary_imu",
    "/lf/sportmodestate",
    "/loco_sdk",
    "/log_system_inbound",
    "/log_system_outbound",
    "/lowcmd",
    "/lowstate",
    "/lowstate_doubleimu",
    "/multiplestate",
    "/no_warning_clouds",
    "/odom",
    "/odommodestate",
    "/parameter_events",
    "/planner_map",
    "/pre_collision_clouds",
    "/pre_safe_clouds",
    "/public_network_status",
    "/rosout",
    "/rtc/state",
    "/rtc_status",
    "/safe_clouds",
    "/secondary_imu",
    "/selftest",
    "/servicestate",
    "/servicestateactivate",
    "/slam_info",
    "/slam_key_info",
    "/sportmodestate",
    "/unitree/slam_mapping/odom",
    "/unitree/slam_mapping/points",
    "/unitree/slam_relocation/global_map",
    "/unitree/slam_relocation/odom",
    "/unitree/slam_relocation/points",
    "/unitree/slam_relocation/web_points",
    "/unitree_slam/waypoints",
    "/user_lowcmd",
    "/utlidar/cloud_deskewed",
    "/utlidar/cloud_livox_mid360",
    "/utlidar/imu_livox_mid360",
    "/utlidar/range_info",
    "/videohub/inner",
    "/warning_clouds",
    "/webrtcreq",
    "/webrtcres",
    "/wirelesscontroller",
    "/xfk_webrtcreq",
    "/xfk_webrtcres",
}


def normalize_topic(topic: str) -> str:
    topic = topic.strip()
    if not topic:
        return topic
    return topic if topic.startswith("/") else f"/{topic}"


def positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite value > 0")
    return parsed


def nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def load_topics(path: Path) -> set[str]:
    topics: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        topics.add(normalize_topic(line))
    return topics


def qos_profile(reliable: bool) -> QoSProfile:
    reliability = ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=reliability,
        durability=DurabilityPolicy.VOLATILE,
    )


def compact_value(value: Any, max_sequence: int, max_string: int, depth: int) -> Any:
    if depth <= 0:
        return "<max_depth>"
    if isinstance(value, bytes):
        return {
            "__bytes_len__": len(value),
            "preview_hex": value[:max_sequence].hex(),
        }
    if isinstance(value, bytearray):
        return {
            "__bytearray_len__": len(value),
            "preview_hex": bytes(value[:max_sequence]).hex(),
        }
    if isinstance(value, str):
        if len(value) > max_string:
            return {
                "__string_len__": len(value),
                "preview": value[:max_string],
            }
        return value
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    if isinstance(value, array.array):
        preview = [
            compact_value(item, max_sequence, max_string, depth - 1)
            for item in value[:max_sequence]
        ]
        if len(value) > max_sequence:
            return {
                "__array_typecode__": value.typecode,
                "__array_len__": len(value),
                "preview": preview,
            }
        return preview
    if isinstance(value, (list, tuple)):
        preview = [
            compact_value(item, max_sequence, max_string, depth - 1)
            for item in value[:max_sequence]
        ]
        if len(value) > max_sequence:
            return {"__sequence_len__": len(value), "preview": preview}
        return preview
    if hasattr(value, "get_fields_and_field_types"):
        return compact_message(value, max_sequence, max_string, depth - 1)
    return str(value)


def compact_message(msg: Any, max_sequence: int, max_string: int, depth: int) -> dict[str, Any]:
    fields = msg.get_fields_and_field_types()
    return {
        name: compact_value(getattr(msg, name), max_sequence, max_string, depth)
        for name in fields
    }


class CsvTopicMonitor(Node):
    def __init__(
        self,
        output_path: Path,
        topics: set[str] | None,
        *,
        monitor_all: bool,
        discovery_period: float,
        reliable: bool,
        max_sequence: int,
        max_string: int,
        max_depth: int,
    ) -> None:
        super().__init__("topic_csv_monitor")
        self.output_path = output_path
        self.target_topics = topics
        self.monitor_all = monitor_all
        self.discovery_period = discovery_period
        self.qos = qos_profile(reliable)
        self.max_sequence = max_sequence
        self.max_string = max_string
        self.max_depth = max_depth
        self.subscriptions_by_topic: dict[str, Any] = {}
        self.lock = threading.Lock()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_file = output_path.open("a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.csv_file,
            fieldnames=[
                "wall_time_ns",
                "ros_time_ns",
                "topic",
                "type",
                "message_json",
            ],
        )
        if output_path.stat().st_size == 0:
            self.writer.writeheader()
            self.csv_file.flush()

        self.discover_topics()
        self.timer = self.create_timer(discovery_period, self.discover_topics)

    def close(self) -> None:
        with self.lock:
            self.csv_file.flush()
            self.csv_file.close()

    def discover_topics(self) -> None:
        topic_types = dict(self.get_topic_names_and_types())
        for topic, type_names in sorted(topic_types.items()):
            topic = normalize_topic(topic)
            if topic in self.subscriptions_by_topic:
                continue
            if not self.monitor_all and self.target_topics is not None and topic not in self.target_topics:
                continue
            if not type_names:
                continue
            self.subscribe_to_topic(topic, type_names[0])

    def subscribe_to_topic(self, topic: str, type_name: str) -> None:
        try:
            msg_type = get_message(type_name)
        except (AttributeError, ModuleNotFoundError, ValueError) as exc:
            self.get_logger().warning(f"Skipping {topic}: cannot import {type_name}: {exc}")
            return

        def callback(msg: Any, *, topic_name: str = topic, topic_type: str = type_name) -> None:
            self.write_message(topic_name, topic_type, msg)

        self.subscriptions_by_topic[topic] = self.create_subscription(
            msg_type,
            topic,
            callback,
            self.qos,
        )
        self.get_logger().info(f"Subscribed to {topic} [{type_name}]")

    def write_message(self, topic: str, type_name: str, msg: Any) -> None:
        row = {
            "wall_time_ns": time.time_ns(),
            "ros_time_ns": self.get_clock().now().nanoseconds,
            "topic": topic,
            "type": type_name,
            "message_json": json.dumps(
                compact_message(msg, self.max_sequence, self.max_string, self.max_depth),
                ensure_ascii=True,
                separators=(",", ":"),
            ),
        }
        with self.lock:
            self.writer.writerow(row)
            self.csv_file.flush()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Subscribe to ROS 2 topics and append timestamped messages to CSV."
    )
    parser.add_argument(
        "-o",
        "--output",
        default="topic_data.csv",
        help="CSV output path. Existing files are appended to. Default: topic_data.csv",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Monitor every discovered topic instead of the built-in Unitree topic list.",
    )
    parser.add_argument(
        "--topics-file",
        type=Path,
        help="Text file with one topic per line. Overrides the built-in topic list.",
    )
    parser.add_argument(
        "--topic",
        action="append",
        default=[],
        help="Topic to monitor. Can be passed multiple times. Overrides the built-in topic list.",
    )
    parser.add_argument(
        "--discovery-period",
        type=positive_float,
        default=2.0,
        help="Seconds between topic discovery scans. Default: 2.0",
    )
    parser.add_argument(
        "--reliable",
        action="store_true",
        help="Request reliable QoS. Default is best-effort for broad sensor compatibility.",
    )
    parser.add_argument(
        "--max-sequence",
        type=nonnegative_int,
        default=32,
        help="Maximum list/array items stored in each message field preview. Default: 32",
    )
    parser.add_argument(
        "--max-string",
        type=nonnegative_int,
        default=2000,
        help="Maximum string characters stored before summarizing. Default: 2000",
    )
    parser.add_argument(
        "--max-depth",
        type=nonnegative_int,
        default=8,
        help="Maximum nested message depth converted to JSON. Default: 8",
    )
    return parser.parse_args(argv)


def selected_topics(args: argparse.Namespace) -> set[str] | None:
    if args.all:
        return None
    if args.topic:
        return {normalize_topic(topic) for topic in args.topic if topic.strip()}
    if args.topics_file:
        return load_topics(args.topics_file)
    return set(DEFAULT_TOPICS)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    topics = selected_topics(args)

    rclpy.init()
    monitor = CsvTopicMonitor(
        Path(args.output).expanduser(),
        topics,
        monitor_all=args.all,
        discovery_period=args.discovery_period,
        reliable=args.reliable,
        max_sequence=args.max_sequence,
        max_string=args.max_string,
        max_depth=args.max_depth,
    )

    stop_requested = False

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        monitor.get_logger().info(f"Received signal {signum}; stopping")

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        while rclpy.ok() and not stop_requested:
            rclpy.spin_once(monitor, timeout_sec=0.2)
    finally:
        monitor.close()
        monitor.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
