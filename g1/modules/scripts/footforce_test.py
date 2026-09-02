#!/usr/bin/env python3
"""Read-only ground-reaction-force discovery probe for a Unitree G1.

The Go2 SDK exposes foot loads as ``LowState.foot_force`` and often also
``foot_force_est`` on the DDS topic ``rt/lowstate``.  This tool checks whether
the G1 publishes either of those fields, discovers similarly named ROS 2
topics, and prints one ROS message from every relevant topic it can sample.

It never publishes commands or changes robot state.

Examples:
  python3 modules/scripts/footforce_test.py
  python3 modules/scripts/footforce_test.py --seconds 5 --iface eth0
  python3 modules/scripts/footforce_test.py --no-ros
"""
from __future__ import annotations

import argparse
from collections import deque
import json
import math
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


MODULES_DIR = Path(__file__).resolve().parents[1]
if str(MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(MODULES_DIR))

from dds_env import default_dds_iface, ensure_channel_factory_initialized
from sdk_sensors import resolve_lowstate_type


GO2_LOWSTATE_TOPIC = "rt/lowstate"
ROS_FORCE_WORDS = ("foot", "force", "contact", "pressure", "wrench")
FORCE_FIELD_WORDS = ("foot", "force", "contact", "pressure", "wrench", "load")


def bounded_positive_float(maximum: float):
    def parse(value: str) -> float:
        try:
            parsed = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("must be a number") from exc
        if not math.isfinite(parsed) or not 0.0 < parsed <= maximum:
            raise argparse.ArgumentTypeError(f"must be finite and between 0 and {maximum:g}")
        return parsed

    return parse


def domain_id(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if not 0 <= parsed <= 232:
        raise argparse.ArgumentTypeError("must be between 0 and 232")
    return parsed


def nonempty_iface(value: str) -> str:
    parsed = str(value).strip()
    if not parsed:
        raise argparse.ArgumentTypeError("must not be empty")
    return parsed


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=bounded_positive_float(300.0), default=3.0,
                        help="DDS collection time (default: 3).")
    parser.add_argument("--iface", type=nonempty_iface, default=default_dds_iface("eth0"),
                        help="DDS NIC; use auto to let CycloneDDS choose.")
    parser.add_argument("--domain-id", type=domain_id, default=0, help="DDS domain ID.")
    parser.add_argument("--ros-timeout", type=bounded_positive_float(60.0), default=3.0,
                        help="Timeout for ROS 2 discovery commands (default: 3).")
    parser.add_argument("--ros-echo-seconds", type=bounded_positive_float(60.0), default=2.0,
                        help="Wait time for one message per relevant ROS topic (default: 2).")
    parser.add_argument("--no-ros", action="store_true", help="Skip ROS 2 discovery.")
    parser.add_argument("--json", action="store_true", help="Also print machine-readable results.")
    return parser.parse_args(argv)


def as_values(value: Any) -> list[float | int] | None:
    """Convert a scalar or IDL sequence to JSON-safe numeric values."""
    if isinstance(value, (str, bytes, bytearray)):
        return None
    try:
        raw = list(value)
    except TypeError:
        raw = [value]
    except Exception:
        return None
    converted: list[float | int] = []
    for item in raw:
        try:
            number = float(item)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        converted.append(int(number) if number.is_integer() else number)
    return converted


def numeric_force_fields(msg: Any) -> dict[str, list[float | int]]:
    """Find force/contact-like LowState members without assuming a G1 schema."""
    fields: dict[str, list[float | int]] = {}
    for name in dir(msg):
        normalized = name.lower()
        if name.startswith("_") or not any(word in normalized for word in FORCE_FIELD_WORDS):
            continue
        try:
            values = as_values(getattr(msg, name))
        except Exception:
            continue
        if values is not None:
            fields[name] = values
    return fields


class LowStateProbe:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.samples: deque[dict[str, list[float | int]]] = deque(maxlen=1000)
        self.samples_received = 0
        self.field_names: set[str] = set()
        self.ever_nonzero: dict[str, bool] = {}

    def callback(self, msg: Any) -> None:
        fields = numeric_force_fields(msg)
        with self._lock:
            self.samples_received += 1
            self.samples.append(fields)
            self.field_names.update(fields)
            for name, values in fields.items():
                self.ever_nonzero[name] = self.ever_nonzero.get(name, False) or any(
                    value != 0 for value in values
                )

    def result(self) -> dict[str, Any]:
        with self._lock:
            samples = list(self.samples)
            samples_received = self.samples_received
            field_names = sorted(self.field_names)
            nonzero = dict(self.ever_nonzero)
        latest = samples[-1] if samples else {}
        return {
            "topic": GO2_LOWSTATE_TOPIC,
            "samples_received": samples_received,
            "force_like_fields": field_names,
            "latest": latest,
            "any_nonzero": nonzero,
            # Exact Go2 convention, checked explicitly even if schema introspection changes.
            "go2_compatible_fields": {
                name: latest.get(name) for name in ("foot_force", "foot_force_est") if name in latest
            },
        }


def run_lowstate_probe(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {"topic": GO2_LOWSTATE_TOPIC, "error": None}
    msg_type = resolve_lowstate_type()
    if msg_type is None:
        result["error"] = "Unitree LowState_ IDL type is not available."
        return result
    try:
        from unitree_sdk2py.core.channel import ChannelSubscriber
        ensure_channel_factory_initialized(args.domain_id, args.iface)
        probe = LowStateProbe()
        subscriber = ChannelSubscriber(GO2_LOWSTATE_TOPIC, msg_type)
        subscriber.Init(probe.callback, 20)
        time.sleep(args.seconds)
        result.update(probe.result())
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def run_command(command: list[str], timeout_s: float) -> tuple[int, str]:
    try:
        completed = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, timeout=timeout_s, check=False)
        return completed.returncode, completed.stdout.strip()
    except FileNotFoundError:
        return 127, "ros2 command not found"
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout.strip() if isinstance(exc.stdout, str) else ""
        return 124, (output + "\n" if output else "") + f"timed out after {timeout_s:.1f}s"
    except Exception as exc:
        return 1, f"{type(exc).__name__}: {exc}"


def relevant_ros_topics(listing: str) -> list[tuple[str, str]]:
    topics: list[tuple[str, str]] = []
    for line in listing.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        topic, type_name = parts
        # ROS 2 Foxy prints types as "[package/msg/Type]".
        type_name = type_name.strip().strip("[]")
        text = f"{topic} {type_name}".lower()
        # Sample explicitly force-like topics plus actual LowState messages;
        # do not mistake a topic merely named "lowstate_doubleimu" for one.
        if any(word in text for word in ROS_FORCE_WORDS) or "lowstate" in type_name.lower():
            topics.append((topic, type_name))
    return topics


def sample_ros_topic(topic: str, type_name: str, timeout_s: float) -> tuple[int, str]:
    """Receive one ROS message without depending on Foxy's echo CLI options."""
    try:
        import rclpy
        from rosidl_runtime_py.utilities import get_message
    except Exception as exc:
        return 1, f"ROS Python imports unavailable: {type(exc).__name__}: {exc}"

    node = None
    initialized_here = False
    received: list[Any] = []
    try:
        if not rclpy.ok():
            rclpy.init(args=None)
            initialized_here = True
        message_type = get_message(type_name)
        node = rclpy.create_node("g1_footforce_probe", start_parameter_services=False)
        subscription = node.create_subscription(message_type, topic, lambda msg: received.append(msg), 10)
        deadline = time.monotonic() + timeout_s
        while not received and time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            rclpy.spin_once(node, timeout_sec=min(0.1, remaining))
        node.destroy_subscription(subscription)
        if not received:
            return 124, "(no message received before timeout)"
        return 0, str(received[0])
    except Exception as exc:
        return 1, f"{type(exc).__name__}: {exc}"
    finally:
        if node is not None:
            node.destroy_node()
        if initialized_here:
            rclpy.shutdown()


def run_ros_probe(args: argparse.Namespace) -> dict[str, Any]:
    if shutil.which("ros2") is None:
        return {"available": False, "error": "ros2 command not found", "topics": []}
    code, listing = run_command(["ros2", "topic", "list", "-t"], args.ros_timeout)
    output: dict[str, Any] = {"available": code == 0, "list_exit_code": code,
                              "list_output": listing, "topics": []}
    if code != 0:
        output["error"] = "Could not list ROS 2 topics. Source ROS 2 and check DDS permissions/domain."
        return output
    for topic, type_name in relevant_ros_topics(listing):
        echo_code, echo = sample_ros_topic(topic, type_name, args.ros_echo_seconds)
        if len(echo) > 6000:
            echo = echo[:6000] + "\n... (remaining fields omitted)"
        output["topics"].append({"topic": topic, "type": type_name, "echo_exit_code": echo_code,
                                 "sample": echo or "(no message received)"})
    return output


def print_human(dds: dict[str, Any], ros: dict[str, Any] | None) -> None:
    print("G1 foot-force / ground-reaction-force probe (read-only)")
    print(f"\nDDS {dds.get('topic')}:")
    if dds.get("error"):
        print(f"  ERROR: {dds['error']}")
    else:
        print(f"  LowState samples received: {dds.get('samples_received', 0)}")
        fields = dds.get("force_like_fields", [])
        if not fields:
            print("  No foot/force/contact/pressure-like fields exist in this G1 LowState schema.")
        for name in fields:
            values = dds.get("latest", {}).get(name, [])
            suffix = "non-zero observed" if dds.get("any_nonzero", {}).get(name) else "only zero observed"
            print(f"  {name}: {values} ({suffix})")
        go2 = dds.get("go2_compatible_fields", {})
        if go2:
            print(f"  Go2-compatible foot-force fields found: {go2}")
        elif fields:
            print("  Go2 fields foot_force/foot_force_est were not present; inspect fields above.")
        if dds.get("samples_received", 0) == 0:
            print("  No rt/lowstate data arrived: verify DDS interface/domain and that the robot is on.")

    if ros is None:
        return
    print("\nROS 2 discovery:")
    if ros.get("error"):
        print(f"  ERROR: {ros['error']}")
        detail = ros.get("list_output")
        if detail:
            print(f"  ros2 output: {detail}")
        return
    print("  ros2 topic list -t:")
    listing = str(ros.get("list_output") or "(no topics)")
    print("    " + listing.replace("\n", "\n    "))
    topics = ros.get("topics", [])
    print(f"  Force-related topic samples: {len(topics)}")
    for entry in topics:
        exit_code = entry["echo_exit_code"]
        status = "sample received" if exit_code == 0 else "no sample" if exit_code == 124 else f"subscriber exit {exit_code}"
        print(f"  {entry['topic']} [{entry['type']}] ({status}):")
        print("    " + str(entry["sample"]).replace("\n", "\n    "))


def main() -> int:
    args = parse_args()
    ros = None if args.no_ros else run_ros_probe(args)
    # rclpy and unitree_sdk2py both create CycloneDDS participants. ROS must
    # run first on this Foxy installation, otherwise rclpy cannot create a
    # node after the SDK channel factory is initialized.
    dds = run_lowstate_probe(args)
    print_human(dds, ros)
    if args.json:
        print("\nJSON:")
        print(json.dumps({"dds": dds, "ros": ros}, indent=2, sort_keys=True, allow_nan=False))
    # A missing force field/data is a probe finding, not a script failure.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
