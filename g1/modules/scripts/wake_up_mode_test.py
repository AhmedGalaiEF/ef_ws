#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dds_env import default_dds_iface, ensure_channel_factory_initialized
from sdk_boot import create_loco_client

try:
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.g1.loco.g1_loco_api import (
        ROBOT_API_ID_LOCO_GET_BALANCE_MODE,
        ROBOT_API_ID_LOCO_GET_FSM_ID,
        ROBOT_API_ID_LOCO_GET_FSM_MODE,
        ROBOT_API_ID_LOCO_GET_PHASE,
        ROBOT_API_ID_LOCO_GET_STAND_HEIGHT,
        ROBOT_API_ID_LOCO_GET_SWING_HEIGHT,
    )
except ImportError as exc:
    raise SystemExit(
        "unitree_sdk2py is not installed. Install it with:\n"
        "  pip install -e <path-to-unitree_sdk2_python>"
    ) from exc


ERROR_HINTS = {
    0: "success",
    7001: "request parameter error",
    7002: "service busy; retry",
    7004: "unsupported mode name",
    7005: "internal command execute error",
    7006: "check command execute error",
    7007: "switch command execute error",
    7008: "release command execute error",
    7009: "custom config set error",
}

LOCO_READ_APIS = {
    "fsm_id": ROBOT_API_ID_LOCO_GET_FSM_ID,
    "fsm_mode": ROBOT_API_ID_LOCO_GET_FSM_MODE,
    "balance_mode": ROBOT_API_ID_LOCO_GET_BALANCE_MODE,
    "swing_height": ROBOT_API_ID_LOCO_GET_SWING_HEIGHT,
    "stand_height": ROBOT_API_ID_LOCO_GET_STAND_HEIGHT,
    "phase": ROBOT_API_ID_LOCO_GET_PHASE,
}

DEFAULT_DDS_TOPIC_FILTER = "lowstate|sport|wireless|mode|fsm|state|odom|tf"
DEFAULT_ROS_SAMPLE_FILTER = "state|mode|fsm|low|sport|wireless|odom"
ROS_ECHO_OMIT_TOKENS = ("joint", "lowstate", "low_state", "odom", "/tf", "velocity")
OMITTED_STATE_KEYS = {
    "acceleration",
    "angular_velocity",
    "ddq",
    "dq",
    "foot_force",
    "foot_force_est",
    "linear_velocity",
    "motor_cmd",
    "motor_state",
    "omega",
    "q",
    "tau",
    "tau_est",
    "vel",
    "velocity",
}


@dataclass(frozen=True)
class DdsTopicSpec:
    topic: str
    module_path: str
    type_name: str


DEFAULT_DDS_SAMPLES = (
    DdsTopicSpec("rt/lowstate", "unitree_sdk2py.idl.unitree_hg.msg.dds_", "LowState_"),
    DdsTopicSpec("rt/lowstate", "unitree_sdk2py.idl.unitree_go.msg.dds_", "LowState_"),
    DdsTopicSpec("rt/sportmodestate", "unitree_sdk2py.idl.unitree_go.msg.dds_", "SportModeState_"),
    DdsTopicSpec("rt/wirelesscontroller", "unitree_sdk2py.idl.unitree_go.msg.dds_", "WirelessController_"),
)


def _result_code(result: Any) -> int:
    if isinstance(result, tuple):
        return int(result[0])
    if result is None:
        return 0
    return int(result)


def _error_hint(code: int) -> str:
    return ERROR_HINTS.get(int(code), "unknown SDK return code")


def _json_value(
    value: Any,
    *,
    depth: int = 0,
    max_depth: int = 5,
    max_items: int = 80,
    omit_motion_fields: bool = True,
) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return list(value[:max_items])
    if depth >= max_depth:
        return repr(value)
    if isinstance(value, dict):
        return {
            str(k): _json_value(
                v,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                omit_motion_fields=omit_motion_fields,
            )
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
            if not (omit_motion_fields and str(k).lower() in OMITTED_STATE_KEYS)
        }
    if isinstance(value, (list, tuple)):
        return [
            _json_value(
                v,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                omit_motion_fields=omit_motion_fields,
            )
            for v in list(value)[:max_items]
        ]
    if hasattr(value, "__dict__"):
        return _object_to_dict(
            value,
            depth=depth + 1,
            max_depth=max_depth,
            max_items=max_items,
            omit_motion_fields=omit_motion_fields,
        )

    attrs: dict[str, Any] = {}
    for name in dir(value):
        if name.startswith("_"):
            continue
        try:
            attr = getattr(value, name)
        except Exception:
            continue
        if callable(attr):
            continue
        if omit_motion_fields and name.lower() in OMITTED_STATE_KEYS:
            continue
        attrs[name] = _json_value(
            attr,
            depth=depth + 1,
            max_depth=max_depth,
            max_items=max_items,
            omit_motion_fields=omit_motion_fields,
        )
    if attrs:
        return attrs
    return repr(value)


def _object_to_dict(
    value: Any,
    *,
    depth: int,
    max_depth: int,
    max_items: int,
    omit_motion_fields: bool,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    names: set[str] = set()
    if hasattr(value, "__dict__"):
        names.update(str(k) for k in vars(value).keys())
    for cls in type(value).__mro__:
        slots = getattr(cls, "__slots__", ())
        if isinstance(slots, str):
            names.add(slots)
        else:
            names.update(str(s) for s in slots)

    for name in sorted(n for n in names if not n.startswith("_")):
        if omit_motion_fields and name.lower() in OMITTED_STATE_KEYS:
            continue
        try:
            out[name] = _json_value(
                getattr(value, name),
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                omit_motion_fields=omit_motion_fields,
            )
        except Exception as exc:
            out[name] = {"error": str(exc)}
    return out


def _load_type(module_path: str, type_name: str) -> type | None:
    try:
        module = __import__(module_path, fromlist=[type_name])
        return getattr(module, type_name)
    except Exception:
        return None


def _load_class(module_path: str, class_name: str) -> type | None:
    return _load_type(module_path, class_name)


class LatestDdsSample:
    def __init__(self, spec: DdsTopicSpec) -> None:
        msg_type = _load_type(spec.module_path, spec.type_name)
        if msg_type is None:
            raise RuntimeError(f"{spec.module_path}.{spec.type_name} is not available")
        self.spec = spec
        self._lock = threading.Lock()
        self._latest: Any = None
        self._count = 0
        self._subscriber = ChannelSubscriber(spec.topic, msg_type)
        self._subscriber.Init(self._on_message, 10)

    def _on_message(self, msg: Any) -> None:
        with self._lock:
            self._latest = msg
            self._count += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            msg = self._latest
            count = self._count
        return {
            "topic": self.spec.topic,
            "type": f"{self.spec.module_path}.{self.spec.type_name}",
            "samples_seen": count,
            "data": None if msg is None else _json_value(msg),
        }


class RobotStateScanner:
    def __init__(
        self,
        iface: str,
        domain_id: int,
        timeout: float,
        dds_wait_s: float,
        include_ros: bool,
        ros_timeout_s: float,
        ros_sample_filter: str,
        include_discovery: bool,
        discovery_filter: str,
    ) -> None:
        self.iface = iface
        self.domain_id = int(domain_id)
        self.timeout = float(timeout)
        self.dds_wait_s = float(dds_wait_s)
        self.include_ros = bool(include_ros)
        self.ros_timeout_s = float(ros_timeout_s)
        self.ros_sample_filter = str(ros_sample_filter)
        self.include_discovery = bool(include_discovery)
        self.discovery_filter = str(discovery_filter)

        ensure_channel_factory_initialized(self.domain_id, self.iface)
        self.motion = MotionSwitcherClient()
        self.motion.SetTimeout(self.timeout)
        self.motion.Init()
        self.loco = create_loco_client(self.domain_id, self.iface, timeout=self.timeout)
        self.audio_client, self.audio_error = self._optional_rpc_client(
            "unitree_sdk2py.g1.audio.g1_audio_client",
            "AudioClient",
        )
        self.vui_client, self.vui_error = self._optional_rpc_client(
            "unitree_sdk2py.go2.vui.vui_client",
            "VuiClient",
        )
        self.dds_readers = self._start_dds_readers(DEFAULT_DDS_SAMPLES)

    def _optional_rpc_client(self, module_path: str, class_name: str) -> tuple[Any | None, str | None]:
        client_cls = _load_class(module_path, class_name)
        if client_cls is None:
            return None, f"{module_path}.{class_name} is not available"
        try:
            client = client_cls()
            client.SetTimeout(self.timeout)
            client.Init()
            return client, None
        except Exception as exc:
            return None, str(exc)

    def _start_dds_readers(self, specs: Iterable[DdsTopicSpec]) -> list[LatestDdsSample]:
        readers: list[LatestDdsSample] = []
        seen: set[tuple[str, str, str]] = set()
        for spec in specs:
            key = (spec.topic, spec.module_path, spec.type_name)
            if key in seen:
                continue
            seen.add(key)
            try:
                readers.append(LatestDdsSample(spec))
            except Exception as exc:
                print(f"warning: DDS reader unavailable for {spec.topic} {spec.type_name}: {exc}", flush=True)
        return readers

    def snapshot(self, label: str) -> dict[str, Any]:
        time.sleep(max(0.0, self.dds_wait_s))
        snap: dict[str, Any] = {
            "label": label,
            "timestamp_unix": time.time(),
            "iface": self.iface,
            "domain_id": self.domain_id,
            "motion_switcher": self._motion_snapshot(),
            "loco_rpc": self._loco_snapshot(),
            "audio_client": self._client_snapshot(
                self.audio_client,
                self.audio_error,
                ("GetVolume", "GetApiVersion", "GetServerApiVersion", "GetLeaseId"),
            ),
            "vui_client": self._client_snapshot(
                self.vui_client,
                self.vui_error,
                ("GetSwitch", "GetVolume", "GetBrightness", "GetApiVersion", "GetServerApiVersion", "GetLeaseId"),
            ),
            "dds_samples": [reader.snapshot() for reader in self.dds_readers],
        }
        if self.include_discovery:
            snap["dds_discovery"] = discover_dds_topics(
                self.domain_id, self.iface, self.dds_wait_s, self.discovery_filter
            )
        if self.include_ros:
            snap["ros2"] = ros2_snapshot(self.ros_timeout_s, self.ros_sample_filter)
        return snap

    def _client_snapshot(
        self,
        client: Any | None,
        init_error: str | None,
        method_names: tuple[str, ...],
    ) -> dict[str, Any]:
        if client is None:
            return {"available": False, "error": init_error}
        out: dict[str, Any] = {"available": True}
        for method_name in method_names:
            method = getattr(client, method_name, None)
            if not callable(method):
                continue
            try:
                out[method_name] = _json_value(method(), omit_motion_fields=False)
            except Exception as exc:
                out[method_name] = {"error": str(exc)}
        return out

    def _motion_snapshot(self) -> dict[str, Any]:
        try:
            code, data = self.motion.CheckMode()
            code = int(code)
            return {
                "check_mode_code": code,
                "check_mode_hint": _error_hint(code),
                "data": _json_value(data),
            }
        except Exception as exc:
            return {"error": str(exc)}

    def _loco_snapshot(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, api_id in LOCO_READ_APIS.items():
            try:
                code, data = self.loco._Call(api_id, "{}")  # type: ignore[attr-defined]
                parsed = None
                if data:
                    try:
                        parsed = json.loads(data)
                    except Exception:
                        parsed = data
                out[name] = {"api_id": int(api_id), "code": int(code), "data": parsed}
            except Exception as exc:
                out[name] = {"api_id": int(api_id), "error": str(exc)}

        for method_name in ("GetFsmId", "GetFsmMode", "GetBalanceMode", "GetSwingHeight", "GetStandHeight"):
            method = getattr(self.loco, method_name, None)
            if not callable(method):
                continue
            try:
                out[f"method.{method_name}"] = _json_value(method())
            except Exception as exc:
                out[f"method.{method_name}"] = {"error": str(exc)}
        return out


def discover_dds_topics(domain_id: int, iface: str, seconds: float, pattern: str) -> dict[str, Any]:
    try:
        import re
        from cyclonedds.builtin import (
            BuiltinDataReader,
            BuiltinTopicDcpsPublication,
            BuiltinTopicDcpsSubscription,
            BuiltinTopicDcpsTopic,
        )
        from cyclonedds.domain import Domain, DomainParticipant
        from unitree_sdk2py.core import channel as channel_module
    except Exception as exc:
        return {"error": str(exc)}

    try:
        channel_module.ChannelConfigHasInterface = """<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS>
  <Domain Id="any">
    <General>
      <Interfaces>
        <NetworkInterface name="$__IF_NAME__$" priority="default" multicast="default"/>
      </Interfaces>
    </General>
  </Domain>
</CycloneDDS>"""
        config = channel_module.ChannelConfigHasInterface.replace("$__IF_NAME__$", iface)
        domain = Domain(domain_id, config)
        participant = DomainParticipant(domain_id)
        readers = {
            "topics": BuiltinDataReader(participant, BuiltinTopicDcpsTopic),
            "publications": BuiltinDataReader(participant, BuiltinTopicDcpsPublication),
            "subscriptions": BuiltinDataReader(participant, BuiltinTopicDcpsSubscription),
        }
        regex = re.compile(pattern, flags=re.IGNORECASE) if pattern else None
        found: dict[str, list[dict[str, str]]] = {key: [] for key in readers}
        seen: set[tuple[str, str, str]] = set()
        deadline = time.monotonic() + max(0.1, seconds)
        while time.monotonic() < deadline:
            for group, reader in readers.items():
                for sample in reader.take(100) or []:
                    topic = str(getattr(sample, "topic_name", ""))
                    type_name = str(getattr(sample, "type_name", ""))
                    text = f"{topic} {type_name} {sample}"
                    if regex and not regex.search(text):
                        continue
                    key = (group, topic, type_name)
                    if key in seen:
                        continue
                    seen.add(key)
                    found[group].append({"topic": topic, "type": type_name})
            time.sleep(0.1)
        del domain
        return found
    except Exception as exc:
        return {"error": str(exc)}


def _run_command(args: list[str], timeout_s: float) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            args,
            text=True,
            capture_output=True,
            timeout=max(0.5, timeout_s),
            check=False,
        )
        return {
            "cmd": args,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:
        return {"cmd": args, "error": str(exc)}


def ros2_snapshot(timeout_s: float, sample_filter: str) -> dict[str, Any]:
    if shutil.which("ros2") is None:
        return {"available": False, "error": "ros2 executable not found in PATH"}
    out: dict[str, Any] = {
        "available": True,
        "node_list": _run_command(["ros2", "node", "list"], timeout_s),
        "topic_list_t": _run_command(["ros2", "topic", "list", "-t"], timeout_s),
    }

    topics: list[dict[str, str]] = []
    stdout = str(out["topic_list_t"].get("stdout", ""))
    for line in stdout.splitlines():
        if not line.strip():
            continue
        name, _, type_part = line.partition(" ")
        msg_type = type_part.strip().strip("[]")
        topics.append({"name": name.strip(), "type": msg_type})
    out["topics"] = topics

    import re

    regex = re.compile(sample_filter, flags=re.IGNORECASE) if sample_filter else None
    samples: dict[str, Any] = {}
    for topic in topics:
        name = topic["name"]
        if regex and not regex.search(f"{name} {topic['type']}"):
            continue
        topic_text = f"{name} {topic['type']}".lower()
        echo_result: dict[str, Any]
        if any(token in topic_text for token in ROS_ECHO_OMIT_TOKENS):
            echo_result = {"skipped": "motion-heavy topic; joint/velocity payload omitted"}
        else:
            echo_result = _run_command(["ros2", "topic", "echo", "--once", name], timeout_s)
        samples[name] = {
            "type": topic["type"],
            "info": _run_command(["ros2", "topic", "info", name], timeout_s),
            "echo_once": echo_result,
        }
    out["sampled_topics"] = samples
    return out


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten(item, child))
        return out
    if isinstance(value, list):
        out = {prefix: f"<list len={len(value)}>"}
        for idx, item in enumerate(value[:40]):
            child = f"{prefix}[{idx}]"
            out.update(flatten(item, child))
        return out
    return {prefix: value}


def diff_snapshots(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    before_flat = flatten(before)
    after_flat = flatten(after)
    ignore_prefixes = ("timestamp_unix", "label")
    changes: list[dict[str, Any]] = []
    for path in sorted(set(before_flat) | set(after_flat)):
        if path.startswith(ignore_prefixes):
            continue
        old = before_flat.get(path, "<missing>")
        new = after_flat.get(path, "<missing>")
        if old != new:
            changes.append({"path": path, "before": old, "after": new})
    return changes


def likely_mode_change(change: dict[str, Any]) -> bool:
    path = str(change.get("path", "")).lower()
    if any(token in path for token in ("mode", "fsm", "state", "sport", "wireless", "button", "key", "audio", "vui", "voice")):
        return True
    before = str(change.get("before", "")).lower()
    after = str(change.get("after", "")).lower()
    return any(token in f"{before} {after}" for token in ("wake", "key", "close", "ai_sport", "fsm", "vui", "voice"))


def print_snapshot_summary(snapshot: dict[str, Any]) -> None:
    motion = snapshot.get("motion_switcher", {})
    loco = snapshot.get("loco_rpc", {})
    lowstate_bits: list[str] = []
    for sample in snapshot.get("dds_samples", []):
        data = sample.get("data")
        if not data:
            continue
        topic = sample.get("topic")
        msg_type = str(sample.get("type", "")).rsplit(".", 1)[-1]
        mode_machine = data.get("mode_machine") if isinstance(data, dict) else None
        mode_pr = data.get("mode_pr") if isinstance(data, dict) else None
        lowstate_bits.append(
            f"{topic}/{msg_type}: samples={sample.get('samples_seen')} "
            f"mode_machine={mode_machine} mode_pr={mode_pr}"
        )
    print(f"\n[{snapshot['label']}]")
    print(f"  MotionSwitcher: {json.dumps(motion, default=str)}")
    print(f"  Loco RPC: {json.dumps(loco, default=str)}")
    print(f"  AudioClient: {json.dumps(snapshot.get('audio_client', {}), default=str)}")
    print(f"  VuiClient: {json.dumps(snapshot.get('vui_client', {}), default=str)}")
    for line in lowstate_bits:
        print(f"  DDS {line}")
    ros = snapshot.get("ros2")
    if isinstance(ros, dict):
        print(f"  ROS2: available={ros.get('available')} topics={len(ros.get('topics') or [])}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Passively compare robot state before and after a remote-controller "
            "switch into wake-up, close-interaction, or keymode."
        )
    )
    parser.add_argument("--iface", default=default_dds_iface("eth0"), help="DDS network interface.")
    parser.add_argument("--domain", type=int, default=0, help="DDS domain id.")
    parser.add_argument("--timeout", type=float, default=5.0, help="SDK RPC timeout in seconds.")
    parser.add_argument("--dds-wait-s", type=float, default=1.5, help="Seconds to collect DDS samples per scan.")
    parser.add_argument("--output", default="", help="JSON report path. Defaults to /tmp/wake_up_mode_report_<timestamp>.json.")
    parser.add_argument("--no-ros", action="store_true", help="Skip ROS 2 node/topic probes.")
    parser.add_argument("--ros-timeout-s", type=float, default=2.0, help="Timeout per ros2 command.")
    parser.add_argument(
        "--ros-sample-filter",
        default=DEFAULT_ROS_SAMPLE_FILTER,
        help="Regex for ROS topics to sample with `ros2 topic echo --once`.",
    )
    parser.add_argument("--no-dds-discovery", action="store_true", help="Skip builtin DDS discovery.")
    parser.add_argument(
        "--dds-discovery-filter",
        default=DEFAULT_DDS_TOPIC_FILTER,
        help="Regex for DDS builtin topic discovery output.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of manual remote-controller mode changes to capture.",
    )
    return parser.parse_args()


def prompt_mode_label(index: int) -> str:
    prompt = (
        f"\nUse the remote controller to put the robot into one target mode for scan {index}:\n"
        "  - wake-up mode\n"
        "  - close interaction\n"
        "  - keymode\n"
        "Type the mode name after the robot is visibly in that mode, or q to abort: "
    )
    while True:
        value = input(prompt).strip()
        if value.lower() in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        if value:
            return value
        print("Please type the mode you selected so the report can label this scan.")


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    scanner = RobotStateScanner(
        iface=args.iface,
        domain_id=args.domain,
        timeout=args.timeout,
        dds_wait_s=args.dds_wait_s,
        include_ros=not args.no_ros,
        ros_timeout_s=args.ros_timeout_s,
        ros_sample_filter=args.ros_sample_filter,
        include_discovery=not args.no_dds_discovery,
        discovery_filter=args.dds_discovery_filter,
    )

    report_path = Path(args.output) if args.output else Path(f"/tmp/wake_up_mode_report_{int(time.time())}.json")
    print(f"Connected on domain={args.domain} iface={args.iface!r}.")
    print("This tool does not command a mode switch. It records state before/after manual remote-controller changes.")

    try:
        snapshots: list[dict[str, Any]] = []
        comparisons: list[dict[str, Any]] = []

        print("\nScanning baseline robot state...")
        baseline = scanner.snapshot("baseline")
        snapshots.append(baseline)
        print_snapshot_summary(baseline)

        previous = baseline
        for index in range(1, max(1, int(args.iterations)) + 1):
            mode_label = prompt_mode_label(index)
            print(f"Rescanning after manual selection: {mode_label!r}...")
            current = scanner.snapshot(f"manual_{index}_{mode_label}")
            snapshots.append(current)
            print_snapshot_summary(current)

            changes = diff_snapshots(previous, current)
            likely = [change for change in changes if likely_mode_change(change)]
            comparisons.append(
                {
                    "from": previous["label"],
                    "to": current["label"],
                    "operator_label": mode_label,
                    "changed_count": len(changes),
                    "likely_mode_related_changes": likely,
                    "all_changes": changes,
                }
            )
            print(f"\nChanged fields: {len(changes)}")
            print("Likely mode-related changes:")
            for change in likely[:80]:
                print(f"  {change['path']}: {change['before']!r} -> {change['after']!r}")
            if len(likely) > 80:
                print(f"  ... {len(likely) - 80} more in the JSON report")
            previous = current

        report = {
            "created_unix": time.time(),
            "purpose": (
                "Identify robot variables/settings that change when the remote controller "
                "switches wake-up, close-interaction, or keymode."
            ),
            "snapshots": snapshots,
            "comparisons": comparisons,
        }
        write_report(report_path, report)
        print(f"\nSaved full state/diff report to {report_path}")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted; no robot command was sent.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
