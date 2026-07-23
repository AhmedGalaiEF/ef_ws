#!/usr/bin/env python3
"""Print robot battery and temperature status from Unitree DDS topics."""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any


MODULES_DIR = Path(__file__).resolve().parents[1]
if str(MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(MODULES_DIR))


DEFAULT_BMS_TOPICS = (
    "rt/lf/bmsstate",
    "rt/lf/agvbmsstate",
    "rt/bmsstate",
    "rt/agvbmsstate",
)

LOWSTATE_TOPIC = "rt/lowstate"


def _read_attr(obj: Any, name: str) -> Any | None:
    if obj is None or not hasattr(obj, name):
        return None
    return getattr(obj, name)


def _as_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except Exception:
        return None


def _as_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except Exception:
        return None


def _int_list(value: Any) -> list[int]:
    try:
        return [int(item) for item in value]
    except Exception:
        return []


def _read_robot_temperature(msg: Any) -> dict[str, Any]:
    imu = _read_attr(msg, "imu_state")
    return {
        "imu_temperature_c": _as_float(_read_attr(imu, "temperature")),
        "temperature_ntc1_c": _as_float(_read_attr(msg, "temperature_ntc1")),
        "temperature_ntc2_c": _as_float(_read_attr(msg, "temperature_ntc2")),
    }


def battery_status_from_lowstate(msg: Any) -> dict[str, Any]:
    bms = _read_attr(msg, "bms_state")
    bit_flag = _as_int(_read_attr(msg, "bit_flag"))
    battery_timeout = None if bit_flag is None else bool(bit_flag & 0x10)

    status: dict[str, Any] = {
        "timestamp": time.time(),
        "power_v": _as_float(_read_attr(msg, "power_v")),
        "power_a": _as_float(_read_attr(msg, "power_a")),
        "bit_flag": bit_flag,
        "battery_timeout": battery_timeout,
        "robot_temperature": _read_robot_temperature(msg),
        "bms": None,
    }

    if bms is not None:
        status["bms"] = {
            "status": _as_int(_read_attr(bms, "status")),
            "soc": _as_int(_read_attr(bms, "soc")),
            "current": _as_int(_read_attr(bms, "current")),
            "cycle": _as_int(_read_attr(bms, "cycle")),
            "version_high": _as_int(_read_attr(bms, "version_high")),
            "version_low": _as_int(_read_attr(bms, "version_low")),
            "bq_ntc": _int_list(_read_attr(bms, "bq_ntc")),
            "mcu_ntc": _int_list(_read_attr(bms, "mcu_ntc")),
            "temperature": _int_list(_read_attr(bms, "temperature")),
            "cell_vol": _int_list(_read_attr(bms, "cell_vol")),
            "bmsvoltage": _int_list(_read_attr(bms, "bmsvoltage")),
            "bmsstate": _int_list(_read_attr(bms, "bmsstate")),
        }

    return status


def battery_status_from_bms_msg(msg: Any, *, topic: str) -> dict[str, Any]:
    return {
        "timestamp": time.time(),
        "source": topic,
        "power_v": None,
        "power_a": None,
        "bit_flag": None,
        "battery_timeout": None,
        "robot_temperature": None,
        "bms": {
            "status": _as_int(_read_attr(msg, "status")),
            "soc": _as_int(_read_attr(msg, "soc")),
            "soh": _as_int(_read_attr(msg, "soh")),
            "current": _as_int(_read_attr(msg, "current")),
            "cycle": _as_int(_read_attr(msg, "cycle")),
            "version_high": _as_int(_read_attr(msg, "version_high")),
            "version_low": _as_int(_read_attr(msg, "version_low")),
            "bq_ntc": _int_list(_read_attr(msg, "bq_ntc")),
            "mcu_ntc": _int_list(_read_attr(msg, "mcu_ntc")),
            "temperature": _int_list(_read_attr(msg, "temperature")),
            "cell_vol": _int_list(_read_attr(msg, "cell_vol")),
            "bmsvoltage": _int_list(_read_attr(msg, "bmsvoltage")),
            "bmsstate": _int_list(_read_attr(msg, "bmsstate")),
        },
    }


def add_lowstate_temperature(status: dict[str, Any], msg: Any | None) -> dict[str, Any]:
    if msg is None:
        return status
    status["robot_temperature"] = _read_robot_temperature(msg)
    return status


class LatestBmsSubscriber:
    def __init__(self, topics: list[str]) -> None:
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import BmsState_ as GoBmsState
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import BmsState_ as HgBmsState

        self._lock = threading.Lock()
        self._latest: tuple[Any, str, float] | None = None
        self._subscribers: list[Any] = []

        for topic in topics:
            for msg_type in (HgBmsState, GoBmsState):
                sub = ChannelSubscriber(topic, msg_type)
                sub.Init(self._make_callback(topic), 10)
                self._subscribers.append(sub)

    def _make_callback(self, topic: str):
        def _callback(msg: Any) -> None:
            with self._lock:
                self._latest = (msg, topic, time.time())

        return _callback

    def get_latest(self) -> tuple[Any, str, float] | None:
        with self._lock:
            return self._latest

    def wait(self, timeout: float) -> tuple[Any, str, float] | None:
        deadline = time.time() + max(0.0, float(timeout))
        while time.time() < deadline:
            latest = self.get_latest()
            if latest is not None:
                return latest
            time.sleep(0.05)
        return self.get_latest()


class LatestLowStateSubscriber:
    def __init__(self, topic: str = LOWSTATE_TOPIC) -> None:
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as GoLowState
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as HgLowState

        self._lock = threading.Lock()
        self._latest: tuple[Any, float] | None = None
        self._subscribers: list[Any] = []

        for msg_type in (HgLowState, GoLowState):
            sub = ChannelSubscriber(topic, msg_type)
            sub.Init(self._callback, 10)
            self._subscribers.append(sub)

    def _callback(self, msg: Any) -> None:
        with self._lock:
            self._latest = (msg, time.time())

    def get_latest(self) -> tuple[Any, float] | None:
        with self._lock:
            return self._latest

    def wait(self, timeout: float) -> tuple[Any, float] | None:
        deadline = time.time() + max(0.0, float(timeout))
        while time.time() < deadline:
            latest = self.get_latest()
            if latest is not None:
                return latest
            time.sleep(0.05)
        return self.get_latest()


def print_status(status: dict[str, Any], *, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(status, indent=2, sort_keys=True))
        return

    bms = status.get("bms")
    if isinstance(bms, dict):
        soc = bms.get("soc")
        if soc is not None:
            print(f"Battery percentage: {soc}%")
        else:
            print("Battery percentage: unavailable")
    else:
        print("Battery percentage: unavailable")

    voltage = status.get("power_v")
    current = status.get("power_a")
    if voltage is not None:
        print(f"Battery voltage: {voltage:.2f} V")
    if current is not None:
        print(f"Battery current: {current:.2f} A")
    if status.get("battery_timeout") is not None:
        print(f"Battery timeout flag: {status.get('battery_timeout')}")

    robot_temperature = status.get("robot_temperature")
    if isinstance(robot_temperature, dict):
        imu_temp = robot_temperature.get("imu_temperature_c")
        ntc1 = robot_temperature.get("temperature_ntc1_c")
        ntc2 = robot_temperature.get("temperature_ntc2_c")
        if imu_temp is not None:
            print(f"Robot IMU temperature: {imu_temp:.1f} C")
        if ntc1 is not None or ntc2 is not None:
            ntc1_text = "n/a" if ntc1 is None else f"{ntc1:.1f} C"
            ntc2_text = "n/a" if ntc2 is None else f"{ntc2:.1f} C"
            print(f"Robot body NTC temperatures: {ntc1_text} / {ntc2_text}")

    if not isinstance(bms, dict):
        print("BMS state: unavailable on this lowstate message")
        return

    if bms.get("soh") is not None:
        print(f"BMS health: {bms.get('soh')}%")
    if bms.get("status") is not None:
        print(f"BMS status: {bms.get('status')}")
    if bms.get("current") is not None:
        print(f"BMS current: {bms.get('current')}")
    if bms.get("cycle") is not None:
        print(f"BMS cycle count: {bms.get('cycle')}")
    if status.get("source"):
        print(f"Source: {status.get('source')}")
    if bms.get("bq_ntc") or bms.get("mcu_ntc"):
        print(f"BMS temps bq/mcu: {bms.get('bq_ntc')} / {bms.get('mcu_ntc')}")
    if bms.get("temperature"):
        print(f"BMS temperature: {bms.get('temperature')}")
    if bms.get("bmsvoltage"):
        print(f"BMS voltage raw: {bms.get('bmsvoltage')}")
    if bms.get("cell_vol"):
        print(f"Cell voltages: {bms.get('cell_vol')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print G1 battery and robot temperature from DDS topics.")
    parser.add_argument("--iface", default=os.environ.get("G1_IFACE", "eth0"), help="DDS network interface.")
    parser.add_argument("--domain-id", type=int, default=int(os.environ.get("ROS_DOMAIN_ID", "0")))
    parser.add_argument("--timeout", type=float, default=3.0, help="Seconds to wait for a battery message.")
    parser.add_argument(
        "--source",
        choices=("auto", "bms", "lowstate"),
        default="auto",
        help="Read direct BMS DDS topics, lowstate, or try BMS first then lowstate.",
    )
    parser.add_argument(
        "--topic",
        action="append",
        default=[],
        help="BMS DDS topic to try. Can be repeated. Defaults to common G1 BMS topics.",
    )
    parser.add_argument("--watch", action="store_true", help="Keep printing until Ctrl-C.")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between watch-mode prints.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    return parser.parse_args()


def _run_bms(args: argparse.Namespace) -> bool:
    from dds_env import ensure_channel_factory_initialized

    ensure_channel_factory_initialized(int(args.domain_id), str(args.iface))
    topics = [str(topic) for topic in args.topic] or list(DEFAULT_BMS_TOPICS)
    bms_sub = LatestBmsSubscriber(topics)
    lowstate_sub = LatestLowStateSubscriber()
    latest = bms_sub.wait(float(args.timeout))
    if latest is None:
        print(
            f"No BMS message received within {args.timeout:.1f}s "
            f"(iface={args.iface}, domain_id={args.domain_id}, topics={', '.join(topics)}).",
            file=sys.stderr,
        )
        return False

    while True:
        latest = bms_sub.get_latest()
        if latest is None:
            return False
        msg, topic, _ts = latest
        lowstate = lowstate_sub.wait(0.2)
        lowstate_msg = None if lowstate is None else lowstate[0]
        status = add_lowstate_temperature(battery_status_from_bms_msg(msg, topic=topic), lowstate_msg)
        print_status(status, as_json=bool(args.json))
        if not args.watch:
            return True
        print()
        time.sleep(max(0.1, float(args.interval)))


def _run_lowstate(args: argparse.Namespace) -> bool:
    from sdk_client import Robot

    robot = Robot(
        iface=str(args.iface),
        domain_id=int(args.domain_id),
        safety_boot=False,
        recover_dev_mode_on_init=False,
        auto_start_sensors=True,
    )

    if not robot.wait_for_low_state(timeout=float(args.timeout)):
        print(
            f"No rt/lowstate message received within {args.timeout:.1f}s "
            f"(iface={args.iface}, domain_id={args.domain_id}).",
            file=sys.stderr,
        )
        return False

    while True:
        msg = robot.get_low_state_msg()
        if msg is None:
            print("No lowstate message available", file=sys.stderr)
            return False
        print_status(battery_status_from_lowstate(msg), as_json=bool(args.json))
        if not args.watch:
            return True
        print()
        time.sleep(max(0.1, float(args.interval)))


def main() -> int:
    args = parse_args()

    try:
        if args.source == "bms":
            return 0 if _run_bms(args) else 1
        if args.source == "lowstate":
            return 0 if _run_lowstate(args) else 1
        if _run_bms(args):
            return 0
        print("Falling back to rt/lowstate.", file=sys.stderr)
        return 0 if _run_lowstate(args) else 1
    except KeyboardInterrupt:
        return 130

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
