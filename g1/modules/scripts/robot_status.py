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
STALE_AFTER_S = 1.5


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


def _vec(value: Any, size: int) -> list[float] | None:
    try:
        values = [float(value[i]) for i in range(size)]
    except Exception:
        return None
    return values


def _read_robot_temperature(msg: Any) -> dict[str, Any]:
    imu = _read_attr(msg, "imu_state")
    return {
        "imu_temperature_c": _as_float(_read_attr(imu, "temperature")),
        "temperature_ntc1_c": _as_float(_read_attr(msg, "temperature_ntc1")),
        "temperature_ntc2_c": _as_float(_read_attr(msg, "temperature_ntc2")),
    }


def _read_lowstate_status(msg: Any) -> dict[str, Any]:
    imu = _read_attr(msg, "imu_state")
    return {
        "topic": LOWSTATE_TOPIC,
        "age_s": None,
        "fresh": None,
        "mode_machine": _as_int(_read_attr(msg, "mode_machine")),
        "mode_pr": _as_int(_read_attr(msg, "mode_pr")),
        "tick": _as_int(_read_attr(msg, "tick")),
        "imu": {
            "rpy": _vec(_read_attr(imu, "rpy"), 3),
            "gyro": _vec(_read_attr(imu, "gyroscope"), 3),
            "accelerometer": _vec(_read_attr(imu, "accelerometer"), 3),
            "quaternion": _vec(_read_attr(imu, "quaternion"), 4),
        },
        "foot_force": _int_list(_read_attr(msg, "foot_force")),
        "foot_force_est": _int_list(_read_attr(msg, "foot_force_est")),
        "fan_frequency": _int_list(_read_attr(msg, "fan_frequency")),
    }


def _set_lowstate_age(status: dict[str, Any], timestamp: float | None) -> None:
    lowstate = status.get("lowstate")
    if not isinstance(lowstate, dict) or timestamp is None:
        return
    age_s = max(0.0, time.time() - float(timestamp))
    lowstate["age_s"] = age_s
    lowstate["fresh"] = age_s <= STALE_AFTER_S


def _nonzero_ints(values: Any) -> list[int]:
    return [int(value) for value in values if int(value) != 0] if isinstance(values, list) else []


def _derive_status(status: dict[str, Any], args: argparse.Namespace | None = None) -> dict[str, Any]:
    bms = status.get("bms")
    derived: dict[str, Any] = {
        "voltage_v": _as_float(status.get("power_v")),
        "current_a": _as_float(status.get("power_a")),
        "charge_state": None,
        "cell_min_v": None,
        "cell_max_v": None,
        "cell_imbalance_v": None,
        "bms_temperature_max_c": None,
        "foot_loaded": None,
    }

    if isinstance(bms, dict):
        if derived["current_a"] is None and bms.get("current") is not None:
            derived["current_a"] = float(bms["current"]) / 1000.0
        current_a = derived.get("current_a")
        if current_a is not None:
            if float(current_a) > 0.05:
                derived["charge_state"] = "charging"
            elif float(current_a) < -0.05:
                derived["charge_state"] = "discharging"
            else:
                derived["charge_state"] = "idle"

        voltages = _nonzero_ints(bms.get("bmsvoltage"))
        if derived["voltage_v"] is None and voltages:
            derived["voltage_v"] = max(voltages) / 1000.0

        cells = _nonzero_ints(bms.get("cell_vol"))
        if cells:
            cell_values_v = [value / 1000.0 for value in cells]
            derived["cell_min_v"] = min(cell_values_v)
            derived["cell_max_v"] = max(cell_values_v)
            derived["cell_imbalance_v"] = derived["cell_max_v"] - derived["cell_min_v"]

        bms_temps = _nonzero_ints(bms.get("temperature"))
        if bms_temps:
            derived["bms_temperature_max_c"] = max(bms_temps)

    lowstate = status.get("lowstate")
    if isinstance(lowstate, dict):
        foot_force = _nonzero_ints(lowstate.get("foot_force"))
        if foot_force:
            derived["foot_loaded"] = sum(foot_force) > 40

    status["derived"] = derived
    status["warnings"] = _build_warnings(status, args)
    status["health"] = "CRITICAL" if any(w["severity"] == "critical" for w in status["warnings"]) else (
        "WARNING" if status["warnings"] else "OK"
    )
    return status


def _build_warnings(status: dict[str, Any], args: argparse.Namespace | None = None) -> list[dict[str, str]]:
    warn_battery = float(getattr(args, "warn_battery", 20.0))
    critical_battery = float(getattr(args, "critical_battery", 10.0))
    warn_imu_temp = float(getattr(args, "warn_imu_temp", 80.0))
    critical_imu_temp = float(getattr(args, "critical_imu_temp", 90.0))
    warn_bms_temp = float(getattr(args, "warn_bms_temp", 55.0))
    critical_bms_temp = float(getattr(args, "critical_bms_temp", 65.0))
    warn_cell_imbalance = float(getattr(args, "warn_cell_imbalance", 0.08))
    critical_cell_imbalance = float(getattr(args, "critical_cell_imbalance", 0.20))

    warnings: list[dict[str, str]] = []
    bms = status.get("bms")
    derived = status.get("derived") if isinstance(status.get("derived"), dict) else {}

    soc = bms.get("soc") if isinstance(bms, dict) else None
    if soc is not None:
        if float(soc) <= critical_battery:
            warnings.append({"severity": "critical", "message": f"battery low: {soc}%"})
        elif float(soc) <= warn_battery:
            warnings.append({"severity": "warning", "message": f"battery low: {soc}%"})

    robot_temperature = status.get("robot_temperature")
    imu_temp = robot_temperature.get("imu_temperature_c") if isinstance(robot_temperature, dict) else None
    if imu_temp is not None:
        if float(imu_temp) >= critical_imu_temp:
            warnings.append({"severity": "critical", "message": f"robot IMU temperature high: {imu_temp:.1f} C"})
        elif float(imu_temp) >= warn_imu_temp:
            warnings.append({"severity": "warning", "message": f"robot IMU temperature high: {imu_temp:.1f} C"})

    bms_temp = derived.get("bms_temperature_max_c")
    if bms_temp is not None:
        if float(bms_temp) >= critical_bms_temp:
            warnings.append({"severity": "critical", "message": f"BMS temperature high: {bms_temp:.1f} C"})
        elif float(bms_temp) >= warn_bms_temp:
            warnings.append({"severity": "warning", "message": f"BMS temperature high: {bms_temp:.1f} C"})

    imbalance = derived.get("cell_imbalance_v")
    if imbalance is not None:
        if float(imbalance) >= critical_cell_imbalance:
            warnings.append({"severity": "critical", "message": f"cell imbalance high: {imbalance:.3f} V"})
        elif float(imbalance) >= warn_cell_imbalance:
            warnings.append({"severity": "warning", "message": f"cell imbalance high: {imbalance:.3f} V"})

    if status.get("battery_timeout") is True:
        warnings.append({"severity": "critical", "message": "battery timeout flag is set"})

    lowstate = status.get("lowstate")
    if isinstance(lowstate, dict) and lowstate.get("fresh") is False:
        warnings.append({"severity": "warning", "message": f"lowstate stale: {lowstate.get('age_s'):.2f}s old"})

    return warnings


def robot_status_from_lowstate(msg: Any, *, timestamp: float | None = None, args: argparse.Namespace | None = None) -> dict[str, Any]:
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
        "lowstate": _read_lowstate_status(msg),
        "bms": None,
    }
    _set_lowstate_age(status, timestamp)

    if bms is not None:
        status["bms"] = {
            "fn": _as_int(_read_attr(bms, "fn")),
            "status": _as_int(_read_attr(bms, "status")),
            "soc": _as_int(_read_attr(bms, "soc")),
            "soh": _as_int(_read_attr(bms, "soh")),
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
            "manufacturer_date": _as_int(_read_attr(bms, "manufacturer_date")),
        }

    return _derive_status(status, args)


def battery_status_from_lowstate(msg: Any) -> dict[str, Any]:
    return robot_status_from_lowstate(msg)


def battery_status_from_bms_msg(msg: Any, *, topic: str) -> dict[str, Any]:
    return {
        "timestamp": time.time(),
        "source": topic,
        "power_v": None,
        "power_a": None,
        "bit_flag": None,
        "battery_timeout": None,
        "robot_temperature": None,
        "lowstate": None,
        "bms": {
            "fn": _as_int(_read_attr(msg, "fn")),
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
            "manufacturer_date": _as_int(_read_attr(msg, "manufacturer_date")),
        },
    }


def add_lowstate_status(status: dict[str, Any], msg: Any | None, *, timestamp: float | None = None, args: argparse.Namespace | None = None) -> dict[str, Any]:
    if msg is None:
        return _derive_status(status, args)
    status["robot_temperature"] = _read_robot_temperature(msg)
    status["lowstate"] = _read_lowstate_status(msg)
    _set_lowstate_age(status, timestamp)
    return _derive_status(status, args)


def add_lowstate_temperature(status: dict[str, Any], msg: Any | None) -> dict[str, Any]:
    return add_lowstate_status(status, msg)


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


def _format_vec(values: Any, precision: int = 3) -> str:
    if not values:
        return "n/a"
    return "[" + ", ".join(f"{float(value):.{precision}f}" for value in values) + "]"


def print_status(
    status: dict[str, Any],
    *,
    as_json: bool = False,
    brief: bool = False,
    full: bool = False,
) -> None:
    if as_json:
        print(json.dumps(status, indent=2, sort_keys=True))
        return

    print(f"Robot status: {status.get('health', 'UNKNOWN')}")
    warnings = status.get("warnings")
    if isinstance(warnings, list) and warnings:
        for warning in warnings:
            print(f"{str(warning.get('severity', 'warning')).upper()}: {warning.get('message')}")

    bms = status.get("bms")
    if isinstance(bms, dict):
        soc = bms.get("soc")
        if soc is not None:
            print(f"Battery percentage: {soc}%")
        else:
            print("Battery percentage: unavailable")
    else:
        print("Battery percentage: unavailable")

    derived = status.get("derived") if isinstance(status.get("derived"), dict) else {}
    voltage = derived.get("voltage_v")
    current = derived.get("current_a")
    if voltage is not None:
        print(f"Battery voltage: {voltage:.2f} V")
    if current is not None:
        print(f"Battery current: {current:.2f} A")
    if derived.get("charge_state") is not None:
        print(f"Battery state: {derived.get('charge_state')}")
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

    bms_temp = derived.get("bms_temperature_max_c")
    if bms_temp is not None:
        print(f"BMS max temperature: {bms_temp:.1f} C")

    if derived.get("cell_min_v") is not None and derived.get("cell_max_v") is not None:
        print(
            "Cell voltage range: "
            f"{derived.get('cell_min_v'):.3f} V - {derived.get('cell_max_v'):.3f} V, "
            f"imbalance: {derived.get('cell_imbalance_v'):.3f} V"
        )

    lowstate = status.get("lowstate")
    if isinstance(lowstate, dict):
        mode_bits = []
        if lowstate.get("mode_machine") is not None:
            mode_bits.append(f"mode_machine={lowstate.get('mode_machine')}")
        if lowstate.get("mode_pr") is not None:
            mode_bits.append(f"mode_pr={lowstate.get('mode_pr')}")
        if lowstate.get("tick") is not None:
            mode_bits.append(f"tick={lowstate.get('tick')}")
        if mode_bits:
            print(f"Lowstate: {', '.join(mode_bits)}")
        if lowstate.get("age_s") is not None:
            freshness = "fresh" if lowstate.get("fresh") else "stale"
            print(f"Lowstate age: {lowstate.get('age_s'):.2f}s ({freshness})")

    if brief:
        return

    if isinstance(lowstate, dict):
        imu = lowstate.get("imu") if isinstance(lowstate.get("imu"), dict) else {}
        print(f"IMU RPY(rad): {_format_vec(imu.get('rpy'))}")
        if full:
            print(f"IMU gyro(rad/s): {_format_vec(imu.get('gyro'))}")
            print(f"IMU accelerometer(m/s^2): {_format_vec(imu.get('accelerometer'))}")
            print(f"IMU quaternion: {_format_vec(imu.get('quaternion'))}")

        foot_force = lowstate.get("foot_force")
        if foot_force:
            print(f"Foot force: {foot_force}")
            if derived.get("foot_loaded") is not None:
                print(f"Feet loaded: {derived.get('foot_loaded')}")
        foot_force_est = lowstate.get("foot_force_est")
        if full and foot_force_est:
            print(f"Foot force estimated: {foot_force_est}")
        fan_frequency = lowstate.get("fan_frequency")
        if fan_frequency:
            print(f"Fan frequency: {fan_frequency}")

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
    if full and (bms.get("bq_ntc") or bms.get("mcu_ntc")):
        print(f"BMS temps bq/mcu: {bms.get('bq_ntc')} / {bms.get('mcu_ntc')}")
    if full and bms.get("temperature"):
        print(f"BMS temperature: {bms.get('temperature')}")
    if full and bms.get("bmsvoltage"):
        print(f"BMS voltage raw: {bms.get('bmsvoltage')}")
    if full and bms.get("cell_vol"):
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
    parser.add_argument("--clear", action="store_true", help="Clear the terminal before each watch-mode update.")
    parser.add_argument("--changes-only", action="store_true", help="In watch mode, only print when status changes.")
    parser.add_argument("--brief", action="store_true", help="Print only the health, battery, temperature, and key state summary.")
    parser.add_argument("--full", action="store_true", help="Print detailed IMU, BMS, foot-force, and raw battery fields.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    parser.add_argument("--warn-battery", type=float, default=20.0, help="Warning battery percentage threshold.")
    parser.add_argument("--critical-battery", type=float, default=10.0, help="Critical battery percentage threshold.")
    parser.add_argument("--warn-imu-temp", type=float, default=80.0, help="Warning robot IMU temperature in C.")
    parser.add_argument("--critical-imu-temp", type=float, default=90.0, help="Critical robot IMU temperature in C.")
    parser.add_argument("--warn-bms-temp", type=float, default=55.0, help="Warning BMS temperature in C.")
    parser.add_argument("--critical-bms-temp", type=float, default=65.0, help="Critical BMS temperature in C.")
    parser.add_argument("--warn-cell-imbalance", type=float, default=0.08, help="Warning cell imbalance in volts.")
    parser.add_argument("--critical-cell-imbalance", type=float, default=0.20, help="Critical cell imbalance in volts.")
    return parser.parse_args()


def _status_key(status: dict[str, Any]) -> str:
    compact = dict(status)
    compact.pop("timestamp", None)
    lowstate = compact.get("lowstate")
    if isinstance(lowstate, dict):
        lowstate = dict(lowstate)
        lowstate.pop("age_s", None)
        compact["lowstate"] = lowstate
    return json.dumps(compact, sort_keys=True, default=str)


def _maybe_print_status(status: dict[str, Any], args: argparse.Namespace, last_key: str | None) -> str:
    key = _status_key(status)
    if bool(args.changes_only) and last_key == key:
        return last_key
    if bool(args.clear):
        print("\033[2J\033[H", end="")
    if bool(args.watch):
        print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print_status(status, as_json=bool(args.json), brief=bool(args.brief), full=bool(args.full))
    return key


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

    last_key: str | None = None
    while True:
        latest = bms_sub.get_latest()
        if latest is None:
            return False
        msg, topic, _ts = latest
        lowstate = lowstate_sub.wait(0.2)
        lowstate_msg = None if lowstate is None else lowstate[0]
        lowstate_ts = None if lowstate is None else lowstate[1]
        status = add_lowstate_status(battery_status_from_bms_msg(msg, topic=topic), lowstate_msg, timestamp=lowstate_ts, args=args)
        last_key = _maybe_print_status(status, args, last_key)
        if not args.watch:
            return True
        if not bool(args.clear):
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

    last_key: str | None = None
    while True:
        lowstate_latest = robot._lowstate_sub.get_latest() if robot._lowstate_sub is not None else (None, 0.0)
        msg = lowstate_latest[0]
        lowstate_ts = lowstate_latest[1]
        if msg is None:
            print("No lowstate message available", file=sys.stderr)
            return False
        status = robot_status_from_lowstate(msg, timestamp=lowstate_ts, args=args)
        last_key = _maybe_print_status(status, args, last_key)
        if not args.watch:
            return True
        if not bool(args.clear):
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
