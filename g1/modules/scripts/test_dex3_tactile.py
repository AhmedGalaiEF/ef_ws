#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from dds_env import default_dds_iface, ensure_channel_factory_initialized
from unitree_sdk2py.core import channel as channel_module
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_


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
channel_module.ChannelConfigAutoDetermine = """<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS>
  <Domain Id="any">
    <General>
      <Interfaces>
        <NetworkInterface autodetermine="true" priority="default" multicast="default" />
      </Interfaces>
    </General>
  </Domain>
</CycloneDDS>"""
os.environ.setdefault(
    "CYCLONEDDS_URI",
    "<CycloneDDS><Domain><Tracing><Category>none</Category></Tracing></Domain></CycloneDDS>",
)


HAND_STATE_TOPIC_BY_SIDE = {
    "left": "rt/dex3/left/state",
    "right": "rt/dex3/right/state",
}

INVALID_DOC_VALUE = 30000.0
VALID_DOC_THRESHOLD = 100000.0
WARN_CHANGE_THRESHOLD = 2000.0
HOT_CHANGE_THRESHOLD = 5000.0

ANSI_RESET = "\033[0m"
ANSI_DIM = "\033[2m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_RED = "\033[31m"
ANSI_CLEAR = "\033[2J\033[H"

SENSOR_LABELS_BY_COUNT = {
    6: {
        0: "thumb? pad A",
        1: "thumb? pad B",
        2: "index? pad A",
        3: "index? pad B",
        4: "middle? pad A",
        5: "middle? pad B",
    },
    9: {
        0: "thumb? pad A",
        1: "thumb? pad B",
        2: "thumb? pad C",
        3: "index? pad A",
        4: "index? pad B",
        5: "index? pad C",
        6: "middle? pad A",
        7: "middle? pad B",
        8: "middle? pad C",
    },
}


@dataclass
class TactileSensor:
    index: int
    sensor_id: int | None
    values: list[float]
    temperatures: list[float]
    temp: float | None
    lost: int | None

    @property
    def valid_count(self) -> int:
        return sum(1 for value in self.values if is_valid_tactile_value(value))

    @property
    def invalid_count(self) -> int:
        return sum(1 for value in self.values if is_invalid_tactile_value(value))

    @property
    def active_count(self) -> int:
        return sum(1 for value in self.values if value > 0.0 and not is_invalid_tactile_value(value))

    @property
    def max_value(self) -> float | None:
        return max(self.values) if self.values else None

    @property
    def mean_value(self) -> float | None:
        return sum(self.values) / len(self.values) if self.values else None


@dataclass
class HandTactileSnapshot:
    hand: str
    topic: str
    timestamp: float
    motor_count: int
    sensors: list[TactileSensor] = field(default_factory=list)
    power_v: float | None = None
    power_a: float | None = None
    system_v: float | None = None
    device_v: float | None = None
    error: list[int] = field(default_factory=list)

    @property
    def tactile_value_count(self) -> int:
        return sum(len(sensor.values) for sensor in self.sensors)

    @property
    def valid_count(self) -> int:
        return sum(sensor.valid_count for sensor in self.sensors)

    @property
    def invalid_count(self) -> int:
        return sum(sensor.invalid_count for sensor in self.sensors)

    @property
    def active_count(self) -> int:
        return sum(sensor.active_count for sensor in self.sensors)

    @property
    def max_value(self) -> float | None:
        values = [sensor.max_value for sensor in self.sensors if sensor.max_value is not None]
        return max(values) if values else None


class LatestTactileState:
    def __init__(self, hand: str, topic: str, queue_len: int) -> None:
        self.hand = hand
        self.topic = topic
        self._lock = threading.Lock()
        self._event = threading.Event()
        self.snapshot: HandTactileSnapshot | None = None
        self.count = 0
        self.subscriber = ChannelSubscriber(topic, HandState_)
        self.subscriber.Init(self._callback, int(queue_len))

    def _callback(self, msg: Any) -> None:
        snapshot = parse_hand_state(self.hand, self.topic, msg)
        with self._lock:
            self.snapshot = snapshot
            self.count += 1
        self._event.set()

    def wait(self, timeout_s: float) -> HandTactileSnapshot | None:
        if not self._event.wait(max(0.0, float(timeout_s))):
            return None
        with self._lock:
            return self.snapshot

    def latest(self) -> HandTactileSnapshot | None:
        with self._lock:
            return self.snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live monitor for Dex3 tactile sensor data."
    )
    parser.add_argument("--iface", default=None, help="DDS network interface. Default: live eth0, else auto.")
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain id.")
    parser.add_argument("--hand", choices=("left", "right", "both"), default="both")
    parser.add_argument("--timeout-s", type=float, default=8.0, help="Seconds to wait for hand state.")
    parser.add_argument("--sample-s", type=float, default=0.0,
                        help="In --once mode, keep sampling for this many seconds after first data.")
    parser.add_argument("--once", action="store_true", help="Print one snapshot and exit.")
    parser.add_argument("--refresh-hz", type=float, default=5.0, help="Live display refresh rate.")
    parser.add_argument("--no-clear", action="store_true", help="Do not clear the terminal between updates.")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI color highlighting.")
    parser.add_argument("--change-threshold", type=float, default=WARN_CHANGE_THRESHOLD,
                        help="Raw pressure delta from baseline required before coloring a cell.")
    parser.add_argument("--hot-change-threshold", type=float, default=HOT_CHANGE_THRESHOLD,
                        help="Raw pressure delta from baseline colored red.")
    parser.add_argument("--queue-len", type=int, default=20, help="DDS subscriber queue length.")
    parser.add_argument("--json", action="store_true", help="Print one snapshot as JSON and exit.")
    parser.add_argument("--raw", action="store_true", help="Print all 12 values for each tactile sensor.")
    parser.add_argument("--scale", type=float, default=10000.0,
                        help="Display scale for raw pressure values. Unitree docs recommend 10000.")
    parser.add_argument("--require-valid", action="store_true",
                        help="Fail unless at least one value is >= 100000, per Unitree docs.")
    return parser.parse_args()


def is_invalid_tactile_value(value: float) -> bool:
    return abs(float(value) - INVALID_DOC_VALUE) < 0.5


def is_valid_tactile_value(value: float) -> bool:
    return float(value) >= VALID_DOC_THRESHOLD


def scalar_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def series_from_attrs(obj: Any, names: tuple[str, ...]) -> list[float]:
    for name in names:
        try:
            raw = getattr(obj, name)
        except Exception:
            continue
        if raw is None:
            continue
        try:
            return [float(value) for value in list(raw)]
        except Exception:
            continue
    return []


def int_from_attrs(obj: Any, names: tuple[str, ...]) -> int | None:
    for name in names:
        try:
            raw = getattr(obj, name)
        except Exception:
            continue
        if raw is None:
            continue
        try:
            return int(raw)
        except Exception:
            continue
    return None


def parse_tactile_sensor(index: int, sensor: Any) -> TactileSensor:
    values = series_from_attrs(sensor, ("pressure", "data"))
    temperatures = series_from_attrs(sensor, ("temperature",))
    temp = scalar_or_none(getattr(sensor, "temp", None))
    if temp is None and len(temperatures) == 1:
        temp = temperatures[0]
    return TactileSensor(
        index=index,
        sensor_id=int_from_attrs(sensor, ("id", "sensor_id")),
        values=values,
        temperatures=temperatures,
        temp=temp,
        lost=int_from_attrs(sensor, ("lost",)),
    )


def parse_hand_state(hand: str, topic: str, msg: Any) -> HandTactileSnapshot:
    press_sensor_state = list(getattr(msg, "press_sensor_state", []) or [])
    sensors = [
        parse_tactile_sensor(index, sensor)
        for index, sensor in enumerate(press_sensor_state)
    ]
    return HandTactileSnapshot(
        hand=hand,
        topic=topic,
        timestamp=time.time(),
        motor_count=len(list(getattr(msg, "motor_state", []) or [])),
        sensors=sensors,
        power_v=scalar_or_none(getattr(msg, "power_v", None)),
        power_a=scalar_or_none(getattr(msg, "power_a", None)),
        system_v=scalar_or_none(getattr(msg, "system_v", None)),
        device_v=scalar_or_none(getattr(msg, "device_v", None)),
        error=[int(value) for value in list(getattr(msg, "error", []) or [])],
    )


def sensor_to_dict(sensor: TactileSensor, scale: float) -> dict[str, Any]:
    return {
        "index": sensor.index,
        "id": sensor.sensor_id,
        "values": sensor.values,
        "scaled_values": [value / scale for value in sensor.values] if scale else sensor.values,
        "temperature": sensor.temp,
        "temperatures": sensor.temperatures,
        "lost": sensor.lost,
        "valid_count": sensor.valid_count,
        "invalid_count": sensor.invalid_count,
        "active_count": sensor.active_count,
        "max": sensor.max_value,
        "mean": sensor.mean_value,
    }


def snapshot_to_dict(snapshot: HandTactileSnapshot, scale: float) -> dict[str, Any]:
    return {
        "hand": snapshot.hand,
        "topic": snapshot.topic,
        "timestamp": snapshot.timestamp,
        "age_s": max(0.0, time.time() - snapshot.timestamp),
        "motor_count": snapshot.motor_count,
        "sensor_count": len(snapshot.sensors),
        "tactile_value_count": snapshot.tactile_value_count,
        "valid_count": snapshot.valid_count,
        "invalid_count": snapshot.invalid_count,
        "active_count": snapshot.active_count,
        "max": snapshot.max_value,
        "power_v": snapshot.power_v,
        "power_a": snapshot.power_a,
        "system_v": snapshot.system_v,
        "device_v": snapshot.device_v,
        "error": snapshot.error,
        "sensors": [sensor_to_dict(sensor, scale) for sensor in snapshot.sensors],
    }


def format_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def colorize(text: str, color: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{color}{text}{ANSI_RESET}"


def sensor_label(sensor: TactileSensor, sensor_count: int) -> str:
    labels = SENSOR_LABELS_BY_COUNT.get(sensor_count, {})
    return labels.get(sensor.index, "unknown packet")


def baseline_value(
    baseline: HandTactileSnapshot | None,
    sensor_index: int,
    value_index: int,
) -> float | None:
    if baseline is None:
        return None
    if sensor_index >= len(baseline.sensors):
        return None
    values = baseline.sensors[sensor_index].values
    if value_index >= len(values):
        return None
    return values[value_index]


def tactile_change_color(
    value: float,
    base_value: float | None,
    *,
    change_threshold: float,
    hot_change_threshold: float,
    enabled: bool,
) -> str:
    if is_invalid_tactile_value(value):
        return ANSI_DIM if enabled else ""
    if base_value is None:
        return ""
    delta = abs(float(value) - float(base_value))
    if delta >= hot_change_threshold:
        return ANSI_RED if enabled else ""
    if delta >= change_threshold:
        return ANSI_YELLOW if enabled else ""
    return ""


def format_tactile_value(
    value: float,
    base_value: float | None,
    *,
    scale: float,
    color: bool,
    change_threshold: float,
    hot_change_threshold: float,
) -> str:
    shown = value / scale if scale else value
    text = f"{shown:7.4f}" if scale else f"{shown:8.1f}"
    ansi = tactile_change_color(
        value,
        base_value,
        change_threshold=change_threshold,
        hot_change_threshold=hot_change_threshold,
        enabled=color,
    )
    return colorize(text, ansi, bool(ansi))


def format_sensor_matrix(
    sensor: TactileSensor,
    baseline: HandTactileSnapshot | None,
    *,
    scale: float,
    color: bool,
    change_threshold: float,
    hot_change_threshold: float,
) -> list[str]:
    values = sensor.values
    cells = list(values[:12])
    while len(cells) < 12:
        cells.append(0.0)
    rows = []
    for row_idx in range(3):
        offset = row_idx * 4
        row = cells[offset:offset + 4]
        rows.append(" ".join(
            format_tactile_value(
                value,
                baseline_value(baseline, sensor.index, offset + cell_idx),
                scale=scale,
                color=color,
                change_threshold=change_threshold,
                hot_change_threshold=hot_change_threshold,
            )
            for cell_idx, value in enumerate(row)
        ))
    return rows


def print_sensor_map(sensor_count: int) -> None:
    labels = SENSOR_LABELS_BY_COUNT.get(sensor_count)
    print("Sensor packet map:")
    if sensor_count == 9:
        print("  inferred from observed 9 packets: [0,1,2]=thumb?, [3,4,5]=index?, [6,7,8]=middle?")
        print("  Press one fingertip at a time to confirm the mapping on this firmware.")
    elif sensor_count == 6:
        print("  public docs mention 6 tactile locations: [0,1]=thumb?, [2,3]=index?, [4,5]=middle?")
        print("  Packet IDs are not exposed by this Python IDL, so verify by pressing one finger at a time.")
    else:
        print(f"  no built-in label map for {sensor_count} packets; use the packet indices below.")
    if labels:
        print("  " + ", ".join(f"{idx}:{label}" for idx, label in labels.items()))


def print_snapshot(
    snapshot: HandTactileSnapshot,
    *,
    scale: float,
    raw: bool,
    color: bool,
    baseline: HandTactileSnapshot | None,
    change_threshold: float,
    hot_change_threshold: float,
) -> None:
    age_s = max(0.0, time.time() - snapshot.timestamp)
    print(
        f"{snapshot.hand}: topic={snapshot.topic} age={age_s:.3f}s "
        f"motors={snapshot.motor_count} sensors={len(snapshot.sensors)} "
        f"values={snapshot.tactile_value_count} valid>={int(VALID_DOC_THRESHOLD)}:{snapshot.valid_count} "
        f"invalid=={int(INVALID_DOC_VALUE)}:{snapshot.invalid_count} active:{snapshot.active_count} "
        f"max={format_float(snapshot.max_value, 1)}"
    )
    print_sensor_map(len(snapshot.sensors))
    print(
        "Legend: "
        + colorize("changed", ANSI_YELLOW, color)
        + f" delta>={int(change_threshold)} raw, "
        + colorize("big change", ANSI_RED, color)
        + f" delta>={int(hot_change_threshold)} raw, "
        + colorize("invalid/no value", ANSI_DIM, color)
        + f" ==30000. Display scale={format_float(scale, 0)} raw units per 1.0000."
    )
    for sensor in snapshot.sensors:
        label = f"  sensor[{sensor.index}] {sensor_label(sensor, len(snapshot.sensors))}"
        if sensor.sensor_id is not None:
            label += f" id={sensor.sensor_id}"
        label += (
            f" count={len(sensor.values)} valid={sensor.valid_count} "
            f"invalid={sensor.invalid_count} active={sensor.active_count} "
            f"max={format_float(sensor.max_value, 1)} mean={format_float(sensor.mean_value, 1)}"
        )
        if sensor.lost is not None:
            label += f" lost={sensor.lost}"
        if sensor.temp is not None:
            label += f" temp={format_float(sensor.temp, 1)}"
        print(label)
        if raw:
            for row in format_sensor_matrix(
                sensor,
                baseline,
                scale=scale,
                color=color,
                change_threshold=change_threshold,
                hot_change_threshold=hot_change_threshold,
            ):
                print(f"    {row}")


def wait_for_initial_snapshots(
    subscribers: list[LatestTactileState],
    timeout_s: float,
) -> dict[str, HandTactileSnapshot]:
    deadline = time.time() + max(0.1, float(timeout_s))
    snapshots: dict[str, HandTactileSnapshot] = {}
    for sub in subscribers:
        remaining = max(0.0, deadline - time.time())
        snapshot = sub.wait(remaining)
        if snapshot is not None:
            snapshots[sub.hand] = snapshot
    return snapshots


def collect_latest_snapshots(subscribers: list[LatestTactileState]) -> dict[str, HandTactileSnapshot]:
    snapshots: dict[str, HandTactileSnapshot] = {}
    for sub in subscribers:
        snapshot = sub.latest()
        if snapshot is not None:
            snapshots[sub.hand] = snapshot
    return snapshots


def print_missing(subscribers: list[LatestTactileState], snapshots: dict[str, HandTactileSnapshot]) -> None:
    for sub in subscribers:
        if sub.hand not in snapshots:
            print(f"{sub.hand}: no state received from {sub.topic}")


def print_snapshots(
    subscribers: list[LatestTactileState],
    snapshots: dict[str, HandTactileSnapshot],
    *,
    scale: float,
    raw: bool,
    color: bool,
    baselines: dict[str, HandTactileSnapshot] | None,
    change_threshold: float,
    hot_change_threshold: float,
) -> None:
    for idx, sub in enumerate(subscribers):
        if idx:
            print()
        snapshot = snapshots.get(sub.hand)
        if snapshot is None:
            print(f"{sub.hand}: no state received from {sub.topic}")
        else:
            print_snapshot(
                snapshot,
                scale=scale,
                raw=raw,
                color=color,
                baseline=None if baselines is None else baselines.get(sub.hand),
                change_threshold=change_threshold,
                hot_change_threshold=hot_change_threshold,
            )


def validate_result(
    subscribers: list[LatestTactileState],
    snapshots: dict[str, HandTactileSnapshot],
    require_valid: bool,
) -> int:
    missing = [sub.hand for sub in subscribers if sub.hand not in snapshots]
    empty = [
        hand for hand, snapshot in snapshots.items()
        if snapshot.tactile_value_count == 0
    ]
    no_valid = [
        hand for hand, snapshot in snapshots.items()
        if snapshot.valid_count == 0
    ]

    if missing:
        print("Missing hand state for: " + ", ".join(missing), file=sys.stderr)
        return 1
    if empty:
        print("No tactile sensor values in state for: " + ", ".join(empty), file=sys.stderr)
        return 2
    if require_valid and no_valid:
        print(
            "No tactile values reached the documented valid threshold for: "
            + ", ".join(no_valid),
            file=sys.stderr,
        )
        return 3
    return 0


def selected_hands(hand_arg: str) -> tuple[str, ...]:
    return ("left", "right") if hand_arg == "both" else (hand_arg,)


def main() -> int:
    args = parse_args()
    iface = str(args.iface) if args.iface else default_dds_iface("eth0")
    ensure_channel_factory_initialized(int(args.domain_id), iface)

    subscribers = [
        LatestTactileState(hand, HAND_STATE_TOPIC_BY_SIDE[hand], args.queue_len)
        for hand in selected_hands(args.hand)
    ]
    print(
        "Waiting for Dex3 tactile state on "
        + ", ".join(sub.topic for sub in subscribers)
        + f" (domain={args.domain_id}, iface={iface})"
    )

    snapshots = wait_for_initial_snapshots(subscribers, float(args.timeout_s))

    if args.sample_s > 0.0 and snapshots:
        time.sleep(float(args.sample_s))
        snapshots = collect_latest_snapshots(subscribers)

    if args.json:
        print(json.dumps(
            {
                hand: snapshot_to_dict(snapshot, float(args.scale))
                for hand, snapshot in snapshots.items()
            },
            indent=2,
            sort_keys=True,
        ))
        return validate_result(subscribers, snapshots, bool(args.require_valid))

    color = not bool(args.no_color)
    baselines = dict(snapshots)
    if args.once:
        print_snapshots(
            subscribers,
            snapshots,
            scale=float(args.scale),
            raw=bool(args.raw),
            color=color,
            baselines=None,
            change_threshold=float(args.change_threshold),
            hot_change_threshold=float(args.hot_change_threshold),
        )
        return validate_result(subscribers, snapshots, bool(args.require_valid))

    refresh_s = 1.0 / max(0.2, float(args.refresh_hz))
    try:
        while True:
            snapshots = collect_latest_snapshots(subscribers)
            if not args.no_clear:
                print(ANSI_CLEAR, end="")
            print(
                "Dex3 tactile live monitor "
                f"(domain={args.domain_id}, iface={iface}, refresh={1.0 / refresh_s:.1f} Hz)"
            )
            print(
                "Press Ctrl+C to stop. Colors show change from the first received baseline; "
                "restart the script to reset the baseline.\n"
            )
            print_snapshots(
                subscribers,
                snapshots,
                scale=float(args.scale),
                raw=bool(args.raw),
                color=color,
                baselines=baselines,
                change_threshold=float(args.change_threshold),
                hot_change_threshold=float(args.hot_change_threshold),
            )
            sys.stdout.flush()
            time.sleep(refresh_s)
    except KeyboardInterrupt:
        if not args.no_clear:
            print()
        print("Stopped by Ctrl+C.")
        return validate_result(subscribers, collect_latest_snapshots(subscribers), bool(args.require_valid))


if __name__ == "__main__":
    raise SystemExit(main())
