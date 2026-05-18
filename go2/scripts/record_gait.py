import argparse
import curses
import json
import os
import struct
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_


TOPIC_LOWSTATE = "rt/lowstate"
LEG_NAMES = [
    "FR_0", "FR_1", "FR_2",
    "FL_0", "FL_1", "FL_2",
    "RR_0", "RR_1", "RR_2",
    "RL_0", "RL_1", "RL_2",
]


def age_text(ts: float) -> str:
    if ts <= 0.0:
        return "--"
    return f"{max(0.0, time.time() - ts):4.1f}s"


def decode_buttons(data):
    data1 = int(data[2])
    data2 = int(data[3])
    return {
        "R1": (data1 >> 0) & 1,
        "L1": (data1 >> 1) & 1,
        "Start": (data1 >> 2) & 1,
        "Select": (data1 >> 3) & 1,
        "R2": (data1 >> 4) & 1,
        "L2": (data1 >> 5) & 1,
        "F1": (data1 >> 6) & 1,
        "F3": (data1 >> 7) & 1,
        "A": (data2 >> 0) & 1,
        "B": (data2 >> 1) & 1,
        "X": (data2 >> 2) & 1,
        "Y": (data2 >> 3) & 1,
        "Up": (data2 >> 4) & 1,
        "Right": (data2 >> 5) & 1,
        "Down": (data2 >> 6) & 1,
        "Left": (data2 >> 7) & 1,
    }


def decode_remote(data):
    raw = bytes(data)
    return {
        "lx": struct.unpack("<f", raw[4:8])[0],
        "rx": struct.unpack("<f", raw[8:12])[0],
        "ry": struct.unpack("<f", raw[12:16])[0],
        "ly": struct.unpack("<f", raw[20:24])[0],
        "buttons": decode_buttons(raw),
        "raw_hex": raw.hex(),
    }


def build_sample(msg: LowState_):
    remote = decode_remote(msg.wireless_remote)
    joints = []
    for idx in range(12):
        motor = msg.motor_state[idx]
        joints.append(
            {
                "name": LEG_NAMES[idx],
                "index": idx,
                "q": float(motor.q),
                "dq": float(motor.dq),
                "ddq": float(motor.ddq),
                "tau_est": float(motor.tau_est),
                "temperature": int(motor.temperature),
                "mode": int(motor.mode),
            }
        )

    return {
        "wall_time": time.time(),
        "tick": int(msg.tick),
        "power_v": float(msg.power_v),
        "power_a": float(msg.power_a),
        "imu_rpy": [float(v) for v in msg.imu_state.rpy],
        "imu_gyro": [float(v) for v in msg.imu_state.gyroscope],
        "imu_acc": [float(v) for v in msg.imu_state.accelerometer],
        "foot_force": [int(v) for v in msg.foot_force],
        "foot_force_est": [int(v) for v in msg.foot_force_est],
        "remote": remote,
        "joints": joints,
    }


@dataclass
class RecorderState:
    last_sample: Optional[dict] = None
    last_sample_time: float = 0.0
    recorded_samples: int = 0
    dropped_writes: int = 0
    recording: bool = False
    output_path: str = ""


class GaitRecorder:
    def __init__(self, output_path: str):
        self.lock = threading.Lock()
        self.state = RecorderState(output_path=output_path)
        self.writer = None
        self.subscriber = None

    def init(self):
        self.subscriber = ChannelSubscriber(TOPIC_LOWSTATE, LowState_)
        self.subscriber.Init(self._low_state_handler, 20)

    def toggle_recording(self):
        with self.lock:
            if self.state.recording:
                if self.writer is not None:
                    self.writer.flush()
                    self.writer.close()
                    self.writer = None
                self.state.recording = False
                return

            os.makedirs(os.path.dirname(self.state.output_path) or ".", exist_ok=True)
            self.writer = open(self.state.output_path, "a", encoding="utf-8")
            self.state.recording = True

    def close(self):
        with self.lock:
            if self.writer is not None:
                self.writer.flush()
                self.writer.close()
                self.writer = None
            self.state.recording = False

    def snapshot(self):
        with self.lock:
            return {
                "last_sample": self.state.last_sample,
                "last_sample_time": self.state.last_sample_time,
                "recording": self.state.recording,
                "recorded_samples": self.state.recorded_samples,
                "dropped_writes": self.state.dropped_writes,
                "output_path": self.state.output_path,
            }

    def _low_state_handler(self, msg: LowState_):
        sample = build_sample(msg)
        line = json.dumps(sample, separators=(",", ":"))
        with self.lock:
            self.state.last_sample = sample
            self.state.last_sample_time = time.time()
            if not self.state.recording or self.writer is None:
                return
            try:
                self.writer.write(line + "\n")
                self.state.recorded_samples += 1
                if self.state.recorded_samples % 50 == 0:
                    self.writer.flush()
            except Exception:
                self.state.dropped_writes += 1


def fmt_vec(values):
    return f"{values[0]: .2f} {values[1]: .2f} {values[2]: .2f}"


def draw(stdscr, recorder: GaitRecorder):
    stdscr.erase()
    h, w = stdscr.getmaxyx()
    snap = recorder.snapshot()
    sample = snap["last_sample"]

    lines = [
        "Record Gait",
        "r: start/stop recording  q: quit",
        f"Topic: {TOPIC_LOWSTATE}",
        f"Recording: {'ON' if snap['recording'] else 'OFF'}",
        f"Output: {snap['output_path']}",
        f"Samples written: {snap['recorded_samples']}  dropped writes: {snap['dropped_writes']}",
        f"Last sample age: {age_text(snap['last_sample_time'])}",
        "",
    ]

    if sample is None:
        lines.append("Waiting for rt/lowstate ...")
    else:
        remote = sample["remote"]
        buttons = remote["buttons"]
        lines.extend(
            [
                f"Tick: {sample['tick']}  Power V/A: {sample['power_v']:.2f} / {sample['power_a']:.2f}",
                f"IMU rpy:  {fmt_vec(sample['imu_rpy'])}",
                f"IMU gyro: {fmt_vec(sample['imu_gyro'])}",
                f"IMU acc:  {fmt_vec(sample['imu_acc'])}",
                f"Foot force: {' '.join(str(v) for v in sample['foot_force'])}",
                "",
                f"Joystick lx/ly: {remote['lx']:+.2f} {remote['ly']:+.2f}",
                f"Joystick rx/ry: {remote['rx']:+.2f} {remote['ry']:+.2f}",
                "Buttons: " + " ".join(name for name, value in buttons.items() if value) if any(buttons.values()) else "Buttons: none",
                "",
                "Joints: name q dq tau temp",
            ]
        )
        for joint in sample["joints"]:
            lines.append(
                f"{joint['name']:>4} {joint['q']: .3f} {joint['dq']: .3f} {joint['tau_est']: .3f} {joint['temperature']:>3}"
            )

    if h < 24 or w < 100:
        lines.append("")
        lines.append("Terminal is small; widen it for the full monitor.")

    for idx, line in enumerate(lines[: max(0, h - 1)]):
        stdscr.addnstr(idx, 0, line, max(0, w - 1))
    stdscr.refresh()


def tui_main(stdscr, recorder: GaitRecorder):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(100)
    while True:
        draw(stdscr, recorder)
        key = stdscr.getch()
        if key in (ord("r"), ord("R")):
            recorder.toggle_recording()
        elif key in (ord("q"), ord("Q")):
            break


def parse_args():
    parser = argparse.ArgumentParser(
        description="Monitor Go2 low-level joint state and record joint/remote-controller data."
    )
    parser.add_argument("--iface", required=True, help="Robot network interface")
    parser.add_argument(
        "--output",
        default=f"record_gait_{time.strftime('%Y%m%d_%H%M%S')}.jsonl",
        help="JSONL output path",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ChannelFactoryInitialize(0, args.iface)

    recorder = GaitRecorder(args.output)
    recorder.init()
    try:
        curses.wrapper(tui_main, recorder)
    finally:
        recorder.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
