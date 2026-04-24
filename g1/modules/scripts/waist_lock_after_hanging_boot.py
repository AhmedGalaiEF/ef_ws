#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from sdk_boot import hanger_boot_sequence

try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
    from unitree_sdk2py.utils.crc import CRC
except ImportError as exc:
    raise SystemExit(
        "unitree_sdk2py is not installed. Install it with:\n"
        "  pip install -e <path-to-unitree_sdk2_python>"
    ) from exc


BODY_COMMAND_TOPIC = "rt/lowcmd"
LOWSTATE_TOPIC = "rt/lowstate"
NOT_USED_IDX = 29
WAIST_ROLL_IDX = 13
WAIST_PITCH_IDX = 14
WAIST_LIMIT_RAD = 0.52


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the hanging boot sequence, then lock waist roll and pitch at fixed targets."
    )
    parser.add_argument("--iface", default="enp1s0", help="DDS network interface.")
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain id.")
    parser.add_argument("--rate-hz", type=float, default=100.0, help="Publish rate for the waist hold command.")
    parser.add_argument("--kp", type=float, default=30.0, help="Waist proportional gain.")
    parser.add_argument("--kd", type=float, default=1.5, help="Waist derivative gain.")
    parser.add_argument("--tau", type=float, default=0.0, help="Waist feed-forward torque.")
    parser.add_argument(
        "--state-timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for rt/lowstate after boot.",
    )
    parser.add_argument(
        "--roll-target",
        type=float,
        default=None,
        help="Explicit waist roll target in radians. Defaults to the current measured angle.",
    )
    parser.add_argument(
        "--pitch-target",
        type=float,
        default=None,
        help="Explicit waist pitch target in radians. Defaults to the current measured angle.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Optional hold duration in seconds. Use 0 to hold until Ctrl-C.",
    )
    return parser.parse_args()


def resolve_lowstate_type() -> type | None:
    for module_path in (
        "unitree_sdk2py.idl.unitree_hg.msg.dds_",
        "unitree_sdk2py.idl.unitree_go.msg.dds_",
    ):
        try:
            module = __import__(module_path, fromlist=["LowState_"])
        except Exception:
            continue
        if hasattr(module, "LowState_"):
            return getattr(module, "LowState_")
    return None


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


@dataclass
class WaistSnapshot:
    roll: float
    pitch: float
    mode_machine: int
    timestamp: float


class WaistStateSubscriber:
    def __init__(self) -> None:
        self._snapshot: WaistSnapshot | None = None
        lowstate_type = resolve_lowstate_type()
        if lowstate_type is None:
            raise RuntimeError("LowState_ type not found in unitree_sdk2py.")
        self._sub = ChannelSubscriber(LOWSTATE_TOPIC, lowstate_type)
        self._sub.Init(self._callback, 50)

    def _callback(self, msg: Any) -> None:
        try:
            roll = float(msg.motor_state[WAIST_ROLL_IDX].q)
            pitch = float(msg.motor_state[WAIST_PITCH_IDX].q)
            mode_machine = int(getattr(msg, "mode_machine", 0))
        except Exception:
            return
        self._snapshot = WaistSnapshot(
            roll=roll,
            pitch=pitch,
            mode_machine=mode_machine,
            timestamp=time.time(),
        )

    def wait_for_snapshot(self, timeout: float) -> WaistSnapshot:
        deadline = time.time() + max(0.1, float(timeout))
        while time.time() < deadline:
            snapshot = self._snapshot
            if snapshot is not None:
                return snapshot
            time.sleep(0.02)
        raise TimeoutError(f"No waist state received on {LOWSTATE_TOPIC} within {timeout:.1f}s.")

    def snapshot(self) -> WaistSnapshot | None:
        return self._snapshot


class WaistLockController:
    def __init__(self, *, iface: str, domain_id: int) -> None:
        self._crc = CRC()
        ChannelFactoryInitialize(int(domain_id), str(iface))
        self._pub = ChannelPublisher(BODY_COMMAND_TOPIC, LowCmd_)
        self._pub.Init()
        self._cmd = unitree_hg_msg_dds__LowCmd_()
        self._cmd.mode_pr = 0
        self._cmd.mode_machine = 0
        self._cmd.motor_cmd[NOT_USED_IDX].q = 1.0
        for joint in (WAIST_ROLL_IDX, WAIST_PITCH_IDX):
            self._cmd.motor_cmd[joint].mode = 1

    def write_targets_once(
        self,
        *,
        roll: float,
        pitch: float,
        kp: float,
        kd: float,
        tau: float,
        mode_machine: int,
    ) -> None:
        self._cmd.mode_machine = int(mode_machine)
        for joint, target in ((WAIST_ROLL_IDX, roll), (WAIST_PITCH_IDX, pitch)):
            mc = self._cmd.motor_cmd[joint]
            mc.mode = 1
            mc.q = float(target)
            mc.dq = 0.0
            mc.kp = float(kp)
            mc.kd = float(kd)
            mc.tau = float(tau)
        self._cmd.crc = self._crc.Crc(self._cmd)
        self._pub.Write(self._cmd)

    def release_once(self, *, roll: float, pitch: float, mode_machine: int) -> None:
        self.write_targets_once(
            roll=roll,
            pitch=pitch,
            kp=0.0,
            kd=0.0,
            tau=0.0,
            mode_machine=mode_machine,
        )


def main() -> int:
    args = parse_args()
    rate_hz = max(1.0, float(args.rate_hz))
    period_s = 1.0 / rate_hz

    ChannelFactoryInitialize(int(args.domain_id), str(args.iface))
    state_sub = WaistStateSubscriber()

    print("Running hanging boot sequence. Ensure the robot is correctly supported on the hanger.")
    hanger_boot_sequence(iface=args.iface, domain_id=args.domain_id)

    snapshot = state_sub.wait_for_snapshot(timeout=float(args.state_timeout))
    roll_target = snapshot.roll if args.roll_target is None else clamp(args.roll_target, -WAIST_LIMIT_RAD, WAIST_LIMIT_RAD)
    pitch_target = (
        snapshot.pitch if args.pitch_target is None else clamp(args.pitch_target, -WAIST_LIMIT_RAD, WAIST_LIMIT_RAD)
    )

    controller = WaistLockController(iface=args.iface, domain_id=args.domain_id)

    print(
        "Locking waist joints:",
        f"roll={roll_target:.3f} rad",
        f"pitch={pitch_target:.3f} rad",
        f"mode_machine={snapshot.mode_machine}",
        f"rate={rate_hz:.1f} Hz",
    )
    if float(args.duration) > 0.0:
        print(f"Will stop after {float(args.duration):.2f} s.")
    else:
        print("Holding until Ctrl-C.")

    start = time.monotonic()
    try:
        while True:
            latest = state_sub.snapshot()
            mode_machine = snapshot.mode_machine if latest is None else latest.mode_machine
            controller.write_targets_once(
                roll=roll_target,
                pitch=pitch_target,
                kp=float(args.kp),
                kd=float(args.kd),
                tau=float(args.tau),
                mode_machine=mode_machine,
            )
            if float(args.duration) > 0.0 and (time.monotonic() - start) >= float(args.duration):
                break
            time.sleep(period_s)
    except KeyboardInterrupt:
        print("\nInterrupted, releasing waist lock gains.")
    finally:
        latest = state_sub.snapshot()
        mode_machine = snapshot.mode_machine if latest is None else latest.mode_machine
        controller.release_once(
            roll=roll_target,
            pitch=pitch_target,
            mode_machine=mode_machine,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
