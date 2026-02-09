#!/usr/bin/env python3
"""Play huddle.wav between two arm motions (extend → audio → lift → lower)."""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from typing import Dict, List, Tuple

try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
    from unitree_sdk2py.utils.crc import CRC
except ImportError as exc:
    raise SystemExit(
        "unitree_sdk2py is not installed. Install it with:\n"
        "  pip install -e <path-to-unitree_sdk2_python>"
    ) from exc


def _find_player() -> list[str] | None:
    for cmd in ("aplay", "paplay", "ffplay"):
        if subprocess.call(["/usr/bin/env", "which", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
            return [cmd]
    return None


def _load_audio_client():
    try:
        from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient  # type: ignore

        return AudioClient
    except Exception as exc:
        raise SystemExit(
            "unitree_sdk2py AudioClient is not available. Install unitree_sdk2_python and ensure AudioClient exists."
        ) from exc


def _parse_level(value: str) -> int:
    try:
        level = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("level must be an integer 0-100") from exc
    if not 0 <= level <= 100:
        raise argparse.ArgumentTypeError("level must be in range 0-100")
    return level


def _set_volume(level: int) -> None:
    AudioClient = _load_audio_client()
    client = AudioClient()
    client.SetTimeout(3.0)
    client.Init()
    code = client.SetVolume(level)
    if code != 0:
        raise SystemExit(f"SetVolume failed: code={code}")


def _set_brightness(level: int) -> None:
    AudioClient = _load_audio_client()
    client = AudioClient()
    client.SetTimeout(3.0)
    client.Init()
    val = max(0, min(255, int(level * 255 / 100)))
    code = client.LedControl(val, val, val)
    if code != 0:
        raise SystemExit(f"LedControl failed: code={code}")


def _play_wav(wav_path: str) -> None:
    if not os.path.exists(wav_path):
        print(f"Missing wav file: {wav_path}")
        raise SystemExit(1)

    player = _find_player()
    if not player:
        print("No audio player found. Install aplay/paplay/ffplay.")
        raise SystemExit(2)

    cmd = player + [wav_path]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Audio player failed: {exc}")
        raise SystemExit(exc.returncode)


class ArmSdkController:
    """Simple arm SDK pose sequencer using rt/arm_sdk (LowCmd)."""

    _WAIST_YAW_IDX = 12
    _NOT_USED_IDX = 29  # enable arm sdk when q = 1

    def __init__(self, iface: str, arm: str, cmd_hz: float, kp: float, kd: float) -> None:
        self._arm = arm
        self._cmd_hz = max(1.0, cmd_hz)
        self._kp = kp
        self._kd = kd
        self._cmd_q: Dict[int, float] = {}
        self._crc = CRC()

        ChannelFactoryInitialize(0, iface)

        self._pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self._pub.Init()

        self._cmd = unitree_hg_msg_dds__LowCmd_()
        self._cmd.motor_cmd[self._NOT_USED_IDX].q = 1

        if arm == "left":
            self._joint_idx = [15, 16, 17, 18, 19, 20, 21]
        else:
            self._joint_idx = [22, 23, 24, 25, 26, 27, 28]

        for idx in self._joint_idx:
            self._cmd_q[idx] = 0.0
        self._cmd_q[self._WAIST_YAW_IDX] = 0.0

    def _apply_targets(self, targets: Dict[int, float]) -> None:
        for j_idx, q_val in targets.items():
            mc = self._cmd.motor_cmd[j_idx]
            mc.q = float(q_val)
            mc.kp = float(self._kp)
            mc.kd = float(self._kd)
            mc.tau = 0.0
        self._cmd.crc = self._crc.Crc(self._cmd)
        self._pub.Write(self._cmd)

    def ramp_to_pose(
        self,
        pose: List[Tuple[int, float]],
        duration: float,
        easing: str = "smooth",
    ) -> None:
        """Ramp the arm joints to the target pose over duration seconds."""
        target = {j: q for j, q in pose}
        start = {j: self._cmd_q.get(j, 0.0) for j in target}

        steps = max(1, int(self._cmd_hz * max(0.0, duration)))
        dt = 1.0 / self._cmd_hz

        for step in range(1, steps + 1):
            alpha = step / steps
            if easing == "smooth":
                alpha = 0.5 - 0.5 * math.cos(math.pi * alpha)
            cur = {j: start[j] + (target[j] - start[j]) * alpha for j in target}
            self._apply_targets(cur)
            time.sleep(dt)

        self._cmd_q.update(target)

    def hold_pose(self, pose: List[Tuple[int, float]], hold_s: float) -> None:
        if hold_s <= 0:
            return
        target = {j: q for j, q in pose}
        steps = max(1, int(self._cmd_hz * hold_s))
        dt = 1.0 / self._cmd_hz
        for _ in range(steps):
            self._apply_targets(target)
            time.sleep(dt)


def _default_poses(arm: str) -> Dict[str, List[Tuple[int, float]]]:
    # These are conservative poses based on existing scripts (run_geoff_gui.py).
    # Tune as needed for your robot.
    if arm == "left":
        side = [
            (12, 0.0),
            (15, +0.211),  # shoulder pitch
            (16, +0.181),  # shoulder roll
            (17, -0.284),  # shoulder yaw
            (18, +0.672),  # elbow
            (19, -0.379),  # wrist roll
            (20, -0.852),  # wrist pitch
            (21, -0.019),  # wrist yaw
        ]
        extend = [
            (12, 0.0),
            (15, +0.380),
            (16, +0.060),
            (17, -0.240),
            (18, +0.420),
            (19, -0.500),
            (20, -1.000),
            (21, -0.050),
        ]
        lift = [
            (12, 0.0),
            (15, -0.300),
            (16, +0.080),
            (17, -0.200),
            (18, +0.520),
            (19, -0.420),
            (20, -0.650),
            (21, -0.050),
        ]
    else:
        side = [
            (12, 0.0),
            (22, -0.023),  # shoulder pitch
            (23, -0.225),  # shoulder roll
            (24, +0.502),  # shoulder yaw
            (25, +1.317),  # elbow
            (26, +0.185),  # wrist pitch
            (27, +0.125),  # wrist roll
            (28, -0.182),  # wrist yaw
        ]
        extend = [
            (12, 0.0),
            (22, +0.200),
            (23, -0.300),
            (24, +0.280),
            (25, +0.520),
            (26, +0.050),
            (27, -1.050),
            (28, -0.176),
        ]
        lift = [
            (12, 0.0),
            (22, -0.550),
            (23, -0.320),
            (24, +0.280),
            (25, +0.691),
            (26, +0.160),
            (27, -0.650),
            (28, -0.176),
        ]

    return {"side": side, "extend": extend, "lift": lift}


def main() -> None:
    parser = argparse.ArgumentParser(description="Huddle sequence: extend → audio → lift → lower.")
    parser.add_argument("--iface", default="eth0", help="network interface for DDS")
    parser.add_argument("--arm", choices=["left", "right"], default="right", help="which arm to move")
    parser.add_argument("--file", default="huddle.wav", help="path to wav file")
    parser.add_argument("--volume", type=_parse_level, default=None, help="set robot speaker volume (0-10)")
    parser.add_argument("--brightness", type=_parse_level, default=None, help="set headlight brightness (0-10)")
    parser.add_argument("--cmd-hz", type=float, default=50.0, help="command rate for arm SDK")
    parser.add_argument("--kp", type=float, default=40.0, help="arm joint kp")
    parser.add_argument("--kd", type=float, default=1.0, help="arm joint kd")
    parser.add_argument("--extend-sec", type=float, default=2.5, help="seconds to extend hand")
    parser.add_argument("--hold-extend", type=float, default=0.3, help="seconds to hold extend pose")
    parser.add_argument("--lift-sec", type=float, default=1.5, help="seconds to lift hand")
    parser.add_argument("--hold-lift", type=float, default=0.3, help="seconds to hold lift pose")
    parser.add_argument("--lower-sec", type=float, default=2.5, help="seconds to lower back to side")
    parser.add_argument(
        "--easing",
        choices=["linear", "smooth"],
        default="smooth",
        help="easing profile for arm ramps",
    )
    args = parser.parse_args()

    wav_path = args.file
    if not os.path.isabs(wav_path):
        wav_path = os.path.join(os.path.dirname(__file__), wav_path)

    poses = _default_poses(args.arm)

    arm = ArmSdkController(args.iface, args.arm, args.cmd_hz, args.kp, args.kd)

    print("Step 1: extend hand in front (palm down).")
    arm.ramp_to_pose(poses["extend"], args.extend_sec, easing=args.easing)
    arm.hold_pose(poses["extend"], args.hold_extend)

    print("Step 2: play audio.")
    if args.brightness is not None:
        _set_brightness(args.brightness)
    if args.volume is not None:
        _set_volume(args.volume)
    _play_wav(wav_path)

    print("Step 3: lift hand (rotate shoulder).")
    arm.ramp_to_pose(poses["lift"], args.lift_sec, easing=args.easing)
    arm.hold_pose(poses["lift"], args.hold_lift)

    print("Step 4: lower arm back to side (slow).")
    arm.ramp_to_pose(poses["side"], args.lower_sec, easing=args.easing)


if __name__ == "__main__":
    main()
