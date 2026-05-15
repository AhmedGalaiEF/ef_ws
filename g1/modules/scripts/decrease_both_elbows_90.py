#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import time

from low_level_commands import BODY_JOINTS, LowLevelJointExample, clamp


LEFT_ELBOW_NAME = "left_arm.elbow"
RIGHT_ELBOW_NAME = "right_arm.elbow"
LEFT_SHOULDER_YAW_NAME = "left_arm.shoulder_yaw"
RIGHT_SHOULDER_YAW_NAME = "right_arm.shoulder_yaw"
DEFAULT_DECREASE_RAD = math.pi / 2.0
DEFAULT_SHOULDER_YAW_INWARD_RAD = math.pi / 6.0
DEFAULT_MAX_DELTA_RAD = 0.1
DEFAULT_RAMP_DURATION_S = 5.0
LEFT_SHOULDER_YAW_INWARD_SIGN = -1.0
RIGHT_SHOULDER_YAW_INWARD_SIGN = 1.0


def joint_by_name(name: str) -> tuple[int, str, int, float, float]:
    for local_idx, spec in enumerate(BODY_JOINTS):
        joint_name, motor_idx, lo, hi = spec
        if joint_name == name:
            return local_idx, joint_name, motor_idx, lo, hi
    raise RuntimeError(f"Joint not found in BODY_JOINTS: {name}")


class DecreaseBothElbows:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.rate_hz = max(1.0, float(args.rate_hz))
        self.dt = 1.0 / self.rate_hz
        self.controller = LowLevelJointExample(args)
        self.elbows = [joint_by_name(LEFT_ELBOW_NAME), joint_by_name(RIGHT_ELBOW_NAME)]
        self.shoulder_yaws = [
            joint_by_name(LEFT_SHOULDER_YAW_NAME),
            joint_by_name(RIGHT_SHOULDER_YAW_NAME),
        ]

    def run(self) -> int:
        decrease_rad = abs(float(self.args.decrease_rad))
        shoulder_yaw_rad = abs(float(self.args.shoulder_yaw_inward_rad))
        max_delta = min(DEFAULT_MAX_DELTA_RAD, abs(float(self.args.max_delta_rad)))
        ramp_duration_s = max(0.1, float(self.args.ramp_duration_s))
        if max_delta <= 0.0:
            raise ValueError("--max-delta-rad must be greater than 0")

        print("This script uses true low-level body control on rt/lowcmd.")
        print(f"Requested elbow decrease: {decrease_rad:.4f} rad ({math.degrees(decrease_rad):.1f} deg)")
        print(
            f"Requested shoulder-yaw inward move: {shoulder_yaw_rad:.4f} rad "
            f"({math.degrees(shoulder_yaw_rad):.1f} deg)"
        )
        print(f"Max command delta per publish step: {max_delta:.4f} rad")
        print(f"Requested ramp duration: {ramp_duration_s:.1f}s")
        if not self.args.yes:
            print("Dry run only. Re-run with --yes to release motion switcher and send commands.")
            for _local_idx, name, motor_idx, lo, hi in self.elbows + self.shoulder_yaws:
                print(f"body {motor_idx:02d} {name} limits=[{lo:+.4f}, {hi:+.4f}]")
            return 0

        self.controller.setup_dds()
        self.controller.enter_dev_mode()
        assert self.controller.body_state is not None

        snapshot = self.controller.body_state.wait(float(self.args.timeout))
        body_targets = list(snapshot.positions)
        mode_machine = int(snapshot.mode_machine)

        starts: dict[int, float] = {}
        targets: dict[int, float] = {}
        max_distance = 0.0
        for local_idx, name, motor_idx, lo, hi in self.elbows:
            start = clamp(body_targets[local_idx], lo, hi)
            target = clamp(start - decrease_rad, lo, hi)
            body_targets[local_idx] = start
            starts[local_idx] = start
            targets[local_idx] = target
            max_distance = max(max_distance, abs(target - start))
            print(f"body {motor_idx:02d} {name}: {start:+.4f} -> {target:+.4f} rad")
        for local_idx, name, motor_idx, lo, hi in self.shoulder_yaws:
            start = clamp(body_targets[local_idx], lo, hi)
            sign = LEFT_SHOULDER_YAW_INWARD_SIGN if name == LEFT_SHOULDER_YAW_NAME else RIGHT_SHOULDER_YAW_INWARD_SIGN
            target = clamp(start + sign * shoulder_yaw_rad, lo, hi)
            body_targets[local_idx] = start
            starts[local_idx] = start
            targets[local_idx] = target
            max_distance = max(max_distance, abs(target - start))
            print(f"body {motor_idx:02d} {name}: {start:+.4f} -> {target:+.4f} rad")

        if max_distance <= 1e-6:
            print("All requested joints are already at their target limits; nothing to move.")
            return 0

        print(f"Holding current pose for {self.args.initial_hold_s:.1f}s")
        self.controller.hold_all(body_targets, {}, mode_machine, float(self.args.initial_hold_s))

        requested_steps = max(1, math.ceil(ramp_duration_s * self.rate_hz))
        min_safe_steps = max(1, math.ceil(max_distance / max_delta))
        steps = max(requested_steps, min_safe_steps)
        ramp_seconds = steps / self.rate_hz
        print(f"Ramping elbows and shoulder yaw over {steps} publish steps at {self.rate_hz:.1f} Hz ({ramp_seconds:.1f}s)")
        try:
            for step_idx in range(1, steps + 1):
                ratio = float(step_idx) / float(steps)
                for local_idx in targets:
                    body_targets[local_idx] = starts[local_idx] + (targets[local_idx] - starts[local_idx]) * ratio
                self.controller.write_body(body_targets, mode_machine)
                time.sleep(self.dt)

            if self.args.hold_forever:
                print("Holding final pose until interrupted.")
                while True:
                    self.controller.write_body(body_targets, mode_machine)
                    time.sleep(self.dt)
            print(f"Holding final pose for {self.args.final_hold_s:.1f}s")
            self.controller.hold_all(body_targets, {}, mode_machine, float(self.args.final_hold_s))
        finally:
            if self.args.zero_gains_on_exit:
                print("zeroing body gains on exit")
                self.controller.write_body(body_targets, mode_machine, kp_scale=0.0)
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decrease both G1 elbow joints by 90 degrees and move both shoulder-yaw joints inward with a slow low-level ramp."
    )
    parser.add_argument("--iface", default="eth0", help="DDS network interface.")
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain ID.")
    parser.add_argument("--timeout", type=float, default=5.0, help="Seconds to wait for SDK RPC/state.")
    parser.add_argument(
        "--decrease-rad",
        type=float,
        default=DEFAULT_DECREASE_RAD,
        help="Positive elbow decrease amount in radians. Default is pi/2.",
    )
    parser.add_argument(
        "--shoulder-yaw-inward-rad",
        type=float,
        default=DEFAULT_SHOULDER_YAW_INWARD_RAD,
        help="Positive inward shoulder-yaw amount in radians. Default is pi/6.",
    )
    parser.add_argument(
        "--max-delta-rad",
        type=float,
        default=DEFAULT_MAX_DELTA_RAD,
        help="Maximum commanded position change per publish step. Values above 0.1 are capped.",
    )
    parser.add_argument("--ramp-duration-s", type=float, default=DEFAULT_RAMP_DURATION_S, help="Requested ramp duration in seconds.")
    parser.add_argument("--initial-hold-s", type=float, default=1.0, help="Hold current pose before moving.")
    parser.add_argument("--final-hold-s", type=float, default=30.0, help="Hold final pose before exit if --no-hold-forever is set.")
    hold_group = parser.add_mutually_exclusive_group()
    hold_group.add_argument("--hold-forever", dest="hold_forever", action="store_true", help="Keep publishing the final pose until interrupted.")
    hold_group.add_argument("--no-hold-forever", dest="hold_forever", action="store_false", help="Hold for --final-hold-s, then exit.")
    parser.set_defaults(hold_forever=True)
    parser.add_argument("--rate-hz", type=float, default=50.0, help="Low-level publish rate.")
    parser.add_argument("--zero-gains-on-exit", action="store_true", help="Send one zero-gain command on exit.")
    parser.add_argument("--yes", action="store_true", help="Actually release motion switcher and send low-level commands.")

    args = parser.parse_args()
    args.hands = "none"
    args.hand_kp = 0.5
    args.hand_kd = 0.1
    args.require_hands = False
    return args


def main() -> int:
    return DecreaseBothElbows(parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(main())
