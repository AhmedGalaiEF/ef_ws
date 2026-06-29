from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from sdk_client import HAND_JOINT_NAMES, Robot
from sdk_hand import clamp_hand_targets, hand_calibration_path


POSES = ("open", "closed")
HANDS = ("left", "right")


def open_close_controller_side(hand: str) -> str:
    side = str(hand).strip().lower()
    if side == "right":
        return "left"
    if side == "left":
        return "right"
    return side


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Release Dex3 fingers, let you manually place them, then save open/closed "
            "targets used by Robot.hand_open() and Robot.hand_close()."
        )
    )
    parser.add_argument("--hand", choices=("left", "right", "both"), default="both")
    parser.add_argument("--pose", choices=("open", "closed", "both"), default="both")
    parser.add_argument("--iface", default="eth0")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--rate-hz", type=float, default=50.0)
    parser.add_argument("--settle-s", type=float, default=0.3)
    parser.add_argument(
        "--output",
        type=Path,
        default=hand_calibration_path(),
        help="Calibration JSON path. Defaults to sdk_hand.py's calibration path.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Do not ask before overwriting existing pose entries.",
    )
    return parser.parse_args()


def selected(value: str, choices: tuple[str, ...]) -> list[str]:
    return list(choices) if value == "both" else [value]


def load_calibration(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "version": 1,
            "joint_order": list(HAND_JOINT_NAMES),
            "hands": {},
        }
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Calibration file must contain a JSON object: {path}")
    data.setdefault("version", 1)
    data["joint_order"] = list(HAND_JOINT_NAMES)
    hands = data.setdefault("hands", {})
    if not isinstance(hands, dict):
        raise ValueError(f"Calibration file has invalid 'hands' object: {path}")
    return data


def save_calibration(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def wait_for_positions(robot: Robot, hand: str, timeout_s: float = 3.0) -> list[float]:
    deadline = time.time() + timeout_s
    last_snapshot: dict[str, Any] | None = None
    while time.time() < deadline:
        snapshot = robot.get_hand_state_snapshot(hand)
        if snapshot is not None:
            last_snapshot = snapshot
            positions_by_name = snapshot.get("positions", {})
            positions = [positions_by_name.get(name) for name in HAND_JOINT_NAMES]
            if all(value is not None for value in positions):
                return clamp_hand_targets(hand, [float(value) for value in positions])
        time.sleep(0.05)
    source = None if last_snapshot is None else last_snapshot.get("source")
    raise TimeoutError(f"Timed out waiting for {hand} hand state from {source or 'DDS'}.")


def confirm_overwrite(data: dict[str, Any], hand: str, pose: str, assume_yes: bool) -> None:
    if assume_yes:
        return
    hand_data = data.get("hands", {}).get(hand, {})
    if pose not in hand_data:
        return
    answer = input(f"{hand} {pose} already exists. Overwrite it? [y/N] ").strip().lower()
    if answer not in {"y", "yes"}:
        raise RuntimeError(f"Skipped {hand} {pose}; existing calibration was kept.")


def capture_pose(
    robot: Robot,
    *,
    public_hand: str,
    controller_hand: str,
    pose: str,
    rate_hz: float,
    settle_s: float,
) -> list[float]:
    print(
        f"\nReleasing {public_hand} hand for {pose} calibration "
        f"(Dex3 controller side: {controller_hand})."
    )
    robot.release_fingers(hand=controller_hand, rate_hz=rate_hz, persistent=True)
    input(f"Move the {public_hand} fingers to the {pose} position, then press Enter to save. ")
    time.sleep(max(0.0, settle_s))
    targets = wait_for_positions(robot, controller_hand)
    robot.stop_release_fingers(controller_hand)
    print(f"Captured {public_hand} {pose}: {[round(value, 6) for value in targets]}")
    return targets


def main() -> int:
    args = parse_args()
    path = args.output.expanduser()
    data = load_calibration(path)
    robot = Robot(iface=args.iface, domain_id=args.domain_id)
    public_hands = selected(args.hand, HANDS)
    poses = selected(args.pose, POSES)

    try:
        for pose in poses:
            for public_hand in public_hands:
                controller_hand = open_close_controller_side(public_hand)
                confirm_overwrite(data, controller_hand, pose, args.yes)
                targets = capture_pose(
                    robot,
                    public_hand=public_hand,
                    controller_hand=controller_hand,
                    pose=pose,
                    rate_hz=args.rate_hz,
                    settle_s=args.settle_s,
                )
                hand_data = data.setdefault("hands", {}).setdefault(controller_hand, {})
                hand_data[pose] = targets
                hand_data["public_label"] = public_hand
                data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                save_calibration(path, data)
                print(f"Saved {public_hand} {pose} as controller side {controller_hand} to {path}")
    finally:
        if args.hand == "both":
            robot.stop_release_fingers("both")
        else:
            robot.stop_release_fingers(open_close_controller_side(args.hand))

    print("\nDone. New Robot.hand_open() and Robot.hand_close() calls will use this calibration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
