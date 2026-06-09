#!/usr/bin/env python3
from __future__ import annotations
from secure_boot import force_normal_gait
from sdk_boot import (
    BALANCED_STAND_FSM_IDS,
    create_loco_client,
    fsm_mode,
    is_balanced_stand_state,
    read_fsm_state,
)

import argparse
import os
import sys
import time


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODULES_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if MODULES_DIR not in sys.path:
    sys.path.insert(0, MODULES_DIR)


BALANCED_STAND_FSM_LABEL = "/".join(str(fsm_id) for fsm_id in sorted(BALANCED_STAND_FSM_IDS))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hanger boot, shake hands, then walk forward briefly."
    )
    parser.add_argument("--iface", default="eth0", help="Robot network interface.")
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain id.")
    parser.add_argument(
        "--forward-speed",
        type=float,
        default=0.3,
        help="Forward walking speed in m/s.",
    )
    parser.add_argument(
        "--forward-seconds",
        type=float,
        default=2.0,
        help="How long to walk forward after shaking hands.",
    )
    parser.add_argument(
        "--extend-delay",
        type=float,
        default=0.75,
        help="Delay after the first ShakeHand call before walking forward.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip operator confirmation prompts.",
    )
    return parser.parse_args()


def show_fsm(loco: object, tag: str) -> tuple[int | None, int | None]:
    cur_id, cur_mode = read_fsm_state(loco, retries=2, retry_delay=0.05)
    print(f"{tag:<12s} -> FSM {cur_id}   mode {cur_mode}")
    return cur_id, cur_mode


def is_balanced_stand(loco: object) -> bool:
    cur_id, cur_mode = read_fsm_state(loco)
    return is_balanced_stand_state(cur_id, cur_mode)


def wait_for_balanced_stand(
    loco: object,
    timeout_s: float = 8.0,
    poll_s: float = 0.2,
) -> bool:
    deadline = time.time() + max(0.0, float(timeout_s))
    while time.time() < deadline:
        if is_balanced_stand(loco):
            return True
        time.sleep(max(0.01, float(poll_s)))
    return is_balanced_stand(loco)


def stop_loco(loco: object | None) -> None:
    if loco is None:
        return
    if hasattr(loco, "StopMove"):
        loco.StopMove()
        return
    if hasattr(loco, "Move"):
        loco.Move(0.0, 0.0, 0.0, continous_move=False)


def run_secure_boot(args: argparse.Namespace) -> object:
    loco = create_loco_client(domain_id=args.domain_id, iface=args.iface)

    cur_id, cur_mode = show_fsm(loco, "initial")
    if is_balanced_stand_state(cur_id, cur_mode):
        print("Robot is already in balanced stand; skipping hanger boot.")
        force_normal_gait(loco)
        return loco

    loco.Damp()
    show_fsm(loco, "damp")

    loco.SetFsmId(4)
    show_fsm(loco, "stand_up")

    height = 0.0
    while True:
        height = 0.0
        while height < 0.5:
            height += 0.02
            loco.SetStandHeight(height)
            show_fsm(loco, f"height {height:.2f} m")
            if fsm_mode(loco) == 0 and height > 0.2:
                break

        if fsm_mode(loco) == 0:
            break

        print(
            f"Feet still unloaded (mode {fsm_mode(loco)}) after reaching {height:.2f} m. "
            "Adjust the hanger height, then press Enter to retry."
        )
        try:
            loco.SetStandHeight(0.0)
            show_fsm(loco, "reset")
        except Exception:
            pass
        input()

    if not args.yes:
        input("Robot appears loaded. Press Enter to command balanced stand...")

    loco.BalanceStand(0)
    show_fsm(loco, "balance")
    loco.SetStandHeight(height)
    show_fsm(loco, "height_ok")
    loco.Start()
    show_fsm(loco, "start")
    force_normal_gait(loco)
    show_fsm(loco, "normal_gait")

    if not wait_for_balanced_stand(loco):
        cur_id, cur_mode = read_fsm_state(loco)
        raise RuntimeError(
            "Secure boot did not reach balanced stand "
            f"(expected FSM {BALANCED_STAND_FSM_LABEL}, mode 0; got FSM {cur_id}, mode {cur_mode})."
        )

    show_fsm(loco, "balanced")
    return loco


def main() -> int:
    args = parse_args()

    print("WARNING: Please ensure there are no obstacles around the robot.")
    if not args.yes:
        input("Press Enter to run hanger boot, shake hands, then walk forward...")

    loco = None
    try:
        loco = run_secure_boot(args)
    except Exception as exc:
        print(f"Failed to boot/connect to robot: {exc}")
        return 1

    try:
        if not hasattr(loco, "ShakeHand"):
            raise AttributeError("Current locomotion client does not support ShakeHand().")
        if not hasattr(loco, "Move"):
            raise AttributeError("Current locomotion client does not support Move().")
        if not is_balanced_stand(loco):
            cur_id, cur_mode = read_fsm_state(loco)
            raise RuntimeError(
                "Refusing to run handshake/walk because robot is not in balanced stand "
                f"(FSM {cur_id}, mode {cur_mode})."
            )

        print("Running loco client id 11: extend hand.")
        loco.ShakeHand()
        time.sleep(max(0.0, float(args.extend_delay)))

        print(
            f"Walking forward at {args.forward_speed:.2f} m/s "
            f"for {args.forward_seconds:.2f} seconds while hand is extended."
        )
        loco.Move(float(args.forward_speed), 0.0, 0.0, continous_move=True)
        time.sleep(max(0.0, float(args.forward_seconds)))
        stop_loco(loco)
        time.sleep(0.25)

        print("Retracting hand with second loco client id 11 call.")
        loco.ShakeHand()
        time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nInterrupted. Sending stop command.")
        stop_loco(loco)
        return 1
    except Exception as exc:
        print(f"Sequence failed: {exc}")
        stop_loco(loco)
        return 1
    finally:
        try:
            stop_loco(loco)
        except Exception:
            pass

    print("Sequence complete. Stop command sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
