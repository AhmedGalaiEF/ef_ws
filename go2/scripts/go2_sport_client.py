from __future__ import annotations

import argparse
import json
import math
import time
from typing import Any
from dataclasses import dataclass


@dataclass
class TestOption:
    name: str
    id: int

option_list = [
    TestOption(name="damp", id=0),         
    TestOption(name="stand_up", id=1),     
    TestOption(name="stand_down", id=2),   
    TestOption(name="move forward", id=3),         
    TestOption(name="move lateral", id=4),    
    TestOption(name="move rotate", id=5),  
    TestOption(name="stop_move", id=6),  
    TestOption(name="hand stand", id=7),
    TestOption(name="balanced stand", id=9),     
    TestOption(name="recovery", id=10),       
    TestOption(name="left flip", id=11),      
    TestOption(name="back flip", id=12),
    TestOption(name="free walk", id=13),  
    TestOption(name="free bound", id=14), 
    TestOption(name="free avoid", id=15),  
    TestOption(name="walk upright", id=17),
    TestOption(name="cross step", id=18),
    TestOption(name="free jump", id=19),
    TestOption(name="lowered stand", id=20),
    TestOption(name="lowered walk", id=21),
]

LOWER_BODY_HEIGHT = 0.16  # meters; adjust if too low for your robot
BODYHEIGHT_API_ID = 1013


def nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def positive_finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite value > 0")
    return parsed


def try_set_body_height(sport_client: Any, height: float) -> bool:
    # BodyHeight is not registered in the Go2 client, so register and call it.
    sport_client._RegistApi(BODYHEIGHT_API_ID, 0)
    code, _ = sport_client._Call(BODYHEIGHT_API_ID, json.dumps({"data": height}))
    if code != 0:
        print(f"BodyHeight failed with code {code}; using normal height.")
        return False
    return True


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive Go2 SportClient command runner.")
    parser.add_argument("legacy_iface", nargs="?", help=argparse.SUPPRESS)
    parser.add_argument("--iface", help="Robot network interface. Defaults to SDK auto-detect.")
    parser.add_argument("--domain-id", type=nonnegative_int, default=0)
    parser.add_argument("--timeout", type=positive_finite_float, default=10.0)
    parser.add_argument("--yes", action="store_true", help="Confirm commands that can move the robot.")
    parser.add_argument("--list", action="store_true", help="List available commands and exit.")
    parser.add_argument("--once", help="Run one command by id or exact name, then exit.")
    return parser.parse_args(argv)


def print_options() -> None:
    for option in option_list:
        print(f"{option.name}, id: {option.id}")


def resolve_option(value: str) -> TestOption | None:
    cleaned = str(value).strip()
    for option in option_list:
        if cleaned == option.name or cleaned == str(option.id):
            return option
    return None


def run_option(sport_client: Any, option: TestOption) -> None:
    print(f"Running: {option.name}, id: {option.id}")
    if option.id == 0:
        print("ret:", sport_client.Damp())
    elif option.id == 1:
        print("ret:", sport_client.StandUp())
    elif option.id == 2:
        print("ret:", sport_client.StandDown())
    elif option.id == 3:
        print("ret:", sport_client.Move(0.3, 0, 0))
    elif option.id == 4:
        print("ret:", sport_client.Move(0, 0.3, 0))
    elif option.id == 5:
        print("ret:", sport_client.Move(0, 0, 0.5))
    elif option.id == 6:
        print("ret:", sport_client.StopMove())
    elif option.id == 7:
        print("ret:", sport_client.HandStand(True))
        time.sleep(4)
        print("ret:", sport_client.HandStand(False))
    elif option.id == 9:
        print("ret:", sport_client.BalanceStand())
    elif option.id == 10:
        print("ret:", sport_client.RecoveryStand())
    elif option.id == 11:
        print("ret:", sport_client.LeftFlip())
    elif option.id == 12:
        print("ret:", sport_client.BackFlip())
    elif option.id == 13:
        print("ret:", sport_client.FreeWalk())
    elif option.id == 14:
        print("ret:", sport_client.FreeBound(True))
        time.sleep(2)
        print("ret:", sport_client.FreeBound(False))
    elif option.id == 15:
        print("ret:", sport_client.FreeAvoid(True))
        time.sleep(2)
        print("ret:", sport_client.FreeAvoid(False))
    elif option.id == 17:
        print("ret:", sport_client.WalkUpright(True))
        time.sleep(4)
        print("ret:", sport_client.WalkUpright(False))
    elif option.id == 18:
        print("ret:", sport_client.CrossStep(True))
        time.sleep(4)
        print("ret:", sport_client.CrossStep(False))
    elif option.id == 19:
        print("ret:", sport_client.FreeJump(True))
        time.sleep(4)
        print("ret:", sport_client.FreeJump(False))
    elif option.id == 20:
        try_set_body_height(sport_client, LOWER_BODY_HEIGHT)
        print("ret:", sport_client.BalanceStand())
    elif option.id == 21:
        try_set_body_height(sport_client, LOWER_BODY_HEIGHT)
        print("ret:", sport_client.Move(0.3, 0, 0))


def main() -> int:
    args = parse_args()
    if args.list:
        print_options()
        return 0
    if not args.yes:
        print("This script can move the robot. Re-run with --yes to confirm.")
        return 2

    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.go2.sport.sport_client import SportClient
    except ImportError as exc:
        raise SystemExit(
            "unitree_sdk2py is not installed. Install it with:\n"
            "  pip install -e <path-to-unitree_sdk2_python>"
        ) from exc

    iface = args.iface or args.legacy_iface
    if iface:
        ChannelFactoryInitialize(args.domain_id, iface)
    else:
        ChannelFactoryInitialize(args.domain_id)

    sport_client = SportClient()
    sport_client.SetTimeout(float(args.timeout))
    sport_client.Init()

    if args.once:
        option = resolve_option(args.once)
        if option is None:
            print(f"No matching command: {args.once!r}")
            print_options()
            return 2
        run_option(sport_client, option)
        return 0

    print("Type 'list' to show commands, 'q' to quit.")
    while True:
        try:
            raw = input("Enter id or name: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if raw in {"q", "quit", "exit"}:
            return 0
        if raw == "list":
            print_options()
            continue
        option = resolve_option(raw)
        if option is None:
            print("No matching test option found.")
            continue
        run_option(sport_client, option)
        time.sleep(1)


if __name__ == "__main__":
    raise SystemExit(main())
