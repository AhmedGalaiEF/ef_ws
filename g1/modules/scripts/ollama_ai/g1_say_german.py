#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
ROBOT_SAY_ONCE = SCRIPTS_DIR / "robot_say_once.py"


def volume_arg(value: str) -> int:
    volume = int(value)
    if not 0 <= volume <= 100:
        raise argparse.ArgumentTypeError("volume must be between 0 and 100")
    return volume


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prompt for German text and speak it through the G1 speaker."
    )
    parser.add_argument("--iface", default="eth0", help="DDS network interface.")
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain ID.")
    parser.add_argument("--volume", type=volume_arg, default=None, help="Optional playback volume 0-100.")
    parser.add_argument("--language", default="de", help="TTS language code. Default: de.")
    parser.add_argument(
        "--text",
        default=None,
        help="Speak this text once instead of prompting interactively.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Keep prompting after each spoken line. Empty input exits.",
    )
    return parser.parse_args()


def speak(args: argparse.Namespace, text: str) -> int:
    text = " ".join(text.split())
    if not text:
        return 0
    if not ROBOT_SAY_ONCE.exists():
        print(f"Missing speech helper: {ROBOT_SAY_ONCE}", file=sys.stderr)
        return 2

    command = [
        sys.executable,
        str(ROBOT_SAY_ONCE),
        text,
        "--iface",
        str(args.iface),
        "--domain-id",
        str(int(args.domain_id)),
        "--language",
        str(args.language),
    ]
    if args.volume is not None:
        command.extend(["--volume", str(int(args.volume))])

    env = os.environ.copy()
    env.setdefault("CYCLONEDDS_HOME", "/home/unitree/cyclonedds_ws/install/cyclonedds")
    env.setdefault("CYCLONEDDS_URI", "/home/unitree/cyclonedds_ws/cyclonedds.xml")
    return int(subprocess.call(command, env=env))


def prompt_once() -> str:
    try:
        return input("Text fuer die deutsche G1-Stimme: ").strip()
    except EOFError:
        return ""


def main() -> int:
    args = parse_args()
    if args.text is not None:
        return speak(args, args.text)

    if args.loop:
        while True:
            text = prompt_once()
            if not text:
                return 0
            code = speak(args, text)
            if code != 0:
                return code

    text = prompt_once()
    if not text:
        print("Kein Text eingegeben.")
        return 0
    return speak(args, text)


if __name__ == "__main__":
    raise SystemExit(main())
