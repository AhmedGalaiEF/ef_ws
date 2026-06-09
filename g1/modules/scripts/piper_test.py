#!/usr/bin/env python3
from __future__ import annotations
from sdk_client import Robot

import argparse
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


BIRTHDAY_LINES = {
    "en": (
        "Happy birthday, {name}.",
        "We wish you joy, laughter, and a wonderful day.",
        "Happy birthday, dear {name}.",
    ),
    "de": (
        "Alles Gute zum Geburtstag, {name}.",
        "Wir wuenschen dir Freude, Lachen und einen wunderschoenen Tag.",
        "Alles Gute, lieber {name}.",
    ),
    "fr": (
        "Joyeux anniversaire, {name}.",
        "Nous te souhaitons de la joie, des rires, et une tres belle journee.",
        "Joyeux anniversaire, cher {name}.",
    ),
    "es": (
        "Feliz cumpleanos, {name}.",
        "Te deseamos alegria, risas, y un dia maravilloso.",
        "Feliz cumpleanos, querido {name}.",
    ),
    "ar": (
        "Eid milad saeed, {name}.",
        "Natamanna laka al farah wal ibtisam fi hatha al yawm al jameel.",
        "Eid milad saeed ya {name}.",
    ),
}


LANGUAGE_NAMES = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "ar": "Arabic",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make the robot sing a short Piper birthday greeting in multiple languages."
    )
    parser.add_argument("name", help="Person's name to include in the birthday greeting.")
    parser.add_argument("--iface", default="eth0",
                        help="DDS network interface, for example eth0 or enp3s0.")
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain ID.")
    parser.add_argument("--volume", type=int, default=None, help="Optional playback volume 0-100.")
    parser.add_argument(
        "--languages",
        nargs="+",
        default=("en", "de", "fr", "es", "ar"),
        choices=tuple(BIRTHDAY_LINES),
        help="Languages to sing, in order.",
    )
    parser.add_argument("--pause-s", type=float, default=0.5, help="Pause between languages.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated lines without connecting to the robot.",
    )
    return parser.parse_args()


def birthday_text(language: str, name: str) -> str:
    return " ".join(line.format(name=name) for line in BIRTHDAY_LINES[language])


def main() -> int:
    args = parse_args()

    language_texts = [(language, birthday_text(language, args.name)) for language in args.languages]
    if args.dry_run:
        for language, text in language_texts:
            print(f"{LANGUAGE_NAMES[language]} ({language}): {text}")
        return 0

    robot = Robot(
        iface=args.iface,
        domain_id=args.domain_id,
        safety_boot=False,
        recover_dev_mode_on_init=False,
        auto_start_sensors=False,
    )

    for language, text in language_texts:
        print(f"Singing in {LANGUAGE_NAMES[language]} ({language})...")
        code = robot.say(text, volume=args.volume, language=language)
        print(f"Robot.say returned {code}")
        time.sleep(max(0.0, float(args.pause_s)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
