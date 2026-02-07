#!/usr/bin/env python3
"""Play greeting.wav on the robot."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def _find_player() -> list[str] | None:
    for cmd in ("aplay", "paplay", "ffplay"):
        if subprocess.call(["/usr/bin/env", "which", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
            return [cmd]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Play greeting.wav.")
    parser.add_argument("--file", default="greeting.wav", help="path to wav file")
    args = parser.parse_args()

    wav_path = args.file
    if not os.path.isabs(wav_path):
        wav_path = os.path.join(os.path.dirname(__file__), wav_path)

    if not os.path.exists(wav_path):
        print(f"Missing wav file: {wav_path}")
        sys.exit(1)

    player = _find_player()
    if not player:
        print("No audio player found. Install aplay/paplay/ffplay.")
        sys.exit(2)

    cmd = player + [wav_path]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Audio player failed: {exc}")
        sys.exit(exc.returncode)


if __name__ == "__main__":
    main()
