#!/usr/bin/env python3
"""Record a WAV file with Enter-to-start and Enter-to-stop controls."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


def _find_arecord() -> str | None:
    for cmd in ("arecord", "parec", "ffmpeg"):
        if subprocess.call(["/usr/bin/env", "which", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
            return cmd
    return None


def _record_with_arecord(out_path: str, rate: int, channels: int, fmt: str) -> int:
    cmd = ["arecord", "-q", "-f", fmt, "-r", str(rate), "-c", str(channels), out_path]
    print(f"Recording with: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)
    try:
        input("Press Enter to stop recording... ")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()
    return 0


def _record_with_parec(out_path: str, rate: int, channels: int) -> int:
    cmd = ["parec", "--raw", f"--rate={rate}", f"--channels={channels}"]
    ffmpeg = [
        "ffmpeg",
        "-y",
        "-f",
        "s16le",
        "-ar",
        str(rate),
        "-ac",
        str(channels),
        "-i",
        "-",
        out_path,
    ]
    print(f"Recording with: {' '.join(cmd)} | {' '.join(ffmpeg)}")
    parec = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    ff = subprocess.Popen(ffmpeg, stdin=parec.stdout)
    try:
        input("Press Enter to stop recording... ")
    finally:
        parec.terminate()
        ff.terminate()
        for p in (parec, ff):
            try:
                p.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                p.kill()
    return 0


def _record_with_ffmpeg(out_path: str, rate: int, channels: int) -> int:
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "alsa",
        "-i",
        "default",
        "-ar",
        str(rate),
        "-ac",
        str(channels),
        out_path,
    ]
    print(f"Recording with: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)
    try:
        input("Press Enter to stop recording... ")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Record a WAV file with Enter-to-start and Enter-to-stop.")
    parser.add_argument("outfile", help="output WAV filename")
    parser.add_argument("--rate", type=int, default=16000, help="sample rate (Hz)")
    parser.add_argument("--channels", type=int, default=1, help="number of channels")
    parser.add_argument("--format", default="S16_LE", help="arecord format (default: S16_LE)")
    args = parser.parse_args()

    out_path = args.outfile
    if not out_path.lower().endswith(".wav"):
        out_path += ".wav"

    if os.path.exists(out_path):
        print(f"Output already exists: {out_path}")
        return 2

    print("Press Enter to start recording...")
    input()

    recorder = _find_arecord()
    if recorder == "arecord":
        return _record_with_arecord(out_path, args.rate, args.channels, args.format)
    if recorder == "parec":
        return _record_with_parec(out_path, args.rate, args.channels)
    if recorder == "ffmpeg":
        return _record_with_ffmpeg(out_path, args.rate, args.channels)

    print("No recorder found. Install 'arecord' (alsa-utils) or 'ffmpeg'.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
