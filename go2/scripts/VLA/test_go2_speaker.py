#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from sdk_safety import assert_go2_audio_supported, require_known_interface


def _parse_volume(value: str) -> int:
    try:
        volume = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("volume must be an integer in range 0-100") from exc
    if not 0 <= volume <= 100:
        raise argparse.ArgumentTypeError("volume must be an integer in range 0-100")
    return volume


def _resolve_wav(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preflight Go2 speaker arguments without calling unsupported SDK audio APIs."
    )
    parser.add_argument("--iface", default="enp0s31f6", help="DDS network interface, e.g. enp0s31f6")
    parser.add_argument("--volume", type=_parse_volume, default=None, help="reserved for future Go2 audio support")
    parser.add_argument("--timeout", type=float, default=5.0, help="reserved for future Go2 audio support")
    parser.add_argument("--speaker-id", type=int, default=0, help="reserved for future Go2 audio support")
    parser.add_argument("--app-name", default="go2-speaker-test", help="reserved for future Go2 audio support")
    parser.add_argument("--stream-id", default="stream-1", help="reserved for future Go2 audio support")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--text", help="text that would be spoken if a Go2 audio client were available")
    mode.add_argument("--wav", help="path to mono 16-bit 16kHz WAV file that would be streamed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.wav:
        wav_path = _resolve_wav(args.wav)
        if not wav_path.exists():
            raise SystemExit(f"WAV file not found: {wav_path}")

    require_known_interface(args.iface)
    assert_go2_audio_supported()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
