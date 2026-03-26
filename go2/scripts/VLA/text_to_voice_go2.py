#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from sdk_safety import assert_go2_audio_supported, require_known_interface
from text_to_wav import text_to_wav


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
        description="Convert text to WAV locally. Go2 playback requires a Go2 audio client SDK."
    )
    parser.add_argument("text", help="text to speak")
    parser.add_argument("--iface", default="enp0s31f6", help="DDS network interface, e.g. enp0s31f6")
    parser.add_argument("--volume", type=_parse_volume, default=None, help="reserved for future Go2 audio support")
    parser.add_argument("--timeout", type=float, default=5.0, help="reserved for future Go2 audio support")
    parser.add_argument("--speaker-id", type=int, default=0, help="reserved for future Go2 audio support")
    parser.add_argument("--lang", default="en", help="language code for gTTS conversion")
    parser.add_argument("--app-name", default="go2-tts", help="reserved for future Go2 audio support")
    parser.add_argument("--stream-id", default="stream-1", help="reserved for future Go2 audio support")
    parser.add_argument(
        "--mode",
        choices=("robot-tts", "gtts-stream"),
        default="robot-tts",
        help="kept for CLI compatibility; neither mode can play on Go2 with the installed SDK",
    )
    parser.add_argument(
        "--save-wav",
        help="output WAV path; if omitted, a temporary WAV is created and immediately discarded",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    wav_path: Path | None = None
    if args.mode == "gtts-stream":
        if args.save_wav:
            wav_path = _resolve_wav(args.save_wav)
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            text_to_wav(text=args.text, output=str(wav_path), lang=args.lang)
            print(f"Wrote WAV: {wav_path}")
        else:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav:
                text_to_wav(text=args.text, output=temp_wav.name, lang=args.lang)
                print("Generated temporary WAV successfully.")

    try:
        require_known_interface(args.iface)
        assert_go2_audio_supported()
    except SystemExit as exc:
        message = str(exc)
        if wav_path is not None:
            raise SystemExit(f"{message}\nLocal WAV generation succeeded: {wav_path}") from exc
        raise SystemExit(message) from exc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
