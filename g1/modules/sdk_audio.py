from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
import wave
from pathlib import Path
from typing import Optional


_NAMED_COLORS = {
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "pink": (255, 105, 180),
}


def _load_audio_client():
    from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

    return AudioClient


def parse_color(value: str | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(value, tuple) and len(value) == 3:
        return tuple(int(max(0, min(255, v))) for v in value)

    lowered = str(value).strip().lower()
    if lowered in _NAMED_COLORS:
        return _NAMED_COLORS[lowered]
    if re.fullmatch(r"#?[0-9a-fA-F]{6}", lowered):
        hexval = lowered.lstrip("#")
        return (int(hexval[0:2], 16), int(hexval[2:4], 16), int(hexval[4:6], 16))
    if re.fullmatch(r"\d{1,3},\d{1,3},\d{1,3}", lowered):
        parts = [int(p) for p in lowered.split(",")]
        if all(0 <= p <= 255 for p in parts):
            return (parts[0], parts[1], parts[2])
    raise ValueError("color must be a name, #RRGGBB, or R,G,B")


def scale_color(rgb: tuple[int, int, int], intensity: int) -> tuple[int, int, int]:
    level = max(0, min(100, int(intensity)))
    if level >= 100:
        return rgb
    scale = level / 100.0
    return (int(rgb[0] * scale), int(rgb[1] * scale), int(rgb[2] * scale))


class RobotAudio:
    def __init__(self) -> None:
        audio_client_cls = _load_audio_client()
        self._client = audio_client_cls()
        self._client.SetTimeout(5.0)
        self._client.Init()

    def set_headlight(
        self,
        color: str | tuple[int, int, int] = "white",
        intensity: int = 100,
        duration: float | None = None,
    ) -> int:
        rgb = scale_color(parse_color(color), intensity)
        code = int(self._client.LedControl(*rgb))
        if code != 0:
            return code
        if duration is None:
            time.sleep(0.25)
            return 0
        time.sleep(max(0.0, float(duration)))
        time.sleep(0.25)
        return int(self._client.LedControl(0, 0, 0))

    def set_volume(self, level: int) -> int:
        return int(self._client.SetVolume(int(level)))

    def play_wav(self, wav_path: str | os.PathLike[str], volume: Optional[int] = None) -> int:
        if volume is not None:
            code = self.set_volume(volume)
            if code != 0:
                return code

        with wave.open(str(wav_path), "rb") as wf:
            if wf.getnchannels() != 1 or wf.getframerate() != 16000 or wf.getsampwidth() != 2:
                raise ValueError("WAV must be mono 16-bit PCM at 16kHz for robot playback")
            pcm = wf.readframes(wf.getnframes())

        code, _data = self._client.PlayStream("sdk_client", "sdk-client-1", pcm)
        return int(code)

    def speak(self, text: str, volume: Optional[int] = None) -> int:
        if subprocess.call(
            ["/usr/bin/env", "which", "espeak"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ) != 0:
            raise RuntimeError("espeak is required for say(); SDK audio playback is available but TTS is not built in")

        with tempfile.TemporaryDirectory(prefix="g1_say_") as td:
            wav_path = Path(td) / "speech.wav"
            subprocess.run(["espeak", "-w", str(wav_path), text], check=True)
            return self.play_wav(wav_path, volume=volume)


__all__ = ["RobotAudio", "parse_color"]
