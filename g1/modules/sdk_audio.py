from __future__ import annotations

import audioop
import os
import re
import subprocess
import tempfile
import time
import wave
from pathlib import Path
from typing import Optional

from dds_env import ensure_cyclonedds_environment


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

_PIPER_VOICE_DIR = Path.home() / ".local" / "share" / "piper" / "voices"
_DEFAULT_PIPER_VOICES = {
    "en": "en_US-lessac-medium",
    "en_us": "en_US-lessac-medium",
    "english": "en_US-lessac-medium",
    "de": "de_DE-thorsten-medium",
    "de_de": "de_DE-thorsten-medium",
    "german": "de_DE-thorsten-medium",
    "fr": "fr_FR-siwis-medium",
    "fr_fr": "fr_FR-siwis-medium",
    "french": "fr_FR-siwis-medium",
    "es": "es_ES-davefx-medium",
    "es_es": "es_ES-davefx-medium",
    "spanish": "es_ES-davefx-medium",
    "ar": "ar_JO-kareem-medium",
    "ar_jo": "ar_JO-kareem-medium",
    "arabic": "ar_JO-kareem-medium",
}


def _load_audio_client():
    ensure_cyclonedds_environment()
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


def _convert_wav_for_robot(src_path: Path, dst_path: Path) -> Path:
    with wave.open(str(src_path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frame_rate = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())

    if channels == 2:
        pcm = audioop.tomono(pcm, sample_width, 0.5, 0.5)
        channels = 1
    elif channels != 1:
        raise ValueError(f"WAV must be mono or stereo PCM, got {channels} channels")

    if sample_width != 2:
        pcm = audioop.lin2lin(pcm, sample_width, 2)
        sample_width = 2

    if frame_rate != 16000:
        pcm, _state = audioop.ratecv(pcm, sample_width, channels, frame_rate, 16000, None)
        frame_rate = 16000

    with wave.open(str(dst_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm)

    return dst_path


def _find_piper_binary() -> str:
    piper_bin = os.environ.get("G1_PIPER_BIN") or os.environ.get("PIPER_BIN") or "piper"
    if subprocess.call(
        ["/usr/bin/env", "which", piper_bin],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ) != 0:
        local_piper = Path.home() / ".local" / "bin" / "piper"
        if piper_bin == "piper" and local_piper.exists():
            return str(local_piper)
        raise RuntimeError(
            "piper is required for say(); install Piper or set G1_PIPER_BIN/PIPER_BIN to the piper executable"
        )
    return piper_bin


def _piper_voice_model_path(voice_name: str) -> Path:
    return _PIPER_VOICE_DIR / voice_name / f"{voice_name}.onnx"


def _resolve_piper_model(
    model: str | os.PathLike[str] | None = None,
    language: str | None = None,
) -> Path:
    value = model
    if not value and language:
        voice_name = _DEFAULT_PIPER_VOICES.get(language.strip().lower().replace("-", "_"))
        if not voice_name:
            supported = ", ".join(("en", "de", "fr", "es"))
            raise ValueError(
                f"unsupported Piper language {language!r}; supported languages: {supported}")
        value = _piper_voice_model_path(voice_name)
    if not value:
        value = os.environ.get("G1_PIPER_MODEL") or os.environ.get("PIPER_MODEL")

    if not value:
        default_models = (
            _piper_voice_model_path("en_US-lessac-medium"),
            Path(__file__).resolve().parent / ".piper_voices" /
            "en_US-lessac-medium" / "en_US-lessac-medium.onnx",
        )
        for default_model in default_models:
            if default_model.exists():
                return default_model
        raise RuntimeError(
            "Piper voice model is required for say(); set G1_PIPER_MODEL to a .onnx voice file")

    model_path = Path(value).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"Piper voice model does not exist: {model_path}")
    return model_path


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

    def speak(
        self,
        text: str,
        volume: Optional[int] = None,
        model: str | os.PathLike[str] | None = None,
        language: str | None = None,
        speaker: int | None = None,
    ) -> int:
        piper_bin = _find_piper_binary()
        model_path = _resolve_piper_model(model, language=language)
        with tempfile.TemporaryDirectory(prefix="g1_say_") as td:
            wav_path = Path(td) / "speech.wav"
            robot_wav_path = Path(td) / "speech_robot.wav"
            command = [piper_bin, "--model", str(model_path), "--output-file", str(wav_path)]
            if speaker is not None:
                command.extend(["--speaker", str(int(speaker))])
            subprocess.run(command, input=text, text=True, check=True)
            return self.play_wav(_convert_wav_for_robot(wav_path, robot_wav_path), volume=volume)


__all__ = ["RobotAudio", "parse_color"]
