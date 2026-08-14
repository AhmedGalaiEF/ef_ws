from __future__ import annotations

import sys
import types
import wave
from pathlib import Path

import pytest


OLLAMA_AI_DIR = Path(__file__).resolve().parents[1] / "g1" / "modules" / "scripts" / "ollama_ai"
if str(OLLAMA_AI_DIR) not in sys.path:
    sys.path.insert(0, str(OLLAMA_AI_DIR))

import local_asr  # noqa: E402


def test_host_only_asr_url_uses_navbot_default_port() -> None:
    assert local_asr.normalize_post_url("127.0.0.1") == "http://127.0.0.1:8096/asr"
    assert local_asr.normalize_post_url("http://robot:9000/command") == "http://robot:9000/command"


def test_asr_url_rejects_non_http_schemes() -> None:
    with pytest.raises(SystemExit, match="http or https"):
        local_asr.normalize_post_url("ftp://robot/asr")


def test_invalid_url_port_is_reported_as_system_exit() -> None:
    with pytest.raises(SystemExit, match="Invalid --url port"):
        local_asr.normalize_post_url("http://robot:99999/asr")


def test_parse_args_rejects_invalid_capture_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["local_asr.py", "--text-input", "--gain", "nan"])
    with pytest.raises(SystemExit, match="--gain"):
        local_asr.parse_args()


def test_brand_corrections_are_applied_to_transcripts() -> None:
    assert local_asr.apply_brand_corrections("Fahre zu Pukt Heinz") == "Fahre zu Punkt eins"
    assert local_asr.apply_brand_corrections("Unitree gee one") == "Unitree G1"


def test_vosk_uses_wav_sample_rate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, int] = {}

    class FakeRecognizer:
        def __init__(self, _model: object, sample_rate: int) -> None:
            captured["sample_rate"] = sample_rate

        def AcceptWaveform(self, _data: bytes) -> None:
            return None

        def FinalResult(self) -> str:
            return '{"text": "test"}'

    fake_vosk = types.SimpleNamespace(
        Model=lambda _path: object(),
        KaldiRecognizer=FakeRecognizer,
    )
    monkeypatch.setitem(sys.modules, "vosk", fake_vosk)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    wav_path = tmp_path / "sample.wav"
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(b"\x00\x00" * 100)

    backend = local_asr.VoskBackend(str(model_dir), "de")
    assert backend.transcribe(wav_path) == "test"
    assert captured["sample_rate"] == 8000
