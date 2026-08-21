from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest


DEV_DIR = Path(__file__).resolve().parents[1] / "dev"
if str(DEV_DIR) not in sys.path:
    sys.path.insert(0, str(DEV_DIR))

from ai_control import ollama_client, tools  # noqa: E402
from ai_control.cli import _parse_args  # noqa: E402
from ai_control.config import AIConfig  # noqa: E402


class RecordingBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def _call(self, name: str, **kwargs: object) -> str:
        self.calls.append((name, kwargs))
        return "ok"

    def move(self, **kwargs: object) -> str:
        return self._call("move", **kwargs)

    def navigate_to(self, **kwargs: object) -> str:
        return self._call("navigate_to", **kwargs)

    def stop(self) -> str:
        return self._call("stop")

    def hand_open(self, **kwargs: object) -> str:
        return self._call("hand_open", **kwargs)

    def hand_close(self, **kwargs: object) -> str:
        return self._call("hand_close", **kwargs)

    def gesture(self, **kwargs: object) -> str:
        return self._call("gesture", **kwargs)

    def release_arms(self) -> str:
        return self._call("release_arms")

    def say(self, **kwargs: object) -> str:
        return self._call("say", **kwargs)

    def navbot_command(self, **kwargs: object) -> str:
        return self._call("navbot_command", **kwargs)

    def capture_frame(self) -> bytes | None:
        return None


def test_move_rejects_non_finite_and_overlong_commands() -> None:
    backend = RecordingBackend()
    with pytest.raises(ValueError, match="finite"):
        tools.dispatch("move", {"vx": math.nan}, backend)
    with pytest.raises(ValueError, match="<= 30"):
        tools.dispatch("move", {"duration": 31}, backend)
    assert backend.calls == []


def test_move_accepts_defaults_and_normalizes_numeric_strings() -> None:
    backend = RecordingBackend()
    assert tools.dispatch("move", {"vx": "0.2", "duration": "1.5"}, backend) == "ok"
    assert backend.calls == [("move", {"vx": 0.2, "vy": 0.0, "vyaw": 0.0, "duration": 1.5})]


def test_tool_payloads_validate_hands_and_text() -> None:
    backend = RecordingBackend()
    with pytest.raises(ValueError, match="hand must"):
        tools.dispatch("hand_open", {"hand": "middle"}, backend)
    with pytest.raises(ValueError, match="must not be empty"):
        tools.dispatch("say", {"text": "  "}, backend)


def test_config_rejects_invalid_runtime_values() -> None:
    with pytest.raises(ValueError, match="positive finite"):
        AIConfig(request_timeout_s=math.inf)
    with pytest.raises(ValueError, match="0 through 232"):
        AIConfig(domain_id=233)
    with pytest.raises(ValueError, match="absolute ROS topic"):
        AIConfig(navbot_command_topic="relative topic")


def test_config_strips_host_trailing_slash() -> None:
    config = AIConfig(ollama_host="http://localhost:11434/")
    assert config.ollama_host == "http://localhost:11434"


def test_cli_rejects_invalid_dds_domain_ids() -> None:
    assert _parse_args(["--domain-id", "232"]).domain_id == 232
    with pytest.raises(SystemExit):
        _parse_args(["--domain-id", "233"])


def test_json_extraction_handles_prose_and_nested_braces() -> None:
    raw = 'Here is the result: {"response":"keep {this}","tool_call":null} done.'
    assert ollama_client.extract_json_object(raw) == {
        "response": "keep {this}",
        "tool_call": None,
    }


def test_json_extraction_rejects_non_object_json() -> None:
    with pytest.raises(ollama_client.OllamaError, match="JSON object"):
        ollama_client.extract_json_object("[1, 2, 3]")
