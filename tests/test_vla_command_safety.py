from __future__ import annotations

import argparse
import math

import pytest

from go2.scripts.VLA.ollama_vla import sport_actor
from go2.scripts.VLA.ollama_vla.agents import ActorAgent
from go2.scripts.VLA.ollama_vla.config import RuntimeConfig
from go2.scripts.VLA.cli_validation import positive_finite_float, positive_int
from go2.scripts.VLA.intent_summary import summarize_commands


def _dry_executor(monkeypatch: pytest.MonkeyPatch) -> sport_actor.SportCommandExecutor:
    monkeypatch.setattr(sport_actor, "SportClient", lambda: object())
    return sport_actor.SportCommandExecutor(dry_run=True)


@pytest.mark.parametrize(
    "command",
    [
        {"name": "move", "args": {"vx": "nan"}, "duration_sec": 1.0},
        {"name": "move", "args": {"vx": 0.36}, "duration_sec": 1.0},
        {"name": "move", "args": {"vx": 0.2}, "duration_sec": 0.0},
        {"name": "move", "args": [], "duration_sec": 1.0},
        {"name": "unknown", "args": {}, "duration_sec": 0.0},
        {"name": "speed_level", "args": {"level": 2}, "duration_sec": 0.0},
    ],
)
def test_dry_run_validates_commands(
    monkeypatch: pytest.MonkeyPatch, command: dict[str, object]
) -> None:
    with pytest.raises(ValueError):
        _dry_executor(monkeypatch).execute(command)


def test_dry_run_returns_normalized_command(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _dry_executor(monkeypatch).execute(
        {
            "name": "move",
            "args": {"vx": "0.2", "vy": 0, "vyaw": "-0.35"},
            "duration_sec": "1.5",
        }
    )
    assert result.args == {"vx": 0.2, "vy": 0.0, "vyaw": -0.35}
    assert result.duration_sec == pytest.approx(1.5)


def test_live_move_stops_when_sleep_is_interrupted(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []

    class Client:
        def SetTimeout(self, _timeout: float) -> None:
            pass

        def Init(self) -> None:
            pass

        def Move(self, *args: object) -> int:
            calls.append(("move", *args))
            return 0

        def StopMove(self) -> int:
            calls.append(("stop",))
            return 0

    client = Client()
    monkeypatch.setattr(sport_actor, "SportClient", lambda: client)

    def interrupt(_duration: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(sport_actor.time, "sleep", interrupt)
    executor = sport_actor.SportCommandExecutor(dry_run=False)
    with pytest.raises(KeyboardInterrupt):
        executor.execute(
            {"name": "move", "args": {"vx": 0.2, "vy": 0.0, "vyaw": 0.0}, "duration_sec": 1.0}
        )
    assert calls == [("move", 0.2, 0.0, 0.0), ("stop",)]


def test_live_move_attempts_stop_when_move_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class Client:
        def Move(self, *_args: object) -> int:
            calls.append("move")
            raise RuntimeError("transport failed after sending")

        def StopMove(self) -> int:
            calls.append("stop")
            return 0

    monkeypatch.setattr(sport_actor, "SportClient", Client)
    executor = sport_actor.SportCommandExecutor(dry_run=False)
    with pytest.raises(RuntimeError, match="transport failed"):
        executor.execute(
            {"name": "move", "args": {"vx": 0.2, "vy": 0.0, "vyaw": 0.0}, "duration_sec": 1.0}
        )
    assert calls == ["move", "stop"]


def test_command_batch_is_validated_before_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class Client:
        def StandUp(self) -> int:
            calls.append("stand_up")
            return 0

    monkeypatch.setattr(sport_actor, "SportClient", Client)
    executor = sport_actor.SportCommandExecutor(dry_run=False)
    with pytest.raises(ValueError):
        executor.execute_many(
            [
                {"name": "stand_up", "args": {}, "duration_sec": 0.0},
                {"name": "unknown", "args": {}, "duration_sec": 0.0},
            ]
        )
    assert calls == []


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf"])
def test_vla_cli_float_validator_rejects_invalid_periods(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        positive_finite_float(value)


@pytest.mark.parametrize("value", ["0", "-1", "nan"])
def test_vla_cli_int_validator_rejects_invalid_prediction_counts(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int(value)


def test_intent_summary_survives_malformed_model_numbers() -> None:
    summary = summarize_commands(
        [{"name": "move", "args": {"vx": "nan", "vy": "not-a-number", "vyaw": "inf"}}]
    )
    assert summary == "make a small adjustment in place"


def test_actor_output_matches_executor_motion_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = RuntimeConfig()
    runtime.allowed_actions.update(max_vx=99.0, max_vy=99.0, max_vyaw=99.0)
    actor = ActorAgent(client=None, system_prompt="", runtime=runtime)  # type: ignore[arg-type]
    output = actor.map_actions(
        {
            "suggested_actions": [
                {
                    "name": "move",
                    "args": {"vx": 99.0, "vy": -99.0, "vyaw": 99.0},
                    "duration_sec": 99.0,
                }
            ]
        }
    )
    command = output["commands"][0]
    limits = sport_actor.MOVE_LIMITS
    assert abs(command["args"]["vx"]) <= limits[0]
    assert abs(command["args"]["vy"]) <= limits[1]
    assert abs(command["args"]["vyaw"]) <= limits[2]
    assert command["duration_sec"] <= runtime.allowed_actions["max_duration_sec"]
    assert all(math.isfinite(value) for value in command["args"].values())

    _dry_executor(monkeypatch).execute(command)


def test_actor_replaces_non_finite_model_values() -> None:
    actor = ActorAgent(client=None, system_prompt="", runtime=RuntimeConfig())  # type: ignore[arg-type]
    command = actor.map_actions(
        {
            "suggested_actions": [
                {
                    "name": "move",
                    "args": {"vx": "nan", "vy": "inf", "vyaw": "-inf"},
                    "duration_sec": "nan",
                }
            ]
        }
    )["commands"][0]
    assert all(math.isfinite(value) for value in command["args"].values())
    assert math.isfinite(command["duration_sec"])
