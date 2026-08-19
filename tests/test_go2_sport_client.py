from __future__ import annotations

import argparse

import pytest

from go2.scripts import go2_sport_client
from go2.scripts.go2_sport_client import (
    parse_args,
    positive_finite_float,
    run_timed_move,
    run_timed_toggle,
)


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf"])
def test_positive_finite_float_rejects_invalid_timeout(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        positive_finite_float(value)


def test_parse_args_validates_robot_timeout() -> None:
    args = parse_args(["--timeout", "2.5", "--domain-id", "3"])
    assert args.timeout == pytest.approx(2.5)
    assert args.domain_id == 3

    with pytest.raises(SystemExit):
        parse_args(["--timeout", "nan"])


def test_timed_move_stops_when_sleep_is_interrupted(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []

    class Client:
        def Move(self, *args: object) -> int:
            calls.append(("move", *args))
            return 0

        def StopMove(self) -> int:
            calls.append(("stop",))
            return 0

    def interrupt(_duration: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(go2_sport_client.time, "sleep", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_timed_move(Client(), 0.2, 0.0, 0.0, 1.0)
    assert calls == [("move", 0.2, 0.0, 0.0), ("stop",)]


def test_timed_move_attempts_stop_when_move_raises() -> None:
    calls: list[str] = []

    class Client:
        def Move(self, *_args: object) -> int:
            calls.append("move")
            raise RuntimeError("transport failed after sending")

        def StopMove(self) -> int:
            calls.append("stop")
            return 0

    with pytest.raises(RuntimeError, match="transport failed"):
        run_timed_move(Client(), 0.2, 0.0, 0.0, 1.0)
    assert calls == ["move", "stop"]


def test_timed_toggle_disables_mode_when_sleep_is_interrupted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states: list[bool] = []

    def action(enabled: bool) -> int:
        states.append(enabled)
        return 0

    def interrupt(_duration: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(go2_sport_client.time, "sleep", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_timed_toggle(action, 1.0)
    assert states == [True, False]
