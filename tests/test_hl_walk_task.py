from __future__ import annotations

import pytest

import go2.scripts.hl_walk_task as hl_walk_task


class FakeClient:
    def __init__(self) -> None:
        self.moves = 0
        self.stops = 0

    def Move(self, *_args: float) -> None:
        self.moves += 1

    def StopMove(self) -> None:
        self.stops += 1


def test_walk_distance_raises_and_stops_when_pose_does_not_change(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hl_walk_task, "last_odom_pos", [0.0, 0.0])
    monkeypatch.setattr(hl_walk_task, "last_sport_pos", None)
    client = FakeClient()

    with pytest.raises(TimeoutError, match="Walk timed out"):
        hl_walk_task._walk_distance(client, speed=0.2, distance=1.0, tick=0.001, timeout=0.01)

    assert client.moves > 0
    assert client.stops == 1


def test_turn_raises_and_stops_when_yaw_does_not_change(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hl_walk_task, "last_imu_yaw", 0.0)
    client = FakeClient()

    with pytest.raises(TimeoutError, match="Turn timed out"):
        hl_walk_task._turn_to_delta(client, delta_yaw=1.0, yaw_rate=0.5, tick=0.001, timeout=0.01)

    assert client.moves > 0
    assert client.stops == 1
