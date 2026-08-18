from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from unitree_sdk2py.go2.sport.sport_client import SportClient


MAX_COMMAND_DURATION_SEC = 30.0
MOVE_LIMITS = (0.35, 0.20, 0.60)


@dataclass
class ExecutedCommand:
    name: str
    args: Dict[str, Any]
    duration_sec: float
    code: int
    timestamp: float = field(default_factory=time.time)


class SportCommandExecutor:
    def __init__(self, timeout_sec: float = 5.0, dry_run: bool = True):
        if not math.isfinite(float(timeout_sec)) or float(timeout_sec) <= 0.0:
            raise ValueError("timeout_sec must be a finite value > 0")
        self._client = SportClient()
        self._timeout_sec = float(timeout_sec)
        self._dry_run = dry_run

    def start(self) -> None:
        if self._dry_run:
            return
        self._client.SetTimeout(self._timeout_sec)
        self._client.Init()

    def execute_many(self, commands: List[Dict[str, Any]]) -> List[ExecutedCommand]:
        executed: List[ExecutedCommand] = []
        for command in commands:
            executed.append(self.execute(command))
        return executed

    def execute(self, command: Dict[str, Any]) -> ExecutedCommand:
        name = str(command.get("name", "stop_move"))
        args = dict(command.get("args", {}) or {})
        try:
            duration_sec = float(command.get("duration_sec", 0.0) or 0.0)
        except (TypeError, ValueError) as exc:
            raise ValueError("duration_sec must be a finite number") from exc
        if not math.isfinite(duration_sec) or not 0.0 <= duration_sec <= MAX_COMMAND_DURATION_SEC:
            raise ValueError(f"duration_sec must be finite and between 0 and {MAX_COMMAND_DURATION_SEC:g} seconds")

        if self._dry_run:
            return ExecutedCommand(name=name, args=args, duration_sec=duration_sec, code=0)

        code = self._dispatch(name, args, duration_sec)
        return ExecutedCommand(name=name, args=args, duration_sec=duration_sec, code=code)

    def _dispatch(self, name: str, args: Dict[str, Any], duration_sec: float) -> int:
        if name == "damp":
            return self._client.Damp()
        if name == "stop_move":
            return self._client.StopMove()
        if name == "stand_up":
            return self._client.StandUp()
        if name == "stand_down":
            return self._client.StandDown()
        if name == "balance_stand":
            return self._client.BalanceStand()
        if name == "recovery":
            return self._client.RecoveryStand()
        if name == "hello":
            return self._client.Hello()
        if name == "stretch":
            return self._client.Stretch()
        if name == "content":
            return self._client.Content()
        if name == "free_walk":
            return self._client.FreeWalk()
        if name == "pose_on":
            return self._client.Pose(True)
        if name == "pose_off":
            return self._client.Pose(False)
        if name == "dance1":
            return self._client.Dance1()
        if name == "dance2":
            return self._client.Dance2()
        if name == "static_walk":
            return self._client.StaticWalk()
        if name == "trot_run":
            return self._client.TrotRun()
        if name == "walk_upright_on":
            return self._client.WalkUpright(True)
        if name == "walk_upright_off":
            return self._client.WalkUpright(False)
        if name == "classic_walk_on":
            return self._client.ClassicWalk(True)
        if name == "classic_walk_off":
            return self._client.ClassicWalk(False)
        if name == "switch_avoid_mode":
            return self._client.SwitchAvoidMode()
        if name == "speed_level":
            level = int(args.get("level", 1) or 1)
            if level not in {-1, 0, 1}:
                raise ValueError("speed level must be -1, 0, or 1")
            return self._client.SpeedLevel(level)
        if name == "sit":
            return self._client.Sit()
        if name == "rise_sit":
            return self._client.RiseSit()
        if name == "move":
            try:
                vx = float(args.get("vx", 0.0))
                vy = float(args.get("vy", 0.0))
                vyaw = float(args.get("vyaw", 0.0))
            except (TypeError, ValueError) as exc:
                raise ValueError("move values must be finite numbers") from exc
            if not all(math.isfinite(value) for value in (vx, vy, vyaw)):
                raise ValueError("move values must be finite numbers")
            if not all(abs(value) <= limit for value, limit in zip((vx, vy, vyaw), MOVE_LIMITS)):
                raise ValueError("move command exceeds conservative Go2 velocity limits")
            code = self._client.Move(vx, vy, vyaw)
            if duration_sec > 0.0:
                time.sleep(duration_sec)
                self._client.StopMove()
            return code
        raise ValueError(f"unsupported action: {name}")
