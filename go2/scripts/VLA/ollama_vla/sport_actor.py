from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from unitree_sdk2py.go2.sport.sport_client import SportClient


@dataclass
class ExecutedCommand:
    name: str
    args: Dict[str, Any]
    duration_sec: float
    code: int
    timestamp: float = field(default_factory=time.time)


class SportCommandExecutor:
    def __init__(self, timeout_sec: float = 5.0, dry_run: bool = True):
        self._client = SportClient()
        self._timeout_sec = timeout_sec
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
        duration_sec = float(command.get("duration_sec", 0.0) or 0.0)

        if self._dry_run:
            return ExecutedCommand(name=name, args=args, duration_sec=duration_sec, code=0)

        code = self._dispatch(name, args, duration_sec)
        return ExecutedCommand(name=name, args=args, duration_sec=duration_sec, code=code)

    def _dispatch(self, name: str, args: Dict[str, Any], duration_sec: float) -> int:
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
        if name == "free_walk":
            return self._client.FreeWalk()
        if name == "sit":
            return self._client.Sit()
        if name == "rise_sit":
            return self._client.RiseSit()
        if name == "move":
            vx = float(args.get("vx", 0.0))
            vy = float(args.get("vy", 0.0))
            vyaw = float(args.get("vyaw", 0.0))
            code = self._client.Move(vx, vy, vyaw)
            if duration_sec > 0.0:
                time.sleep(duration_sec)
                self._client.StopMove()
            return code
        raise ValueError(f"unsupported action: {name}")
