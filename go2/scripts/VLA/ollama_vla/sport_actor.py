from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping

from .config import CONSERVATIVE_MOVE_LIMITS

try:
    from unitree_sdk2py.go2.sport.sport_client import SportClient
except ImportError:  # Allow command validation and dry runs off-robot.
    SportClient = None  # type: ignore[assignment]


MAX_COMMAND_DURATION_SEC = 30.0
MOVE_LIMITS = CONSERVATIVE_MOVE_LIMITS
SUPPORTED_ACTIONS = frozenset(
    {
        "damp",
        "stop_move",
        "stand_up",
        "stand_down",
        "balance_stand",
        "recovery",
        "hello",
        "stretch",
        "content",
        "free_walk",
        "pose_on",
        "pose_off",
        "dance1",
        "dance2",
        "static_walk",
        "trot_run",
        "walk_upright_on",
        "walk_upright_off",
        "classic_walk_on",
        "classic_walk_off",
        "switch_avoid_mode",
        "speed_level",
        "sit",
        "rise_sit",
        "move",
    }
)


@dataclass
class ExecutedCommand:
    name: str
    args: Dict[str, Any]
    duration_sec: float
    code: int
    timestamp: float = field(default_factory=time.time)


class SportCommandExecutor:
    def __init__(self, timeout_sec: float = 5.0, dry_run: bool = True):
        try:
            parsed_timeout = float(timeout_sec)
        except (TypeError, ValueError) as exc:
            raise ValueError("timeout_sec must be a finite value > 0") from exc
        if not math.isfinite(parsed_timeout) or parsed_timeout <= 0.0:
            raise ValueError("timeout_sec must be a finite value > 0")
        if not dry_run and SportClient is None:
            raise RuntimeError("unitree_sdk2py is required for live robot commands")
        self._client = None if dry_run else SportClient()
        self._timeout_sec = parsed_timeout
        self._dry_run = bool(dry_run)

    def start(self) -> None:
        if self._dry_run:
            return
        if self._client is None:  # Defensive guard for dynamically changed environments.
            raise RuntimeError("unitree_sdk2py is required for live robot commands")
        self._client.SetTimeout(self._timeout_sec)
        self._client.Init()

    def execute_many(self, commands: List[Dict[str, Any]]) -> List[ExecutedCommand]:
        validated = [self._validate_command(command) for command in commands]
        return [self._execute_validated(*command) for command in validated]

    def execute(self, command: Dict[str, Any]) -> ExecutedCommand:
        return self._execute_validated(*self._validate_command(command))

    def _execute_validated(
        self, name: str, args: Dict[str, Any], duration_sec: float
    ) -> ExecutedCommand:
        if self._dry_run:
            return ExecutedCommand(name=name, args=args, duration_sec=duration_sec, code=0)

        code = self._dispatch(name, args, duration_sec)
        return ExecutedCommand(name=name, args=args, duration_sec=duration_sec, code=code)

    def _validate_command(self, command: Mapping[str, Any]) -> tuple[str, Dict[str, Any], float]:
        if not isinstance(command, Mapping):
            raise ValueError("command must be an object")
        name = str(command.get("name", "stop_move")).strip()
        if name not in SUPPORTED_ACTIONS:
            raise ValueError(f"unsupported action: {name}")

        raw_args = command.get("args", {})
        if raw_args is None:
            raw_args = {}
        if not isinstance(raw_args, Mapping):
            raise ValueError("command args must be an object")
        args = dict(raw_args)
        try:
            duration_sec = float(command.get("duration_sec", 0.0) or 0.0)
        except (TypeError, ValueError) as exc:
            raise ValueError("duration_sec must be a finite number") from exc
        if not math.isfinite(duration_sec) or not 0.0 <= duration_sec <= MAX_COMMAND_DURATION_SEC:
            raise ValueError(f"duration_sec must be finite and between 0 and {MAX_COMMAND_DURATION_SEC:g} seconds")

        if name == "speed_level":
            raw_level = args.get("level", 1)
            try:
                numeric_level = float(raw_level)
            except (TypeError, ValueError) as exc:
                raise ValueError("speed level must be -1, 0, or 1") from exc
            if not math.isfinite(numeric_level) or not numeric_level.is_integer():
                raise ValueError("speed level must be -1, 0, or 1")
            level = int(numeric_level)
            if level not in {-1, 0, 1}:
                raise ValueError("speed level must be -1, 0, or 1")
            args = {"level": level}
        elif name == "move":
            if duration_sec <= 0.0:
                raise ValueError("move duration_sec must be > 0")
            values = []
            try:
                for key in ("vx", "vy", "vyaw"):
                    values.append(float(args.get(key, 0.0)))
            except (TypeError, ValueError) as exc:
                raise ValueError("move values must be finite numbers") from exc
            if not all(math.isfinite(value) for value in values):
                raise ValueError("move values must be finite numbers")
            if not all(abs(value) <= limit for value, limit in zip(values, MOVE_LIMITS)):
                raise ValueError("move command exceeds conservative Go2 velocity limits")
            args = dict(zip(("vx", "vy", "vyaw"), values))
        else:
            args = {}

        return name, args, duration_sec

    def _dispatch(self, name: str, args: Dict[str, Any], duration_sec: float) -> int:
        if self._client is None:
            raise RuntimeError("unitree_sdk2py is required for live robot commands")
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
            return self._client.SpeedLevel(args["level"])
        if name == "sit":
            return self._client.Sit()
        if name == "rise_sit":
            return self._client.RiseSit()
        if name == "move":
            try:
                code = self._client.Move(args["vx"], args["vy"], args["vyaw"])
                time.sleep(duration_sec)
                return code
            finally:
                self._client.StopMove()
        raise ValueError(f"unsupported action: {name}")
