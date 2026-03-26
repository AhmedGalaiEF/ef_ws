from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import RuntimeConfig
from .ollama_client import OllamaChatClient, OllamaError
from .video_source import Go2VideoSource, VideoFrame


@dataclass
class PerceptionResult:
    prompt: str
    frame_timestamp: float
    data: Dict[str, Any]
    created_at: float = field(default_factory=time.time)


class PerceptionAgent:
    def __init__(self, client: OllamaChatClient, system_prompt: str):
        self._client = client
        self._system_prompt = system_prompt

    def observe(self, frame: VideoFrame, task_prompt: str) -> PerceptionResult:
        user_prompt = (
            f"Planner task:\n{task_prompt}\n\n"
            "Return only the requested JSON object."
        )
        data = self._client.chat_json(
            system_prompt=self._system_prompt,
            user_prompt=user_prompt,
            images=[frame.to_base64()],
        )
        return PerceptionResult(
            prompt=task_prompt,
            frame_timestamp=frame.timestamp,
            data=data,
        )


class PerceptionWorker:
    def __init__(
        self,
        video_source: Go2VideoSource,
        agent: PerceptionAgent,
        period_sec: float,
        initial_prompt: str,
    ):
        self._video_source = video_source
        self._agent = agent
        self._period_sec = period_sec
        self._prompt = initial_prompt
        self._lock = threading.Lock()
        self._latest: Optional[PerceptionResult] = None
        self._latest_error: Optional[str] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def set_prompt(self, prompt: str) -> None:
        with self._lock:
            self._prompt = prompt

    def latest(self) -> Optional[PerceptionResult]:
        with self._lock:
            return self._latest

    def latest_error(self) -> Optional[str]:
        with self._lock:
            return self._latest_error

    def _run(self) -> None:
        while not self._stop.is_set():
            frame = self._video_source.latest()
            if frame is not None:
                with self._lock:
                    prompt = self._prompt
                try:
                    result = self._agent.observe(frame, prompt)
                    with self._lock:
                        self._latest = result
                        self._latest_error = None
                except Exception as exc:  # noqa: BLE001
                    with self._lock:
                        self._latest_error = str(exc)
            self._stop.wait(self._period_sec)


class PlannerAgent:
    def __init__(self, client: OllamaChatClient, system_prompt: str):
        self._client = client
        self._system_prompt = system_prompt

    def plan(
        self,
        latest_perception: Optional[PerceptionResult],
        last_commands: List[Dict[str, Any]],
        now: float,
    ) -> Dict[str, Any]:
        perception_payload = latest_perception.data if latest_perception else None
        prompt = {
            "timestamp": now,
            "latest_perception": perception_payload,
            "latest_perception_prompt": latest_perception.prompt if latest_perception else None,
            "last_commands": last_commands,
            "instruction": (
                "Update the next perception prompt and suggest the next short-horizon action."
            ),
        }
        return self._client.chat_json(
            system_prompt=self._system_prompt,
            user_prompt=json.dumps(prompt, indent=2),
        )


class ActorAgent:
    def __init__(self, client: OllamaChatClient, system_prompt: str, runtime: RuntimeConfig):
        self._client = client
        self._system_prompt = system_prompt
        self._runtime = runtime

    def map_actions(self, planner_output: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "planner_output": planner_output,
            "limits": self._runtime.allowed_actions,
            "instruction": "Produce executable commands only.",
        }
        raw = self._client.chat_json(
            system_prompt=self._system_prompt,
            user_prompt=json.dumps(payload, indent=2),
        )
        return self._sanitize(raw)

    def _sanitize(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        commands = raw.get("commands", [])
        if not isinstance(commands, list) or not commands:
            return {"commands": [{"name": "stop_move", "args": {}, "duration_sec": 0.0}]}

        cleaned: List[Dict[str, Any]] = []
        max_vx = float(self._runtime.allowed_actions["max_vx"])
        max_vy = float(self._runtime.allowed_actions["max_vy"])
        max_vyaw = float(self._runtime.allowed_actions["max_vyaw"])
        max_duration = float(self._runtime.allowed_actions["max_duration_sec"])
        allowed = {
            "stop_move",
            "stand_up",
            "stand_down",
            "balance_stand",
            "recovery",
            "move",
            "hello",
            "stretch",
            "free_walk",
            "sit",
            "rise_sit",
        }

        for command in commands[:2]:
            name = str(command.get("name", "stop_move"))
            if name not in allowed:
                continue
            args = dict(command.get("args", {}) or {})
            duration_sec = max(0.0, min(float(command.get("duration_sec", 0.0) or 0.0), max_duration))
            if name == "move":
                args = {
                    "vx": _clamp(args.get("vx", 0.0), -max_vx, max_vx),
                    "vy": _clamp(args.get("vy", 0.0), -max_vy, max_vy),
                    "vyaw": _clamp(args.get("vyaw", 0.0), -max_vyaw, max_vyaw),
                }
                if duration_sec <= 0.0:
                    duration_sec = 0.75
            else:
                args = {}
                duration_sec = 0.0
            cleaned.append({"name": name, "args": args, "duration_sec": duration_sec})

        if not cleaned:
            cleaned.append({"name": "stop_move", "args": {}, "duration_sec": 0.0})
        return {"commands": cleaned}


def _clamp(value: Any, low: float, high: float) -> float:
    value = float(value)
    if value < low:
        return low
    if value > high:
        return high
    return value


@dataclass
class ControlStepResult:
    planner_output: Dict[str, Any]
    actor_output: Dict[str, Any]
    perception: Optional[PerceptionResult]
    perception_error: Optional[str]


class VLAController:
    def __init__(
        self,
        planner: PlannerAgent,
        actor: ActorAgent,
        perception_worker: PerceptionWorker,
    ):
        self._planner = planner
        self._actor = actor
        self._perception_worker = perception_worker
        self._last_commands: List[Dict[str, Any]] = []

    def step(self) -> ControlStepResult:
        perception = self._perception_worker.latest()
        planner_output = self._planner.plan(
            latest_perception=perception,
            last_commands=self._last_commands,
            now=time.time(),
        )
        perception_prompt = planner_output.get("perception_prompt")
        if isinstance(perception_prompt, str) and perception_prompt.strip():
            self._perception_worker.set_prompt(perception_prompt.strip())

        actor_output = self._actor.map_actions(planner_output)
        self._last_commands = list(actor_output.get("commands", []))

        return ControlStepResult(
            planner_output=planner_output,
            actor_output=actor_output,
            perception=perception,
            perception_error=self._perception_worker.latest_error(),
        )
