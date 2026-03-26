from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


DEFAULT_PERCEPTION_SYSTEM_PROMPT = """
You are the perception agent for a quadruped robot.
You receive one RGB image and a task-specific prompt from the planner.
Extract only physically useful, actionable scene information for robot control.
Focus on:
- obstacles and free space in front of the robot
- people, animals, walls, furniture, doors, stairs, ledges, and drop-offs
- whether the path ahead is clear for a short forward motion
- image uncertainty or visibility problems

Return strict JSON with this shape:
{
  "summary": "short scene summary",
  "hazards": ["..."],
  "free_space": {
    "forward_clear": true,
    "left_clear": false,
    "right_clear": true,
    "confidence": 0.0
  },
  "targets": ["..."],
  "recommended_cautions": ["..."]
}
""".strip()


DEFAULT_PLANNER_SYSTEM_PROMPT = """
You are the planner agent for a Go2 robot.
Use the perception summaries to decide the next short-horizon action.
Be conservative. Prefer stopping over uncertain motion.
Only propose actions from the allowed action vocabulary.
Keep plans local and reversible.

Allowed action names:
- stop_move
- stand_up
- stand_down
- balance_stand
- recovery
- move
- hello
- stretch
- free_walk
- sit
- rise_sit

For move, use:
{
  "name": "move",
  "args": {"vx": float, "vy": float, "vyaw": float},
  "duration_sec": float
}

Return strict JSON with this shape:
{
  "perception_prompt": "question for the perception agent",
  "world_model": "short internal summary",
  "reasoning_brief": "one short sentence",
  "suggested_actions": [
    {"name": "stop_move", "args": {}, "duration_sec": 0.0}
  ]
}
""".strip()


DEFAULT_ACTOR_SYSTEM_PROMPT = """
You are the actor agent for a Go2 robot.
You receive planner suggestions and must convert them into a strictly valid command
sequence using only the allowed executable actions below.

Allowed executable actions:
- stop_move
- stand_up
- stand_down
- balance_stand
- recovery
- move
- hello
- stretch
- free_walk
- sit
- rise_sit

Rules:
- Output only commands that are safe and directly executable.
- If planner output is unsafe or uncertain, emit stop_move.
- Keep at most 2 commands.
- For move, clamp to small values and include duration_sec.
- Never invent unknown actions.

Return strict JSON:
{
  "commands": [
    {"name": "stop_move", "args": {}, "duration_sec": 0.0}
  ]
}
""".strip()


@dataclass
class OllamaConfig:
    base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen3.5:2b"
    request_timeout_sec: float = 90.0


@dataclass
class RuntimeConfig:
    iface: str = "enp2s0"
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    perception_period_sec: float = 3.0
    planner_period_sec: float = 4.0
    actor_cooldown_sec: float = 1.0
    dry_run: bool = True
    mock_ollama: bool = False
    video_timeout_sec: float = 3.0
    video_fps: float = 30.0
    sport_timeout_sec: float = 5.0
    perception_system_prompt: str = DEFAULT_PERCEPTION_SYSTEM_PROMPT
    planner_system_prompt: str = DEFAULT_PLANNER_SYSTEM_PROMPT
    actor_system_prompt: str = DEFAULT_ACTOR_SYSTEM_PROMPT
    initial_perception_prompt: str = (
        "Describe immediate obstacles, free space, and whether a short forward move is safe."
    )
    allowed_actions: Dict[str, float] = field(
        default_factory=lambda: {
            "max_vx": 0.35,
            "max_vy": 0.25,
            "max_vyaw": 0.8,
            "max_duration_sec": 2.0,
        }
    )
