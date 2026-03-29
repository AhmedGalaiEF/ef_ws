from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from ollama_vla.config import (
    DEFAULT_ACTOR_SYSTEM_PROMPT,
    DEFAULT_PERCEPTION_SYSTEM_PROMPT,
    DEFAULT_PLANNER_SYSTEM_PROMPT,
)


@dataclass
class HuggingFaceConfig:
    api_url: str = "https://router.huggingface.co/v1/chat/completions"
    model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    api_token: str = ""
    request_timeout_sec: float = 90.0
    temperature: float = 0.1


@dataclass
class RuntimeConfig:
    iface: str = "enp2s0"
    hf: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    perception_period_sec: float = 3.0
    planner_period_sec: float = 4.0
    actor_cooldown_sec: float = 1.0
    dry_run: bool = True
    mock_hf: bool = False
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
