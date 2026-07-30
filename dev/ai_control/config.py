from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AIConfig:
    """Runtime configuration for the Ollama-backed control stack.

    Defaults match the currently provisioned robot image. Swap the tags for
    whatever is actually pulled on the target machine (`ollama pull <tag>`).
    """

    ollama_host: str = "http://localhost:11434"

    # Small/fast model: classifies intent before any heavy reasoning runs.
    router_model: str = "qwen2.5:0.5b"

    # Larger reasoning model: drafts the response and any tool call.
    thinker_model: str = "qwen3.5:9b"

    # Vision-language model: only invoked when a camera frame is needed.
    vision_model: str = "qwen2.5vl:7b"

    request_timeout_s: float = 60.0

    # Robot connection (only used when --robot is passed to the CLI).
    iface: str = "eth0"
    domain_id: int = 0
    navbot_command_topic: str = "/model_api/navbot_command"
