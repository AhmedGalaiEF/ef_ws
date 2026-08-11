from __future__ import annotations

from dataclasses import dataclass
import math
from urllib.parse import urlparse


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

    def __post_init__(self) -> None:
        host = str(self.ollama_host).strip().rstrip("/")
        parsed = urlparse(host)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("ollama_host must be an http:// or https:// URL")
        self.ollama_host = host

        for field_name in ("router_model", "thinker_model", "vision_model", "iface", "navbot_command_topic"):
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError(f"{field_name} must not be empty")
            setattr(self, field_name, value)

        if isinstance(self.request_timeout_s, bool):
            raise ValueError("request_timeout_s must be a positive finite number")
        self.request_timeout_s = float(self.request_timeout_s)
        if not math.isfinite(self.request_timeout_s) or self.request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be a positive finite number")

        if isinstance(self.domain_id, bool) or not isinstance(self.domain_id, int):
            raise ValueError("domain_id must be an integer from 0 through 232")
        if not 0 <= self.domain_id <= 232:
            raise ValueError("domain_id must be an integer from 0 through 232")

        if not self.navbot_command_topic.startswith("/") or any(char.isspace() for char in self.navbot_command_topic):
            raise ValueError("navbot_command_topic must be an absolute ROS topic without whitespace")
