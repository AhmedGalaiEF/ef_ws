"""ASR runtime state used by /asr and /monitor."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

from .monitor import MonitorEventBus


@dataclass
class AsrRuntimeState:
    microphone_enabled: bool = True
    asr_enabled: bool = True
    audio_to_cognition: bool = True
    audio_to_state: bool = True
    listening: bool = False
    confidence: Optional[float] = None
    partial_transcript: str = ""
    final_transcript: str = ""
    last_accepted_prompt: str = ""
    last_rejected_input: str = ""
    last_accepted_at: Optional[float] = None
    last_rejected_at: Optional[float] = None
    silence_timeout_ms: int = 1200
    input_topic: str = "/audio_msg"
    last_error: str = ""

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        return {
            "microphone_enabled": self.microphone_enabled,
            "asr_enabled": self.asr_enabled,
            "audio_to_cognition": self.audio_to_cognition,
            "audio_to_state": self.audio_to_state,
            "listening": self.listening,
            "confidence": self.confidence,
            "partial_transcript": self.partial_transcript,
            "final_transcript": self.final_transcript,
            "last_accepted_prompt": self.last_accepted_prompt,
            "last_rejected_input": self.last_rejected_input,
            "last_accepted_prompt_age_s": None
            if self.last_accepted_at is None
            else max(0.0, now - self.last_accepted_at),
            "silence_timeout_ms": self.silence_timeout_ms,
            "input_topic": self.input_topic,
            "last_error": self.last_error,
        }


class AsrRuntime:
    def __init__(self, *, monitor: Optional[MonitorEventBus] = None) -> None:
        self.monitor = monitor
        self.state = AsrRuntimeState()

    def update_settings(self, settings: Any) -> None:
        self.state.microphone_enabled = bool(settings.audio.input_enabled)
        self.state.asr_enabled = bool(settings.audio.asr_enabled and settings.asr.enabled)
        self.state.audio_to_cognition = bool(settings.audio.asr_enabled and settings.asr.enabled)
        self.state.audio_to_state = bool(settings.audio.audio_to_state_enabled)
        self.state.silence_timeout_ms = int(settings.asr.silence_timeout_ms)

    def started(self) -> None:
        self.state.listening = True
        self._emit("audio_input_started", "ASR input listening")

    def stopped(self) -> None:
        self.state.listening = False
        self._emit("audio_input_stopped", "ASR input stopped")

    def partial(self, text: str, confidence: Optional[float] = None) -> None:
        self.state.partial_transcript = text
        self.state.confidence = confidence
        self._emit("asr_partial", text[:120], confidence=confidence)

    def final(self, text: str, confidence: Optional[float] = None) -> bool:
        self.state.final_transcript = text
        self.state.confidence = confidence
        self._emit("asr_final", text[:120], confidence=confidence)
        threshold = 0.0 if confidence is None else float(confidence)
        return confidence is None or threshold >= 0.0

    def accepted(self, text: str, confidence: Optional[float] = None) -> None:
        self.state.last_accepted_prompt = text
        self.state.last_accepted_at = time.time()
        self.state.confidence = confidence
        self._emit("user_audio_prompt_created", text[:120], confidence=confidence, input_source="audio")

    def rejected(self, text: str, reason: str, confidence: Optional[float] = None) -> None:
        self.state.last_rejected_input = text
        self.state.last_rejected_at = time.time()
        self.state.confidence = confidence
        self._emit("asr_rejected", reason, confidence=confidence, text=text[:120])

    def error(self, message: str) -> None:
        self.state.last_error = message

    def snapshot(self) -> dict[str, Any]:
        return self.state.snapshot()

    def _emit(self, event: str, summary: str, **metadata: Any) -> None:
        if self.monitor is not None:
            self.monitor.emit("audio", event, summary, metadata=metadata)
