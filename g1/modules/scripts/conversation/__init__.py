"""Speech, chat, and LLM-facing tooling."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType


_MODULES = {
    "chat": "modules.scripts.chat",
    "chat_with_FAQs": "modules.scripts.chat_with_FAQs",
    "chat_with_ef_knowledge": "modules.scripts.chat_with_ef_knowledge",
    "chat_with_knowledge": "modules.scripts.chat_with_knowledge",
    "hear_and_repeat": "modules.scripts.hear_and_repeat",
    "llm_client": "modules.scripts.llm_client",
    "naive_VLA": "modules.scripts.naive_VLA",
    "robot_say_once": "modules.scripts.robot_say_once",
    "save_audio_msg_and_reply": "modules.scripts.save_audio_msg_and_reply",
}

__all__ = sorted(_MODULES)


def __getattr__(name: str) -> ModuleType:
    if name not in _MODULES:
        raise AttributeError(name)
    module = import_module(_MODULES[name])
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(list(globals()) + __all__)
