"""System, hardware, and one-shot utility tooling."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType


_MODULES = {
    "altegro_client": "modules.scripts.altegro_client",
    "altegro_server": "modules.scripts.altegro_server",
    "hue_wheel_headlight": "modules.scripts.hue_wheel_headlight",
    "low_level_commands": "modules.scripts.low_level_commands",
    "persistent_headlight_control": "modules.scripts.persistent_headlight_control",
    "piper_test": "modules.scripts.piper_test",
    "robot_headlight_once": "modules.scripts.robot_headlight_once",
    "usage": "modules.scripts.usage",
    "walk_forward_5s_turn_90": "modules.scripts.walk_forward_5s_turn_90",
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
