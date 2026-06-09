"""Sensor, SLAM, and perception tooling."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType


_MODULES = {
    "dds_discover_topics": "modules.scripts.dds_discover_topics",
    "real_sense": "modules.scripts.real_sense",
    "rgbd_client": "modules.scripts.rgbd_client",
    "slam_points_viewer": "modules.scripts.slam_points_viewer",
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
