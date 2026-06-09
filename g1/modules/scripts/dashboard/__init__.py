"""Dashboard, web UI, and operator panel tooling."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType


_MODULES = {
    "dash_robot_control": "modules.scripts.dash_robot_control",
    "dashboard_2504_working_but_laggy": "modules.scripts.dashboard_2504_working_but_laggy",
    "mode_control": "modules.scripts.mode_control",
    "sensor_monitor_pyqt": "modules.scripts.sensor_monitor_pyqt",
    "slam_web_app": "modules.scripts.slam_web_app",
    "slam_webapp_with_tasks": "modules.scripts.slam_webapp_with_tasks",
    "upper_body_control_dash": "modules.scripts.upper_body_control_dash",
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
