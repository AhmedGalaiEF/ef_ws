"""Hand, gripper, and finger control tooling."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType


_MODULES = {
    "close_both_dex3_hands": "modules.scripts.close_both_dex3_hands",
    "dex3_cmd_probe": "modules.scripts.dex3_cmd_probe",
    "dex3_joint_slider": "modules.scripts.dex3_joint_slider",
    "dex3_slow_fingers": "modules.scripts.dex3_slow_fingers",
    "gripping": "modules.scripts.gripping",
    "gripping_cli": "modules.scripts.gripping_cli",
    "inspire_individual_fingers": "modules.scripts.inspire_individual_fingers",
    "left_dex3_finger_joints": "modules.scripts.left_dex3_finger_joints",
    "test_dex3": "modules.scripts.test_dex3",
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
