"""Arm control and pose tooling."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType


_MODULES = {
    "arm_hand_pose_teach_ui": "modules.scripts.arm_hand_pose_teach_ui",
    "arm_joint_nudge_cli": "modules.scripts.arm_joint_nudge_cli",
    "arm_joint_slider": "modules.scripts.arm_joint_slider",
    "arm_pose_capture": "modules.scripts.arm_pose_capture",
    "decrease_both_elbows_90": "modules.scripts.decrease_both_elbows_90",
    "dual_arm_mirror_ui": "modules.scripts.dual_arm_mirror_ui",
    "dual_arm_mirror_ui_with_waist": "modules.scripts.dual_arm_mirror_ui_with_waist",
    "extend_hand_forward": "modules.scripts.extend_hand_forward",
    "pick_and_place": "modules.scripts.pick_and_place",
    "regression_model_arm_motion": "modules.scripts.regression_model_arm_motion",
    "replay_saved_arm_pose": "modules.scripts.replay_saved_arm_pose",
    "robot_joint_slider": "modules.scripts.robot_joint_slider",
    "run_saved_arm_hand_pose_sequence": "modules.scripts.run_saved_arm_hand_pose_sequence",
    "shake_hand_trajectory_cli": "modules.scripts.shake_hand_trajectory_cli",
    "shake_hand_walk_forward": "modules.scripts.shake_hand_walk_forward",
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
