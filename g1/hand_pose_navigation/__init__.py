"""
hand_pose_navigation
====================

Pipeline for guiding the G1 robot hand to a visually-detected target pose
using RGB-D perception and ROS 2 TF.

This package intentionally avoids importing the full perception / ROS stack at
module import time. Several callers only need lightweight kinematics modules
such as `arm_fk` or `arm_ik`, and eagerly importing the rest of the package can
pull in native dependencies that interfere with unrelated runtime subsystems.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "CameraTFPublisher",
    "TargetDetector",
    "DetectionResult",
    "DetectedPosePublisher",
    "ArmFK",
    "TFUtils",
    "GraspPlanner",
    "ArmIK",
    "ReachabilityChecker",
    "ArmExecutor",
    "TrackingLoop",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "CameraTFPublisher": (".camera_tf_publisher", "CameraTFPublisher"),
    "TargetDetector": (".target_detector", "TargetDetector"),
    "DetectionResult": (".target_detector", "DetectionResult"),
    "DetectedPosePublisher": (".detected_pose_publisher", "DetectedPosePublisher"),
    "ArmFK": (".arm_fk", "ArmFK"),
    "TFUtils": (".tf_utils", "TFUtils"),
    "GraspPlanner": (".grasp_planner", "GraspPlanner"),
    "ArmIK": (".arm_ik", "ArmIK"),
    "ReachabilityChecker": (".reachability_checker", "ReachabilityChecker"),
    "ArmExecutor": (".arm_executor", "ArmExecutor"),
    "TrackingLoop": (".tracking_loop", "TrackingLoop"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
