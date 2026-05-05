"""
hand_pose_navigation
====================

Pipeline for guiding the G1 robot hand to a visually-detected target pose
using RGB-D perception and ROS 2 TF.

Table 1 steps:
  1  camera_tf_publisher    — Calibrate: static TF camera_link -> camera_color_optical_frame
  2  target_detector        — Detect target pose in RGB-D (T_camera_object)
  3  detected_pose_publisher— Broadcast object_visible_pose TF frame
  4  arm_fk                 — FK from joint states -> T_base_hand
  5  tf_utils               — lookupTransform base_link <- object_visible_pose
  6  grasp_planner          — Define desired hand pose with offset
  7  arm_ik                 — Solve IK for q_arm_desired
  8  reachability_checker   — Collision / joint-limit check
  9  arm_executor           — Send ll_joint_move command
  10 tracking_loop          — Continuous TF-feedback controller
"""

from .camera_tf_publisher import CameraTFPublisher
from .target_detector import TargetDetector, DetectionResult
from .detected_pose_publisher import DetectedPosePublisher
from .arm_fk import ArmFK
from .tf_utils import TFUtils
from .grasp_planner import GraspPlanner
from .arm_ik import ArmIK
from .reachability_checker import ReachabilityChecker
from .arm_executor import ArmExecutor
from .tracking_loop import TrackingLoop

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
