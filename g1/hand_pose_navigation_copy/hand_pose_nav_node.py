"""
hand_pose_nav_node.py
======================
ROS 2 node that orchestrates all 10 pipeline steps.

Run:
    python3 hand_pose_nav_node.py              # interactive demo
    ros2 run hand_pose_navigation hand_pose_nav_node

ROS 2 params (all overrideable via --ros-args -p):
    arm             right | left
    detection_method  aruco | color | center
    aruco_id        0
    marker_size_m   0.05
    standoff_m      0.08
    rate_hz         10.0
    timeout_s       30.0
    ik_solver       dls | scipy | pin
    camera_frame    camera_color_optical_frame
    base_frame      base_link
    object_frame    object_visible_pose
"""
from __future__ import annotations

import sys
import os
from typing import Dict, Optional

# ── path setup ─────────────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULES = os.path.join(_DIR, "..", "modules")
if _MODULES not in sys.path:
    sys.path.insert(0, _MODULES)

import rclpy
from rclpy.node import Node

try:
    from .camera_tf_publisher import CameraTFPublisher      # Step 1
    from .target_detector import TargetDetector              # Step 2
    from .detected_pose_publisher import DetectedPosePublisher  # Step 3
    from .arm_fk import ArmFK                               # Step 4
    from .tf_utils import TFUtils                           # Step 5
    from .grasp_planner import GraspPlanner                 # Step 6
    from .arm_ik import ArmIK                               # Step 7
    from .reachability_checker import ReachabilityChecker   # Step 8
    from .arm_executor import ArmExecutor                   # Step 9
    from .tracking_loop import TrackingLoop                 # Step 10
except ImportError:
    from camera_tf_publisher import CameraTFPublisher      # type: ignore
    from target_detector import TargetDetector              # type: ignore
    from detected_pose_publisher import DetectedPosePublisher  # type: ignore
    from arm_fk import ArmFK                               # type: ignore
    from tf_utils import TFUtils                           # type: ignore
    from grasp_planner import GraspPlanner                 # type: ignore
    from arm_ik import ArmIK                               # type: ignore
    from reachability_checker import ReachabilityChecker   # type: ignore
    from tracking_loop import TrackingLoop                 # type: ignore

try:
    from sdk_client import Robot
    _ROBOT_AVAILABLE = True
except ImportError:
    _ROBOT_AVAILABLE = False
    Robot = None


class HandPoseNavNode(Node):
    """
    Main orchestrator node.  Spins up all sub-components and starts the
    tracking loop.

    The node runs two ROS 2 sub-nodes internally (CameraTFPublisher and
    DetectedPosePublisher) so that their TF broadcasts are part of the
    same executor context.
    """

    def __init__(self, config: Optional[Dict] = None, robot=None) -> None:
        super().__init__("hand_pose_nav_node")
        config = config or {}

        # ── Declare all parameters ────────────────────────────────────
        self.declare_parameter("arm", config.get("arm", "right"))
        self.declare_parameter("detection_method", config.get("detection_method", "aruco"))
        self.declare_parameter("aruco_id", config.get("aruco_id", 0))
        self.declare_parameter("marker_size_m", config.get("marker_size_m", 0.05))
        self.declare_parameter("standoff_m", config.get("standoff_m", 0.08))
        self.declare_parameter("rate_hz", config.get("rate_hz", 10.0))
        self.declare_parameter("timeout_s", config.get("timeout_s", 30.0))
        self.declare_parameter("ik_solver", config.get("ik_solver", "dls"))
        self.declare_parameter("camera_frame", config.get("camera_frame", "camera_color_optical_frame"))
        self.declare_parameter("base_frame", config.get("base_frame", "base_link"))
        self.declare_parameter("object_frame", config.get("object_frame", "object_visible_pose"))
        self.declare_parameter("iface", config.get("iface", "eth0"))
        self.declare_parameter("domain_id", config.get("domain_id", 0))
<<<<<<< HEAD
        self.declare_parameter("rgbd_host", config.get("rgbd_host", "10.34.0.83"))
        self.declare_parameter("rgbd_port", config.get("rgbd_port", 5555))
        self.declare_parameter("rgbd_topic", config.get("rgbd_topic", ""))
=======
>>>>>>> 8cefa02351752afd10164323bae8d90d9d7ce454

        arm               = self.get_parameter("arm").value
        detection_method  = self.get_parameter("detection_method").value
        aruco_id          = self.get_parameter("aruco_id").value
        marker_size_m     = self.get_parameter("marker_size_m").value
        standoff_m        = self.get_parameter("standoff_m").value
        rate_hz           = self.get_parameter("rate_hz").value
        timeout_s         = self.get_parameter("timeout_s").value
        ik_solver         = self.get_parameter("ik_solver").value
        self._config = {
            "arm": arm,
            "detection_method": detection_method,
            "aruco_id": aruco_id,
            "marker_size_m": marker_size_m,
            "standoff_m": standoff_m,
            "rate_hz": rate_hz,
            "timeout_s": timeout_s,
            "ik_solver": ik_solver,
            "iface": self.get_parameter("iface").value,
            "domain_id": self.get_parameter("domain_id").value,
<<<<<<< HEAD
            "rgbd_host": self.get_parameter("rgbd_host").value,
            "rgbd_port": self.get_parameter("rgbd_port").value,
            "rgbd_topic": self.get_parameter("rgbd_topic").value,
=======
>>>>>>> 8cefa02351752afd10164323bae8d90d9d7ce454
            "mock": bool(config.get("mock", False)),
            "sdk_preinit_error": config.get("sdk_preinit_error", ""),
        }

        self.get_logger().info(f"[HPN] Starting hand_pose_nav — arm={arm}, method={detection_method}")

        # ── Step 1: camera TF ─────────────────────────────────────────
        self._cam_tf_pub = CameraTFPublisher()
        self.get_logger().info("[Step 1] Camera TF publisher created.")

        # ── Step 2: target detector ───────────────────────────────────
        self._detector = TargetDetector(
            method=detection_method,
            aruco_id=aruco_id,
            marker_size_m=marker_size_m,
        )
        self.get_logger().info(f"[Step 2] TargetDetector({detection_method}) created.")

        # ── Step 3: detected pose publisher ──────────────────────────
        self._pose_pub = DetectedPosePublisher()
        self.get_logger().info("[Step 3] DetectedPosePublisher created.")

        # ── Step 4: FK ────────────────────────────────────────────────
        self._fk = ArmFK(arm=arm)
        self.get_logger().info("[Step 4] ArmFK created.")

        # ── Step 5: TF utils ─────────────────────────────────────────
        self._tf_utils = TFUtils(node=self)
        self.get_logger().info("[Step 5] TFUtils created.")

        # ── Step 6: grasp planner ────────────────────────────────────
        self._grasp_planner = GraspPlanner(arm=arm, standoff_m=standoff_m)
        self.get_logger().info("[Step 6] GraspPlanner created.")

        # ── Step 7: IK solver ─────────────────────────────────────────
        self._ik = ArmIK(arm=arm, solver=ik_solver)
        self.get_logger().info(f"[Step 7] ArmIK({ik_solver}) created.")

        # ── Step 8: reachability checker ──────────────────────────────
        self._checker = ReachabilityChecker(arm=arm)
        self.get_logger().info("[Step 8] ReachabilityChecker created.")

        # ── Robot SDK + Step 9 / 10 ───────────────────────────────────
        if robot is not None:
            self._robot = robot
            self._robot_mode = "sdk"
            self._sdk_error = ""
            self.get_logger().info("[SDK] Using pre-connected robot.")
        elif _ROBOT_AVAILABLE and not config.get("mock", False):
            iface     = self.get_parameter("iface").value
            domain_id = self.get_parameter("domain_id").value
<<<<<<< HEAD
            rgbd_host = self.get_parameter("rgbd_host").value
            rgbd_port = self.get_parameter("rgbd_port").value
            rgbd_topic = self.get_parameter("rgbd_topic").value
            try:
                self._robot = Robot(
                    iface=iface,
                    domain_id=domain_id,
                    rgbd_host=rgbd_host,
                    rgbd_port=rgbd_port,
                    rgbd_topic=rgbd_topic,
                )
=======
            try:
                self._robot = Robot(iface=iface, domain_id=domain_id)
>>>>>>> 8cefa02351752afd10164323bae8d90d9d7ce454
                self._robot.start_sensors()
                self._robot_mode = "sdk"
                self._sdk_error = ""
                self.get_logger().info("[SDK] Robot connected, sensors started.")
            except Exception as exc:
                self._robot = _MockRobot()
                self._robot_mode = "mock"
                self._sdk_error = repr(exc)
                self.get_logger().error(
                    f"[SDK] Robot init failed - using mock robot: {exc!r}"
                )
        else:
            self._robot = _MockRobot()
            self._robot_mode = "mock"
            self._sdk_error = config.get("sdk_preinit_error", "")
            if config.get("mock", False):
                self.get_logger().warn("[SDK] mock mode requested - using mock robot.")
            else:
                self.get_logger().warn("[SDK] sdk_client not available - using mock robot.")

        self._executor_obj = ArmExecutor(self._robot, arm=arm)
        self.get_logger().info("[Step 9] ArmExecutor created.")

        self._tracking_loop = TrackingLoop(
            robot=self._robot,
            detector=self._detector,
            pose_publisher=self._pose_pub,
            tf_utils=self._tf_utils,
            fk=self._fk,
            grasp_planner=self._grasp_planner,
            ik=self._ik,
            checker=self._checker,
            executor=self._executor_obj,
            arm=arm,
            rate_hz=rate_hz,
            timeout_s=timeout_s,
            on_converge=self._on_converge,
        )
        self.get_logger().info("[Step 10] TrackingLoop created.")

        # Start the loop in a background thread so ROS can spin
        self._status = self._tracking_loop.start(blocking=False)
        self.get_logger().info("[HPN] Tracking loop started.")

    # ------------------------------------------------------------------
    def _on_converge(self) -> None:
        self.get_logger().info("[HPN] Hand converged to target pose!")

    def status_snapshot(self) -> Dict:
        status = self._tracking_loop.status.to_dict()
        status.update({
            "node": self.get_name(),
            "robot_mode": self._robot_mode,
            "sdk_error": self._sdk_error,
            "config": dict(self._config),
        })
        return status

    def destroy_node(self) -> None:
        self._tracking_loop.stop()
        super().destroy_node()


# ---------------------------------------------------------------------------
# Mock robot for testing without hardware
# ---------------------------------------------------------------------------

class _MockRobot:
    def get_rgbd(self, timeout=2.0):
        import numpy as np
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = np.full((480, 640), 0.8, dtype=np.float32)
        return {"rgb_bgr": rgb, "depth_m": depth}

    def get_joint_states(self):
        return {"joints": {f"j{i}": {"index": i, "position": 0.0} for i in range(30)}}

    def move_upper_body_joint(self, **kw):
        pass


# ---------------------------------------------------------------------------

def main(args=None) -> None:
    rclpy.init(args=args)
    node = HandPoseNavNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
