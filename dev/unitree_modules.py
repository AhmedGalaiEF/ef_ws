###################################
# Minimal imports
###################################

import importlib
import json
import math
import threading
import time, os, sys

import pandas as pd

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
from unitree_sdk2py.g1.loco.g1_loco_api import (
    ROBOT_API_ID_LOCO_GET_FSM_ID,
    ROBOT_API_ID_LOCO_GET_FSM_MODE,
)
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient


###################################
# Constants
###################################

HL_ARM_ACTIONS = {
    "release arm": 99, "two-hand kiss": 11, "left kiss": 12, "right kiss": 13,
    "hands up": 15, "clap": 17, "high five": 18, "hug": 19, "heart": 20,
    "right heart": 21, "reject": 22, "right hand up": 23, "x-ray": 24,
    "face wave": 25, "high wave": 26, "shake hand": 27,
}
HL_ARM_ALIASES = {
    "release": "release arm", "two hand kiss": "two-hand kiss",
    "left hand kiss": "left kiss", "right hand kiss": "right kiss",
    "xray": "x-ray", "x ray": "x-ray",
}
_ARM_RELEASE_DELAY_S = 2.0


def _ensure_g1_modules():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "g1", "modules"))
    if path not in sys.path:
        sys.path.insert(0, path)


###################################
# G1 Class
###################################


class G1():
    def __init__(self, iface="eth0", domain_id=0, timeout=10.0):
        self.iface = iface
        self.domain_id = int(domain_id)
        ChannelFactoryInitialize(self.domain_id, self.iface)
        self.loco_client = LocoClient()
        self.loco_client.SetTimeout(float(timeout))
        self.loco_client.Init()
        self.ms_client = MotionSwitcherClient()
        self.ms_client.SetTimeout(float(timeout))
        self.ms_client.Init()

        self._lock = threading.Lock()
        self._sport = None
        self._lowstate = None
        self._lidar_cloud = None
        self._odom = None

        self._video_client = None
        self._arm_pub = None
        self._arm_cmd = None
        self._arm_crc = None
        self._arm_action_client = None
        self._audio = None
        self._hands = {}
        self._slam_client = None
        self._slam_info_sub = None
        self._path_points = []
        self.slam_is_running = False

        self._start_sensors()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _start_sensors(self):
        try:
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
            sub = ChannelSubscriber("rt/odommodestate", SportModeState_)
            sub.Init(self._on_sport, 10)
        except Exception:
            pass

        lowstate_type = None
        for path in ("unitree_sdk2py.idl.unitree_hg.msg.dds_", "unitree_sdk2py.idl.unitree_go.msg.dds_"):
            try:
                mod = importlib.import_module(path)
                if hasattr(mod, "LowState_"):
                    lowstate_type = mod.LowState_
                    break
            except Exception:
                pass
        if lowstate_type:
            sub = ChannelSubscriber("rt/lowstate", lowstate_type)
            sub.Init(self._on_lowstate, 10)

        try:
            from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
            sub = ChannelSubscriber("rt/utlidar/cloud_deskewed", PointCloud2_)
            sub.Init(self._on_lidar, 10)
        except Exception:
            pass

        try:
            from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_
            sub = ChannelSubscriber("rt/odom", Odometry_)
            sub.Init(self._on_odom, 10)
        except Exception:
            pass

    def _on_sport(self, msg):
        with self._lock:
            self._sport = msg

    def _on_lowstate(self, msg):
        with self._lock:
            self._lowstate = msg

    def _on_lidar(self, msg):
        with self._lock:
            self._lidar_cloud = msg

    def _on_odom(self, msg):
        with self._lock:
            self._odom = msg

    def _get_video_client(self):
        if self._video_client is None:
            for path in ("unitree_sdk2py.g1.video.video_client", "unitree_sdk2py.go2.video.video_client"):
                try:
                    cls = importlib.import_module(path).VideoClient
                    self._video_client = cls()
                    self._video_client.SetTimeout(2.0)
                    self._video_client.Init()
                    break
                except Exception:
                    pass
        return self._video_client

    def _get_arm_sdk(self):
        if self._arm_pub is None:
            from unitree_sdk2py.core.channel import ChannelPublisher
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
            from unitree_sdk2py.utils.crc import CRC
            self._arm_pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
            self._arm_pub.Init()
            self._arm_cmd = unitree_hg_msg_dds__LowCmd_()
            self._arm_crc = CRC()
        return self._arm_pub, self._arm_cmd, self._arm_crc

    def _get_arm_action_client(self):
        if self._arm_action_client is None:
            from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient
            self._arm_action_client = G1ArmActionClient()
            self._arm_action_client.SetTimeout(10.0)
            self._arm_action_client.Init()
        return self._arm_action_client

    def _get_audio(self):
        if self._audio is None:
            _ensure_g1_modules()
            from sdk_audio import RobotAudio
            self._audio = RobotAudio()
        return self._audio

    def _get_hand(self, hand="right"):
        side = str(hand).strip().lower()
        if side not in self._hands:
            _ensure_g1_modules()
            from sdk_hand import Dex3HandController
            self._hands[side] = Dex3HandController(side, iface=self.iface, domain_id=self.domain_id)
        return self._hands[side]

    def _get_slam_client(self):
        if self._slam_client is None:
            _ensure_g1_modules()
            from sdk_slam import SlamOperateClient
            self._slam_client = SlamOperateClient()
            self._slam_client.Init()
            self._slam_client.SetTimeout(10.0)
        return self._slam_client

    def _get_slam_info_sub(self):
        if self._slam_info_sub is None:
            _ensure_g1_modules()
            from sdk_slam import SlamInfoSubscriber
            self._slam_info_sub = SlamInfoSubscriber("rt/slam_info", "rt/slam_key_info")
            self._slam_info_sub.start()
        return self._slam_info_sub

    def _rpc(self, api_id):
        try:
            code, data = self.loco_client._Call(api_id, "{}")
            return int(json.loads(data).get("data")) if code == 0 and data else None
        except Exception:
            return None

    def _run_pose_nav(self, x, y, yaw=0.0):
        qz = math.sin(float(yaw) * 0.5)
        qw = math.cos(float(yaw) * 0.5)
        return int(self._get_slam_client().pose_nav(float(x), float(y), 0.0, 0.0, 0.0, qz, qw, mode=1).code)

    # ------------------------------------------------------------------
    # Motion switcher
    # ------------------------------------------------------------------

    def toggle_service(self):
        code, data = self.ms_client.CheckMode()
        name = (data or {}).get("name", "")
        if name:
            return self.ms_client.ReleaseMode()
        return self.ms_client.SelectMode("ai")

    # ------------------------------------------------------------------
    # State snapshot
    # ------------------------------------------------------------------

    def get_state(self, df=True):
        fsm = self.get_fsm()
        imu = self.get_sensors_imu()
        odom = self.get_odomstate()
        joints = self.get_joint_states()

        row = {"fsm_id": fsm.get("id"), "fsm_mode": fsm.get("mode")}
        if odom:
            row.update({"odom_x": odom[0], "odom_y": odom[1], "odom_yaw": odom[2]})
        if imu:
            row.update({
                "imu_roll": imu.get("roll"), "imu_pitch": imu.get("pitch"), "imu_yaw": imu.get("yaw"),
                "gyro_x": imu["gyro"][0], "gyro_y": imu["gyro"][1], "gyro_z": imu["gyro"][2],
                "acc_x": imu["acc"][0], "acc_y": imu["acc"][1], "acc_z": imu["acc"][2],
            })
        if joints:
            for idx, vals in joints.items():
                row.update({f"j{idx}_q": vals["q"], f"j{idx}_dq": vals["dq"], f"j{idx}_tau": vals["tau"]})

        return pd.DataFrame([row]) if df else row

    def get_robot_state(self):
        return {
            "fsm": self.get_fsm(),
            "mode": self.get_mode(),
            "gait": self.get_gait(),
            "body_height": self.get_body_height(),
            "position": self.get_position(),
            "velocity": self.get_velocity(),
            "yaw": self.get_yaw(),
            "is_moving": self.is_moving(),
            "imu": self.get_imu(),
            "odom_pose": self.get_odomstate(),
            "slam_is_running": bool(self.slam_is_running),
            "queued_path_points": len(self._path_points),
        }

    # ------------------------------------------------------------------
    # FSM
    # ------------------------------------------------------------------

    def get_fsm(self):
        return {"id": self._rpc(ROBOT_API_ID_LOCO_GET_FSM_ID), "mode": self._rpc(ROBOT_API_ID_LOCO_GET_FSM_MODE)}

    def switch_fsm(self, FSM_ID):
        return self.loco_client.SetFsmId(int(FSM_ID))

    def fsm_0_zt(self):
        if hasattr(self.loco_client, "ZeroTorque"):
            self.loco_client.ZeroTorque()
        elif hasattr(self.loco_client, "SetFsmId"):
            self.loco_client.SetFsmId(0)

    def fsm_1_damp(self):
        if hasattr(self.loco_client, "Damp"):
            self.loco_client.Damp()
        elif hasattr(self.loco_client, "SetFsmId"):
            self.loco_client.SetFsmId(1)

    def fsm_2_airborne(self):
        self.loco_client.SetFsmId(2)

    def fsm_2_squat(self):
        self.fsm_2_airborne()

    # ------------------------------------------------------------------
    # Locomotion
    # ------------------------------------------------------------------

    def loco_move(self, vx, vy, vyaw):
        return self.loco_client.Move(float(vx), float(vy), float(vyaw), continous_move=True)

    def move_for(self, duration, vx=0.0, vy=0.0, vyaw=0.0):
        result = self.loco_move(vx, vy, vyaw)
        try:
            time.sleep(float(duration))
        finally:
            self.stop()
        return result

    def stop_moving(self):
        if hasattr(self.loco_client, "StopMove"):
            self.loco_client.StopMove()
        else:
            self.loco_client.Move(0.0, 0.0, 0.0, continous_move=False)

    def stop(self):
        self.stop_moving()

    def zero_torque(self):
        self.fsm_0_zt()

    def damp(self):
        self.fsm_1_damp()

    def walk_mode(self):
        self.loco_client.SetFsmId(501)

    def run_mode(self):
        self.loco_client.SetFsmId(802)

    def dev_mode(self):
        if hasattr(self.loco_client, "SetGaitType"):
            self.loco_client.SetGaitType(3)
        elif hasattr(self.loco_client, "SetBalanceMode"):
            self.loco_client.SetBalanceMode(3)

    # ------------------------------------------------------------------
    # Sport state queries
    # ------------------------------------------------------------------

    def get_mode(self):
        with self._lock:
            msg = self._sport
        if msg is None:
            return None
        try:
            return int(msg.mode)
        except Exception:
            return None

    def get_gait(self):
        with self._lock:
            msg = self._sport
        if msg is None:
            return None
        for key in ("gait_type", "gaitType", "gait"):
            try:
                return int(getattr(msg, key))
            except Exception:
                pass
        return None

    def get_body_height(self):
        with self._lock:
            msg = self._sport
        if msg is None:
            return None
        for key in ("body_height", "bodyHeight", "stand_height", "standHeight"):
            try:
                return float(getattr(msg, key))
            except Exception:
                pass
        return None

    def get_position(self):
        with self._lock:
            msg = self._sport
        if msg is None:
            return None
        for key in ("position", "pos", "position_w"):
            try:
                v = getattr(msg, key)
                return (float(v[0]), float(v[1]), float(v[2]))
            except Exception:
                pass
        return None

    def get_velocity(self):
        with self._lock:
            msg = self._sport
        if msg is None:
            return None
        for key in ("velocity", "vel", "velocity_w"):
            try:
                v = getattr(msg, key)
                return (float(v[0]), float(v[1]), float(v[2]))
            except Exception:
                pass
        return None

    def get_yaw(self):
        imu = self.get_imu()
        return float(imu["rpy"][2]) if imu else None

    def is_moving(self, linear_eps=0.03, yaw_eps=0.08):
        vel = self.get_velocity()
        if vel is None:
            return False
        return math.hypot(vel[0], vel[1]) > linear_eps or abs(vel[2]) > yaw_eps

    # ------------------------------------------------------------------
    # IMU / sensors
    # ------------------------------------------------------------------

    def get_imu(self):
        with self._lock:
            msg = self._sport
        if msg is None:
            return None
        imu = getattr(msg, "imu_state", None)
        if imu is None:
            return None
        rpy = gyro = acc = quat = temp = None
        try:
            rpy = tuple(float(imu.rpy[i]) for i in range(3))
        except Exception:
            rpy = (0.0, 0.0, 0.0)
        try:
            gyro = tuple(float(imu.gyroscope[i]) for i in range(3))
        except Exception:
            pass
        try:
            acc = tuple(float(imu.accelerometer[i]) for i in range(3))
        except Exception:
            pass
        try:
            quat = tuple(float(imu.quaternion[i]) for i in range(4))
        except Exception:
            pass
        try:
            temp = float(imu.temperature)
        except Exception:
            pass
        return {"rpy": rpy, "gyro": gyro, "acc": acc, "quat": quat, "temp": temp}

    def get_sensors_imu(self):
        imu = self.get_imu()
        if imu is None:
            return None
        rpy = imu["rpy"]
        return {"roll": rpy[0], "pitch": rpy[1], "yaw": rpy[2], "gyro": imu["gyro"], "acc": imu["acc"]}

    def get_joint_states(self):
        with self._lock:
            msg = self._lowstate
        if msg is None:
            return None
        joints = {}
        for i, motor in enumerate(msg.motor_state or []):
            try:
                joints[i] = {"q": float(motor.q), "dq": float(motor.dq), "tau": float(motor.tau_est)}
            except Exception:
                joints[i] = {"q": None, "dq": None, "tau": None}
        return joints

    def get_odomstate(self):
        with self._lock:
            msg = self._odom or self._sport
        if msg is None:
            return None
        try:
            pos = msg.pose.pose.position
            ori = msg.pose.pose.orientation
            x, y = float(pos.x), float(pos.y)
            qx, qy, qz, qw = float(ori.x), float(ori.y), float(ori.z), float(ori.w)
        except Exception:
            try:
                pos = msg.position()
                q = msg.imu_state().quaternion()
                x, y = float(pos[0]), float(pos[1])
                qw, qx, qy, qz = float(q[0]), float(q[1]), float(q[2]), float(q[3])
            except Exception:
                return None
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        return (x, y, float(yaw))

    def get_odom_pose(self):
        return self.get_odomstate()

    # ------------------------------------------------------------------
    # Camera / RGBD
    # ------------------------------------------------------------------

    def get_camera_image_jpeg(self):
        code, data = self._get_video_client().GetImageSample()
        if int(code) != 0:
            raise RuntimeError(f"GetImageSample failed: {code}")
        return bytes(data)

    def get_sensors_rgbd(self):
        return self.get_camera_image_jpeg()

    def get_camera_frame_bgr(self):
        import cv2, numpy as np
        return cv2.imdecode(np.frombuffer(self.get_camera_image_jpeg(), dtype=np.uint8), cv2.IMREAD_COLOR)

    def get_camera_frame_rgb(self):
        import cv2
        return cv2.cvtColor(self.get_camera_frame_bgr(), cv2.COLOR_BGR2RGB)

    def get_rgbd(self):
        import cv2
        bgr = self.get_camera_frame_bgr()
        jpeg = self.get_camera_image_jpeg()
        return {"rgb_bgr": bgr, "rgb_rgb": cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), "jpeg": jpeg}

    # ------------------------------------------------------------------
    # Lidar
    # ------------------------------------------------------------------

    def get_sensors_lidar(self):
        with self._lock:
            msg = self._lidar_cloud
        if msg is None:
            return None
        try:
            import numpy as np
            w, h, step = int(msg.width), int(msg.height), int(msg.point_step)
            raw = bytes(msg.data)
            fields = {f.name.lower(): f for f in msg.fields}
            dtype = np.dtype({
                "names": ["x", "y", "z"], "formats": ["<f4", "<f4", "<f4"],
                "offsets": [fields["x"].offset, fields["y"].offset, fields["z"].offset],
                "itemsize": step,
            })
            arr = np.frombuffer(raw, dtype=dtype, count=min(w * h, len(raw) // step))
            pts = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype("float32")
            return pts[np.isfinite(pts).all(axis=1)]
        except Exception:
            return None

    def get_lidar_points(self, max_points=20000):
        pts = self.get_sensors_lidar()
        if pts is None:
            return []
        if max_points and len(pts) > max_points:
            import numpy as np
            pts = pts[np.linspace(0, len(pts) - 1, max_points, dtype=np.int64)]
        return [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in pts]

    # ------------------------------------------------------------------
    # Arm SDK (direct joint control)
    # ------------------------------------------------------------------

    def move_joint(self, q, dq, kp, kd, tau):
        pub, cmd, crc = self._get_arm_sdk()
        targets = q if isinstance(q, dict) else {i: float(v) for i, v in enumerate(q)}
        for idx, pos in targets.items():
            mc = cmd.motor_cmd[int(idx)]
            mc.mode = 1
            mc.q = float(pos)
            mc.dq = float(dq[idx] if isinstance(dq, dict) else dq)
            mc.kp = float(kp[idx] if isinstance(kp, dict) else kp)
            mc.kd = float(kd[idx] if isinstance(kd, dict) else kd)
            mc.tau = float(tau[idx] if isinstance(tau, dict) else tau)
        cmd.crc = crc.Crc(cmd)
        pub.Write(cmd)

    # ------------------------------------------------------------------
    # Arm action client (high-level gestures)
    # ------------------------------------------------------------------

    def execute_arm_action(self, action, release_after_s=None):
        if isinstance(action, str):
            key = " ".join(str(action).strip().lower().replace("_", " ").split())
            key = HL_ARM_ALIASES.get(key, key)
            action_id = HL_ARM_ACTIONS[key]
        else:
            action_id = int(action)
        client = self._get_arm_action_client()
        code = int(client.ExecuteAction(action_id))
        if release_after_s is not None:
            time.sleep(max(0.0, float(release_after_s)))
            client.ExecuteAction(HL_ARM_ACTIONS["release arm"])
        return code

    def release_arm(self):
        return self.execute_arm_action("release arm")

    def shake_hand(self, release_after_s=_ARM_RELEASE_DELAY_S):
        return self.execute_arm_action("shake hand", release_after_s=release_after_s)

    def high_five(self, release_after_s=_ARM_RELEASE_DELAY_S):
        return self.execute_arm_action("high five", release_after_s=release_after_s)

    def hug(self, release_after_s=_ARM_RELEASE_DELAY_S):
        return self.execute_arm_action("hug", release_after_s=release_after_s)

    def high_wave(self):
        return self.execute_arm_action("high wave")

    def clap(self):
        return self.execute_arm_action("clap")

    def face_wave(self):
        return self.execute_arm_action("face wave")

    def left_kiss(self):
        return self.execute_arm_action("left kiss")

    def right_kiss(self):
        return self.execute_arm_action("right kiss")

    def two_hand_kiss(self):
        return self.execute_arm_action("two-hand kiss")

    def heart(self, release_after_s=_ARM_RELEASE_DELAY_S):
        return self.execute_arm_action("heart", release_after_s=release_after_s)

    def right_heart(self, release_after_s=_ARM_RELEASE_DELAY_S):
        return self.execute_arm_action("right heart", release_after_s=release_after_s)

    def hands_up(self, release_after_s=_ARM_RELEASE_DELAY_S):
        return self.execute_arm_action("hands up", release_after_s=release_after_s)

    def x_ray(self, release_after_s=_ARM_RELEASE_DELAY_S):
        return self.execute_arm_action("x-ray", release_after_s=release_after_s)

    def right_hand_up(self, release_after_s=_ARM_RELEASE_DELAY_S):
        return self.execute_arm_action("right hand up", release_after_s=release_after_s)

    def reject(self, release_after_s=_ARM_RELEASE_DELAY_S):
        return self.execute_arm_action("reject", release_after_s=release_after_s)

    # ------------------------------------------------------------------
    # Audio / headlight
    # ------------------------------------------------------------------

    def say(self, text, volume=None):
        return self._get_audio().speak(text, volume=volume)

    def play_wav(self, wav_path, volume=None):
        return self._get_audio().play_wav(wav_path, volume=volume)

    def headlight(self, color="white", intensity=100, duration=None):
        return self._get_audio().set_headlight(color=color, intensity=intensity, duration=duration)

    # ------------------------------------------------------------------
    # Hand (Dex3)
    # ------------------------------------------------------------------

    def hand_open(self, hand="right", hold_s=0.6, rate_hz=50.0, ramp_s=None):
        self._get_hand(hand).open(hold_s=hold_s, rate_hz=rate_hz, ramp_s=ramp_s)

    def hand_close(self, hand="right", hold_s=0.6, rate_hz=50.0, ramp_s=None):
        self._get_hand(hand).close(hold_s=hold_s, rate_hz=rate_hz, ramp_s=ramp_s)

    def hand_pose(self, targets, hand="right", hold_s=0.6, rate_hz=50.0, kp=1.2, kd=0.05, tau=0.05, ramp_s=None):
        self._get_hand(hand).set_targets(targets, hold_s=hold_s, rate_hz=rate_hz, kp=kp, kd=kd, tau=tau, ramp_s=ramp_s)

    # ------------------------------------------------------------------
    # SLAM / navigation
    # ------------------------------------------------------------------

    def start_slam(self, slam_type="indoor"):
        response = self._get_slam_client().start_mapping(slam_type=slam_type)
        self.slam_is_running = response.code == 0
        return int(response.code)

    def stop_slam(self, save_path=None):
        client = self._get_slam_client()
        response = client.end_mapping(save_path) if save_path else client.close_slam()
        self.slam_is_running = False
        return int(response.code)

    def get_slam_pose(self, timeout_s=0.4):
        sub = self._get_slam_info_sub()
        t0 = time.time()
        while time.time() - t0 < max(0.05, float(timeout_s)):
            pose = sub.get_pose()
            if pose is not None:
                return pose
            time.sleep(0.03)
        return None

    def set_path_point(self, x, y, yaw=0.0):
        self._path_points.append((float(x), float(y), float(yaw)))

    def get_path_points(self):
        return list(self._path_points)

    def clear_path_points(self):
        self._path_points.clear()

    def navigate_path(self, clear_on_finish=True):
        if not self._path_points:
            raise RuntimeError("No path points. Call set_path_point() first.")
        if not self.slam_is_running:
            print("[navigate_path] SLAM not running.")
            return False
        try:
            self.walk_mode()
        except Exception:
            pass
        ok = True
        try:
            for i, (x, y, yaw) in enumerate(self._path_points, 1):
                pos = self.get_position()
                if pos is not None and math.hypot(x - pos[0], y - pos[1]) <= 0.20:
                    print(f"[navigate_path] step={i} skipped: already within 0.20m of target.")
                    continue
                rc = self._run_pose_nav(x, y, yaw)
                print(f"[navigate_path] step={i} pose_nav rc={rc}")
                if rc != 0:
                    print(f"[navigate_path] failed at point {i}: ({x:.3f},{y:.3f},{yaw:.3f})")
                    ok = False
                    break
        finally:
            if clear_on_finish:
                self._path_points.clear()
        return ok
