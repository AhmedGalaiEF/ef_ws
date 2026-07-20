#!/usr/bin/env python3
from __future__ import annotations
from sdk_client import Robot

import argparse
import importlib
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

try:
    from PyQt5.QtCore import QTimer, Qt
    from PyQt5.QtGui import QImage, QPixmap
    from PyQt5.QtWidgets import (
        QApplication,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QPlainTextEdit,
        QSizePolicy,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:
    raise SystemExit("PyQt5 is required. Install it with: pip install PyQt5") from exc

try:
    import numpy as np
    import pyqtgraph as pg
except ImportError as exc:
    raise SystemExit(
        "numpy and pyqtgraph are required. Install them with: pip install numpy pyqtgraph") from exc

try:
    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import Imu_ as SensorImu_
except ImportError as exc:
    raise SystemExit(
        "unitree_sdk2py is not installed. Install it with:\n"
        "  pip install -e <path-to-unitree_sdk2_python>"
    ) from exc


DEFAULT_RGB_TOPIC = ""
DEFAULT_DEPTH_TOPIC = ""
DEFAULT_SECONDARY_IMU_TOPIC = "rt/secondary_imu"
DEFAULT_LIDAR_MAX_POINTS = 4000
DEFAULT_BODY_MAX_JOINTS = 29


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PyQt sensor monitor for Unitree G1 Jetson-side RGB/depth/lidar/IMU/joint telemetry."
    )
    parser.add_argument("--iface", default="eth0", help="Network interface for DDS traffic.")
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain id.")
    parser.add_argument(
        "--rgb-source",
        choices=("video_client", "dds"),
        default="video_client",
        help="Use VideoClient RPC for RGB or subscribe to a DDS RGB topic.",
    )
    parser.add_argument(
        "--rgb-topic",
        default=DEFAULT_RGB_TOPIC,
        help="DDS RGB topic. Required when --rgb-source dds.",
    )
    parser.add_argument(
        "--rgb-type",
        default="unitree_go::msg::dds_::Go2FrontVideoData_",
        help="Preferred DDS RGB type when --rgb-source dds.",
    )
    parser.add_argument(
        "--depth-topic",
        default=DEFAULT_DEPTH_TOPIC,
        help="Optional DDS depth topic. Leave empty if depth is unavailable.",
    )
    parser.add_argument(
        "--depth-type",
        default="sensor_msgs::msg::dds_::Image_",
        help="Preferred DDS depth message type.",
    )
    parser.add_argument(
        "--secondary-imu-topic",
        default=DEFAULT_SECONDARY_IMU_TOPIC,
        help="DDS topic for the secondary or crotch IMU. Use empty string to disable.",
    )
    parser.add_argument(
        "--video-field",
        default="video720p",
        choices=("video720p", "video360p", "video180p"),
        help="Compressed field to decode if DDS RGB uses a front-video message type.",
    )
    parser.add_argument(
        "--poll-hz",
        type=float,
        default=12.0,
        help="Background polling frequency for SDK data.",
    )
    parser.add_argument(
        "--ui-hz",
        type=float,
        default=8.0,
        help="UI refresh rate.",
    )
    parser.add_argument(
        "--lidar-max-points",
        type=int,
        default=DEFAULT_LIDAR_MAX_POINTS,
        help="Maximum lidar points to render in the top-down scatter plot.",
    )
    parser.add_argument(
        "--max-joints",
        type=int,
        default=DEFAULT_BODY_MAX_JOINTS,
        help="Maximum number of body joints to show from rt/lowstate.",
    )
    parser.add_argument(
        "--safety-boot",
        action="store_true",
        help="Run the safety boot sequence before monitoring. Off by default.",
    )
    return parser.parse_args()


def _resolve_type(path: str) -> Any:
    if "::" in path:
        parts = [p for p in path.split("::") if p]
        module = importlib.import_module(".".join(parts[:-1]))
        return getattr(module, parts[-1])
    if ":" in path:
        module_name, class_name = path.split(":", 1)
    else:
        module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _resolve_first(candidates: list[str]) -> type | None:
    for candidate in candidates:
        try:
            return _resolve_type(candidate)
        except Exception:
            continue
    return None


def _rgb_type_candidates(user_type: str) -> list[str]:
    return [
        user_type,
        "unitree_go::msg::dds_::Go2FrontVideoData_",
        "unitree_sdk2py.idl.unitree_go.msg.dds_:Go2FrontVideoData_",
        "unitree_sdk2py.idl.unitree_go.msg.dds_.Go2FrontVideoData_",
        "sensor_msgs::msg::dds_::Image_",
        "unitree_sdk2py.idl.sensor_msgs.msg.dds_:Image_",
        "unitree_sdk2py.idl.sensor_msgs.msg.dds_.Image_",
    ]


def _depth_type_candidates(user_type: str) -> list[str]:
    return [
        user_type,
        "sensor_msgs::msg::dds_::Image_",
        "unitree_sdk2py.idl.sensor_msgs.msg.dds_:Image_",
        "unitree_sdk2py.idl.sensor_msgs.msg.dds_.Image_",
    ]


def _bytes_from_seq(data: Any) -> bytes:
    if isinstance(data, bytes):
        return data
    if isinstance(data, bytearray):
        return bytes(data)
    return bytes(bytearray(data))


def _decode_ros_image(msg: Any):
    try:
        height = int(msg.height)
        width = int(msg.width)
        step = int(msg.step)
        encoding = str(getattr(msg, "encoding", "")).lower()
        buf = _bytes_from_seq(msg.data)
    except Exception:
        return None

    if height <= 0 or width <= 0 or not buf:
        return None

    if encoding in ("bgr8", "rgb8"):
        dtype, channels = np.uint8, 3
    elif encoding in ("bgra8", "rgba8"):
        dtype, channels = np.uint8, 4
    elif encoding in ("mono8", "8uc1"):
        dtype, channels = np.uint8, 1
    elif encoding in ("mono16", "16uc1", "z16"):
        dtype, channels = np.uint16, 1
    else:
        if len(buf) == height * width * 3:
            dtype, channels, step = np.uint8, 3, width * 3
        elif len(buf) == height * width * 2:
            dtype, channels, step = np.uint16, 1, width * 2
        else:
            return None

    elem_size = int(np.dtype(dtype).itemsize)
    min_step = width * channels * elem_size
    if step < min_step:
        step = min_step
    if len(buf) < height * step:
        return None

    if channels == 1:
        img = np.ndarray((height, width), dtype=dtype, buffer=buf, strides=(step, elem_size)).copy()
    else:
        img = np.ndarray(
            (height, width, channels),
            dtype=dtype,
            buffer=buf,
            strides=(step, channels * elem_size, elem_size),
        ).copy()

    if encoding == "bgr8":
        img = img[:, :, ::-1]
    elif encoding == "bgra8":
        img = img[:, :, [2, 1, 0, 3]]
    return img


def _decode_go2_front(msg: Any, preferred_field: str):
    import cv2

    for field in (preferred_field, "video720p", "video360p", "video180p"):
        payload = getattr(msg, field, None)
        if payload is None:
            continue
        arr = np.frombuffer(_bytes_from_seq(payload), dtype=np.uint8)
        if arr.size == 0:
            continue
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is not None:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def _decode_any_image(msg: Any, preferred_field: str):
    if hasattr(msg, "height") and hasattr(msg, "width") and hasattr(msg, "data"):
        return _decode_ros_image(msg)
    if hasattr(msg, "video720p") or hasattr(msg, "video360p") or hasattr(msg, "video180p"):
        return _decode_go2_front(msg, preferred_field)
    return None


def _depth_to_color(depth: np.ndarray | None) -> np.ndarray | None:
    import cv2

    if depth is None:
        return None
    if depth.ndim == 3 and depth.shape[2] >= 3:
        if depth.shape[2] == 4:
            return depth[:, :, :3]
        return depth[:, :, :3]
    depth_f = depth.astype(np.float32, copy=False)
    valid = np.isfinite(depth_f)
    if not np.any(valid):
        return None
    vals = depth_f[valid]
    dmin = float(np.min(vals))
    dmax = float(np.max(vals))
    if dmax <= dmin:
        display = np.zeros_like(depth_f, dtype=np.uint8)
    else:
        display = (255.0 * (depth_f - dmin) / (dmax - dmin)).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(display, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)


def _format_vec3(values: tuple[float, float, float] | None) -> str:
    if values is None:
        return "n/a"
    return ", ".join(f"{float(v):+.3f}" for v in values)


def _extract_imu_values(msg: Any) -> dict[str, Any]:
    if msg is None:
        return {"rpy": None, "gyro": None, "acc": None, "quat": None, "temp": None}

    def _triple(obj: Any, *names: str) -> tuple[float, float, float] | None:
        for name in names:
            try:
                value = getattr(obj, name)
                return (float(value[0]), float(value[1]), float(value[2]))
            except Exception:
                continue
        return None

    def _quat(obj: Any, *names: str) -> tuple[float, float, float, float] | None:
        for name in names:
            try:
                value = getattr(obj, name)
                return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
            except Exception:
                continue
        return None

    temp = None
    for key in ("temperature", "temp"):
        try:
            temp = float(getattr(msg, key))
            break
        except Exception:
            continue

    return {
        "rpy": _triple(msg, "rpy"),
        "gyro": _triple(msg, "gyroscope", "gyro", "angular_velocity"),
        "acc": _triple(msg, "accelerometer", "acc", "linear_acceleration"),
        "quat": _quat(msg, "quaternion"),
        "temp": temp,
    }


def _image_to_qpixmap(image: np.ndarray | None, fallback_text: str) -> QPixmap:
    if image is None:
        pix = QPixmap(640, 360)
        pix.fill(Qt.black)
        return pix

    if image.ndim == 2:
        rgb = np.stack((image, image, image), axis=-1).astype(np.uint8, copy=False)
    else:
        rgb = image
        if rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)

    if not rgb.flags["C_CONTIGUOUS"]:
        rgb = np.ascontiguousarray(rgb)

    h, w, _ = rgb.shape
    qimg = QImage(rgb.data, w, h, int(rgb.strides[0]), QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


@dataclass
class MonitorSnapshot:
    timestamps: dict[str, float]
    rgb_frame: np.ndarray | None
    depth_frame: np.ndarray | None
    lidar_points: list[tuple[float, float, float]]
    body_imu: dict[str, Any]
    secondary_imu: dict[str, Any]
    lidar_imu: dict[str, Any]
    joints: list[dict[str, float]]
    mode_text: str
    gait_text: str
    position_text: str
    velocity_text: str
    status_lines: list[str]


class ImageSubscriber:
    def __init__(self, topic: str, msg_type: type, video_field: str) -> None:
        self.topic = str(topic)
        self.video_field = str(video_field)
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._timestamp = 0.0
        self._sub = ChannelSubscriber(self.topic, msg_type)
        self._sub.Init(self._callback, 10)

    def _callback(self, msg: Any) -> None:
        frame = _decode_any_image(msg, self.video_field)
        if frame is None:
            return
        with self._lock:
            self._frame = frame
            self._timestamp = time.time()

    def snapshot(self) -> tuple[np.ndarray | None, float]:
        with self._lock:
            if self._frame is None:
                return None, float(self._timestamp)
            return self._frame.copy(), float(self._timestamp)


class ImuSubscriber:
    def __init__(self, topic: str) -> None:
        self.topic = str(topic)
        self._lock = threading.Lock()
        self._values: dict[str, Any] = {"rpy": None,
                                        "gyro": None, "acc": None, "quat": None, "temp": None}
        self._timestamp = 0.0
        self._sub = ChannelSubscriber(self.topic, SensorImu_)
        self._sub.Init(self._callback, 20)

    def _callback(self, msg: Any) -> None:
        values = _extract_imu_values(msg)
        with self._lock:
            self._values = values
            self._timestamp = time.time()

    def snapshot(self) -> tuple[dict[str, Any], float]:
        with self._lock:
            return dict(self._values), float(self._timestamp)


class MonitorBackend:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.robot = Robot(
            iface=args.iface,
            domain_id=args.domain_id,
            safety_boot=bool(args.safety_boot),
            auto_start_sensors=True,
        )
        self._secondary_imu: ImuSubscriber | None = None
        self._rgb_sub: ImageSubscriber | None = None
        self._depth_sub: ImageSubscriber | None = None
        self._lock = threading.Lock()
        self._snapshot = MonitorSnapshot(
            timestamps={},
            rgb_frame=None,
            depth_frame=None,
            lidar_points=[],
            body_imu={"rpy": None, "gyro": None, "acc": None, "quat": None, "temp": None},
            secondary_imu={"rpy": None, "gyro": None, "acc": None, "quat": None, "temp": None},
            lidar_imu={"rpy": None, "gyro": None, "acc": None, "quat": None, "temp": None},
            joints=[],
            mode_text="n/a",
            gait_text="n/a",
            position_text="n/a",
            velocity_text="n/a",
            status_lines=["Initializing..."],
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        if args.rgb_source == "dds":
            if not args.rgb_topic:
                raise SystemExit("--rgb-topic is required when --rgb-source dds")
            rgb_type = _resolve_first(_rgb_type_candidates(args.rgb_type))
            if rgb_type is None:
                raise SystemExit(f"Could not resolve RGB DDS type from {args.rgb_type!r}")
            self._rgb_sub = ImageSubscriber(args.rgb_topic, rgb_type, args.video_field)

        if args.depth_topic:
            depth_type = _resolve_first(_depth_type_candidates(args.depth_type))
            if depth_type is None:
                raise SystemExit(f"Could not resolve depth DDS type from {args.depth_type!r}")
            self._depth_sub = ImageSubscriber(args.depth_topic, depth_type, args.video_field)

        if args.secondary_imu_topic:
            self._secondary_imu = ImuSubscriber(args.secondary_imu_topic)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="sensor-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.5)

    def snapshot(self) -> MonitorSnapshot:
        with self._lock:
            snap = self._snapshot
            return MonitorSnapshot(
                timestamps=dict(snap.timestamps),
                rgb_frame=None if snap.rgb_frame is None else snap.rgb_frame.copy(),
                depth_frame=None if snap.depth_frame is None else snap.depth_frame.copy(),
                lidar_points=list(snap.lidar_points),
                body_imu=dict(snap.body_imu),
                secondary_imu=dict(snap.secondary_imu),
                lidar_imu=dict(snap.lidar_imu),
                joints=[dict(row) for row in snap.joints],
                mode_text=str(snap.mode_text),
                gait_text=str(snap.gait_text),
                position_text=str(snap.position_text),
                velocity_text=str(snap.velocity_text),
                status_lines=list(snap.status_lines),
            )

    def _run(self) -> None:
        period = 1.0 / max(1.0, float(self.args.poll_hz))
        while not self._stop.is_set():
            t0 = time.time()
            status_lines: list[str] = []
            timestamps = self.robot.get_sensor_timestamps()

            rgb_frame = None
            if self.args.rgb_source == "video_client":
                try:
                    rgb_frame = self.robot.get_camera_frame_rgb()
                    timestamps["rgb"] = time.time()
                except Exception as exc:
                    status_lines.append(f"RGB VideoClient unavailable: {exc}")
            elif self._rgb_sub is not None:
                rgb_frame, rgb_ts = self._rgb_sub.snapshot()
                timestamps["rgb"] = rgb_ts
                if rgb_frame is None:
                    status_lines.append(f"Waiting for DDS RGB on {self.args.rgb_topic}")

            depth_frame = None
            if self._depth_sub is not None:
                depth_frame, depth_ts = self._depth_sub.snapshot()
                timestamps["depth"] = depth_ts
                if depth_frame is None:
                    status_lines.append(f"Waiting for DDS depth on {self.args.depth_topic}")
            else:
                status_lines.append(
                    "Depth feed not configured. Pass --depth-topic if the board exposes one.")

            body = self.robot.get_imu()
            body_imu = {
                "rpy": None if body is None else body.rpy,
                "gyro": None if body is None else body.gyro,
                "acc": None if body is None else body.acc,
                "quat": None if body is None else body.quat,
                "temp": None if body is None else body.temp,
            }
            if body is None:
                status_lines.append("Waiting for body IMU from sport state.")

            secondary_imu = {"rpy": None, "gyro": None, "acc": None, "quat": None, "temp": None}
            if self._secondary_imu is not None:
                secondary_imu, secondary_ts = self._secondary_imu.snapshot()
                timestamps["secondary_imu"] = secondary_ts
                if secondary_ts <= 0.0:
                    status_lines.append(
                        f"Waiting for secondary IMU on {self.args.secondary_imu_topic}")
            else:
                status_lines.append("Secondary or crotch IMU subscription disabled.")

            lidar_imu_msg = self.robot.get_lidar_imu()
            lidar_imu = _extract_imu_values(lidar_imu_msg)
            if timestamps.get("lidar_imu", 0.0) <= 0.0:
                status_lines.append("Waiting for lidar IMU.")

            joints: list[dict[str, float]] = []
            lowstate = self.robot.get_low_state_snapshot()
            if lowstate is not None:
                count = min(
                    len(lowstate.joint_positions),
                    len(lowstate.joint_velocities),
                    len(lowstate.joint_torques),
                    max(1, int(self.args.max_joints)),
                )
                for idx in range(count):
                    joints.append(
                        {
                            "index": float(idx),
                            "q": float(lowstate.joint_positions[idx]),
                            "dq": float(lowstate.joint_velocities[idx]),
                            "tau": float(lowstate.joint_torques[idx]),
                        }
                    )
            else:
                status_lines.append("Waiting for rt/lowstate joint telemetry.")

            lidar_points = self.robot.get_lidar_points(
                max_points=max(100, int(self.args.lidar_max_points)))
            if not lidar_points:
                status_lines.append("Waiting for lidar point cloud.")

            mode = self.robot.get_mode()
            gait = self.robot.get_gait()
            position = self.robot.get_position()
            velocity = self.robot.get_velocity()

            if not status_lines:
                status_lines.append("All configured feeds are updating.")

            snapshot = MonitorSnapshot(
                timestamps=timestamps,
                rgb_frame=rgb_frame,
                depth_frame=depth_frame,
                lidar_points=lidar_points,
                body_imu=body_imu,
                secondary_imu=secondary_imu,
                lidar_imu=lidar_imu,
                joints=joints,
                mode_text="n/a" if mode is None else str(int(mode)),
                gait_text="n/a" if gait is None else str(int(gait)),
                position_text=_format_vec3(position),
                velocity_text=_format_vec3(velocity),
                status_lines=status_lines,
            )
            with self._lock:
                self._snapshot = snapshot

            dt = time.time() - t0
            time.sleep(max(0.0, period - dt))


class ImagePane(QLabel):
    def __init__(self, title: str) -> None:
        super().__init__()
        self.title = title
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 220)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background:#111; color:#ddd; border:1px solid #444;")
        self.setText(f"{title}\nWaiting for data...")

    def set_frame(self, image: np.ndarray | None, waiting_text: str) -> None:
        if image is None:
            self.setPixmap(QPixmap())
            self.setText(f"{self.title}\n{waiting_text}")
            return
        pix = _image_to_qpixmap(image, waiting_text)
        self.setPixmap(pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.setText("")

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        pix = self.pixmap()
        if pix is not None and not pix.isNull():
            self.setPixmap(pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


class SensorMonitorWindow(QMainWindow):
    def __init__(self, backend: MonitorBackend, args: argparse.Namespace) -> None:
        super().__init__()
        self.backend = backend
        self.args = args
        self.setWindowTitle("G1 Sensor Monitor")
        self.resize(1680, 980)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        splitter = QSplitter(Qt.Vertical)
        root.addWidget(splitter)

        top = QWidget()
        top_layout = QHBoxLayout(top)
        self.rgb_pane = ImagePane("RGB")
        self.depth_pane = ImagePane("Depth")
        top_layout.addWidget(self.rgb_pane, 1)
        top_layout.addWidget(self.depth_pane, 1)
        splitter.addWidget(top)

        middle = QWidget()
        middle_layout = QGridLayout(middle)
        splitter.addWidget(middle)

        self.lidar_plot = pg.PlotWidget(background="#101010")
        self.lidar_plot.setTitle("Lidar Top-Down XY")
        self.lidar_plot.setLabel("left", "Y", units="m")
        self.lidar_plot.setLabel("bottom", "X", units="m")
        self.lidar_plot.showGrid(x=True, y=True, alpha=0.25)
        self.lidar_plot.setAspectLocked(True)
        self.lidar_scatter = pg.ScatterPlotItem(size=3, pen=None)
        self.lidar_plot.addItem(self.lidar_scatter)
        middle_layout.addWidget(self.lidar_plot, 0, 0, 2, 1)

        state_box = QGroupBox("Robot State")
        state_form = QFormLayout(state_box)
        self.mode_label = QLabel("n/a")
        self.gait_label = QLabel("n/a")
        self.position_label = QLabel("n/a")
        self.velocity_label = QLabel("n/a")
        state_form.addRow("Mode", self.mode_label)
        state_form.addRow("Gait", self.gait_label)
        state_form.addRow("Position", self.position_label)
        state_form.addRow("Velocity", self.velocity_label)
        middle_layout.addWidget(state_box, 0, 1)

        imu_box = QGroupBox("IMUs")
        imu_layout = QVBoxLayout(imu_box)
        self.body_imu_text = QPlainTextEdit()
        self.secondary_imu_text = QPlainTextEdit()
        self.lidar_imu_text = QPlainTextEdit()
        for widget in (self.body_imu_text, self.secondary_imu_text, self.lidar_imu_text):
            widget.setReadOnly(True)
            widget.setMaximumBlockCount(200)
            widget.setLineWrapMode(QPlainTextEdit.NoWrap)
        imu_layout.addWidget(QLabel("Body IMU"))
        imu_layout.addWidget(self.body_imu_text)
        imu_layout.addWidget(QLabel("Secondary / Crotch IMU"))
        imu_layout.addWidget(self.secondary_imu_text)
        imu_layout.addWidget(QLabel("Lidar IMU"))
        imu_layout.addWidget(self.lidar_imu_text)
        middle_layout.addWidget(imu_box, 1, 1)

        bottom = QWidget()
        bottom_layout = QHBoxLayout(bottom)
        splitter.addWidget(bottom)

        joint_box = QGroupBox("Joint States and Torques")
        joint_layout = QVBoxLayout(joint_box)
        self.joint_table = QTableWidget(0, 4)
        self.joint_table.setHorizontalHeaderLabels(
            ["Joint", "q [rad]", "dq [rad/s]", "tau_est [Nm]"])
        self.joint_table.verticalHeader().setVisible(False)
        self.joint_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.joint_table.setSelectionMode(QTableWidget.NoSelection)
        joint_layout.addWidget(self.joint_table)
        bottom_layout.addWidget(joint_box, 2)

        status_box = QGroupBox("Feed Status")
        status_layout = QVBoxLayout(status_box)
        self.status_text = QPlainTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setLineWrapMode(QPlainTextEdit.NoWrap)
        status_layout.addWidget(self.status_text)
        bottom_layout.addWidget(status_box, 1)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 2)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        interval_ms = int(1000.0 / max(1.0, float(args.ui_hz)))
        self._timer.start(max(50, interval_ms))

    @staticmethod
    def _imu_text(values: dict[str, Any]) -> str:
        lines = [
            f"rpy : {_format_vec3(values.get('rpy'))}",
            f"gyro: {_format_vec3(values.get('gyro'))}",
            f"acc : {_format_vec3(values.get('acc'))}",
        ]
        quat = values.get("quat")
        if quat is None:
            lines.append("quat: n/a")
        else:
            lines.append("quat: " + ", ".join(f"{float(v):+.4f}" for v in quat))
        temp = values.get("temp")
        lines.append("temp: n/a" if temp is None else f"temp: {float(temp):.2f}")
        return "\n".join(lines)

    def refresh(self) -> None:
        snap = self.backend.snapshot()

        rgb_wait = "Waiting for RGB feed..."
        depth_wait = "Waiting for depth feed..."
        self.rgb_pane.set_frame(snap.rgb_frame, rgb_wait)
        depth_vis = _depth_to_color(snap.depth_frame)
        if self.args.depth_topic:
            self.depth_pane.set_frame(depth_vis, depth_wait)
        else:
            self.depth_pane.set_frame(None, "Depth not configured.")

        self.mode_label.setText(snap.mode_text)
        self.gait_label.setText(snap.gait_text)
        self.position_label.setText(snap.position_text)
        self.velocity_label.setText(snap.velocity_text)

        self.body_imu_text.setPlainText(self._imu_text(snap.body_imu))
        self.secondary_imu_text.setPlainText(self._imu_text(snap.secondary_imu))
        self.lidar_imu_text.setPlainText(self._imu_text(snap.lidar_imu))

        self._update_lidar(snap.lidar_points)
        self._update_joint_table(snap.joints)
        self._update_status(snap.timestamps, snap.status_lines)

    def _update_lidar(self, points: list[tuple[float, float, float]]) -> None:
        if not points:
            self.lidar_scatter.setData([], [])
            return

        arr = np.asarray(points, dtype=np.float32)
        xy = arr[:, :2]
        z = arr[:, 2]
        zmin = float(np.min(z))
        zmax = float(np.max(z))
        if zmax <= zmin:
            norm = np.zeros_like(z, dtype=np.float32)
        else:
            norm = (z - zmin) / (zmax - zmin)

        colors = np.empty((arr.shape[0], 4), dtype=np.ubyte)
        colors[:, 0] = np.clip(255.0 * norm, 0, 255).astype(np.ubyte)
        colors[:, 1] = np.clip(255.0 * (1.0 - np.abs(norm - 0.5) * 2.0), 0, 255).astype(np.ubyte)
        colors[:, 2] = np.clip(255.0 * (1.0 - norm), 0, 255).astype(np.ubyte)
        colors[:, 3] = 220

        self.lidar_scatter.setData(x=xy[:, 0], y=xy[:, 1], brush=colors, size=3)

    def _update_joint_table(self, joints: list[dict[str, float]]) -> None:
        self.joint_table.setRowCount(len(joints))
        for row, item in enumerate(joints):
            values = [
                f"{int(item['index'])}",
                f"{float(item['q']):+.4f}",
                f"{float(item['dq']):+.4f}",
                f"{float(item['tau']):+.4f}",
            ]
            for col, text in enumerate(values):
                cell = self.joint_table.item(row, col)
                if cell is None:
                    cell = QTableWidgetItem()
                    self.joint_table.setItem(row, col, cell)
                cell.setText(text)
                cell.setTextAlignment(Qt.AlignCenter)
        self.joint_table.resizeColumnsToContents()

    def _update_status(self, timestamps: dict[str, float], status_lines: list[str]) -> None:
        now = time.time()
        lines = []
        for key in sorted(timestamps):
            ts = float(timestamps[key])
            age = math.inf if ts <= 0.0 else (now - ts)
            age_text = "never" if not math.isfinite(age) else f"{age:.2f}s old"
            lines.append(f"{key:16s} {age_text}")
        if status_lines:
            lines.append("")
            lines.extend(status_lines)
        self.status_text.setPlainText("\n".join(lines))

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._timer.stop()
        self.backend.stop()
        super().closeEvent(event)


def main() -> int:
    args = parse_args()
    app = QApplication(sys.argv)
    pg.setConfigOptions(antialias=False)
    backend = MonitorBackend(args)
    backend.start()
    window = SensorMonitorWindow(backend, args)
    window.show()
    return int(app.exec_())


if __name__ == "__main__":
    raise SystemExit(main())
