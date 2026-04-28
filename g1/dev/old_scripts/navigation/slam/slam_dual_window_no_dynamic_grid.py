#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import heapq
import json
import logging
import math
import os
import re
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Iterable


ROOT = Path(__file__).resolve().parent


def _force_cyclonedds_no_shm() -> None:
    """Disable CycloneDDS shared-memory transport for this launcher.

    The Unitree SDK path in this environment is tripping a CycloneDDS
    assertion in `dds_writecdr_impl_common` on the first real motion command.
    That assert is consistent with an Iceoryx/shared-memory mismatch between
    publisher state and sample allocation. Force plain UDP transport here.
    """

    if os.environ.get("CYCLONEDDS_URI"):
        return
    os.environ["CYCLONEDDS_URI"] = (
        "<CycloneDDS>"
        "<Domain>"
        "<General>"
        "<Interfaces><NetworkInterface autodetermine=\"true\"/></Interfaces>"
        "</General>"
        "<SharedMemory><Enable>false</Enable></SharedMemory>"
        "</Domain>"
        "</CycloneDDS>"
    )


_force_cyclonedds_no_shm()


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if "run_geoff_stack" not in sys.modules:
    _load_module("run_geoff_stack", ROOT / "geoff_stack_backup.py")

LEGACY = _load_module(
    "legacy_run_geoff_gui_input",
    ROOT / "unreliabale" / "run_geoff_gui_input.py",
)
RGBD_CLIENT = _load_module(
    "rgbd_client_module",
    (ROOT / "../../sensors/rgbd_client.py").resolve(),
)

from PySide6 import QtCore, QtWidgets  # type: ignore  # noqa: E402


def _install_native_stdout_filter(*, allow_livox: bool) -> None:
    """Filter native (C/C++) stdout spam (Livox SDK prints bypass Python logging)."""

    if allow_livox:
        return

    import os

    patterns = [
        re.compile(r"^\[\d{4}-\d{2}-\d{2} .*\] \[console\] \[info\]"),
        re.compile(r"Handle detection data"),
        re.compile(r"Detection lidars failed"),
        re.compile(r"general_command_handler\.cpp"),
        re.compile(r"device_manager\.cpp"),
        re.compile(r"mid360_command_handler\.cpp"),
        re.compile(r"parse_cfg_file\.cpp"),
        re.compile(r"params_check\.cpp"),
        re.compile(r"data_handler\.cpp"),
    ]

    try:
        orig_fd = os.dup(1)
        r_fd, w_fd = os.pipe()
        os.dup2(w_fd, 1)
        os.close(w_fd)
    except Exception:
        return

    def _writer(msg: str) -> None:
        try:
            os.write(orig_fd, msg.encode("utf-8", errors="replace"))
        except Exception:
            pass

    def _reader() -> None:
        buf = b""
        try:
            while True:
                chunk = os.read(r_fd, 4096)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    s = line.decode("utf-8", errors="replace")
                    if any(p.search(s) for p in patterns):
                        continue
                    _writer(s + "\n")
        finally:
            try:
                if buf:
                    s = buf.decode("utf-8", errors="replace")
                    if not any(p.search(s) for p in patterns):
                        _writer(s)
            except Exception:
                pass
            try:
                os.close(r_fd)
            except Exception:
                pass

    t = threading.Thread(target=_reader, daemon=True)
    t.start()


def _install_console_noise_filter(*, allow_livox: bool) -> None:
    """Reduce console spam from Livox + SDK while keeping file logs intact."""

    if allow_livox:
        return

    noisy = [
        re.compile(r"\\[Livox2\\] frame \\d+ pts"),
        re.compile(r"\\b\\[Livox2\\]\\b"),
        re.compile(r"Handle detection data"),
        re.compile(r"general_command_handler\\.cpp"),
    ]

    class _NoiseFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
            try:
                msg = record.getMessage()
            except Exception:
                return True
            return not any(p.search(msg) for p in noisy)

    root = logging.getLogger()
    filt = _NoiseFilter()
    for h in list(root.handlers):
        # Keep file logs; only filter the console handler(s).
        if hasattr(h, "baseFilename"):
            continue
        try:
            h.addFilter(filt)
        except Exception:
            pass


class _SlamSession:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.mapping_enabled = True
        self.reset_requested = False
        self.demo = None
        self.error: str | None = None
        self.last_update = 0.0

    def request_reset(self) -> None:
        with self.lock:
            self.reset_requested = True

    def consume_reset(self) -> bool:
        with self.lock:
            val = self.reset_requested
            self.reset_requested = False
            return val

    def set_mapping(self, enabled: bool) -> None:
        with self.lock:
            self.mapping_enabled = enabled

    def is_mapping_enabled(self) -> bool:
        with self.lock:
            return self.mapping_enabled

    def set_demo(self, demo) -> None:
        with self.lock:
            self.demo = demo
            self.error = None

    def set_error(self, msg: str | None) -> None:
        with self.lock:
            self.error = msg

    def get_error(self) -> str | None:
        with self.lock:
            return self.error

    def touch_update(self) -> None:
        with self.lock:
            self.last_update = time.time()

    def age_sec(self) -> float:
        with self.lock:
            if self.last_update <= 0.0:
                return float("inf")
            return max(0.0, time.time() - self.last_update)


SLAM_SESSION = _SlamSession()


class _RgbdSession:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.enabled = True
        self.max_fps = 8.0

    def set_enabled(self, enabled: bool) -> None:
        with self.lock:
            self.enabled = enabled

    def is_enabled(self) -> bool:
        with self.lock:
            return self.enabled

    def set_max_fps(self, fps: float) -> None:
        with self.lock:
            self.max_fps = max(0.1, fps)

    def get_max_fps(self) -> float:
        with self.lock:
            return self.max_fps


RGBD_SESSION = _RgbdSession()


def _compute_forward_depth_min(depth, depth_scale) -> float | None:
    try:
        import numpy as np  # type: ignore
    except Exception:
        return None
    if depth is None or depth_scale is None:
        return None
    h, w = depth.shape[:2]
    x0, x1 = int(w * 0.30), int(w * 0.70)
    y0, y1 = int(h * 0.30), int(h * 0.85)
    roi = depth[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    meters = roi.astype(np.float32) * float(depth_scale)
    valid = np.isfinite(meters) & (meters > 0.05) & (meters < 10.0)
    if not np.any(valid):
        return None
    return float(np.min(meters[valid]))


class _Lidar3DSession:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.enabled = True

    def set_enabled(self, enabled: bool) -> None:
        with self.lock:
            self.enabled = enabled

    def is_enabled(self) -> bool:
        with self.lock:
            return self.enabled


LIDAR3D_SESSION = _Lidar3DSession()

_raw_xyz_lock = threading.Lock()
_raw_xyz_latest = None  # latest raw LiDAR frame, always updated even when mapping frozen


class _NoOpPublisher:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def Init(self, *args, **kwargs) -> bool:  # noqa: D401
        return True

    def Write(self, *args, **kwargs) -> bool:  # noqa: D401
        return True


class _DisabledPublisher:
    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError("ChannelPublisher disabled in slam_dual_window.py")


class _DisabledSubscriber:
    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError("ChannelSubscriber disabled in slam_dual_window.py")


class _DisabledDex3Client:
    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError("Dex3 disabled in slam_dual_window.py")


@contextmanager
def _patch_upper_body_channels():
    patched: list[tuple[object, str, object]] = []

    def _try_patch(module_name: str, attr: str, value: object) -> None:
        try:
            module = __import__(module_name, fromlist=[attr])
            original = getattr(module, attr)
            setattr(module, attr, value)
            patched.append((module, attr, original))
        except Exception:
            pass

    _try_patch("unitree_sdk2py.core.channel", "ChannelPublisher", _DisabledPublisher)
    _try_patch("unitree_sdk2py.core.channel", "ChannelSubscriber", _DisabledSubscriber)
    _try_patch("unitree_sdk2py.dex3", "Dex3Client", _DisabledDex3Client)

    try:
        yield
    finally:
        for module, attr, original in reversed(patched):
            try:
                setattr(module, attr, original)
            except Exception:
                pass


def _install_dual_slam_runner() -> None:
    def _run_slam(stop_evt: threading.Event):  # pragma: no cover - needs HW
        try:
            SLAM_SESSION.set_error(None)
            LEGACY._patch_live_slam_for_pyqt()

            import live_slam as _ls  # type: ignore

            def _current_corrected_pose(demo):
                pose = demo._slam.last_pose.copy()  # type: ignore[attr-defined]
                mount_tf = getattr(_ls, "_R_MOUNT", None)
                if mount_tf is not None:
                    pose = mount_tf @ pose
                return pose

            def _current_corrected_cloud(demo):
                try:
                    cloud = demo._slam.get_map()
                except AttributeError:
                    cloud = demo._slam.local_map.point_cloud()
                mount_tf = getattr(_ls, "_R_MOUNT", None)
                if mount_tf is not None:
                    cloud = (cloud @ mount_tf[:3, :3].T).astype(cloud.dtype, copy=False)
                if cloud.shape[0] > demo._vis_max_points:
                    step = int(cloud.shape[0] / demo._vis_max_points) + 1
                    cloud = cloud[::step]
                return cloud

            if not getattr(_ls.LiveSLAMDemo.handle_points, "_dual_window_wrapped", False):
                _orig_hp = _ls.LiveSLAMDemo.handle_points

                def _safe_hp(self, xyz):
                    global _raw_xyz_latest
                    with _raw_xyz_lock:
                        _raw_xyz_latest = xyz.copy()
                    mapping_enabled = SLAM_SESSION.is_mapping_enabled()
                    was_mapping_enabled = getattr(self, "_dual_mapping_enabled_prev", True)
                    try:
                        _orig_hp(self, xyz)
                    except Exception as exc:  # pylint: disable=broad-except
                        try:
                            self._viewer.push(xyz, None)
                        except Exception:
                            pass
                        # KISS-ICP can fail transiently and can spam the console.
                        now = time.monotonic()
                        last = getattr(_safe_hp, "_last_kiss_log", 0.0)
                        if now - last > 8.0:
                            setattr(_safe_hp, "_last_kiss_log", now)
                            print("[slam_dual_window] KISS-ICP frame failed:", exc)
                    else:
                        if mapping_enabled:
                            self._dual_frozen_cloud = None
                        else:
                            if was_mapping_enabled or getattr(self, "_dual_frozen_cloud", None) is None:
                                try:
                                    self._dual_frozen_cloud = _current_corrected_cloud(self).copy()
                                except Exception:
                                    self._dual_frozen_cloud = None
                            frozen_cloud = getattr(self, "_dual_frozen_cloud", None)
                            if frozen_cloud is not None:
                                try:
                                    self._viewer.push(frozen_cloud, _current_corrected_pose(self))
                                except Exception:
                                    pass
                    finally:
                        self._dual_mapping_enabled_prev = mapping_enabled

                _safe_hp._dual_window_wrapped = True  # type: ignore[attr-defined]
                _ls.LiveSLAMDemo.handle_points = _safe_hp  # type: ignore[assignment]

            def _start_demo():
                demo = _ls.LiveSLAMDemo()
                SLAM_SESSION.set_demo(demo)
                spin_fn = getattr(demo, "spin", None)
                if callable(spin_fn):
                    t_spin = threading.Thread(target=spin_fn, daemon=True)
                    t_spin.start()
                return demo

            demo = _start_demo()
            try:
                while not stop_evt.is_set():
                    if SLAM_SESSION.consume_reset():
                        try:
                            demo.shutdown()
                        except Exception:
                            pass
                        demo = _start_demo()
                    try:
                        demo._viewer.tick()  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    time.sleep(0.05)
            finally:
                SLAM_SESSION.set_demo(None)
                try:
                    demo.shutdown()
                except Exception:
                    pass

        except Exception as exc:  # pylint: disable=broad-except
            SLAM_SESSION.set_error(str(exc))
            print("[slam_dual_window] SLAM thread disabled:", exc, file=sys.stderr)

    LEGACY._run_slam = _run_slam


_install_dual_slam_runner()


def _disabled_rx_realsense(_stop_evt: threading.Event) -> None:
    return


def _disabled_keyboard_controller(_stop_evt: threading.Event, _iface: str, _backend: str) -> None:
    return


def _disabled_rx_battery(_stop_evt: threading.Event, _iface: str) -> None:
    return


def _disabled_arm_tick(self) -> None:  # noqa: D401
    return


def _disabled_hand_tick(self) -> None:  # noqa: D401
    return


def _rx_rgbd_zmq(
    stop_evt: threading.Event,
    host: str,
    port: int,
    *,
    topic: str = "",
    timeout_ms: int = 1000,
    max_depth_m: float = 4.0,
) -> None:  # pragma: no cover - needs HW
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        import zmq  # type: ignore

        endpoint = f"tcp://{host}:{port}"
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.SUBSCRIBE, topic.encode("utf-8"))
        socket.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
        socket.connect(endpoint)

        frame_count = 0
        t_start = time.time()
        last_emit = 0.0
        blank_sent = False

        try:
            while not stop_evt.is_set():
                if not RGBD_SESSION.is_enabled():
                    if not blank_sent:
                        with LEGACY._state_lock:
                            LEGACY._state["rgbd"] = None
                        blank_sent = True
                    time.sleep(0.1)
                    continue

                try:
                    parts = socket.recv_multipart()
                except zmq.Again:
                    continue

                if len(parts) < 3:
                    continue

                now = time.time()
                min_dt = 1.0 / RGBD_SESSION.get_max_fps()
                if now - last_emit < min_dt:
                    continue

                color = RGBD_CLIENT._decode_color(parts[0])
                depth = RGBD_CLIENT._decode_depth(parts[1])
                depth_scale = RGBD_CLIENT._decode_scale(parts[2])
                if color is None:
                    continue

                if depth is None:
                    depth_vis = np.zeros_like(color)
                    cv2.putText(
                        depth_vis,
                        "No depth payload",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 180, 255),
                        2,
                    )
                    probe = None
                    depth_min_m = None
                else:
                    depth_vis = RGBD_CLIENT._colorize_depth(depth, max_depth_m, depth_scale)
                    probe = (depth.shape[1] // 2, depth.shape[0] // 2)
                    depth_min_m = _compute_forward_depth_min(depth, depth_scale)

                frame_count += 1
                elapsed = max(time.time() - t_start, 1e-6)
                fps = frame_count / elapsed
                color, depth_vis = RGBD_CLIENT._overlay_info(
                    color,
                    depth_vis,
                    depth,
                    depth_scale,
                    fps,
                    probe,
                )

                if color.shape[:2] != depth_vis.shape[:2]:
                    depth_vis = cv2.resize(
                        depth_vis,
                        (color.shape[1], color.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )

                combo = cv2.hconcat([color, depth_vis])
                with LEGACY._state_lock:
                    LEGACY._state["rgbd"] = combo
                    LEGACY._state["depth_guard_m"] = depth_min_m
                last_emit = now
                blank_sent = False
        finally:
            socket.close(0)
            context.term()

    except Exception as exc:  # pylint: disable=broad-except
        print("[slam_dual_window] RGBD receiver disabled:", exc, file=sys.stderr)


class ControlWindow(QtWidgets.QMainWindow):
    def __init__(self, owner: "DualWindow"):
        super().__init__()
        self._owner = owner
        self.setWindowTitle("SLAM Control")
        self.resize(460, 420)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self.mapping_lbl = QtWidgets.QLabel("Mapping: --")
        self.rgbd_lbl = QtWidgets.QLabel("RGBD: --")
        self.lidar3d_lbl = QtWidgets.QLabel("3D LiDAR: --")
        self.slam_lbl = QtWidgets.QLabel("SLAM: --")
        self.goal_lbl = QtWidgets.QLabel("Goal: --")
        self.replan_lbl = QtWidgets.QLabel("Replanning: --")
        self.pose_lbl = QtWidgets.QLabel("Pose: --")
        self.help_lbl = QtWidgets.QLabel(
            "Click the occupancy map in the viewer window to send a goal.\n"
            "W/A/S/D/Q/E or Shift for base tele-op still work in the Qt app."
        )
        self.help_lbl.setWordWrap(True)

        for lbl in (
            self.mapping_lbl,
            self.rgbd_lbl,
            self.lidar3d_lbl,
            self.slam_lbl,
            self.goal_lbl,
            self.replan_lbl,
            self.pose_lbl,
        ):
            lbl.setStyleSheet("font: 12pt 'DejaVu Sans Mono';")

        layout.addWidget(self.mapping_lbl)
        layout.addWidget(self.rgbd_lbl)
        layout.addWidget(self.lidar3d_lbl)
        layout.addWidget(self.slam_lbl)
        layout.addWidget(self.goal_lbl)
        layout.addWidget(self.replan_lbl)
        layout.addWidget(self.pose_lbl)
        layout.addWidget(self.help_lbl)

        grid = QtWidgets.QGridLayout()
        layout.addLayout(grid)

        def _btn(text: str, slot, row: int, col: int):
            btn = QtWidgets.QPushButton(text)
            btn.clicked.connect(slot)  # type: ignore[arg-type]
            grid.addWidget(btn, row, col)
            return btn

        _btn("Start Mapping", self._owner.start_mapping, 0, 0)
        _btn("Finish Mapping", self._owner.finish_mapping, 0, 1)
        _btn("Reset Mapping", self._owner.reset_mapping, 1, 0)
        _btn("Save Snapshot", self._owner.save_snapshot, 1, 1)
        _btn("Stop Robot", self._owner.stop_motion, 2, 0)
        _btn("Clear Goal", self._owner.clear_goal, 2, 1)
        _btn("Go To Target", self._owner.start_goal_navigation, 3, 0)
        _btn("Toggle Replanning", self._owner.toggle_obstacle_avoidance, 3, 1)
        self._rgbd_btn = _btn("Toggle RGBD", self._owner.toggle_rgbd, 4, 0)
        self._lidar3d_btn = _btn("Toggle 3D LiDAR", self._owner.toggle_lidar3d, 4, 1)

        nav_form = QtWidgets.QFormLayout()
        self._speed_spin = QtWidgets.QDoubleSpinBox()
        self._speed_spin.setRange(0.05, 1.00)
        self._speed_spin.setSingleStep(0.05)
        self._speed_spin.setDecimals(2)
        self._speed_spin.setValue(self._owner.nav_speed())
        self._speed_spin.valueChanged.connect(self._owner.set_nav_speed)  # type: ignore[arg-type]
        nav_form.addRow("Nav Speed (m/s)", self._speed_spin)

        self._duration_spin = QtWidgets.QDoubleSpinBox()
        self._duration_spin.setRange(0.20, 5.00)
        self._duration_spin.setSingleStep(0.10)
        self._duration_spin.setDecimals(2)
        self._duration_spin.setValue(self._owner.nav_cmd_duration())
        self._duration_spin.valueChanged.connect(self._owner.set_nav_cmd_duration)  # type: ignore[arg-type]
        nav_form.addRow("Cmd Duration (s)", self._duration_spin)

        self._obstacle_spin = QtWidgets.QDoubleSpinBox()
        self._obstacle_spin.setRange(0.10, 2.00)
        self._obstacle_spin.setSingleStep(0.05)
        self._obstacle_spin.setDecimals(2)
        self._obstacle_spin.setValue(self._owner.min_obstacle_distance())
        self._obstacle_spin.valueChanged.connect(self._owner.set_min_obstacle_distance)  # type: ignore[arg-type]
        nav_form.addRow("Min Obstacle (m)", self._obstacle_spin)

        self._tolerance_spin = QtWidgets.QDoubleSpinBox()
        self._tolerance_spin.setRange(0.10, 2.00)
        self._tolerance_spin.setSingleStep(0.05)
        self._tolerance_spin.setDecimals(2)
        self._tolerance_spin.setValue(self._owner.nav_goal_tolerance())
        self._tolerance_spin.valueChanged.connect(self._owner.set_nav_goal_tolerance)  # type: ignore[arg-type]
        nav_form.addRow("Goal Tolerance (m)", self._tolerance_spin)
        layout.addLayout(nav_form)

        layout.addStretch(1)

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()

    def _refresh(self) -> None:
        mapping_text = "ON" if SLAM_SESSION.is_mapping_enabled() else "FROZEN"
        self.mapping_lbl.setText(f"Mapping: {mapping_text}")
        self.rgbd_lbl.setText(f"RGBD: {'ON' if self._owner.rgbd_enabled() else 'OFF'}")
        self.lidar3d_lbl.setText(f"3D LiDAR: {'ON' if self._owner.lidar3d_enabled() else 'OFF'}")
        self.slam_lbl.setText(f"SLAM: {self._owner.slam_status_text()}")
        self.goal_lbl.setText(f"Goal: {self._owner.goal_status_text()}")
        self.replan_lbl.setText(
            f"Replanning: {'ON' if self._owner.obstacle_avoidance_enabled() else 'OFF'}"
            f"  min={self._owner.min_obstacle_distance():.2f}m"
        )
        self.pose_lbl.setText(f"Pose: {self._owner.pose_status_text()}")


class DualWindow(LEGACY.GeoffWindow):  # type: ignore[misc]
    def _configure_arm_variables(self):  # noqa: D401
        """Disable all low-level upper-body control for this launcher."""
        self._arm_pub = None
        self._dex3 = None
        self._arm_joint_idx = []
        self._cmd_q = {}
        self._joint_cur = {}
        self._pose_seq = []
        self._seq_idx = 0
        self._nav_status = getattr(self, "_nav_status", "stable stand only")

    def _maybe_arm_inference(self, _key_name: str) -> None:  # noqa: D401
        return

    def _maybe_hand_control(self, _key_name: str) -> None:  # noqa: D401
        return

    def _on_arm_tick(self) -> None:  # noqa: D401
        return

    def _on_hand_tick(self) -> None:  # noqa: D401
        return

    def _on_damp_pressed(self) -> None:  # noqa: D401
        self._nav_status = "upper-body low-level control disabled"

    def __init__(
        self,
        iface: str,
        ground_clear_in: float,
        *,
        enable_robot_control: bool,
        rgbd_host: str,
        rgbd_port: int,
        rgbd_topic: str = "",
        gui_fps: float,
        rgbd_fps: float,
        rgbd_enabled: bool,
        slam_fps: float,
        map_fps: float,
        max_points: int,
        **kwargs,
    ):
        input_backend = kwargs.get("input_backend", "qt")
        requested_input_backend = input_backend
        # Do not let the legacy constructor boot the robot client or start its
        # keyboard controller. We bring the robot client up lazily on first use
        # so the UI can start even on deployments where initial DDS writes are
        # unstable.
        kwargs["input_backend"] = "curses"

        orig_rx = LEGACY._rx_realsense
        orig_kb = getattr(LEGACY, "_run_keyboard_controller", None)
        orig_batt = getattr(LEGACY, "_rx_battery", None)
        orig_arm_tick = getattr(LEGACY.GeoffWindow, "_on_arm_tick", None)
        orig_hand_tick = getattr(LEGACY.GeoffWindow, "_on_hand_tick", None)
        LEGACY._rx_realsense = _disabled_rx_realsense
        if orig_kb is not None:
            LEGACY._run_keyboard_controller = _disabled_keyboard_controller
        if orig_batt is not None:
            LEGACY._rx_battery = _disabled_rx_battery
        if orig_arm_tick is not None:
            LEGACY.GeoffWindow._on_arm_tick = _disabled_arm_tick  # type: ignore[assignment]
        if orig_hand_tick is not None:
            LEGACY.GeoffWindow._on_hand_tick = _disabled_hand_tick  # type: ignore[assignment]
        try:
            with _patch_upper_body_channels():
                super().__init__(iface, ground_clear_in, **kwargs)
        finally:
            LEGACY._rx_realsense = orig_rx
            if orig_kb is not None:
                LEGACY._run_keyboard_controller = orig_kb
            if orig_batt is not None:
                LEGACY._rx_battery = orig_batt
            if orig_arm_tick is not None:
                LEGACY.GeoffWindow._on_arm_tick = orig_arm_tick  # type: ignore[assignment]
            if orig_hand_tick is not None:
                LEGACY.GeoffWindow._on_hand_tick = orig_hand_tick  # type: ignore[assignment]
        self._input_backend = requested_input_backend
        self._bot = None
        self._motion_proc = None
        self._motion_stdout_thread = None
        self._motion_stderr_thread = None
        self._last_motion_error = None
        self._nav_speed_mps = 0.18
        self._nav_cmd_duration_s = 5.0
        self._min_obstacle_dist_m = 0.3
        self._goal_tolerance_m = 0.5
        self._latest_depth_guard_m = None

        self.win.setWindowTitle("SLAM Viewer")
        self.control_win = ControlWindow(self)

        self._latest_pose = None
        self._nav_goal_world: tuple[float, float] | None = None
        self._nav_waypoints_world: list[tuple[float, float]] = []
        self._nav_status = "stable stand"
        self._nav_enabled = True
        self._rgbd_host = rgbd_host
        self._rgbd_port = rgbd_port
        self._rgbd_topic = rgbd_topic
        self._gui_fps = max(1.0, gui_fps)
        self._slam_fps = max(0.5, slam_fps)
        self._map_fps = max(0.2, map_fps)
        self._max_points = max(5_000, int(max_points))
        self._last_gui_tick = 0.0
        self._last_rgbd_ref = None
        self._last_slam_ref = None
        self._last_slam_draw = 0.0
        self._last_map_draw = 0.0
        self._last_pose_draw = 0.0
        self._pending_slam = None
        self._plan_lock = threading.Lock()
        self._plan_result = None
        self._plan_request_id = 0
        self._map_lock = threading.Lock()
        self._map_result = None
        self._map_worker_busy = False
        self._map_canvas = None
        self._ground_z_smooth = None
        self._dynamic_occ_map = None
        self._hover_px: tuple[int, int] | None = None
        self._clicked_px: tuple[int, int] | None = None
        self._goal_px: tuple[int, int] | None = None
        self._goal_ready = False
        self._nav_autonomous_active = False
        self._avoid_obstacles = True
        self._last_replan_t = 0.0
        self._replan_worker_busy = False
        self._map_seq = 0
        self._last_plan_map_seq = -1
        self._last_wp_log_t = 0.0
        self._last_wp_remaining = None
        self._last_nav_motion_cmd_t = 0.0
        self._iface_name = iface
        self._iface_valid = iface in {name for _, name in socket.if_nameindex()}
        self._robot_control_enabled = bool(enable_robot_control)
        self._robot_boot_failed = False
        LIDAR3D_SESSION.set_enabled(True)
        RGBD_SESSION.set_max_fps(rgbd_fps)
        RGBD_SESSION.set_enabled(rgbd_enabled)
        try:
            import pyqtgraph as pg  # type: ignore

            self._cmap = pg.colormap.get("turbo")  # type: ignore[attr-defined]
        except Exception:
            self._cmap = None
        try:
            self._map_img.setOpts(axisOrder="row-major")  # type: ignore[attr-defined]
        except Exception:
            pass

        try:
            self._btn_damp.setEnabled(False)
            self._btn_damp.setToolTip("Disabled in slam_dual_window.py")
        except Exception:
            pass
        try:
            self._arm_selector.setEnabled(False)
            self._arm_selector.setToolTip("Arm control disabled in slam_dual_window.py")
        except Exception:
            pass
        try:
            if hasattr(self, "_arm_timer"):
                self._arm_timer.stop()
        except Exception:
            pass
        try:
            if hasattr(self, "_hand_timer"):
                self._hand_timer.stop()
        except Exception:
            pass
        if not self._robot_control_enabled:
            self._bot = None
            self._nav_status = "robot control disabled"
        elif not self._iface_valid:
            self._nav_status = f"invalid iface: {iface}"
        else:
            if self._bot is None:
                self._nav_status = "robot ready to connect"

        self._threads.append(
            threading.Thread(
                target=_rx_rgbd_zmq,
                args=(self._stop_evt, self._rgbd_host, self._rgbd_port),
                kwargs={"topic": self._rgbd_topic},
                daemon=True,
            )
        )
        self._threads[-1].start()

        self._set_rgbd_widgets_visible(RGBD_SESSION.is_enabled())
        self._install_gl_status_label()

        try:
            self._refresh.setInterval(max(33, int(1000.0 / self._gui_fps)))
        except Exception:
            pass
        self._nav_timer = QtCore.QTimer(self)
        self._nav_timer.setInterval(100)
        self._nav_timer.timeout.connect(self._on_nav_tick)
        self._nav_timer.start()
        try:
            self.map_view.scene().sigMouseMoved.connect(self._on_map_hover)
        except Exception:
            pass

    def _ensure_robot_client(self) -> bool:
        if not self._robot_control_enabled:
            self._nav_status = "robot control disabled"
            return False
        if not self._iface_valid:
            self._nav_status = f"invalid iface: {self._iface_name}"
            return False
        if self._motion_proc is not None and self._motion_proc.poll() is None:
            return True
        if self._robot_boot_failed:
            self._nav_status = "robot client unavailable (DDS/CycloneDDS?)"
            return False
        try:
            _force_cyclonedds_no_shm()
            worker = ROOT / "slam_motion_worker.py"
            self._motion_proc = subprocess.Popen(
                [sys.executable, str(worker), "--iface", self._iface_name],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            self._start_motion_pipe_threads(self._motion_proc)
            self._send_motion_cmd({"cmd": "boot"})
            self._nav_status = "robot connecting..."
            return True
        except Exception as exc:  # pylint: disable=broad-except
            self._robot_boot_failed = True
            self._nav_status = "robot client unavailable (DDS/CycloneDDS?)"
            print("[slam_dual_window] Robot boot failed:", exc, file=sys.stderr)
            self._motion_proc = None
            return False

    def _start_motion_pipe_threads(self, proc) -> None:
        def _stdout_reader() -> None:
            stream = proc.stdout
            if stream is None:
                return
            try:
                for raw in stream:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                    except Exception:
                        print(f"[slam_motion_worker] {line}")
                        continue
                    if msg.get("ok"):
                        cmd = msg.get("cmd", "?")
                        if cmd == "boot":
                            self._nav_status = "robot connected"
                        elif cmd == "move":
                            self._last_motion_error = None
                    else:
                        err = str(msg.get("error", "worker error"))
                        cmd = msg.get("cmd", "?")
                        self._last_motion_error = err
                        self._nav_status = f"{cmd} failed: {err[:48]}"
                        print(f"[slam_motion_worker] {cmd} failed: {err}", file=sys.stderr)
            finally:
                if self._motion_proc is proc and proc.poll() is not None:
                    self._nav_status = "motion worker exited"

        def _stderr_reader() -> None:
            stream = proc.stderr
            if stream is None:
                return
            for raw in stream:
                line = raw.rstrip()
                if not line:
                    continue
                self._last_motion_error = line
                print(f"[slam_motion_worker] {line}", file=sys.stderr)

        self._motion_stdout_thread = threading.Thread(target=_stdout_reader, daemon=True)
        self._motion_stderr_thread = threading.Thread(target=_stderr_reader, daemon=True)
        self._motion_stdout_thread.start()
        self._motion_stderr_thread.start()

    def _send_motion_cmd(self, payload: dict) -> bool:
        proc = self._motion_proc
        if proc is None or proc.poll() is not None or proc.stdin is None:
            if self._last_motion_error:
                self._nav_status = f"robot client unavailable: {self._last_motion_error[:40]}"
            else:
                self._nav_status = "robot client unavailable"
            return False
        try:
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
            return True
        except Exception as exc:  # pylint: disable=broad-except
            self._nav_status = f"motion worker failed: {exc}"
            return False

    def _kill_motion_worker(self) -> None:
        proc = self._motion_proc
        self._motion_proc = None
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception:
            pass
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception:
            pass
        self._bal_mode = 0

    def run(self):  # noqa: D401
        self.win.show()
        self.control_win.show()
        sys.exit(self.app.exec())

    def _on_quit(self):  # noqa: D401
        try:
            self.control_win.close()
        except Exception:
            pass
        try:
            self._send_motion_cmd({"cmd": "quit"})
        except Exception:
            pass
        try:
            if self._motion_proc is not None:
                self._motion_proc.terminate()
        except Exception:
            pass
        self._stop_evt.set()
        try:
            self._nav_timer.stop()
        except Exception:
            pass
        try:
            if hasattr(self, "_refresh"):
                self._refresh.stop()
        except Exception:
            pass
        try:
            if hasattr(self, "_drive_timer"):
                self._drive_timer.stop()
        except Exception:
            pass
        for t in getattr(self, "_threads", []):
            try:
                t.join(timeout=1.0)
            except Exception:
                pass

    def _on_tick(self):
        now = time.time()
        min_dt = 1.0 / self._gui_fps
        if now - self._last_gui_tick < min_dt:
            return
        self._last_gui_tick = now

        self._update_key_overlay()

        with LEGACY._state_lock:
            rgbd = LEGACY._state.get("rgbd")
            vx, vy, om = LEGACY._state.get("vel", (0.0, 0.0, 0.0))
            soc = LEGACY._state.get("soc")
            self._latest_depth_guard_m = LEGACY._state.get("depth_guard_m")

        if rgbd is not None and rgbd is not self._last_rgbd_ref and getattr(rgbd, "shape", None) == (480, 1280, 3):
            self._last_rgbd_ref = rgbd
            rgb, depth = rgbd[:, :640], rgbd[:, 640:]
            px1, px2 = self._numpy_to_qpix(rgb), self._numpy_to_qpix(depth)
            if px1:
                scaled = px1.scaled(
                    self.rgb_lbl.size(),
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation,
                )
                self.rgb_lbl.setPixmap(scaled)
            if px2:
                scaled = px2.scaled(
                    self.depth_lbl.size(),
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation,
                )
                self.depth_lbl.setPixmap(scaled)

        status_txt = f"vx {vx:+.2f}  vy {vy:+.2f}  omega {om:+.2f}"
        if soc is not None:
            status_txt += f"   battery {soc:3d}%"
        else:
            with LEGACY._state_lock:
                volt = LEGACY._state.get("voltage")
            if volt is not None:
                status_txt += f"   V {volt:5.1f}"
        self.status.setText(status_txt)

        with LEGACY._slam_lock:
            data = LEGACY._slam_latest
        if data is not None:
            self._last_slam_ref = data
            self._pending_slam = data
            SLAM_SESSION.touch_update()
            _, pose = data
            if pose is not None and getattr(pose, "shape", None) == (4, 4):
                self._latest_pose = pose.copy()

        self._apply_pending_plan_result()
        self._apply_pending_map_result()
        if self._map_canvas is not None and self._latest_pose is not None:
            self._render_map_overlay(self._map_canvas)

        if self._pending_slam is None:
            self._show_no_slam_placeholders()
            return

        xyz, pose = self._pending_slam
        if getattr(xyz, "shape", None) is None or xyz.shape[0] == 0:
            self._show_no_slam_placeholders()
            return

        if LIDAR3D_SESSION.is_enabled() and now - self._last_slam_draw >= 1.0 / self._slam_fps:
            self._last_slam_draw = now
            if xyz.shape[0] > self._max_points:
                xyz_draw = xyz[:: int(xyz.shape[0] / self._max_points) + 1]
            else:
                xyz_draw = xyz

            z_ft = xyz_draw[:, 2] * 3.28084
            z_rel = z_ft - z_ft.min()
            v = self._clip01(z_rel / 9.0)
            v_gamma = v ** 0.35
            if self._cmap is not None:
                colors = self._cmap.map(v_gamma, mode="float")
            else:
                import numpy as np  # local

                gray = (255.0 * v_gamma).astype(np.uint8)
                colors = np.stack(
                    [
                        gray.astype(np.float32) / 255.0,
                        gray.astype(np.float32) / 255.0,
                        gray.astype(np.float32) / 255.0,
                        np.ones_like(v_gamma),
                    ],
                    axis=1,
                )
            self._scatter.setData(pos=xyz_draw, size=1.0, color=colors)

            if pose is not None and pose.shape == (4, 4) and now - self._last_pose_draw >= 0.5:
                self._last_pose_draw = now
                self._update_pose_axes(pose, xyz_draw)

        if now - self._last_map_draw >= 1.0 / self._map_fps:
            self._last_map_draw = now
            self._schedule_map_update(xyz, pose)

    @staticmethod
    def _clip01(arr):
        import numpy as np  # local

        return np.clip(arr, 0.0, 1.0)

    def _on_drive_tick(self):  # noqa: D401
        # Prevent the legacy teleop loop from fighting autonomous navigation.
        # Otherwise it keeps issuing Move(0,0,0) when no keys are pressed,
        # which produces stop-go "stutter" during nav.
        if getattr(self, "_nav_autonomous_active", False):
            return

        lim = self._current_speed_limit()

        if self._is_pressed("w") and not self._is_pressed("s"):
            self._vx = self._clamp(self._vx + self._LIN_STEP, lim)
        elif self._is_pressed("s") and not self._is_pressed("w"):
            self._vx = self._clamp(self._vx - self._LIN_STEP, lim)
        else:
            self._vx = 0.0

        if self._is_pressed("q") and not self._is_pressed("e"):
            self._vy = self._clamp(self._vy + self._LIN_STEP, lim)
        elif self._is_pressed("e") and not self._is_pressed("q"):
            self._vy = self._clamp(self._vy - self._LIN_STEP, lim)
        else:
            self._vy = 0.0

        if self._is_pressed("a") and not self._is_pressed("d"):
            self._omega = self._clamp(self._omega + self._ANG_STEP, lim)
        elif self._is_pressed("d") and not self._is_pressed("a"):
            self._omega = self._clamp(self._omega - self._ANG_STEP, lim)
        else:
            self._omega = 0.0

        if self._is_pressed("space"):
            self._vx = self._vy = self._omega = 0.0

        # Do not publish continuous zero-velocity commands while idle. The
        # legacy GUI wrote Move(0,0,0) every tick, but in this launcher that
        # causes unnecessary DDS traffic and can trip CycloneDDS assertions on
        # some deployments before the user even starts teleop or navigation.
        if (
            self._vx == 0.0
            and self._vy == 0.0
            and self._omega == 0.0
            and not any(self._is_pressed(k) for k in ("w", "a", "s", "d", "q", "e", "space"))
        ):
            return

        if self._is_pressed("z") or self._is_pressed("esc"):
            try:
                self._send_motion_cmd({"cmd": "stop"})
            except Exception:
                pass
            self.app.quit()
            return

        if self._motion_proc is not None and self._motion_proc.poll() is None:
            try:
                self._send_motion_cmd({"cmd": "move", "vx": self._vx, "vy": self._vy, "omega": self._omega})
                desired_mode = 0 if (self._vx == self._vy == self._omega == 0.0) else 1
                if desired_mode != self._bal_mode:
                    try:
                        self._send_motion_cmd({"cmd": "set_balance", "mode": desired_mode})
                        self._bal_mode = desired_mode
                    except Exception:
                        pass
            except Exception as exc:
                print("[slam_dual_window] Move failed:", exc, file=sys.stderr)
                self._motion_proc = None

        with LEGACY._state_lock:
            LEGACY._state["vel"] = (self._vx, self._vy, self._omega)

    def _on_map_click(self, ev):  # noqa: D401
        if self._occ_map is None or self._map_meta is None:
            return

        pt = self._scene_to_map_px(ev.scenePos())
        if pt is None:
            return
        gx, gy = pt
        if not (0 <= gx < 480 and 0 <= gy < 480):
            return

        rob_px = getattr(self, "_robot_px", None)
        if rob_px is None:
            return

        try:
            if ev.button() != QtCore.Qt.LeftButton:  # type: ignore[attr-defined]
                return
        except Exception:
            pass

        occ = self._occ_map.copy()
        map_meta = self._map_meta
        self._clicked_px = (gx, gy)
        self._goal_px = (gx, gy)
        self._goal_ready = False
        self._nav_autonomous_active = False
        self._stop_robot()
        if self._map_canvas is not None:
            self._render_map_overlay(self._map_canvas)
        request_id = self._plan_request_id + 1
        self._plan_request_id = request_id
        self._nav_status = "planning path..."

        def _worker():
            path = self._plan_path(rob_px[0], rob_px[1], gx, gy, occ)
            world_path = self._px_path_to_world_with_meta(path, map_meta) if path else None
            with self._plan_lock:
                self._plan_result = (request_id, path, world_path, map_meta)

        threading.Thread(target=_worker, daemon=True).start()

    def _on_map_hover(self, pos) -> None:  # noqa: D401
        self._hover_px = self._scene_to_map_px(pos)
        if self._map_canvas is not None:
            self._render_map_overlay(self._map_canvas)

    def _scene_to_map_px(self, scene_pos):
        try:
            view_pt = self._map_vb.mapSceneToView(scene_pos)  # type: ignore[attr-defined]
            gx, gy = int(round(view_pt.x())), int(round(view_pt.y()))
        except Exception:
            return None
        if 0 <= gx < 480 and 0 <= gy < 480:
            return (gx, gy)
        return None

    def _apply_pending_plan_result(self) -> None:
        with self._plan_lock:
            result = self._plan_result
            self._plan_result = None
        if result is None:
            return
        request_id, path, world_path, _map_meta = result
        if request_id != self._plan_request_id:
            return
        if path is None or len(path) <= 1:
            print("[slam_dual_window] No path found to clicked target.")
            self._nav_status = "no path found"
            self._goal_ready = False
            self._nav_autonomous_active = False
            return

        self._path_px = path
        self._nav_waypoints_world = self._compress_waypoints(world_path)
        self._nav_goal_world = self._nav_waypoints_world[-1] if self._nav_waypoints_world else None
        self._goal_ready = self._nav_goal_world is not None
        self._nav_autonomous_active = False
        self._last_plan_map_seq = self._map_seq
        self._nav_status = f"goal ready: {len(self._nav_waypoints_world)} waypoints"

        if self._nav_goal_world is not None:
            gxw, gyw = self._nav_goal_world
            print(f"[slam_dual_window] Goal set to ({gxw:+.2f}, {gyw:+.2f})")

    def start_goal_navigation(self) -> None:
        if not self._ensure_robot_client():
            self._nav_status = "robot client unavailable"
            return
        if not self._goal_ready or not self._nav_waypoints_world or self._nav_goal_world is None:
            self._nav_status = "define a target first"
            return
        self._nav_autonomous_active = True
        self._last_replan_t = 0.0
        self._last_wp_log_t = 0.0
        self._last_nav_motion_cmd_t = 0.0
        self._nav_status = f"autonomous nav: {len(self._nav_waypoints_world)} waypoints"
        gx, gy = self._nav_goal_world
        print(
            f"[slam_dual_window] Nav start -> goal ({gx:+.2f}, {gy:+.2f}) "
            f"avoid={self._avoid_obstacles} speed={self._nav_speed_mps:.2f} dur={self._nav_cmd_duration_s:.2f}"
        )

    def nav_speed(self) -> float:
        return float(self._nav_speed_mps)

    def set_nav_speed(self, value: float) -> None:
        self._nav_speed_mps = max(0.05, min(1.0, float(value)))

    def nav_cmd_duration(self) -> float:
        return float(self._nav_cmd_duration_s)

    def set_nav_cmd_duration(self, value: float) -> None:
        self._nav_cmd_duration_s = max(0.20, min(5.0, float(value)))

    def min_obstacle_distance(self) -> float:
        return float(self._min_obstacle_dist_m)

    def set_min_obstacle_distance(self, value: float) -> None:
        self._min_obstacle_dist_m = max(0.10, min(2.0, float(value)))
        if self._avoid_obstacles:
            self._last_replan_t = 0.0

    def nav_goal_tolerance(self) -> float:
        return float(self._goal_tolerance_m)

    def set_nav_goal_tolerance(self, value: float) -> None:
        self._goal_tolerance_m = max(0.10, min(2.0, float(value)))

    def obstacle_avoidance_enabled(self) -> bool:
        return bool(self._avoid_obstacles)

    def toggle_obstacle_avoidance(self) -> None:
        self._avoid_obstacles = not self._avoid_obstacles
        state = "ON" if self._avoid_obstacles else "OFF"
        print(f"[slam_dual_window] Obstacle avoidance: {state}")
        # If we are already navigating, enabling avoidance should replan ASAP.
        if self._avoid_obstacles and self._nav_autonomous_active:
            self._last_replan_t = 0.0

    def _maybe_replan(self) -> None:
        if not self._avoid_obstacles or not self._nav_autonomous_active:
            return
        if self._replan_worker_busy:
            return
        if self._map_meta is None:
            return
        occ = self._occ_map
        if occ is None:
            return
        rob_px = getattr(self, "_robot_px", None)
        if rob_px is None:
            return
        if self._goal_px is None:
            return
        now = time.monotonic()
        if now - self._last_replan_t < 1.0:
            return
        if self._map_seq <= self._last_plan_map_seq:
            return

        rx, ry = rob_px
        gx, gy = self._goal_px
        occ = occ.copy()
        map_meta = self._map_meta
        self._last_replan_t = now
        self._replan_worker_busy = True
        request_id = self._plan_request_id + 1
        self._plan_request_id = request_id

        def _worker():
            try:
                path = self._plan_path(rx, ry, gx, gy, occ)
                world_path = self._px_path_to_world_with_meta(path, map_meta) if path else None
                with self._plan_lock:
                    self._plan_result = (request_id, path, world_path, map_meta)
            finally:
                self._replan_worker_busy = False

        threading.Thread(target=_worker, daemon=True).start()
        print(f"[slam_dual_window] Replan map_seq={self._map_seq} from=({rx},{ry}) to=({gx},{gy})")

    def _px_path_to_world(self, path_px: Iterable[tuple[int, int]]) -> list[tuple[float, float]]:
        return self._px_path_to_world_with_meta(path_px, self._map_meta)

    @staticmethod
    def _px_path_to_world_with_meta(path_px: Iterable[tuple[int, int]], map_meta) -> list[tuple[float, float]]:
        min_x, min_y, scale = map_meta
        out = []
        for px, py in path_px:
            yw = (float(px) - 5.0) / scale + min_y
            xw = (474.0 - float(py)) / scale + min_x
            out.append((xw, yw))
        return out

    @staticmethod
    def _compress_waypoints(path: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if not path:
            return []
        keep = [path[0]]
        stride = 10
        for idx in range(stride, max(len(path) - 1, 1), stride):
            keep.append(path[idx])
        if keep[-1] != path[-1]:
            keep.append(path[-1])
        return keep

    @staticmethod
    def _yaw_from_pose(pose) -> float:
        return float(math.atan2(pose[1, 0], pose[0, 0]))

    @staticmethod
    def _clamp(val: float, limit: float) -> float:
        return max(-limit, min(limit, val))

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def _stop_robot(self) -> None:
        if self._motion_proc is None or self._motion_proc.poll() is not None:
            return
        try:
            self._send_motion_cmd({"cmd": "stop"})
        except Exception:
            pass

    def _is_path_blocked(self) -> bool:
        occ = self._occ_map
        if occ is None or not self._path_px:
            return False
        for col, row in self._path_px:
            if 0 <= row < occ.shape[0] and 0 <= col < occ.shape[1]:
                if occ[row, col]:
                    return True
        return False

    def _on_nav_tick(self) -> None:
        if self._motion_proc is None or self._motion_proc.poll() is not None or not self._nav_waypoints_world or not self._nav_autonomous_active:
            return
        if self._latest_pose is None:
            return

        self._maybe_replan()

        if self._latest_depth_guard_m is not None and self._latest_depth_guard_m <= self._min_obstacle_dist_m:
            self._nav_status = f"depth stop: obstacle {self._latest_depth_guard_m:.2f}m"
            self._nav_autonomous_active = False
            self._stop_robot()
            print(
                f"[slam_dual_window] Depth guard stop: {self._latest_depth_guard_m:.2f}m <= {self._min_obstacle_dist_m:.2f}m"
            )
            return

        if self._is_path_blocked():
            self._nav_status = "path blocked by obstacle"
            self._nav_autonomous_active = False
            self._stop_robot()
            print("[slam_dual_window] Path blocked — stopping navigation")
            return

        pose = self._latest_pose
        px, py = float(pose[0, 3]), float(pose[1, 3])

        while self._nav_waypoints_world:
            tx, ty = self._nav_waypoints_world[0]
            if math.hypot(tx - px, ty - py) > self._goal_tolerance_m:
                break
            self._nav_waypoints_world.pop(0)
            remaining = len(self._nav_waypoints_world)
            now = time.monotonic()
            if self._last_wp_remaining != remaining and now - self._last_wp_log_t > 1.0:
                self._last_wp_log_t = now
                self._last_wp_remaining = remaining
                print(f"[slam_dual_window] Waypoint reached, remaining={remaining}")

        if not self._nav_waypoints_world:
            self._nav_status = "goal reached"
            self._nav_goal_world = None
            self._goal_ready = False
            self._nav_autonomous_active = False
            self._stop_robot()
            print("[slam_dual_window] Goal reached")
            return

        tx, ty = self._nav_waypoints_world[0]
        dx = tx - px
        dy = ty - py
        dist = math.hypot(dx, dy)

        yaw = self._yaw_from_pose(pose)
        target_yaw = math.atan2(dy, dx)
        heading_err = self._wrap_angle(target_yaw - yaw)

        vy = 0.0
        if abs(heading_err) > 0.75:
            vx = 0.0
            omega = self._clamp(-0.9 * heading_err, 0.30)
        else:
            vx = self._clamp(0.55 * dist, self._nav_speed_mps)
            if dist < 0.40:
                vx = self._clamp(0.35 * dist, min(0.16, self._nav_speed_mps))
            omega = self._clamp(-0.7 * heading_err, 0.22)
            if abs(heading_err) > 0.30:
                vx *= 0.65

        try:
            # Match legacy teleop behaviour: static balance when stopped, gait when moving.
            desired_mode = 0 if (vx == 0.0 and vy == 0.0 and omega == 0.0) else 1
            if getattr(self, "_bal_mode", None) != desired_mode:
                try:
                    self._send_motion_cmd({"cmd": "set_balance", "mode": desired_mode})
                    self._bal_mode = desired_mode
                except Exception:
                    pass
            now_cmd = time.monotonic()
            min_cmd_dt = max(0.10, self._nav_cmd_duration_s * 0.85)
            if now_cmd - self._last_nav_motion_cmd_t < min_cmd_dt:
                self._nav_status = (
                    f"nav wp={len(self._nav_waypoints_world)} "
                    f"err={dist:.2f}m hdg={math.degrees(heading_err):+.0f}deg waiting"
                )
                return
            if not self._send_motion_cmd(
                {
                    "cmd": "move",
                    "vx": vx,
                    "vy": vy,
                    "omega": omega,
                    "duration": self._nav_cmd_duration_s,
                }
            ):
                raise RuntimeError(self._nav_status)
            self._last_nav_motion_cmd_t = now_cmd
            with LEGACY._state_lock:
                LEGACY._state["vel"] = (vx, vy, omega)
            self._nav_status = (
                f"nav wp={len(self._nav_waypoints_world)} "
                f"err={dist:.2f}m hdg={math.degrees(heading_err):+.0f}deg"
            )
        except Exception as exc:  # pylint: disable=broad-except
            self._nav_status = f"nav failed: {exc}"
            self.clear_goal()

        if self._map_canvas is not None:
            self._render_map_overlay(self._map_canvas)

    def start_mapping(self) -> None:
        SLAM_SESSION.set_mapping(True)
        self._nav_status = "mapping live"

    def finish_mapping(self) -> None:
        SLAM_SESSION.set_mapping(False)
        self._nav_status = "mapping frozen"

    def reset_mapping(self) -> None:
        self.clear_goal()
        self._path_px = None
        self._occ_map = None
        SLAM_SESSION.request_reset()
        SLAM_SESSION.set_mapping(True)
        self._nav_status = "resetting slam"

    def save_snapshot(self) -> None:
        import numpy as np  # local import

        out_dir = ROOT / "maps"
        out_dir.mkdir(exist_ok=True)

        default_name = time.strftime("slam_snapshot_%Y%m%d_%H%M%S")
        name, ok = QtWidgets.QInputDialog.getText(
            self.control_win,
            "Save Snapshot",
            "Snapshot name:",
            text=default_name,
        )
        if not ok or not name.strip():
            return
        name = name.strip()

        with LEGACY._slam_lock:
            data = LEGACY._slam_latest
        if data is None:
            self._nav_status = "nothing to save yet"
            return

        xyz, pose = data
        save_path = out_dir / f"{name}.npz"
        np.savez_compressed(
            save_path,
            points=xyz,
            pose=pose,
            occ_map=self._occ_map,
            path_px=self._path_px,
            goal=np.array(self._nav_goal_world) if self._nav_goal_world is not None else np.empty((0,)),
        )
        self._nav_status = f"saved {save_path.name}"
        print(f"[slam_dual_window] Saved snapshot to {save_path}")

    def enable_free_walk(self) -> None:
        self._nav_autonomous_active = False
        if not self._ensure_robot_client():
            self._nav_status = "robot client unavailable"
            return
        try:
            if self._send_motion_cmd({"cmd": "free_walk"}):
                self._nav_status = "free walk enabled"
            else:
                self._nav_status = "FreeWalk failed"
        except Exception as exc:  # pylint: disable=broad-except
            self._nav_status = f"FreeWalk failed: {exc}"

    def stop_motion(self) -> None:
        self._nav_autonomous_active = False
        self.clear_goal()
        self._stop_robot()
        self._last_nav_motion_cmd_t = 0.0
        self._nav_status = "stopped"

    def clear_goal(self) -> None:
        self._nav_waypoints_world.clear()
        self._nav_goal_world = None
        self._path_px = None
        self._clicked_px = None
        self._goal_px = None
        self._goal_ready = False
        self._nav_autonomous_active = False

    def toggle_rgbd(self) -> None:
        enabled = not RGBD_SESSION.is_enabled()
        RGBD_SESSION.set_enabled(enabled)
        if not enabled:
            self.rgb_lbl.clear()
            self.depth_lbl.clear()
        self._set_rgbd_widgets_visible(enabled)

    def rgbd_enabled(self) -> bool:
        return RGBD_SESSION.is_enabled()

    def _set_rgbd_widgets_visible(self, visible: bool) -> None:
        try:
            self.rgb_lbl.setVisible(visible)
            self.depth_lbl.setVisible(visible)
        except Exception:
            pass

    def _install_gl_status_label(self) -> None:
        try:
            self._gl_status_lbl = QtWidgets.QLabel(self.gl_view)
            self._gl_status_lbl.setStyleSheet(
                "color: #dddddd; background-color: rgba(0, 0, 0, 150); padding: 8px; border-radius: 6px;"
            )
            self._gl_status_lbl.move(20, 20)
            self._gl_status_lbl.resize(420, 60)
            self._gl_status_lbl.show()
        except Exception:
            self._gl_status_lbl = None

    def _show_no_slam_placeholders(self) -> None:
        try:
            import numpy as np  # type: ignore
        except Exception:
            return

        msg = self.slam_status_text()
        if self._map_canvas is None:
            canvas = np.full((480, 480, 3), 20, dtype=np.uint8)
            self._map_canvas = canvas
            self._render_map_overlay(canvas)
        if getattr(self, "_gl_status_lbl", None) is not None:
            self._gl_status_lbl.setText(msg)
            self._gl_status_lbl.setVisible(True)

    def slam_status_text(self) -> str:
        if not self._iface_valid:
            return "iface invalid"
        err = SLAM_SESSION.get_error()
        if err:
            msg = err.strip().splitlines()[0]
            return f"error: {msg[:60]}"
        age = SLAM_SESSION.age_sec()
        if age == float("inf"):
            return "waiting for data"
        if age > 3.0:
            return f"stale: {age:.1f}s"
        if getattr(self, "_gl_status_lbl", None) is not None:
            self._gl_status_lbl.setVisible(False)
        return "running"

    def toggle_lidar3d(self) -> None:
        enabled = not LIDAR3D_SESSION.is_enabled()
        LIDAR3D_SESSION.set_enabled(enabled)
        try:
            self.gl_view.setVisible(enabled)
        except Exception:
            pass
        if not enabled:
            try:
                self._scatter.setData(pos=[], color=[])
            except Exception:
                pass
            for item in getattr(self, "_pose_items", []):
                try:
                    self.gl_view.removeItem(item)
                except Exception:
                    pass
            self._pose_items = []

    def lidar3d_enabled(self) -> bool:
        return LIDAR3D_SESSION.is_enabled()

    def goal_status_text(self) -> str:
        if self._nav_goal_world is None:
            return self._nav_status
        gx, gy = self._nav_goal_world
        return f"{self._nav_status} -> ({gx:+.2f}, {gy:+.2f})"

    def pose_status_text(self) -> str:
        if self._latest_pose is None:
            return "--"
        x = float(self._latest_pose[0, 3])
        y = float(self._latest_pose[1, 3])
        yaw_deg = math.degrees(self._yaw_from_pose(self._latest_pose))
        depth_txt = ""
        if self._latest_depth_guard_m is not None:
            depth_txt = f" depth={self._latest_depth_guard_m:.2f}m"
        return f"x={x:+.2f} y={y:+.2f} yaw={yaw_deg:+.1f}deg{depth_txt}"

    def _render_map_overlay(self, canvas) -> None:
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except Exception:
            return

        canvas = np.array(canvas, copy=True)

        if self._path_px and len(self._path_px) > 1:
            cv2.polylines(
                canvas,
                [np.array(self._path_px, dtype=np.int32)],
                isClosed=False,
                color=(0, 0, 180),
                thickness=1,
            )
            gx, gy = self._path_px[-1]
            cv2.circle(canvas, (gx, gy), 4, (0, 0, 255), -1)

        if (
            getattr(self, "_nav_autonomous_active", False)
            and self._nav_waypoints_world
            and self._map_meta is not None
        ):
            min_x_m, min_y_m, scale_m = self._map_meta
            wp_pts: list[tuple[int, int]] = []
            if self._latest_pose is not None:
                rx_wp = int(round((float(self._latest_pose[1, 3]) - min_y_m) * scale_m + 5.0))
                ry_wp = int(round(479.0 - ((float(self._latest_pose[0, 3]) - min_x_m) * scale_m + 5.0)))
                if 0 <= rx_wp < 480 and 0 <= ry_wp < 480:
                    wp_pts.append((rx_wp, ry_wp))
            for xw, yw in self._nav_waypoints_world:
                col = int(round((yw - min_y_m) * scale_m + 5.0))
                row = int(round(479.0 - (xw - min_x_m) * scale_m + 5.0))
                if 0 <= col < 480 and 0 <= row < 480:
                    wp_pts.append((col, row))
            if len(wp_pts) > 1:
                cv2.polylines(
                    canvas,
                    [np.array(wp_pts, dtype=np.int32)],
                    isClosed=False,
                    color=(0, 255, 0),
                    thickness=2,
                )
            if wp_pts:
                cv2.circle(canvas, wp_pts[-1], 6, (0, 255, 0), -1)

        if self._hover_px is not None:
            hx, hy = self._hover_px
            cv2.drawMarker(
                canvas,
                (hx, hy),
                (0, 255, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=14,
                thickness=1,
            )
            cv2.circle(canvas, (hx, hy), 6, (0, 255, 255), 1)

        if self._clicked_px is not None:
            cx, cy = self._clicked_px
            cv2.drawMarker(
                canvas,
                (cx, cy),
                (0, 165, 255),
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=18,
                thickness=2,
            )
            cv2.circle(canvas, (cx, cy), 8, (0, 165, 255), 2)

        if self._clicked_px is not None and self._map_meta is not None:
            try:
                wx, wy = self._px_path_to_world_with_meta([self._clicked_px], self._map_meta)[0]
                label = f"target ({wx:+.2f}, {wy:+.2f})"
                cv2.putText(
                    canvas,
                    label,
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 165, 255),
                    1,
                    cv2.LINE_AA,
                )
            except Exception:
                pass

        if self._latest_pose is not None and self._map_meta is not None:
            try:
                min_x, min_y, scale = self._map_meta
                rx = int(round((float(self._latest_pose[1, 3]) - min_y) * scale + 5.0))
                ry = int(round(479.0 - ((float(self._latest_pose[0, 3]) - min_x) * scale + 5.0)))
                yaw = self._yaw_from_pose(self._latest_pose)
                tip_len = 18
                tip = (
                    int(round(rx + tip_len * math.sin(yaw))),
                    int(round(ry - tip_len * math.cos(yaw))),
                )
                cv2.circle(canvas, (rx, ry), 5, (255, 220, 0), -1)
                cv2.arrowedLine(canvas, (rx, ry), tip, (255, 220, 0), 2, tipLength=0.35)
            except Exception:
                pass

        if self._pending_slam is None:
            cv2.putText(
                canvas,
                self.slam_status_text(),
                (18, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )

        try:
            self._map_img.setImage(canvas, levels=(0, 255))  # type: ignore[arg-type]
        except Exception:
            pass

    def _apply_pending_map_result(self) -> None:
        with self._map_lock:
            result = self._map_result
            self._map_result = None
        if result is None:
            return
        canvas, occ_map, _dynamic_occ_map, map_meta, robot_px, ground_z = result
        self._map_canvas = canvas
        self._occ_map = occ_map
        self._dynamic_occ_map = None
        self._map_meta = map_meta
        self._robot_px = robot_px
        self._ground_z_smooth = ground_z
        self._map_seq += 1
        self._render_map_overlay(canvas)

    def _schedule_map_update(self, xyz, pose) -> None:
        if self._map_worker_busy:
            return

        if xyz.shape[0] > 20000:
            xyz_work = xyz[:: int(xyz.shape[0] / 20000) + 1].copy()
        else:
            xyz_work = xyz.copy()
        pose_work = pose.copy() if pose is not None and getattr(pose, "shape", None) == (4, 4) else None
        ground_z = self._ground_z_smooth
        clear_m = self._clear_m
        min_obstacle_dist_m = self._min_obstacle_dist_m
        with _raw_xyz_lock:
            raw_snap = _raw_xyz_latest.copy() if _raw_xyz_latest is not None else None
        self._map_worker_busy = True

        def _worker():
            try:
                result = self._build_map_snapshot(xyz_work, pose_work, clear_m, ground_z, min_obstacle_dist_m, raw_xyz=raw_snap)
                with self._map_lock:
                    self._map_result = result
            finally:
                self._map_worker_busy = False

        threading.Thread(target=_worker, daemon=True).start()

    @staticmethod
    def _plan_path(sx: int, sy: int, gx: int, gy: int, occ):  # noqa: D401
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        h, w = occ.shape
        if not (0 <= sx < w and 0 <= sy < h and 0 <= gx < w and 0 <= gy < h):
            return None
        if occ[sy, sx] or occ[gy, gx]:
            return None

        free_uint8 = (~occ).astype(np.uint8)
        dist = cv2.distanceTransform(free_uint8, cv2.DIST_L2, 5)
        max_dist = float(dist.max()) or 1.0
        clearance_floor = max(2.0, min(10.0, max_dist * 0.08))
        bias = 5.0

        def cell_cost(x: int, y: int) -> float:
            d = max(float(dist[y, x]), clearance_floor)
            d_norm = min(d / max_dist, 1.0)
            return 1.0 + bias * (1.0 - d_norm)

        open_set: list[tuple[float, tuple[int, int]]] = []
        heapq.heappush(open_set, (0.0, (sx, sy)))
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score = {(sx, sy): 0.0}

        def heuristic(x: int, y: int) -> float:
            return math.hypot(gx - x, gy - y)

        while open_set:
            _, current = heapq.heappop(open_set)
            cx, cy = current
            if current == (gx, gy):
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == dy == 0:
                        continue
                    nx, ny = cx + dx, cy + dy
                    if not (0 <= nx < w and 0 <= ny < h):
                        continue
                    if occ[ny, nx]:
                        continue
                    if dx != 0 and dy != 0:
                        if occ[cy, nx] or occ[ny, cx]:
                            continue
                    step = math.hypot(dx, dy) * cell_cost(nx, ny)
                    tentative = g_score[current] + step
                    if tentative < g_score.get((nx, ny), float("inf")):
                        came_from[(nx, ny)] = current
                        g_score[(nx, ny)] = tentative
                        heapq.heappush(open_set, (tentative + heuristic(nx, ny), (nx, ny)))
        return None

    @staticmethod
    def _inflate_occupancy(occ, scale: float, min_obstacle_dist_m: float):
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        inflate_radius_m = max(0.20, min(2.0, float(min_obstacle_dist_m)))
        wall_thicken_px = max(1, int(round(scale * 0.05)))
        inflate_px = max(wall_thicken_px + 1, int(round(scale * inflate_radius_m)))

        occ_u8 = occ.astype(np.uint8) * 255
        wall_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * wall_thicken_px + 1, 2 * wall_thicken_px + 1),
        )
        thick = cv2.dilate(occ_u8, wall_kernel)
        inflate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * inflate_px + 1, 2 * inflate_px + 1),
        )
        inflated = cv2.dilate(thick, inflate_kernel) > 0
        return thick > 0, inflated

    @staticmethod
    def _build_map_snapshot(xyz, pose, clear_m: float, ground_z_prev, min_obstacle_dist_m: float, raw_xyz=None):  # noqa: D401
        import os
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        if xyz.shape[0] == 0:
            return (
                np.full((480, 480, 3), 30, dtype=np.uint8),
                np.zeros((480, 480), dtype=bool),
                np.zeros((480, 480), dtype=bool),
                (0.0, 0.0, 1.0),
                None,
                ground_z_prev,
            )

        min_x, max_x = float(xyz[:, 0].min()), float(xyz[:, 0].max())
        min_y, max_y = float(xyz[:, 1].min()), float(xyz[:, 1].max())
        span = max(max_x - min_x, max_y - min_y, 1e-6)
        scale = 470.0 / span
        map_meta = (min_x, min_y, scale)

        def world_to_px(xw, yw):
            px = ((yw - min_y) * scale + 5).astype(np.int32)
            py = ((xw - min_x) * scale + 5).astype(np.int32)
            py = 479 - py
            return px, py

        canvas = np.full((480, 480, 3), 30, dtype=np.uint8)
        ground_z_inst = float(np.percentile(xyz[:, 2], 5.0))
        if ground_z_prev is None:
            ground_z = ground_z_inst
        else:
            ground_z = (1.0 - 0.05) * float(ground_z_prev) + 0.05 * ground_z_inst

        try:
            r_xy = float(os.environ.get("LIDAR_SELF_FILTER_RADIUS", 0.30))
            dz = float(os.environ.get("LIDAR_SELF_FILTER_Z", 0.24))
        except ValueError:
            r_xy, dz = 0.08, 0.05

        if pose is not None and pose.shape == (4, 4):
            rob_pos = pose[:3, 3]
            diff = xyz - rob_pos
            dist_xy = np.linalg.norm(diff[:, :2], axis=1)
            keep_mask = ~((dist_xy < r_xy) & (np.abs(diff[:, 2]) < dz))
            if keep_mask.sum() != xyz.shape[0]:
                xyz = xyz[keep_mask]

        thresh = ground_z + clear_m
        pts = xyz[xyz[:, 2] > thresh]
        occ_raw = np.zeros((480, 480), dtype=bool)

        if pts.shape[0] > 0:
            px_obs, py_obs = world_to_px(pts[:, 0], pts[:, 1])
            valid = (px_obs >= 0) & (px_obs < 480) & (py_obs >= 0) & (py_obs < 480)
            px_obs, py_obs = px_obs[valid], py_obs[valid]
            occ_raw[py_obs, px_obs] = True

        occ_display, occ_plan = DualWindow._inflate_occupancy(occ_raw, scale, min_obstacle_dist_m)
        canvas[occ_display] = (110, 110, 110)
        canvas[occ_raw] = (255, 255, 255)

        # This variant intentionally ignores the latest raw LiDAR overlay and
        # plans only against the current SLAM occupancy map.
        dynamic_occ_plan = occ_plan.copy()

        cv2.rectangle(canvas, (0, 0), (479, 479), (255, 255, 255), 1)

        robot_px = None
        if pose is not None and pose.shape == (4, 4):
            rob_pos = pose[:3, 3]
            rx, ry = world_to_px(np.array([rob_pos[0]]), np.array([rob_pos[1]]))
            rx, ry = int(rx[0]), int(ry[0])
            robot_px = (rx, ry)

            rr0, rr1 = max(0, ry - 1), min(480, ry + 2)
            rc0, rc1 = max(0, rx - 1), min(480, rx + 2)
            occ_plan[rr0:rr1, rc0:rc1] = False
            dynamic_occ_plan[rr0:rr1, rc0:rc1] = False

            fwd_vec = pose[:3, 0] * 0.25
            tip_world = rob_pos + fwd_vec
            tx, ty = world_to_px(np.array([tip_world[0]]), np.array([tip_world[1]]))
            tx, ty = int(tx[0]), int(ty[0])
            cv2.arrowedLine(canvas, (rx, ry), (tx, ty), (0, 255, 0), 2, tipLength=0.8)

        return canvas, occ_plan.copy(), dynamic_occ_plan.copy(), map_meta, robot_px, ground_z


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SLAM visualisation with a separate Qt control window, without dynamic occupancy overlay."
    )
    parser.add_argument("--iface", default="eth0", help="NIC connected to the Unitree G-1")
    parser.add_argument(
        "--clear",
        type=float,
        default=18.0,
        help="Clearance in inches above floor before a point becomes an obstacle",
    )
    parser.add_argument("--arm", choices=["left", "right"], default="left")
    parser.add_argument("--hand", choices=["left", "right"], default="left")
    parser.add_argument("--grip-force", type=float, dest="grip_force", default=0.3)
    parser.add_argument("--input", choices=("qt", "pynput", "curses"), default="qt")
    parser.add_argument("--rgbd-host", "--robot-ip", dest="rgbd_host", default="192.168.123.164", help="ZeroMQ RGBD publisher host")
    parser.add_argument("--rgbd-port", type=int, default=5555, help="ZeroMQ RGBD publisher port")
    parser.add_argument("--rgbd-topic", default="", help="Optional ZeroMQ subscription prefix")
    parser.add_argument("--gui-fps", type=float, default=8.0, help="Maximum GUI refresh rate")
    parser.add_argument("--rgbd-fps", type=float, default=6.0, help="Maximum RGBD decode/update rate")
    parser.add_argument("--no-rgbd", action="store_true", help="Start with RGBD widgets disabled")
    parser.add_argument("--slam-fps", type=float, default=2.0, help="Maximum 3-D point-cloud redraw rate")
    parser.add_argument("--map-fps", type=float, default=1.5, help="Maximum occupancy-map redraw rate")
    parser.add_argument("--max-points", type=int, default=30000, help="Maximum rendered point count")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(
        "--enable-robot-control",
        action="store_true",
        help="(Deprecated) Robot control is enabled by default; use --no-robot-control to disable",
    )
    grp.add_argument("--no-robot-control", action="store_true", help="Disable Unitree robot control and tele-op")
    parser.add_argument("--show-livox-logs", action="store_true", help="Do not filter Livox/SDK console spam")
    args = parser.parse_args()

    _install_console_noise_filter(allow_livox=bool(args.show_livox_logs))
    _install_native_stdout_filter(allow_livox=bool(args.show_livox_logs))

    window = DualWindow(
        args.iface,
        args.clear,
        enable_robot_control=bool(args.enable_robot_control) or (not bool(args.no_robot_control)),
        rgbd_host=args.rgbd_host,
        rgbd_port=args.rgbd_port,
        rgbd_topic=args.rgbd_topic,
        gui_fps=args.gui_fps,
        rgbd_fps=args.rgbd_fps,
        rgbd_enabled=not args.no_rgbd,
        slam_fps=args.slam_fps,
        map_fps=args.map_fps,
        max_points=args.max_points,
        hand=args.hand,
        grip_force=args.grip_force,
        input_backend=args.input,
    )
    window._active_arm = args.arm  # type: ignore[attr-defined]
    window._arm_selector.setCurrentIndex(0 if args.arm == "left" else 1)

    try:
        window._configure_arm_variables()
    except Exception as exc:  # pragma: no cover
        print("[slam_dual_window] Initial arm switch failed:", exc, file=sys.stderr)

    window.run()


if __name__ == "__main__":
    main()
