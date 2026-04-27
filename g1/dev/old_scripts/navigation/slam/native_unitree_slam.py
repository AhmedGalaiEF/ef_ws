#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6 import QtCore, QtGui, QtWidgets  # type: ignore

from sdk_client import Robot
from sdk_slam import SlamInfoSubscriber

try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "unitree_sdk2py is not installed. Install it with:\n"
        "  pip install -e <path-to-unitree_sdk2_python>"
    ) from exc


class MultiPointCloudSubscriber:
    def __init__(self, topics: list[str]) -> None:
        self.topics = list(dict.fromkeys(str(t) for t in topics if str(t).strip()))
        self._msgs: dict[str, PointCloud2_ | None] = {topic: None for topic in self.topics}
        self._ts: dict[str, float] = {topic: 0.0 for topic in self.topics}
        self._subs: dict[str, ChannelSubscriber] = {}

    def start(self) -> None:
        for topic in self.topics:
            if topic in self._subs:
                continue
            sub = ChannelSubscriber(topic, PointCloud2_)
            sub.Init(self._make_callback(topic), 10)
            self._subs[topic] = sub

    def _make_callback(self, topic: str):
        def _callback(msg: PointCloud2_) -> None:
            self._msgs[topic] = msg
            self._ts[topic] = time.time()

        return _callback

    def get_latest(self) -> tuple[PointCloud2_ | None, float, str | None]:
        best_topic = None
        best_ts = 0.0
        for topic, ts in self._ts.items():
            if ts > best_ts:
                best_ts = ts
                best_topic = topic
        if best_topic is None:
            return None, 0.0, None
        return self._msgs.get(best_topic), best_ts, best_topic

    def get_topic_stats(self) -> list[tuple[str, float, bool]]:
        return [(topic, self._ts.get(topic, 0.0), self._msgs.get(topic) is not None) for topic in self.topics]


def decode_points_xyz(msg: PointCloud2_, max_points: int = 120000) -> np.ndarray | None:
    try:
        fields = {f.name: f for f in msg.fields}
        if "x" not in fields or "y" not in fields or "z" not in fields:
            return None
        point_step = int(msg.point_step)
        if point_step <= 0:
            return None
        raw = bytes(msg.data)
        if not raw:
            return None
        dtype = np.dtype(
            {
                "names": ["x", "y", "z"],
                "formats": ["<f4", "<f4", "<f4"],
                "offsets": [
                    int(fields["x"].offset),
                    int(fields["y"].offset),
                    int(fields["z"].offset),
                ],
                "itemsize": point_step,
            }
        )
        arr = np.frombuffer(raw, dtype=dtype, count=len(raw) // point_step)
        pts = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32, copy=False)
        mask = np.isfinite(pts).all(axis=1)
        pts = pts[mask]
        if pts.size == 0:
            return None
        if max_points > 0 and pts.shape[0] > max_points:
            step = int(pts.shape[0] / max_points) + 1
            pts = pts[::step]
        return pts
    except Exception:
        return None


class MapCanvas(QtWidgets.QLabel):
    clicked_world = QtCore.Signal(float, float)

    def __init__(self) -> None:
        super().__init__()
        self.setFixedSize(520, 520)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setStyleSheet("background: #111; border: 1px solid #444;")
        self.setText("Waiting for SLAM point topics ...")
        self._pixmap: QtGui.QPixmap | None = None
        self._map_meta: tuple[float, float, float] | None = None

    def set_frame(self, image: np.ndarray, map_meta: tuple[float, float, float] | None) -> None:
        h, w, _ = image.shape
        qimg = QtGui.QImage(image.data, w, h, 3 * w, QtGui.QImage.Format_BGR888).copy()
        self._pixmap = QtGui.QPixmap.fromImage(qimg).scaled(
            self.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self._map_meta = map_meta
        self.setPixmap(self._pixmap)

    def mousePressEvent(self, event) -> None:  # noqa: D401
        if event.button() != QtCore.Qt.LeftButton or self._pixmap is None or self._map_meta is None:
            return
        label_w = self.width()
        label_h = self.height()
        pm_w = self._pixmap.width()
        pm_h = self._pixmap.height()
        off_x = max(0, (label_w - pm_w) // 2)
        off_y = max(0, (label_h - pm_h) // 2)
        px = event.position().x() - off_x
        py = event.position().y() - off_y
        if not (0 <= px < pm_w and 0 <= py < pm_h):
            return
        map_x = int(round(px * 480.0 / max(1, pm_w)))
        map_y = int(round(py * 480.0 / max(1, pm_h)))
        min_x, min_y, scale = self._map_meta
        yw = (float(map_x) - 5.0) / scale + min_y
        xw = (474.0 - float(map_y)) / scale + min_x
        self.clicked_world.emit(float(xw), float(yw))


class NativeUnitreeSlamWindow(QtWidgets.QWidget):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args
        self.robot = Robot(iface=args.iface, domain_id=args.domain_id, auto_start_sensors=False)
        self.info_sub = SlamInfoSubscriber(args.slam_info_topic, args.slam_key_topic)
        self.points_sub = MultiPointCloudSubscriber(args.slam_points_topics)
        self.info_sub.start()
        self.points_sub.start()

        self._last_points_ts = 0.0
        self._latest_points: np.ndarray | None = None
        self._latest_points_source: str | None = None
        self._latest_points_count = 0
        self._latest_pose: tuple[float, float, float] | None = None
        self._goal_world: tuple[float, float] | None = None
        self._goal_yaw = 0.0
        self._map_meta: tuple[float, float, float] | None = None
        self._free_walk_enabled = False
        self._raw_info_type = None
        self._raw_key_type = None

        self.setWindowTitle("Native Unitree SLAM")
        self.resize(860, 620)
        self._build_ui()

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_timer)  # type: ignore[arg-type]
        self._timer.start(150)

    def _build_ui(self) -> None:
        root = QtWidgets.QHBoxLayout(self)

        self.map_canvas = MapCanvas()
        self.map_canvas.clicked_world.connect(self._on_map_clicked)  # type: ignore[arg-type]
        root.addWidget(self.map_canvas, 0)

        panel = QtWidgets.QVBoxLayout()
        root.addLayout(panel, 1)

        self.pose_lbl = QtWidgets.QLabel("Pose: --")
        self.pose_lbl.setStyleSheet("font: 12pt 'DejaVu Sans Mono';")
        panel.addWidget(self.pose_lbl)

        self.status_lbl = QtWidgets.QLabel("Status: starting")
        self.status_lbl.setWordWrap(True)
        panel.addWidget(self.status_lbl)

        self.points_lbl = QtWidgets.QLabel("Points: waiting")
        self.points_lbl.setWordWrap(True)
        self.points_lbl.setStyleSheet("font: 10pt 'DejaVu Sans Mono';")
        panel.addWidget(self.points_lbl)

        self.topic_lbl = QtWidgets.QLabel(
            "Topics:\n"
            f"  iface: {self.args.iface}\n"
            f"  domain: {self.args.domain_id}\n"
            f"  pose: {self.args.slam_info_topic}\n"
            f"  points: {', '.join(self.args.slam_points_topics)}"
        )
        self.topic_lbl.setStyleSheet("font: 10pt 'DejaVu Sans Mono';")
        panel.addWidget(self.topic_lbl)

        self.diag_lbl = QtWidgets.QLabel("Diagnostics:\n  starting ...")
        self.diag_lbl.setWordWrap(True)
        self.diag_lbl.setStyleSheet("font: 10pt 'DejaVu Sans Mono';")
        panel.addWidget(self.diag_lbl)

        btns = QtWidgets.QGridLayout()
        panel.addLayout(btns)

        start_btn = QtWidgets.QPushButton("Start SLAM")
        start_btn.clicked.connect(self.start_slam)  # type: ignore[arg-type]
        btns.addWidget(start_btn, 0, 0)

        stop_btn = QtWidgets.QPushButton("Stop SLAM")
        stop_btn.clicked.connect(self.stop_slam)  # type: ignore[arg-type]
        btns.addWidget(stop_btn, 0, 1)

        self.free_walk_btn = QtWidgets.QPushButton("Enable Free Walk")
        self.free_walk_btn.clicked.connect(self.toggle_free_walk)  # type: ignore[arg-type]
        btns.addWidget(self.free_walk_btn, 1, 0)

        send_btn = QtWidgets.QPushButton("Nav To Target")
        send_btn.clicked.connect(self.send_target)  # type: ignore[arg-type]
        btns.addWidget(send_btn, 1, 1)

        clear_btn = QtWidgets.QPushButton("Clear Target")
        clear_btn.clicked.connect(self.clear_target)  # type: ignore[arg-type]
        btns.addWidget(clear_btn, 2, 0)

        refresh_btn = QtWidgets.QPushButton("Refresh View")
        refresh_btn.clicked.connect(self._refresh_map_only)  # type: ignore[arg-type]
        btns.addWidget(refresh_btn, 2, 1)

        form = QtWidgets.QFormLayout()
        panel.addLayout(form)

        self.slam_type_box = QtWidgets.QComboBox()
        self.slam_type_box.addItems(["indoor", "outdoor"])
        form.addRow("SLAM Type", self.slam_type_box)

        self.stop_save_edit = QtWidgets.QLineEdit()
        self.stop_save_edit.setPlaceholderText("Optional robot save path on stop")
        form.addRow("Save Path", self.stop_save_edit)

        self.goal_x_edit = QtWidgets.QLineEdit()
        self.goal_y_edit = QtWidgets.QLineEdit()
        self.goal_yaw_spin = QtWidgets.QDoubleSpinBox()
        self.goal_yaw_spin.setRange(-180.0, 180.0)
        self.goal_yaw_spin.setSingleStep(5.0)
        self.goal_yaw_spin.setSuffix(" deg")
        self.goal_yaw_spin.valueChanged.connect(self._on_goal_yaw_changed)  # type: ignore[arg-type]
        form.addRow("Goal X", self.goal_x_edit)
        form.addRow("Goal Y", self.goal_y_edit)
        form.addRow("Goal Yaw", self.goal_yaw_spin)

        self.help_lbl = QtWidgets.QLabel(
            "Click the map to place a target, then press `Nav To Target`. If no SLAM point topic is available, "
            "the UI still shows pose and SLAM controls but cannot draw a local map."
        )
        self.help_lbl.setWordWrap(True)
        panel.addWidget(self.help_lbl)
        panel.addStretch(1)

    def _set_status(self, text: str) -> None:
        self.status_lbl.setText(f"Status: {text}")

    def _on_goal_yaw_changed(self, value: float) -> None:
        self._goal_yaw = math.radians(float(value))

    def _on_map_clicked(self, xw: float, yw: float) -> None:
        self._goal_world = (xw, yw)
        self.goal_x_edit.setText(f"{xw:.3f}")
        self.goal_y_edit.setText(f"{yw:.3f}")
        self._set_status(f"target selected ({xw:+.2f}, {yw:+.2f})")
        self._refresh_map_only()

    def start_slam(self) -> None:
        try:
            rc = self.robot.start_slam(self.slam_type_box.currentText())
            self._set_status(f"start_slam rc={rc}")
        except Exception as exc:
            self._set_status(f"start_slam failed: {exc}")

    def stop_slam(self) -> None:
        try:
            save_path = self.stop_save_edit.text().strip() or None
            rc = self.robot.stop_slam(save_path)
            self._set_status(f"stop_slam rc={rc}")
        except Exception as exc:
            self._set_status(f"stop_slam failed: {exc}")

    def toggle_free_walk(self) -> None:
        try:
            if not self._free_walk_enabled:
                self.robot.walk_mode()
                self._free_walk_enabled = True
                self.free_walk_btn.setText("Disable Free Walk")
                self._set_status("free walk enabled")
            else:
                try:
                    self.robot.stop_moving()
                except Exception:
                    pass
                try:
                    self.robot.balanced_stand(0)
                except Exception:
                    pass
                self._free_walk_enabled = False
                self.free_walk_btn.setText("Enable Free Walk")
                self._set_status("free walk disabled")
        except Exception as exc:
            self._set_status(f"free walk toggle failed: {exc}")

    def clear_target(self) -> None:
        self._goal_world = None
        self.goal_x_edit.clear()
        self.goal_y_edit.clear()
        self._set_status("target cleared")
        self._refresh_map_only()

    def send_target(self) -> None:
        try:
            x = float(self.goal_x_edit.text().strip())
            y = float(self.goal_y_edit.text().strip())
        except Exception:
            self._set_status("invalid goal coordinates")
            return
        try:
            rc = self.robot._run_pose_nav(x, y, self._goal_yaw)
            self._goal_world = (x, y)
            self._set_status(f"pose_nav rc={rc} target=({x:+.2f}, {y:+.2f})")
            self._refresh_map_only()
        except Exception as exc:
            self._set_status(f"pose_nav failed: {exc}")

    def _on_timer(self) -> None:
        info_raw = self.info_sub.get_info()
        key_raw = self.info_sub.get_key()
        self._raw_info_type = self._extract_json_type(info_raw)
        self._raw_key_type = self._extract_json_type(key_raw)
        self._latest_pose = self.info_sub.get_pose()
        if self._latest_pose is not None:
            x, y, yaw = self._latest_pose
            self.pose_lbl.setText(f"Pose: x={x:+.2f}  y={y:+.2f}  yaw={math.degrees(yaw):+.1f} deg")
        else:
            self.pose_lbl.setText("Pose: --")

        msg, ts, source = self.points_sub.get_latest()
        if msg is not None and ts > self._last_points_ts:
            pts = decode_points_xyz(msg)
            if pts is not None:
                self._latest_points = pts
                self._last_points_ts = ts
                self._latest_points_source = source
                self._latest_points_count = int(pts.shape[0])
                self.points_lbl.setText(
                    f"Points: source={source}  count={self._latest_points_count}  age=0.0s"
                )
        elif self._latest_points_source is not None:
            age = max(0.0, time.time() - self._last_points_ts)
            self.points_lbl.setText(
                f"Points: source={self._latest_points_source}  count={self._latest_points_count}  age={age:.1f}s"
            )
        self._update_diagnostics()
        self._refresh_map_only()

    def _refresh_map_only(self) -> None:
        if self._latest_points is None or self._latest_points.shape[0] == 0:
            if self._latest_points_source is None:
                self._set_status("waiting for SLAM points topic")
            return
        canvas, meta = self._build_topdown_canvas(self._latest_points, self._latest_pose, self._goal_world)
        self._map_meta = meta
        self.map_canvas.set_frame(canvas, meta)
        if meta is None:
            self._set_status("received points but could not build top-down map")

    @staticmethod
    def _build_topdown_canvas(
        pts: np.ndarray,
        pose: tuple[float, float, float] | None,
        goal_world: tuple[float, float] | None,
    ) -> tuple[np.ndarray, tuple[float, float, float] | None]:
        pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.shape[0] == 0:
            return np.full((480, 480, 3), 24, dtype=np.uint8), None

        xy_norm = np.linalg.norm(pts[:, :2], axis=1)
        pts = pts[(xy_norm > 0.05) & (xy_norm < 45.0)]
        if pts.shape[0] == 0:
            return np.full((480, 480, 3), 24, dtype=np.uint8), None

        min_x = float(pts[:, 0].min())
        max_x = float(pts[:, 0].max())
        min_y = float(pts[:, 1].min())
        max_y = float(pts[:, 1].max())
        span = max(max_x - min_x, max_y - min_y, 1e-6)
        scale = 470.0 / span
        canvas = np.full((480, 480, 3), 24, dtype=np.uint8)

        def world_to_px(xw, yw):
            px = ((yw - min_y) * scale + 5.0).astype(np.int32)
            py = ((xw - min_x) * scale + 5.0).astype(np.int32)
            py = 479 - py
            return px, py

        px, py = world_to_px(pts[:, 0], pts[:, 1])
        valid = (px >= 0) & (px < 480) & (py >= 0) & (py < 480)
        px = px[valid]
        py = py[valid]
        z_vals = pts[:, 2][valid]
        if z_vals.size > 0:
            z_min = float(np.percentile(z_vals, 5.0))
            z_max = float(np.percentile(z_vals, 95.0))
            z_span = max(0.05, z_max - z_min)
            z_norm = np.clip((z_vals - z_min) / z_span, 0.0, 1.0)
            colors = np.stack(
                [
                    (60.0 + 180.0 * z_norm),
                    (120.0 + 100.0 * (1.0 - z_norm)),
                    np.full_like(z_norm, 220.0),
                ],
                axis=1,
            ).astype(np.uint8)
            canvas[py, px] = colors
        else:
            canvas[py, px] = (220, 220, 220)

        if pose is not None:
            rx, ry, yaw = pose
            rpx, rpy = world_to_px(np.array([rx]), np.array([ry]))
            cx = int(rpx[0])
            cy = int(rpy[0])
            if 0 <= cx < 480 and 0 <= cy < 480:
                canvas[max(0, cy - 3): min(480, cy + 4), max(0, cx - 3): min(480, cx + 4)] = (0, 220, 255)
                tip = (
                    int(round(cx + 18.0 * math.sin(yaw))),
                    int(round(cy - 18.0 * math.cos(yaw))),
                )
                NativeUnitreeSlamWindow._draw_line(canvas, (cx, cy), tip, (0, 220, 255))

        if goal_world is not None:
            gxw, gyw = goal_world
            gpx, gpy = world_to_px(np.array([gxw]), np.array([gyw]))
            gx = int(gpx[0])
            gy = int(gpy[0])
            if 0 <= gx < 480 and 0 <= gy < 480:
                NativeUnitreeSlamWindow._draw_cross(canvas, gx, gy, (0, 120, 255))

        canvas[0, :, :] = 255
        canvas[-1, :, :] = 255
        canvas[:, 0, :] = 255
        canvas[:, -1, :] = 255
        return canvas, (min_x, min_y, scale)

    @staticmethod
    def _draw_cross(canvas: np.ndarray, x: int, y: int, color: tuple[int, int, int]) -> None:
        for d in range(-8, 9):
            xx = x + d
            yy = y + d
            if 0 <= xx < 480 and 0 <= yy < 480:
                canvas[yy, xx] = color
            yy = y - d
            if 0 <= xx < 480 and 0 <= yy < 480:
                canvas[yy, xx] = color

    @staticmethod
    def _draw_line(canvas: np.ndarray, p0: tuple[int, int], p1: tuple[int, int], color: tuple[int, int, int]) -> None:
        x0, y0 = p0
        x1, y1 = p1
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            if 0 <= x0 < 480 and 0 <= y0 < 480:
                canvas[y0, x0] = color
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def _update_diagnostics(self) -> None:
        now = time.time()
        lines = ["Diagnostics:"]
        info_raw = self.info_sub.get_info()
        key_raw = self.info_sub.get_key()
        info_pose = self.info_sub.parse_pose(info_raw)
        key_pose = self.info_sub.parse_pose(key_raw)
        lines.append(
            self._format_diag_line(
                self.args.slam_info_topic,
                info_raw is not None,
                None,
                {"pose": info_pose, "json_type": self._raw_info_type, "raw_len": len(info_raw) if info_raw else 0},
                now,
            )
        )
        lines.append(
            self._format_diag_line(
                self.args.slam_key_topic,
                key_raw is not None,
                None,
                {"pose": key_pose, "json_type": self._raw_key_type, "raw_len": len(key_raw) if key_raw else 0},
                now,
            )
        )
        for topic, ts, has_msg in self.points_sub.get_topic_stats():
            pts_state = None
            if topic == self._latest_points_source:
                pts_state = self._latest_points_count
            lines.append(self._format_diag_line(topic, has_msg, ts, pts_state, now))
        self.diag_lbl.setText("\n".join(lines))

    @staticmethod
    def _format_diag_line(
        topic: str,
        has_msg: bool,
        last_ts: float | None,
        extra,
        now: float,
    ) -> str:
        base = f"  {topic}: "
        if last_ts is None:
            state = "seen" if has_msg else "waiting"
            if isinstance(extra, dict):
                pose = extra.get("pose")
                json_type = extra.get("json_type")
                raw_len = extra.get("raw_len")
                suffix = []
                if json_type is not None:
                    suffix.append(f"type={json_type}")
                if raw_len:
                    suffix.append(f"bytes={raw_len}")
                if isinstance(pose, tuple) and len(pose) == 3:
                    x, y, yaw = pose
                    suffix.append(f"pose=({x:+.2f},{y:+.2f},{math.degrees(yaw):+.1f}deg)")
                elif has_msg:
                    suffix.append("pose=unparsed")
                return base + state + (" " + " ".join(suffix) if suffix else "")
            return base + state
        if not has_msg or last_ts <= 0.0:
            return base + "waiting"
        age = max(0.0, now - last_ts)
        if isinstance(extra, int):
            return base + f"seen age={age:.1f}s points={extra}"
        return base + f"seen age={age:.1f}s"

    @staticmethod
    def _extract_json_type(payload_raw: str | None) -> str | None:
        if not payload_raw:
            return None
        try:
            payload = json.loads(payload_raw)
        except Exception:
            return "invalid_json"
        if not isinstance(payload, dict):
            return "non_object"
        val = payload.get("type")
        return str(val) if val is not None else "missing_type"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal UI for Unitree native SLAM topics and controls.")
    parser.add_argument("--iface", default="eth0")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--slam-info-topic", default="rt/slam_info")
    parser.add_argument("--slam-key-topic", default="rt/slam_key_info")
    parser.add_argument(
        "--slam-points-topic",
        dest="slam_points_topics",
        action="append",
        default=[],
        help="Point-cloud topic to try. Can be passed multiple times.",
    )
    args = parser.parse_args()
    if not args.slam_points_topics:
        args.slam_points_topics = [
            "rt/unitree/slam_mapping/points",
            "rt/slam_mapping/points",
            "rt/unitree/slam_relocation/global_map",
            "rt/unitree/slam_relocation/points",
            "rt/slam_relocation/points",
            "rt/utlidar/cloud_livox_mid360",
        ]
    return args


def main() -> None:
    args = parse_args()
    args.domain_id = 0
    ChannelFactoryInitialize(args.domain_id, args.iface)
    app = QtWidgets.QApplication(sys.argv)
    win = NativeUnitreeSlamWindow(args)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
