#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import numpy as np

try:
    from PySide6 import QtCore, QtGui, QtWidgets  # type: ignore

    Signal = QtCore.Signal

    def _event_pos(event):
        return event.position()

except ModuleNotFoundError:
    from PyQt5 import QtCore, QtGui, QtWidgets  # type: ignore

    Signal = QtCore.pyqtSignal

    def _event_pos(event):
        return event.localPos()

from g1_nav_backend import MapSnapshot, NavigationController, TelemetrySnapshot


def rgb_to_qimage(rgb: np.ndarray) -> QtGui.QImage:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected HxWx3 RGB image.")
    arr = np.ascontiguousarray(rgb)
    return QtGui.QImage(
        arr.data,
        arr.shape[1],
        arr.shape[0],
        arr.strides[0],
        QtGui.QImage.Format_RGB888,
    ).copy()


class MapCanvas(QtWidgets.QWidget):
    goalSelected = Signal(float, float)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(640, 640)
        self._snapshot: MapSnapshot | None = None
        self._draw_rect = QtCore.QRectF()
        self._hover_world: tuple[float, float] | None = None
        self.setMouseTracking(True)

    def set_snapshot(self, snapshot: MapSnapshot) -> None:
        self._snapshot = snapshot
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: D401
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor("#14171c"))

        snapshot = self._snapshot
        if snapshot is None or snapshot.rgb is None:
            painter.setPen(QtGui.QColor("#c8d0dc"))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, "Waiting for map data")
            return

        img = rgb_to_qimage(snapshot.rgb)
        target = QtCore.QRectF(self.rect())
        src = QtCore.QRectF(img.rect())
        scaled = QtCore.QSizeF(src.size())
        scaled.scale(target.size(), QtCore.Qt.KeepAspectRatio)
        x = target.x() + 0.5 * (target.width() - scaled.width())
        y = target.y() + 0.5 * (target.height() - scaled.height())
        self._draw_rect = QtCore.QRectF(x, y, scaled.width(), scaled.height())
        painter.drawImage(self._draw_rect, img, src)

        painter.setPen(QtGui.QPen(QtGui.QColor("#c8d0dc"), 1))
        painter.drawRect(self._draw_rect)

        if self._hover_world is not None:
            painter.setPen(QtGui.QColor("#f6c85f"))
            txt = f"x={self._hover_world[0]:+.2f}  y={self._hover_world[1]:+.2f}"
            painter.drawText(self._draw_rect.adjusted(10, 10, -10, -10), txt)

    def mousePressEvent(self, event) -> None:  # noqa: D401
        if event.button() != QtCore.Qt.LeftButton:
            return
        world = self._event_to_world(_event_pos(event))
        if world is not None:
            self.goalSelected.emit(world[0], world[1])

    def mouseMoveEvent(self, event) -> None:  # noqa: D401
        self._hover_world = self._event_to_world(_event_pos(event))
        self.update()

    def _event_to_world(self, pos) -> tuple[float, float] | None:
        snapshot = self._snapshot
        if snapshot is None or snapshot.occupancy is None or snapshot.rgb is None:
            return None
        if not self._draw_rect.contains(pos):
            return None
        u = (pos.x() - self._draw_rect.x()) / max(1.0, self._draw_rect.width())
        v = (pos.y() - self._draw_rect.y()) / max(1.0, self._draw_rect.height())
        px = int(np.clip(round(u * (snapshot.rgb.shape[1] - 1)), 0, snapshot.rgb.shape[1] - 1))
        py = int(np.clip(round(v * (snapshot.rgb.shape[0] - 1)), 0, snapshot.rgb.shape[0] - 1))
        gy = snapshot.rgb.shape[0] - 1 - py
        gx = px
        x = snapshot.origin_x + (float(gx) + 0.5) * snapshot.resolution
        y = snapshot.origin_y + (float(gy) + 0.5) * snapshot.resolution
        return (x, y)


class StatusPill(QtWidgets.QFrame):
    def __init__(self, title: str):
        super().__init__()
        self.setObjectName("statusPill")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        self._title = QtWidgets.QLabel(title)
        self._title.setObjectName("pillTitle")
        self._value = QtWidgets.QLabel("--")
        self._value.setObjectName("pillValue")
        layout.addWidget(self._title)
        layout.addWidget(self._value)

    def set_value(self, value: str) -> None:
        self._value.setText(value)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, controller: NavigationController):
        super().__init__()
        self.controller = controller
        self.setWindowTitle("G1 Navigation Console")
        self.resize(1560, 980)
        self._last_log_count = 0

        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        layout = QtWidgets.QVBoxLayout(root)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        header = QtWidgets.QHBoxLayout()
        self.title_lbl = QtWidgets.QLabel("G1 Navigation Console")
        self.title_lbl.setObjectName("mainTitle")
        self.subtitle_lbl = QtWidgets.QLabel("Direct SDK control, local occupancy mapping, click-to-goal planning")
        self.subtitle_lbl.setObjectName("subTitle")
        head_text = QtWidgets.QVBoxLayout()
        head_text.addWidget(self.title_lbl)
        head_text.addWidget(self.subtitle_lbl)
        header.addLayout(head_text, 1)

        self.connect_btn = QtWidgets.QPushButton("Connect Robot")
        self.connect_btn.clicked.connect(self.controller.connect_robot)  # type: ignore[arg-type]
        header.addWidget(self.connect_btn)
        layout.addLayout(header)

        pills = QtWidgets.QHBoxLayout()
        self.pill_conn = StatusPill("Robot")
        self.pill_pose = StatusPill("Pose")
        self.pill_slam = StatusPill("SLAM")
        self.pill_nav = StatusPill("Navigation")
        self.pill_map = StatusPill("Map")
        for pill in (self.pill_conn, self.pill_pose, self.pill_slam, self.pill_nav, self.pill_map):
            pills.addWidget(pill, 1)
        layout.addLayout(pills)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        layout.addWidget(split, 1)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        self.map_canvas = MapCanvas()
        self.map_canvas.goalSelected.connect(self._goal_selected)  # type: ignore[arg-type]
        left_layout.addWidget(self.map_canvas, 3)

        cam_frame = QtWidgets.QFrame()
        cam_frame.setObjectName("panel")
        cam_layout = QtWidgets.QVBoxLayout(cam_frame)
        cam_title = QtWidgets.QLabel("Front Camera")
        cam_title.setObjectName("panelTitle")
        self.cam_label = QtWidgets.QLabel("Waiting for camera")
        self.cam_label.setAlignment(QtCore.Qt.AlignCenter)
        self.cam_label.setMinimumHeight(260)
        self.cam_label.setObjectName("cameraLabel")
        cam_layout.addWidget(cam_title)
        cam_layout.addWidget(self.cam_label)
        left_layout.addWidget(cam_frame, 2)

        split.addWidget(left)

        right = QtWidgets.QScrollArea()
        right.setWidgetResizable(True)
        right_content = QtWidgets.QWidget()
        right.setWidget(right_content)
        form = QtWidgets.QVBoxLayout(right_content)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(10)

        form.addWidget(self._build_slam_panel())
        form.addWidget(self._build_nav_panel())
        form.addWidget(self._build_teleop_panel())
        form.addWidget(self._build_log_panel(), 1)
        split.addWidget(right)
        split.setSizes([1050, 510])

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(180)
        self._timer.timeout.connect(self._refresh)  # type: ignore[arg-type]
        self._timer.start()

        self.setStyleSheet(
            """
            QWidget { background: #0f1216; color: #eef2f7; font-family: 'DejaVu Sans'; font-size: 13px; }
            #mainTitle { font-size: 28px; font-weight: 700; color: #f7f8fb; }
            #subTitle { font-size: 13px; color: #95a0ae; }
            #panel, #statusPill { background: #171c22; border: 1px solid #252c35; border-radius: 14px; }
            #panelTitle { font-size: 16px; font-weight: 700; color: #f7f8fb; }
            #pillTitle { color: #8f98a6; font-size: 12px; }
            #pillValue { color: #f7f8fb; font-size: 16px; font-weight: 700; }
            QPushButton { background: #25303b; border: 1px solid #33414d; border-radius: 10px; padding: 10px 12px; }
            QPushButton:hover { background: #2d3b47; }
            QPushButton:pressed { background: #1e2831; }
            QLineEdit { background: #11161b; border: 1px solid #33414d; border-radius: 8px; padding: 8px; }
            QTextEdit { background: #11161b; border: 1px solid #33414d; border-radius: 10px; }
            QLabel#cameraLabel { background: #11161b; border-radius: 12px; border: 1px solid #29313b; }
            """
        )

    def _build_slam_panel(self) -> QtWidgets.QFrame:
        panel = QtWidgets.QFrame()
        panel.setObjectName("panel")
        layout = QtWidgets.QVBoxLayout(panel)
        title = QtWidgets.QLabel("SLAM + Mapping")
        title.setObjectName("panelTitle")
        layout.addWidget(title)

        row1 = QtWidgets.QHBoxLayout()
        btn = QtWidgets.QPushButton("Start SLAM")
        btn.clicked.connect(lambda: self.controller.start_slam("indoor"))  # type: ignore[arg-type]
        row1.addWidget(btn)
        btn = QtWidgets.QPushButton("Stop SLAM")
        btn.clicked.connect(lambda: self.controller.stop_slam())  # type: ignore[arg-type]
        row1.addWidget(btn)
        layout.addLayout(row1)

        row2 = QtWidgets.QHBoxLayout()
        self.mapping_chk = QtWidgets.QCheckBox("Local Mapping Enabled")
        self.mapping_chk.setChecked(True)
        self.mapping_chk.toggled.connect(self.controller.set_mapping_enabled)  # type: ignore[arg-type]
        row2.addWidget(self.mapping_chk)
        btn = QtWidgets.QPushButton("Reset Map")
        btn.clicked.connect(self.controller.reset_map)  # type: ignore[arg-type]
        row2.addWidget(btn)
        layout.addLayout(row2)

        save_row = QtWidgets.QHBoxLayout()
        self.snapshot_name = QtWidgets.QLineEdit()
        self.snapshot_name.setPlaceholderText("snapshot name")
        save_row.addWidget(self.snapshot_name, 1)
        btn = QtWidgets.QPushButton("Save Snapshot")
        btn.clicked.connect(self._save_snapshot)  # type: ignore[arg-type]
        save_row.addWidget(btn)
        layout.addLayout(save_row)
        return panel

    def _build_nav_panel(self) -> QtWidgets.QFrame:
        panel = QtWidgets.QFrame()
        panel.setObjectName("panel")
        layout = QtWidgets.QVBoxLayout(panel)
        title = QtWidgets.QLabel("Navigation")
        title.setObjectName("panelTitle")
        layout.addWidget(title)

        row1 = QtWidgets.QHBoxLayout()
        btn = QtWidgets.QPushButton("Start Nav")
        btn.clicked.connect(self.controller.start_navigation)  # type: ignore[arg-type]
        row1.addWidget(btn)
        btn = QtWidgets.QPushButton("Stop Nav")
        btn.clicked.connect(self.controller.stop_navigation)  # type: ignore[arg-type]
        row1.addWidget(btn)
        layout.addLayout(row1)

        row2 = QtWidgets.QHBoxLayout()
        btn = QtWidgets.QPushButton("Clear Goal")
        btn.clicked.connect(self.controller.clear_goal)  # type: ignore[arg-type]
        row2.addWidget(btn)
        btn = QtWidgets.QPushButton("Pause SDK Nav")
        btn.clicked.connect(self.controller.pause_nav)  # type: ignore[arg-type]
        row2.addWidget(btn)
        layout.addLayout(row2)

        row3 = QtWidgets.QHBoxLayout()
        btn = QtWidgets.QPushButton("Resume SDK Nav")
        btn.clicked.connect(self.controller.resume_nav)  # type: ignore[arg-type]
        row3.addWidget(btn)
        self.avoid_chk = QtWidgets.QCheckBox("Avoid Obstacles")
        self.avoid_chk.setChecked(True)
        self.avoid_chk.toggled.connect(lambda _checked: self.controller.toggle_avoidance())  # type: ignore[arg-type]
        row3.addWidget(self.avoid_chk)
        layout.addLayout(row3)

        self.goal_lbl = QtWidgets.QLabel("Click the map to select a goal.")
        self.goal_lbl.setWordWrap(True)
        layout.addWidget(self.goal_lbl)
        return panel

    def _build_teleop_panel(self) -> QtWidgets.QFrame:
        panel = QtWidgets.QFrame()
        panel.setObjectName("panel")
        layout = QtWidgets.QVBoxLayout(panel)
        title = QtWidgets.QLabel("Manual Motion")
        title.setObjectName("panelTitle")
        layout.addWidget(title)

        grid = QtWidgets.QGridLayout()
        self._step_button(grid, "Forward", 0, 1, lambda: self.controller.step_move(0.18, 0.0, 0.0))
        self._step_button(grid, "Left", 1, 0, lambda: self.controller.step_move(0.0, 0.10, 0.0))
        self._step_button(grid, "Stop", 1, 1, self.controller.stop_navigation)
        self._step_button(grid, "Right", 1, 2, lambda: self.controller.step_move(0.0, -0.10, 0.0))
        self._step_button(grid, "Back", 2, 1, lambda: self.controller.step_move(-0.12, 0.0, 0.0))
        self._step_button(grid, "Turn L", 3, 0, lambda: self.controller.step_move(0.0, 0.0, 0.28))
        self._step_button(grid, "Free Walk", 3, 1, self.controller.free_walk)
        self._step_button(grid, "Turn R", 3, 2, lambda: self.controller.step_move(0.0, 0.0, -0.28))
        layout.addLayout(grid)
        return panel

    def _build_log_panel(self) -> QtWidgets.QFrame:
        panel = QtWidgets.QFrame()
        panel.setObjectName("panel")
        layout = QtWidgets.QVBoxLayout(panel)
        title = QtWidgets.QLabel("Status Log")
        title.setObjectName("panelTitle")
        layout.addWidget(title)
        self.log_box = QtWidgets.QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(240)
        layout.addWidget(self.log_box)
        return panel

    def _step_button(self, grid, text: str, row: int, col: int, slot) -> None:
        btn = QtWidgets.QPushButton(text)
        btn.clicked.connect(slot)  # type: ignore[arg-type]
        grid.addWidget(btn, row, col)

    def _goal_selected(self, x: float, y: float) -> None:
        self.controller.set_goal_world(x, y)

    def _save_snapshot(self) -> None:
        self.controller.save_snapshot(self.snapshot_name.text().strip() or None)

    def _refresh(self) -> None:
        state = self.controller.get_ui_state()
        telemetry: TelemetrySnapshot = state["telemetry"]
        map_snapshot: MapSnapshot = state["map"]
        logs: list[str] = state["logs"]

        self.pill_conn.set_value("Connected" if telemetry.robot_connected else "Offline")
        if telemetry.pose is None:
            self.pill_pose.set_value("No pose")
        else:
            self.pill_pose.set_value(f"{telemetry.pose_source}  x={telemetry.pose[0]:+.2f} y={telemetry.pose[1]:+.2f}")
        self.pill_slam.set_value("Running" if telemetry.slam_running else "Stopped")
        self.pill_nav.set_value("Active" if telemetry.nav_active else "Idle")
        age = 0.0 if map_snapshot.updated_at <= 0.0 else max(0.0, time.time() - map_snapshot.updated_at)
        ros_flag = "ROS ok" if telemetry.ros_bridge_ready else "ROS wait"
        self.pill_map.set_value(f"{map_snapshot.width}x{map_snapshot.height}  age={age:.1f}s  {ros_flag}")

        self.map_canvas.set_snapshot(map_snapshot)

        if telemetry.goal is None:
            self.goal_lbl.setText("Click the map to select a goal.")
        else:
            self.goal_lbl.setText(
                f"Goal: ({telemetry.goal[0]:+.2f}, {telemetry.goal[1]:+.2f})\n"
                f"Pose source: {telemetry.pose_source}\n"
                f"Status: {telemetry.status}\n"
                f"ROS bridge: {telemetry.ros_bridge_status}"
            )

        camera_rgb = state["camera_rgb"]
        if camera_rgb is not None:
            qimg = rgb_to_qimage(camera_rgb)
            px = QtGui.QPixmap.fromImage(qimg).scaled(
                self.cam_label.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
            self.cam_label.setPixmap(px)
        else:
            self.cam_label.setText("Camera unavailable")

        if len(logs) != self._last_log_count:
            self._last_log_count = len(logs)
            self.log_box.setPlainText("\n".join(logs))
            cursor = self.log_box.textCursor()
            cursor.movePosition(QtGui.QTextCursor.End)
            self.log_box.setTextCursor(cursor)

    def closeEvent(self, event) -> None:  # noqa: D401
        self.controller.shutdown()
        super().closeEvent(event)


def main() -> None:
    parser = argparse.ArgumentParser(description="New G1 navigation GUI with direct SDK backend.")
    parser.add_argument("--iface", default="enp1s0", help="NIC connected to the robot")
    parser.add_argument("--map-resolution", type=float, default=0.05)
    parser.add_argument("--map-size-m", type=float, default=24.0)
    parser.add_argument("--ros-lidar-topic", default="/livox/points", help="ROS 2 PointCloud2 topic for lidar")
    parser.add_argument("--ros-rgb-topic", default="/rgbd/color/image_raw", help="ROS 2 Image topic for RGB")
    parser.add_argument("--ros-depth-topic", default="/rgbd/depth/image_raw", help="ROS 2 Image topic for depth")
    parser.add_argument("--no-ros-topics", action="store_true", help="Disable ROS 2 CycloneDDS sensor subscriptions")
    args = parser.parse_args()

    app = QtWidgets.QApplication([])
    controller = NavigationController(
        iface=args.iface,
        map_resolution=args.map_resolution,
        map_size_m=args.map_size_m,
        ros_topics_enabled=not args.no_ros_topics,
        ros_lidar_topic=args.ros_lidar_topic,
        ros_rgb_topic=args.ros_rgb_topic,
        ros_depth_topic=args.ros_depth_topic,
    )
    window = MainWindow(controller)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
