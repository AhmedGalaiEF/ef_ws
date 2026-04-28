#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

from PySide6 import QtCore, QtWidgets  # type: ignore

import slam_dual_window as dual


class G1NavControlWindow(QtWidgets.QMainWindow):
    def __init__(self, owner: "G1SlamNavWindow"):
        super().__init__()
        self._owner = owner
        self.setWindowTitle("G1 SLAM Navigation")
        self.resize(500, 500)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self.mapping_lbl = QtWidgets.QLabel("Mapping: --")
        self.slam_lbl = QtWidgets.QLabel("SLAM: --")
        self.robot_lbl = QtWidgets.QLabel("Robot: --")
        self.goal_lbl = QtWidgets.QLabel("Goal: --")
        self.avoid_lbl = QtWidgets.QLabel("Avoid: --")
        self.pose_lbl = QtWidgets.QLabel("Pose: --")
        self.help_lbl = QtWidgets.QLabel(
            "Click the occupancy map in the viewer to choose a target.\n"
            "Use Start Mapping while building the map, then Freeze Map before long navigation runs.\n"
            "W/A/S/D/Q/E still work for manual base motion when robot control is enabled."
        )
        self.help_lbl.setWordWrap(True)

        for lbl in (
            self.mapping_lbl,
            self.slam_lbl,
            self.robot_lbl,
            self.goal_lbl,
            self.avoid_lbl,
            self.pose_lbl,
        ):
            lbl.setStyleSheet("font: 12pt 'DejaVu Sans Mono';")

        layout.addWidget(self.mapping_lbl)
        layout.addWidget(self.slam_lbl)
        layout.addWidget(self.robot_lbl)
        layout.addWidget(self.goal_lbl)
        layout.addWidget(self.avoid_lbl)
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
        _btn("Freeze Map", self._owner.finish_mapping, 0, 1)
        _btn("Reset Map", self._owner.reset_mapping, 1, 0)
        _btn("Save Snapshot", self._owner.save_snapshot, 1, 1)
        _btn("Go To Goal", self._owner.start_goal_navigation, 2, 0)
        _btn("Clear Goal", self._owner.clear_goal, 2, 1)
        _btn("Stop Robot", self._owner.stop_motion, 3, 0)
        _btn("Free Walk", self._owner.enable_free_walk, 3, 1)
        _btn("Toggle Avoid", self._owner.toggle_obstacle_avoidance, 4, 0)
        _btn("Toggle 3D View", self._owner.toggle_lidar3d, 4, 1)
        _btn("Toggle RGBD", self._owner.toggle_rgbd, 5, 0)

        form = QtWidgets.QFormLayout()

        self._speed_spin = QtWidgets.QDoubleSpinBox()
        self._speed_spin.setRange(0.05, 1.00)
        self._speed_spin.setSingleStep(0.05)
        self._speed_spin.setDecimals(2)
        self._speed_spin.setValue(self._owner.nav_speed())
        self._speed_spin.valueChanged.connect(self._owner.set_nav_speed)  # type: ignore[arg-type]
        form.addRow("Speed (m/s)", self._speed_spin)

        self._duration_spin = QtWidgets.QDoubleSpinBox()
        self._duration_spin.setRange(0.20, 5.00)
        self._duration_spin.setSingleStep(0.10)
        self._duration_spin.setDecimals(2)
        self._duration_spin.setValue(self._owner.nav_cmd_duration())
        self._duration_spin.valueChanged.connect(self._owner.set_nav_cmd_duration)  # type: ignore[arg-type]
        form.addRow("Cmd Duration (s)", self._duration_spin)

        self._obstacle_spin = QtWidgets.QDoubleSpinBox()
        self._obstacle_spin.setRange(0.20, 2.00)
        self._obstacle_spin.setSingleStep(0.10)
        self._obstacle_spin.setDecimals(2)
        self._obstacle_spin.setValue(self._owner.min_obstacle_distance())
        self._obstacle_spin.valueChanged.connect(self._owner.set_min_obstacle_distance)  # type: ignore[arg-type]
        form.addRow("Obstacle Margin (m)", self._obstacle_spin)

        layout.addLayout(form)
        layout.addStretch(1)

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()

    def _refresh(self) -> None:
        self.mapping_lbl.setText(
            f"Mapping: {'ON' if dual.SLAM_SESSION.is_mapping_enabled() else 'FROZEN'}"
        )
        self.slam_lbl.setText(f"SLAM: {self._owner.slam_status_text()}")
        self.robot_lbl.setText(f"Robot: {self._owner.robot_status_text()}")
        self.goal_lbl.setText(f"Goal: {self._owner.goal_status_text()}")
        self.avoid_lbl.setText(
            f"Avoid: {'ON' if self._owner.obstacle_avoidance_enabled() else 'OFF'}"
            f"  min={self._owner.min_obstacle_distance():.2f}m"
        )
        self.pose_lbl.setText(f"Pose: {self._owner.pose_status_text()}")


class G1SlamNavWindow(dual.DualWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self.control_win.close()
        except Exception:
            pass
        self.control_win = G1NavControlWindow(self)
        self.win.setWindowTitle("G1 SLAM Navigator")
        self._nav_status = "click map to select goal"

    def robot_status_text(self) -> str:
        if not self._robot_control_enabled:
            return "disabled"
        if not self._iface_valid:
            return f"invalid iface: {self._iface_name}"
        if self._motion_proc is not None and self._motion_proc.poll() is None:
            return "connected"
        if self._robot_boot_failed:
            return "worker failed"
        return "ready"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dedicated G1 SLAM navigation GUI with click-to-goal planning."
    )
    parser.add_argument("--iface", default="eth0", help="NIC connected to the Unitree G1")
    parser.add_argument("--clear", type=float, default=18.0, help="Obstacle threshold above floor in inches")
    parser.add_argument("--rgbd-host", dest="rgbd_host", default="192.168.123.164", help="ZeroMQ RGBD host")
    parser.add_argument("--rgbd-port", type=int, default=5555, help="ZeroMQ RGBD port")
    parser.add_argument("--rgbd-topic", default="", help="Optional ZeroMQ subscription prefix")
    parser.add_argument("--gui-fps", type=float, default=8.0)
    parser.add_argument("--rgbd-fps", type=float, default=4.0)
    parser.add_argument("--slam-fps", type=float, default=2.0)
    parser.add_argument("--map-fps", type=float, default=2.0)
    parser.add_argument("--max-points", type=int, default=30000)
    parser.add_argument("--arm", choices=["left", "right"], default="left")
    parser.add_argument("--hand", choices=["left", "right"], default="left")
    parser.add_argument("--grip-force", type=float, dest="grip_force", default=0.3)
    parser.add_argument("--input", choices=("qt", "pynput", "curses"), default="qt")
    parser.add_argument("--show-rgbd", action="store_true", help="Start with RGBD panes visible")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(
        "--enable-robot-control",
        action="store_true",
        help="Robot control is enabled by default; use --no-robot-control to disable",
    )
    grp.add_argument("--no-robot-control", action="store_true", help="Disable Unitree robot control and tele-op")
    parser.add_argument("--show-livox-logs", action="store_true", help="Do not filter Livox/SDK console spam")
    args = parser.parse_args()

    dual._install_console_noise_filter(allow_livox=bool(args.show_livox_logs))
    dual._install_native_stdout_filter(allow_livox=bool(args.show_livox_logs))

    window = G1SlamNavWindow(
        args.iface,
        args.clear,
        enable_robot_control=bool(args.enable_robot_control) or (not bool(args.no_robot_control)),
        rgbd_host=args.rgbd_host,
        rgbd_port=args.rgbd_port,
        rgbd_topic=args.rgbd_topic,
        gui_fps=args.gui_fps,
        rgbd_fps=args.rgbd_fps,
        rgbd_enabled=bool(args.show_rgbd),
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
        print("[g1_slam_nav_gui] Initial arm switch failed:", exc, file=sys.stderr)

    window.run()


if __name__ == "__main__":
    main()
