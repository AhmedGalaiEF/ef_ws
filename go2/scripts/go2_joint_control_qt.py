import sys
import time

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

import unitree_legged_const as go2

try:
    from PySide6 import QtCore, QtWidgets
except ImportError as exc:
    raise SystemExit(
        "PySide6 is required to run this app. Install it before launching go2_joint_control_qt.py."
    ) from exc


JOINT_NAMES = [
    "FR_0",
    "FR_1",
    "FR_2",
    "FL_0",
    "FL_1",
    "FL_2",
    "RR_0",
    "RR_1",
    "RR_2",
    "RL_0",
    "RL_1",
    "RL_2",
]


class JointControlBackend:
    def __init__(self):
        self.dt = 0.002
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = None
        self.low_cmd_thread = None
        self.crc = CRC()

        self.start_pos = [0.0] * 12
        self.shutdown_start_pos = [0.0] * 12
        self.first_run = True

        self.selected_joint = 0
        self.target_offset = 0.0
        self.commanded_offsets = [0.0] * 12
        self.goal_offsets = [0.0] * 12
        self.gradient_speed = 0.8
        self.kp = 40.0
        self.kd = 3.0

        self.mode = "hold"
        self.shutdown_progress = 0.0
        self.shutdown_steps = 900
        self.shutdown_complete = False
        self.neutralize_count = 0
        self.neutralize_cycles = 1000

    def init(self):
        self._init_low_cmd()

        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self._low_state_handler, 10)

        self.sc = SportClient()
        self.sc.SetTimeout(5.0)
        self.sc.Init()

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        status, result = self.msc.CheckMode()
        while result["name"]:
            self.sc.StandDown()
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)

    def start(self):
        self.low_cmd_thread = RecurrentThread(
            interval=self.dt, target=self._low_cmd_write, name="jointcontrolcmd"
        )
        self.low_cmd_thread.Start()

    def _init_low_cmd(self):
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i in range(20):
            self.low_cmd.motor_cmd[i].mode = 0x01
            self.low_cmd.motor_cmd[i].q = go2.PosStopF
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = go2.VelStopF
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0

    def _low_state_handler(self, msg: LowState_):
        self.low_state = msg

    def _low_cmd_write(self):
        if self.low_state is None:
            return

        if self.first_run:
            for i in range(12):
                self.start_pos[i] = self.low_state.motor_state[i].q
            self.first_run = False

        if self.mode == "neutralize":
            for i in range(20):
                self.low_cmd.motor_cmd[i].q = go2.PosStopF
                self.low_cmd.motor_cmd[i].kp = 0
                self.low_cmd.motor_cmd[i].dq = go2.VelStopF
                self.low_cmd.motor_cmd[i].kd = 0
                self.low_cmd.motor_cmd[i].tau = 0
            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            self.lowcmd_publisher.Write(self.low_cmd)
            self.neutralize_count += 1
            if self.neutralize_count >= self.neutralize_cycles:
                self.shutdown_complete = True
            return

        if self.mode == "shutdown":
            self.shutdown_progress += 1.0 / self.shutdown_steps
            self.shutdown_progress = min(self.shutdown_progress, 1.0)
        else:
            step = self.gradient_speed * self.dt
            for i in range(12):
                error = self.goal_offsets[i] - self.commanded_offsets[i]
                if abs(error) <= step:
                    self.commanded_offsets[i] = self.goal_offsets[i]
                else:
                    self.commanded_offsets[i] += step if error > 0.0 else -step

        for i in range(12):
            if self.mode == "shutdown":
                target_q = (
                    (1.0 - self.shutdown_progress) * self.shutdown_start_pos[i]
                    + self.shutdown_progress * self.start_pos[i]
                )
            else:
                target_q = self.start_pos[i]
                target_q = self.start_pos[i] + self.commanded_offsets[i]

            self.low_cmd.motor_cmd[i].q = target_q
            self.low_cmd.motor_cmd[i].dq = 0
            self.low_cmd.motor_cmd[i].kp = self.kp
            self.low_cmd.motor_cmd[i].kd = self.kd
            self.low_cmd.motor_cmd[i].tau = 0

        if self.mode == "shutdown" and self.shutdown_progress >= 1.0:
            self.mode = "neutralize"

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)

    def request_shutdown(self):
        if self.low_state is None or self.shutdown_complete:
            self.mode = "neutralize"
            return
        for i in range(12):
            self.shutdown_start_pos[i] = self.low_state.motor_state[i].q
        self.shutdown_progress = 0.0
        self.mode = "shutdown"

    def measured_q(self, joint_index: int) -> float:
        if self.low_state is None:
            return 0.0
        return float(self.low_state.motor_state[joint_index].q)

    def set_joint_target_offset(self, joint_index: int, offset: float):
        for i in range(12):
            self.goal_offsets[i] = 0.0
        self.goal_offsets[joint_index] = offset
        self.target_offset = offset


class MainWindow(QtWidgets.QWidget):
    def __init__(self, backend: JointControlBackend):
        super().__init__()
        self.backend = backend
        self._updating_ui = False
        self.setWindowTitle("GO2 Joint Control")
        self.resize(420, 260)
        self._build_ui()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._refresh_status)
        self.timer.start(100)

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        layout.addWidget(QtWidgets.QLabel("Joint"))
        self.joint_combo = QtWidgets.QComboBox(self)
        self.joint_combo.addItems(JOINT_NAMES)
        self.joint_combo.currentIndexChanged.connect(self._on_joint_changed)
        layout.addWidget(self.joint_combo)

        self.target_label = QtWidgets.QLabel(self)
        layout.addWidget(self.target_label)
        self.target_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.target_slider.setRange(-700, 700)
        self.target_slider.setValue(0)
        self.target_slider.valueChanged.connect(self._on_target_changed)
        layout.addWidget(self.target_slider)

        self.kp_label = QtWidgets.QLabel(self)
        layout.addWidget(self.kp_label)
        self.kp_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.kp_slider.setRange(0, 120)
        self.kp_slider.setValue(int(self.backend.kp))
        self.kp_slider.valueChanged.connect(self._on_kp_changed)
        layout.addWidget(self.kp_slider)

        self.kd_label = QtWidgets.QLabel(self)
        layout.addWidget(self.kd_label)
        self.kd_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.kd_slider.setRange(0, 100)
        self.kd_slider.setValue(int(self.backend.kd * 10.0))
        self.kd_slider.valueChanged.connect(self._on_kd_changed)
        layout.addWidget(self.kd_slider)

        self.current_label = QtWidgets.QLabel("Measured q: waiting for lowstate...", self)
        layout.addWidget(self.current_label)

        self._sync_labels()

    def _sync_labels(self):
        self.target_label.setText(
            f"Target offset: {self.backend.target_offset:+.3f} rad"
        )
        self.kp_label.setText(f"Kp: {self.backend.kp:.1f}")
        self.kd_label.setText(f"Kd: {self.backend.kd:.1f}")

    def _sync_selected_joint_controls(self):
        joint_index = self.backend.selected_joint
        offset = self.backend.goal_offsets[joint_index]
        self._updating_ui = True
        try:
            self.target_slider.setValue(int(round(offset * 1000.0)))
        finally:
            self._updating_ui = False
        self.backend.target_offset = offset
        self._sync_labels()

    def _on_joint_changed(self, index: int):
        self.backend.selected_joint = index
        self._sync_selected_joint_controls()
        self._refresh_status()

    def _on_target_changed(self, value: int):
        if self._updating_ui:
            return
        self.backend.set_joint_target_offset(self.backend.selected_joint, value / 1000.0)
        self._sync_labels()

    def _on_kp_changed(self, value: int):
        self.backend.kp = float(value)
        self._sync_labels()

    def _on_kd_changed(self, value: int):
        self.backend.kd = value / 10.0
        self._sync_labels()

    def _refresh_status(self):
        joint_index = self.backend.selected_joint
        measured = self.backend.measured_q(joint_index)
        base = self.backend.start_pos[joint_index]
        target = base + self.backend.goal_offsets[joint_index]
        commanded = base + self.backend.commanded_offsets[joint_index]
        self.current_label.setText(
            f"Measured q: {measured:+.3f} rad | Base: {base:+.3f} rad | Commanded: {commanded:+.3f} rad | Target: {target:+.3f} rad"
        )

    def closeEvent(self, event):
        self.backend.request_shutdown()
        while not self.backend.shutdown_complete:
            QtWidgets.QApplication.processEvents()
            time.sleep(0.05)
        event.accept()


def main():
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    backend = JointControlBackend()
    backend.init()
    backend.start()

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(backend)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
