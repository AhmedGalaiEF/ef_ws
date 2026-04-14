import math
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
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError as exc:
    raise SystemExit(
        "PySide6 is required to run this app. Install it before launching go2_gait_control_qt.py."
    ) from exc


def lerp(a, b, t):
    return a + (b - a) * t


TARGET_POS_1 = [
    0.0, 1.36, -2.65, 0.0, 1.36, -2.65,
    -0.2, 1.36, -2.65, 0.2, 1.36, -2.65,
]
TARGET_POS_2 = [
    0.0, 0.67, -1.3, 0.0, 0.67, -1.3,
    0.0, 0.67, -1.3, 0.0, 0.67, -1.3,
]
STAND_POS = [0.5 * (a + b) for a, b in zip(TARGET_POS_2, TARGET_POS_1)]

LEG_INDEX = {
    "FR": (0, 1, 2),
    "FL": (3, 4, 5),
    "RR": (6, 7, 8),
    "RL": (9, 10, 11),
}
LEG_ORDER = ["FR", "FL", "RR", "RL"]

GAITS = {
    "Walk": {
        "description": "Four-beat lateral walk: RL -> FL -> RR -> FR",
        "phase_offsets": {"RL": 0.00, "FL": 0.25, "RR": 0.50, "FR": 0.75},
        "duty": 0.75,
        "cycle_sec": 2.0,
        "step_height": 0.16,
        "step_length": 0.16,
    },
    "Amble": {
        "description": "Lateral amble: same order as walk, quicker overlap",
        "phase_offsets": {"RL": 0.00, "FL": 0.20, "RR": 0.50, "FR": 0.70},
        "duty": 0.68,
        "cycle_sec": 1.6,
        "step_height": 0.18,
        "step_length": 0.18,
    },
    "Diagonal Amble": {
        "description": "Diagonal amble: RL -> FR -> RR -> FL",
        "phase_offsets": {"RL": 0.00, "FR": 0.25, "RR": 0.50, "FL": 0.75},
        "duty": 0.68,
        "cycle_sec": 1.55,
        "step_height": 0.18,
        "step_length": 0.18,
    },
    "Trot": {
        "description": "Two-beat diagonal gait",
        "phase_offsets": {"FR": 0.00, "RL": 0.00, "FL": 0.50, "RR": 0.50},
        "duty": 0.52,
        "cycle_sec": 1.2,
        "step_height": 0.20,
        "step_length": 0.22,
    },
    "Pace": {
        "description": "Two-beat lateral gait",
        "phase_offsets": {"FR": 0.00, "RR": 0.00, "FL": 0.50, "RL": 0.50},
        "duty": 0.52,
        "cycle_sec": 1.15,
        "step_height": 0.20,
        "step_length": 0.22,
    },
    "Canter": {
        "description": "Three-beat asymmetric gait",
        "phase_offsets": {"RL": 0.00, "RR": 0.18, "FR": 0.18, "FL": 0.55},
        "duty": 0.42,
        "cycle_sec": 0.95,
        "step_height": 0.24,
        "step_length": 0.28,
    },
}


class GaitController:
    def __init__(self):
        self.dt = 0.002
        self.kp = 50.0
        self.kd = 4.0

        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = None
        self.low_cmd_thread = None
        self.crc = CRC()

        self.start_pos = [0.0] * 12
        self.shutdown_start_pos = [0.0] * 12
        self.first_run = True

        self.mode = "stand"
        self.stage = 0
        self.stage_progress = 0.0
        self.duration_1 = 600
        self.duration_2 = 400
        self.duration_3 = 800
        self.shutdown_steps = 900
        self.shutdown_progress = 0.0
        self.shutdown_complete = False
        self.neutralize_count = 0
        self.neutralize_cycles = 1000

        self.gait_name = "Walk"
        self.phase = 0.0
        self.turn_blend = 0.0
        self.move_x = 0.0
        self.move_yaw = 0.0
        self.command_move_x = 0.0
        self.command_move_yaw = 0.0
        self.command_gradient = 1.2

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
            interval=self.dt, target=self._low_cmd_write, name="gaitcontrolcmd"
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

    def _write_q(self, q):
        for i in range(12):
            self.low_cmd.motor_cmd[i].q = q[i]
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kp = self.kp
            self.low_cmd.motor_cmd[i].kd = self.kd
            self.low_cmd.motor_cmd[i].tau = 0.0
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)

    def _ramp_value(self, current, target, rate):
        step = rate * self.dt
        if abs(target - current) <= step:
            return target
        return current + step if target > current else current - step

    def _stand_target(self):
        if self.stage == 0:
            self.stage_progress += 1.0 / self.duration_1
            src = self.start_pos
            dst = TARGET_POS_2
        elif self.stage == 1:
            self.stage_progress += 1.0 / self.duration_2
            src = TARGET_POS_2
            dst = STAND_POS
        else:
            self.stage_progress += 1.0 / self.duration_3
            src = STAND_POS
            dst = STAND_POS

        self.stage_progress = min(self.stage_progress, 1.0)
        q = [lerp(src[i], dst[i], self.stage_progress) for i in range(12)]

        if self.stage_progress >= 1.0:
            self.stage += 1
            self.stage_progress = 0.0
            if self.stage > 2:
                self.mode = "gait"
        return q

    def _stance_swing_value(self, leg_phase, duty):
        if leg_phase < duty:
            stance_phase = leg_phase / duty
            return 1.0 - 2.0 * stance_phase, 0.0
        swing_phase = (leg_phase - duty) / (1.0 - duty)
        sweep = -1.0 + 2.0 * swing_phase
        lift = math.sin(math.pi * swing_phase)
        return sweep, lift

    def _gait_target(self):
        gait = GAITS[self.gait_name]
        self.command_move_x = self._ramp_value(self.command_move_x, self.move_x, self.command_gradient)
        self.command_move_yaw = self._ramp_value(self.command_move_yaw, self.move_yaw, self.command_gradient * 1.6)

        q = list(STAND_POS)
        cycle_sec = gait["cycle_sec"]
        duty = gait["duty"]
        step_length = gait["step_length"] * abs(self.command_move_x)
        step_height = gait["step_height"] * abs(self.command_move_x)
        turn_amount = 0.16 * self.command_move_yaw

        move_mag = max(abs(self.command_move_x), abs(self.command_move_yaw))
        if move_mag < 0.02:
            self.phase = 0.0
            return q

        self.phase = (self.phase + self.dt / cycle_sec) % 1.0

        for leg in LEG_ORDER:
            hip_idx, thigh_idx, calf_idx = LEG_INDEX[leg]
            leg_phase = (self.phase + gait["phase_offsets"][leg]) % 1.0
            sweep, lift = self._stance_swing_value(leg_phase, duty)

            side_sign = 1.0 if leg in ("FL", "RL") else -1.0
            front_sign = 1.0 if leg in ("FR", "FL") else -1.0

            hip_delta = -sweep * step_length * self.command_move_x
            hip_delta += side_sign * turn_amount * (0.55 if front_sign > 0 else 0.95)
            thigh_delta = -0.55 * step_height * lift
            calf_delta = 1.1 * step_height * lift

            q[hip_idx] += hip_delta
            q[thigh_idx] += thigh_delta
            q[calf_idx] += calf_delta

        return q

    def _shutdown_target(self):
        self.shutdown_progress += 1.0 / self.shutdown_steps
        self.shutdown_progress = min(self.shutdown_progress, 1.0)
        q = [
            lerp(self.shutdown_start_pos[i], self.start_pos[i], self.shutdown_progress)
            for i in range(12)
        ]
        if self.shutdown_progress >= 1.0:
            self.mode = "neutralize"
        return q

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

        if self.mode == "stand":
            q = self._stand_target()
        elif self.mode == "shutdown":
            q = self._shutdown_target()
        else:
            q = self._gait_target()

        self._write_q(q)

    def set_gait(self, gait_name):
        if gait_name in GAITS:
            self.gait_name = gait_name
            self.phase = 0.0

    def set_motion_command(self, move_x, move_yaw):
        self.move_x = max(-1.0, min(1.0, move_x))
        self.move_yaw = max(-1.0, min(1.0, move_yaw))

    def ready(self):
        return self.mode == "gait"

    def request_shutdown(self):
        if self.low_state is None or self.shutdown_complete:
            self.mode = "neutralize"
            return
        for i in range(12):
            self.shutdown_start_pos[i] = self.low_state.motor_state[i].q
        self.shutdown_progress = 0.0
        self.mode = "shutdown"


class MainWindow(QtWidgets.QWidget):
    def __init__(self, controller: GaitController):
        super().__init__()
        self.controller = controller
        self.pressed_keys = set()

        self.setWindowTitle("GO2 Gait Control")
        self.resize(520, 260)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self._build_ui()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self.timer.start(40)

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        intro = QtWidgets.QLabel(
            "Stand-up follows the low-level stand example. Use arrow keys after the robot is ready.",
            self,
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        layout.addWidget(QtWidgets.QLabel("Gait", self))
        self.gait_combo = QtWidgets.QComboBox(self)
        self.gait_combo.addItems(list(GAITS.keys()))
        self.gait_combo.currentTextChanged.connect(self._on_gait_changed)
        layout.addWidget(self.gait_combo)

        self.desc_label = QtWidgets.QLabel(self)
        self.desc_label.setWordWrap(True)
        layout.addWidget(self.desc_label)

        self.state_label = QtWidgets.QLabel("State: standing up...", self)
        layout.addWidget(self.state_label)

        self.keys_label = QtWidgets.QLabel(
            "Controls: Up/Down = forward/backward, Left/Right = turn left/right",
            self,
        )
        layout.addWidget(self.keys_label)

        self._on_gait_changed(self.gait_combo.currentText())

    def _on_gait_changed(self, gait_name):
        self.controller.set_gait(gait_name)
        self.desc_label.setText(GAITS[gait_name]["description"])

    def _update_motion_command(self):
        move_x = 0.0
        move_yaw = 0.0
        if QtCore.Qt.Key_Up in self.pressed_keys:
            move_x += 1.0
        if QtCore.Qt.Key_Down in self.pressed_keys:
            move_x -= 1.0
        if QtCore.Qt.Key_Left in self.pressed_keys:
            move_yaw += 1.0
        if QtCore.Qt.Key_Right in self.pressed_keys:
            move_yaw -= 1.0
        self.controller.set_motion_command(move_x, move_yaw)

    def _refresh(self):
        self._update_motion_command()
        state = "ready" if self.controller.ready() else self.controller.mode
        gait = self.gait_combo.currentText()
        self.state_label.setText(
            f"State: {state} | Gait: {gait} | Command x: {self.controller.move_x:+.1f} | yaw: {self.controller.move_yaw:+.1f}"
        )

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.isAutoRepeat():
            return
        self.pressed_keys.add(event.key())
        self._update_motion_command()
        event.accept()

    def keyReleaseEvent(self, event: QtGui.QKeyEvent):
        if event.isAutoRepeat():
            return
        self.pressed_keys.discard(event.key())
        self._update_motion_command()
        event.accept()

    def closeEvent(self, event):
        self.controller.set_motion_command(0.0, 0.0)
        self.controller.request_shutdown()
        while not self.controller.shutdown_complete:
            QtWidgets.QApplication.processEvents()
            time.sleep(0.05)
        event.accept()


def main():
    print("WARNING: Ensure the robot has clear space and stable footing before running.")
    input("Press Enter to continue...")

    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    controller = GaitController()
    controller.init()
    controller.start()

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(controller)
    window.show()
    window.activateWindow()
    window.setFocus()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
