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


class DeepGait:
    def __init__(self):
        self.kp = 60.0
        self.kd = 5.0

        self.target_pos_1 = [
            0.0,
            1.36,
            -2.65,
            0.0,
            1.36,
            -2.65,
            -0.2,
            1.36,
            -2.65,
            0.2,
            1.36,
            -2.65,
        ]
        self.target_pos_2 = [
            0.0,
            0.67,
            -1.3,
            0.0,
            0.67,
            -1.3,
            0.0,
            0.67,
            -1.3,
            0.0,
            0.67,
            -1.3,
        ]

        self.pos_max = self.target_pos_2
        self.pos_low = self.target_pos_1
        self.pos_mid = [
            0.5 * (a + b) for a, b in zip(self.pos_max, self.pos_low)
        ]

        self.transition_progress = 0.0
        self.start_pos = [0.0] * 12
        self.shutdown_start_pos = [0.0] * 12
        self.first_run = True
        self.mode = "raise"
        self.stage = 0
        self.duration_1 = 600
        self.duration_2 = 400
        self.duration_3 = 800
        self.duration_4 = 400
        self.duration_5 = 900
        self.neutralize_count = 0
        self.neutralize_cycles = 1000
        self.shutdown_complete = False

        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = None
        self.low_cmd_thread = None
        self.crc = CRC()

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
            interval=0.002, target=self._low_cmd_write, name="deepgaitcmd"
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

        if self.mode == "raise":
            if self.stage == 0:
                src = self.start_pos
                dst = self.pos_max
                duration = self.duration_1
            elif self.stage == 1:
                src = self.pos_max
                dst = self.pos_mid
                duration = self.duration_2
            elif self.stage == 2:
                src = self.pos_mid
                dst = self.pos_mid
                duration = self.duration_3
            elif self.stage == 3:
                src = self.pos_mid
                dst = self.pos_max
                duration = self.duration_4
            elif self.stage == 4:
                src = self.pos_max
                dst = self.pos_low
                duration = self.duration_5
            else:
                src = self.pos_low
                dst = self.pos_low
                duration = None
        else:
            src = self.shutdown_start_pos
            dst = self.start_pos
            duration = self.duration_5

        if duration is not None and self.transition_progress < 1.0:
            self.transition_progress += 1.0 / duration
            self.transition_progress = min(self.transition_progress, 1.0)

        for i in range(12):
            target_q = (
                (1.0 - self.transition_progress) * src[i]
                + self.transition_progress * dst[i]
            )
            self.low_cmd.motor_cmd[i].q = target_q
            self.low_cmd.motor_cmd[i].dq = 0
            self.low_cmd.motor_cmd[i].kp = self.kp
            self.low_cmd.motor_cmd[i].kd = self.kd
            self.low_cmd.motor_cmd[i].tau = 0

        if self.transition_progress >= 1.0:
            if self.mode == "raise" and self.stage < 5:
                self.stage += 1
                if self.stage < 5:
                    self.transition_progress = 0.0
            elif self.mode == "shutdown":
                self.mode = "neutralize"

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)

    def request_shutdown(self):
        if self.low_state is None or self.shutdown_complete:
            self.mode = "neutralize"
            return
        for i in range(12):
            self.shutdown_start_pos[i] = self.low_state.motor_state[i].q
        self.transition_progress = 0.0
        self.stage = 0
        self.mode = "shutdown"


if __name__ == "__main__":
    print("WARNING: Ensure the robot has clear space and stable footing before running.")
    input("Press Enter to continue...")

    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    controller = DeepGait()
    controller.init()
    controller.start()

    print("Streaming target posture. Press Ctrl+C to lower and release motors.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nLowering to resting pose before release...")
        controller.request_shutdown()
        while not controller.shutdown_complete:
            time.sleep(0.1)
        print("Motors neutralized.")
