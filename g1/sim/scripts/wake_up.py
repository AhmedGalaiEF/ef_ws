import argparse
import os
import time

import numpy as np

from unitree_sdk2py.core import channel as channel_module
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

# Disable CycloneDDS tracing to avoid config print crash
channel_module.ChannelConfigHasInterface = """<?xml version=\"1.0\" encoding=\"UTF-8\" ?>
<CycloneDDS>
  <Domain Id=\"any\">
    <General>
      <Interfaces>
        <NetworkInterface name=\"$__IF_NAME__$\" priority=\"default\" multicast=\"default\"/>
      </Interfaces>
    </General>
  </Domain>
</CycloneDDS>"""
channel_module.ChannelConfigAutoDetermine = """<?xml version=\"1.0\" encoding=\"UTF-8\" ?>
<CycloneDDS>
  <Domain Id=\"any\">
    <General>
      <Interfaces>
        <NetworkInterface autodetermine=\"true\" priority=\"default\" multicast=\"default\" />
      </Interfaces>
    </General>
  </Domain>
</CycloneDDS>"""
os.environ.setdefault(
    "CYCLONEDDS_URI",
    "<CycloneDDS><Domain><Tracing><Category>none</Category></Tracing></Domain></CycloneDDS>",
)

G1_NUM_MOTOR = 29

# Conservative gains from the example
Kp = [
    60, 60, 60, 100, 40, 40,      # legs
    60, 60, 60, 100, 40, 40,      # legs
    60, 40, 40,                   # waist
    40, 40, 40, 40,  40, 40, 40,  # arms
    40, 40, 40, 40,  40, 40, 40   # arms
]

Kd = [
    1, 1, 1, 2, 1, 1,     # legs
    1, 1, 1, 2, 1, 1,     # legs
    1, 1, 1,              # waist
    1, 1, 1, 1, 1, 1, 1,  # arms
    1, 1, 1, 1, 1, 1, 1   # arms
]


class G1JointIndex:
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleRoll = 5
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleRoll = 11
    WaistYaw = 12
    WaistRoll = 13
    WaistPitch = 14
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20
    LeftWristYaw = 21
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27
    RightWristYaw = 28


class Mode:
    PR = 0
    AB = 1


class WakeUpController:
    def __init__(self, control_dt=0.002):
        self.control_dt_ = control_dt
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = None
        self.mode_machine_ = 0
        self.have_mode_machine_ = False
        self.crc = CRC()
        self.time_ = 0.0
        self.phase_start_ = 0.0
        self.phase_index_ = 0
        self.phase_plan_ = []

    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg
        if not self.have_mode_machine_:
            self.mode_machine_ = msg.mode_machine
            self.have_mode_machine_ = True

    def build_plan(self, base_q, scale=1.0):
        # Joint-space waypoints. Tune values as needed for your robot.
        # All poses start from current joint positions, then override key joints.
        def pose(overrides):
            q = np.array(base_q, dtype=float)
            for idx, val in overrides.items():
                q[idx] = val * scale
            return q

        # Phase 1: bring knees under body and put feet on ground.
        phase1 = pose({
            G1JointIndex.LeftHipPitch: 0.4,
            G1JointIndex.RightHipPitch: 0.4,
            G1JointIndex.LeftKnee: -0.9,
            G1JointIndex.RightKnee: -0.9,
            G1JointIndex.LeftAnklePitch: 0.5,
            G1JointIndex.RightAnklePitch: 0.5,
        })

        # Phase 2: place hands for support.
        phase2 = pose({
            G1JointIndex.LeftShoulderPitch: 0.6,
            G1JointIndex.RightShoulderPitch: 0.6,
            G1JointIndex.LeftElbow: -0.8,
            G1JointIndex.RightElbow: -0.8,
        })

        # Phase 3: push up to a crouched stand.
        phase3 = pose({
            G1JointIndex.LeftHipPitch: 0.2,
            G1JointIndex.RightHipPitch: 0.2,
            G1JointIndex.LeftKnee: -0.6,
            G1JointIndex.RightKnee: -0.6,
            G1JointIndex.LeftAnklePitch: 0.3,
            G1JointIndex.RightAnklePitch: 0.3,
            G1JointIndex.LeftShoulderPitch: 0.2,
            G1JointIndex.RightShoulderPitch: 0.2,
            G1JointIndex.LeftElbow: -0.4,
            G1JointIndex.RightElbow: -0.4,
        })

        # Phase 4: balanced stand (neutral-ish upper body).
        phase4 = pose({
            G1JointIndex.LeftHipPitch: 0.0,
            G1JointIndex.RightHipPitch: 0.0,
            G1JointIndex.LeftKnee: 0.0,
            G1JointIndex.RightKnee: 0.0,
            G1JointIndex.LeftAnklePitch: 0.0,
            G1JointIndex.RightAnklePitch: 0.0,
            G1JointIndex.LeftShoulderPitch: 0.0,
            G1JointIndex.RightShoulderPitch: 0.0,
            G1JointIndex.LeftElbow: 0.0,
            G1JointIndex.RightElbow: 0.0,
        })

        self.phase_plan_ = [
            (1.5, phase1),
            (2.0, phase2),
            (2.0, phase3),
            (2.0, phase4),
        ]

    def start(self):
        self.thread = RecurrentThread(
            interval=self.control_dt_, target=self.step, name="wake_up"
        )
        self.thread.Start()

    def step(self):
        if self.low_state is None:
            return
        if not self.have_mode_machine_:
            return

        if not self.phase_plan_:
            base_q = [self.low_state.motor_state[i].q for i in range(G1_NUM_MOTOR)]
            self.build_plan(base_q)
            self.phase_start_ = self.time_

        self.time_ += self.control_dt_

        # Determine current phase
        elapsed = self.time_ - self.phase_start_
        phase_dur, phase_target = self.phase_plan_[self.phase_index_]
        if elapsed > phase_dur:
            self.phase_index_ = min(self.phase_index_ + 1, len(self.phase_plan_) - 1)
            self.phase_start_ = self.time_
            elapsed = 0.0
            phase_dur, phase_target = self.phase_plan_[self.phase_index_]

        # Interpolate from current to target
        alpha = np.clip(elapsed / phase_dur, 0.0, 1.0)
        current_q = np.array([self.low_state.motor_state[i].q for i in range(G1_NUM_MOTOR)], dtype=float)
        desired_q = current_q * (1.0 - alpha) + phase_target * alpha

        self.low_cmd.mode_pr = Mode.PR
        self.low_cmd.mode_machine = self.mode_machine_
        for i in range(G1_NUM_MOTOR):
            self.low_cmd.motor_cmd[i].mode = 1
            self.low_cmd.motor_cmd[i].tau = 0.0
            self.low_cmd.motor_cmd[i].q = float(desired_q[i])
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kp = Kp[i]
            self.low_cmd.motor_cmd[i].kd = Kd[i]

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)


def main():
    parser = argparse.ArgumentParser(description="Wake the robot up using joint-space IK waypoints.")
    parser.add_argument("--iface", default=None, help="Network interface (robot: enp1s0, sim: lo)")
    parser.add_argument("--domain_id", type=int, default=None, help="DDS domain id (robot: 0, sim: 1)")
    args = parser.parse_args()

    if args.iface is not None:
        domain_id = 0 if args.domain_id is None else args.domain_id
        ChannelFactoryInitialize(domain_id, args.iface)
    else:
        domain_id = 1 if args.domain_id is None else args.domain_id
        ChannelFactoryInitialize(domain_id)

    ctl = WakeUpController()
    ctl.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
    ctl.lowcmd_publisher_.Init()

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(ctl.LowStateHandler, 10)

    ctl.start()
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
