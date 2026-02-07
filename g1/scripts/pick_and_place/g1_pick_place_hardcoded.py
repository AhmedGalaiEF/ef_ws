#!/usr/bin/env python3
"""Hardcoded pick/place motion for Unitree G1 (arm_sdk + Dex3 hand).

Sequence (right arm by default):
1) Open hand.
2) Move to a target pose in front of the robot.
3) Close hand with limited force (low kp + small tau).
4) Lift slightly.
5) Rotate shoulder yaw to place a few cm to the right.
6) Lower slowly and open hand to release.
7) Retract to a stow pose.

This is intentionally simple and does not use IK or perception.
"""

import argparse
import os
import time
from typing import Dict, List

from unitree_sdk2py.core import channel as channel_module
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, HandCmd_
from unitree_sdk2py.utils.crc import CRC

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

TOPIC_ARM = "rt/arm_sdk"
TOPIC_HAND_RIGHT = "rt/dex3/right/cmd"

G1_NUM_MOTOR = 29
ARM_ENABLE_IDX = 29  # enable arm_sdk when q = 1 (per Unitree docs)


class G1JointIndex:
    WaistYaw = 12
    WaistRoll = 13
    WaistPitch = 14
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27
    RightWristYaw = 28


ARM_JOINTS = [
    G1JointIndex.RightShoulderPitch,
    G1JointIndex.RightShoulderRoll,
    G1JointIndex.RightShoulderYaw,
    G1JointIndex.RightElbow,
    G1JointIndex.RightWristRoll,
    G1JointIndex.RightWristPitch,
    G1JointIndex.RightWristYaw,
]

# Hand joint order as used by unitree_sdk2py dex3 examples.
# If your hand responds incorrectly, swap index/middle pairs to match your firmware.
HAND_JOINTS = [
    "thumb_0",
    "thumb_1",
    "thumb_2",
    "middle_0",
    "middle_1",
    "index_0",
    "index_1",
]

# Open/closed reference from run_geoff_gui.py (Dex3 right hand).
HAND_OPEN = [
    -0.15717165172100067,
    -0.41322529315948486,
    0.02846403606235981,
    0.17782948911190033,
    -0.025226416066288948,
    0.17983606457710266,
    -0.027690349146723747,
]

HAND_CLOSED = [
    0.07452802360057831,
    0.9478388428688049,
    1.766921877861023,
    -1.4442411661148071,
    -1.4384468793869019,
    -1.5298594236373901,
    -1.4153316020965576,
]

# Hardcoded arm poses (rad) for right arm.
POSES: Dict[str, Dict[int, float]] = {
    # Tucked near torso.
    "stow": {
        G1JointIndex.WaistYaw: 0.0,
        G1JointIndex.RightShoulderPitch: 0.10,
        G1JointIndex.RightShoulderRoll: -0.25,
        G1JointIndex.RightShoulderYaw: 0.25,
        G1JointIndex.RightElbow: 0.75,
        G1JointIndex.RightWristRoll: -0.30,
        G1JointIndex.RightWristPitch: -0.60,
        G1JointIndex.RightWristYaw: -0.10,
    },
    # Target pose in front of robot (reach forward slightly).
    "reach_front": {
        G1JointIndex.WaistYaw: 0.0,
        G1JointIndex.RightShoulderPitch: 0.55,
        G1JointIndex.RightShoulderRoll: -0.20,
        G1JointIndex.RightShoulderYaw: 0.35,
        G1JointIndex.RightElbow: 1.15,
        G1JointIndex.RightWristRoll: -0.10,
        G1JointIndex.RightWristPitch: 0.15,
        G1JointIndex.RightWristYaw: -0.05,
    },
    # Lift slightly after grasp.
    "lift": {
        G1JointIndex.WaistYaw: 0.0,
        G1JointIndex.RightShoulderPitch: 0.40,
        G1JointIndex.RightShoulderRoll: -0.15,
        G1JointIndex.RightShoulderYaw: 0.35,
        G1JointIndex.RightElbow: 1.00,
        G1JointIndex.RightWristRoll: -0.10,
        G1JointIndex.RightWristPitch: 0.05,
        G1JointIndex.RightWristYaw: -0.05,
    },
    # Move can a few cm to the right by rotating shoulder yaw.
    "place_high": {
        G1JointIndex.WaistYaw: 0.0,
        G1JointIndex.RightShoulderPitch: 0.45,
        G1JointIndex.RightShoulderRoll: -0.25,
        G1JointIndex.RightShoulderYaw: 0.75,
        G1JointIndex.RightElbow: 1.05,
        G1JointIndex.RightWristRoll: -0.05,
        G1JointIndex.RightWristPitch: 0.10,
        G1JointIndex.RightWristYaw: 0.00,
    },
    # Lower slowly for placement.
    "place_low": {
        G1JointIndex.WaistYaw: 0.0,
        G1JointIndex.RightShoulderPitch: 0.35,
        G1JointIndex.RightShoulderRoll: -0.25,
        G1JointIndex.RightShoulderYaw: 0.75,
        G1JointIndex.RightElbow: 1.20,
        G1JointIndex.RightWristRoll: -0.05,
        G1JointIndex.RightWristPitch: -0.05,
        G1JointIndex.RightWristYaw: 0.00,
    },
}


def _build_hand_msg(targets: List[float], kp: float, kd: float, tau: float) -> HandCmd_:
    msg = unitree_hg_msg_dds__HandCmd_()
    for i in range(7):
        cmd = msg.motor_cmd[i]
        cmd.mode = 1
        cmd.tau = float(tau)
        cmd.q = float(targets[i])
        cmd.dq = 0.0
        cmd.kp = float(kp)
        cmd.kd = float(kd)
    return msg


def _apply_arm_targets(cmd: unitree_hg_msg_dds__LowCmd_, targets: Dict[int, float], kp: float, kd: float) -> None:
    for j in ARM_JOINTS + [G1JointIndex.WaistYaw]:
        if j not in targets:
            continue
        mc = cmd.motor_cmd[j]
        mc.mode = 1
        mc.q = float(targets[j])
        mc.dq = 0.0
        mc.tau = 0.0
        mc.kp = float(kp)
        mc.kd = float(kd)


def _move_arm(
    arm_pub: ChannelPublisher,
    cmd: unitree_hg_msg_dds__LowCmd_,
    crc: CRC,
    target: Dict[int, float],
    duration: float,
    rate_hz: float,
    kp: float,
    kd: float,
    hand_pub: ChannelPublisher | None = None,
    hand_hold: HandCmd_ | None = None,
) -> None:
    steps = max(1, int(duration * rate_hz))
    # Read current commanded positions as the start of interpolation.
    start = {j: float(cmd.motor_cmd[j].q) for j in target.keys()}
    dt = 1.0 / rate_hz

    for i in range(steps):
        alpha = float(i + 1) / float(steps)
        blended = {j: start[j] + (target[j] - start[j]) * alpha for j in target.keys()}
        _apply_arm_targets(cmd, blended, kp=kp, kd=kd)
        cmd.motor_cmd[ARM_ENABLE_IDX].q = 1.0
        cmd.crc = crc.Crc(cmd)
        arm_pub.Write(cmd)
        if hand_pub is not None and hand_hold is not None:
            hand_pub.Write(hand_hold)
        time.sleep(dt)


def run(iface: str, domain_id: int, rate_hz: float) -> None:
    ChannelFactoryInitialize(domain_id, iface)

    arm_pub = ChannelPublisher(TOPIC_ARM, LowCmd_)
    arm_pub.Init()

    hand_pub = ChannelPublisher(TOPIC_HAND_RIGHT, HandCmd_)
    hand_pub.Init()

    crc = CRC()
    arm_cmd = unitree_hg_msg_dds__LowCmd_()

    # Open hand first (soft, low force).
    hand_open = _build_hand_msg(HAND_OPEN, kp=1.2, kd=0.05, tau=0.05)
    hand_pub.Write(hand_open)
    time.sleep(0.5)

    # Move to stow then reach front.
    _move_arm(arm_pub, arm_cmd, crc, POSES["stow"], duration=1.5, rate_hz=rate_hz, kp=45.0, kd=1.2)
    _move_arm(arm_pub, arm_cmd, crc, POSES["reach_front"], duration=2.0, rate_hz=rate_hz, kp=45.0, kd=1.2)

    # Close hand with limited force. Small tau + low kp acts as a force limit.
    grip_close = _build_hand_msg(HAND_CLOSED, kp=1.0, kd=0.05, tau=0.20)
    for _ in range(int(1.5 * rate_hz)):
        hand_pub.Write(grip_close)
        time.sleep(1.0 / rate_hz)

    # Lift slightly while maintaining grip.
    _move_arm(
        arm_pub,
        arm_cmd,
        crc,
        POSES["lift"],
        duration=1.2,
        rate_hz=rate_hz,
        kp=45.0,
        kd=1.2,
        hand_pub=hand_pub,
        hand_hold=grip_close,
    )

    # Rotate shoulder to place a few cm away.
    _move_arm(
        arm_pub,
        arm_cmd,
        crc,
        POSES["place_high"],
        duration=1.5,
        rate_hz=rate_hz,
        kp=45.0,
        kd=1.2,
        hand_pub=hand_pub,
        hand_hold=grip_close,
    )

    # Lower slowly.
    _move_arm(
        arm_pub,
        arm_cmd,
        crc,
        POSES["place_low"],
        duration=2.5,
        rate_hz=rate_hz,
        kp=35.0,
        kd=1.0,
        hand_pub=hand_pub,
        hand_hold=grip_close,
    )

    # Release and retract.
    hand_pub.Write(hand_open)
    time.sleep(0.5)

    _move_arm(arm_pub, arm_cmd, crc, POSES["stow"], duration=2.0, rate_hz=rate_hz, kp=40.0, kd=1.1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hardcoded G1 pick/place (arm_sdk + Dex3).")
    parser.add_argument("--iface", default="eth0", help="Network interface (robot: eth0)")
    parser.add_argument("--domain_id", type=int, default=0, help="DDS domain id (robot: 0)")
    parser.add_argument("--rate", type=float, default=50.0, help="Command rate (Hz)")
    args = parser.parse_args()

    run(iface=args.iface, domain_id=args.domain_id, rate_hz=args.rate)


if __name__ == "__main__":
    main()
