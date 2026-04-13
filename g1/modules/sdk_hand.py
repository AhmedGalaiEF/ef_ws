from __future__ import annotations

import os
import time
from typing import Dict

from unitree_sdk2py.core import channel as channel_module
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_


channel_module.ChannelConfigHasInterface = """<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS>
  <Domain Id="any">
    <General>
      <Interfaces>
        <NetworkInterface name="$__IF_NAME__$" priority="default" multicast="default"/>
      </Interfaces>
    </General>
  </Domain>
</CycloneDDS>"""
channel_module.ChannelConfigAutoDetermine = """<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS>
  <Domain Id="any">
    <General>
      <Interfaces>
        <NetworkInterface autodetermine="true" priority="default" multicast="default" />
      </Interfaces>
    </General>
  </Domain>
</CycloneDDS>"""
os.environ.setdefault(
    "CYCLONEDDS_URI",
    "<CycloneDDS><Domain><Tracing><Category>none</Category></Tracing></Domain></CycloneDDS>",
)


TOPIC_HAND_BY_SIDE = {
    "left": "rt/dex3/left/cmd",
    "right": "rt/dex3/right/cmd",
}

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

FINGER_TO_IDXS: Dict[str, list[int]] = {
    "thumb": [0, 1, 2],
    "middle": [3, 4],
    "index": [5, 6],
}


def build_hand_msg(targets: list[float], kp: float, kd: float, tau: float) -> HandCmd_:
    if len(targets) != 7:
        raise ValueError("Hand targets must contain 7 joint values.")
    msg = unitree_hg_msg_dds__HandCmd_()
    for idx in range(7):
        cmd = msg.motor_cmd[idx]
        cmd.mode = 1
        cmd.tau = float(tau)
        cmd.q = float(targets[idx])
        cmd.dq = 0.0
        cmd.kp = float(kp)
        cmd.kd = float(kd)
    return msg


class Dex3HandController:
    def __init__(self, hand: str = "right") -> None:
        side = str(hand).strip().lower()
        if side not in TOPIC_HAND_BY_SIDE:
            raise ValueError(f"Invalid hand '{hand}'.")
        self.hand = side
        self._pub = ChannelPublisher(TOPIC_HAND_BY_SIDE[self.hand], HandCmd_)
        self._pub.Init()

    def publish_for(self, msg: HandCmd_, seconds: float, rate_hz: float = 50.0) -> None:
        steps = max(1, int(max(0.01, float(seconds)) * max(1.0, float(rate_hz))))
        dt = 1.0 / max(1.0, float(rate_hz))
        for _ in range(steps):
            self._pub.Write(msg)
            time.sleep(dt)

    def set_targets(
        self,
        targets: list[float],
        hold_s: float = 0.6,
        rate_hz: float = 50.0,
        kp: float = 1.2,
        kd: float = 0.05,
        tau: float = 0.05,
    ) -> None:
        self.publish_for(build_hand_msg(targets, kp=kp, kd=kd, tau=tau), seconds=hold_s, rate_hz=rate_hz)

    def open(self, hold_s: float = 0.6, rate_hz: float = 50.0) -> None:
        self.set_targets(list(HAND_OPEN), hold_s=hold_s, rate_hz=rate_hz)

    def close(self, hold_s: float = 0.6, rate_hz: float = 50.0) -> None:
        self.set_targets(list(HAND_CLOSED), hold_s=hold_s, rate_hz=rate_hz, tau=0.12)

    def move_finger(self, finger_name: str, hold_s: float = 1.0, settle_s: float = 0.6, rate_hz: float = 50.0) -> None:
        finger = str(finger_name).strip().lower()
        if finger not in FINGER_TO_IDXS:
            raise ValueError(f"Unknown finger '{finger_name}'.")
        targets = list(HAND_OPEN)
        for idx in FINGER_TO_IDXS[finger]:
            targets[idx] = HAND_CLOSED[idx]
        self.open(hold_s=settle_s, rate_hz=rate_hz)
        self.set_targets(targets, hold_s=hold_s, rate_hz=rate_hz, tau=0.12)
        self.open(hold_s=settle_s, rate_hz=rate_hz)


__all__ = ["Dex3HandController", "HAND_CLOSED", "HAND_OPEN", "build_hand_msg"]
