#!/usr/bin/env python3
"""Sequential Dex3 finger-motion test for Unitree G1 hand.

Moves each of the three fingers one-by-one in order:
1) thumb
2) middle
3) index

For each finger:
- start from open hand
- close only that finger
- reopen
"""

import argparse
import os
import time
from typing import Dict, List

from unitree_sdk2py.core import channel as channel_module
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_

# Disable CycloneDDS tracing to avoid config print crash.
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

TOPIC_HAND_BY_SIDE = {
    "left": "rt/dex3/left/cmd",
    "right": "rt/dex3/right/cmd",
}

# Right-hand joint order used by current scripts.
HAND_JOINTS = [
    "thumb_0",
    "thumb_1",
    "thumb_2",
    "middle_0",
    "middle_1",
    "index_0",
    "index_1",
]

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

FINGER_TO_IDXS: Dict[str, List[int]] = {
    "thumb": [0, 1, 2],
    "middle": [3, 4],
    "index": [5, 6],
}

FINGER_ORDER = ["thumb", "middle", "index"]


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


def _publish_for(pub: ChannelPublisher, msg: HandCmd_, seconds: float, rate_hz: float) -> None:
    steps = max(1, int(seconds * rate_hz))
    dt = 1.0 / rate_hz
    for _ in range(steps):
        pub.Write(msg)
        time.sleep(dt)


def _finger_target(finger_name: str) -> List[float]:
    target = list(HAND_OPEN)
    for idx in FINGER_TO_IDXS[finger_name]:
        target[idx] = HAND_CLOSED[idx]
    return target


def run(
    iface: str,
    domain_id: int,
    hand: str,
    rate_hz: float,
    hold_s: float,
    settle_s: float,
) -> None:
    ChannelFactoryInitialize(domain_id, iface)

    hand_pub = ChannelPublisher(TOPIC_HAND_BY_SIDE[hand], HandCmd_)
    hand_pub.Init()

    print(f"Opening {hand} hand")
    open_msg = _build_hand_msg(HAND_OPEN, kp=1.2, kd=0.05, tau=0.05)
    _publish_for(hand_pub, open_msg, seconds=settle_s, rate_hz=rate_hz)

    for finger in FINGER_ORDER:
        print(f"Moving finger: {finger}")

        close_one = _build_hand_msg(_finger_target(finger), kp=1.2, kd=0.05, tau=0.12)
        _publish_for(hand_pub, close_one, seconds=hold_s, rate_hz=rate_hz)

        _publish_for(hand_pub, open_msg, seconds=settle_s, rate_hz=rate_hz)

    print(f"{hand.capitalize()} hand finger test complete")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Dex3 finger motions sequentially.")
    parser.add_argument("--hand", choices=["left", "right"], default="right", help="Which hand to command")
    parser.add_argument("--iface", default="eth0", help="Network interface (robot: eth0)")
    parser.add_argument("--domain_id", type=int, default=0, help="DDS domain id (robot: 0)")
    parser.add_argument("--rate", type=float, default=50.0, help="Publish rate (Hz)")
    parser.add_argument("--hold", type=float, default=1.2, help="Seconds to hold each finger close")
    parser.add_argument("--settle", type=float, default=0.6, help="Seconds to hold open between fingers")
    args = parser.parse_args()

    run(
        iface=args.iface,
        domain_id=args.domain_id,
        hand=args.hand,
        rate_hz=args.rate,
        hold_s=args.hold,
        settle_s=args.settle,
    )


if __name__ == "__main__":
    main()
