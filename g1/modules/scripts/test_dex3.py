import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_

TOPICS = {
    "right": (
        "rt/dex3/right/state",
        "rt/lf/dex3/right/state",
        "dex3/right/state",
        "lf/dex3/right/state",
    ),
    "left": (
        "rt/dex3/left/state",
        "rt/lf/dex3/left/state",
        "dex3/left/state",
        "lf/dex3/left/state",
    ),
}

seen = {topic: False for topics in TOPICS.values() for topic in topics}


def cb(topic):
    def inner(msg):
        seen[topic] = True
        print(
            f"{topic}: got state, "
            f"motors={len(msg.motor_state)}, press={len(msg.press_sensor_state)}"
        )
    return inner


ChannelFactoryInitialize(0, "eth0")

subs = []
for topics in TOPICS.values():
    for topic in topics:
        sub = ChannelSubscriber(topic, HandState_)
        sub.Init(cb(topic), 10)
        subs.append(sub)

end = time.time() + 8
while time.time() < end:
    time.sleep(0.2)

print("seen:")
for topic, value in seen.items():
    print(f"  {topic}: {value}")
