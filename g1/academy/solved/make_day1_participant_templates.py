#!/usr/bin/env python3
"""Replace Day 1 reference implementations with participant exercise cells."""

import json
import sys
from pathlib import Path


def source(text: str):
    return text.splitlines(keepends=True)


TEMPLATES = {
    "task1_sdkwrapper_usage.ipynb": {
        3: '''import sys
sys.path.append("..")
from sdk_wrapper import G1

# TODO: Construct exactly one G1 for this kernel using eth0 and DDS domain 0.
g1 = None
''',
        5: '''# TODO: Ask g1 to say a short English sentence at volume 80.
code = None
print("say() return code:", code)
''',
        7: '''# TODO: Set the headlight yellow at 100% intensity for 30 seconds.
code = None
print("set_headlight() return code:", code)
''',
    },
    "task2_necessary_dds_init_pubsub.ipynb": {
        3: '''import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

_factory_config = None

def ensure_channel_factory(domain_id, interface):
    # TODO: initialise once, remember (domain_id, interface), and raise on a conflicting re-call.
    raise NotImplementedError("Implement the one-per-kernel DDS factory guard")

ensure_channel_factory(0, "eth0")
''',
        5: '''class Latest:
    def __init__(self, topic, message_type, queue_len=10):
        # TODO: create/init the subscriber and initialise the newest-message cache.
        raise NotImplementedError("Implement the asynchronous subscriber cache")

    def _callback(self, message):
        # TODO: store the message and its receipt time.
        pass

    def fresh(self, max_age_s=0.5):
        # TODO: return True only for an existing, recent message.
        raise NotImplementedError

lowstate_sub = Latest("rt/lowstate", LowState_)
''',
        7: '''def make_publisher(topic, message_type):
    # TODO: construct, initialise, and return a ChannelPublisher.
    raise NotImplementedError

def diagnose(latest, name, max_age_s=0.5):
    # TODO: return a useful no-message/fresh/stale report including message age.
    raise NotImplementedError

print(diagnose(lowstate_sub, "lowstate"))
''',
    },
    "task3_say_and_headlight_helpers.ipynb": {
        3: '''import sys
import time
sys.path.append("..")
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
from util import play_piper_text
from sdk_wrapper import ensure_channel_factory

ensure_channel_factory(0, "eth0")

# TODO: construct, set timeout, and initialise one AudioClient.
audio_client = None

def say(text, language="en", volume=100):
    # TODO: call the supplied Piper/PlayStream helper with the initialised client.
    raise NotImplementedError

print(say("Headlight and speech helpers online."))
''',
        5: '''import re

_NAMED_COLORS = {
    "white": (255, 255, 255), "red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255),
    "yellow": (255, 255, 0), "cyan": (0, 255, 255), "magenta": (255, 0, 255),
    "orange": (255, 165, 0), "purple": (128, 0, 128), "pink": (255, 105, 180),
}

def parse_color(value):
    # TODO: support a named color, #RRGGBB, and R,G,B; clamp each channel to 0..255.
    raise NotImplementedError

def scale_color(rgb, intensity):
    # TODO: clamp intensity to 0..100 and scale the RGB tuple.
    raise NotImplementedError

parse_color("cyan"), parse_color("#00ffff"), parse_color("0,255,255")
''',
        7: '''import threading
import time

def led_control_was_accepted(code):
    # The G1 LED RPC can time out (3104) even when the command was applied.
    return int(code) in (0, 3104)

class HeadlightThread(threading.Thread):
    # TODO: retain the supplied parameters, refresh LedControl until stopped/expired,
    # and always attempt to turn the LED off in finally.
    pass

_headlight_stop = None
_headlight_thread = None

def set_headlight(color="green", intensity=100, duration_s=3):
    # TODO: issue one immediate LED command; cancel/join a previous worker before
    # starting a new one for positive duration. Treat return codes 0 and 3104 as accepted.
    raise NotImplementedError

set_headlight("yellow", intensity=100, duration_s=30)
''',
    },
    "task4_robot_state_observation.ipynb": {
        2: '''import json, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
import sys
if ".." not in sys.path: sys.path.append("..")
from sdk_wrapper import ensure_channel_factory

ensure_channel_factory(0, "eth0")

class Latest:
    # TODO: cache a newest DDS message and receipt timestamp; add fresh(max_age_s).
    pass

def wait_for_latest(latest, timeout_s=3.0, max_age_s=1.0):
    # TODO: wait briefly for a fresh message, returning None if nothing arrives.
    raise NotImplementedError

# TODO: implement small DataFrame display helpers for None/dict/list results.
''',
        4: '''lowstate_sub = Latest("rt/lowstate", LowState_)

def get_lowstate(timeout_s=3.0):
    # TODO: return timestamp, q/dq/tau_est arrays, and IMU rpy/gyro/acc as a dict.
    raise NotImplementedError

# TODO: implement a compact display_lowstate(data), then call it.
''',
        6: '''from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
odom_sub = Latest("rt/odommodestate", SportModeState_)

def _first_attr(obj, names, default=None):
    # TODO: return the first available attribute name for SDK-version compatibility.
    raise NotImplementedError

def get_odommodestate(timeout_s=3.0):
    # TODO: return position, velocity, mode, and gait_type using defensive attribute lookup.
    raise NotImplementedError
''',
        8: '''import importlib

def _bms_types():
    # TODO: discover BmsState_ in the Unitree HG/GO IDL modules.
    raise NotImplementedError

BMS_TOPICS = ["rt/lf/bmsstate", "rt/lf/agvbmsstate", "rt/bmsstate", "rt/agvbmsstate"]
# TODO: create subscribers and implement get_battery(), preferring fresh dedicated BMS data.
''',
        10: '''from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
slam_info_sub = Latest("rt/slam_info", String_)
slam_key_sub = Latest("rt/slam_key_info", String_)

def get_slam_info(timeout_s=3.0):
    # TODO: read the primary JSON string, then fall back to slam_key_info.
    raise NotImplementedError
''',
        12: '''from unitree_sdk2py.idl.unitree_go.msg.dds_ import HeightMap_
occupancygrid_sub = Latest("rt/utlidar/map_state", HeightMap_, queue_len=5)

def get_occupancygrid(timeout_s=3.0):
    # TODO: validate width, height, resolution, and data length before reshaping the map grid.
    raise NotImplementedError
''',
        14: '''import os, struct

def get_rgbd(endpoints=None):
    # TODO: subscribe over ZMQ, receive RGB/depth/scale multipart data, and close each socket.
    raise NotImplementedError

# TODO: decode and display the RGB image and optional depth image.
''',
        16: '''try:
    from unitree_sdk2py.b2.robot_state.robot_state_client import RobotStateClient
except ImportError:
    from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient

# TODO: initialise RobotStateClient, define SERVICE_CATALOG, and implement get_services().
''',
    },
}


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {Path(sys.argv[0]).name} DAY1_DIRECTORY")
    day_dir = Path(sys.argv[1])
    for filename, replacements in TEMPLATES.items():
        path = day_dir / filename
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell_index, text in replacements.items():
            cell = notebook["cells"][cell_index]
            cell["source"] = source(text)
            cell["outputs"] = []
            cell["execution_count"] = None
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
