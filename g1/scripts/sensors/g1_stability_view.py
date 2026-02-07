import argparse
import math
import threading
import time
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_

try:
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
except Exception:
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_


TOPIC_LOWSTATE = "rt/lowstate"
TOPIC_ODOM = "rt/odom"


class StabilityStream:
    def __init__(self, history_seconds: float, max_tilt_deg: float):
        self.history_seconds = float(history_seconds)
        self.max_tilt_deg = float(max_tilt_deg)
        self.lock = threading.Lock()

        self.last_lowstate = None
        self.last_odom = None

        self.ts = deque()
        self.roll = deque()
        self.pitch = deque()
        self.yaw = deque()
        self.tilt = deque()
        self.stability = deque()

        self.odom_x = deque()
        self.odom_y = deque()

    def lowstate_cb(self, msg: LowState_):
        now = time.time()
        imu = msg.imu_state
        roll = float(imu.rpy[0])
        pitch = float(imu.rpy[1])
        yaw = float(imu.rpy[2])

        tilt = math.degrees(math.sqrt(roll * roll + pitch * pitch))
        stability = max(0.0, 1.0 - (tilt / self.max_tilt_deg))

        with self.lock:
            self.last_lowstate = msg
            self.ts.append(now)
            self.roll.append(math.degrees(roll))
            self.pitch.append(math.degrees(pitch))
            self.yaw.append(math.degrees(yaw))
            self.tilt.append(tilt)
            self.stability.append(stability)
            self._trim(now)

    def odom_cb(self, msg: Odometry_):
        with self.lock:
            self.last_odom = msg
            try:
                self.odom_x.append(float(msg.pose.pose.position.x))
                self.odom_y.append(float(msg.pose.pose.position.y))
            except Exception:
                pass

    def snapshot(self):
        with self.lock:
            if not self.ts:
                return None
            return {
                "t0": self.ts[0],
                "ts": list(self.ts),
                "roll": list(self.roll),
                "pitch": list(self.pitch),
                "yaw": list(self.yaw),
                "tilt": list(self.tilt),
                "stability": list(self.stability),
                "odom_x": list(self.odom_x),
                "odom_y": list(self.odom_y),
            }

    def _trim(self, now):
        cutoff = now - self.history_seconds
        while self.ts and self.ts[0] < cutoff:
            self.ts.popleft()
            self.roll.popleft()
            self.pitch.popleft()
            self.yaw.popleft()
            self.tilt.popleft()
            self.stability.popleft()
        max_odom = max(1, int(self.history_seconds * 10))
        while len(self.odom_x) > max_odom:
            self.odom_x.popleft()
            self.odom_y.popleft()


def main():
    parser = argparse.ArgumentParser(
        description="Stream Unitree G1 IMU + odom and visualize stability."
    )
    parser.add_argument("--iface", default="eth0", help="Network interface")
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain id")
    parser.add_argument("--history", type=float, default=30.0, help="History length (seconds)")
    parser.add_argument(
        "--max-tilt-deg",
        type=float,
        default=20.0,
        help="Tilt (deg) that maps to stability=0.0",
    )
    parser.add_argument(
        "--lowstate-topic",
        default=TOPIC_LOWSTATE,
        help="Lowstate topic name",
    )
    parser.add_argument(
        "--odom-topic",
        default=TOPIC_ODOM,
        help="Odometry topic name",
    )
    args = parser.parse_args()

    print(f"Connecting via iface={args.iface} domain_id={args.domain_id}")
    ChannelFactoryInitialize(args.domain_id, args.iface)

    stream = StabilityStream(args.history, args.max_tilt_deg)

    lowstate_sub = ChannelSubscriber(args.lowstate_topic, LowState_)
    lowstate_sub.Init(stream.lowstate_cb, 10)

    odom_sub = ChannelSubscriber(args.odom_topic, Odometry_)
    odom_sub.Init(stream.odom_cb, 10)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    fig.canvas.manager.set_window_title("G1 Stability")

    ax_rpy = axes[0]
    ax_stab = axes[1]

    ax_rpy.set_title("Roll/Pitch/Yaw (deg)")
    ax_rpy.set_xlabel("Time (s)")
    ax_rpy.set_ylabel("Degrees")
    line_roll, = ax_rpy.plot([], [], label="roll")
    line_pitch, = ax_rpy.plot([], [], label="pitch")
    line_yaw, = ax_rpy.plot([], [], label="yaw")
    line_tilt, = ax_rpy.plot([], [], label="tilt")
    ax_rpy.legend(loc="upper right")

    ax_stab.set_title("Stability (1.0=level)")
    ax_stab.set_xlabel("Time (s)")
    ax_stab.set_ylabel("Score")
    line_stab, = ax_stab.plot([], [], color="tab:green")
    ax_stab.set_ylim(-0.05, 1.05)

    status = ax_stab.text(
        0.01,
        0.92,
        "Waiting for rt/lowstate...",
        transform=ax_stab.transAxes,
    )

    def _update(_frame):
        snap = stream.snapshot()
        if snap is None:
            return line_roll, line_pitch, line_yaw, line_tilt, line_stab, status

        t0 = snap["t0"]
        ts = [t - t0 for t in snap["ts"]]

        line_roll.set_data(ts, snap["roll"])
        line_pitch.set_data(ts, snap["pitch"])
        line_yaw.set_data(ts, snap["yaw"])
        line_tilt.set_data(ts, snap["tilt"])
        line_stab.set_data(ts, snap["stability"])

        ax_rpy.set_xlim(max(0.0, ts[-1] - args.history), ts[-1] + 0.1)
        all_vals = snap["roll"] + snap["pitch"] + snap["yaw"] + snap["tilt"]
        vmin = min(all_vals) - 5.0
        vmax = max(all_vals) + 5.0
        ax_rpy.set_ylim(vmin, vmax)
        ax_stab.set_xlim(max(0.0, ts[-1] - args.history), ts[-1] + 0.1)

        status_text = (
            f"tilt={snap['tilt'][-1]:.2f} deg  stability={snap['stability'][-1]:.2f}"
        )
        if snap["odom_x"] and snap["odom_y"]:
            status_text += f"  odom=({snap['odom_x'][-1]:.2f}, {snap['odom_y'][-1]:.2f})"
        status.set_text(status_text)
        return line_roll, line_pitch, line_yaw, line_tilt, line_stab, status

    FuncAnimation(fig, _update, interval=100, blit=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
