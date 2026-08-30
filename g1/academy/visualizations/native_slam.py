#!/usr/bin/env python3
"""native_slam.py - standalone Unitree G1 SLAM point-cloud/map demo.

Imports ONLY from:
  - the Python standard library
  - third-party libraries (numpy, matplotlib)
  - unitree_sdk2py (Unitree's own SDK)

No local project modules are imported (no sdk_slam.py, no dds_env.py,
no sdk_client.py). Everything those modules would normally give you
(pose parsing, the SLAM RPC calls, DDS env setup, .pcd loading) is
reimplemented inline below, in plain terms, so the whole data path is
visible in one file.

Five phases, one subcommand each:

  scan          Raw LiDAR cloud, works with SLAM completely idle.
                -> "before mapping"
  start-mapping Calls the start_mapping RPC, then live-plots the
                growing SLAM map cloud.
                -> "while mapping"
  end-mapping   Calls the end_mapping RPC to save the map to a .pcd
                file on the robot's filesystem.
                -> "map is finished"
  relocate      Calls the init_pose RPC against a saved map, then
                live-plots the loaded global map cloud coming back
                over DDS.
                -> viewing the finished map live, via the robot
  plot-map      Reads a saved .pcd file straight off disk and plots
                it. No DDS, no kiss-icp, no sdk_slam module needed at
                all -- just a ~40-line PCD parser using numpy/struct.
                -> viewing the finished map, without the robot

Example:
    python3 native_slam.py scan --iface eth0
    python3 native_slam.py start-mapping --slam-type indoor
    # ... walk the robot around ...
    python3 native_slam.py end-mapping --path /home/unitree/test.pcd
    python3 native_slam.py relocate --path /home/unitree/test.pcd
    python3 native_slam.py plot-map /home/unitree/test.pcd   # local copy of the file
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import matplotlib

if not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
    # No X server reachable (headless SSH session, robot's onboard PC, ...).
    # Fall back to a non-interactive backend and save PNG snapshots instead
    # of popping up a window.
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2py.rpc.client import Client


# ---------------------------------------------------------------------------
# 0. DDS bring-up
#
# Equivalent of dds_env.ensure_cyclonedds_environment(), inlined so this file
# needs no local module. Only fills the env vars in if they are not already
# set (e.g. by your shell profile or venv activation) -- most setups will
# never need this at all.
# ---------------------------------------------------------------------------

def configure_cyclonedds_env() -> None:
    if os.environ.get("CYCLONEDDS_HOME"):
        return
    for candidate in (
        Path.home() / "cyclonedds_ws" / "install" / "cyclonedds",
        Path.home() / "unitree_ros2" / "cyclonedds_ws" / "install" / "cyclonedds",
    ):
        if (candidate / "lib" / "libddsc.so").is_file():
            os.environ["CYCLONEDDS_HOME"] = str(candidate)
            return


# ---------------------------------------------------------------------------
# 1. sensor_msgs/PointCloud2 -> (N, 3) float32 XYZ
#
# A PointCloud2 is a flat byte buffer (`msg.data`) plus a `fields` table
# telling you the byte offset of each named field ("x", "y", "z", ...)
# within one point, and `point_step`, the stride in bytes between points.
# We build a numpy structured dtype from that and let numpy do the
# unpacking in one call instead of looping in Python.
# ---------------------------------------------------------------------------

def decode_xyz(msg: PointCloud2_, max_points: int = 20000) -> np.ndarray:
    try:
        fields = {f.name: f for f in msg.fields}
        if not {"x", "y", "z"} <= set(fields):
            return np.empty((0, 3), dtype=np.float32)
        point_step = int(msg.point_step)
        data = bytes(msg.data)
        if point_step <= 0 or not data:
            return np.empty((0, 3), dtype=np.float32)
        dtype = np.dtype(
            {
                "names": ["x", "y", "z"],
                "formats": ["<f4", "<f4", "<f4"],
                "offsets": [int(fields["x"].offset), int(fields["y"].offset), int(fields["z"].offset)],
                "itemsize": point_step,
            }
        )
        raw = np.frombuffer(data, dtype=dtype, count=len(data) // point_step)
        pts = np.stack([raw["x"], raw["y"], raw["z"]], axis=1).astype(np.float32)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if max_points > 0 and pts.shape[0] > max_points:
            idx = np.linspace(0, pts.shape[0] - 1, max_points, dtype=np.int64)
            pts = pts[idx]
        return pts
    except Exception:
        return np.empty((0, 3), dtype=np.float32)


# ---------------------------------------------------------------------------
# 2. Pose from rt/slam_info / rt/slam_key_info (std_msgs/String, JSON body)
# ---------------------------------------------------------------------------

def parse_pose(payload_raw: Optional[str]) -> Optional[tuple[float, float, float]]:
    """Returns (x, y, yaw) from a slam_info/slam_key_info JSON payload, or None."""
    if not payload_raw:
        return None
    try:
        payload = json.loads(payload_raw)
        if int(payload.get("errorCode", 0)) != 0:
            return None
        cur = payload["data"]["currentPose"]
        x, y = float(cur.get("x", 0.0)), float(cur.get("y", 0.0))
        if {"q_x", "q_y", "q_z", "q_w"} <= set(cur):
            qx, qy = float(cur["q_x"]), float(cur["q_y"])
            qz, qw = float(cur["q_z"]), float(cur["q_w"])
            yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        else:
            yaw = float(cur.get("yaw", 0.0))
        if abs(x) < 1e-5 and abs(y) < 1e-5 and abs(yaw) < 1e-5:
            return None  # indistinguishable from "no pose yet"
        return (x, y, yaw)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 3. Minimal SLAM RPC client -- replaces sdk_slam.SlamOperateClient
#
# unitree_sdk2py.rpc.client.Client is the actual third-party piece doing the
# work: it registers numeric API IDs against a named DDS service and turns
# each call into a JSON request / JSON response round-trip.
# ---------------------------------------------------------------------------

API_START_MAPPING = 1801
API_END_MAPPING = 1802
API_INIT_POSE = 1804
API_POSE_NAV = 1102
API_PAUSE_NAV = 1201
API_RESUME_NAV = 1202
API_CLOSE_SLAM = 1901


class SlamRPC(Client):
    def __init__(self) -> None:
        super().__init__("slam_operate", False)

    def Init(self) -> None:
        for api_id in (
            API_START_MAPPING, API_END_MAPPING, API_INIT_POSE,
            API_POSE_NAV, API_PAUSE_NAV, API_RESUME_NAV, API_CLOSE_SLAM,
        ):
            self._RegistApi(api_id, 0)
        self._SetApiVerson("1.0.0.1")
        self.SetTimeout(10.0)

    def _call(self, api_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        code, data = self._Call(api_id, json.dumps(payload, ensure_ascii=True))
        try:
            data = json.loads(data) if isinstance(data, str) else data
        except Exception:
            pass
        return {"code": int(code), "raw": data}

    def start_mapping(self, slam_type: str = "indoor") -> dict[str, Any]:
        return self._call(API_START_MAPPING, {"data": {"slam_type": slam_type}})

    def end_mapping(self, path: str) -> dict[str, Any]:
        return self._call(API_END_MAPPING, {"data": {"address": path}})

    def init_pose(self, x: float, y: float, z: float, qx: float, qy: float, qz: float, qw: float, path: str) -> dict[str, Any]:
        return self._call(
            API_INIT_POSE,
            {"data": {"x": x, "y": y, "z": z, "q_x": qx, "q_y": qy, "q_z": qz, "q_w": qw, "address": path}},
        )

    def close_slam(self) -> dict[str, Any]:
        return self._call(API_CLOSE_SLAM, {"data": {}})


def yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5))


# ---------------------------------------------------------------------------
# 4. Generic "latest message" subscribers
# ---------------------------------------------------------------------------

class LatestCloud:
    def __init__(self, topic: str) -> None:
        self.topic = topic
        self._lock = threading.Lock()
        self._msg: Optional[PointCloud2_] = None
        self._ts = 0.0
        self._count = 0

    def start(self) -> None:
        sub = ChannelSubscriber(self.topic, PointCloud2_)
        sub.Init(self._callback, 10)

    def _callback(self, msg: PointCloud2_) -> None:
        with self._lock:
            self._msg, self._ts, self._count = msg, time.time(), self._count + 1

    def latest(self) -> tuple[Optional[PointCloud2_], float, int]:
        with self._lock:
            return self._msg, self._ts, self._count


class LatestString:
    def __init__(self, topic: str) -> None:
        self.topic = topic
        self._lock = threading.Lock()
        self._data: Optional[str] = None

    def start(self) -> None:
        sub = ChannelSubscriber(self.topic, String_)
        sub.Init(self._callback, 10)

    def _callback(self, msg: String_) -> None:
        with self._lock:
            self._data = str(msg.data)

    def latest(self) -> Optional[str]:
        with self._lock:
            return self._data


# ---------------------------------------------------------------------------
# 5. Live matplotlib top-down viewer, shared by scan / start-mapping / relocate
# ---------------------------------------------------------------------------

def live_view(topic: str, title: str, headless_prefix: str = "snapshot") -> None:
    cloud = LatestCloud(topic)
    cloud.start()
    pose_sub = LatestString("rt/slam_info")
    pose_sub.start()

    fig, ax = plt.subplots(figsize=(7, 7))
    scat = ax.scatter([], [], s=2, c="#55c7ff")
    pose_dot = ax.scatter([], [], s=120, c="#ff3154", marker="D", label="current pose")
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")

    def update(_frame):
        msg, ts, count = cloud.latest()
        if msg is None:
            return scat, pose_dot
        pts = decode_xyz(msg)
        if pts.size:
            scat.set_offsets(pts[:, :2])
            pad = 1.5
            xmin, ymin = pts[:, :2].min(axis=0) - pad
            xmax, ymax = pts[:, :2].max(axis=0) + pad
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
        pose = parse_pose(pose_sub.latest())
        if pose is not None:
            pose_dot.set_offsets([[pose[0], pose[1]]])
        age = time.time() - ts
        ax.set_title(f"{title}  |  {topic}  |  points={pts.shape[0]}  age={age:.1f}s  msgs={count}")
        return scat, pose_dot

    ani = animation.FuncAnimation(fig, update, interval=200, cache_frame_data=False)
    if matplotlib.get_backend().lower() == "agg":
        print(f"No display detected -- saving snapshots to {headless_prefix}_NNN.png every 2s. Ctrl+C to stop.")
        i = 0
        try:
            while True:
                update(i)
                fig.savefig(f"{headless_prefix}_{i:03d}.png", dpi=110)
                print(f"wrote {headless_prefix}_{i:03d}.png")
                i += 1
                time.sleep(2.0)
        except KeyboardInterrupt:
            pass
    else:
        plt.show()


# ---------------------------------------------------------------------------
# 6. Reading a saved .pcd map file straight off disk -- no DDS, no kiss-icp,
#    no sdk_slam. Just the PCD text header + numpy for the binary payload.
#    Handles ascii and binary DATA sections (not binary_compressed, which
#    needs an lzf codec).
# ---------------------------------------------------------------------------

_PCD_TYPE_MAP = {
    ("F", 4): "<f4", ("F", 8): "<f8",
    ("U", 1): "u1", ("U", 2): "<u2", ("U", 4): "<u4", ("U", 8): "<u8",
    ("I", 1): "i1", ("I", 2): "<i2", ("I", 4): "<i4", ("I", 8): "<i8",
}


def read_pcd_xyz(path: str) -> np.ndarray:
    with open(path, "rb") as fh:
        header: dict[str, str] = {}
        while True:
            line = fh.readline()
            if not line:
                raise ValueError("PCD file ended before DATA line")
            text = line.decode("ascii", errors="replace").strip()
            if not text or text.startswith("#"):
                continue
            key, _, rest = text.partition(" ")
            header[key.upper()] = rest.strip()
            if key.upper() == "DATA":
                break

        fields = header["FIELDS"].split()
        sizes = [int(v) for v in header["SIZE"].split()]
        types = header["TYPE"].split()
        counts = [int(v) for v in header.get("COUNT", " ".join(["1"] * len(fields))).split()]
        n_points = int(header["POINTS"])
        data_kind = header["DATA"].strip().lower()

        if data_kind == "ascii":
            xi, yi, zi = fields.index("x"), fields.index("y"), fields.index("z")
            pts = np.empty((n_points, 3), dtype=np.float32)
            for i in range(n_points):
                line = fh.readline()
                if not line:
                    pts = pts[:i]
                    break
                parts = line.decode("ascii").split()
                pts[i] = (float(parts[xi]), float(parts[yi]), float(parts[zi]))
            return pts

        if data_kind == "binary":
            offsets, cursor = [], 0
            for size, typ, count in zip(sizes, types, counts):
                offsets.append(cursor)
                cursor += size * count
            point_step = cursor
            names, formats, use_offsets = [], [], []
            for name, size, typ, count, off in zip(fields, sizes, types, counts, offsets):
                if count != 1:
                    continue  # skip multi-count fields, we only need x/y/z
                names.append(name)
                formats.append(_PCD_TYPE_MAP[(typ, size)])
                use_offsets.append(off)
            dtype = np.dtype({"names": names, "formats": formats, "offsets": use_offsets, "itemsize": point_step})
            raw = np.frombuffer(fh.read(point_step * n_points), dtype=dtype, count=n_points)
            return np.stack([raw["x"], raw["y"], raw["z"]], axis=1).astype(np.float32)

        raise NotImplementedError(f"DATA {data_kind!r} not supported (only ascii/binary); "
                                   "binary_compressed needs an lzf decoder.")


def plot_pcd(path: str) -> None:
    pts = read_pcd_xyz(path)
    print(f"{path}: {pts.shape[0]} points")
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(pts[:, 0], pts[:, 1], s=1, c=pts[:, 2], cmap="viridis")
    ax.set_title(f"{Path(path).name}  ({pts.shape[0]} points)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    if matplotlib.get_backend().lower() == "agg":
        out = str(Path(path).with_suffix(".png"))
        fig.savefig(out, dpi=140)
        print(f"No display detected -- wrote {out}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iface", default=os.environ.get("G1_IFACE", "eth0"))
    parser.add_argument("--domain-id", type=int, default=int(os.environ.get("G1_DOMAIN_ID", "0")))
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_scan = sub.add_parser("scan", help="Live raw LiDAR cloud (before/independent of mapping)")
    p_scan.add_argument("--topic", default="rt/utlidar/cloud_deskewed")

    p_start = sub.add_parser("start-mapping", help="Start SLAM mapping + live map cloud")
    p_start.add_argument("--slam-type", default="indoor", choices=["indoor", "outdoor"])
    p_start.add_argument("--topic", default="rt/unitree/slam_mapping/points")

    p_end = sub.add_parser("end-mapping", help="Save the finished map to a .pcd path on the robot")
    p_end.add_argument("--path", required=True)

    p_reloc = sub.add_parser("relocate", help="Load a saved map + live view of the global map cloud")
    p_reloc.add_argument("--path", required=True)
    p_reloc.add_argument("--x", type=float, default=0.0)
    p_reloc.add_argument("--y", type=float, default=0.0)
    p_reloc.add_argument("--yaw", type=float, default=0.0)
    p_reloc.add_argument("--topic", default="rt/unitree/slam_relocation/global_map")
    p_reloc.add_argument("--use-current-pose", action="store_true",
                          help="Read x/y/yaw from rt/slam_info instead of --x/--y/--yaw")

    p_plot = sub.add_parser("plot-map", help="Plot a saved .pcd file from disk directly (no DDS at all)")
    p_plot.add_argument("path")

    p_stop = sub.add_parser("stop", help="Close the active SLAM session")

    args = parser.parse_args()

    if args.cmd == "plot-map":
        plot_pcd(args.path)
        return

    configure_cyclonedds_env()
    ChannelFactoryInitialize(args.domain_id, args.iface)

    if args.cmd == "scan":
        live_view(args.topic, "Raw LiDAR (before mapping)", headless_prefix="scan")

    elif args.cmd == "start-mapping":
        rpc = SlamRPC()
        rpc.Init()
        print("start_mapping ->", rpc.start_mapping(args.slam_type))
        live_view(args.topic, "SLAM mapping (live)", headless_prefix="mapping")

    elif args.cmd == "end-mapping":
        rpc = SlamRPC()
        rpc.Init()
        print("end_mapping ->", rpc.end_mapping(args.path))

    elif args.cmd == "relocate":
        x, y, yaw = args.x, args.y, args.yaw
        if args.use_current_pose:
            pose_sub = LatestString("rt/slam_info")
            pose_sub.start()
            time.sleep(1.0)  # give it a moment to receive a message
            pose = parse_pose(pose_sub.latest())
            if pose is None:
                print("No current pose available on rt/slam_info; falling back to --x/--y/--yaw.")
            else:
                x, y, yaw = pose
        qx, qy, qz, qw = yaw_to_quat(yaw)
        rpc = SlamRPC()
        rpc.Init()
        print("init_pose ->", rpc.init_pose(x, y, 0.0, qx, qy, qz, qw, args.path))
        live_view(args.topic, "Loaded map (relocated)", headless_prefix="global_map")

    elif args.cmd == "stop":
        rpc = SlamRPC()
        rpc.Init()
        print("close_slam ->", rpc.close_slam())


if __name__ == "__main__":
    main()
