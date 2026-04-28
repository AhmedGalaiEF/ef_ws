"""Minimal Livox-SDK2 packet demo.

This script shows the two stages inside the MID-360 data path:

1. A raw Livox UDP point packet arrives in the SDK callback.
2. The packet payload is decoded into a NumPy ``xyz`` array in metres.

It uses the same SDK2 callback wiring as ``livox2_python.py`` but keeps the
logic intentionally small and print-oriented for inspection/debugging.

Usage:

    python get_udp_packets.py
    python get_udp_packets.py --config mid360_config.json --host-ip 192.168.123.222
"""

from __future__ import annotations

import argparse
import ctypes as _C
import json
import os
import sys
import time
from ctypes import POINTER, c_bool, c_char_p, c_uint8, c_uint16, c_uint32
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def _load_lib():
    try:
        _C.CDLL("libstdc++.so.6", mode=_C.RTLD_GLOBAL)
    except OSError:
        pass

    env_lib = os.getenv("LIVOX_SDK2_LIB")
    if env_lib and os.path.exists(env_lib):
        try:
            return _C.cdll.LoadLibrary(env_lib)
        except OSError:
            pass

    env_dir = os.getenv("LIVOX_SDK2_DIR")
    if env_dir:
        base = Path(env_dir)
        for rel in (
            "build/liblivox_lidar_sdk_shared.so",
            "build/liblivox_lidar_sdk.so",
            "build/lib/liblivox_lidar_sdk_shared.so",
            "build/lib/liblivox_lidar_sdk.so",
            "liblivox_lidar_sdk_shared.so",
            "liblivox_lidar_sdk.so",
        ):
            candidate = base / rel
            if candidate.exists():
                try:
                    return _C.cdll.LoadLibrary(os.fspath(candidate))
                except OSError:
                    pass

    home = Path.home()
    for rel in (
        "Livox-SDK2/build/liblivox_lidar_sdk_shared.so",
        "Livox-SDK2/build/liblivox_lidar_sdk.so",
    ):
        candidate = home / rel
        if candidate.exists():
            try:
                return _C.cdll.LoadLibrary(os.fspath(candidate))
            except OSError:
                pass

    for name in (
        "liblivox_lidar_sdk_shared.so",
        "liblivox_lidar_sdk.so",
        "livox_lidar_sdk.dll",
    ):
        try:
            return _C.cdll.LoadLibrary(name)
        except OSError:
            continue

    raise OSError(
        "liblivox_lidar_sdk shared library not found. Build and install Livox-SDK2 first."
    )


_lib = _load_lib()


class _LivoxLidarEthernetPacket(_C.Structure):
    _pack_ = 1
    _fields_ = [
        ("version", c_uint8),
        ("length", c_uint16),
        ("time_interval", c_uint16),
        ("dot_num", c_uint16),
        ("udp_cnt", c_uint16),
        ("frame_cnt", c_uint8),
        ("data_type", c_uint8),
        ("time_type", c_uint8),
        ("rsvd", c_uint8 * 12),
        ("crc32", c_uint32),
        ("timestamp", c_uint8 * 8),
        ("data", c_uint8 * 1),
    ]


class _CartesianHighPoint(_C.Structure):
    _pack_ = 1
    _fields_ = [
        ("x", _C.c_int32),
        ("y", _C.c_int32),
        ("z", _C.c_int32),
        ("reflectivity", c_uint8),
        ("tag", c_uint8),
    ]


class _CartesianLowPoint(_C.Structure):
    _pack_ = 1
    _fields_ = [
        ("x", _C.c_int16),
        ("y", _C.c_int16),
        ("z", _C.c_int16),
        ("reflectivity", c_uint8),
        ("tag", c_uint8),
    ]


class _LivoxLidarInfo(_C.Structure):
    _fields_ = [
        ("dev_type", c_uint8),
        ("sn", _C.c_char * 16),
        ("lidar_ip", _C.c_char * 16),
    ]


_PointCb = _C.CFUNCTYPE(None, c_uint32, c_uint8, POINTER(_LivoxLidarEthernetPacket), _C.c_void_p)
_InfoChangeCb = _C.CFUNCTYPE(None, c_uint32, POINTER(_LivoxLidarInfo), _C.c_void_p)


_lib.LivoxLidarSdkInit.argtypes = (c_char_p, c_char_p, _C.c_void_p)
_lib.LivoxLidarSdkInit.restype = c_bool
_lib.LivoxLidarSdkStart.argtypes = ()
_lib.LivoxLidarSdkStart.restype = c_bool
_lib.LivoxLidarSdkUninit.argtypes = ()
_lib.LivoxLidarSdkUninit.restype = None
_lib.SetLivoxLidarPointCloudCallBack.argtypes = (_PointCb, _C.c_void_p)
_lib.SetLivoxLidarInfoChangeCallback.argtypes = (_InfoChangeCb, _C.c_void_p)
_lib.SetLivoxLidarWorkMode.argtypes = (c_uint32, c_uint8, _C.c_void_p, _C.c_void_p)
_lib.SetLivoxLidarWorkMode.restype = c_uint32
_lib.EnableLivoxLidarPointSend.argtypes = (c_uint32, _C.c_void_p, _C.c_void_p)
_lib.EnableLivoxLidarPointSend.restype = c_uint32
_lib.SetLivoxLidarPclDataType.argtypes = (c_uint32, c_uint8, _C.c_void_p, _C.c_void_p)


def _timestamp_hex(pkt: _LivoxLidarEthernetPacket) -> str:
    return bytes(pkt.timestamp).hex()


def _raw_payload_preview(pkt: _LivoxLidarEthernetPacket, preview_bytes: int = 32) -> str:
    count = max(0, min(preview_bytes, int(pkt.length)))
    if count == 0:
        return ""
    raw = _C.string_at(_C.byref(pkt.data), count)
    return raw.hex(" ")


def _decode_xyz(pkt: _LivoxLidarEthernetPacket) -> np.ndarray | None:
    n = int(pkt.dot_num)
    if n <= 0:
        return None

    if pkt.data_type == 1:
        arr_type = _CartesianHighPoint * n
        points = _C.cast(pkt.data, POINTER(arr_type)).contents
        arr = np.ctypeslib.as_array(points)
        return np.stack((arr["x"], arr["y"], arr["z"]), axis=1).astype(np.float32) / 1000.0

    if pkt.data_type == 2:
        arr_type = _CartesianLowPoint * n
        points = _C.cast(pkt.data, POINTER(arr_type)).contents
        arr = np.ctypeslib.as_array(points)
        return np.stack((arr["x"], arr["y"], arr["z"]), axis=1).astype(np.float32) / 100.0

    return None


class PacketPrinter:
    def __init__(self, config_path: str | Path, host_ip: str, *, max_packets: int = 20):
        self._config_path = os.fspath(config_path).encode()
        self._host_ip = host_ip
        self._max_packets = max_packets
        self._seen_packets = 0
        self._running = False

        if not _lib.LivoxLidarSdkInit(self._config_path, host_ip.encode(), None):
            raise RuntimeError("LivoxLidarSdkInit failed. Check config path and host IP.")

        self._point_cb = _PointCb(self._on_packet)
        self._info_cb = _InfoChangeCb(self._on_info_change)

        _lib.SetLivoxLidarPointCloudCallBack(self._point_cb, None)
        _lib.SetLivoxLidarInfoChangeCallback(self._info_cb, None)

        if not _lib.LivoxLidarSdkStart():
            _lib.LivoxLidarSdkUninit()
            raise RuntimeError("LivoxLidarSdkStart failed.")

        self._running = True

    def shutdown(self):
        if self._running:
            self._running = False
            _lib.LivoxLidarSdkUninit()

    def spin(self):
        try:
            while self._running:
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def _on_info_change(self, handle: int, info_ptr, _client):
        info = info_ptr.contents if info_ptr else None
        lidar_ip = ""
        if info is not None:
            lidar_ip = bytes(info.lidar_ip).decode("ascii", "ignore").rstrip("\x00")
        print(f"[info] handle={handle} lidar_ip={lidar_ip or 'unknown'}")

        kNormal = 1
        _lib.SetLivoxLidarWorkMode(handle, kNormal, None, None)
        _lib.EnableLivoxLidarPointSend(handle, None, None)
        _lib.SetLivoxLidarPclDataType(handle, 1, None, None)

    def _on_packet(self, handle: int, dev_type: int, pkt_ptr, _client):
        pkt = pkt_ptr.contents
        self._seen_packets += 1

        print(f"\n=== packet {self._seen_packets} ===")
        print(
            "[raw] "
            f"handle={handle} dev_type={dev_type} version={pkt.version} "
            f"length={pkt.length} dots={pkt.dot_num} udp_cnt={pkt.udp_cnt} "
            f"frame_cnt={pkt.frame_cnt} data_type={pkt.data_type} "
            f"timestamp={_timestamp_hex(pkt)}"
        )
        print(f"[raw] payload preview: {_raw_payload_preview(pkt)}")

        xyz = _decode_xyz(pkt)
        if xyz is None:
            print(f"[xyz] unsupported data_type={pkt.data_type}")
        else:
            print(f"[xyz] shape={xyz.shape} dtype={xyz.dtype}")
            print(f"[xyz] first 5 points:\n{xyz[:5]}")

        if self._seen_packets >= self._max_packets:
            print(f"\n[done] captured {self._seen_packets} packets, stopping.")
            self._running = False


def _ensure_default_config(path: Path, host_ip: str) -> None:
    if path.exists():
        return

    data = {
        "MID360": {
            "lidar_net_info": {
                "cmd_data_port": 56100,
                "push_msg_port": 56200,
                "point_data_port": 56300,
                "imu_data_port": 56400,
                "log_data_port": 56500,
            },
            "host_net_info": [
                {
                    "host_ip": host_ip,
                    "multicast_ip": "224.1.1.5",
                    "cmd_data_port": 56101,
                    "push_msg_port": 56201,
                    "point_data_port": 56301,
                    "imu_data_port": 56401,
                    "log_data_port": 56501,
                }
            ],
        }
    }
    path.write_text(json.dumps(data, indent=2))
    print(f"[info] wrote default config to {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Print raw Livox SDK2 packets and decoded xyz points.")
    parser.add_argument(
        "--config",
        default=os.fspath(ROOT / "mid360_config.json"),
        help="Livox SDK2 JSON config path",
    )
    parser.add_argument("--host-ip", default=os.environ.get("HOST_IP", "192.168.123.222"))
    parser.add_argument("--max-packets", type=int, default=20, help="Stop after this many packets")
    args = parser.parse_args()

    cfg = Path(args.config)
    _ensure_default_config(cfg, args.host_ip)

    reader = PacketPrinter(cfg, args.host_ip, max_packets=max(1, args.max_packets))
    reader.spin()
    return 0


if __name__ == "__main__":
    sys.exit(main())
