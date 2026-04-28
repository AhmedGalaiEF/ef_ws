from __future__ import annotations

import ctypes as _C
import os
import sys
import time
from ctypes import POINTER, c_bool, c_char_p, c_uint8, c_uint16, c_uint32
from pathlib import Path

import numpy as np


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

    for name in ("liblivox_lidar_sdk_shared.so", "liblivox_lidar_sdk.so", "livox_lidar_sdk.dll"):
        try:
            return _C.cdll.LoadLibrary(name)
        except OSError:
            continue

    raise OSError("liblivox_lidar_sdk shared library not found. Build and install Livox-SDK2 first.")


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


class Livox2:
    def __init__(self, config_path: str | Path, host_ip: str, *, frame_time: float = 0.20, frame_packets: int = 120):
        self._config_path = os.fspath(config_path).encode()
        if not _lib.LivoxLidarSdkInit(self._config_path, host_ip.encode(), None):
            raise RuntimeError("LivoxLidarSdkInit failed. Check config path and host IP.")

        self._cb = _PointCb(self._on_packet)
        self._info_cb = _InfoChangeCb(self._on_info_change)
        _lib.SetLivoxLidarPointCloudCallBack(self._cb, None)
        _lib.SetLivoxLidarInfoChangeCallback(self._info_cb, None)

        if not _lib.LivoxLidarSdkStart():
            _lib.LivoxLidarSdkUninit()
            raise RuntimeError("LivoxLidarSdkStart failed.")

        self._running = True
        self._frame_time = float(frame_time)
        self._frame_packets = int(frame_packets)

    def shutdown(self) -> None:
        if self._running:
            _lib.LivoxLidarSdkUninit()
            self._running = False

    def handle_points(self, xyz: np.ndarray) -> None:
        print(f"frame {len(xyz)} pts")

    def _on_packet(self, handle: int, dev_type: int, pkt_ptr, _client):
        del dev_type
        pkt = pkt_ptr.contents
        count = int(pkt.dot_num)
        if count <= 0:
            return

        if pkt.data_type == 1:
            arr_type = _CartesianHighPoint * count
            points = _C.cast(pkt.data, POINTER(arr_type)).contents
            arr = np.ctypeslib.as_array(points)
            xyz = np.stack((arr["x"], arr["y"], arr["z"]), axis=1).astype(np.float32) / 1000.0
        elif pkt.data_type == 2:
            class _LowPoint(_C.Structure):
                _fields_ = [
                    ("x", _C.c_int16),
                    ("y", _C.c_int16),
                    ("z", _C.c_int16),
                    ("reflectivity", c_uint8),
                    ("tag", c_uint8),
                ]

            arr_type = _LowPoint * count
            points = _C.cast(pkt.data, POINTER(arr_type)).contents
            arr = np.ctypeslib.as_array(points)
            xyz = np.stack((arr["x"], arr["y"], arr["z"]), axis=1).astype(np.float32) / 100.0
        else:
            return

        state = self.__dict__.setdefault("_frame_state", {})
        buf, last_t = state.get(handle, ([], time.time()))
        buf.append(xyz)

        now = time.time()
        if (now - last_t) >= self._frame_time or len(buf) >= self._frame_packets:
            frame_xyz = np.concatenate(buf, axis=0)
            try:
                self.handle_points(frame_xyz)
            except Exception as exc:
                print(f"Exception in handle_points: {exc}", file=sys.stderr)
            buf = []
            last_t = now

        state[handle] = (buf, last_t)

    def _on_info_change(self, handle: int, info_ptr, _client):
        del info_ptr
        _lib.SetLivoxLidarWorkMode(handle, 1, None, None)
        _lib.EnableLivoxLidarPointSend(handle, None, None)
        _lib.SetLivoxLidarPclDataType(handle, 1, None, None)
