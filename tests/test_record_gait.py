from __future__ import annotations

import importlib.util
import math
import struct
import sys
from pathlib import Path
from types import ModuleType

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "go2" / "scripts" / "record_gait.py"


def load_record_gait(monkeypatch: pytest.MonkeyPatch):
    channel = ModuleType("unitree_sdk2py.core.channel")
    channel.ChannelFactoryInitialize = object()
    channel.ChannelSubscriber = object()
    low_state = ModuleType("unitree_sdk2py.idl.unitree_go.msg.dds_")
    low_state.LowState_ = object

    modules = {
        "unitree_sdk2py": ModuleType("unitree_sdk2py"),
        "unitree_sdk2py.core": ModuleType("unitree_sdk2py.core"),
        "unitree_sdk2py.core.channel": channel,
        "unitree_sdk2py.idl": ModuleType("unitree_sdk2py.idl"),
        "unitree_sdk2py.idl.unitree_go": ModuleType("unitree_sdk2py.idl.unitree_go"),
        "unitree_sdk2py.idl.unitree_go.msg": ModuleType("unitree_sdk2py.idl.unitree_go.msg"),
        "unitree_sdk2py.idl.unitree_go.msg.dds_": low_state,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "_test_record_gait"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def remote_payload(*, lx: float = 0.0, rx: float = 0.0, ry: float = 0.0, ly: float = 0.0) -> bytes:
    payload = bytearray(24)
    payload[4:8] = struct.pack("<f", lx)
    payload[8:12] = struct.pack("<f", rx)
    payload[12:16] = struct.pack("<f", ry)
    payload[20:24] = struct.pack("<f", ly)
    return bytes(payload)


def test_decode_remote_rejects_truncated_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    record_gait = load_record_gait(monkeypatch)

    with pytest.raises(ValueError, match="at least 24 bytes"):
        record_gait.decode_remote(b"short")


def test_command_features_reject_non_finite_axes(monkeypatch: pytest.MonkeyPatch) -> None:
    record_gait = load_record_gait(monkeypatch)
    remote = record_gait.decode_remote(remote_payload(ly=math.nan))

    with pytest.raises(ValueError, match="must be finite"):
        record_gait.derive_command_features(remote)


def test_recorder_counts_invalid_telemetry_without_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record_gait = load_record_gait(monkeypatch)
    recorder = record_gait.GaitRecorder("unused.jsonl", "all")
    message = type("Message", (), {"wireless_remote": b"short"})()

    recorder._low_state_handler(message)

    snapshot = recorder.snapshot()
    assert snapshot["invalid_samples"] == 1
    assert snapshot["last_sample"] is None


def test_parse_args_accepts_explicit_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    record_gait = load_record_gait(monkeypatch)

    args = record_gait.parse_args(["--iface", "eth0", "--capture-filter", "forward"])

    assert args.iface == "eth0"
    assert args.capture_filter == "forward"
