from __future__ import annotations

import builtins
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "g1" / "academy" / "solved" / "lib_solved.py"


def module_with(name: str, **attributes: object) -> ModuleType:
    module = ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    return module


def load_lib_solved(monkeypatch: pytest.MonkeyPatch):
    message_symbols = {
        "String_": object,
        "SportModeState_": object,
        "HandCmd_": object,
        "HandState_": object,
        "LowCmd_": object,
        "LowState_": object,
    }
    modules = {
        "unitree_sdk2py": module_with("unitree_sdk2py"),
        "unitree_sdk2py.core": module_with("unitree_sdk2py.core"),
        "unitree_sdk2py.core.channel": module_with(
            "unitree_sdk2py.core.channel",
            ChannelFactoryInitialize=lambda *_args: None,
            ChannelPublisher=object,
            ChannelSubscriber=object,
        ),
        "unitree_sdk2py.idl": module_with("unitree_sdk2py.idl"),
        "unitree_sdk2py.idl.default": module_with(
            "unitree_sdk2py.idl.default",
            unitree_hg_msg_dds__HandCmd_=object,
            unitree_hg_msg_dds__LowCmd_=object,
        ),
        "unitree_sdk2py.idl.std_msgs": module_with("unitree_sdk2py.idl.std_msgs"),
        "unitree_sdk2py.idl.std_msgs.msg": module_with("unitree_sdk2py.idl.std_msgs.msg"),
        "unitree_sdk2py.idl.std_msgs.msg.dds_": module_with(
            "unitree_sdk2py.idl.std_msgs.msg.dds_", String_=message_symbols["String_"]
        ),
        "unitree_sdk2py.idl.unitree_go": module_with("unitree_sdk2py.idl.unitree_go"),
        "unitree_sdk2py.idl.unitree_go.msg": module_with("unitree_sdk2py.idl.unitree_go.msg"),
        "unitree_sdk2py.idl.unitree_go.msg.dds_": module_with(
            "unitree_sdk2py.idl.unitree_go.msg.dds_", SportModeState_=message_symbols["SportModeState_"]
        ),
        "unitree_sdk2py.idl.unitree_hg": module_with("unitree_sdk2py.idl.unitree_hg"),
        "unitree_sdk2py.idl.unitree_hg.msg": module_with("unitree_sdk2py.idl.unitree_hg.msg"),
        "unitree_sdk2py.idl.unitree_hg.msg.dds_": module_with(
            "unitree_sdk2py.idl.unitree_hg.msg.dds_",
            HandCmd_=message_symbols["HandCmd_"],
            HandState_=message_symbols["HandState_"],
            LowCmd_=message_symbols["LowCmd_"],
            LowState_=message_symbols["LowState_"],
        ),
        "unitree_sdk2py.rpc": module_with("unitree_sdk2py.rpc"),
        "unitree_sdk2py.rpc.client": module_with("unitree_sdk2py.rpc.client", Client=object),
        "unitree_sdk2py.utils": module_with("unitree_sdk2py.utils"),
        "unitree_sdk2py.utils.crc": module_with("unitree_sdk2py.utils.crc", CRC=object),
        "util": module_with(
            "util",
            HAND_CLOSED={},
            HAND_JOINT_NAMES={},
            HAND_OPEN={},
            play_piper_text=lambda *_args, **_kwargs: None,
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "_test_lib_solved"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def valid_right_trajectory() -> dict:
    joints = list(range(22, 29))
    return {
        "format": "trajectory_v1",
        "arm": "right",
        "joints": joints,
        "timestamps": [0.0, 0.1],
        "frames": [
            {str(joint): 0.0 for joint in joints},
            {str(joint): 0.1 for joint in joints},
        ],
    }


def test_trajectory_validation_rejects_non_finite_and_reordered_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lib = load_lib_solved(monkeypatch)
    trajectory = valid_right_trajectory()
    lib._validate_trajectory(trajectory)

    trajectory["frames"][1]["22"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        lib._validate_trajectory(trajectory)

    trajectory = valid_right_trajectory()
    trajectory["joints"] = list(reversed(trajectory["joints"]))
    with pytest.raises(ValueError, match="do not match"):
        lib._validate_trajectory(trajectory)


def test_committed_arm_sequences_pass_safety_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    lib = load_lib_solved(monkeypatch)
    sequences = lib._load_sequences()

    assert sequences
    for sequence in sequences.values():
        lib._validate_trajectory(sequence)


def test_repeat_validates_before_changing_arm_ownership(monkeypatch: pytest.MonkeyPatch) -> None:
    lib = load_lib_solved(monkeypatch)
    trajectory = valid_right_trajectory()
    trajectory["timestamps"] = [0.1, 0.1]
    lib._sequences["unsafe"] = trajectory
    release_calls = []
    monkeypatch.setattr(lib, "release_arms", lambda: release_calls.append(True))

    with pytest.raises(ValueError, match="strictly increasing"):
        lib.repeat("unsafe")

    assert release_calls == []


def test_teach_releases_arms_when_no_frames_are_captured(monkeypatch: pytest.MonkeyPatch) -> None:
    lib = load_lib_solved(monkeypatch)
    release_calls = []
    monkeypatch.setattr(lib, "arms_zero_stiffness", lambda **_kwargs: None)
    monkeypatch.setattr(lib, "release_arms", lambda: release_calls.append(True))
    monkeypatch.setattr(builtins, "input", lambda _prompt: "")

    with pytest.raises(RuntimeError, match="No teach frames"):
        lib.teach("empty", reset=True, arm="right")

    assert release_calls == [True]


def test_sequence_save_is_atomic_and_valid_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lib = load_lib_solved(monkeypatch)
    lib._sequences_path = tmp_path / "arm_sequences.json"
    lib._sequences = {"wave": valid_right_trajectory()}

    lib._save_sequences()

    assert json.loads(lib._sequences_path.read_text(encoding="utf-8")) == lib._sequences
    assert not list(tmp_path.glob(".arm_sequences.json.*.tmp"))
