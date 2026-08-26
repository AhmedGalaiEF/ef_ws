from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "go2" / "scripts" / "make_one_step.py"


def module_with(**attributes: object) -> ModuleType:
    module = ModuleType("stub")
    for name, value in attributes.items():
        setattr(module, name, value)
    return module


def load_make_one_step(monkeypatch: pytest.MonkeyPatch):
    modules = {
        "unitree_sdk2py": module_with(),
        "unitree_sdk2py.comm": module_with(),
        "unitree_sdk2py.comm.motion_switcher": module_with(),
        "unitree_sdk2py.comm.motion_switcher.motion_switcher_client": module_with(
            MotionSwitcherClient=object
        ),
        "unitree_sdk2py.core": module_with(),
        "unitree_sdk2py.core.channel": module_with(
            ChannelFactoryInitialize=object(),
            ChannelPublisher=object,
            ChannelSubscriber=object,
        ),
        "unitree_sdk2py.idl": module_with(),
        "unitree_sdk2py.idl.default": module_with(unitree_go_msg_dds__LowCmd_=object),
        "unitree_sdk2py.idl.unitree_go": module_with(),
        "unitree_sdk2py.idl.unitree_go.msg": module_with(),
        "unitree_sdk2py.idl.unitree_go.msg.dds_": module_with(
            LowCmd_=object,
            LowState_=object,
        ),
        "unitree_sdk2py.utils": module_with(),
        "unitree_sdk2py.utils.crc": module_with(CRC=object),
        "unitree_sdk2py.utils.thread": module_with(RecurrentThread=object),
        "unitree_legged_const": module_with(),
    }
    for name, module in modules.items():
        module.__name__ = name
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "_test_make_one_step"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_parse_args_accepts_safe_step_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    make_one_step = load_make_one_step(monkeypatch)

    args = make_one_step.parse_args(
        [
            "eth0",
            "--step-x",
            "0.04",
            "--step-y",
            "0.03",
            "--turn-step",
            "0.02",
            "--lift-z",
            "0.07",
            "--shift-scale",
            "0.5",
            "--stance-fraction",
            "0.3",
        ]
    )

    assert args.iface == "eth0"
    assert args.step_x == pytest.approx(0.04)
    assert args.shift_scale == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--step-x", "nan"),
        ("--step-x", "0.5"),
        ("--step-y", "0"),
        ("--turn-step", "inf"),
        ("--lift-z", "0.01"),
        ("--shift-scale", "1.1"),
        ("--stance-fraction", "-0.1"),
    ],
)
def test_parse_args_rejects_unsafe_step_configuration(
    monkeypatch: pytest.MonkeyPatch,
    option: str,
    value: str,
) -> None:
    make_one_step = load_make_one_step(monkeypatch)

    with pytest.raises(SystemExit):
        make_one_step.parse_args([option, value])
