from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable


_SDK_CANDIDATES = (
    Path("/home/ag/unitree_sdk2_python"),
    Path(__file__).resolve().parents[4] / "unitree_sdk2_python",
)


def add_local_sdk_path() -> None:
    for candidate in _SDK_CANDIDATES:
        if (candidate / "unitree_sdk2py").is_dir():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return


def available_interfaces() -> list[str]:
    net_dir = Path("/sys/class/net")
    try:
        return sorted(path.name for path in net_dir.iterdir())
    except OSError:
        return []


def _interface_error(iface: str, interfaces: Iterable[str]) -> str:
    known = ", ".join(interfaces) or "none"
    return (
        f"Network interface {iface!r} was not found in /sys/class/net. "
        f"Known interfaces: {known}. "
        "Refusing to call ChannelFactoryInitialize because the Unitree SDK can abort "
        "the Python process on an invalid interface name."
    )


def require_known_interface(iface: str | None) -> str | None:
    if not iface:
        return iface

    interfaces = available_interfaces()
    if interfaces and iface not in interfaces:
        raise SystemExit(_interface_error(iface, interfaces))
    return iface


def init_channel_safely(iface: str | None) -> None:
    add_local_sdk_path()
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "unitree_sdk2py is not available. Install it or keep /home/ag/unitree_sdk2_python present."
        ) from exc

    checked_iface = require_known_interface(iface)
    try:
        if checked_iface:
            ChannelFactoryInitialize(0, checked_iface)
        else:
            ChannelFactoryInitialize(0)
    except Exception as exc:
        target = checked_iface if checked_iface else "autodetect"
        raise SystemExit(
            f"ChannelFactoryInitialize failed for {target}. "
            "This usually indicates a CycloneDDS/network setup issue."
        ) from exc


def init_channel_autodetect(iface: str | None) -> None:
    add_local_sdk_path()
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "unitree_sdk2py is not available. Install it or keep /home/ag/unitree_sdk2_python present."
        ) from exc

    require_known_interface(iface)
    try:
        ChannelFactoryInitialize(0)
    except Exception as exc:
        raise SystemExit(
            "ChannelFactoryInitialize failed in autodetect mode. "
            "The explicit interface-selection path is disabled because it aborts this SDK build; "
            "the remaining failure usually indicates CycloneDDS cannot enumerate or bind a usable NIC."
        ) from exc


def assert_go2_audio_supported() -> None:
    add_local_sdk_path()
    try:
        from unitree_sdk2py.go2.vui.vui_client import VuiClient  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "Go2 VUI support is unavailable in the installed unitree_sdk2py package."
        ) from exc

    # The SDK installed on this machine exposes Go2 VUI controls only. It does not
    # provide a Go2 audio/TTS streaming client comparable to g1.audio.AudioClient.
    if VuiClient is not None:
        raise SystemExit(
            "Go2 speaker playback/TTS is not available in the installed unitree_sdk2py SDK. "
            "This package exposes unitree_sdk2py.go2.vui.VuiClient for switch/volume controls, "
            "but no Go2 audio client. Using unitree_sdk2py.g1.audio.AudioClient against a Go2 "
            "causes the native SDK to abort."
        )
