#!/usr/bin/env python3
from __future__ import annotations

import time
from typing import Any

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize


AI_MODE_NAME = "ai_sport"
_CHANNEL_FACTORY_CONFIG: tuple[int, str | None] | None = None


def _normalize_iface(iface: str | None) -> str | None:
    if iface is None:
        return None
    value = str(iface).strip()
    if not value or value.lower() in {"auto", "default", "none"}:
        return None
    return value


def _ensure_channel_factory_initialized(domain_id: int, iface: str | None) -> None:
    global _CHANNEL_FACTORY_CONFIG

    config = (int(domain_id), _normalize_iface(iface))
    if _CHANNEL_FACTORY_CONFIG is None:
        ChannelFactoryInitialize(config[0], config[1])
        _CHANNEL_FACTORY_CONFIG = config
        return
    if _CHANNEL_FACTORY_CONFIG != config:
        raise RuntimeError(
            "ChannelFactoryInitialize() was already called with "
            f"domain={_CHANNEL_FACTORY_CONFIG[0]} iface={_CHANNEL_FACTORY_CONFIG[1]!r}; "
            f"refusing domain={config[0]} iface={config[1]!r}."
        )


def _motion_client(rpc_timeout: float) -> MotionSwitcherClient:
    client = MotionSwitcherClient()
    client.SetTimeout(float(rpc_timeout))
    client.Init()
    return client


def _mode_name(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    for key in ("name", "mode", "alias"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _result_code(result: Any) -> int:
    if isinstance(result, tuple):
        return int(result[0])
    return int(result)


def enter_dev_mode(
    iface: str = "eth0",
    domain_id: int = 0,
    timeout: float = 10.0,
    rpc_timeout: float = 5.0,
    poll_interval: float = 0.5,
) -> dict[str, Any]:
    """Release MotionSwitcher so low-level / developer commands are accepted."""
    _ensure_channel_factory_initialized(domain_id, iface)
    motion = _motion_client(rpc_timeout)
    deadline = time.monotonic() + float(timeout)

    while time.monotonic() < deadline:
        code, data = motion.CheckMode()
        if int(code) != 0:
            raise RuntimeError(f"CheckMode failed: code={int(code)} data={data!r}")
        if not _mode_name(data):
            return {"active": False, "mode": None, "raw": data}

        release_code = _result_code(motion.ReleaseMode())
        if release_code != 0:
            raise RuntimeError(f"ReleaseMode failed: code={release_code}")
        time.sleep(float(poll_interval))

    raise TimeoutError("Timed out waiting for MotionSwitcher to release.")


def exit_dev_mode(
    iface: str = "eth0",
    domain_id: int = 0,
    timeout: float = 10.0,
    rpc_timeout: float = 5.0,
    poll_interval: float = 0.5,
    mode_name: str = AI_MODE_NAME,
) -> dict[str, Any]:
    """Select the normal AI motion mode again, which exits developer mode."""
    _ensure_channel_factory_initialized(domain_id, iface)
    motion = _motion_client(rpc_timeout)
    deadline = time.monotonic() + float(timeout)

    while time.monotonic() < deadline:
        code, data = motion.CheckMode()
        if int(code) != 0:
            raise RuntimeError(f"CheckMode failed: code={int(code)} data={data!r}")

        current_mode = _mode_name(data)
        if current_mode == mode_name:
            return {"active": True, "mode": current_mode, "raw": data}

        select_code = _result_code(motion.SelectMode(mode_name))
        if select_code != 0:
            raise RuntimeError(f"SelectMode({mode_name!r}) failed: code={select_code}")
        time.sleep(float(poll_interval))

    raise TimeoutError(f"Timed out waiting for MotionSwitcher to select {mode_name!r}.")


# Shorter versions using the local helper modules in `..`:
#
# import os
# import sys
#
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
#
# from ll_sdk import LLSdk
#
# sdk = LLSdk(iface="eth0", domain_id=0)
# sdk.enter_dev_mode(timeout=10.0)
#
# from dds_env import ensure_channel_factory_initialized
#
# There is no matching exit_dev_mode() wrapper in ../ll_sdk.py or ../sdk_client.py
# right now, so the short version still uses MotionSwitcherClient directly:
#
# ensure_channel_factory_initialized(0, "eth0")
# motion = MotionSwitcherClient()
# motion.SetTimeout(5.0)
# motion.Init()
# motion.SelectMode("ai_sport")
