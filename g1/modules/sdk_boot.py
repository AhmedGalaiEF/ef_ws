from __future__ import annotations

import json
import logging
from typing import Optional

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_api import (
    ROBOT_API_ID_LOCO_GET_FSM_ID,
    ROBOT_API_ID_LOCO_GET_FSM_MODE,
)
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient


def create_loco_client(domain_id: int, iface: str, timeout: float = 10.0) -> LocoClient:
    ChannelFactoryInitialize(int(domain_id), iface)
    client = LocoClient()
    client.SetTimeout(float(timeout))
    client.Init()
    return client


def rpc_get_int(client: LocoClient, api_id: int) -> Optional[int]:
    try:
        code, data = client._Call(api_id, "{}")  # type: ignore[attr-defined]
        if code != 0 or not data:
            return None
        return int(json.loads(data).get("data"))
    except Exception:
        return None


def fsm_id(client: LocoClient) -> Optional[int]:
    return rpc_get_int(client, ROBOT_API_ID_LOCO_GET_FSM_ID)


def fsm_mode(client: LocoClient) -> Optional[int]:
    return rpc_get_int(client, ROBOT_API_ID_LOCO_GET_FSM_MODE)


def hanger_boot_sequence(
    iface: str,
    domain_id: int = 0,
    step: float = 0.02,
    max_height: float = 0.5,
    logger: logging.Logger | None = None,
) -> LocoClient:
    if logger is None:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        logger = logging.getLogger("hanger_boot")

    bot = create_loco_client(domain_id=domain_id, iface=iface)

    try:
        cur_id = fsm_id(bot)
        cur_mode = fsm_mode(bot)
        if cur_id == 200 and cur_mode is not None and cur_mode != 2:
            logger.info(
                "Robot already in balanced stand (FSM 200, mode %s); skipping boot sequence.",
                cur_mode,
            )
            return bot
    except Exception:
        pass

    def show(tag: str) -> None:
        logger.info("%-12s -> FSM %s   mode %s", tag, fsm_id(bot), fsm_mode(bot))

    bot.Damp()
    show("damp")

    bot.SetFsmId(4)
    show("stand_up")

    while True:
        height = 0.0
        while height < max_height:
            height += step
            bot.SetStandHeight(height)
            show(f"height {height:.2f} m")
            if fsm_mode(bot) == 0 and height > 0.2:
                break

        if fsm_mode(bot) == 0:
            break

        logger.warning(
            "Feet still unloaded (mode %s) after reaching %.2f m. "
            "Adjust the hanger height, then press Enter to retry.",
            fsm_mode(bot),
            height,
        )
        try:
            bot.SetStandHeight(0.0)
            show("reset")
        except Exception:
            pass
        input()

    bot.BalanceStand(0)
    show("balance")
    bot.SetStandHeight(height)
    show("height_ok")
    bot.Start()
    show("start")
    return bot


__all__ = ["create_loco_client", "rpc_get_int", "hanger_boot_sequence"]
