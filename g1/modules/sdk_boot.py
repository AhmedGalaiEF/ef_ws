from __future__ import annotations

import json
import logging
import time
from typing import Optional

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_api import (
    ROBOT_API_ID_LOCO_GET_FSM_ID,
    ROBOT_API_ID_LOCO_GET_FSM_MODE,
)
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

BALANCED_STAND_FSM_ID = 501
BALANCED_STAND_FSM_IDS = frozenset((BALANCED_STAND_FSM_ID,))


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


def read_fsm_state(
    client: LocoClient,
    retries: int = 5,
    retry_delay: float = 0.1,
) -> tuple[Optional[int], Optional[int]]:
    attempts = max(1, int(retries))
    delay_s = max(0.0, float(retry_delay))
    last_id: Optional[int] = None
    last_mode: Optional[int] = None
    for attempt in range(attempts):
        cur_id = fsm_id(client)
        cur_mode = fsm_mode(client)
        if cur_id is not None:
            last_id = cur_id
        if cur_mode is not None:
            last_mode = cur_mode
        if last_id is not None and last_mode is not None:
            break
        if attempt + 1 < attempts and delay_s > 0.0:
            time.sleep(delay_s)
    return last_id, last_mode


def is_balanced_stand_state(fsm_id_value: object, fsm_mode_value: object = None) -> bool:
    try:
        return int(fsm_id_value) == BALANCED_STAND_FSM_ID
    except Exception:
        return False


def force_balanced_stand_fsm(client: LocoClient) -> int:
    if not hasattr(client, "SetFsmId"):
        raise AttributeError("Current locomotion client does not support SetFsmId().")
    return int(client.SetFsmId(BALANCED_STAND_FSM_ID))


def is_balanced_stand(client: LocoClient) -> bool:
    cur_id, cur_mode = read_fsm_state(client)
    return is_balanced_stand_state(cur_id, cur_mode)


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
        cur_id, cur_mode = read_fsm_state(bot)
        if is_balanced_stand_state(cur_id, cur_mode):
            logger.info(
                "Robot already in balanced stand (FSM %s, mode %s); skipping boot sequence.",
                cur_id,
                cur_mode,
            )
            return bot
    except Exception:
        pass

    def show(tag: str) -> None:
        cur_id, cur_mode = read_fsm_state(bot, retries=2, retry_delay=0.05)
        logger.info("%-12s -> FSM %s   mode %s", tag, cur_id, cur_mode)

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
    force_balanced_stand_fsm(bot)
    show("balanced")
    return bot


__all__ = [
    "BALANCED_STAND_FSM_ID",
    "BALANCED_STAND_FSM_IDS",
    "create_loco_client",
    "force_balanced_stand_fsm",
    "rpc_get_int",
    "read_fsm_state",
    "is_balanced_stand_state",
    "is_balanced_stand",
    "hanger_boot_sequence",
]
