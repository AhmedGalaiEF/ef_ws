from __future__ import annotations

import json
import logging
import sys
import time
from typing import Callable, Optional

from dds_env import ensure_cyclonedds_environment

ensure_cyclonedds_environment()

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
    max_attempts: int = 3,
    client_timeout: float = 2.0,
    require_confirmation: bool = True,
    interactive_retry: bool | None = None,
    retry_callback: Callable[[str, int], bool] | None = None,
    confirm_callback: Callable[[str], bool] | None = None,
    logger: logging.Logger | None = None,
) -> LocoClient:
    if logger is None:
        logger = logging.getLogger("hanger_boot")
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    bot = create_loco_client(domain_id=domain_id, iface=iface, timeout=float(client_timeout))
    attempts_limit = max(1, int(max_attempts))
    if interactive_retry is None:
        interactive_retry = True

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

    def show(tag: str) -> tuple[Optional[int], Optional[int]]:
        cur_id, cur_mode = read_fsm_state(bot, retries=3, retry_delay=0.05)
        logger.info("%-12s -> FSM %s   mode %s", tag, cur_id, cur_mode)
        return cur_id, cur_mode

    def loaded_standup_state(
        cur_id: Optional[int],
        cur_mode: Optional[int],
        current_height: float,
    ) -> bool:
        return cur_mode == 0 and (cur_id == 4 or cur_id is None) and current_height > 0.2

    def wait_for_operator(prompt: str) -> bool:
        if confirm_callback is not None:
            return bool(confirm_callback(prompt))
        if interactive_retry:
            logger.warning("%s Press Enter to continue.", prompt)
            try:
                input()
            except EOFError:
                return False
            return True
        return False

    logger.info("stand_up command -> SetFsmId(4)")
    bot.SetFsmId(4)
    show("stand_up")

    attempt = 0
    while True:
        attempt += 1
        loaded_height: Optional[float] = None
        height = 0.0
        while height < max_height:
            height += step
            logger.info("height %.2f m command -> SetStandHeight", height)
            bot.SetStandHeight(height)
            cur_id, cur_mode = show(f"height {height:.2f} m")
            if loaded_standup_state(cur_id, cur_mode, height):
                loaded_height = height
                break

        cur_id, cur_mode = read_fsm_state(bot, retries=5, retry_delay=0.08)
        if loaded_height is not None and loaded_standup_state(cur_id, cur_mode, loaded_height):
            height = loaded_height
            break
        if loaded_height is not None:
            logger.warning(
                "Loaded stand was observed at %.2f m, but the confirmation read was "
                "incomplete (FSM %s, mode %s); accepting the last good height.",
                loaded_height,
                cur_id,
                cur_mode,
            )
            height = loaded_height
            break

        logger.warning(
            "Feet still unloaded (FSM %s, mode %s) after reaching %.2f m on attempt %d/%d.",
            cur_id,
            cur_mode,
            height,
            attempt,
            attempts_limit,
        )
        try:
            bot.SetStandHeight(0.0)
            show("reset")
        except Exception:
            pass
        if attempt >= attempts_limit:
            raise TimeoutError(
                "Hanger boot did not reach a loaded stand state after "
                f"{attempts_limit} attempt(s). Adjust the hanger height/support and retry."
            )
        prompt = (
            "Feet still unloaded after the stand-height sweep. Adjust the hanger "
            "height/support so the soles are just in contact with the ground."
        )
        if retry_callback is not None:
            if not bool(retry_callback(prompt, attempt)):
                raise TimeoutError(f"Boot retry cancelled after attempt {attempt}.")
            continue
        if interactive_retry:
            logger.warning("%s Press Enter to retry.", prompt)
            try:
                input()
            except EOFError as exc:
                raise TimeoutError("Boot retry needs operator confirmation, but input is unavailable.") from exc
            continue
        logger.warning(
            "Retrying stand-height sweep automatically (%d/%d).",
            attempt + 1,
            attempts_limit,
        )

    if require_confirmation and not wait_for_operator(
        "Robot appears loaded. Confirm before commanding balanced stand."
    ):
        raise TimeoutError("Balanced-stand confirmation was not received.")

    bot.BalanceStand(0)
    show("balance")
    bot.SetStandHeight(height)
    show("height_ok")
    bot.Start()
    show("start")
    for _ in range(3):
        force_balanced_stand_fsm(bot)
        cur_id, cur_mode = show("balanced")
        if is_balanced_stand_state(cur_id, cur_mode):
            break
        time.sleep(0.1)
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
