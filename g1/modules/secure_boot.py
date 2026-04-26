from __future__ import annotations

from typing import Any

from sdk_boot import force_balanced_stand_fsm, hanger_boot_sequence


def force_normal_gait(client: Any) -> int:
    """Force the locomotion client into normal gait / non-running mode."""
    last_exc: Exception | None = None
    result = 0

    if hasattr(client, "BalanceStand"):
        try:
            result = int(client.BalanceStand(0))
        except Exception as exc:
            last_exc = exc

    for method_name in ("SetGaitType", "SetBalanceMode"):
        if not hasattr(client, method_name):
            continue
        try:
            result = int(getattr(client, method_name)(0))
            break
        except Exception as exc:
            last_exc = exc

    if hasattr(client, "SetFsmId"):
        try:
            force_balanced_stand_fsm(client)
            return result
        except Exception as exc:
            last_exc = exc
            raise

    if hasattr(client, "BalanceStand"):
        return result
    if last_exc is not None:
        raise last_exc
    raise AttributeError("Current locomotion client does not support gait mode setting API.")


def secure_boot(
    iface: str,
    domain_id: int = 0,
    step: float = 0.02,
    max_height: float = 0.5,
    logger: Any | None = None,
) -> Any:
    client = hanger_boot_sequence(
        iface=iface,
        domain_id=int(domain_id),
        step=float(step),
        max_height=float(max_height),
        logger=logger,
    )
    force_normal_gait(client)
    return client


__all__ = ["force_normal_gait", "secure_boot"]
