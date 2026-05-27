from __future__ import annotations

import os
import threading
from pathlib import Path

_CHANNEL_FACTORY_LOCK = threading.Lock()
_CHANNEL_FACTORY_CONFIG: tuple[int, str] | None = None


def _is_valid_cyclonedds_home(path: str | None) -> bool:
    if not path:
        return False
    root = Path(path).expanduser()
    return (root / "lib" / "libddsc.so").is_file()


def _iter_cyclonedds_home_candidates() -> list[Path]:
    home = Path.home()
    return [
        home / "cyclonedds_ws" / "install" / "cyclonedds",
        home / "unitree_ros2" / "cyclonedds_ws" / "install" / "cyclonedds",
        home / "academy_prep" / "academy_content" / "docs" / "repos" / "cyclonedds_0_10" / "install_0_10",
        home / "cyclonedds" / "install",
        home / "Desktop" / "unitree" / "cyclonedds" / "install",
        home / "academy_prep" / "academy_content" / "docs" / "repos" / "cyclonedds" / "install",
    ]


def _resolve_cyclonedds_home() -> str | None:
    current = os.environ.get("CYCLONEDDS_HOME")
    if _is_valid_cyclonedds_home(current):
        return str(Path(current).expanduser())

    for candidate in _iter_cyclonedds_home_candidates():
        if _is_valid_cyclonedds_home(str(candidate)):
            return str(candidate)
    return None


def _looks_like_xml(value: str | None) -> bool:
    return bool(value and value.lstrip().startswith("<"))


def _iter_cyclonedds_uri_candidates() -> list[Path]:
    home = Path.home()
    return [
        home / "cyclonedds_ws" / "cyclonedds.xml",
        home / "unitree_ros2" / "cyclonedds_ws" / "src" / "cyclonedds.xml",
        home / "Desktop" / "unitree" / "unitree_sdk2_python" / "cyclonedds.xml",
        home / "academy_prep" / "academy_content" / "docs" / "repos" / "unitree_sdk2_python" / "cyclonedds.xml",
    ]


def _resolve_cyclonedds_uri() -> str | None:
    current = os.environ.get("CYCLONEDDS_URI")
    if _looks_like_xml(current):
        return current
    if current and Path(current).expanduser().is_file():
        return str(Path(current).expanduser())

    for candidate in _iter_cyclonedds_uri_candidates():
        if candidate.is_file():
            return str(candidate)
    return None


def ensure_cyclonedds_environment() -> None:
    home = _resolve_cyclonedds_home()
    if home:
        os.environ["CYCLONEDDS_HOME"] = home

    uri = _resolve_cyclonedds_uri()
    if uri:
        os.environ["CYCLONEDDS_URI"] = uri


def ensure_channel_factory_initialized(domain_id: int = 0, iface: str = "eth0") -> None:
    """Initialize the Unitree SDK channel factory once per process."""
    global _CHANNEL_FACTORY_CONFIG

    resolved_domain = int(domain_id)
    resolved_iface = str(iface)
    ensure_cyclonedds_environment()
    with _CHANNEL_FACTORY_LOCK:
        if _CHANNEL_FACTORY_CONFIG is not None:
            if _CHANNEL_FACTORY_CONFIG != (resolved_domain, resolved_iface):
                raise RuntimeError(
                    "Unitree channel factory already initialized for "
                    f"domain={_CHANNEL_FACTORY_CONFIG[0]} iface={_CHANNEL_FACTORY_CONFIG[1]}, "
                    f"refusing domain={resolved_domain} iface={resolved_iface}."
                )
            return
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize

        ChannelFactoryInitialize(resolved_domain, resolved_iface)
        _CHANNEL_FACTORY_CONFIG = (resolved_domain, resolved_iface)


__all__ = ["ensure_channel_factory_initialized", "ensure_cyclonedds_environment"]
