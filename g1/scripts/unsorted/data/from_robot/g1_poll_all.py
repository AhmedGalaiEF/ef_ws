#!/usr/bin/env python3
"""
Best-effort polling and discovery for Unitree G1 DDS topics and SDK clients.

- Discovers DDS topics via Cyclone DDS built-in topics.
- Tries to map discovered DDS type names to unitree_sdk2py IDL classes.
- Subscribes to configured topics (from JSON/YAML or built-in profiles) and logs received samples.
- Optionally scans SDK client classes and logs which ones can be imported/instantiated.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import pkgutil
import sys
import time
from dataclasses import is_dataclass, asdict
from types import ModuleType
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

def _configure_logging(level: int) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


LOG = logging.getLogger("g1_poll_all")
RGBD_TOPIC_KEYWORDS = ("rgbd", "depth", "rgb", "color", "image", "camera")


class TopicStats:
    def __init__(self) -> None:
        self.sample_count: int = 0
        self.last_sample: str = ""
        self.last_ts: float = 0.0
        self.last_error: str = ""

    def update_sample(self, sample: Any) -> None:
        self.sample_count += 1
        self.last_sample = _truncate(_format_sample(sample), 400)
        self.last_ts = time.time()
        self.last_error = ""

    def update_error(self, exc: Exception) -> None:
        self.last_error = str(exc)
        self.last_ts = time.time()


def _try_import(module_path: str) -> Optional[ModuleType]:
    try:
        return importlib.import_module(module_path)
    except Exception:
        LOG.debug("Failed to import %s", module_path, exc_info=True)
        return None


def _load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    if path.endswith(".json"):
        return json.loads(data)
    if path.endswith(".yaml") or path.endswith(".yml"):
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError("PyYAML not installed but YAML config provided") from exc
        return yaml.safe_load(data) or {}
    # Try JSON as a fallback
    return json.loads(data)


def _type_name_to_idl_module(type_name: str) -> Optional[Tuple[str, str]]:
    """
    Map DDS type names like 'unitree_go::msg::dds_::LowState_' to
    unitree_sdk2py IDL module/class names.

    Returns (module_path, class_name) or None if mapping is impossible.
    """
    if not type_name or "::" not in type_name:
        return None
    parts = type_name.split("::")
    if len(parts) < 2:
        return None
    class_name = parts[-1]
    namespace = parts[:-1]

    # unitree SDK2 Python IDL modules are under unitree_sdk2py.idl.<namespace>.<class_module>
    # and message modules are named with a leading underscore, e.g. _LowState_.py
    module_path = "unitree_sdk2py.idl." + ".".join(namespace) + "._" + class_name
    return module_path, class_name


def _format_sample(sample: Any) -> str:
    try:
        if is_dataclass(sample):
            return json.dumps(asdict(sample), ensure_ascii=True)
    except Exception:
        pass
    # If message has a to_dict() helper
    try:
        if hasattr(sample, "to_dict"):
            return json.dumps(sample.to_dict(), ensure_ascii=True)
    except Exception:
        pass
    try:
        return repr(sample)
    except Exception:
        return "<unprintable sample>"


class DdsDiscovery:
    def __init__(self, domain_id: int) -> None:
        from cyclonedds.domain import DomainParticipant
        from cyclonedds.builtin import BuiltinDataReader, BuiltinTopicDcpsTopic, BuiltinTopicDcpsPublication

        self.participant = DomainParticipant(domain_id)
        self.topic_reader = BuiltinDataReader(self.participant, BuiltinTopicDcpsTopic)
        self.pub_reader = BuiltinDataReader(self.participant, BuiltinTopicDcpsPublication)

        self.seen_topics: Dict[str, str] = {}
        self.seen_publications: Set[Tuple[str, str]] = set()

    def poll(self) -> List[Tuple[str, str]]:
        discovered: List[Tuple[str, str]] = []
        try:
            topics = self.topic_reader.read(64)
        except Exception:
            LOG.debug("Failed to read builtin topics", exc_info=True)
            topics = []

        for t in topics:
            try:
                topic_name = t.topic_name
                type_name = t.type_name
            except Exception:
                continue
            if topic_name not in self.seen_topics:
                self.seen_topics[topic_name] = type_name
                discovered.append((topic_name, type_name))

        try:
            pubs = self.pub_reader.read(64)
        except Exception:
            pubs = []

        for p in pubs:
            try:
                key = (p.topic_name, p.type_name)
            except Exception:
                continue
            if key not in self.seen_publications:
                self.seen_publications.add(key)
        return discovered


class TopicReader:
    def __init__(self, participant: Any, topic_name: str, msg_type: Any, type_label: str) -> None:
        from cyclonedds.topic import Topic
        from cyclonedds.sub import DataReader

        self.topic_name = topic_name
        self.msg_type = msg_type
        self.type_label = type_label
        self.topic = Topic(participant, topic_name, msg_type)
        self.reader = DataReader(participant, self.topic)

    def read(self, max_samples: int = 8) -> Iterable[Any]:
        try:
            return self.reader.read(max_samples)
        except Exception as exc:
            raise RuntimeError(f"read failed for {self.topic_name}") from exc


def _resolve_type(type_path: str) -> Any:
    """Resolve a python type from a dotted path like package.module:ClassName or package.module.ClassName."""
    if ":" in type_path:
        module_path, class_name = type_path.split(":", 1)
    else:
        module_path, class_name = type_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _type_path_candidates(type_path: str) -> List[str]:
    """
    Return candidate python type paths for a given DDS type name or python path.
    """
    candidates: List[str] = []
    if "::" not in type_path:
        return [type_path]

    # Primary mapping based on DDS namespace
    mapped = _type_name_to_idl_module(type_path)
    if mapped:
        module_path, class_name = mapped
        candidates.append(f"{module_path}:{class_name}")

    # Common ROS2 IDL location in unitree_sdk2py
    if type_path.startswith("sensor_msgs::msg::dds_::"):
        class_name = type_path.split("::")[-1]
        candidates.append(f"unitree_sdk2py.idl.ros2._{class_name}:{class_name}")

    # Fallback to raw input (may already be a python path even if it contains ::)
    candidates.append(type_path)
    return candidates


def _build_profiles() -> Dict[str, Dict[str, Any]]:
    # Canonical profiles based on provided G1 docs
    return {
        "g1_basic": {
            "topics": [
                {"topic": "rt/lowstate", "type": "unitree_hg::msg::dds_::LowState_"},
                # Publish topics included for inspection only
                {"topic": "rt/lowcmd", "type": "unitree_hg::msg::dds_::LowCmd_"},
                {"topic": "rt/dex3/left/state", "type": "unitree_hg::msg::dds_::HandState_"},
                {"topic": "rt/dex3/right/state", "type": "unitree_hg::msg::dds_::HandState_"},
                {"topic": "rt/dex3/left/cmd", "type": "unitree_hg::msg::dds_::HandCmd_"},
                {"topic": "rt/dex3/right/cmd", "type": "unitree_hg::msg::dds_::HandCmd_"},
            ],
        },
        "g1_sport": {
            "topics": [
                {"topic": "rt/arm_sdk", "type": "unitree_hg::msg::dds_::LowCmd_"},
            ],
        },
        "g1_odom": {
            "topics": [
                {"topic": "rt/odommodestate", "type": "unitree_go::msg::dds_::SportModeState_"},
                {"topic": "rt/lf/odommodestate", "type": "unitree_go::msg::dds_::SportModeState_"},
            ],
        },
        "g1_lidar": {
            "topics": [
                {"topic": "rt/utlidar/cloud_livox_mid360", "type": "sensor_msgs::msg::dds_::PointCloud2_"},
                {"topic": "rt/utlidar/imu_livox_mid360", "type": "sensor_msgs::msg::dds_::Imu_"},
            ],
        },
    }


def _build_readers_from_config(
    participant: Any,
    config: Dict[str, Any],
) -> Dict[str, TopicReader]:
    readers: Dict[str, TopicReader] = {}
    for item in config.get("topics", []) or []:
        topic = item.get("topic")
        type_path = item.get("type")
        if not topic or not type_path:
            LOG.warning("Skipping topic entry missing 'topic' or 'type': %s", item)
            continue
        try:
            last_exc: Optional[Exception] = None
            msg_type = None
            for candidate in _type_path_candidates(type_path):
                try:
                    msg_type = _resolve_type(candidate)
                    break
                except Exception as exc:
                    last_exc = exc
            if msg_type is None:
                raise last_exc or RuntimeError("No candidate types resolved")
            readers[topic] = TopicReader(participant, topic, msg_type, type_path)
            LOG.info("Subscribed (config): %s -> %s", topic, type_path)
        except Exception as exc:
            if isinstance(exc, ModuleNotFoundError):
                missing = getattr(exc, "name", None) or str(exc)
                LOG.warning(
                    "Skipping topic %s (%s). Missing module: %s. Update config or install the IDL module.",
                    topic,
                    type_path,
                    missing,
                )
            else:
                LOG.exception("Failed to subscribe (config) to %s with %s", topic, type_path)
    return readers


def _try_add_reader_for_discovered(
    participant: Any,
    readers: Dict[str, TopicReader],
    topic_name: str,
    type_name: str,
) -> None:
    if topic_name in readers:
        return
    mapped = _type_name_to_idl_module(type_name)
    if not mapped:
        LOG.info("Discovered %s (%s) but cannot map type", topic_name, type_name)
        return
    module_path, class_name = mapped
    try:
        module = importlib.import_module(module_path)
        msg_type = getattr(module, class_name)
    except Exception:
        LOG.info("Discovered %s (%s) but import failed: %s.%s", topic_name, type_name, module_path, class_name)
        return
    try:
        readers[topic_name] = TopicReader(participant, topic_name, msg_type, type_name)
        LOG.info("Subscribed (discovered): %s -> %s", topic_name, type_name)
    except Exception:
        LOG.exception("Failed to subscribe to discovered topic %s (%s)", topic_name, type_name)


def _scan_rpc_clients() -> None:
    sdk = _try_import("unitree_sdk2py")
    if not sdk:
        LOG.warning("unitree_sdk2py not installed; skipping RPC scan")
        return

    base_mod = _try_import("unitree_sdk2py.rpc.client_base")
    if not base_mod or not hasattr(base_mod, "ClientBase"):
        LOG.warning("unitree_sdk2py.rpc.client_base not available; skipping RPC scan")
        return

    ClientBase = getattr(base_mod, "ClientBase")

    LOG.info("Scanning SDK modules for RPC client classes...")
    for modinfo in pkgutil.walk_packages(sdk.__path__, sdk.__name__ + "."):
        try:
            mod = importlib.import_module(modinfo.name)
        except Exception:
            LOG.debug("Failed to import module %s", modinfo.name, exc_info=True)
            continue

        for name, obj in vars(mod).items():
            if not isinstance(obj, type):
                continue
            if obj is ClientBase:
                continue
            if issubclass(obj, ClientBase) or name.endswith("Client"):
                LOG.info("RPC client class: %s.%s", modinfo.name, name)
                try:
                    _ = obj()  # Best-effort; many clients require args
                    LOG.info("Instantiated: %s.%s", modinfo.name, name)
                except Exception:
                    LOG.info("Instantiation failed: %s.%s", modinfo.name, name, exc_info=True)


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def main() -> int:
    parser = argparse.ArgumentParser(description="Poll Unitree G1 DDS topics and SDK clients.")
    parser.add_argument("--domain", type=int, default=0, help="DDS domain id (default: 0)")
    parser.add_argument("--iface", type=str, default="", help="Network interface name (for unitree_sdk2py ChannelFactoryInitialize)")
    parser.add_argument("--config", type=str, default="", help="Path to JSON/YAML config with topics list")
    parser.add_argument(
        "--profile",
        type=str,
        default="",
        help="Built-in topic profile: g1_basic, g1_sport, g1_odom, g1_lidar (comma-separated)",
    )
    parser.add_argument("--poll", type=float, default=0.05, help="Polling interval seconds")
    parser.add_argument("--no-discover", action="store_true", help="Disable DDS discovery")
    parser.add_argument("--rpc-scan", action="store_true", help="Scan SDK modules for RPC client classes")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level (DEBUG, INFO, WARNING)")

    args = parser.parse_args()

    level = getattr(logging, args.log_level.upper(), logging.INFO)
    _configure_logging(level)

    # Best-effort initialize unitree SDK2 channel factory if available
    if args.iface:
        try:
            chan = _try_import("unitree_sdk2py.core.channel")
            if chan and hasattr(chan, "ChannelFactoryInitialize"):
                chan.ChannelFactoryInitialize(args.domain, args.iface)
                LOG.info("ChannelFactoryInitialize(domain=%s, iface=%s) OK", args.domain, args.iface)
        except Exception:
            LOG.exception("ChannelFactoryInitialize failed")

    if args.rpc_scan:
        _scan_rpc_clients()

    # DDS discovery + subscriptions
    discovery = None
    discover_enabled = not args.no_discover

    try:
        from cyclonedds.domain import DomainParticipant
    except Exception as exc:
        LOG.error("cyclonedds is required for DDS polling: %s", exc)
        return 2

    participant = DomainParticipant(args.domain)

    if discover_enabled:
        try:
            discovery = DdsDiscovery(args.domain)
        except Exception:
            LOG.exception("DDS discovery unavailable; continuing without discovery")
            discovery = None

    config = _load_config(args.config) if args.config else {}
    profiles = _build_profiles()
    if args.profile:
        combined: Dict[str, Any] = {"topics": []}
        for name in [p.strip() for p in args.profile.split(",") if p.strip()]:
            if name not in profiles:
                LOG.warning("Unknown profile %s (available: %s)", name, ", ".join(sorted(profiles.keys())))
                continue
            combined["topics"].extend(profiles[name].get("topics", []))
        if combined["topics"]:
            if config:
                config = {"topics": (config.get("topics", []) or []) + combined["topics"]}
            else:
                config = combined
    readers = _build_readers_from_config(participant, config)

    LOG.info("Polling started. Press Ctrl+C to stop.")
    try:
        stats: Dict[str, TopicStats] = {}
        while True:
            if discovery:
                for topic_name, type_name in discovery.poll():
                    LOG.info("Discovered topic: %s (%s)", topic_name, type_name)
                    if any(key in topic_name.lower() for key in RGBD_TOPIC_KEYWORDS):
                        LOG.info("Possible RGBD topic discovered: %s (%s)", topic_name, type_name)
                    _try_add_reader_for_discovered(participant, readers, topic_name, type_name)

            for topic_name, reader in list(readers.items()):
                stat = stats.setdefault(topic_name, TopicStats())
                try:
                    for sample in reader.read(8):
                        stat.update_sample(sample)
                        LOG.info("%s: %s", topic_name, stat.last_sample)
                except Exception as exc:
                    stat.update_error(exc)
                    LOG.exception("Read failed for topic %s", topic_name)
            time.sleep(args.poll)
    except KeyboardInterrupt:
        LOG.info("Stopping...")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
