#!/usr/bin/env python3
"""
Best-effort polling and discovery for Unitree G1 DDS topics and SDK clients.

- Discovers DDS topics via Cyclone DDS built-in topics.
- Tries to map discovered DDS type names to unitree_sdk2py IDL classes.
- Subscribes to configured topics (from JSON/YAML) and logs received samples.
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

try:
    from rich.logging import RichHandler  # type: ignore

    def _configure_logging(level: int) -> None:
        logging.basicConfig(
            level=level,
            format="%(message)s",
            datefmt="[%H:%M:%S]",
            handlers=[RichHandler(rich_tracebacks=True)],
        )

except Exception:

    def _configure_logging(level: int) -> None:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )


LOG = logging.getLogger("g1_poll_all")


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
    def __init__(self, participant: Any, topic_name: str, msg_type: Any) -> None:
        from cyclonedds.topic import Topic
        from cyclonedds.sub import DataReader

        self.topic_name = topic_name
        self.msg_type = msg_type
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
            msg_type = _resolve_type(type_path)
            readers[topic] = TopicReader(participant, topic, msg_type)
            LOG.info("Subscribed (config): %s -> %s", topic, type_path)
        except Exception:
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
        readers[topic_name] = TopicReader(participant, topic_name, msg_type)
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Poll Unitree G1 DDS topics and SDK clients.")
    parser.add_argument("--domain", type=int, default=0, help="DDS domain id (default: 0)")
    parser.add_argument("--iface", type=str, default="", help="Network interface name (for unitree_sdk2py ChannelFactoryInitialize)")
    parser.add_argument("--config", type=str, default="", help="Path to JSON/YAML config with topics list")
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
    readers = _build_readers_from_config(participant, config)

    LOG.info("Polling started. Press Ctrl+C to stop.")
    try:
        while True:
            if discovery:
                for topic_name, type_name in discovery.poll():
                    LOG.info("Discovered topic: %s (%s)", topic_name, type_name)
                    _try_add_reader_for_discovered(participant, readers, topic_name, type_name)

            for topic_name, reader in list(readers.items()):
                try:
                    for sample in reader.read(8):
                        LOG.info("%s: %s", topic_name, _format_sample(sample))
                except Exception:
                    LOG.exception("Read failed for topic %s", topic_name)
            time.sleep(args.poll)
    except KeyboardInterrupt:
        LOG.info("Stopping...")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
