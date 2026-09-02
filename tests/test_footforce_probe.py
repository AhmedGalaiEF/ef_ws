from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "g1" / "modules" / "scripts" / "footforce_test.py"


def load_footforce_probe(monkeypatch: pytest.MonkeyPatch):
    dds_env = ModuleType("dds_env")
    dds_env.default_dds_iface = lambda _preferred: "eth0"
    dds_env.ensure_channel_factory_initialized = lambda *_args, **_kwargs: None
    sdk_sensors = ModuleType("sdk_sensors")
    sdk_sensors.resolve_lowstate_type = lambda: None
    monkeypatch.setitem(sys.modules, "dds_env", dds_env)
    monkeypatch.setitem(sys.modules, "sdk_sensors", sdk_sensors)

    module_name = "_test_footforce_probe"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "301"])
def test_probe_rejects_invalid_collection_duration(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    probe = load_footforce_probe(monkeypatch)

    with pytest.raises(SystemExit):
        probe.parse_args(["--seconds", value])


def test_probe_cli_validates_domain_and_supports_explicit_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = load_footforce_probe(monkeypatch)

    args = probe.parse_args(["--domain-id", "232", "--seconds", "2.5", "--no-ros"])
    assert args.domain_id == 232
    assert args.seconds == pytest.approx(2.5)
    with pytest.raises(SystemExit):
        probe.parse_args(["--domain-id", "233"])


def test_numeric_force_fields_reject_non_finite_values(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = load_footforce_probe(monkeypatch)
    message = SimpleNamespace(foot_force=[1.0, float("nan")], contact_load=[2.0, 3.0])

    assert probe.numeric_force_fields(message) == {"contact_load": [2, 3]}


def test_probe_bounds_history_but_preserves_total_and_nonzero_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe_module = load_footforce_probe(monkeypatch)
    probe = probe_module.LowStateProbe()
    probe.callback(SimpleNamespace(foot_force=[5.0]))
    for _ in range(1100):
        probe.callback(SimpleNamespace(foot_force=[0.0]))

    result = probe.result()
    assert result["samples_received"] == 1101
    assert len(probe.samples) == 1000
    assert result["any_nonzero"]["foot_force"] is True


def test_ros_topic_filter_uses_type_and_force_names(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = load_footforce_probe(monkeypatch)
    listing = "\n".join(
        [
            "/joint_states [sensor_msgs/msg/JointState]",
            "/left_foot_wrench [geometry_msgs/msg/WrenchStamped]",
            "/state [unitree_hg/msg/LowState]",
        ]
    )

    assert probe.relevant_ros_topics(listing) == [
        ("/left_foot_wrench", "geometry_msgs/msg/WrenchStamped"),
        ("/state", "unitree_hg/msg/LowState"),
    ]
