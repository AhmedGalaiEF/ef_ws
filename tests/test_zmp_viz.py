from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "g1" / "academy" / "visualizations" / "zmp_viz.py"


def component(*_args: object, **_kwargs: object) -> dict:
    return {}


def load_zmp_viz(monkeypatch: pytest.MonkeyPatch):
    dash = ModuleType("dash")

    class Dash:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.layout = None
            self.title = ""

        def callback(self, *_args: object, **_kwargs: object):
            return lambda function: function

    # Provide the concrete components constructed while the module imports.
    dcc = SimpleNamespace(Input=component, Graph=component, Interval=component)
    html = SimpleNamespace(H3=component, Label=component, Div=component, Br=component, Span=component)
    dash.Dash = Dash
    dash.Input = component
    dash.Output = component
    dash.State = component
    dash.dcc = dcc
    dash.html = html
    dash.ctx = SimpleNamespace(triggered_id=None)

    dbc = ModuleType("dash_bootstrap_components")
    dbc.themes = SimpleNamespace(DARKLY="darkly")
    for name in ("Container", "Row", "Col", "Input", "Button", "Badge"):
        setattr(dbc, name, component)

    graph_objects = ModuleType("plotly.graph_objects")
    graph_objects.Figure = object
    graph_objects.Scatter = object
    plotly = ModuleType("plotly")
    plotly.graph_objects = graph_objects

    monkeypatch.setitem(sys.modules, "dash", dash)
    monkeypatch.setitem(sys.modules, "dash_bootstrap_components", dbc)
    monkeypatch.setitem(sys.modules, "plotly", plotly)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", graph_objects)

    module_name = "_test_zmp_viz"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_cli_rejects_invalid_domain_and_port(monkeypatch: pytest.MonkeyPatch) -> None:
    zmp = load_zmp_viz(monkeypatch)

    assert zmp._parse_args(["--domain-id", "232", "--port", "65535"]).port == 65535
    with pytest.raises(SystemExit):
        zmp._parse_args(["--domain-id", "233"])
    with pytest.raises(SystemExit):
        zmp._parse_args(["--port", "0"])


def test_kinematics_rejects_non_finite_joint_positions(monkeypatch: pytest.MonkeyPatch) -> None:
    zmp = load_zmp_viz(monkeypatch)

    with pytest.raises(ValueError, match="non-finite"):
        zmp.body_frames([0.0] * 11 + [float("nan")])


def test_invalid_imu_falls_back_to_odometry_yaw(monkeypatch: pytest.MonkeyPatch) -> None:
    zmp = load_zmp_viz(monkeypatch)

    yaw = zmp._extract_yaw(
        {"pose": [1.0, 2.0, 0.4]},
        {"rpy": [0.0, 0.0, float("nan")]},
    )

    assert yaw == pytest.approx(0.4)
    assert zmp._extract_yaw({"pose": [1.0, 2.0]}, None) is None
    assert zmp._imu_acceleration_world({"rpy": [0, 0, 0], "acc": [np.nan, 0, 0]}) is None


def test_disconnect_stops_poll_thread_before_reconnect(monkeypatch: pytest.MonkeyPatch) -> None:
    zmp = load_zmp_viz(monkeypatch)
    sdk_wrapper = ModuleType("sdk_wrapper")
    sdk_wrapper.G1 = lambda *_args, **_kwargs: object()
    monkeypatch.setitem(sys.modules, "sdk_wrapper", sdk_wrapper)
    monkeypatch.setattr(zmp.ZmpLink, "_poll_loop", lambda _self, stop: stop.wait())

    link = zmp.ZmpLink("eth0", 0)
    link.connect()
    first_thread = link._poll_thread
    assert first_thread is not None and first_thread.is_alive()

    link.disconnect()
    assert not first_thread.is_alive()

    link.connect()
    try:
        assert link._poll_thread is not first_thread
    finally:
        link.disconnect()
