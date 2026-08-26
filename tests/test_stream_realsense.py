from __future__ import annotations

from types import SimpleNamespace

import pytest

import dev.stream_realsense as stream_realsense


def test_streamer_reports_optional_dependency_at_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stream_realsense, "rs", None)

    with pytest.raises(RuntimeError, match="pyrealsense2 is not installed"):
        stream_realsense.run()


@pytest.mark.parametrize("option", ["--width", "--height", "--fps", "--timeout-ms"])
def test_streamer_cli_rejects_non_positive_dimensions_and_timing(option: str) -> None:
    parser = stream_realsense.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([option, "0"])


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"rgb_width": 0}, "rgb_width"),
        ({"rgb_height": -1}, "rgb_height"),
        ({"fps": True}, "fps"),
        ({"timeout_ms": 0}, "timeout_ms"),
        ({"display": "sometimes"}, "display"),
        ({"serial": "  "}, "serial"),
    ],
)
def test_streamer_validates_programmatic_configuration(
    overrides: dict[str, object], message: str
) -> None:
    defaults: dict[str, object] = {
        "rgb_width": 640,
        "rgb_height": 480,
        "fps": 30,
        "timeout_ms": 15000,
        "display": "auto",
        "serial": None,
    }
    defaults.update(overrides)

    with pytest.raises(ValueError, match=message):
        stream_realsense.validate_run_config(**defaults)


def test_select_device_honours_requested_serial(monkeypatch: pytest.MonkeyPatch) -> None:
    serial_info = object()
    monkeypatch.setattr(
        stream_realsense,
        "rs",
        SimpleNamespace(camera_info=SimpleNamespace(serial_number=serial_info)),
    )

    class Device:
        def __init__(self, serial: str) -> None:
            self.serial = serial

        def get_info(self, info: object) -> str:
            assert info is serial_info
            return self.serial

    first = Device("first")
    requested = Device("requested")
    context = SimpleNamespace(query_devices=lambda: [first, requested])

    assert stream_realsense.select_device(context, "requested") is requested
    assert stream_realsense.select_device(context, "missing") is None


def test_wait_for_device_retries_during_usb_reconnect(monkeypatch: pytest.MonkeyPatch) -> None:
    serial_info = object()

    class Device:
        def get_info(self, info: object) -> str:
            assert info is serial_info
            return "camera-1"

    contexts = iter(
        [
            SimpleNamespace(query_devices=lambda: []),
            SimpleNamespace(query_devices=lambda: [Device()]),
        ]
    )
    fake_rs = SimpleNamespace(
        camera_info=SimpleNamespace(serial_number=serial_info),
        context=lambda: next(contexts),
    )
    monkeypatch.setattr(stream_realsense, "rs", fake_rs)
    monkeypatch.setattr(stream_realsense.time, "sleep", lambda _delay: None)

    context, device = stream_realsense.wait_for_device("camera-1", timeout_s=1.0)

    assert context is not None
    assert device is not None
