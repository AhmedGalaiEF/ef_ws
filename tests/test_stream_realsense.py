from __future__ import annotations

import pytest

import dev.stream_realsense as stream_realsense


def test_streamer_reports_optional_dependency_at_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stream_realsense, "rs", None)

    with pytest.raises(RuntimeError, match="pyrealsense2 is not installed"):
        stream_realsense.run()
