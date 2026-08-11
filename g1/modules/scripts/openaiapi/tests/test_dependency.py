from __future__ import annotations

import pytest


def test_agent_test_dependency_is_available() -> None:
    pydantic = pytest.importorskip("pydantic", reason="openaiapi agent tests require pydantic v2")
    if not pydantic.VERSION.startswith("2"):
        pytest.skip(f"openaiapi agent tests require pydantic v2 (found {pydantic.VERSION})")
