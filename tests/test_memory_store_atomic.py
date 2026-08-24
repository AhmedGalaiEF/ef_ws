from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


AGENT_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "g1" / "modules" / "scripts" / "openaiapi"
if str(AGENT_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_SCRIPTS_DIR))

import agent.memory as memory_package  # noqa: E402
from agent.memory.procedural import ProceduralAdaptation, ProceduralStore  # noqa: E402
from agent.memory.semantic import SemanticClaim, SemanticStore  # noqa: E402


@pytest.mark.parametrize("store_kind", ["semantic", "procedural"])
def test_memory_store_preserves_existing_file_when_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    store_kind: str,
) -> None:
    path = tmp_path / f"{store_kind}.json"
    if store_kind == "semantic":
        store = SemanticStore(path)
        store.upsert(SemanticClaim(claim="existing"))
        write = lambda: store.upsert(SemanticClaim(claim="new"))
    else:
        store = ProceduralStore(path)
        store.add(ProceduralAdaptation(skill="existing"))
        write = lambda: store.add(ProceduralAdaptation(skill="new"))

    original = json.loads(path.read_text(encoding="utf-8"))

    def fail_replace(*_args: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(memory_package.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        write()

    assert json.loads(path.read_text(encoding="utf-8")) == original
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))
