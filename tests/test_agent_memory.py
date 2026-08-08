from __future__ import annotations

import sys
from pathlib import Path

import pytest

pydantic = pytest.importorskip("pydantic")
if not pydantic.VERSION.startswith("2"):
    pytest.skip(f"agent package requires pydantic>=2 (found {pydantic.VERSION})", allow_module_level=True)

AGENT_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "g1" / "modules" / "scripts"
if str(AGENT_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_SCRIPTS_DIR))

from agent.memory.autobiography import AutobiographyStore  # noqa: E402
from agent.memory.episodic import EpisodicStore, new_episode  # noqa: E402
from agent.memory.manager import MemoryManager, MemoryProposalError  # noqa: E402
from agent.memory.procedural import ProceduralAdaptation, ProceduralStore  # noqa: E402
from agent.memory.semantic import SemanticClaim, SemanticStore  # noqa: E402
from agent.models import MemoryProposal  # noqa: E402


def test_episodic_append_and_keyword_search(tmp_path: Path) -> None:
    store = EpisodicStore(tmp_path / "episodic.jsonl")
    store.append(new_episode(goal="wave at the visitor", outcome="succeeded"))
    store.append(new_episode(goal="pick up the bottle", outcome="failed: grip slipped"))

    hits = store.search("bottle")
    assert len(hits) == 1
    assert "bottle" in hits[0].goal


def test_semantic_upsert_dedups_and_keeps_higher_confidence(tmp_path: Path) -> None:
    store = SemanticStore(tmp_path / "semantic.json")
    store.upsert(SemanticClaim(claim="right knee runs hot after standing", confidence=0.4, supporting_episodes=["e1"]))
    store.upsert(SemanticClaim(claim="right knee runs hot after standing", confidence=0.9, supporting_episodes=["e2"]))

    claims = store.all()
    assert len(claims) == 1
    assert claims[0].confidence == pytest.approx(0.9)
    assert set(claims[0].supporting_episodes) == {"e1", "e2"}


def test_procedural_store_filters_by_skill(tmp_path: Path) -> None:
    store = ProceduralStore(tmp_path / "procedural.json")
    store.add(ProceduralAdaptation(skill="stand_from_crouch", condition={"knee_temp_c": ">60"}, recommended_parameters={"rise_speed_scale": 0.7}))
    store.add(ProceduralAdaptation(skill="wave", condition={}, recommended_parameters={}))

    assert len(store.for_skill("stand_from_crouch")) == 1
    assert len(store.for_skill("wave")) == 1
    assert len(store.for_skill("unknown")) == 0


def test_autobiography_summary_is_oldest_first_and_bounded(tmp_path: Path) -> None:
    store = AutobiographyStore(tmp_path / "autobiography.jsonl")
    for i in range(3):
        store.append(f"event {i}", timestamp=1000.0 + i)

    summary = store.summary(max_entries=2)
    assert summary is not None
    lines = summary.splitlines()
    assert len(lines) == 2
    assert "event 1" in lines[0]
    assert "event 2" in lines[1]


def test_autobiography_summary_none_when_empty(tmp_path: Path) -> None:
    store = AutobiographyStore(tmp_path / "autobiography.jsonl")
    assert store.summary() is None


def test_memory_manager_apply_proposal_writes_to_correct_store(tmp_path: Path) -> None:
    manager = MemoryManager(base_dir=tmp_path)
    manager.apply_proposal(
        MemoryProposal(kind="semantic", content={"claim": "battery drains faster in cold weather"}, confidence=0.7)
    )
    claims = manager.semantic.all()
    assert len(claims) == 1
    assert claims[0].claim == "battery drains faster in cold weather"


def test_memory_manager_rejects_unknown_kind(tmp_path: Path) -> None:
    manager = MemoryManager(base_dir=tmp_path)
    with pytest.raises(MemoryProposalError):
        manager.apply_proposal(MemoryProposal(kind="not_a_real_kind", content={}))


def test_memory_manager_rejects_missing_required_field(tmp_path: Path) -> None:
    manager = MemoryManager(base_dir=tmp_path)
    with pytest.raises(MemoryProposalError):
        manager.apply_proposal(MemoryProposal(kind="episodic", content={"no_goal_field": True}))


def test_memory_manager_retrieve_returns_all_three_kinds(tmp_path: Path) -> None:
    manager = MemoryManager(base_dir=tmp_path)
    manager.episodic.append(new_episode(goal="find the charging dock", outcome="succeeded"))
    manager.semantic.upsert(SemanticClaim(claim="the charging dock is near the kitchen", confidence=0.8))

    refs = manager.retrieve("charging dock")
    assert refs["episodic"] and refs["episodic"][0].source_type == "episodic"
    assert refs["semantic"] and refs["semantic"][0].source_type == "semantic"
