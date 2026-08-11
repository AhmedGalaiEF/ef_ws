"""Memory retrieval + proposal-validation facade (spec sections 9, 23).

The planner never writes memory directly -- it returns a
``MemoryProposal`` (see ``agent/models.py``) that this manager validates
before applying, matching spec section 9's "a deterministic memory/
learning manager validates, deduplicates, and versions canonical
changes." Validation here is intentionally simple (kind must be a known
store, content must be well-formed for that store); real deduplication
against near-duplicate *episodes* and clustering/consolidation are
deferred (see the plan's TODO list) -- ``SemanticStore.upsert`` already
dedups on exact claim text, which is as far as this phase goes.
"""
from __future__ import annotations

from pathlib import Path
import os
from typing import Optional

from ..models import KnowledgeRef, MemoryProposal
from .autobiography import AutobiographyStore
from .episodic import EpisodicStore, new_episode
from .procedural import ProceduralAdaptation, ProceduralStore
from .semantic import SemanticClaim, SemanticStore


DEFAULT_MEMORY_DIR = Path(os.environ.get("G1_AGENT_MEMORY_DIR", "~/.g1_agent/memory")).expanduser()


class MemoryProposalError(ValueError):
    pass


class MemoryManager:
    def __init__(self, *, base_dir: Optional[Path | str] = None) -> None:
        base = Path(base_dir).expanduser() if base_dir is not None else DEFAULT_MEMORY_DIR
        base.mkdir(parents=True, exist_ok=True)
        self.episodic = EpisodicStore(base / "episodic.jsonl")
        self.semantic = SemanticStore(base / "semantic.json")
        self.procedural = ProceduralStore(base / "procedural.json")
        self.autobiography = AutobiographyStore(base / "autobiography.jsonl")

    # -- retrieval, for PlannerInput construction -----------------------

    def retrieve(self, query: str, *, top_k: int = 5) -> dict[str, list[KnowledgeRef]]:
        episodic_refs = [
            KnowledgeRef(
                source_type="episodic",
                source=ep.id,
                text=f"goal={ep.goal!r} outcome={ep.outcome!r}",
                trust="medium",
                note="historical experience",
            )
            for ep in self.episodic.search(query, top_k=top_k)
        ]
        semantic_refs = [
            KnowledgeRef(
                source_type="semantic",
                source="semantic_memory",
                text=claim.claim,
                trust="medium" if claim.confidence < 0.85 else "authoritative",
                note=f"confidence={claim.confidence:.2f}",
            )
            for claim in self.semantic.search(query, top_k=top_k)
        ]
        procedural_refs = [
            KnowledgeRef(
                source_type="procedural",
                source=adaptation.skill,
                text=f"condition={adaptation.condition} -> {adaptation.recommended_parameters}",
                trust="medium" if adaptation.confidence < 0.85 else "authoritative",
                note=f"confidence={adaptation.confidence:.2f}",
            )
            for adaptation in self.procedural.all()
        ][:top_k]
        return {"episodic": episodic_refs, "semantic": semantic_refs, "procedural": procedural_refs}

    def autobiography_summary(self) -> Optional[str]:
        return self.autobiography.summary()

    # -- validated write path --------------------------------------------

    def apply_proposal(self, proposal: MemoryProposal) -> None:
        kind = proposal.kind.strip().lower()
        content = proposal.content

        if kind == "episodic":
            if "goal" not in content:
                raise MemoryProposalError("episodic memory_proposal.content requires a 'goal' field")
            self.episodic.append(
                new_episode(
                    goal=str(content["goal"]),
                    initial_state=content.get("initial_state") or {},
                    actions=content.get("actions") or [],
                    observations=content.get("observations") or [],
                    outcome=str(content.get("outcome", "")),
                    anomalies=content.get("anomalies") or [],
                )
            )
        elif kind == "semantic":
            if "claim" not in content:
                raise MemoryProposalError("semantic memory_proposal.content requires a 'claim' field")
            self.semantic.upsert(
                SemanticClaim(
                    claim=str(content["claim"]),
                    confidence=float(proposal.confidence),
                    supporting_episodes=list(proposal.derived_from),
                )
            )
        elif kind == "procedural":
            if "skill" not in content:
                raise MemoryProposalError("procedural memory_proposal.content requires a 'skill' field")
            self.procedural.add(
                ProceduralAdaptation(
                    skill=str(content["skill"]),
                    condition=content.get("condition") or {},
                    recommended_parameters=content.get("recommended_parameters") or {},
                    confidence=float(proposal.confidence),
                    derived_from=list(proposal.derived_from),
                )
            )
        elif kind == "autobiographical":
            if "summary" not in content:
                raise MemoryProposalError("autobiographical memory_proposal.content requires a 'summary' field")
            self.autobiography.append(str(content["summary"]))
        else:
            raise MemoryProposalError(f"unknown memory_proposal.kind {proposal.kind!r}")
