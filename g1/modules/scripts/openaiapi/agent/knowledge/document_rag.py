"""Documentary RAG (spec section 20).

Thin wrapper around the existing ``ollama_ai/nav_bot.py:
KnowledgeRetriever`` -- reused, not replaced, per spec section 20's
"prefer the project's existing RAG implementation where possible".
``KnowledgeRetriever`` is a keyword/score-based retriever over local
JSON/text knowledge files; there is no vector-embedding index anywhere in
this repo or sandbox to swap in instead (a real one is a deferred TODO).

``nav_bot.py`` requires the Unitree SDK2 Python stack to import at all
(via ``sdk_slam``/``dds_env``), which is not installed in this dev
sandbox -- this module degrades to raising ``DocumentRAGUnavailable``
rather than crashing the whole planner input construction, so callers can
catch it and proceed with an empty ``documentary_rag`` list.
"""
from __future__ import annotations

import sys
from pathlib import Path

from ..models import KnowledgeRef


def _bootstrap_repo_paths() -> None:
    here = Path(__file__).resolve()
    scripts_dir = here.parents[2]  # g1/modules/scripts
    modules_dir = here.parents[3]  # g1/modules
    g1_dir = here.parents[4]  # g1
    ollama_ai_dir = scripts_dir / "ollama_ai"
    wbc_dir = g1_dir / "WBC"
    for path in (modules_dir, scripts_dir, ollama_ai_dir, wbc_dir):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


class DocumentRAGUnavailable(RuntimeError):
    pass


class DocumentRAG:
    def __init__(self, knowledge_files: list[str]) -> None:
        _bootstrap_repo_paths()
        try:
            from nav_bot import KnowledgeRetriever
        except Exception as exc:
            raise DocumentRAGUnavailable(f"nav_bot.KnowledgeRetriever unavailable: {exc}") from exc

        paths = [Path(f).expanduser() for f in knowledge_files]
        existing = [p for p in paths if p.exists()]
        if not existing:
            raise DocumentRAGUnavailable(f"none of the given knowledge files exist: {knowledge_files}")
        self._retriever = KnowledgeRetriever(existing)

    def search(self, query: str, *, top_k: int = 4, min_score: float = 0.06) -> list[KnowledgeRef]:
        results = self._retriever.search(query, top_k=top_k, min_score=min_score)
        return [
            KnowledgeRef(
                source_type="documentary",
                source=f"{entry.source} ({entry.title})",
                text=entry.text,
                trust="authoritative",
                note=f"score={score:.3f}",
            )
            for entry, score in results
        ]
