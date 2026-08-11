"""AST/symbol index over sdk_wrapper_v3.py (spec section 19).

Classified explicitly, on every retrieved chunk, as:

    IMPLEMENTATION KNOWLEDGE -- IMPORTANT -- POTENTIALLY FALLIBLE

Indexes class/function/method definitions with their line ranges and a
content hash for provenance, so the planner can retrieve small, targeted
source slices instead of the whole ~58KB file. This module is a knowledge
source only -- nothing in ``agent/skills.py`` or ``agent/capabilities.py``
calls through it.

Documented cross-reference gap (see the Phase 1 plan): every live script
in this repo that actually drives the robot at runtime (``llm_client``,
``scene_executor``, ``nav_bot``, ``chatbot_with_tactile_dex3``,
``grasp_pipeline``) calls through ``g1/modules/sdk_client.py``'s ``Robot``
class, not the ``G1`` class this file indexes. ``sdk_wrapper_v3.py``
defines an independent implementation with different method names for the
same hardware. This module indexes it because that is what was designated
as boot-time knowledge (spec section 19) -- indexing it is not a claim
that it is what actually executes commands.
"""
from __future__ import annotations

import ast
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..models import KnowledgeRef

DEFAULT_SDK_WRAPPER_PATH = Path(
    os.environ.get(
        "G1_AGENT_SDK_WRAPPER",
        str(Path(__file__).resolve().parents[4] / "dev" / "sdk_wrapper_v3.py"),
    )
).expanduser()


@dataclass
class SymbolEntry:
    kind: str  # "class" | "function" | "method"
    qualname: str
    line_start: int
    line_end: int
    docstring: Optional[str]


class SdkWrapperKnowledge:
    """Indexes one sdk_wrapper.py-shaped file into retrievable symbol chunks."""

    def __init__(self, path: Path | str = DEFAULT_SDK_WRAPPER_PATH) -> None:
        self.path = Path(path).expanduser()
        self._symbols: list[SymbolEntry] = []
        self._source_lines: list[str] = []
        self._file_hash: Optional[str] = None
        self._indexed = False

    def index(self) -> None:
        if not self.path.exists():
            self._indexed = True
            return
        source = self.path.read_text(encoding="utf-8")
        self._source_lines = source.splitlines()
        self._file_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
        tree = ast.parse(source, filename=str(self.path))
        self._symbols = list(self._walk(tree))
        self._indexed = True

    @staticmethod
    def _walk(tree: ast.Module):
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                yield SymbolEntry(
                    kind="class",
                    qualname=node.name,
                    line_start=node.lineno,
                    line_end=getattr(node, "end_lineno", node.lineno),
                    docstring=ast.get_docstring(node),
                )
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        yield SymbolEntry(
                            kind="method",
                            qualname=f"{node.name}.{child.name}",
                            line_start=child.lineno,
                            line_end=getattr(child, "end_lineno", child.lineno),
                            docstring=ast.get_docstring(child),
                        )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield SymbolEntry(
                    kind="function",
                    qualname=node.name,
                    line_start=node.lineno,
                    line_end=getattr(node, "end_lineno", node.lineno),
                    docstring=ast.get_docstring(node),
                )

    def available(self) -> bool:
        if not self._indexed:
            self.index()
        return bool(self._symbols)

    def symbol_names(self) -> list[str]:
        if not self._indexed:
            self.index()
        return [entry.qualname for entry in self._symbols]

    def search(self, query: str, *, top_k: int = 5) -> list[KnowledgeRef]:
        if not self._indexed:
            self.index()
        if not self._symbols:
            return []

        terms = {term.lower() for term in query.split() if term}
        scored: list[tuple[int, SymbolEntry]] = []
        for entry in self._symbols:
            haystack = f"{entry.qualname} {entry.docstring or ''}".lower()
            score = sum(1 for term in terms if term in haystack) if terms else 0
            scored.append((score, entry))
        scored.sort(key=lambda item: item[0], reverse=True)

        chosen = [entry for score, entry in scored if score > 0][:top_k]
        if not chosen:  # no keyword hit (or an empty query) -- fall back to first N symbols
            chosen = [entry for _, entry in scored[:top_k]]

        return [
            KnowledgeRef(
                source_type="implementation",
                source=f"sdk_wrapper_v3.py#{self._file_hash or 'unknown'}",
                text=entry.docstring or f"{entry.kind} {entry.qualname}",
                line_range=f"{entry.line_start}-{entry.line_end}",
                trust="low",
                note=(
                    "IMPLEMENTATION KNOWLEDGE -- IMPORTANT -- POTENTIALLY FALLIBLE. "
                    "Not the runtime binding: live scripts call sdk_client.Robot, not this class."
                ),
            )
            for entry in chosen
        ]
