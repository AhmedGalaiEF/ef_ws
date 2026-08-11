"""Read-only views over acquired empirical and procedural knowledge."""
from __future__ import annotations

import time
from typing import Any, Optional


class TacitKnowledgeService:
    def __init__(self, *, agent: Any) -> None:
        self.agent = agent

    def snapshot(self, *, panel: str = "recent", item_id: Optional[str] = None) -> dict[str, Any]:
        panel = (panel or "recent").strip().lower()
        learned = self.agent.learning.learned.all()
        semantic = self.agent.memory.semantic.all()
        procedural = self.agent.memory.procedural.all()
        empirical_items = [self._learned_item(claim, idx) for idx, claim in enumerate(learned, start=1)]
        empirical_items.extend(self._semantic_item(claim, idx) for idx, claim in enumerate(semantic, start=1))
        procedural_items = [self._procedural_item(item, idx) for idx, item in enumerate(procedural, start=1)]
        candidates = [item for item in empirical_items if item.get("status") == "candidate"]
        contested = [item for item in empirical_items if item.get("status") == "contested"]
        deprecated = [item for item in empirical_items if item.get("status") == "deprecated"]
        history = self._history()
        all_items = empirical_items + procedural_items
        selected = self._find_item(all_items, item_id) if item_id else None
        return {
            "panel": panel,
            "summary": {
                "learning_enabled": bool(self.agent.settings.effective().learning.enabled),
                "active_empirical_claims": sum(1 for item in empirical_items if item.get("status") == "active"),
                "procedural_rules": len(procedural_items),
                "candidate_adaptations": len(candidates),
                "contested_claims": len(contested),
                "deprecated_claims": len(deprecated),
                "last_consolidation": self._last_event_age("maintenance", "maintenance_completed"),
            },
            "empirical": empirical_items,
            "procedural": procedural_items,
            "candidates": candidates,
            "contested": contested,
            "deprecated": deprecated,
            "recent": (procedural_items + empirical_items)[-10:],
            "history": history,
            "forgotten": [
                event for event in history
                if any(token in event.get("event", "") for token in ("archived", "deprecated", "merged", "consolidated"))
            ],
            "selected": selected,
        }

    def render_lines(self, *, panel: str = "recent", item_id: Optional[str] = None, max_items: int = 10) -> list[str]:
        snap = self.snapshot(panel=panel, item_id=item_id)
        summary = snap["summary"]
        if panel == "show" and snap.get("selected") is not None:
            return self._render_item(snap["selected"])
        if panel == "evidence" and snap.get("selected") is not None:
            return self._render_evidence(snap["selected"])
        if panel == "stats":
            return [
                "TACIT / LEARNED KNOWLEDGE STATS",
                f"Learning enabled          {self._yes(summary['learning_enabled'])}",
                f"Active empirical claims   {summary['active_empirical_claims']}",
                f"Procedural rules          {summary['procedural_rules']}",
                f"Candidate adaptations     {summary['candidate_adaptations']}",
                f"Contested claims          {summary['contested_claims']}",
                f"Deprecated claims         {summary['deprecated_claims']}",
                f"Last consolidation        {summary['last_consolidation']}",
            ]
        if panel in {"empirical", "procedural", "candidates", "contested", "deprecated", "recent"}:
            items = snap.get(panel) or []
            return self._render_collection(panel.upper(), items[:max_items])
        if panel in {"forgotten", "history"}:
            rows = snap.get("forgotten" if panel == "forgotten" else "history") or []
            lines = [panel.upper()]
            for event in rows[-max_items:]:
                lines.append(f"{event.get('time')}  {event.get('event')}  {event.get('summary')}")
            return lines if len(lines) > 1 else [panel.upper(), "(no learning lifecycle events)"]
        lines = [
            "TACIT / LEARNED KNOWLEDGE",
            f"Learning enabled          {self._yes(summary['learning_enabled'])}",
            f"Active empirical claims   {summary['active_empirical_claims']}",
            f"Procedural rules          {summary['procedural_rules']}",
            f"Candidate adaptations     {summary['candidate_adaptations']}",
            f"Contested claims          {summary['contested_claims']}",
            f"Deprecated claims         {summary['deprecated_claims']}",
            "",
        ]
        lines.extend(self._render_collection("RECENT LEARNED KNOWLEDGE", snap["recent"][:max_items]))
        return lines

    def _learned_item(self, claim: Any, idx: int) -> dict[str, Any]:
        tool_evidence = []
        try:
            tool_evidence = list((claim.applicable_context or {}).get("tool_evidence") or [])
        except Exception:
            tool_evidence = []
        return {
            "id": claim.id or f"E-{idx:03d}",
            "kind": "empirical",
            "claim": claim.claim,
            "status": claim.status,
            "confidence": float(claim.confidence),
            "support": len(claim.supporting_episodes),
            "contradictions": len(claim.contradicting_episodes),
            "created": claim.created_at,
            "last_updated": claim.last_validation_time,
            "applicable_context": claim.applicable_context,
            "provenance": "derived consolidation / robot experience",
            "supporting_episodes": list(claim.supporting_episodes),
            "contradicting_episodes": list(claim.contradicting_episodes),
            "tool_evidence": tool_evidence,
            "behavioral_use": "ADVISORY ONLY",
            "version": claim.version,
        }

    def _semantic_item(self, claim: Any, idx: int) -> dict[str, Any]:
        return {
            "id": f"S-{idx:03d}",
            "kind": "semantic",
            "claim": claim.claim,
            "status": "active",
            "confidence": float(claim.confidence),
            "support": len(claim.supporting_episodes),
            "contradictions": 0,
            "created": None,
            "last_updated": None,
            "applicable_context": {},
            "provenance": "semantic memory / validated proposal",
            "supporting_episodes": list(claim.supporting_episodes),
            "contradicting_episodes": [],
            "behavioral_use": "PLANNER RETRIEVAL",
            "version": 1,
        }

    def _procedural_item(self, adaptation: Any, idx: int) -> dict[str, Any]:
        active = self._procedural_active(adaptation)
        self_effect = ""
        try:
            record = self.agent.self_model.model.skills.records.get(adaptation.skill)
            if record and record.success_rate is not None:
                self_effect = f"{adaptation.skill} reliability={record.success_rate:.2f} confidence={record.confidence:.2f}"
        except Exception:
            self_effect = ""
        tool_evidence = []
        try:
            tool_evidence = list((adaptation.condition or {}).get("tool_evidence") or [])
        except Exception:
            tool_evidence = []
        return {
            "id": f"P-{idx:03d}",
            "kind": "procedural",
            "skill": adaptation.skill,
            "condition": adaptation.condition,
            "adaptation": adaptation.recommended_parameters,
            "status": "validated" if active else "candidate",
            "validation_level": 3 if active else 2,
            "confidence": float(adaptation.confidence),
            "baseline_success": None,
            "adapted_success": None,
            "trials": len(adaptation.derived_from),
            "active": active,
            "behavioral_use": "ACTIVE" if active else "ADVISORY ONLY",
            "self_model_effect": self_effect,
            "version": 1,
            "provenance": "procedural/tacit memory",
            "supporting_episodes": list(adaptation.derived_from),
            "contradicting_episodes": [],
            "tool_evidence": tool_evidence,
        }

    def _procedural_active(self, adaptation: Any) -> bool:
        settings = self.agent.settings.effective()
        if int(getattr(settings.learning, "automatic_level_max", 1)) < 3:
            return False
        return bool(adaptation.recommended_parameters.get("pre_pose"))

    def _history(self) -> list[dict[str, Any]]:
        events = []
        for event in self.agent.monitor_bus.recent(200):
            if event.category not in {"learning", "memory", "maintenance", "reset"}:
                continue
            events.append(
                {
                    "timestamp": event.timestamp,
                    "time": time.strftime("%H:%M", time.localtime(event.timestamp)),
                    "category": event.category,
                    "event": event.event,
                    "summary": event.summary,
                    "references": list(event.references),
                }
            )
        return events

    def _last_event_age(self, category: str, event_name: str) -> str:
        now = time.time()
        for event in reversed(self.agent.monitor_bus.recent(200)):
            if event.category == category and event.event == event_name:
                return f"{(now - event.timestamp) / 60.0:.0f} min ago"
        return "never"

    @staticmethod
    def _find_item(items: list[dict[str, Any]], item_id: Optional[str]) -> Optional[dict[str, Any]]:
        if item_id is None:
            return None
        wanted = item_id.strip().lower()
        for item in items:
            if str(item.get("id", "")).lower() == wanted:
                return item
        return None

    def _render_collection(self, title: str, items: list[dict[str, Any]]) -> list[str]:
        lines = [title]
        if not items:
            return lines + ["(empty)"]
        settings = self.agent.settings.effective().tacit
        for item in items:
            lines.append("")
            lines.append(f"[{item.get('id')}] {item.get('skill') or item.get('claim', '')[:70]}")
            lines.append(f"Status          {item.get('status')}")
            if settings.show_confidence:
                lines.append(f"Confidence      {float(item.get('confidence') or 0):.2f}")
            if settings.show_evidence_counts:
                lines.append(f"Support         {item.get('support', item.get('trials', 0))}")
                lines.append(f"Contradictions  {item.get('contradictions', 0)}")
            if item.get("kind") == "procedural":
                lines.append(f"Behavioral use  {item.get('behavioral_use')}")
                if item.get("self_model_effect"):
                    lines.append(f"Self-model      {item.get('self_model_effect')}")
                lines.append(f"Condition       {item.get('condition')}")
                lines.append(f"Adaptation      {item.get('adaptation')}")
            else:
                lines.append(f"Behavioral use  {item.get('behavioral_use')}")
                lines.append(f"Claim           {item.get('claim')}")
            lines.append(f"Provenance      {item.get('provenance')}")
            tools = self._tool_evidence_names(item)
            if tools:
                lines.append(f"Investigated    {', '.join(tools)}")
        return lines

    def _render_item(self, item: dict[str, Any]) -> list[str]:
        lines = [f"{item.get('id')}"]
        for key in (
            "kind",
            "skill",
            "claim",
            "condition",
            "adaptation",
            "status",
            "validation_level",
            "confidence",
            "support",
            "contradictions",
            "behavioral_use",
            "provenance",
            "version",
        ):
            if key in item and item.get(key) is not None:
                lines.append(f"{key:<18} {item.get(key)}")
        tools = self._tool_evidence_names(item)
        if tools:
            lines.append(f"{'investigated_using':<18} {', '.join(tools)}")
        return lines

    @classmethod
    def _render_evidence(cls, item: dict[str, Any]) -> list[str]:
        lines = [f"EVIDENCE FOR {item.get('id')}"]
        support = item.get("supporting_episodes") or []
        contradictions = item.get("contradicting_episodes") or []
        tools = cls._tool_evidence_names(item)
        lines.append(f"Supporting episodes    {len(support)}")
        lines.extend(f"  + {episode_id}" for episode_id in support[:20])
        lines.append(f"Contradicting episodes {len(contradictions)}")
        lines.extend(f"  - {episode_id}" for episode_id in contradictions[:20])
        if tools:
            lines.append("Investigated using")
            lines.extend(f"  * {tool}" for tool in tools[:20])
        return lines

    @staticmethod
    def _tool_evidence_names(item: dict[str, Any]) -> list[str]:
        names = []
        for evidence in item.get("tool_evidence") or []:
            if isinstance(evidence, dict):
                name = evidence.get("tool") or evidence.get("name")
            else:
                name = str(evidence)
            if name and name not in names:
                names.append(str(name))
        return names

    @staticmethod
    def _yes(value: bool) -> str:
        return "yes" if value else "no"
