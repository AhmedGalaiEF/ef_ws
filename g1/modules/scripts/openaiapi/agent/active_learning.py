"""Deterministic active-learning question gate and provenance builder."""
from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from .models import LearningQuestionProposal, MemoryProposal
from .monitor import MonitorEventBus


DECLINE_PATTERNS = (
    "i don't know",
    "i do not know",
    "dont ask",
    "don't ask",
    "skip",
    "no idea",
    "not now",
    "keine ahnung",
    "frag nicht",
)


@dataclass
class LearningQuestionRecord:
    id: str
    proposal: LearningQuestionProposal
    status: str
    created_at: float = field(default_factory=time.time)
    shown_at: Optional[float] = None
    answered_at: Optional[float] = None
    answer_text: str = ""
    outcome: str = ""

    def snapshot(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "question": self.proposal.question,
            "topic": self.proposal.topic,
            "reason_summary": self.proposal.reason_summary,
            "intended_memory_type": self.proposal.intended_memory_type,
            "status": self.status,
            "created_at": self.created_at,
            "shown_at": self.shown_at,
            "answered_at": self.answered_at,
            "answer_text": self.answer_text,
            "outcome": self.outcome,
        }


class ActiveLearningManager:
    def __init__(self, *, monitor: Optional[MonitorEventBus] = None) -> None:
        self.monitor = monitor
        self.pending: Optional[LearningQuestionRecord] = None
        self.history: List[LearningQuestionRecord] = []
        self.last_autonomous_learning_question_timestamp: Optional[float] = None

    def consider(
        self,
        proposal: Optional[LearningQuestionProposal],
        *,
        settings: Any,
        interaction_state: str = "idle",
    ) -> Optional[LearningQuestionRecord]:
        if proposal is None:
            return None
        self._emit("learning_question_proposed", proposal.question, metadata=proposal.model_dump())
        allowed, reason = self._allowed(proposal, settings=settings, interaction_state=interaction_state)
        if not allowed:
            record = LearningQuestionRecord(
                id=self._new_id(),
                proposal=proposal,
                status="deferred",
                outcome=reason,
            )
            if settings.active_learning.store_rejected_questions:
                self.history.append(record)
            self._emit("learning_question_deferred", reason, references=[record.id])
            return None
        record = LearningQuestionRecord(id=self._new_id(), proposal=proposal, status="pending", shown_at=time.time())
        self.pending = record
        self.history.append(record)
        self.last_autonomous_learning_question_timestamp = record.shown_at
        self._emit("learning_question_shown", proposal.question, references=[record.id])
        return record

    def has_pending(self, *, settings: Any) -> bool:
        self._expire_if_needed(settings)
        return self.pending is not None

    def answer(self, text: str, *, settings: Any) -> tuple[Optional[LearningQuestionRecord], Optional[MemoryProposal]]:
        self._expire_if_needed(settings)
        if self.pending is None:
            return None, None
        record = self.pending
        self.pending = None
        cleaned = str(text).strip()
        record.answered_at = time.time()
        record.answer_text = cleaned
        if self._is_decline(cleaned):
            record.status = "declined"
            record.outcome = "declined_by_user"
            self._emit("learning_question_declined", cleaned[:120], references=[record.id])
            return record, None
        record.status = "answered"
        record.outcome = "user_answer_received"
        self._emit("learning_question_answered", cleaned[:160], references=[record.id])
        proposal = self._memory_proposal_from_answer(record)
        self._emit("learning_memory_proposed", proposal.kind, references=[record.id], metadata=proposal.model_dump())
        return record, proposal

    def skip(self, *, reason: str = "operator_skip") -> Optional[LearningQuestionRecord]:
        if self.pending is None:
            return None
        record = self.pending
        self.pending = None
        record.status = "declined"
        record.outcome = reason
        record.answered_at = time.time()
        self._emit("learning_question_declined", reason, references=[record.id])
        return record

    def should_consume_text_as_answer(self, text: str, *, settings: Any) -> bool:
        if str(text).lstrip().startswith("/"):
            return False
        return self.has_pending(settings=settings)

    def snapshot(self, *, settings: Any) -> Dict[str, Any]:
        self._expire_if_needed(settings)
        now = time.time()
        last_ts = self.last_autonomous_learning_question_timestamp
        cooldown = float(settings.active_learning.cooldown_s)
        remaining = 0.0 if last_ts is None else max(0.0, cooldown - (now - last_ts))
        today_start = now - 24.0 * 3600.0
        recent = [record for record in self.history if record.created_at >= today_start]
        answered = [record for record in self.history if record.status == "answered"]
        declined = [record for record in self.history if record.status == "declined"]
        last = self.history[-1] if self.history else None
        return {
            "enabled": bool(settings.active_learning.enabled),
            "autonomous_questions_allowed": bool(settings.active_learning.allow_autonomous_questions),
            "cooldown_remaining_s": remaining,
            "pending_question": None if self.pending is None else self.pending.snapshot(),
            "questions_today": len(recent),
            "answered": len(answered),
            "declined": len(declined),
            "last_question": None if last is None else last.snapshot(),
            "last_answer": answered[-1].answer_text if answered else "",
            "last_learning_result": last.outcome if last is not None else "",
        }

    def _allowed(self, proposal: LearningQuestionProposal, *, settings: Any, interaction_state: str) -> tuple[bool, str]:
        cfg = settings.active_learning
        if not cfg.enabled:
            return False, "active_learning.enabled=false"
        if not cfg.allow_autonomous_questions:
            return False, "active_learning.allow_autonomous_questions=false"
        if self.pending is not None:
            return False, "another learning question is already pending"
        if int(cfg.maximum_pending_questions) <= 0:
            return False, "maximum_pending_questions=0"
        gap = proposal.confidence_gap
        if gap is not None and float(gap) < float(cfg.minimum_confidence_gap):
            return False, "confidence gap below active_learning.minimum_confidence_gap"
        now = time.time()
        if self.last_autonomous_learning_question_timestamp is not None:
            elapsed = now - self.last_autonomous_learning_question_timestamp
            if elapsed < float(cfg.cooldown_s):
                return False, f"cooldown active for {float(cfg.cooldown_s) - elapsed:.1f}s"
        if interaction_state == "idle" and not cfg.allow_during_idle:
            return False, "questions during idle are disabled"
        if interaction_state == "task" and not cfg.allow_during_task_execution:
            return False, "questions during task execution are disabled"
        if interaction_state == "scenario" and not cfg.allow_during_active_scenario:
            return False, "questions during active scenario are disabled"
        duplicate = self._find_duplicate(proposal, max_age_s=float(cfg.duplicate_suppression_s))
        if duplicate is not None:
            return False, f"duplicate of recent question {duplicate.id}"
        return True, "allowed"

    def _find_duplicate(self, proposal: LearningQuestionProposal, *, max_age_s: float) -> Optional[LearningQuestionRecord]:
        wanted = self._normalize(proposal.question)
        now = time.time()
        for record in reversed(self.history):
            if now - record.created_at > max_age_s:
                continue
            score = self._similarity(wanted, self._normalize(record.proposal.question))
            if score >= 0.78 or (record.proposal.topic == proposal.topic and score >= 0.40):
                return record
        if self.pending is not None:
            score = self._similarity(wanted, self._normalize(self.pending.proposal.question))
            if score >= 0.78 or (self.pending.proposal.topic == proposal.topic and score >= 0.40):
                return self.pending
        return None

    def _expire_if_needed(self, settings: Any) -> None:
        if self.pending is None:
            return
        shown_at = self.pending.shown_at or self.pending.created_at
        if time.time() - shown_at <= float(settings.active_learning.unanswered_timeout_s):
            return
        record = self.pending
        self.pending = None
        record.status = "unanswered"
        record.outcome = "timeout"
        self._emit("learning_question_timeout", record.proposal.question, references=[record.id])

    def _memory_proposal_from_answer(self, record: LearningQuestionRecord) -> MemoryProposal:
        intended = record.proposal.intended_memory_type
        provenance = {
            "source_type": "user_answer",
            "question_id": record.id,
            "timestamp": record.answered_at,
            "topic": record.proposal.topic,
            "question": record.proposal.question,
            "answer": record.answer_text,
            "trust": "operator-provided",
            "intended_memory_type": intended,
            "reason_summary": record.proposal.reason_summary,
            "related_memory_ids": list(record.proposal.related_memory_ids),
            "related_episode_ids": list(record.proposal.related_episode_ids),
        }
        kind = "semantic"
        if intended == "episodic":
            kind = "episodic"
        elif intended == "procedural_hint":
            kind = "procedural"
        confidence = 0.85 if intended in {"user_preference", "object_fact", "environment_fact"} else 0.55
        if kind == "episodic":
            content = {
                "goal": f"user answered active-learning question about {record.proposal.topic}",
                "observations": [provenance],
                "outcome": "answered",
            }
        elif kind == "procedural":
            content = {
                "skill": str(record.proposal.topic or "unknown"),
                "condition": {"source_type": "user_answer", "question_id": record.id},
                "recommended_parameters": {
                    "operator_hint": record.answer_text,
                    "question": record.proposal.question,
                    "requires_validation": True,
                },
            }
        else:
            content = {
                "claim": (
                    f"Operator answer for {record.proposal.topic}: {record.answer_text} "
                    f"(question_id={record.id}, source_type=user_answer, trust=operator-provided)"
                ),
                "provenance": provenance,
            }
        return MemoryProposal(
            kind=kind,
            content=content,
            confidence=confidence,
            derived_from=[record.id] + list(record.proposal.related_memory_ids) + list(record.proposal.related_episode_ids),
        )

    @staticmethod
    def _is_decline(text: str) -> bool:
        lowered = text.strip().lower()
        return any(pattern in lowered for pattern in DECLINE_PATTERNS)

    @staticmethod
    def _normalize(text: str) -> str:
        lowered = text.lower()
        lowered = re.sub(r"[^a-z0-9äöüß]+", " ", lowered)
        tokens = [
            token
            for token in lowered.split()
            if token
            not in {
                "the",
                "a",
                "an",
                "to",
                "do",
                "you",
                "why",
                "should",
                "prefer",
                "before",
                "first",
                "my",
                "i",
            }
        ]
        return " ".join(tokens)

    @staticmethod
    def _similarity(left: str, right: str) -> float:
        seq = SequenceMatcher(None, left, right).ratio()
        left_tokens = set(left.split())
        right_tokens = set(right.split())
        if not left_tokens or not right_tokens:
            return seq
        jaccard = len(left_tokens & right_tokens) / float(len(left_tokens | right_tokens))
        return max(seq, jaccard)

    @staticmethod
    def _new_id() -> str:
        return "lq_" + uuid.uuid4().hex[:12]

    def _emit(
        self,
        event: str,
        summary: str,
        *,
        references: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.monitor is not None:
            self.monitor.emit("active_learning", event, summary, references=references, metadata=metadata)
