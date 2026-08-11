"""Bounded experiential learning on top of the existing memory stores."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import json
import os
import shutil
import time
import uuid

from pydantic import BaseModel, ConfigDict, Field

from .memory.episodic import Episode, new_episode
from .memory.manager import MemoryManager
from .memory.procedural import ProceduralAdaptation
from .monitor import MonitorEventBus
from .outcomes import SkillOutcome
from .semantic_state import SemanticState


class LearnedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: "lm_" + uuid.uuid4().hex[:12])
    claim: str
    confidence: float = 0.5
    supporting_episodes: List[str] = Field(default_factory=list)
    contradicting_episodes: List[str] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    last_validation_time: float = Field(default_factory=time.time)
    status: str = "candidate"
    applicable_context: Dict[str, Any] = Field(default_factory=dict)
    version: int = 1
    risk_level: int = 1


class LearnedMemoryStore:
    def __init__(self, path: Union[Path, str]) -> None:
        self.path = Path(path).expanduser()

    def all(self) -> List[LearnedClaim]:
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8") or "[]")
        except Exception:
            return []
        claims: List[LearnedClaim] = []
        for item in raw if isinstance(raw, list) else []:
            try:
                claims.append(LearnedClaim.model_validate(item))
            except Exception:
                continue
        return claims

    def save_all(self, claims: List[LearnedClaim]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            json.dumps([claim.model_dump() for claim in claims], ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    def upsert(self, claim: LearnedClaim) -> LearnedClaim:
        claims = self.all()
        for idx, existing in enumerate(claims):
            if existing.id == claim.id or existing.claim == claim.claim:
                merged = claim.model_copy(update={"id": existing.id, "created_at": existing.created_at, "version": existing.version + 1})
                claims[idx] = merged
                self.save_all(claims)
                return merged
        claims.append(claim)
        self.save_all(claims)
        return claim

    def stats(self) -> Dict[str, int]:
        counts = Counter(claim.status for claim in self.all())
        return dict(counts)


class SkillStatisticsStore:
    def __init__(self, path: Union[Path, str]) -> None:
        self.path = Path(path).expanduser()

    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8") or "{}")
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def save(self, data: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)


class LearningManager:
    def __init__(
        self,
        memory: MemoryManager,
        *,
        monitor: Optional[MonitorEventBus] = None,
        base_dir: Optional[Union[Path, str]] = None,
    ) -> None:
        self.memory = memory
        base = Path(base_dir).expanduser() if base_dir is not None else self._infer_base_dir(memory)
        base.mkdir(parents=True, exist_ok=True)
        self.base_dir = base
        self.learned = LearnedMemoryStore(base / "learned_semantic.json")
        self.stats_store = SkillStatisticsStore(base / "skill_stats.json")
        self.pins_path = base / "pinned_memories.json"
        self.monitor = monitor
        self.low_value_suppressed = False

    @staticmethod
    def _infer_base_dir(memory: MemoryManager) -> Path:
        try:
            return memory.episodic.path.parent
        except Exception:
            return Path(os.environ.get("G1_AGENT_MEMORY_DIR", "~/.g1_agent/memory")).expanduser()

    def record_skill_outcome(
        self,
        outcome: SkillOutcome,
        *,
        before: SemanticState,
        after: SemanticState,
        settings: Any,
    ) -> Optional[str]:
        if not bool(getattr(getattr(settings, "learning", object()), "enabled", True)):
            return None
        is_low_value = outcome.goal_reached and not outcome.anomalies and outcome.failure_type is None
        if self.low_value_suppressed and is_low_value:
            self._emit("memory", "episode_suppressed", f"suppressed routine episode for {outcome.skill_id}")
            return None
        episode = new_episode(
            goal=f"skill:{outcome.skill_id}",
            initial_state=before.model_dump(),
            actions=[outcome.skill_id],
            observations=[
                f"command_accepted={outcome.command_accepted}",
                f"execution_completed={outcome.execution_completed}",
                f"goal_reached={outcome.goal_reached}",
                f"safe={outcome.safe}",
            ],
            outcome="success" if outcome.goal_reached else "failure",
            anomalies=list(outcome.anomalies) + ([outcome.failure_type] if outcome.failure_type else []),
        )
        self.memory.episodic.append(episode)
        self._update_skill_stats(outcome, episode.id)
        self._emit("memory", "episode_created", f"{episode.id} stored for {outcome.skill_id}", references=[episode.id])
        self._maybe_propose_empirical_claim(outcome, settings=settings)
        self._maybe_propose_wave_adaptation(outcome, settings=settings)
        return episode.id

    def propose_empirical_memory(
        self,
        *,
        claim: str,
        supporting_episodes: List[str],
        contradicting_episodes: Optional[List[str]] = None,
        confidence: float = 0.5,
        applicable_context: Optional[Dict[str, Any]] = None,
        risk_level: int = 1,
    ) -> LearnedClaim:
        verified_support = self._existing_episode_ids(set(supporting_episodes))
        verified_contradictions = self._existing_episode_ids(set(contradicting_episodes or []))
        confidence = max(0.0, min(1.0, float(confidence)))
        if not verified_support:
            confidence = min(confidence, 0.25)
        learned = self.learned.upsert(
            LearnedClaim(
                claim=str(claim).strip(),
                confidence=confidence,
                supporting_episodes=sorted(verified_support),
                contradicting_episodes=sorted(verified_contradictions),
                applicable_context=dict(applicable_context or {}),
                risk_level=int(risk_level),
            )
        )
        self._emit("learning", "memory_proposed", learned.claim, references=learned.supporting_episodes)
        return learned

    def propose_procedural_adaptation(
        self,
        *,
        skill: str,
        condition: Dict[str, Any],
        recommended_parameters: Dict[str, Any],
        confidence: float,
        derived_from: List[str],
        settings: Any,
    ) -> ProceduralAdaptation:
        verified = sorted(self._existing_episode_ids(set(derived_from)))
        adaptation = ProceduralAdaptation(
            skill=skill,
            condition=condition,
            recommended_parameters=recommended_parameters,
            confidence=max(0.0, min(1.0, float(confidence))),
            derived_from=verified,
        )
        automatic_max = int(getattr(getattr(settings, "learning", object()), "automatic_level_max", 1))
        if automatic_max >= 2 and verified:
            self.memory.procedural.add(adaptation)
            self._emit("learning", "procedural_adaptation_proposed", f"{skill}: {recommended_parameters}", references=verified)
        else:
            self._emit("learning", "procedural_adaptation_candidate", f"{skill}: {recommended_parameters}", references=verified)
        return adaptation

    def report_memory_contradiction(self, claim_id: str, episode_id: str) -> Optional[LearnedClaim]:
        claims = self.learned.all()
        for idx, claim in enumerate(claims):
            if claim.id != claim_id:
                continue
            contradictions = sorted(set(claim.contradicting_episodes + [episode_id]))
            support = len(claim.supporting_episodes)
            confidence = support / max(1, support + len(contradictions))
            updated = claim.model_copy(
                update={
                    "contradicting_episodes": contradictions,
                    "confidence": confidence,
                    "status": "contested" if contradictions else claim.status,
                    "last_validation_time": time.time(),
                    "version": claim.version + 1,
                }
            )
            claims[idx] = updated
            self.learned.save_all(claims)
            self._emit("learning", "memory_contested", updated.claim, references=[episode_id])
            return updated
        return None

    def request_episode_search(self, query: str, *, top_k: int = 5) -> List[Episode]:
        return self.memory.episodic.search(query, top_k=top_k)

    def pinned_ids(self) -> Set[str]:
        if not self.pins_path.exists():
            return set()
        try:
            raw = json.loads(self.pins_path.read_text(encoding="utf-8") or "[]")
            return {str(item) for item in raw if str(item).strip()}
        except Exception:
            return set()

    def pin(self, memory_id: str) -> None:
        pins = self.pinned_ids()
        pins.add(str(memory_id))
        self.pins_path.parent.mkdir(parents=True, exist_ok=True)
        self.pins_path.write_text(json.dumps(sorted(pins), indent=2), encoding="utf-8")
        self._emit("memory", "memory_pinned", str(memory_id), references=[str(memory_id)])

    def unpin(self, memory_id: str) -> None:
        pins = self.pinned_ids()
        pins.discard(str(memory_id))
        self.pins_path.parent.mkdir(parents=True, exist_ok=True)
        self.pins_path.write_text(json.dumps(sorted(pins), indent=2), encoding="utf-8")
        self._emit("memory", "memory_unpinned", str(memory_id), references=[str(memory_id)])

    def consolidate(self, *, settings: Any) -> Dict[str, Any]:
        start = time.time()
        self._emit("maintenance", "maintenance_started", "memory consolidation started")
        self._apply_disk_policy(settings=settings)
        moved = self._bound_hot_episodes(settings=settings)
        stats = self.stats_store.load()
        result = {
            "duration_s": max(0.0, time.time() - start),
            "episodes_archived": moved,
            "skill_stats": len(stats),
            "learned_claims": len(self.learned.all()),
            "disk": self.disk_stats(settings=settings),
        }
        self._emit("maintenance", "maintenance_completed", f"archived={moved} learned={result['learned_claims']}")
        return result

    def memory_stats(self, *, settings: Optional[Any] = None) -> Dict[str, Any]:
        episodes = self.memory.episodic.all()
        learned = self.learned.all()
        procedural = self.memory.procedural.all()
        semantic = self.memory.semantic.all()
        hot_limit = int(getattr(getattr(settings, "memory", object()), "hot_episode_limit", 5000)) if settings else 5000
        return {
            "working_events": self.monitor.max_events if self.monitor else 0,
            "hot_episodes": len(episodes),
            "hot_episode_limit": hot_limit,
            "semantic_memories": len(semantic),
            "procedural_memories": len(procedural),
            "learned_claims": len(learned),
            "learned_status": self.learned.stats(),
            "pinned": len(self.pinned_ids()),
            "base_dir": str(self.base_dir),
        }

    def disk_stats(self, *, settings: Optional[Any] = None) -> Dict[str, Any]:
        total_size = 0
        for path in self.base_dir.rglob("*"):
            if path.is_file():
                try:
                    total_size += path.stat().st_size
                except OSError:
                    pass
        usage = shutil.disk_usage(self.base_dir)
        free_pct = 100.0 * usage.free / max(1, usage.total)
        warning = float(getattr(getattr(settings, "disk", object()), "warning_free_pct", 20)) if settings else 20.0
        critical = float(getattr(getattr(settings, "disk", object()), "critical_free_pct", 10)) if settings else 10.0
        return {
            "memory_bytes": total_size,
            "memory_mb": total_size / (1024 * 1024),
            "free_pct": free_pct,
            "status": "critical" if free_pct <= critical else "warning" if free_pct <= warning else "ok",
        }

    def _maybe_propose_empirical_claim(self, outcome: SkillOutcome, *, settings: Any) -> None:
        if not bool(getattr(getattr(settings, "learning", object()), "empirical_memory_enabled", True)):
            return
        if outcome.goal_reached:
            return
        minimum = int(getattr(getattr(settings, "learning", object()), "minimum_support_for_candidate", 3))
        stats = self.stats_store.load()
        key = f"{outcome.skill_id}:{outcome.failure_type or 'failure'}"
        record = stats.get(key, {})
        failures = list(record.get("failure_episode_ids", []))
        if len(failures) < minimum:
            return
        claim = (
            f"{outcome.skill_id} has repeated {outcome.failure_type or 'failure'} outcomes "
            f"in similar semantic contexts."
        )
        confidence = min(0.9, 0.45 + 0.08 * len(failures))
        self.propose_empirical_memory(
            claim=claim,
            supporting_episodes=failures,
            confidence=confidence,
            applicable_context={"skill": outcome.skill_id, "failure_type": outcome.failure_type},
            risk_level=1,
        )

    def _maybe_propose_wave_adaptation(self, outcome: SkillOutcome, *, settings: Any) -> None:
        if outcome.skill_id != "wave" or outcome.goal_reached:
            return
        if not bool(getattr(getattr(settings, "learning", object()), "procedural_learning_enabled", True)):
            return
        minimum = int(getattr(getattr(settings, "learning", object()), "minimum_support_for_procedure", 10))
        stats = self.stats_store.load()
        failures = list(stats.get("wave:goal_not_reached", {}).get("failure_episode_ids", []))
        if len(failures) < minimum:
            return
        self.propose_procedural_adaptation(
            skill="wave",
            condition={"recent_failure_type": "goal_not_reached"},
            recommended_parameters={"pre_pose": "neutral_before_wave", "risk_level": 2},
            confidence=min(0.85, 0.55 + 0.03 * len(failures)),
            derived_from=failures,
            settings=settings,
        )

    def _update_skill_stats(self, outcome: SkillOutcome, episode_id: str) -> None:
        stats = self.stats_store.load()
        key = f"{outcome.skill_id}:{outcome.failure_type or 'success'}"
        record = dict(stats.get(key, {}))
        record["samples"] = int(record.get("samples", 0)) + 1
        if outcome.goal_reached:
            record["success"] = int(record.get("success", 0)) + 1
        else:
            record["failure"] = int(record.get("failure", 0)) + 1
            ids = list(record.get("failure_episode_ids", []))
            ids.append(episode_id)
            record["failure_episode_ids"] = ids[-200:]
        durations = list(record.get("duration_s", []))
        durations.append(float(outcome.metrics.get("duration_s", 0.0)))
        record["duration_s"] = durations[-200:]
        stats[key] = record
        self.stats_store.save(stats)

    def _bound_hot_episodes(self, *, settings: Any) -> int:
        limit = int(getattr(getattr(settings, "memory", object()), "hot_episode_limit", 5000))
        episodes = self.memory.episodic.all()
        if len(episodes) <= limit:
            return 0
        protected = self._protected_episode_ids()
        scored = sorted(episodes, key=lambda ep: self._retention_score(ep, protected), reverse=True)
        keep = scored[:limit]
        archive = scored[limit:]
        archive_path = self.base_dir / "episodic_archive.jsonl"
        if archive:
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            with archive_path.open("a", encoding="utf-8") as handle:
                for episode in archive:
                    handle.write(episode.to_json() + "\n")
            self.memory.episodic.path.write_text(
                "".join(episode.to_json() + "\n" for episode in sorted(keep, key=lambda ep: ep.timestamp)),
                encoding="utf-8",
            )
        return len(archive)

    def _protected_episode_ids(self) -> Set[str]:
        protected: Set[str] = set()
        protected.update(self.pinned_ids())
        for claim in self.learned.all():
            if claim.status in {"active", "candidate", "contested"}:
                protected.update(claim.supporting_episodes)
                protected.update(claim.contradicting_episodes)
        for adaptation in self.memory.procedural.all():
            protected.update(adaptation.derived_from)
        return protected

    @staticmethod
    def _retention_score(episode: Episode, protected: Set[str]) -> float:
        score = 0.0
        age_days = max(0.0, (time.time() - episode.timestamp) / 86400.0)
        score += max(0.0, 30.0 - age_days)
        if episode.id in protected:
            score += 1000.0
        text = " ".join([episode.goal, episode.outcome, " ".join(episode.anomalies)]).lower()
        if any(term in text for term in ("failure", "safety", "fall", "fault", "critical", "denied")):
            score += 200.0
        if episode.anomalies:
            score += 50.0
        return score

    def _apply_disk_policy(self, *, settings: Any) -> None:
        disk = self.disk_stats(settings=settings)
        self.low_value_suppressed = disk["status"] == "critical"
        if disk["status"] != "ok":
            self._emit("maintenance", "disk_pressure", f"disk status={disk['status']} free={disk['free_pct']:.1f}%")

    def _existing_episode_ids(self, wanted: Set[str]) -> Set[str]:
        if not wanted:
            return set()
        return {episode.id for episode in self.memory.episodic.all() if episode.id in wanted}

    def _emit(self, category: str, event: str, summary: str, *, references: Optional[List[str]] = None) -> None:
        if self.monitor is not None:
            self.monitor.emit(category, event, summary, references=references or [])
