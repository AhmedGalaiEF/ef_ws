from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shutil
import tempfile
import unittest

from agent.attention import AttentionManager
from agent.capabilities import PolicyDecision
from agent.checkpoint import CheckpointStore
from agent.cli.router import G1Agent
from agent.learning import LearningManager
from agent.memory.manager import MemoryManager
from agent.memory.episodic import new_episode
from agent.memory.procedural import ProceduralAdaptation
from agent.models import EventType, IntentType, MemoryProposal, PlannerDecision
from agent.outcomes import OutcomeEvaluator
from agent.scheduler import CognitiveScheduler
from agent.semantic_state import SemanticState
from agent.settings.manager import SettingsManager
from agent.skills import Skill, SkillInvocationOutcome, SkillRegistry, SkillResult
from agent.state import MockRobotStateSource


class CountingPlanner:
    def __init__(self) -> None:
        self.calls = 0

    def decide(self, planner_input):
        self.calls += 1
        return PlannerDecision(intent=IntentType.NO_ACTION)


class RuntimeRevisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="g1_agent_tests_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_attention_ignores_unimportant_tick(self) -> None:
        decision = AttentionManager().decide(
            event_type="cognitive_tick",
            semantic_state=SemanticState(),
            semantic_changes=[],
            settings=SettingsManager().effective(),
        )
        self.assertIn(decision.action, {"ignore", "cognition_background"})

    def test_task_failure_triggers_cognition(self) -> None:
        decision = AttentionManager().decide(
            event_type="skill_failed",
            semantic_state=SemanticState(),
            semantic_changes=[],
            settings=SettingsManager().effective(),
        )
        self.assertEqual(decision.action, "cognition_now")
        self.assertEqual(decision.priority, 3)

    def test_scheduler_p1_before_background(self) -> None:
        scheduler = CognitiveScheduler()
        scheduler.enqueue(EventType.COGNITIVE_TICK, priority=5)
        scheduler.enqueue(EventType.USER_MESSAGE, priority=1)
        self.assertEqual(scheduler.pop_next().event_type, EventType.USER_MESSAGE)

    def test_tick_short_circuits_planner_when_no_attention(self) -> None:
        planner = CountingPlanner()
        agent = G1Agent(
            planner=planner,
            skills=SkillRegistry(skills={}, backend_label="test"),
            state_source=MockRobotStateSource(),
            settings=SettingsManager(path=self.tmp / "settings.json"),
            memory=MemoryManager(base_dir=self.tmp),
            checkpoint_store=CheckpointStore(self.tmp / "checkpoint.json"),
            auto_confirm=True,
        )
        agent.boot()
        planner.calls = 0
        outcome = agent.handle_cognitive_tick()
        self.assertEqual(outcome.decision.intent, IntentType.NO_ACTION)
        self.assertEqual(planner.calls, 0)

    def test_skill_failure_creates_episode_and_candidate(self) -> None:
        settings = SettingsManager(path=self.tmp / "settings.json")
        settings.set("learning.minimum_support_for_candidate", 2, persist=False)
        memory = MemoryManager(base_dir=self.tmp)
        learning = LearningManager(memory, base_dir=self.tmp)
        evaluator = OutcomeEvaluator()
        before = SemanticState(arm_state="commandable")
        after = SemanticState(arm_state="commandable")
        invocation = SkillInvocationOutcome(
            policy=PolicyDecision(allowed=True, requires_approval=False, risk="low", reason="test"),
            skill_mode=settings.get_skill_mode("wave"),
            status="executed",
            result=SkillResult(ok=False, message="goal not reached"),
        )
        for idx in range(2):
            outcome = evaluator.evaluate(
                skill_id="wave",
                invocation_id=f"wave_{idx}",
                invocation_outcome=invocation,
                before=before,
                after=after,
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
            )
            learning.record_skill_outcome(outcome, before=before, after=after, settings=settings.effective())
        self.assertEqual(len(memory.episodic.all()), 2)
        self.assertGreaterEqual(len(learning.learned.all()), 1)

    def test_contradiction_updates_confidence(self) -> None:
        memory = MemoryManager(base_dir=self.tmp)
        ep = memory.episodic.search("", top_k=1)
        if not ep:
            memory.episodic.append(new_episode(goal="skill:wave"))
        episode_id = memory.episodic.all()[0].id
        learning = LearningManager(memory, base_dir=self.tmp)
        claim = learning.propose_empirical_memory(
            claim="wave sometimes fails",
            supporting_episodes=[episode_id],
            confidence=0.8,
        )
        updated = learning.report_memory_contradiction(claim.id, episode_id)
        self.assertIsNotNone(updated)
        self.assertEqual(updated.status, "contested")
        self.assertLess(updated.confidence, 0.8)

    def test_memory_forgetting_keeps_failure_episode(self) -> None:
        settings = SettingsManager(path=self.tmp / "settings.json")
        settings.set("memory.hot_episode_limit", 10, persist=False)
        memory = MemoryManager(base_dir=self.tmp)
        for idx in range(12):
            memory.apply_proposal(
                MemoryProposal(
                    kind="episodic",
                    content={"goal": f"routine {idx}", "outcome": "success"},
                    confidence=1.0,
                )
            )
        memory.apply_proposal(
            MemoryProposal(
                kind="episodic",
                content={"goal": "safety incident", "outcome": "failure", "anomalies": ["safety"]},
                confidence=1.0,
            )
        )
        learning = LearningManager(memory, base_dir=self.tmp)
        learning.consolidate(settings=settings.effective())
        remaining = memory.episodic.all()
        self.assertLessEqual(len(remaining), 10)
        self.assertTrue(any("safety" in ep.goal for ep in remaining))

    def test_monitor_snapshot_redacts_raw_control_streams(self) -> None:
        agent = G1Agent(
            planner=CountingPlanner(),
            skills=SkillRegistry(skills={}, backend_label="test"),
            state_source=MockRobotStateSource(),
            settings=SettingsManager(path=self.tmp / "settings.json"),
            memory=MemoryManager(base_dir=self.tmp),
            checkpoint_store=CheckpointStore(self.tmp / "checkpoint.json"),
            auto_confirm=True,
        )
        snapshot_text = str(agent.monitor_snapshot())
        self.assertNotIn("joint_positions", snapshot_text)
        self.assertNotIn("joint_velocities", snapshot_text)
        self.assertNotIn("tau", snapshot_text)

    def test_validated_procedural_rule_changes_future_skill_kwargs(self) -> None:
        captured = {}

        def wave_handler(**kwargs):
            captured.update(kwargs)
            return SkillResult(ok=True, message="wave ok", detail=kwargs)

        settings = SettingsManager(path=self.tmp / "settings.json")
        settings.set("learning.automatic_level_max", 3, persist=False)
        memory = MemoryManager(base_dir=self.tmp)
        memory.procedural.add(
            ProceduralAdaptation(
                skill="wave",
                condition={"recent_failure_type": "goal_not_reached"},
                recommended_parameters={"pre_pose": "neutral_before_wave"},
                confidence=0.9,
                derived_from=["ep_test"],
            )
        )
        agent = G1Agent(
            planner=CountingPlanner(),
            skills=SkillRegistry(
                skills={"wave": Skill("wave", "test wave", wave_handler, "test")},
                backend_label="test",
            ),
            state_source=MockRobotStateSource(),
            settings=settings,
            memory=memory,
            checkpoint_store=CheckpointStore(self.tmp / "checkpoint.json"),
            auto_confirm=True,
        )
        planner_input = agent._build_planner_input(
            event=EventType.USER_MESSAGE,
            timestamp=0.0,
            user_text="wave",
            input_source="chat",
            robot_state=MockRobotStateSource().read(),
        )
        decision = PlannerDecision(intent=IntentType.EXECUTE_TASK, requested_skills=["wave"])
        agent._execute_decision(decision, planner_input)
        self.assertEqual(captured.get("learned_pre_pose"), "neutral_before_wave")


if __name__ == "__main__":
    unittest.main()
