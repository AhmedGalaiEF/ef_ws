from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import unittest

from agent.checkpoint import CheckpointStore
from agent.cli.router import G1Agent
from agent.learning import LearningManager
from agent.memory.manager import MemoryManager
from agent.memory.episodic import new_episode
from agent.memory.procedural import ProceduralAdaptation
from agent.memory.semantic import SemanticClaim
from agent.models import EventType, IntentType, LifecycleState, PlannerDecision, RuntimeCheckpoint
from agent.reset import ResetManager
from agent.settings.manager import SettingsManager
from agent.skills import SkillRegistry
from agent.state import MockRobotStateSource


class NoopPlanner:
    def decide(self, planner_input):
        return PlannerDecision(intent=IntentType.NO_ACTION)


class ResetTacitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="g1_reset_tacit_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def make_agent(self) -> G1Agent:
        return G1Agent(
            planner=NoopPlanner(),
            skills=SkillRegistry(skills={}, backend_label="test"),
            state_source=MockRobotStateSource(),
            settings=SettingsManager(path=self.tmp / "settings.json"),
            memory=MemoryManager(base_dir=self.tmp),
            checkpoint_store=CheckpointStore(self.tmp / "checkpoint.json"),
            auto_confirm=True,
        )

    def test_runtime_reset_preserves_learned_knowledge(self) -> None:
        agent = self.make_agent()
        agent.memory.semantic.upsert(SemanticClaim(claim="operator prefers neutral wave", confidence=0.9))
        agent.memory.procedural.add(
            ProceduralAdaptation(skill="wave", recommended_parameters={"pre_pose": "neutral"}, confidence=0.9)
        )
        agent.checkpoint_store.save(
            RuntimeCheckpoint(
                last_cognitive_timestamp=1.0,
                lifecycle_state=LifecycleState.AWAKE,
                last_event_type=EventType.USER_MESSAGE,
                last_decision=IntentType.CONVERSATION,
                last_robot_state_summary={},
            )
        )
        result = agent.reset("runtime")
        self.assertTrue(result.ok)
        self.assertFalse((self.tmp / "checkpoint.json").exists())
        self.assertEqual(len(agent.memory.semantic.all()), 1)
        self.assertEqual(len(agent.memory.procedural.all()), 1)

    def test_learned_reset_clears_tacit_but_keeps_episodes_and_static_context(self) -> None:
        agent = self.make_agent()

        agent.memory.episodic.append(new_episode(goal="raw episode"))
        agent.memory.semantic.upsert(SemanticClaim(claim="learned fact", confidence=0.8))
        agent.memory.procedural.add(ProceduralAdaptation(skill="wave", recommended_parameters={"pre_pose": "neutral"}))
        agent.learning.propose_empirical_memory(claim="empirical claim", supporting_episodes=[], confidence=0.7)
        result = agent.reset("learned")
        self.assertTrue(result.ok)
        self.assertEqual(len(agent.memory.episodic.all()), 1)
        self.assertEqual(agent.memory.semantic.all(), [])
        self.assertEqual(agent.memory.procedural.all(), [])
        self.assertEqual(agent.learning.learned.all(), [])

    def test_full_reset_clears_experience_and_marks_first_boot(self) -> None:
        agent = self.make_agent()

        agent.memory.episodic.append(new_episode(goal="old episode"))
        agent.memory.semantic.upsert(SemanticClaim(claim="old learned fact"))
        agent.memory.procedural.add(ProceduralAdaptation(skill="wave"))
        agent.memory.autobiography.append("old autobiography")
        result = agent.reset("full")
        self.assertTrue(result.ok)
        self.assertEqual(agent.boot_event, EventType.AGENT_FIRST_BOOT)
        self.assertEqual(agent.memory.episodic.all(), [])
        self.assertEqual(agent.memory.semantic.all(), [])
        self.assertEqual(agent.memory.procedural.all(), [])
        self.assertEqual(agent.memory.autobiography.all(), [])

    def test_confirmation_requires_exact_text(self) -> None:
        self.assertFalse(ResetManager.confirmation_matches("full", "y"))
        self.assertFalse(ResetManager.confirmation_matches("full", "reset full"))
        self.assertTrue(ResetManager.confirmation_matches("full", "RESET FULL"))

    def test_tacit_ui_shows_behavioral_use_without_raw_telemetry(self) -> None:
        agent = self.make_agent()
        agent.settings.set("learning.automatic_level_max", 3, persist=False)
        agent.memory.procedural.add(
            ProceduralAdaptation(
                skill="wave",
                condition={"workspace": "rear"},
                recommended_parameters={"pre_pose": "neutral_before_wave"},
                confidence=0.93,
                derived_from=["ep_1", "ep_2"],
            )
        )
        lines = "\n".join(agent.tacit_lines(panel="procedural"))
        self.assertIn("Behavioral use  ACTIVE", lines)
        self.assertNotIn("joint_positions", lines)
        self.assertNotIn("chain-of-thought", lines.lower())

    def test_reset_learned_removes_deleted_item_from_retrieval(self) -> None:
        agent = self.make_agent()

        agent.memory.semantic.upsert(SemanticClaim(claim="wave secret learned pattern", confidence=0.9))
        self.assertTrue(agent.memory.retrieve("wave secret")["semantic"])
        agent.reset("learned")
        self.assertFalse(agent.memory.retrieve("wave secret")["semantic"])


if __name__ == "__main__":
    unittest.main()
