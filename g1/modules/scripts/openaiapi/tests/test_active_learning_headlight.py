from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import unittest

from agent.active_learning import ActiveLearningManager
from agent.activity import ActivityManager
from agent.cli.router import G1Agent
from agent.checkpoint import CheckpointStore
from agent.memory.manager import MemoryManager
from agent.models import IntentType, LearningQuestionProposal, PlannerDecision
from agent.settings.manager import SettingsManager
from agent.skills import SkillRegistry
from agent.state import MockRobotStateSource


class ScriptPlanner:
    def __init__(self, decision: PlannerDecision) -> None:
        self.decision = decision
        self.inputs = []

    def decide(self, planner_input):
        self.inputs.append(planner_input)
        return self.decision


class FakeHeadlightRobot:
    def __init__(self, fail: bool = False) -> None:
        self.calls = []
        self.fail = fail

    def headlight(self, *, color="white", intensity=100, duration=None):
        self.calls.append({"color": color, "intensity": intensity, "duration": duration})
        if self.fail:
            raise RuntimeError("headlight failed")
        return 0


def proposal(question: str = "Why return to neutral before waving?") -> LearningQuestionProposal:
    return LearningQuestionProposal(
        question=question,
        topic="wave",
        reason_summary="Repeated wave failures have ambiguous cause.",
        intended_memory_type="procedural_hint",
        confidence_gap=0.5,
    )


class ActiveLearningHeadlightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="g1_active_learning_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def settings(self) -> SettingsManager:
        return SettingsManager(path=self.tmp / "settings.json")

    def test_active_learning_disabled_does_not_show_question(self) -> None:
        settings = self.settings()
        settings.set("active_learning.allow_autonomous_questions", False, persist=False)
        manager = ActiveLearningManager()
        shown = manager.consider(proposal(), settings=settings.effective())
        self.assertIsNone(shown)
        self.assertIsNone(manager.pending)

    def test_cooldown_defers_second_question(self) -> None:
        settings = self.settings()
        settings.set("active_learning.cooldown_s", 120.0, persist=False)
        manager = ActiveLearningManager()
        self.assertIsNotNone(manager.consider(proposal("Why use neutral before waving?"), settings=settings.effective()))
        manager.skip()
        second = manager.consider(proposal("Should I neutralize before a wave?"), settings=settings.effective())
        self.assertIsNone(second)

    def test_duplicate_suppression(self) -> None:
        settings = self.settings()
        settings.set("active_learning.cooldown_s", 0.0, persist=False)
        manager = ActiveLearningManager()
        self.assertIsNotNone(manager.consider(proposal("Why should I move my arm to neutral first?"), settings=settings.effective()))
        manager.skip()
        duplicate = manager.consider(
            proposal("Why do you prefer a neutral arm pose before waving?"),
            settings=settings.effective(),
        )
        self.assertIsNone(duplicate)

    def test_answer_association_and_memory_provenance(self) -> None:
        settings = self.settings()
        manager = ActiveLearningManager()
        shown = manager.consider(proposal(), settings=settings.effective())
        self.assertIsNotNone(shown)
        record, memory_proposal = manager.answer(
            "Because the elbow otherwise starts close to the rear limit.",
            settings=settings.effective(),
        )
        self.assertIsNotNone(record)
        self.assertIsNotNone(memory_proposal)
        self.assertEqual(record.id, memory_proposal.derived_from[0])
        self.assertEqual(memory_proposal.kind, "procedural")
        self.assertEqual(memory_proposal.content["condition"]["source_type"], "user_answer")
        self.assertEqual(memory_proposal.content["condition"]["question_id"], record.id)

    def test_cli_command_not_consumed_as_learning_answer(self) -> None:
        settings = self.settings()
        agent = G1Agent(
            planner=ScriptPlanner(PlannerDecision(intent=IntentType.NO_ACTION)),
            skills=SkillRegistry(skills={}, backend_label="test"),
            state_source=MockRobotStateSource(),
            settings=settings,
            memory=MemoryManager(base_dir=self.tmp),
            checkpoint_store=CheckpointStore(self.tmp / "checkpoint.json"),
            auto_confirm=True,
        )
        agent.active_learning.consider(proposal(), settings=settings.effective())
        self.assertFalse(agent.active_learning.should_consume_text_as_answer("/settings", settings=settings.effective()))
        outcome = agent.handle_cli_text("Because it avoids a joint limit.")
        self.assertEqual(agent.active_learning.pending, None)
        self.assertEqual(agent.planner.inputs[-1].input_source, "learning_answer")
        self.assertIn("learning evidence", outcome.grounded_response)

    def test_purple_headlight_restored_after_thinking(self) -> None:
        settings = self.settings().effective()
        robot = FakeHeadlightRobot()
        activity = ActivityManager(robot=robot)
        with activity.activity("thinking", settings=settings, reason="test"):
            self.assertEqual(robot.calls[0]["color"], (55, 20, 75))
        self.assertEqual(robot.calls[-1]["color"], (0, 0, 0))

    def test_headlight_restored_after_exception(self) -> None:
        settings = self.settings().effective()
        robot = FakeHeadlightRobot()
        activity = ActivityManager(robot=robot)
        with self.assertRaises(RuntimeError):
            with activity.activity("thinking", settings=settings, reason="test"):
                raise RuntimeError("planner failed")
        self.assertEqual(robot.calls[-1]["color"], (0, 0, 0))

    def test_nested_retrieval_keeps_single_restore(self) -> None:
        settings = self.settings().effective()
        robot = FakeHeadlightRobot()
        activity = ActivityManager(robot=robot)
        with activity.activity("thinking", settings=settings, reason="outer"):
            with activity.activity("retrieving", settings=settings, reason="inner"):
                pass
            self.assertNotEqual(robot.calls[-1]["color"], (0, 0, 0))
        off_calls = [call for call in robot.calls if call["color"] == (0, 0, 0)]
        self.assertEqual(len(off_calls), 1)

    def test_operator_override_skips_thinking_light(self) -> None:
        settings = self.settings().effective()
        robot = FakeHeadlightRobot()
        activity = ActivityManager(robot=robot)
        activity.headlight.set_operator_override(True)
        with activity.activity("thinking", settings=settings, reason="test"):
            pass
        self.assertEqual(robot.calls, [])


if __name__ == "__main__":
    unittest.main()
