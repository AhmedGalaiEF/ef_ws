from __future__ import annotations

from pathlib import Path
import random
import shutil
import tempfile
import unittest

from agent.asr import AsrRuntime
from agent.capabilities import CapabilityResolver
from agent.cli.router import G1Agent
from agent.expressive_motion import ExpressiveMotionController
from agent.models import IntentType, PlannerDecision
from agent.navigation import EXPECTED_ROS_TOPICS, NavigationAdapter
from agent.settings.manager import SettingsManager
from agent.skills import SkillRegistry, build_offline_registry
from agent.state import MockRobotStateSource
from agent.visual_observation import VisualObservationTracker


class NoopPlanner:
    def decide(self, planner_input):
        return PlannerDecision(intent=IntentType.NO_ACTION)


class FakeRepeatRobot:
    def __init__(self) -> None:
        self.calls = []

    def repeat(self, **kwargs):
        self.calls.append(kwargs)
        return {"ok": True, **kwargs}


class RobotSideIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="g1_robot_side_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_expected_ros_topics_use_canonical_lowstate_and_odommodestate(self) -> None:
        self.assertEqual(EXPECTED_ROS_TOPICS["/lowstate"], "unitree_hg/msg/LowState")
        self.assertEqual(EXPECTED_ROS_TOPICS["/odommodestate"], "unitree_go/msg/SportModeState")
        self.assertNotIn("rt/lowstate", str(EXPECTED_ROS_TOPICS))

    def test_expressive_motion_selects_csv_and_calls_repeat(self) -> None:
        for name in ("thinking_01.csv", "thinking_02.csv"):
            (self.tmp / name).write_text("t_s,right_arm.shoulder_pitch\n0,0\n", encoding="utf-8")
        settings = SettingsManager(path=self.tmp / "settings.json")
        settings.set("announcements.gesture_enabled", True, persist=False)
        settings.set("expressive_motion.motion_directory", str(self.tmp), persist=False)
        robot = FakeRepeatRobot()
        controller = ExpressiveMotionController(robot=robot, rng=random.Random(1))
        result = controller.run("thinking", settings=settings.effective(), reason="test")
        self.assertTrue(result.ok)
        self.assertEqual(len(robot.calls), 1)
        self.assertTrue(Path(robot.calls[0]["motion_file"]).name.startswith("thinking_"))

    def test_expressive_motion_missing_files_fails_gracefully(self) -> None:
        settings = SettingsManager(path=self.tmp / "settings.json")
        settings.set("announcements.gesture_enabled", True, persist=False)
        settings.set("expressive_motion.motion_directory", str(self.tmp), persist=False)
        result = ExpressiveMotionController(robot=FakeRepeatRobot()).run(
            "thanking", settings=settings.effective(), reason="test"
        )
        self.assertFalse(result.ok)
        self.assertIn("thanking_*.csv", result.message)

    def test_walk_run_mode_capability_uses_setting(self) -> None:
        settings = SettingsManager(path=self.tmp / "settings.json")
        resolver = CapabilityResolver()
        robot_state = MockRobotStateSource(stability="stable").read()
        self.assertTrue(resolver.resolve_skill("walk_mode", settings=settings.effective(), robot_state=robot_state).allowed)
        settings.set("motion.allow_locomotion_mode_change", False, persist=False)
        self.assertFalse(resolver.resolve_skill("run_mode", settings=settings.effective(), robot_state=robot_state).allowed)

    def test_asr_runtime_tracks_accepted_prompt(self) -> None:
        asr = AsrRuntime()
        asr.started()
        asr.final("hello robot", confidence=0.9)
        asr.accepted("hello robot", confidence=0.9)
        snap = asr.snapshot()
        self.assertTrue(snap["listening"])
        self.assertEqual(snap["last_accepted_prompt"], "hello robot")
        self.assertEqual(snap["input_topic"], "/audio_msg")

    def test_visual_observation_change_detection(self) -> None:
        tracker = VisualObservationTracker()
        first = tracker.observe_from_answer(answer="I see a table and a coffee cup.", model="gpt-4o-mini")
        self.assertTrue(first.notable_changes)
        duplicate = tracker.observe_from_answer(answer="I see a table and a coffee cup.", model="gpt-4o-mini")
        self.assertFalse(duplicate.notable_changes)
        changed = tracker.observe_from_answer(answer="I see a table, a coffee cup, and a person.", model="gpt-4o-mini")
        self.assertTrue(changed.notable_changes)

    def test_navigation_snapshot_exposes_topics(self) -> None:
        snap = NavigationAdapter().snapshot().as_dict()
        self.assertIn("/lowstate", snap["topics"])
        self.assertIn("/odommodestate", snap["topics"])

    def test_agent_monitor_has_robot_side_sections(self) -> None:
        agent = G1Agent(
            planner=NoopPlanner(),
            skills=SkillRegistry(skills={}, backend_label="test"),
            state_source=MockRobotStateSource(),
            settings=SettingsManager(path=self.tmp / "settings.json"),
            auto_confirm=True,
        )
        snapshot = agent.monitor_snapshot()
        self.assertIn("navigation", snapshot)
        self.assertIn("asr", snapshot)
        self.assertIn("vision", snapshot)
        self.assertIn("expressive", snapshot)

    def test_offline_registry_contains_walk_run_and_expressive_skills(self) -> None:
        registry = build_offline_registry()
        for name in ("walk_mode", "run_mode", "thinking_motion", "explain_motion", "thanking_motion"):
            self.assertIn(name, registry.skills)


if __name__ == "__main__":
    unittest.main()
