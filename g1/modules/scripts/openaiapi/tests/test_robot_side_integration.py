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
from agent.llctl import LlctlAdapter
from agent.models import IntentType, PlannerDecision
from agent.navigation import EXPECTED_ROS_TOPICS, NavigationAdapter
from agent.slam import SlamBackend
from agent.checkpoint import CheckpointStore
from agent.memory.manager import MemoryManager
from agent.settings.manager import SettingsManager
from agent.skills import Skill, SkillRegistry, SkillResult, build_offline_registry
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


class FakeLlctlLink:
    def __init__(self) -> None:
        self.connected = False
        self.dev_mode = False
        self.arm_engaged = False
        self.commands = []

    def connect(self) -> None:
        self.connected = True

    def snapshot(self):
        return {
            "connected": self.connected,
            "dev_mode": self.dev_mode,
            "arm_engaged": self.arm_engaged,
            "arm_weight": 1.0 if self.arm_engaged else 0.0,
            "service_row": {"status": 1},
        }

    def joint_modal_defaults(self, joint_id):
        class Spec:
            id = int(joint_id)
            name = "right_shoulder_pitch"
            group = "right_arm"
            q_min = -3.0
            q_max = 3.0

        return {"spec": Spec(), "sensed_q": 0.0, "q": 0.0, "dq": 0.0, "kp": 30.0, "kd": 1.5, "tau": 0.0, "ramp_s": 0.6, "locked": False}

    def toggle_dev_mode(self):
        self.dev_mode = not self.dev_mode
        return True, "dev toggled"

    def set_joint_target(self, joint_id, q, dq, kp, kd, tau, ramp_s):
        self.commands.append(("joint", joint_id, q, dq, kp, kd, tau, ramp_s, self.dev_mode))
        self.arm_engaged = not self.dev_mode
        return True, f"joint {joint_id} set"

    def ee_pose_snapshot(self, side):
        return {"x": 0.2, "y": -0.2 if side == "right" else 0.2, "z": 0.4, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}

    def set_arm_ee_target(self, side, x, y, z, roll, pitch, yaw):
        self.commands.append(("ee", side, x, y, z, roll, pitch, yaw))
        return True, f"{side} ee moved", {"iterations": 1}

    def release_arms(self):
        self.arm_engaged = False
        self.commands.append(("release_arms",))
        return True, "released"


class FakeSlamBackend:
    def start_relocation(self, path=None):
        return "init_pose failed: {'succeed': False, 'errorCode': 509, 'info': 'The current location matching degree is low.'}; hint=relocation scan matching confidence is low"


class FakeSlamState:
    def status(self):
        return {"topics": []}


class FakeSlamRobotNoLidar:
    def start_sensors(self):
        return None

    def sensors_stale(self, max_age=2.0):
        return {"lidar_cloud": True, "lidar_imu": True, "lowstate": False, "slam_odom": True}

    def _service_status(self, name, timeout=1.0):
        return 0

    def get_slam_info(self):
        return "zero-pose status only"


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

    def test_navigation_relocation_failure_is_kept_as_last_error(self) -> None:
        nav = NavigationAdapter(slam_backend=FakeSlamBackend())
        result = nav.action("start_relocation")
        self.assertIn("matching degree is low", result)
        self.assertIn("matching degree is low", nav.snapshot().as_dict()["last_error"])

    def test_slam_backend_reads_nested_error_code(self) -> None:
        self.assertEqual(SlamBackend._error_code({"code": 0, "ok": False, "raw": {"errorCode": 509}}), 509)
        hinted = SlamBackend()._append_slam_data_hint("start_mapping failed", {"code": 0, "ok": False, "raw": {"errorCode": 501}})
        self.assertIn("Lack of lidar or imu data", hinted)

    def test_slam_preflight_reports_root_cause_when_lidar_and_imu_are_stale(self) -> None:
        backend = SlamBackend(robot=FakeSlamRobotNoLidar())
        backend._state = FakeSlamState()
        diagnostics = backend.preflight()
        self.assertIn("neither the required lidar point cloud nor lidar IMU", diagnostics["root_cause"])
        self.assertIn("ros2 topic hz /utlidar/cloud_livox_mid360", diagnostics["next_checks"])

    def test_german_reach_hand_forward_routes_to_reach_forward(self) -> None:
        calls = []
        agent = G1Agent(
            planner=NoopPlanner(),
            skills=SkillRegistry(
                skills={
                    "reach_forward": Skill(
                        "reach_forward",
                        "test reach",
                        lambda **kwargs: calls.append(kwargs) or SkillResult(ok=True, message="reached"),
                        "test",
                    )
                },
                backend_label="test",
            ),
            state_source=MockRobotStateSource(),
            settings=SettingsManager(path=self.tmp / "settings.json"),
            memory=MemoryManager(base_dir=self.tmp / "memory_reach"),
            checkpoint_store=CheckpointStore(self.tmp / "checkpoint_reach.json"),
            auto_confirm=True,
        )
        agent.settings.set("interface.command_language", "de", persist=False)
        outcome = agent.handle_cli_text("kannst du die rechte Hand nach vorne bringen ?")
        self.assertEqual(outcome.decision.intent, IntentType.EXECUTE_TASK)
        self.assertEqual(outcome.skill_outcomes[0][0], "reach_forward")

    def test_learned_question_reports_grounded_counts(self) -> None:
        agent = G1Agent(
            planner=NoopPlanner(),
            skills=SkillRegistry(skills={}, backend_label="test"),
            state_source=MockRobotStateSource(),
            settings=SettingsManager(path=self.tmp / "settings.json"),
            memory=MemoryManager(base_dir=self.tmp / "memory_learned"),
            checkpoint_store=CheckpointStore(self.tmp / "checkpoint_learned.json"),
            auto_confirm=True,
        )
        agent.settings.set("interface.command_language", "de", persist=False)
        outcome = agent.handle_cli_text("Was hast du gelernt ?")
        self.assertIn("Episoden:", outcome.grounded_response or "")
        self.assertIn("Gelernte semantische Claims:", outcome.grounded_response or "")

    def test_llctl_adapter_executes_dashboard_joint_ee_and_release(self) -> None:
        settings = SettingsManager(path=self.tmp / "settings.json").effective()
        adapter = LlctlAdapter()
        adapter._dashboard_module = object()
        adapter._robot_link = FakeLlctlLink()
        self.assertEqual(adapter.enable_session(settings), "llctl session enabled")
        joint_result = adapter.command_joint(settings, joint="22", q=0.1, dq=0.0, kp=30.0, kd=1.5, tau=0.0, ramp_s=0.6, backend="arm_sdk")
        self.assertIn("ok:", joint_result)
        ee_result = adapter.command_ee_delta(settings, side="right", dx=0.01, dz=0.02)
        self.assertIn("ok:", ee_result)
        release = adapter.release_arms(settings)
        self.assertIn("ok:", release)
        self.assertEqual(adapter._robot_link.commands[0][0], "joint")
        self.assertEqual(adapter._robot_link.commands[1][0], "ee")
        self.assertEqual(adapter._robot_link.commands[2][0], "release_arms")

    def test_llctl_cli_parser_routes_joint_command(self) -> None:
        agent = G1Agent(
            planner=NoopPlanner(),
            skills=SkillRegistry(skills={}, backend_label="test"),
            state_source=MockRobotStateSource(),
            settings=SettingsManager(path=self.tmp / "settings.json"),
            auto_confirm=True,
        )
        agent.llctl._dashboard_module = object()
        agent.llctl._robot_link = FakeLlctlLink()
        self.assertIn("enabled", agent.llctl_command(["enable"]))
        result = agent.llctl_command(["joint", "22", "q", "0.1", "dq", "0", "kp", "30", "kq", "1.5", "ramp", "0.6"])
        self.assertIn("ok:", result)
        ee_result = agent.llctl_command(["ee", "right", "0.2", "0", "0.3", "0", "0", "0"])
        self.assertIn("ok:", ee_result)
        self.assertEqual(agent.llctl._robot_link.commands[-1], ("ee", "right", 0.2, 0.0, 0.3, 0.0, 0.0, 0.0))

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
