from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from agent.checkpoint import CheckpointStore
from agent.cli.router import G1Agent
from agent.memory.manager import MemoryManager
from agent.models import IntentType, PlannerDecision
from agent.settings.manager import SettingsManager
from agent.skills import Skill, SkillRegistry, SkillResult
from agent.state import MockRobotStateSource
from agent.tools import ToolAvailability, ToolErrorCode


class RecordingPlanner:
    def __init__(self) -> None:
        self.inputs = []

    def decide(self, planner_input):
        self.inputs.append(planner_input)
        return PlannerDecision(intent=IntentType.CONVERSATION, response_text="ok")


class ToolArchitectureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="g1_tools_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def make_agent(self, *, settings: SettingsManager | None = None, skills: SkillRegistry | None = None) -> G1Agent:
        lowstate = {
            "timestamp": 123.0,
            "joint_count": 2,
            "joint_positions": [1.2, 2.3],
            "joint_velocities": [0.1, 0.2],
            "joint_torques": [0.01, 0.02],
            "imu": {"rpy": (0.0, 0.0, 0.0)},
            "source": "/lowstate",
        }
        return G1Agent(
            planner=RecordingPlanner(),
            skills=skills or SkillRegistry(skills={}, backend_label="test"),
            state_source=MockRobotStateSource(lowstate=lowstate, battery_pct=55.0, stability="stable"),
            settings=settings or SettingsManager(path=self.tmp / "settings.json"),
            memory=MemoryManager(base_dir=self.tmp),
            checkpoint_store=CheckpointStore(self.tmp / "checkpoint.json"),
            auto_confirm=True,
        )

    def context(self, agent: G1Agent, profile: str = "diagnostic"):
        robot_state = agent.state_source.read()
        settings = agent.settings.effective()
        return agent._make_tool_context(settings=settings, robot_state=robot_state, event="test", profile=profile)

    def test_planner_context_is_compact_and_joint_detail_is_tool_lookup(self) -> None:
        agent = self.make_agent()
        agent.memory.autobiography.append("old boot detail")
        agent.handle_chat("what is your arm state?")

        planner_input = agent.planner.inputs[-1]
        self.assertEqual(planner_input.episodic_memory, [])
        self.assertEqual(planner_input.semantic_memory, [])
        self.assertEqual(planner_input.procedural_memory, [])
        self.assertEqual(planner_input.documentary_rag, [])
        self.assertEqual(planner_input.sdk_wrapper_knowledge, [])
        self.assertNotIn("joint_positions", planner_input.robot_state.model_dump_json())

        result = agent.tool_registry.invoke("get_joint_state", {"joint": "right_elbow"}, self.context(agent))
        self.assertTrue(result.ok)
        self.assertEqual(result.content["position"], 1.2)
        self.assertEqual(result.source_type, "robot_state")

    def test_dynamic_exposure_and_operator_only_tools(self) -> None:
        settings = SettingsManager(path=self.tmp / "settings.json")
        settings.set("tools.actions.enabled", False, persist=False)
        wave_skill = Skill(
            name="wave",
            description="test wave",
            source="test",
            handler=lambda **_kwargs: SkillResult(ok=True, message="waved"),
        )
        agent = self.make_agent(settings=settings, skills=SkillRegistry(skills={"wave": wave_skill}, backend_label="test"))
        rows = agent.tool_registry.available_for(self.context(agent), include_unavailable=True)
        by_name = {row["name"]: row for row in rows}

        self.assertEqual(by_name["wave"]["availability"], ToolAvailability.DISABLED_BY_SETTING.value)
        self.assertEqual(by_name["llctl_joint_control"]["availability"], ToolAvailability.OPERATOR_ONLY.value)
        available_names = {row["name"] for row in agent.tool_registry.available_for(self.context(agent))}
        self.assertNotIn("wave", available_names)
        self.assertNotIn("llctl_joint_control", available_names)

    def test_tool_budget_and_zero_budget_schema_hiding(self) -> None:
        settings = SettingsManager(path=self.tmp / "settings.json")
        settings.set("tools.max_calls_per_turn", 1, persist=False)
        agent = self.make_agent(settings=settings)
        ctx = self.context(agent)
        session = agent.tool_registry.session(ctx)

        self.assertTrue(json.loads(session.invoke("get_robot_state"))["ok"])
        second = json.loads(session.invoke("get_battery_state"))
        self.assertFalse(second["ok"])
        self.assertEqual(second["error_code"], ToolErrorCode.BUDGET_EXCEEDED.value)

        settings.set("tools.max_calls_per_turn", 0, persist=False)
        zero_ctx = self.context(agent)
        self.assertEqual(agent.tool_registry.schemas_for(zero_ctx), [])
        self.assertEqual(agent.tool_registry.callables_for(agent.tool_registry.session(zero_ctx)), {})

    def test_source_and_sdk_provenance_are_not_physical_truth(self) -> None:
        agent = self.make_agent()
        result = agent.tool_registry.invoke("inspect_sdk_wrapper", {"query": "headlight", "limit": 1}, self.context(agent))
        self.assertTrue(result.ok)
        self.assertEqual(result.source_type, "implementation_source")
        self.assertEqual(result.provenance["source_type"], "SDK_WRAPPER")
        self.assertFalse(result.provenance["authoritative_physical_truth"])

    def test_learning_tool_keeps_tool_evidence_and_tacit_view_shows_it(self) -> None:
        agent = self.make_agent()
        ctx = self.context(agent)
        result = agent.tool_registry.invoke(
            "propose_empirical_memory",
            {
                "claim": "Wave failures resemble prior right elbow boundary cases.",
                "supporting_episodes": ["ep_1"],
                "confidence": 0.7,
                "tool_evidence": [{"tool": "get_joint_summary", "result_ref": "tool_result_1"}],
            },
            ctx,
        )
        self.assertTrue(result.ok)
        learned = agent.learning.learned.all()[0]
        self.assertEqual(learned.applicable_context["tool_evidence"][0]["tool"], "get_joint_summary")
        lines = "\n".join(agent.tacit_lines(panel="empirical"))
        self.assertIn("Investigated    get_joint_summary", lines)

    def test_monitor_and_tools_cli_show_tool_activity_and_mcp_health(self) -> None:
        agent = self.make_agent()
        agent.tool_registry.invoke("get_robot_state", {}, self.context(agent))

        snapshot = agent.monitor_snapshot(panel="tooling")
        self.assertIn("tooling", snapshot)
        self.assertIn("documentation", snapshot["tooling"]["mcp"])
        events = [event for event in snapshot["events"] if event["category"] == "tool"]
        self.assertTrue(any(event["event"] == "tool_call_completed" for event in events))

        mcp_text = agent.cmd_tooling(["mcp"])
        self.assertIn("documentation", mcp_text)
        self.assertIn("connected=False", mcp_text)

    def test_reset_full_preserves_tool_infrastructure(self) -> None:
        agent = self.make_agent()
        agent.learning.propose_empirical_memory(
            claim="temporary learned tool evidence",
            supporting_episodes=[],
            confidence=0.5,
            applicable_context={"tool_evidence": [{"tool": "get_robot_state"}]},
        )
        self.assertTrue(agent.learning.learned.all())
        result = agent.reset("full")
        self.assertTrue(result.ok)
        self.assertFalse(agent.learning.learned.all())
        self.assertIn("get_robot_state", agent.tool_registry.names())


if __name__ == "__main__":
    unittest.main()
