from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shutil
import tempfile
import unittest

from agent.attention import AttentionManager
from agent.checkpoint import CheckpointStore
from agent.cli.router import G1Agent
from agent.memory.manager import MemoryManager
from agent.memory.procedural import ProceduralAdaptation
from agent.models import IntentType, PlannerDecision
from agent.outcomes import SkillOutcome
from agent.semantic_state import SemanticState
from agent.self_model import SelfModelStore
from agent.settings.manager import SettingsManager
from agent.skills import SkillRegistry
from agent.state import MockRobotStateSource


class NoopPlanner:
    def __init__(self) -> None:
        self.inputs = []

    def decide(self, planner_input):
        self.inputs.append(planner_input)
        return PlannerDecision(intent=IntentType.NO_ACTION)


def outcome(skill: str, ok: bool, failure_type: str = "") -> SkillOutcome:
    now = datetime.now(timezone.utc)
    return SkillOutcome(
        skill_id=skill,
        command_accepted=True,
        execution_completed=True,
        goal_reached=ok,
        safe=True,
        failure_type=failure_type or None,
        started_at=now,
        completed_at=now,
        metrics={"duration_s": 1.0},
    )


class SelfModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="g1_self_model_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def make_agent(self, robot_id: str = "g1_test") -> G1Agent:
        settings = SettingsManager(path=self.tmp / f"{robot_id}_settings.json")
        settings.set("self_model.robot_id", robot_id, persist=False)
        return G1Agent(
            planner=NoopPlanner(),
            skills=SkillRegistry(skills={}, backend_label="test"),
            state_source=MockRobotStateSource(battery_pct=80.0, stability="stable"),
            settings=settings,
            memory=MemoryManager(base_dir=self.tmp / robot_id),
            checkpoint_store=CheckpointStore(self.tmp / robot_id / "checkpoint.json"),
            auto_confirm=True,
        )

    def test_first_boot_baseline_has_no_invented_traits(self) -> None:
        agent = self.make_agent()
        summary = agent.self_model.summary()
        self.assertEqual(summary["robot_id"], "g1_test")
        self.assertEqual(summary["notable_body_facts"], [])
        self.assertEqual(summary["skill_confidence"], {})
        self.assertFalse(summary["energy"]["calibrated"])

    def test_skill_outcomes_update_reliability_and_persist(self) -> None:
        store = SelfModelStore(base_dir=self.tmp, robot_id="g1_a")
        store.update_from_skill_outcome(outcome("wave", True), episode_id="ep_1")
        store.update_from_skill_outcome(outcome("wave", False, "goal_not_reached"), episode_id="ep_2")
        record = store.model.skills.records["wave"]
        self.assertEqual(record.attempts, 2)
        self.assertEqual(record.success_rate, 0.5)
        self.assertIn("goal_not_reached", record.common_failure_modes)

        reloaded = SelfModelStore(base_dir=self.tmp, robot_id="g1_a")
        self.assertEqual(reloaded.model.skills.records["wave"].success_rate, 0.5)

    def test_validated_procedure_changes_skill_kwargs_via_self_model(self) -> None:
        agent = self.make_agent()
        agent.settings.set("learning.automatic_level_max", 3, persist=False)
        adaptation = ProceduralAdaptation(
            skill="wave",
            condition={"workspace": "rear"},
            recommended_parameters={"pre_pose": "neutral_before_wave"},
            confidence=0.93,
            derived_from=["ep_1"],
        )
        agent.self_model.apply_procedural_adaptation(adaptation)
        kwargs = agent._learned_skill_kwargs("wave", agent.settings.effective())
        self.assertEqual(kwargs["learned_pre_pose"], "neutral_before_wave")

    def test_energy_calibration_changes_future_estimate(self) -> None:
        store = SelfModelStore(base_dir=self.tmp, robot_id="g1_energy")
        before = store.estimate_energy_cost("step_forward")
        store.calibrate_energy(task="step_forward", observed_cost_pct=3.0, evidence_ref="ep_energy")
        after = store.estimate_energy_cost("step_forward")
        self.assertNotEqual(before, after)
        self.assertTrue(store.model.energy.calibrated)

    def test_self_model_boosts_attention_for_learned_thermal_sensitivity(self) -> None:
        store = SelfModelStore(base_dir=self.tmp, robot_id="g1_attention")
        store.add_body_constraint(
            description="right knee thermal sensitivity predicts instability",
            condition={"joint": "right_knee", "signal": "thermal"},
            confidence=0.87,
            status="active",
        )
        manager = AttentionManager()
        decision = manager.decide(
            event_type="semantic_event",
            semantic_state=SemanticState(thermal="elevated"),
            semantic_changes=["thermal:nominal->elevated"],
            self_model=store,
        )
        self.assertEqual(decision.reason_code, "self_model_relevance")
        self.assertLessEqual(decision.priority, 2)

    def test_contradiction_reduces_confidence_through_invalidation(self) -> None:
        store = SelfModelStore(base_dir=self.tmp, robot_id="g1_inv")
        store.add_body_constraint(description="right arm calibration bias", condition={"part": "right_arm"}, confidence=0.8)
        store.invalidate("right_arm")
        constraint = store.model.body.learned_constraints[0]
        self.assertEqual(constraint.status, "deprecated")
        self.assertLessEqual(constraint.confidence, 0.2)

    def test_reset_learned_returns_self_model_to_baseline(self) -> None:
        agent = self.make_agent()
        agent.self_model.update_from_skill_outcome(outcome("wave", True), episode_id="ep_1")
        self.assertIn("wave", agent.self_model.model.skills.records)
        result = agent.reset("learned")
        self.assertTrue(result.ok)
        self.assertEqual(agent.self_model.model.skills.records, {})

    def test_full_reset_preserves_baseline_static_identity_only(self) -> None:
        agent = self.make_agent()
        agent.self_model.add_body_constraint(description="left hand thin-object weakness", confidence=0.7)
        result = agent.reset("full")
        self.assertTrue(result.ok)
        self.assertEqual(agent.self_model.model.robot_id, "g1_test")
        self.assertEqual(agent.self_model.model.body.learned_constraints, [])

    def test_robot_ids_are_separate_namespaces(self) -> None:
        first = SelfModelStore(base_dir=self.tmp, robot_id="g1_01")
        second = SelfModelStore(base_dir=self.tmp, robot_id="g1_02")
        first.update_from_skill_outcome(outcome("wave", True), episode_id="ep_1")
        second.reload()
        self.assertIn("wave", first.model.skills.records)
        self.assertNotIn("wave", second.model.skills.records)

    def test_self_tools_and_self_cli_are_available(self) -> None:
        agent = self.make_agent()
        ctx = agent._make_tool_context(
            settings=agent.settings.effective(),
            robot_state=agent.state_source.read(),
            event="test",
            profile="diagnostic",
        )
        result = agent.tool_registry.invoke("get_self_summary", {}, ctx)
        self.assertTrue(result.ok)
        self.assertEqual(result.content["robot_id"], "g1_test")
        self.assertIn("G1 SELF MODEL", agent.cmd_self(["summary"]))
        self.assertIn("self", agent.monitor_snapshot(panel="self"))

    def test_planner_receives_compact_self_summary(self) -> None:
        agent = self.make_agent()
        agent.handle_chat("hello")
        runtime_self = agent.planner.inputs[-1].runtime["self"]
        self.assertEqual(runtime_self["robot_id"], "g1_test")
        self.assertIn("skill_confidence", runtime_self)


if __name__ == "__main__":
    unittest.main()
