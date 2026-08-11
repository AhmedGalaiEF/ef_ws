"""Standalone REPL entry point (spec section 31's example CLI session).

Run with ``python -m agent.cli`` from ``g1/modules/scripts/openaiapi`` (no
``--robot``, no OpenAI key needed -- offline mock backend + MockPlanner),
or ``python -m agent.cli --robot --openai`` on the real deployment target.
"""
from __future__ import annotations

import argparse
import atexit
import curses
import os
import sys
import threading
import time
from pathlib import Path


def _bootstrap_scripts_path() -> None:
    here = Path(__file__).resolve()
    package_root = next((parent for parent in here.parents if (parent / "agent").is_dir()), here.parents[2])
    scripts_dir = next((parent for parent in here.parents if (parent / "llm_client").exists()), here.parents[2])
    modules_dir = scripts_dir.parent if (scripts_dir.parent / "sdk_client.py").exists() else None
    # ``agent`` is nested in openaiapi, while shared transports remain one
    # directory above it.  Put the package root first so this entrypoint
    # cannot accidentally resolve a different package with the same name.
    for path in (modules_dir, scripts_dir, package_root):
        if path is not None and str(path) not in sys.path:
            sys.path.insert(0, str(path))


_bootstrap_scripts_path()

from agent.cli.router import G1Agent, TurnOutcome  # noqa: E402
from agent.capabilities import CapabilityResolver  # noqa: E402
from agent.knowledge.sdk_wrapper_knowledge import SdkWrapperKnowledge  # noqa: E402
from agent.memory.manager import MemoryManager  # noqa: E402
from agent.planner import MockPlanner, OpenAIPlanner, Planner, PlannerError  # noqa: E402
from agent.settings.manager import SettingsManager  # noqa: E402
from agent.skills import SkillUnavailable, build_live_registry, build_offline_registry  # noqa: E402
from agent.state import MockRobotStateSource, SdkClientRobotStateSource  # noqa: E402


PRINT_LOCK = threading.RLock()
SETTING_CHOICES = {
    "asr.confidence_threshold": [0.5, 0.65, 0.72, 0.8, 0.9],
    "active_learning.cooldown_s": [0.0, 60.0, 120.0, 300.0, 600.0],
    "active_learning.minimum_confidence_gap": [0.0, 0.1, 0.2, 0.35, 0.5],
    "active_learning.maximum_pending_questions": [0, 1, 2, 3],
    "active_learning.duplicate_suppression_s": [300.0, 1200.0, 3600.0, 7200.0],
    "active_learning.unanswered_timeout_s": [60.0, 180.0, 300.0, 600.0],
    "asr.silence_timeout_ms": [500, 800, 1200, 1600, 2500],
    "announcements.tts_language": ["", "en", "de", "fr", "es", "ar"],
    "announcements.tts_speaker": [-1, 0, 1, 2, 3],
    "cognition.periodic_interval_s": [5.0, 10.0, 30.0, 60.0, 300.0],
    "learning.automatic_level_max": [0, 1, 2, 3],
    "learning.minimum_support_for_candidate": [2, 3, 5, 10],
    "learning.minimum_support_for_procedure": [3, 5, 10, 20],
    "memory.hot_episode_limit": [100, 500, 1000, 5000, 10000],
    "memory.hot_disk_limit_mb": [100, 250, 500, 1000],
    "memory.consolidation_interval_min": [5, 15, 30, 60, 180],
    "monitor.refresh_hz": [1.0, 2.0, 3.0, 5.0],
    "monitor.event_buffer_size": [100, 500, 1000, 5000],
    "expressive_motion.thinking.cooldown_s": [0.0, 5.0, 8.0, 15.0],
    "expressive_motion.explain.cooldown_s": [0.0, 5.0, 10.0],
    "expressive_motion.thanking.cooldown_s": [0.0, 5.0, 10.0],
    "expressive_motion.explain_minimum_speech_chars": [40, 80, 120, 200],
    "headlight.thinking.color.r": [0, 55, 80, 128, 255],
    "headlight.thinking.color.g": [0, 20, 40, 80, 255],
    "headlight.thinking.color.b": [0, 75, 128, 180, 255],
    "headlight.thinking.intensity": [25, 50, 75, 100],
    "disk.warning_free_pct": [10.0, 15.0, 20.0, 30.0],
    "disk.critical_free_pct": [5.0, 10.0, 15.0],
    "reset.require_confirmation": [True, False],
    "reset.create_backup": [True, False],
    "reset.full_preserve_settings": [True, False],
    "reset.preserve_audit_log": [True, False],
    "tacit.ui_enabled": [True, False],
    "tacit.show_confidence": [True, False],
    "tacit.show_evidence_counts": [True, False],
    "tacit.show_performance_metrics": [True, False],
    "tools.enabled": [True, False],
    "tools.max_calls_per_turn": [0, 4, 8, 12, 20],
    "tools.max_parallel_read_calls": [1, 2, 4, 8],
    "tools.observation.enabled": [True, False],
    "tools.memory.enabled": [True, False],
    "tools.knowledge.enabled": [True, False],
    "tools.diagnostics.enabled": [True, False],
    "tools.actions.enabled": [True, False],
    "mcp.enabled": [True, False],
    "mcp.reconnect.enabled": [True, False],
    "mcp.reconnect.initial_backoff_s": [0.5, 1.0, 2.0, 5.0],
    "mcp.reconnect.max_backoff_s": [10.0, 30.0, 60.0],
    "mcp.servers.memory.enabled": [True, False],
    "mcp.servers.documentation.enabled": [True, False],
    "mcp.servers.diagnostics.enabled": [True, False],
    "self_model.enabled": [True, False],
    "self_model.update_from_skill_outcomes": [True, False],
    "self_model.reset_learned_components_on_reset_learned": [True, False],
    "interface.command_language": ["en", "de", "both"],
    "interface.reply_language": ["auto", "en", "de"],
    "response.max_chars": [300, 500, 700, 1000, 1500],
    "response.memory_max_entries": [1, 2, 3, 5, 10],
    "vision.openai_model": ["gpt-4o-mini", "gpt-4o"],
    "llctl.session_timeout_s": [15.0, 30.0, 60.0, 120.0],
}
SETTING_LABELS = {
    "announcements.tts_language": "announcements.tts_language (voice)",
    "announcements.tts_voice_model": "announcements.tts_voice_model",
    "announcements.tts_speaker": "announcements.tts_speaker",
    "active_learning.enabled": "active_learning.enabled",
    "active_learning.allow_autonomous_questions": "active_learning.allow_autonomous_questions",
    "active_learning.cooldown_s": "active_learning.cooldown_s",
    "active_learning.minimum_confidence_gap": "active_learning.minimum_confidence_gap",
    "active_learning.maximum_pending_questions": "active_learning.maximum_pending_questions",
    "active_learning.allow_during_active_scenario": "active_learning.allow_during_active_scenario",
    "active_learning.allow_during_idle": "active_learning.allow_during_idle",
    "active_learning.allow_during_task_execution": "active_learning.allow_during_task_execution",
    "active_learning.duplicate_suppression_s": "active_learning.duplicate_suppression_s",
    "active_learning.unanswered_timeout_s": "active_learning.unanswered_timeout_s",
    "active_learning.store_rejected_questions": "active_learning.store_rejected_questions",
    "asr.enabled": "asr.enabled",
    "asr.confidence_threshold": "asr.confidence_threshold",
    "asr.partial_display": "asr.partial_display",
    "asr.silence_timeout_ms": "asr.silence_timeout_ms",
    "asr.wake_word_enabled": "asr.wake_word_enabled",
    "interface.command_language": "interface.command_language (commands)",
    "interface.reply_language": "interface.reply_language (answers)",
    "response.max_chars": "response.max_chars",
    "response.memory_max_entries": "response.memory_max_entries",
    "cognition.periodic_enabled": "cognition.periodic_enabled",
    "cognition.periodic_interval_s": "cognition.periodic_interval_s",
    "cognition.attention_enabled": "cognition.attention_enabled",
    "cognition.background_enabled": "cognition.background_enabled",
    "learning.enabled": "learning.enabled",
    "learning.empirical_memory_enabled": "learning.empirical_memory_enabled",
    "learning.procedural_learning_enabled": "learning.procedural_learning_enabled",
    "learning.automatic_level_max": "learning.automatic_level_max",
    "learning.minimum_support_for_candidate": "learning.minimum_support_for_candidate",
    "learning.minimum_support_for_procedure": "learning.minimum_support_for_procedure",
    "memory.working_event_max": "memory.working_event_max",
    "memory.hot_episode_limit": "memory.hot_episode_limit",
    "memory.hot_disk_limit_mb": "memory.hot_disk_limit_mb",
    "memory.cold_archive_limit_gb": "memory.cold_archive_limit_gb",
    "memory.routine_retention_days": "memory.routine_retention_days",
    "memory.significant_retention_days": "memory.significant_retention_days",
    "memory.consolidation_interval_min": "memory.consolidation_interval_min",
    "monitor.refresh_hz": "monitor.refresh_hz",
    "monitor.event_buffer_size": "monitor.event_buffer_size",
    "disk.warning_free_pct": "disk.warning_free_pct",
    "disk.critical_free_pct": "disk.critical_free_pct",
    "reset.require_confirmation": "reset.require_confirmation",
    "reset.create_backup": "reset.create_backup",
    "reset.full_preserve_settings": "reset.full_preserve_settings",
    "reset.preserve_audit_log": "reset.preserve_audit_log",
    "tacit.ui_enabled": "tacit.ui_enabled",
    "tacit.show_confidence": "tacit.show_confidence",
    "tacit.show_evidence_counts": "tacit.show_evidence_counts",
    "tacit.show_performance_metrics": "tacit.show_performance_metrics",
    "tools.enabled": "tools.enabled",
    "tools.max_calls_per_turn": "tools.max_calls_per_turn",
    "tools.max_parallel_read_calls": "tools.max_parallel_read_calls",
    "tools.observation.enabled": "tools.observation.enabled",
    "tools.memory.enabled": "tools.memory.enabled",
    "tools.knowledge.enabled": "tools.knowledge.enabled",
    "tools.diagnostics.enabled": "tools.diagnostics.enabled",
    "tools.actions.enabled": "tools.actions.enabled",
    "mcp.enabled": "mcp.enabled",
    "mcp.reconnect.enabled": "mcp.reconnect.enabled",
    "mcp.reconnect.initial_backoff_s": "mcp.reconnect.initial_backoff_s",
    "mcp.reconnect.max_backoff_s": "mcp.reconnect.max_backoff_s",
    "mcp.servers.memory.enabled": "mcp.servers.memory.enabled",
    "mcp.servers.documentation.enabled": "mcp.servers.documentation.enabled",
    "mcp.servers.diagnostics.enabled": "mcp.servers.diagnostics.enabled",
    "self_model.enabled": "self_model.enabled",
    "self_model.robot_id": "self_model.robot_id",
    "self_model.update_from_skill_outcomes": "self_model.update_from_skill_outcomes",
    "self_model.reset_learned_components_on_reset_learned": "self_model.reset_learned_components_on_reset_learned",
    "expressive_motion.enabled": "expressive_motion.enabled",
    "expressive_motion.thinking.enabled": "expressive_motion.thinking.enabled",
    "expressive_motion.thinking.cooldown_s": "expressive_motion.thinking.cooldown_s",
    "expressive_motion.explain.enabled": "expressive_motion.explain.enabled",
    "expressive_motion.explain.cooldown_s": "expressive_motion.explain.cooldown_s",
    "expressive_motion.thanking.enabled": "expressive_motion.thanking.enabled",
    "expressive_motion.thanking.cooldown_s": "expressive_motion.thanking.cooldown_s",
    "expressive_motion.explain_minimum_speech_chars": "expressive_motion.explain_minimum_speech_chars",
    "expressive_motion.motion_directory": "expressive_motion.motion_directory",
    "headlight.cognitive_indicators_enabled": "headlight.cognitive_indicators_enabled",
    "headlight.thinking.enabled": "headlight.thinking.enabled",
    "headlight.thinking.color.r": "headlight.thinking.color.r",
    "headlight.thinking.color.g": "headlight.thinking.color.g",
    "headlight.thinking.color.b": "headlight.thinking.color.b",
    "headlight.thinking.intensity": "headlight.thinking.intensity",
    "headlight.restore_previous_state": "headlight.restore_previous_state",
    "llctl.enabled": "llctl.enabled",
    "llctl.allow_joint_control": "llctl.allow_joint_control",
    "llctl.allow_ik_control": "llctl.allow_ik_control",
    "llctl.require_explicit_enable_each_session": "llctl.require_explicit_enable_each_session",
    "llctl.session_timeout_s": "llctl.session_timeout_s",
    "motion.allow_locomotion_mode_change": "motion.allow_locomotion_mode_change",
    "vision.rgbd_enabled": "vision.rgbd_enabled (camera)",
    "vision.rgbd_host": "vision.rgbd_host",
    "vision.rgbd_port": "vision.rgbd_port",
    "vision.rgbd_topic": "vision.rgbd_topic",
    "vision.rgbd_timeout_s": "vision.rgbd_timeout_s",
    "vision.openai_model": "vision.openai_model",
}
GERMAN_SETTINGS_ARGS = {
    "anzeigen": "show",
    "zeige": "show",
    "lesen": "get",
    "setzen": "set",
    "faehigkeit": "skill",
    "fähigkeit": "skill",
    "faehigkeiten": "skills",
    "fähigkeiten": "skills",
}


class Color:
    enabled = sys.stdout.isatty() and "NO_COLOR" not in os.environ
    reset = "\033[0m" if enabled else ""
    dim = "\033[2m" if enabled else ""
    cyan = "\033[36m" if enabled else ""
    green = "\033[32m" if enabled else ""
    yellow = "\033[33m" if enabled else ""
    red = "\033[31m" if enabled else ""
    magenta = "\033[35m" if enabled else ""
    blue = "\033[34m" if enabled else ""


def _style(text: str, color: str) -> str:
    return f"{color}{text}{Color.reset}" if Color.enabled else text


def _print(text: str = "") -> None:
    with PRINT_LOCK:
        print(text, flush=True)


def _setup_readline_history() -> None:
    """Enable up-arrow command history when readline is available."""
    try:
        import readline
    except Exception:
        return

    history_file = Path(os.environ.get("G1_AGENT_HISTORY", "~/.g1_agent_history")).expanduser()
    try:
        readline.read_history_file(history_file)
    except FileNotFoundError:
        pass
    except Exception as exc:
        print(f"[warn] could not read history file {history_file}: {exc}")

    try:
        readline.set_history_length(1000)
    except Exception:
        pass

    def _write_history() -> None:
        try:
            history_file.parent.mkdir(parents=True, exist_ok=True)
            readline.write_history_file(history_file)
        except Exception as exc:
            print(f"[warn] could not write history file {history_file}: {exc}")

    atexit.register(_write_history)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="G1 persistent cognitive agent CLI (Phase 1).")
    parser.add_argument("--robot", action="store_true", help="connect to a live robot (requires the Unitree SDK2 stack + DDS)")
    parser.add_argument("--iface", default="eth0")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--openai", action="store_true", help="use OpenAIPlanner (requires OPENAI_API_KEY)")
    parser.add_argument("--model", default=OpenAIPlanner.DEFAULT_MODEL)
    parser.add_argument(
        "--knowledge-file", action="append", default=[], help="documentary RAG knowledge file(s); repeatable"
    )
    parser.add_argument(
        "--auto-confirm", action="store_true", help="skip y/N prompts for skills set to 'confirm' (non-interactive runs)"
    )
    parser.add_argument(
        "--tick-interval",
        type=float,
        default=30.0,
        help="seconds between periodic cognitive ticks while the REPL is idle",
    )
    parser.add_argument(
        "--no-periodic-ticks",
        action="store_true",
        help="disable background periodic cognition",
    )
    return parser


def build_agent(args: argparse.Namespace) -> G1Agent:
    settings = SettingsManager()
    memory = MemoryManager()
    sdk_knowledge = SdkWrapperKnowledge()

    document_rag = None
    knowledge_files = list(args.knowledge_file or [])
    default_knowledge_file = Path(__file__).resolve().parents[1] / "knowledge" / "default_sdk_knowledge.md"
    if default_knowledge_file.exists():
        knowledge_files.insert(0, str(default_knowledge_file))
    if knowledge_files:
        try:
            from agent.knowledge.document_rag import DocumentRAG

            document_rag = DocumentRAG(knowledge_files)
        except Exception as exc:
            print(f"[warn] documentary RAG unavailable: {exc}")

    robot = None
    resolver = CapabilityResolver()
    if args.robot:
        try:
            from agent.skills import _bootstrap_repo_paths

            _bootstrap_repo_paths()
            from sdk_client import Robot  # requires the Unitree SDK2 stack + a live DDS connection

            robot = Robot(iface=args.iface, domain_id=args.domain_id)
        except Exception as exc:
            print(f"[warn] could not connect to a live robot ({exc}); falling back to offline mode.")

    if robot is not None:
        try:
            skills = build_live_registry(robot=robot, iface=args.iface, domain_id=args.domain_id)
            state_source: object = SdkClientRobotStateSource(robot)
            resolver = CapabilityResolver(
                arm_sdk_available=lambda: hasattr(robot, "extend_arm_forward"),
                low_cmd_available=lambda: False,
            )
        except SkillUnavailable as exc:
            print(f"[warn] live skill registry unavailable ({exc}); falling back to offline mode.")
            skills = build_offline_registry()
            state_source = MockRobotStateSource()
            resolver = CapabilityResolver()
    else:
        skills = build_offline_registry()
        state_source = MockRobotStateSource()

    planner: Planner
    if args.openai:
        try:
            planner = OpenAIPlanner(model=args.model)
        except PlannerError as exc:
            print(f"[warn] OpenAIPlanner unavailable ({exc}); falling back to MockPlanner.")
            planner = MockPlanner()
    else:
        planner = MockPlanner()

    agent = G1Agent(
        planner=planner,
        skills=skills,
        state_source=state_source,  # type: ignore[arg-type]
        settings=settings,
        memory=memory,
        sdk_knowledge=sdk_knowledge,
        document_rag=document_rag,
        resolver=resolver,
        auto_confirm=args.auto_confirm,
        robot=robot,
    )
    if robot is not None:
        try:
            from agent.slam import SlamBackend

            agent.navigation.slam_backend = SlamBackend(iface=args.iface, domain_id=args.domain_id, robot=robot)
        except Exception as exc:
            agent.monitor_bus.emit("navigation", "navigation_adapter_unavailable", str(exc))
    return agent


def _print_turn(outcome: TurnOutcome) -> None:
    with PRINT_LOCK:
        decision = _style("decision", Color.magenta)
        print(f"  [{decision}] intent={outcome.decision.intent.value} target={outcome.decision.target}", flush=True)
        if outcome.grounded_response:
            print(f"{_style('agent>', Color.green)} {outcome.grounded_response}", flush=True)
        if outcome.learning_question is not None:
            print("", flush=True)
            print(_style("G1 wants to learn something:", Color.yellow), flush=True)
            print(f"{outcome.learning_question.proposal.question}", flush=True)
            print(_style("(answer normally, use /skip to decline, or run any /command first)", Color.dim), flush=True)
        for skill_name, skill_outcome in outcome.skill_outcomes:
            message = skill_outcome.result.message if skill_outcome.result else skill_outcome.policy.reason
            status_color = Color.green if skill_outcome.status == "executed" else Color.yellow
            print(
                f"  [{_style('skill', Color.blue)}] {skill_name} -> "
                f"{_style(skill_outcome.status, status_color)}: {message}",
                flush=True,
            )


def _dispatch(agent: G1Agent, line: str) -> None:
    if line.startswith("/chat "):
        _print_turn(agent.handle_chat(line[len("/chat "):]))
    elif line.startswith("/audio_msg "):
        outcome = agent.handle_audio_msg(line[len("/audio_msg "):])
        if outcome is None:
            _print(_style("(ASR disabled: audio.asr_enabled=false -- no conversational event generated)", Color.dim))
        else:
            _print_turn(outcome)
    elif line.startswith("/vision "):
        _print_turn(agent.handle_vision_question(line[len("/vision "):]))
    elif line.startswith("/sehen "):
        _print_turn(agent.handle_vision_question(line[len("/sehen "):]))
    elif line in ("/settings-ui", "/setting-ui", "/settings ui", "/einstellungen-ui"):
        _run_settings_ui(agent)
    elif line.startswith("/monitor"):
        parts = line.split()
        panel = parts[1] if len(parts) > 1 else "overview"
        _run_monitor_ui(agent, panel=panel)
    elif line.startswith("/navigation"):
        parts = line.split()
        panel = parts[1] if len(parts) > 1 else "status"
        if panel in {"start", "start_mapping"}:
            _print(agent.navigation_action("start_mapping"))
        elif panel in {"stop", "stop_slam"}:
            _print(agent.navigation_action("stop_slam"))
        elif panel in {"relocate", "localize", "start_relocation"}:
            _print(agent.navigation_action("start_relocation"))
        elif panel == "save_map":
            _print(agent.navigation_action("save_map"))
        elif panel == "preflight":
            _print(agent.navigation_action("preflight"))
        else:
            _run_navigation_ui(agent, panel=panel)
    elif line.startswith("/asr"):
        _run_asr_ui(agent)
    elif line.startswith("/llctl"):
        parts = line.split()
        if len(parts) > 1 and parts[1] == "enable":
            _print(agent.llctl_enable())
        elif len(parts) > 1 and parts[1] in {"disable", "off"}:
            _print(agent.llctl_disable())
        else:
            _run_llctl_ui(agent)
    elif line.startswith("/reset"):
        _handle_reset_command(agent, line.split()[1:])
    elif line.startswith("/tacit"):
        _handle_tacit_command(agent, line.split()[1:])
    elif line.startswith("/self"):
        _print(agent.cmd_self(line.split()[1:]))
    elif line in ("/skip", "/learning-skip"):
        record = agent.active_learning.skip(reason="operator_skip")
        _print("learning question skipped" if record is not None else "no learning question is pending")
    elif line.startswith("/settings") or line.startswith("/einstellungen"):
        args = line.split()[1:]
        if line.startswith("/einstellungen") and not args:
            args = ["show"]
        if line.startswith("/einstellungen") and args:
            args[0] = GERMAN_SETTINGS_ARGS.get(args[0].lower(), args[0])
        _print(agent.cmd_settings(args))
    elif line == "/status":
        _print(agent.cmd_status())
    elif line in ("/faults", "/fehler"):
        _print(agent.cmd_faults())
    elif line.startswith("/memory") or line.startswith("/speicher"):
        _print(agent.cmd_memory(line.split()[1:]))
    elif line.startswith("/tools") or line.startswith("/werkzeuge"):
        args = line.split()[1:]
        _print(agent.cmd_tooling(args) if args else agent.cmd_tools())
    elif line in ("/help", "/hilfe"):
        _print(agent.cmd_help())
    elif line.startswith("/"):
        _print(_style(f"(unknown command: {line.split()[0]} -- try /help)", Color.yellow))
    else:
        # Bare text defaults to /chat, mirroring llm_client/cli.py's REPL.
        _print_turn(agent.handle_cli_text(line))


def _setting_rows(agent: G1Agent) -> list[tuple[str, object]]:
    settings = agent.settings.effective()
    rows = sorted(
        (key, getattr(value, "value", value))
        for key, value in settings.as_flat_dict().items()
    )
    for skill_name in agent.skills.names():
        rows.append((f"skill.{skill_name}", settings.get_skill_mode(skill_name).value))
    return rows


def _format_ui_value(key: str, value: object) -> str:
    if key in ("announcements.tts_language", "announcements.tts_voice_model") and str(value) == "":
        return "default"
    if key == "announcements.tts_speaker" and int(value) < 0:
        return "default"
    if isinstance(value, dict):
        cleaned = {
            item_key: getattr(item_value, "value", item_value)
            for item_key, item_value in value.items()
        }
        return str(cleaned)
    return str(getattr(value, "value", value))


def _format_ui_choices(key: str) -> str:
    choices = SETTING_CHOICES.get(key)
    if not choices:
        return ""
    labels = [str("default" if choice == "" or choice == -1 else choice) for choice in choices]
    return "  [" + " | ".join(labels) + "]"


def _change_setting(agent: G1Agent, key: str, value: object, delta: int) -> str:
    if key.startswith("skill."):
        modes = ["auto", "confirm", "disabled"]
        current = str(value)
        next_value = modes[(modes.index(current) + delta) % len(modes)] if current in modes else "confirm"
        agent.settings.set_skill_mode(key[len("skill."):], next_value)
        return next_value
    if isinstance(value, bool):
        next_value = not value
        agent.settings.set(key, next_value)
        return str(next_value)
    if key in SETTING_CHOICES:
        choices = SETTING_CHOICES[key]
        current = getattr(value, "value", str(value))
        next_value = choices[(choices.index(current) + delta) % len(choices)] if current in choices else choices[0]
        agent.settings.set(key, next_value)
        return _format_ui_value(key, next_value)
    return str(value)


def _edit_setting_value(stdscr: object, agent: G1Agent, key: str, old_value: object) -> str:
    curses.echo()
    max_y, max_x = stdscr.getmaxyx()
    prompt = f"New value for {key} [{_format_ui_value(key, old_value)}]: "
    stdscr.move(max_y - 2, 0)
    stdscr.clrtoeol()
    stdscr.addnstr(max_y - 2, 0, prompt, max_x - 1)
    raw = stdscr.getstr(max_y - 2, min(len(prompt), max_x - 1), 200).decode("utf-8").strip()
    curses.noecho()
    if not raw:
        return "unchanged"
    value = G1Agent._coerce_setting_value(raw)
    try:
        if key.startswith("skill."):
            agent.settings.set_skill_mode(key[len("skill."):], str(value))
        else:
            agent.settings.set(key, value)
    except Exception as exc:
        return f"error: {exc}"
    return f"set {key} = {value}"


def _settings_ui(stdscr: object, agent: G1Agent) -> None:
    curses.curs_set(0)
    stdscr.keypad(True)
    selected = 0
    top = 0
    status = "arrows navigate/change, space toggles, Enter edits, q exits"
    while True:
        rows = _setting_rows(agent)
        max_y, max_x = stdscr.getmaxyx()
        visible = max(1, max_y - 4)
        selected = max(0, min(selected, len(rows) - 1))
        if selected < top:
            top = selected
        elif selected >= top + visible:
            top = selected - visible + 1

        stdscr.erase()
        stdscr.addnstr(0, 0, "Settings", max_x - 1, curses.A_BOLD)
        stdscr.addnstr(1, 0, status, max_x - 1)
        for idx, (key, value) in enumerate(rows[top:top + visible], start=top):
            marker = "> " if idx == selected else "  "
            value_text = _format_ui_value(key, value)
            choices = _format_ui_choices(key) if idx == selected else ""
            label = SETTING_LABELS.get(key, key)
            line = f"{marker}{label:<42} {value_text}{choices}"
            attr = curses.A_REVERSE if idx == selected else curses.A_NORMAL
            stdscr.addnstr(3 + idx - top, 0, line, max_x - 1, attr)
        stdscr.refresh()

        ch = stdscr.getch()
        if ch in (ord("q"), 27):
            return
        if ch == curses.KEY_UP:
            selected -= 1
        elif ch == curses.KEY_DOWN:
            selected += 1
        elif ch in (curses.KEY_LEFT, curses.KEY_RIGHT, ord(" ")):
            key, value = rows[selected]
            delta = -1 if ch == curses.KEY_LEFT else 1
            status = f"{key} = {_change_setting(agent, key, value, delta)}"
        elif ch in (10, 13):
            key, value = rows[selected]
            status = _edit_setting_value(stdscr, agent, key, value)


def _run_settings_ui(agent: G1Agent) -> None:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        _print("settings UI requires an interactive terminal")
        return
    with PRINT_LOCK:
        curses.wrapper(_settings_ui, agent)


MONITOR_PANELS = [
    "overview",
    "state",
    "events",
    "memory",
    "learning",
    "tools",
    "tooling",
    "self",
    "vision",
    "navigation",
    "asr",
    "expressive",
    "activity",
    "active_learning",
]


def _add_line(stdscr: object, y: int, x: int, text: str, width: int, attr: int = 0) -> int:
    max_y, _ = stdscr.getmaxyx()
    if y >= max_y:
        return y
    stdscr.addnstr(y, x, text, max(0, width - x - 1), attr)
    return y + 1


def _monitor_lines(snapshot: dict, panel: str) -> list[str]:
    semantic = snapshot.get("semantic_state") or {}
    memory = snapshot.get("memory") or {}
    learning = snapshot.get("learning") or {}
    disk = snapshot.get("disk") or {}
    tools = snapshot.get("tools") or {}
    events = snapshot.get("events") or []
    if panel == "state":
        return [
            "AGENT STATE",
            f"Posture              {semantic.get('posture')}",
            f"Balance              {semantic.get('balance')}",
            f"Battery              {semantic.get('battery')}",
            f"Interaction          {semantic.get('interaction')}",
            f"Current task         {semantic.get('task')}",
            f"Arm control          {semantic.get('arm_state')}",
            f"Thermal              {semantic.get('thermal')}",
            f"Active faults        {', '.join(semantic.get('active_faults') or []) or 'none'}",
        ]
    if panel == "memory":
        return [
            "MEMORY",
            f"Working events       {memory.get('working_events')} / {snapshot.get('events', []) and len(snapshot.get('events', []))}",
            f"Hot episodes         {memory.get('hot_episodes')} / {memory.get('hot_episode_limit')}",
            f"Semantic memories    {memory.get('semantic_memories')}",
            f"Procedural memories  {memory.get('procedural_memories')}",
            f"Learned claims       {memory.get('learned_claims')}",
            f"Pinned memories      {memory.get('pinned')}",
            f"Memory dir           {memory.get('base_dir')}",
            f"Disk                 {disk.get('status')} free={disk.get('free_pct', 0):.1f}% memory={disk.get('memory_mb', 0):.1f}MB",
        ]
    if panel == "learning":
        lines = [
            "LEARNING",
            f"Candidate claims     {learning.get('candidate_claims')}",
            f"Active claims        {learning.get('active_claims')}",
            f"Procedural rules     {learning.get('procedural_rules')}",
            "",
            "Latest:",
        ]
        for claim in learning.get("latest") or []:
            lines.append(
                f"- {claim.get('id')} status={claim.get('status')} confidence={claim.get('confidence', 0):.2f}"
            )
            lines.append(f"  {claim.get('claim')}")
        return lines
    if panel == "tools":
        return ["TOOLS / CAPABILITIES"] + [f"{key:<22} {value}" for key, value in sorted(tools.items())]
    if panel == "tooling":
        tooling = snapshot.get("tooling") or {}
        lines = [
            "TOOLS / MCP",
            f"Available tools      {tooling.get('available_tools')}",
            f"Read-only            {tooling.get('read_only')}",
            f"Action               {tooling.get('action')}",
            f"Max calls/turn       {tooling.get('max_calls_per_turn')}",
            "",
            "MCP",
        ]
        for name, health in sorted((tooling.get("mcp") or {}).items()):
            lines.append(
                f"{name:<16} connected={health.get('connected')} tools={health.get('available_tools')} "
                f"last_error={health.get('last_error') or 'none'}"
            )
        lines.extend(["", "RECENT TOOL EVENTS"])
        for event in events[-30:]:
            if event.get("category") == "tool":
                lines.append(
                    f"{time.strftime('%H:%M:%S', time.localtime(event.get('timestamp', 0)))}  "
                    f"{event.get('event')}  {event.get('summary')}"
                )
        return lines
    if panel == "self":
        self_snapshot = snapshot.get("self") or {}
        lines = [
            "SELF",
            f"Robot ID             {self_snapshot.get('robot_id')}",
            f"Self-model version   {self_snapshot.get('version')}",
            f"Condition            {self_snapshot.get('current_high_level_condition')}",
            f"Overall confidence   {self_snapshot.get('overall_confidence')}",
            "",
            "Learned body notes:",
        ]
        notes = self_snapshot.get("notable_body_facts") or []
        lines.extend(f"- {note}" for note in notes[:6])
        if not notes:
            lines.append("(none)")
        lines.extend(["", "Current commitments:"])
        commitments = self_snapshot.get("current_commitments") or []
        lines.extend(f"- {item}" for item in commitments[:6])
        if not commitments:
            lines.append("(none)")
        lines.extend(["", "Skill confidence:"])
        for skill, info in sorted((self_snapshot.get("skill_confidence") or {}).items()):
            lines.append(f"{skill:<20} {float(info.get('success_rate') or 0):.2f}")
        energy = self_snapshot.get("energy") or {}
        lines.extend(["", "Energy model:", f"calibrated           {energy.get('calibrated')}", f"prediction error     {energy.get('mean_prediction_error_pct')}"])
        return lines
    if panel == "vision":
        vision = snapshot.get("vision") or {}
        return [
            "VISION",
            f"Person count         {vision.get('person_count')}",
            f"Important objects    {', '.join(vision.get('important_objects') or []) or 'none'}",
            f"Last change          {vision.get('last_semantic_visual_change') or 'none'}",
            f"Confidence           {vision.get('vision_confidence')}",
            f"Observation age      {vision.get('last_observation_age_s')}",
            f"Model                {vision.get('model')}",
            f"Summary              {vision.get('scene_summary') or ''}",
        ]
    if panel == "navigation":
        nav = snapshot.get("navigation") or {}
        return _navigation_lines(nav)
    if panel == "asr":
        return _asr_lines(snapshot.get("asr") or {})
    if panel == "expressive":
        expressive = snapshot.get("expressive") or {}
        return [
            "EXPRESSIVE",
            f"Current motion       {expressive.get('current_expressive_motion') or 'none'}",
            f"Source file          {expressive.get('motion_source_file') or ''}",
            f"Reason               {expressive.get('reason') or ''}",
            f"Started at           {expressive.get('started_at')}",
            f"Last completed       {expressive.get('last_completed_at')}",
            f"Last error           {expressive.get('last_error') or 'none'}",
        ]
    if panel == "activity":
        return _activity_lines(snapshot.get("activity") or {})
    if panel == "active_learning":
        return _active_learning_lines(snapshot.get("active_learning") or {})
    if panel == "events":
        return ["ATTENTION / COGNITION STREAM"] + [
            f"{time.strftime('%H:%M:%S', time.localtime(event.get('timestamp', 0)))}  "
            f"{str(event.get('category', '')).upper():<9} {event.get('event', ''):<28} {event.get('summary', '')}"
            for event in events[-60:]
        ]
    objectives = snapshot.get("objectives") or []
    lines = [
        "AGENT STATE",
        f"Posture              {semantic.get('posture')}",
        f"Balance              {semantic.get('balance')}",
        f"Battery              {semantic.get('battery')}",
        f"Interaction          {semantic.get('interaction')}",
        f"Current task         {semantic.get('task')}",
        f"Arm control          {semantic.get('arm_state')}",
        f"Active faults        {', '.join(semantic.get('active_faults') or []) or 'none'}",
        "",
        "VISION",
        f"Objects              {', '.join((snapshot.get('vision') or {}).get('important_objects') or []) or 'none'}",
        "NAVIGATION",
        f"SLAM                 {(snapshot.get('navigation') or {}).get('slam')}",
        "ASR",
        f"Listening            {(snapshot.get('asr') or {}).get('listening')}",
        "EXPRESSIVE",
        f"Motion               {(snapshot.get('expressive') or {}).get('current_expressive_motion') or 'none'}",
        "SELF",
        f"Robot ID             {(snapshot.get('self') or {}).get('robot_id')}",
        f"Version              {(snapshot.get('self') or {}).get('version')}",
        "ACTIVITY",
        f"Current              {(snapshot.get('activity') or {}).get('current_activity')}",
        "ACTIVE LEARNING",
        f"Pending              {((snapshot.get('active_learning') or {}).get('pending_question') or {}).get('question') or 'none'}",
        "",
        "CURRENT OBJECTIVES",
    ]
    for idx, objective in enumerate(objectives, start=1):
        lines.append(f"{idx}. {objective.get('summary')}                 priority {objective.get('priority')}")
    lines.extend(["", "RECENT EVENTS"])
    for event in events[-12:]:
        lines.append(
            f"{time.strftime('%H:%M:%S', time.localtime(event.get('timestamp', 0)))}  "
            f"{str(event.get('category', '')).upper():<9} {event.get('summary', '')}"
        )
    return lines


def _monitor_ui(stdscr: object, agent: G1Agent, initial_panel: str) -> None:
    curses.curs_set(0)
    stdscr.keypad(True)
    panel = initial_panel if initial_panel in MONITOR_PANELS else "overview"
    status = "q exits, left/right switches panels"
    while True:
        settings = agent.settings.effective()
        refresh_s = 1.0 / max(0.5, min(10.0, float(settings.monitor.refresh_hz)))
        snapshot = agent.monitor_snapshot(panel=panel)
        max_y, max_x = stdscr.getmaxyx()
        stdscr.erase()
        title = (
            f"G1 AGENT MONITOR  panel={panel}  Lifecycle={snapshot.get('lifecycle')}  "
            f"Model={snapshot.get('model')}"
        )
        _add_line(stdscr, 0, 0, title, max_x, curses.A_BOLD)
        last_age = snapshot.get("last_cognition_age_s")
        next_check = snapshot.get("next_scheduled_check_s")
        subtitle = (
            f"Last cognition: {'never' if last_age is None else f'{last_age:.1f}s ago'}   "
            f"Next scheduled check: {'n/a' if next_check is None else f'{next_check:.1f}s'}   "
            f"Attention queue: {snapshot.get('attention_queue')}   {status}"
        )
        _add_line(stdscr, 1, 0, subtitle, max_x)
        _add_line(stdscr, 2, 0, "Panels: " + " | ".join(MONITOR_PANELS), max_x, curses.A_DIM)
        y = 4
        for line in _monitor_lines(snapshot, panel):
            attr = curses.A_BOLD if line.isupper() or line in {"AGENT STATE", "CURRENT OBJECTIVES", "RECENT EVENTS"} else 0
            y = _add_line(stdscr, y, 0, line, max_x, attr)
            if y >= max_y - 1:
                break
        stdscr.refresh()
        stdscr.timeout(int(refresh_s * 1000))
        ch = stdscr.getch()
        if ch in (ord("q"), 27):
            return
        if ch == curses.KEY_RIGHT:
            panel = MONITOR_PANELS[(MONITOR_PANELS.index(panel) + 1) % len(MONITOR_PANELS)]
        elif ch == curses.KEY_LEFT:
            panel = MONITOR_PANELS[(MONITOR_PANELS.index(panel) - 1) % len(MONITOR_PANELS)]
        elif ch in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5"), ord("6")):
            panel = MONITOR_PANELS[int(chr(ch)) - 1]


def _run_monitor_ui(agent: G1Agent, *, panel: str = "overview") -> None:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        snapshot = agent.monitor_snapshot(panel=panel)
        _print(f"lifecycle={snapshot.get('lifecycle')} model={snapshot.get('model')}")
        _print(f"semantic_state={snapshot.get('semantic_state')}")
        for event in snapshot.get("events", [])[-20:]:
            _print(f"{event.get('category')}:{event.get('event')} {event.get('summary')}")
        return
    with PRINT_LOCK:
        curses.wrapper(_monitor_ui, agent, panel)


def _navigation_lines(nav: dict) -> list[str]:
    lines = [
        "NAVIGATION / SLAM",
        f"SLAM                {nav.get('slam')}",
        f"Navigation          {nav.get('navigation')}",
        f"Localization        {nav.get('localization')}",
        f"Map                 {nav.get('map') or ''}",
        f"Current pose        {nav.get('current_pose')}",
        f"Goal                {nav.get('goal') or ''}",
        f"Goal status         {nav.get('goal_status')}",
        f"Velocity command    {nav.get('velocity_command') or ''}",
        f"Planner status      {nav.get('planner_status') or ''}",
        f"Recovery status     {nav.get('recovery_status') or ''}",
        f"Last error          {nav.get('last_error') or 'none'}",
        "",
        "TOPICS",
    ]
    for name, info in sorted((nav.get("topics") or {}).items()):
        alive = info.get("alive")
        state = "unknown" if alive is None else ("alive" if alive else "stale/dead")
        age = info.get("age_s")
        lines.append(f"{name:<36} {state:<12} {info.get('type')} age={age}")
    return lines


def _navigation_ui(stdscr: object, agent: G1Agent, initial_panel: str) -> None:
    curses.curs_set(0)
    stdscr.keypad(True)
    status = "q exits, s start mapping, x stop SLAM, r relocate, p preflight"
    while True:
        nav = agent.navigation_snapshot()
        max_y, max_x = stdscr.getmaxyx()
        stdscr.erase()
        _add_line(stdscr, 0, 0, f"NAVIGATION / SLAM  panel={initial_panel}", max_x, curses.A_BOLD)
        _add_line(stdscr, 1, 0, status, max_x, curses.A_DIM)
        y = 3
        for line in _navigation_lines(nav):
            y = _add_line(stdscr, y, 0, line, max_x, curses.A_BOLD if line.isupper() else 0)
            if y >= max_y - 1:
                break
        stdscr.refresh()
        stdscr.timeout(500)
        ch = stdscr.getch()
        if ch in (ord("q"), 27):
            return
        if ch == ord("s"):
            status = agent.navigation_action("start_mapping")
        elif ch == ord("x"):
            status = agent.navigation_action("stop_slam")
        elif ch == ord("r"):
            status = agent.navigation_action("start_relocation")
        elif ch == ord("p"):
            status = agent.navigation_action("preflight")


def _run_navigation_ui(agent: G1Agent, *, panel: str = "status") -> None:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        for line in _navigation_lines(agent.navigation_snapshot()):
            _print(line)
        return
    with PRINT_LOCK:
        curses.wrapper(_navigation_ui, agent, panel)


def _asr_lines(asr: dict) -> list[str]:
    return [
        "ASR",
        f"Microphone           {asr.get('microphone_enabled')}",
        f"ASR                  {asr.get('asr_enabled')}",
        f"Audio -> cognition   {asr.get('audio_to_cognition')}",
        f"Audio -> state       {asr.get('audio_to_state')}",
        f"Listening            {asr.get('listening')}",
        f"Current confidence   {asr.get('confidence')}",
        f"Partial transcript   {asr.get('partial_transcript') or ''}",
        f"Final transcript     {asr.get('final_transcript') or ''}",
        f"Last accepted prompt {asr.get('last_accepted_prompt') or ''}",
        f"Last rejected input  {asr.get('last_rejected_input') or ''}",
        f"Silence timeout      {asr.get('silence_timeout_ms')} ms",
        f"Input topic          {asr.get('input_topic')}",
        f"Last error           {asr.get('last_error') or 'none'}",
    ]


def _activity_lines(activity: dict) -> list[str]:
    headlight = activity.get("headlight") or {}
    return [
        "ACTIVITY",
        f"Current activity     {activity.get('current_activity')}",
        f"Since                {activity.get('age_s', 0):.1f} s",
        f"Stack                {', '.join(activity.get('stack') or []) or 'none'}",
        f"Cognitive indicator  {headlight.get('current_indicator')}",
        f"Headlight color      {headlight.get('color')}",
        f"Headlight depth      {headlight.get('active_depth')}",
        f"Headlight override   {headlight.get('operator_override')}",
        f"Headlight error      {headlight.get('last_error') or 'none'}",
    ]


def _active_learning_lines(active_learning: dict) -> list[str]:
    pending = active_learning.get("pending_question") or {}
    last = active_learning.get("last_question") or {}
    return [
        "ACTIVE LEARNING",
        f"Enabled              {active_learning.get('enabled')}",
        f"Question permission  {active_learning.get('autonomous_questions_allowed')}",
        f"Cooldown remaining   {active_learning.get('cooldown_remaining_s', 0):.1f} s",
        f"Pending question     {pending.get('question') or 'none'}",
        f"Questions today      {active_learning.get('questions_today')}",
        f"Answered             {active_learning.get('answered')}",
        f"Declined             {active_learning.get('declined')}",
        f"Last question        {last.get('question') or ''}",
        f"Last answer          {active_learning.get('last_answer') or ''}",
        f"Learning result      {active_learning.get('last_learning_result') or ''}",
    ]


def _asr_ui(stdscr: object, agent: G1Agent) -> None:
    curses.curs_set(0)
    stdscr.keypad(True)
    while True:
        max_y, max_x = stdscr.getmaxyx()
        stdscr.erase()
        _add_line(stdscr, 0, 0, "ASR MONITOR  q exits", max_x, curses.A_BOLD)
        y = 2
        for line in _asr_lines(agent.asr_snapshot()):
            y = _add_line(stdscr, y, 0, line, max_x, curses.A_BOLD if line == "ASR" else 0)
            if y >= max_y - 1:
                break
        stdscr.refresh()
        stdscr.timeout(500)
        ch = stdscr.getch()
        if ch in (ord("q"), 27):
            return


def _run_asr_ui(agent: G1Agent) -> None:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        for line in _asr_lines(agent.asr_snapshot()):
            _print(line)
        return
    with PRINT_LOCK:
        curses.wrapper(_asr_ui, agent)


def _llctl_lines(snapshot: dict) -> list[str]:
    return [
        "LLCTL",
        f"Session enabled      {snapshot.get('session_enabled')}",
        f"Commands allowed     {snapshot.get('manual_commands_allowed')}",
        f"Permission reason    {snapshot.get('permission_reason') or ''}",
        f"Joint control        {snapshot.get('allow_joint_control')}",
        f"IK control           {snapshot.get('allow_ik_control')}",
        f"Backend available    {snapshot.get('backend_available')}",
        f"Backend              {snapshot.get('control_backend')}",
        f"Dashboard            {snapshot.get('dashboard_path')}",
        f"Features             {', '.join(snapshot.get('dashboard_features') or [])}",
        f"Backend error        {snapshot.get('backend_error') or 'none'}",
    ]


def _llctl_ui(stdscr: object, agent: G1Agent) -> None:
    curses.curs_set(0)
    stdscr.keypad(True)
    status = "q exits, e enable session, d disable session"
    while True:
        snap = agent.llctl_snapshot()
        max_y, max_x = stdscr.getmaxyx()
        stdscr.erase()
        _add_line(stdscr, 0, 0, "LLCTL ENGINEERING CONTROL", max_x, curses.A_BOLD)
        _add_line(stdscr, 1, 0, status, max_x, curses.A_DIM)
        y = 3
        for line in _llctl_lines(snap):
            y = _add_line(stdscr, y, 0, line, max_x, curses.A_BOLD if line == "LLCTL" else 0)
            if y >= max_y - 1:
                break
        stdscr.refresh()
        stdscr.timeout(500)
        ch = stdscr.getch()
        if ch in (ord("q"), 27):
            return
        if ch == ord("e"):
            status = agent.llctl_enable()
        elif ch == ord("d"):
            status = agent.llctl_disable()


def _run_llctl_ui(agent: G1Agent) -> None:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        for line in _llctl_lines(agent.llctl_snapshot()):
            _print(line)
        return
    with PRINT_LOCK:
        curses.wrapper(_llctl_ui, agent)


RESET_SCOPES = ["runtime", "conversation", "learned", "autobiography", "full"]
TACIT_PANELS = ["recent", "empirical", "procedural", "candidates", "contested", "deprecated", "stats", "history", "forgotten"]


def _reset_scope_description(scope: str) -> list[str]:
    descriptions = {
        "runtime": [
            "Clears runtime continuity, checkpoint, pending question and monitor events.",
            "Preserves settings, memory, learned knowledge and autobiography.",
        ],
        "conversation": [
            "Clears current dialogue continuity and pending learning question.",
            "Preserves robot experience, tacit knowledge, autobiography and settings.",
        ],
        "learned": [
            "Clears semantic learned memory, procedural/tacit memory and learned statistics.",
            "Preserves raw episodic history and static/documentary knowledge.",
        ],
        "autobiography": [
            "Clears autobiographical summary/history only.",
            "Preserves episodes, semantic memory, procedural memory and settings.",
        ],
        "full": [
            "Clears experiential cognitive identity: episodes, learned memory, tacit memory, autobiography and runtime.",
            "Preserves source code, static/documentary RAG, SDK knowledge, safety configuration and live robot state.",
        ],
    }
    return descriptions.get(scope, [])


def _confirm_reset_scope(scope: str) -> bool:
    phrase = f"RESET {scope.upper()}"
    _print("")
    _print(_style(f"RESET: {scope.upper()} AGENT STATE", Color.yellow))
    for line in _reset_scope_description(scope):
        _print(line)
    _print("")
    _print(f"Type exactly: {phrase}")
    try:
        typed = input(_style("confirm> ", Color.cyan)).strip()
    except (EOFError, KeyboardInterrupt):
        _print()
        return False
    return typed == phrase


def _handle_reset_command(agent: G1Agent, args: list[str]) -> None:
    if not args:
        _run_reset_ui(agent)
        return
    scope = args[0].strip().lower()
    if scope == "backups":
        _print(agent.reset_backups())
        return
    if scope not in RESET_SCOPES:
        _print("usage: /reset [runtime|conversation|learned|autobiography|full|backups]")
        return
    if agent.reset_manager.requires_confirmation(scope, agent.settings.effective()):
        if not _confirm_reset_scope(scope):
            agent.monitor_bus.emit("reset", "reset_cancelled", f"{scope} confirmation did not match")
            _print("reset cancelled")
            return
        agent.monitor_bus.emit("reset", "reset_confirmed", scope)
    _print(agent.cmd_reset(scope))


def _reset_lines(agent: G1Agent, selected: int) -> list[str]:
    lines = [
        "RESET AGENT COGNITIVE STATE",
        "arrows navigate, Enter executes selected scope, q exits",
        "",
    ]
    for idx, scope in enumerate(RESET_SCOPES):
        marker = "> " if idx == selected else "  "
        destructive = " destructive" if scope in agent.reset_manager.DESTRUCTIVE_SCOPES else ""
        lines.append(f"{marker}{scope:<16}{destructive}")
    lines.extend(["", "Selected scope:"])
    lines.extend(_reset_scope_description(RESET_SCOPES[selected]))
    return lines


def _reset_ui(stdscr: object, agent: G1Agent) -> None:
    curses.curs_set(0)
    stdscr.keypad(True)
    selected = 0
    status = ""
    while True:
        max_y, max_x = stdscr.getmaxyx()
        stdscr.erase()
        y = 0
        for line in _reset_lines(agent, selected):
            attr = curses.A_BOLD if y == 0 else (curses.A_REVERSE if line.startswith("> ") else 0)
            y = _add_line(stdscr, y, 0, line, max_x, attr)
            if y >= max_y - 2:
                break
        _add_line(stdscr, max_y - 1, 0, status, max_x, curses.A_DIM)
        stdscr.refresh()
        ch = stdscr.getch()
        if ch in (ord("q"), 27):
            return
        if ch == curses.KEY_UP:
            selected = (selected - 1) % len(RESET_SCOPES)
        elif ch == curses.KEY_DOWN:
            selected = (selected + 1) % len(RESET_SCOPES)
        elif ch in (10, 13):
            scope = RESET_SCOPES[selected]
            if agent.reset_manager.requires_confirmation(scope, agent.settings.effective()):
                phrase = f"RESET {scope.upper()}"
                curses.echo()
                stdscr.move(max_y - 2, 0)
                stdscr.clrtoeol()
                stdscr.addnstr(max_y - 2, 0, f"Type exactly '{phrase}': ", max_x - 1)
                typed = stdscr.getstr(max_y - 2, min(len(phrase) + 17, max_x - 1), 80).decode("utf-8").strip()
                curses.noecho()
                if typed != phrase:
                    status = "reset cancelled"
                    continue
                agent.monitor_bus.emit("reset", "reset_confirmed", scope)
            status = agent.cmd_reset(scope)


def _run_reset_ui(agent: G1Agent) -> None:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        for line in _reset_lines(agent, 0):
            _print(line)
        return
    with PRINT_LOCK:
        curses.wrapper(_reset_ui, agent)


def _handle_tacit_command(agent: G1Agent, args: list[str]) -> None:
    if args and args[0] in {"show", "evidence"}:
        item_id = args[1] if len(args) > 1 else None
        for line in agent.tacit_lines(panel=args[0], item_id=item_id):
            _print(line)
        return
    panel = args[0] if args else "recent"
    if panel not in TACIT_PANELS:
        _print("usage: /tacit [empirical|procedural|candidates|contested|deprecated|recent|stats|history|forgotten|show <id>|evidence <id>]")
        return
    _run_tacit_ui(agent, panel=panel)


def _tacit_ui(stdscr: object, agent: G1Agent, initial_panel: str) -> None:
    curses.curs_set(0)
    stdscr.keypad(True)
    panel = initial_panel if initial_panel in TACIT_PANELS else "recent"
    status = "q exits, left/right switches panels"
    while True:
        max_y, max_x = stdscr.getmaxyx()
        stdscr.erase()
        _add_line(stdscr, 0, 0, f"TACIT / LEARNED KNOWLEDGE  panel={panel}", max_x, curses.A_BOLD)
        _add_line(stdscr, 1, 0, status, max_x, curses.A_DIM)
        _add_line(stdscr, 2, 0, "Panels: " + " | ".join(TACIT_PANELS), max_x, curses.A_DIM)
        y = 4
        for line in agent.tacit_lines(panel=panel):
            attr = curses.A_BOLD if line.isupper() or line.startswith("[") else 0
            y = _add_line(stdscr, y, 0, line, max_x, attr)
            if y >= max_y - 1:
                break
        stdscr.refresh()
        stdscr.timeout(800)
        ch = stdscr.getch()
        if ch in (ord("q"), 27):
            return
        if ch == curses.KEY_RIGHT:
            panel = TACIT_PANELS[(TACIT_PANELS.index(panel) + 1) % len(TACIT_PANELS)]
        elif ch == curses.KEY_LEFT:
            panel = TACIT_PANELS[(TACIT_PANELS.index(panel) - 1) % len(TACIT_PANELS)]


def _run_tacit_ui(agent: G1Agent, *, panel: str = "recent") -> None:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        for line in agent.tacit_lines(panel=panel):
            _print(line)
        return
    with PRINT_LOCK:
        curses.wrapper(_tacit_ui, agent, panel)


def _tick_loop(agent: G1Agent, interval_s: float, stop_event: threading.Event, agent_lock: threading.RLock) -> None:
    interval = max(1.0, float(interval_s))
    while not stop_event.wait(interval):
        try:
            with agent_lock:
                outcome = agent.handle_cognitive_tick()
            if (
                outcome.decision.intent.value != "no_action"
                or outcome.grounded_response
                or outcome.skill_outcomes
            ):
                _print(_style("\n[tick]", Color.dim))
                _print_turn(outcome)
        except Exception as exc:
            _print(_style(f"\n[tick error] {exc}", Color.red))


def _extract_asr_text(record: object) -> str:
    if not isinstance(record, dict):
        return ""
    payload = record.get("payload")
    if isinstance(payload, dict):
        for key in ("text", "transcript", "result", "data"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    text = record.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    raw = record.get("raw")
    return raw.strip() if isinstance(raw, str) else ""


def _asr_loop(agent: G1Agent, stop_event: threading.Event, agent_lock: threading.RLock) -> None:
    robot = getattr(agent, "robot", None)
    if robot is None or not hasattr(robot, "get_mic"):
        return
    last_text = ""
    last_ts = 0.0
    last_error = ""
    last_error_ts = 0.0
    agent.asr_runtime.started()
    while not stop_event.is_set():
        try:
            settings = agent.settings.effective()
            agent.asr_runtime.update_settings(settings)
            if not (settings.audio.input_enabled and settings.audio.asr_enabled and settings.asr.enabled):
                stop_event.wait(0.5)
                continue
            records = robot.get_mic(
                duration_s=1.0,
                max_messages=1,
                print_messages=False,
                use_cli=True,
            )
            now = time.time()
            for record in records:
                text = _extract_asr_text(record)
                if not text:
                    continue
                confidence = None
                payload = record.get("payload") if isinstance(record, dict) else None
                if isinstance(payload, dict) and payload.get("confidence") is not None:
                    try:
                        confidence = float(payload.get("confidence"))
                    except Exception:
                        confidence = None
                agent.asr_runtime.final(text, confidence=confidence)
                if confidence is not None and confidence < float(settings.asr.confidence_threshold):
                    agent.asr_runtime.rejected(text, "confidence below threshold", confidence=confidence)
                    continue
                if text == last_text and now - last_ts < 3.0:
                    continue
                last_text = text
                last_ts = now
                with agent_lock:
                    outcome = agent.handle_audio_msg(text)
                if outcome is not None:
                    agent.asr_runtime.accepted(text, confidence=confidence)
                    _print(_style(f"\n[asr] {text}", Color.dim))
                    _print_turn(outcome)
                else:
                    agent.asr_runtime.rejected(text, "ASR cognition disabled", confidence=confidence)
        except Exception as exc:
            now = time.time()
            message = str(exc)
            agent.asr_runtime.error(message)
            if message != last_error or now - last_error_ts > 30.0:
                _print(_style(f"\n[asr error] {message}", Color.red))
                last_error = message
                last_error_ts = now
            stop_event.wait(2.0)
    agent.asr_runtime.stopped()


def repl(agent: G1Agent, *, tick_interval_s: float = 30.0, periodic_ticks: bool = True) -> None:
    _setup_readline_history()
    agent_lock = threading.RLock()
    decision = agent.boot()
    _print(f"{_style('[boot]', Color.cyan)} event={agent.boot_event.value} decision={decision.intent.value}")
    if decision.response_text:
        _print(f"{_style('agent>', Color.green)} {decision.response_text}")
    _print(_style("Type /help for commands, /exit to quit.", Color.dim))

    stop_event = threading.Event()
    tick_thread: threading.Thread | None = None
    asr_thread: threading.Thread | None = None
    if periodic_ticks and tick_interval_s > 0:
        tick_thread = threading.Thread(
            target=_tick_loop,
            args=(agent, tick_interval_s, stop_event, agent_lock),
            daemon=True,
        )
        tick_thread.start()
        _print(_style(f"[ticks] periodic cognition every {tick_interval_s:g}s", Color.dim))
    if getattr(agent, "robot", None) is not None and hasattr(getattr(agent, "robot", None), "get_mic"):
        asr_thread = threading.Thread(
            target=_asr_loop,
            args=(agent, stop_event, agent_lock),
            daemon=True,
        )
        asr_thread.start()
        _print(_style("[asr] listening on /audio_msg when audio input is enabled", Color.dim))

    try:
        while True:
            try:
                line = input(_style("you> ", Color.cyan)).strip()
            except (EOFError, KeyboardInterrupt):
                _print()
                break
            if not line:
                continue
            if line in ("/exit", "/quit", "/ende"):
                break
            with agent_lock:
                _dispatch(agent, line)
    finally:
        stop_event.set()
        if tick_thread is not None:
            tick_thread.join(timeout=2.0)
        if asr_thread is not None:
            asr_thread.join(timeout=2.0)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    agent = build_agent(args)
    repl(agent, tick_interval_s=args.tick_interval, periodic_ticks=not args.no_periodic_ticks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
