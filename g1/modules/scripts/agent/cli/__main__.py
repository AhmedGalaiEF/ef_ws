"""Standalone REPL entry point (spec section 31's example CLI session).

Run with ``python -m agent.cli`` from ``g1/modules/scripts`` (no
``--robot``, no OpenAI key needed -- offline mock backend + MockPlanner),
or ``python -m agent.cli --robot --openai`` on the real deployment target.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap_scripts_path() -> None:
    scripts_dir = Path(__file__).resolve().parents[2]  # g1/modules/scripts
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))


_bootstrap_scripts_path()

from agent.cli.router import G1Agent, TurnOutcome  # noqa: E402
from agent.knowledge.sdk_wrapper_knowledge import SdkWrapperKnowledge  # noqa: E402
from agent.memory.manager import MemoryManager  # noqa: E402
from agent.planner import MockPlanner, OpenAIPlanner, Planner, PlannerError  # noqa: E402
from agent.settings.manager import SettingsManager  # noqa: E402
from agent.skills import SkillUnavailable, build_live_registry, build_offline_registry  # noqa: E402
from agent.state import MockRobotStateSource, SdkClientRobotStateSource  # noqa: E402


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
    return parser


def build_agent(args: argparse.Namespace) -> G1Agent:
    settings = SettingsManager()
    memory = MemoryManager()
    sdk_knowledge = SdkWrapperKnowledge()

    document_rag = None
    if args.knowledge_file:
        try:
            from agent.knowledge.document_rag import DocumentRAG

            document_rag = DocumentRAG(args.knowledge_file)
        except Exception as exc:
            print(f"[warn] documentary RAG unavailable: {exc}")

    robot = None
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
            skills = build_live_registry(robot=robot)
            state_source: object = SdkClientRobotStateSource(robot)
        except SkillUnavailable as exc:
            print(f"[warn] live skill registry unavailable ({exc}); falling back to offline mode.")
            skills = build_offline_registry()
            state_source = MockRobotStateSource()
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

    return G1Agent(
        planner=planner,
        skills=skills,
        state_source=state_source,  # type: ignore[arg-type]
        settings=settings,
        memory=memory,
        sdk_knowledge=sdk_knowledge,
        document_rag=document_rag,
        auto_confirm=args.auto_confirm,
    )


def _print_turn(outcome: TurnOutcome) -> None:
    print(f"  [decision] intent={outcome.decision.intent.value} target={outcome.decision.target}")
    if outcome.grounded_response:
        print(f"agent> {outcome.grounded_response}")
    for skill_name, skill_outcome in outcome.skill_outcomes:
        message = skill_outcome.result.message if skill_outcome.result else skill_outcome.policy.reason
        print(f"  [skill] {skill_name} -> {skill_outcome.status}: {message}")


def _dispatch(agent: G1Agent, line: str) -> None:
    if line.startswith("/chat "):
        _print_turn(agent.handle_chat(line[len("/chat "):]))
    elif line.startswith("/audio_msg "):
        outcome = agent.handle_audio_msg(line[len("/audio_msg "):])
        if outcome is None:
            print("(ASR disabled: audio.asr_enabled=false -- no conversational event generated)")
        else:
            _print_turn(outcome)
    elif line.startswith("/settings"):
        print(agent.cmd_settings(line.split()[1:]))
    elif line == "/status":
        print(agent.cmd_status())
    elif line.startswith("/memory"):
        print(agent.cmd_memory(line.split()[1:]))
    elif line == "/tools":
        print(agent.cmd_tools())
    elif line == "/help":
        print(agent.cmd_help())
    elif line.startswith("/"):
        print(f"(unknown command: {line.split()[0]} -- try /help)")
    else:
        # Bare text defaults to /chat, mirroring llm_client/cli.py's REPL.
        _print_turn(agent.handle_chat(line))


def repl(agent: G1Agent) -> None:
    decision = agent.boot()
    print(f"[boot] event={agent.boot_event.value} decision={decision.intent.value}")
    if decision.response_text:
        print(f"agent> {decision.response_text}")
    print("Type /help for commands, /exit to quit.")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/exit", "/quit"):
            break
        _dispatch(agent, line)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    agent = build_agent(args)
    repl(agent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
