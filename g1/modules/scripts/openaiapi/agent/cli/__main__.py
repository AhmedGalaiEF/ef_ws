"""Standalone REPL entry point (spec section 31's example CLI session).

Run with ``python -m agent.cli`` from ``g1/modules/scripts`` (no
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
    scripts_dir = next((parent for parent in here.parents if (parent / "llm_client").exists()), here.parents[2])
    modules_dir = scripts_dir.parent if (scripts_dir.parent / "sdk_client.py").exists() else None
    for path in (scripts_dir, modules_dir):
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
    if args.knowledge_file:
        try:
            from agent.knowledge.document_rag import DocumentRAG

            document_rag = DocumentRAG(args.knowledge_file)
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
            skills = build_live_registry(robot=robot)
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

    return G1Agent(
        planner=planner,
        skills=skills,
        state_source=state_source,  # type: ignore[arg-type]
        settings=settings,
        memory=memory,
        sdk_knowledge=sdk_knowledge,
        document_rag=document_rag,
        resolver=resolver,
        auto_confirm=args.auto_confirm,
    )


def _print_turn(outcome: TurnOutcome) -> None:
    with PRINT_LOCK:
        decision = _style("decision", Color.magenta)
        print(f"  [{decision}] intent={outcome.decision.intent.value} target={outcome.decision.target}", flush=True)
        if outcome.grounded_response:
            print(f"{_style('agent>', Color.green)} {outcome.grounded_response}", flush=True)
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
    elif line in ("/settings-ui", "/settings ui"):
        _run_settings_ui(agent)
    elif line.startswith("/settings"):
        _print(agent.cmd_settings(line.split()[1:]))
    elif line == "/status":
        _print(agent.cmd_status())
    elif line == "/faults":
        _print(agent.cmd_faults())
    elif line.startswith("/memory"):
        _print(agent.cmd_memory(line.split()[1:]))
    elif line == "/tools":
        _print(agent.cmd_tools())
    elif line == "/help":
        _print(agent.cmd_help())
    elif line.startswith("/"):
        _print(_style(f"(unknown command: {line.split()[0]} -- try /help)", Color.yellow))
    else:
        # Bare text defaults to /chat, mirroring llm_client/cli.py's REPL.
        _print_turn(agent.handle_chat(line))


def _setting_rows(agent: G1Agent) -> list[tuple[str, object]]:
    settings = agent.settings.effective()
    rows = sorted(settings.as_flat_dict().items())
    for skill_name in agent.skills.names():
        rows.append((f"skill.{skill_name}", settings.get_skill_mode(skill_name).value))
    return rows


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
    return str(value)


def _edit_setting_value(stdscr: object, agent: G1Agent, key: str, old_value: object) -> str:
    curses.echo()
    max_y, max_x = stdscr.getmaxyx()
    prompt = f"New value for {key} [{old_value}]: "
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
            line = f"{marker}{key:<42} {value}"
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
    if periodic_ticks and tick_interval_s > 0:
        tick_thread = threading.Thread(
            target=_tick_loop,
            args=(agent, tick_interval_s, stop_event, agent_lock),
            daemon=True,
        )
        tick_thread.start()
        _print(_style(f"[ticks] periodic cognition every {tick_interval_s:g}s", Color.dim))

    try:
        while True:
            try:
                line = input(_style("you> ", Color.cyan)).strip()
            except (EOFError, KeyboardInterrupt):
                _print()
                break
            if not line:
                continue
            if line in ("/exit", "/quit"):
                break
            with agent_lock:
                _dispatch(agent, line)
    finally:
        stop_event.set()
        if tick_thread is not None:
            tick_thread.join(timeout=2.0)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    agent = build_agent(args)
    repl(agent, tick_interval_s=args.tick_interval, periodic_ticks=not args.no_periodic_ticks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
