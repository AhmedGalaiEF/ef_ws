#!/usr/bin/env python3
"""Interactive REPL chat with Claude Sonnet, full robot tool access.

Usage::

    python llm_client/cli.py                              # default: enp1s0, sonnet, robot online
    python llm_client/cli.py --no-robot                   # offline test (no tools)
    python llm_client/cli.py --model claude-haiku-4-5     # different model
    python llm_client/cli.py --iface eth0

In-session commands:
    /exit                quit
    /clear               wipe history (keeps system prompt)
    /system <text>       replace the system prompt + clear history
    /tools               list available tools
    /help                show these commands
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
from types import SimpleNamespace
from typing import Any, Dict, List

# Allow running as `python llm_client/cli.py` (no package context) AND
# as `python -m llm_client.cli` (proper package context).
if __package__ in (None, ""):
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from llm_client import chat as chat_module
    from llm_client.chat import send_chat_with_tool_usage_loop
    from llm_client.robot_tools import build_robot_tools
    try:
        import llm_client.secrets  # noqa: F401  (side effect: sets env vars)
    except Exception:
        pass
else:
    from . import chat as chat_module
    from .chat import send_chat_with_tool_usage_loop
    from .robot_tools import build_robot_tools
    try:
        from . import secrets as _secrets  # noqa: F401
    except Exception:
        pass


PROVIDERS = {
    "anthropic": {
        "base": "https://api.anthropic.com/v1",
        "model": "claude-sonnet-4-6",
        "env_var": "ANTHROPIC_API_KEY",
        "key_hint": "sk-ant-...",
    },
    "openai": {
        "base": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "env_var": "OPENAI_API_KEY",
        "key_hint": "sk-...",
    },
}
DEFAULT_PROVIDER = "anthropic"
DEFAULT_SYSTEM = (
    "You are an embodied agent controlling a Unitree G1 humanoid robot. "
    "The user's name is Mamadou — address them by name in your replies. "
    "You have tools to move the base (move), reach the arm (reach_forward), "
    "close the gripper around an object (grab), and open it (release). "
    "When the user asks you to act in the physical world, plan with the "
    "tools step by step and call them. After acting, briefly confirm what "
    "you did. If a request is ambiguous, ask one clarifying question first. "
    "Keep replies short — they will be spoken out loud by the robot."
)

# Hard cap on what we send to robot.say() so a long answer does not
# block the CLI for minutes of TTS.
SAY_MAX_CHARS = 400


# ----------------------------------------------------------------------
# Auth: build the header lambda the chat module expects.
# ----------------------------------------------------------------------

def configure_provider_auth(env_var: str, key_hint: str) -> None:
    if env_var not in os.environ:
        sys.stderr.write(
            f"ERROR: {env_var} is not set.\n"
            f"  - Either fill it into llm_client/secrets.py, or\n"
            f"  - export {env_var}={key_hint} before running.\n"
        )
        raise SystemExit(2)
    chat_module.dnabot_auth = SimpleNamespace(
        get_auth_header=lambda: {"Authorization": f"Bearer {os.environ[env_var]}"}
    )


# ----------------------------------------------------------------------
# Tool-call printer
# ----------------------------------------------------------------------

def make_tool_printer(verbose: bool):
    def _printer(name: str, args: Dict[str, Any], output: str) -> None:
        if not verbose:
            return
        arg_text = ", ".join(f"{k}={v!r}" for k, v in args.items())
        out_short = output if len(output) <= 200 else output[:197] + "..."
        print(f"  [tool] {name}({arg_text}) -> {out_short}", flush=True)
    return _printer


def make_confirm_prompt(enabled: bool):
    """Return a confirm_tool_call hook that asks the user before each call.

    Recognised replies:
      y / yes / <enter>  -> approve this single call
      n / no             -> deny this call (LLM is told the user blocked it)
      a / all            -> approve and remember; subsequent calls auto-yes
      q / quit           -> deny and remember; subsequent calls auto-no
    """
    if not enabled:
        return None

    state = {"auto": None}  # None | True | False

    def _confirm(name: str, args: Dict[str, Any]) -> bool:
        if state["auto"] is not None:
            decision = state["auto"]
            print(f"  [auto] {name}(...) -> {'approved' if decision else 'denied'}", flush=True)
            return decision
        arg_text = ", ".join(f"{k}={v!r}" for k, v in args.items())
        prompt = f"  [confirm] run {name}({arg_text}) ?  [Y/n/a=all/q=deny-all] "
        try:
            reply = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        if reply in ("", "y", "yes", "j", "ja"):
            return True
        if reply in ("a", "all"):
            state["auto"] = True
            return True
        if reply in ("q", "quit", "deny-all"):
            state["auto"] = False
            return False
        return False  # n / no / anything else

    return _confirm


# ----------------------------------------------------------------------
# Speak the answer through the robot's speaker.
# ----------------------------------------------------------------------

def say_safely(robot, text: str) -> None:
    """Best-effort robot.say(): truncate, swallow errors, never block the REPL."""
    if robot is None or not text:
        return
    payload = text.strip().replace("\n", " ")
    if len(payload) > SAY_MAX_CHARS:
        payload = payload[:SAY_MAX_CHARS - 1].rstrip() + "…"
    try:
        robot.say(payload)
    except Exception as exc:
        print(f"  (say failed: {exc})", flush=True)


# ----------------------------------------------------------------------
# Robot bring-up (or stub when --no-robot)
# ----------------------------------------------------------------------

def setup_robot_and_tools(iface: str, no_robot: bool):
    if no_robot:
        return None, {}, []
    # Local import so --no-robot mode doesn't pull in the SDK.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from sdk_client import Robot  # noqa: WPS433
    robot = Robot(iface)
    tools, schemas = build_robot_tools(robot)
    return robot, tools, schemas


# ----------------------------------------------------------------------
# Slash command handling
# ----------------------------------------------------------------------

def handle_slash(line: str, state: Dict[str, Any]) -> bool:
    """Return True if a slash command was handled (caller should skip LLM call)."""
    if line == "/exit":
        raise SystemExit(0)
    if line == "/clear":
        state["messages"] = [{"role": "system", "content": state["system"]}]
        print("(history cleared)")
        return True
    if line == "/help":
        print(textwrap.dedent(
            """
            /exit                quit
            /clear               wipe history (keeps system prompt)
            /system <text>       replace the system prompt and clear history
            /tools               list available tools
            /help                show this message
            """
        ).strip())
        return True
    if line == "/tools":
        names = list(state["tools"].keys())
        if not names:
            print("(no tools — running with --no-robot)")
        else:
            print("tools:", ", ".join(names))
        return True
    if line.startswith("/system "):
        state["system"] = line[len("/system "):].strip()
        state["messages"] = [{"role": "system", "content": state["system"]}]
        print("(system prompt replaced; history cleared)")
        return True
    if line.startswith("/"):
        print(f"(unknown command: {line.split()[0]} — try /help)")
        return True
    return False


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="LLM REPL with full robot tool access (Anthropic or OpenAI).")
    parser.add_argument("--provider", choices=list(PROVIDERS), default=DEFAULT_PROVIDER,
                        help="LLM provider: anthropic (default) or openai")
    parser.add_argument("--model", default=None,
                        help="model ID (default depends on --provider)")
    parser.add_argument("--iface", default="enp1s0", help="DDS network interface for the Robot")
    parser.add_argument("--system", default=DEFAULT_SYSTEM, help="initial system prompt")
    parser.add_argument("--max-iterations", type=int, default=10,
                        help="max tool-use rounds per user turn")
    parser.add_argument("--base", default=None, help="API base URL (default depends on --provider)")
    parser.add_argument("--no-robot", action="store_true",
                        help="skip Robot init (offline / no tools)")
    parser.add_argument("--quiet-tools", action="store_true",
                        help="hide tool-call debug output")
    parser.add_argument("--no-speak", action="store_true",
                        help="do not call robot.say() on each reply")
    parser.add_argument("--confirm-tools", action="store_true",
                        help="prompt y/n before every tool call (and a/q for approve-all/deny-all)")
    args = parser.parse_args()

    provider_cfg = PROVIDERS[args.provider]
    base = args.base or provider_cfg["base"]
    model = args.model or provider_cfg["model"]
    configure_provider_auth(provider_cfg["env_var"], provider_cfg["key_hint"])
    robot, tools, schemas = setup_robot_and_tools(args.iface, args.no_robot)

    state: Dict[str, Any] = {
        "system": args.system,
        "messages": [{"role": "system", "content": args.system}],
        "tools": tools,
    }

    tool_printer = make_tool_printer(verbose=not args.quiet_tools)
    tool_confirmer = make_confirm_prompt(enabled=args.confirm_tools)

    banner = (
        f"LLM REPL  provider={args.provider}  model={model}  base={base}\n"
        f"  tools: {list(tools.keys()) if tools else '(none — --no-robot)'}\n"
        f"  type /help for commands, /exit to quit"
    )
    print(banner)
    print("-" * 60)

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if handle_slash(user_input, state):
            continue

        state["messages"].append({"role": "user", "content": user_input})
        try:
            content = send_chat_with_tool_usage_loop(
                model_key=model,
                messages=state["messages"],
                base=base,
                tools=tools,
                tool_schemas=schemas,
                tool_choice="auto" if tools else None,
                max_iterations=args.max_iterations,
                on_tool_call=tool_printer,
                confirm_tool_call=tool_confirmer,
            )
        except KeyboardInterrupt:
            print("\n(interrupted)")
            # Keep history clean: drop the trailing user message we just appended.
            if state["messages"] and state["messages"][-1].get("role") == "user":
                state["messages"].pop()
            continue
        except Exception as exc:
            print(f"error: {exc}")
            if state["messages"] and state["messages"][-1].get("role") == "user":
                state["messages"].pop()
            continue

        # Append the assistant's final answer so multi-turn context is kept.
        state["messages"].append({"role": "assistant", "content": content})
        print(f"\nclaude> {content}\n")
        if not args.no_speak:
            say_safely(robot, content)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
