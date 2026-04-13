"""
g1_codex_control.py
===================

Natural-language control loop for the Unitree G1 robot.

Sends the user's command + the Robot class API to Codex (OpenAI),
which returns the exact method call(s) to execute on the Robot instance.
"""
from __future__ import annotations

import inspect
import json
import os
import sys
import textwrap
from pathlib import Path

# Ensure modules dir is importable
MODULES_DIR = Path(__file__).resolve().parent
if str(MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(MODULES_DIR))

from sdk_client import Robot

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_robot_api_description() -> str:
    """Build a concise description of every public method on Robot."""
    lines: list[str] = []
    for name, method in inspect.getmembers(Robot, predicate=inspect.isfunction):
        if name.startswith("_"):
            continue
        sig = inspect.signature(method)
        # skip 'self'
        params = [
            f"{p.name}: {p.annotation.__name__ if hasattr(p.annotation, '__name__') else p.annotation}"
            + (f" = {p.default!r}" if p.default is not inspect.Parameter.empty else "")
            for p in list(sig.parameters.values())[1:]
        ]
        doc = (inspect.getdoc(method) or "").split("\n")[0]
        lines.append(f"  robot.{name}({', '.join(params)})")
        if doc:
            lines.append(f"      # {doc}")
    return "\n".join(lines)


def _build_system_prompt(api_desc: str) -> str:
    return textwrap.dedent(f"""\
    You are controlling a Unitree G1 humanoid robot via Python.
    You have access to a `robot` object of class Robot with these methods:

    {api_desc}

    The user will give you a natural-language command.
    Respond with ONLY a JSON array of method calls to execute, in order.
    Each element must be an object with:
      "method": "<method_name>",
      "kwargs": {{<keyword arguments>}}

    Rules:
    - Only use methods listed above. Never invent methods.
    - Use keyword arguments matching the signatures above.
    - If the command is unclear or dangerous, return an empty array [].
    - Do NOT include any explanation—just the JSON array.
    """)


def _query_codex(system_prompt: str, user_command: str) -> list[dict]:
    """Call OpenAI API and parse the JSON response."""
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("openai package not installed. Run: pip install openai")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY environment variable.")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=os.environ.get("CODEX_MODEL", "gpt-4o"),
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_command},
        ],
    )

    raw = response.choices[0].message.content.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]

    calls = json.loads(raw)
    if not isinstance(calls, list):
        raise ValueError(f"Expected JSON array, got: {type(calls).__name__}")
    return calls


def _execute_calls(robot: Robot, calls: list[dict]) -> None:
    """Execute the method calls returned by codex on the robot."""
    for i, call in enumerate(calls, 1):
        method_name = call.get("method", "")
        kwargs = call.get("kwargs", {})

        if method_name.startswith("_") or not hasattr(robot, method_name):
            print(f"[{i}] SKIP unknown/private method: {method_name}")
            continue

        method = getattr(robot, method_name)
        if not callable(method):
            print(f"[{i}] SKIP {method_name} is not callable")
            continue

        print(f"[{i}] robot.{method_name}({', '.join(f'{k}={v!r}' for k, v in kwargs.items())})")
        result = method(**kwargs)
        if result is not None:
            print(f"     -> {result}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Codex-driven G1 robot control")
    parser.add_argument("--iface", default="eth0", help="Network interface")
    parser.add_argument("--no-safety", action="store_true", help="Skip hanger boot sequence")
    parser.add_argument("--no-sensors", action="store_true", help="Skip auto sensor start")
    parser.add_argument("--dry-run", action="store_true", help="Print calls without executing")
    parser.add_argument("command", nargs="*", help="Natural-language command (or omit for interactive mode)")
    args = parser.parse_args()

    api_desc = _get_robot_api_description()
    system_prompt = _build_system_prompt(api_desc)

    # Lazy-init robot only when we actually need to execute
    robot: Robot | None = None

    def get_robot() -> Robot:
        nonlocal robot
        if robot is None:
            print("[init] Connecting to robot...")
            robot = Robot(
                iface=args.iface,
                safety_boot=not args.no_safety,
                auto_start_sensors=not args.no_sensors,
            )
        return robot

    def handle_command(user_cmd: str) -> None:
        print(f"\n> {user_cmd}")
        calls = _query_codex(system_prompt, user_cmd)

        if not calls:
            print("[codex] No actions returned.")
            return

        print(f"[codex] Plan ({len(calls)} step(s)):")
        for i, c in enumerate(calls, 1):
            print(f"  {i}. {c.get('method', '?')}({c.get('kwargs', {})})")

        if args.dry_run:
            print("[dry-run] Skipping execution.")
            return

        _execute_calls(get_robot(), calls)

    # One-shot or interactive
    if args.command:
        handle_command(" ".join(args.command))
    else:
        print("G1 Codex Control  (type 'quit' to exit)")
        print(f"Model: {os.environ.get('CODEX_MODEL', 'gpt-4o')}")
        print()
        while True:
            try:
                user_input = input("codex> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user_input or user_input.lower() in ("quit", "exit", "q"):
                break
            handle_command(user_input)


if __name__ == "__main__":
    main()
