from __future__ import annotations

import argparse
import sys
from typing import Any

from ai_control import ollama_client, router, thinker, tools, vision
from ai_control.config import AIConfig
from ai_control.robot_backend import MockRobotBackend, RealRobotBackend, RobotBackend

BANNER = """\
ai_control -- placeholder multi-model robot control CLI
  router:  {router_model}
  thinker: {thinker_model}
  vision:  {vision_model}
  backend: {backend}
Type a message, or 'exit'/'quit' to leave. Ctrl-C also exits.
"""


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", action="store_true", help="Connect to a real G1 via sdk_lib instead of the mock backend.")
    parser.add_argument("--iface", default=AIConfig.iface, help="Network interface for the real robot connection.")
    parser.add_argument("--domain-id", type=int, default=AIConfig.domain_id, help="DDS domain id for the real robot connection.")
    parser.add_argument("--host", default=AIConfig.ollama_host, help="Ollama server base URL.")
    parser.add_argument("--router-model", default=AIConfig.router_model)
    parser.add_argument("--thinker-model", default=AIConfig.thinker_model)
    parser.add_argument("--vision-model", default=AIConfig.vision_model)
    return parser.parse_args(argv)


def _build_backend(args: argparse.Namespace) -> RobotBackend:
    if not args.robot:
        return MockRobotBackend()
    print(f"Connecting to real robot on iface={args.iface} domain_id={args.domain_id} ...")
    return RealRobotBackend(iface=args.iface, domain_id=args.domain_id)


def _confirm_and_dispatch(tool_call: dict[str, Any], backend: RobotBackend) -> str:
    name = str(tool_call.get("name", ""))
    args = tool_call.get("args") or {}
    spec = tools.TOOL_SPECS.get(name)
    if spec is None:
        message = f"Model proposed unknown tool {name!r}; ignoring."
        print(f"  ! {message}")
        return message

    print(f"  Proposed tool call [{spec.category}]: {name}({args})")
    answer = input("  Run this on the robot? [y/N] ").strip().lower()
    if answer not in ("y", "yes"):
        message = f"User declined tool call {name}({args})."
        print(f"  - declined")
        return message

    try:
        outcome = tools.dispatch(name, args, backend)
        message = f"Executed {name}({args}) -> {outcome}"
        print(f"  - {message}")
        return message
    except Exception as exc:  # noqa: BLE001 -- surfaced to the model/operator, not swallowed
        message = f"Tool call {name}({args}) failed: {exc}"
        print(f"  ! {message}")
        return message


def run(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    cfg = AIConfig(
        ollama_host=args.host,
        router_model=args.router_model,
        thinker_model=args.thinker_model,
        vision_model=args.vision_model,
        iface=args.iface,
        domain_id=args.domain_id,
    )
    backend = _build_backend(args)

    print(
        BANNER.format(
            router_model=cfg.router_model,
            thinker_model=cfg.thinker_model,
            vision_model=cfg.vision_model,
            backend="real (sdk_lib.G1)" if args.robot else "mock",
        )
    )

    history: list[dict[str, str]] = []
    while True:
        try:
            user_text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text.lower() in ("exit", "quit"):
            break

        try:
            route = router.classify(user_text, cfg)
        except ollama_client.OllamaError as exc:
            print(f"[router error, defaulting to chat] {exc}")
            route = router.RouteResult(intent="chat", raw="")
        print(f"  (intent: {route.intent})")

        vision_context: str | None = None
        if route.intent == "vision_query":
            frame = backend.capture_frame()
            if frame is None:
                vision_context = "No camera frame available (mock backend or camera offline)."
            else:
                try:
                    vision_context = vision.describe(frame, user_text, cfg)
                except ollama_client.OllamaError as exc:
                    vision_context = f"Vision model error: {exc}"

        try:
            result = thinker.think(history, user_text, cfg, vision_context=vision_context)
        except ollama_client.OllamaError as exc:
            print(f"assistant> [thinker error] {exc}")
            continue

        print(f"assistant> {result.response}")
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": result.response})

        if result.tool_call:
            outcome_message = _confirm_and_dispatch(result.tool_call, backend)
            history.append({"role": "system", "content": outcome_message})


if __name__ == "__main__":
    run(sys.argv[1:])
