from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time

from intent_summary import build_intent_statement
from local_speech import LocalSpeechAnnouncer
from ollama_vla.agents import (
    ActorAgent,
    PerceptionAgent,
    PerceptionWorker,
    PlannerAgent,
    VLAController,
)
from ollama_vla.config import RuntimeConfig
from ollama_vla.ollama_client import (
    DryRunOllamaChatClient,
    FallbackOllamaChatClient,
    OllamaChatClient,
)
from ollama_vla.sport_actor import SportCommandExecutor
from ollama_vla.video_source import Go2VideoSource, RgbVisualizer
from planner_prompt_editor import PlannerPromptEditor
from sdk_safety import init_channel_autodetect


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a 3-agent Ollama VLA control stack for Go2.")
    parser.add_argument("--iface", default="enp2s0")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3.5:0.8b")
    parser.add_argument(
        "--vision-model",
        default="",
        help="Optional vision-capable Ollama model for perception. Defaults to --model.",
    )
    parser.add_argument("--request-timeout-sec", type=float, default=90.0)
    parser.add_argument("--text-num-predict", type=int, default=96)
    parser.add_argument("--vision-num-predict", type=int, default=160)
    parser.add_argument("--keep-alive", default="10m")
    parser.add_argument("--perception-period", type=float, default=3.0)
    parser.add_argument("--planner-period", type=float, default=4.0)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument(
        "--mock-ollama",
        action="store_true",
        default=False,
        help="Use deterministic local fallback responses instead of querying Ollama.",
    )
    parser.add_argument("--print-json", action="store_true", default=False)
    parser.add_argument("--visualize-rgb", action="store_true", default=False)
    parser.add_argument(
        "--interactive-planner-prompt",
        action="store_true",
        default=False,
        help="Ask before each planner inference whether to edit the planner system prompt.",
    )
    parser.add_argument("--speak-intent", dest="speak_intent", action="store_true", default=True)
    parser.add_argument("--no-speak-intent", dest="speak_intent", action="store_false")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime = RuntimeConfig(
        iface=args.iface,
        perception_period_sec=args.perception_period,
        planner_period_sec=args.planner_period,
        dry_run=args.dry_run,
        mock_ollama=args.mock_ollama,
    )
    runtime.ollama.base_url = args.ollama_url
    runtime.ollama.model = args.model
    runtime.ollama.request_timeout_sec = args.request_timeout_sec
    vision_model = args.vision_model or args.model

    text_options = {
        "temperature": 0.1,
        "num_predict": max(32, args.text_num_predict),
        "num_ctx": 1024,
    }
    vision_options = {
        "temperature": 0.1,
        "num_predict": max(64, args.vision_num_predict),
        "num_ctx": 1536,
    }

    init_channel_autodetect(runtime.iface)

    if runtime.mock_ollama:
        text_client = DryRunOllamaChatClient(model=f"{runtime.ollama.model}-dry-run")
        vision_client = DryRunOllamaChatClient(model=f"{vision_model}-dry-run")
    else:
        primary_text_client = OllamaChatClient(
            base_url=runtime.ollama.base_url,
            model=runtime.ollama.model,
            timeout_sec=runtime.ollama.request_timeout_sec,
            default_options=text_options,
            keep_alive=args.keep_alive,
        )
        primary_vision_client = OllamaChatClient(
            base_url=runtime.ollama.base_url,
            model=vision_model,
            timeout_sec=runtime.ollama.request_timeout_sec,
            default_options=vision_options,
            keep_alive=args.keep_alive,
        )
        if runtime.dry_run:
            text_fallback = DryRunOllamaChatClient(model=f"{runtime.ollama.model}-dry-run")
            vision_fallback = DryRunOllamaChatClient(model=f"{vision_model}-dry-run")
            warned = False

            def _warn_fallback(exc: Exception) -> None:
                nonlocal warned
                if warned:
                    return
                warned = True
                print(
                    "Ollama backend unavailable in --dry-run; "
                    "falling back to deterministic local responses: "
                    f"{exc}. Adjust with --request-timeout-sec, ensure the model is pulled, "
                    "or use --mock-ollama for forced offline behavior.",
                    file=sys.stderr,
                )

            text_client = FallbackOllamaChatClient(
                primary=primary_text_client,
                fallback=text_fallback,
                on_error=_warn_fallback,
            )
            vision_client = FallbackOllamaChatClient(
                primary=primary_vision_client,
                fallback=vision_fallback,
                on_error=_warn_fallback,
            )
        else:
            text_client = primary_text_client
            vision_client = primary_vision_client
    video = Go2VideoSource(timeout_sec=runtime.video_timeout_sec, fps=runtime.video_fps)
    video.start()
    visualizer = RgbVisualizer(video) if args.visualize_rgb else None
    if visualizer is not None:
        visualizer.start()

    perception_agent = PerceptionAgent(vision_client, runtime.perception_system_prompt)
    planner_agent = PlannerAgent(text_client, runtime.planner_system_prompt)
    actor_agent = ActorAgent(text_client, runtime.actor_system_prompt, runtime)
    perception_worker = PerceptionWorker(
        video_source=video,
        agent=perception_agent,
        period_sec=runtime.perception_period_sec,
        initial_prompt=runtime.initial_perception_prompt,
    )
    executor = SportCommandExecutor(timeout_sec=runtime.sport_timeout_sec, dry_run=runtime.dry_run)
    executor.start()

    controller = VLAController(
        planner=planner_agent,
        actor=actor_agent,
        perception_worker=perception_worker,
    )
    perception_worker.start()
    announcer = LocalSpeechAnnouncer(enabled=args.speak_intent)
    prompt_editor = PlannerPromptEditor(enabled=args.interactive_planner_prompt)
    current_planner_prompt = runtime.planner_system_prompt

    stop = threading.Event()
    shutting_down = threading.Event()
    exit_code = 0

    def _stop_handler(_signum, _frame) -> None:
        stop.set()
        if shutting_down.is_set():
            return
        raise KeyboardInterrupt

    def _term_handler(_signum, _frame) -> None:
        stop.set()

    signal.signal(signal.SIGINT, _stop_handler)
    signal.signal(signal.SIGTERM, _term_handler)

    print(
        f"Starting Ollama VLA stack: iface={runtime.iface}, text_model={runtime.ollama.model}, "
        f"vision_model={vision_model}, dry_run={runtime.dry_run}, mock_ollama={runtime.mock_ollama}"
    )
    if args.speak_intent and not announcer.available():
        print(
            "Local speech is enabled but no supported TTS command was found "
            "(tried: spd-say, espeak-ng, espeak, say).",
            file=sys.stderr,
        )
    if vision_model == runtime.ollama.model:
        print(
            "Perception is using the same Ollama model as planner/actor. "
            "For HF-like image understanding, pass --vision-model with a local vision-capable model.",
            file=sys.stderr,
        )

    try:
        while not stop.is_set():
            current_planner_prompt = prompt_editor.maybe_update(current_planner_prompt)
            planner_agent.set_system_prompt(current_planner_prompt)
            step = controller.step()
            announcer.announce(
                build_intent_statement(
                    planner_output=step.planner_output,
                    actor_output=step.actor_output,
                )
            )
            executed = executor.execute_many(step.actor_output.get("commands", []))

            if args.print_json:
                print(
                    json.dumps(
                        {
                            "perception": step.perception.data if step.perception else None,
                            "perception_error": step.perception_error,
                            "video_error": video.latest_error(),
                            "planner_output": step.planner_output,
                            "actor_output": step.actor_output,
                            "executed": [e.__dict__ for e in executed],
                        },
                        indent=2,
                    )
                )
            else:
                perception_summary = None
                if step.perception is not None:
                    perception_summary = step.perception.data.get("summary")
                print(
                    f"[{time.strftime('%H:%M:%S')}] perception={perception_summary!r} "
                    f"video_error={video.latest_error()!r} "
                    f"commands={step.actor_output.get('commands', [])}"
                )

            _wait_with_visualizer(stop, runtime.planner_period_sec, visualizer)
    except KeyboardInterrupt:
        stop.set()
        exit_code = 130
        print("\nStopping Ollama VLA stack.")
    finally:
        shutting_down.set()
        signal.signal(signal.SIGINT, _term_handler)
        stop.set()
        perception_worker.stop()
        if visualizer is not None:
            visualizer.stop()
        video.stop()
        announcer.close()
        if not runtime.dry_run:
            try:
                executor.execute({"name": "stop_move", "args": {}, "duration_sec": 0.0})
            except KeyboardInterrupt:
                exit_code = 130

    return exit_code


def _wait_with_visualizer(
    stop: threading.Event,
    duration_sec: float,
    visualizer: RgbVisualizer | None,
) -> None:
    deadline = time.time() + max(0.0, duration_sec)
    while not stop.is_set():
        remaining = deadline - time.time()
        if remaining <= 0.0:
            return
        if visualizer is not None:
            visualizer.render_latest()
        stop.wait(min(0.03, remaining))


if __name__ == "__main__":
    sys.exit(main())
