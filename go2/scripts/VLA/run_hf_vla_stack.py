from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time

from hf_vla.config import RuntimeConfig
from cli_validation import positive_finite_float
from hf_vla.hf_client import (
    DryRunHuggingFaceChatClient,
    FallbackHuggingFaceChatClient,
    HuggingFaceChatClient,
)
from intent_summary import build_intent_statement
from local_speech import LocalSpeechAnnouncer
from ollama_vla.agents import (
    ActorAgent,
    PerceptionAgent,
    PerceptionWorker,
    PlannerAgent,
    VLAController,
)
from ollama_vla.sport_actor import SportCommandExecutor
from ollama_vla.video_source import Go2VideoSource, RgbVisualizer
from sdk_safety import init_channel_autodetect


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a 3-agent Hugging Face VLA control stack for Go2."
    )
    parser.add_argument("--iface", default="enp2s0")
    parser.add_argument("--hf-api-url", default="https://router.huggingface.co/v1/chat/completions")
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--perception-period", type=positive_finite_float, default=3.0)
    parser.add_argument("--planner-period", type=positive_finite_float, default=4.0)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument(
        "--mock-hf",
        action="store_true",
        default=False,
        help="Use deterministic local fallback responses instead of querying Hugging Face.",
    )
    parser.add_argument("--print-json", action="store_true", default=False)
    parser.add_argument("--visualize-rgb", action="store_true", default=False)
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
        mock_hf=args.mock_hf,
    )
    runtime.hf.api_url = args.hf_api_url
    runtime.hf.api_token = args.hf_token
    runtime.hf.model = args.model

    init_channel_autodetect(runtime.iface)

    if runtime.mock_hf:
        model_client = DryRunHuggingFaceChatClient(model=f"{runtime.hf.model}-dry-run")
    else:
        primary_client = HuggingFaceChatClient(
            api_url=runtime.hf.api_url,
            model=runtime.hf.model,
            api_token=runtime.hf.api_token,
            timeout_sec=runtime.hf.request_timeout_sec,
            temperature=runtime.hf.temperature,
        )
        if runtime.dry_run:
            fallback_client = DryRunHuggingFaceChatClient(model=f"{runtime.hf.model}-dry-run")
            warned = False

            def _warn_fallback(exc: Exception) -> None:
                nonlocal warned
                if warned:
                    return
                warned = True
                print(
                    "Hugging Face backend unavailable in --dry-run; "
                    f"falling back to deterministic local responses: {exc}",
                    file=sys.stderr,
                )

            model_client = FallbackHuggingFaceChatClient(
                primary=primary_client,
                fallback=fallback_client,
                on_error=_warn_fallback,
            )
        else:
            model_client = primary_client
    video = Go2VideoSource(timeout_sec=runtime.video_timeout_sec, fps=runtime.video_fps)
    video.start()
    visualizer = RgbVisualizer(video) if args.visualize_rgb else None
    if visualizer is not None:
        visualizer.start()

    perception_agent = PerceptionAgent(model_client, runtime.perception_system_prompt)
    planner_agent = PlannerAgent(model_client, runtime.planner_system_prompt)
    actor_agent = ActorAgent(model_client, runtime.actor_system_prompt, runtime)
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
        f"Starting Hugging Face VLA stack: iface={runtime.iface}, model={runtime.hf.model}, "
        f"dry_run={runtime.dry_run}, mock_hf={runtime.mock_hf}"
    )
    if args.speak_intent and not announcer.available():
        print(
            "Local speech is enabled but no supported TTS command was found "
            "(tried: spd-say, espeak-ng, espeak, say).",
            file=sys.stderr,
        )

    try:
        while not stop.is_set():
            if visualizer is not None:
                visualizer.render_latest()
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
        print("\nStopping Hugging Face VLA stack.")
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
