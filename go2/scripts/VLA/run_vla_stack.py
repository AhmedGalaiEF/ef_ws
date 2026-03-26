from __future__ import annotations

import argparse
import json
import signal
import sys
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from ollama_vla.agents import (
    ActorAgent,
    PerceptionAgent,
    PerceptionWorker,
    PlannerAgent,
    VLAController,
)
from ollama_vla.config import RuntimeConfig
from ollama_vla.ollama_client import OllamaChatClient
from ollama_vla.sport_actor import SportCommandExecutor
from ollama_vla.video_source import Go2VideoSource, RgbVisualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a 3-agent Ollama VLA control stack for Go2.")
    parser.add_argument("--iface", default="enp2s0")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3.5:2b")
    parser.add_argument("--perception-period", type=float, default=3.0)
    parser.add_argument("--planner-period", type=float, default=4.0)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--print-json", action="store_true", default=False)
    parser.add_argument("--visualize-rgb", action="store_true", default=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime = RuntimeConfig(
        iface=args.iface,
        perception_period_sec=args.perception_period,
        planner_period_sec=args.planner_period,
        dry_run=args.dry_run,
    )
    runtime.ollama.base_url = args.ollama_url
    runtime.ollama.model = args.model

    ChannelFactoryInitialize(0, runtime.iface)

    ollama = OllamaChatClient(
        base_url=runtime.ollama.base_url,
        model=runtime.ollama.model,
        timeout_sec=runtime.ollama.request_timeout_sec,
    )
    video = Go2VideoSource(timeout_sec=runtime.video_timeout_sec, fps=runtime.video_fps)
    video.start()
    visualizer = RgbVisualizer(video) if args.visualize_rgb else None
    if visualizer is not None:
        visualizer.start()

    perception_agent = PerceptionAgent(ollama, runtime.perception_system_prompt)
    planner_agent = PlannerAgent(ollama, runtime.planner_system_prompt)
    actor_agent = ActorAgent(ollama, runtime.actor_system_prompt, runtime)
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

    stop = False

    def _stop_handler(_signum, _frame) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _stop_handler)
    signal.signal(signal.SIGTERM, _stop_handler)

    print(
        f"Starting Ollama VLA stack: iface={runtime.iface}, model={runtime.ollama.model}, "
        f"dry_run={runtime.dry_run}"
    )

    try:
        while not stop:
            step = controller.step()
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

            time.sleep(runtime.planner_period_sec)
    finally:
        perception_worker.stop()
        if visualizer is not None:
            visualizer.stop()
        video.stop()
        if not runtime.dry_run:
            executor.execute({"name": "stop_move", "args": {}, "duration_sec": 0.0})

    return 0


if __name__ == "__main__":
    sys.exit(main())
