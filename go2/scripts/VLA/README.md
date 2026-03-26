# Ollama VLA Stack

This directory contains a minimal 3-agent VLA control stack for Go2 using Ollama-hosted `qwen3.5:2b` models.

## Agents

- `PerceptionAgent`: polls `VideoClient`, sends the latest RGB frame plus a planner-provided prompt to Ollama, and produces structured scene observations.
- `PlannerAgent`: consumes the latest perception result and proposes the next short-horizon robot actions while also updating the next perception prompt.
- `ActorAgent`: converts planner suggestions into a constrained executable action list.
- `SportCommandExecutor`: deterministically maps allowed actions to `SportClient` methods.

## Files

- `run_vla_stack.py`: entry point
- `ollama_vla/config.py`: prompts and runtime configuration
- `ollama_vla/ollama_client.py`: minimal Ollama HTTP client
- `ollama_vla/video_source.py`: `VideoClient` wrapper
- `ollama_vla/agents.py`: planner/perception/actor orchestration
- `ollama_vla/sport_actor.py`: `SportClient` execution layer

## Usage

Run in dry-run mode first:

```bash
python run_vla_stack.py --iface enp2s0 --dry-run --print-json
```

Run with robot actuation enabled:

```bash
python run_vla_stack.py --iface enp2s0
```

Show the RGB feed in a live window while the agent stack runs:

```bash
python run_vla_stack.py --iface enp2s0 --dry-run --visualize-rgb
```

## Assumptions

- `unitree_sdk2py` is installed in the Python environment used to launch the script.
- Ollama is running locally at `http://127.0.0.1:11434`.
- Model `qwen3.5:2b` is already pulled in Ollama.

## Safety

The executable action vocabulary is intentionally narrow and clamps `move` commands to small velocities and short durations.
With `--dry-run`, the stack still runs perception, planning, and action selection, but no `SportClient` motion command is sent to the robot.
