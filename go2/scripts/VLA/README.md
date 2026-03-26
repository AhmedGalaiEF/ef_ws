# VLA Stacks

This directory contains two minimal 3-agent VLA control stacks for Go2:

- an Ollama-backed stack using locally hosted chat models
- a Hugging Face-backed stack using hosted inference models

## Agents

- `PerceptionAgent`: polls `VideoClient`, sends the latest RGB frame plus a planner-provided prompt to the configured model backend, and produces structured scene observations.
- `PlannerAgent`: consumes the latest perception result and proposes the next short-horizon robot actions while also updating the next perception prompt.
- `ActorAgent`: converts planner suggestions into a constrained executable action list.
- `SportCommandExecutor`: deterministically maps allowed actions to `SportClient` methods.

## Files

- `run_vla_stack.py`: Ollama entry point
- `run_hf_vla_stack.py`: Hugging Face entry point
- `ollama_vla/config.py`: prompts and runtime configuration
- `ollama_vla/ollama_client.py`: minimal Ollama HTTP client
- `hf_vla/config.py`: Hugging Face runtime configuration
- `hf_vla/hf_client.py`: minimal Hugging Face chat completions client
- `ollama_vla/video_source.py`: `VideoClient` wrapper
- `ollama_vla/agents.py`: planner/perception/actor orchestration
- `ollama_vla/sport_actor.py`: `SportClient` execution layer

## Usage

Run the Ollama stack with live perception/planning but no robot actuation:

```bash
python run_vla_stack.py --iface enp0s31f6 --dry-run --print-json
```

Run the Hugging Face stack with live perception/planning but no robot actuation:

```bash
HF_TOKEN=hf_xxx python run_hf_vla_stack.py --iface enp0s31f6 --dry-run --print-json
```

Run the Hugging Face stack fully offline with deterministic local fallback responses:

```bash
python run_hf_vla_stack.py --iface enp0s31f6 --dry-run --mock-hf --print-json
```

Run fully offline with deterministic local fallback responses:

```bash
python run_vla_stack.py --iface enp0s31f6 --dry-run --mock-ollama --print-json
```

Run with robot actuation enabled:

```bash
python run_vla_stack.py --iface enp0s31f6
```

Show the RGB feed in a live window while the agent stack runs:

```bash
python run_vla_stack.py --iface enp0s31f6 --dry-run --visualize-rgb
```

Preflight the Go2 speaker path:

```bash
python test_go2_speaker.py --iface enp0s31f6 --text "Speaker test successful"
```

Preflight streaming a WAV file to the Go2 speaker:

```bash
python test_go2_speaker.py --iface enp0s31f6 --wav /path/to/file.wav
```

The WAV path must be mono 16-bit PCM at 16 kHz.

Convert text to WAV locally:

```bash
python text_to_wav.py "Speaker test successful" --output output.wav
```

Convert text locally, then check Go2 audio support:

```bash
python text_to_voice_go2.py "Speaker test successful" --iface enp0s31f6 --save-wav output.wav
```

Check Go2 audio support after generating a gTTS WAV:

```bash
python text_to_voice_go2.py "Speaker test successful" --iface enp0s31f6 --mode gtts-stream --save-wav output.wav
```

## Assumptions

- `unitree_sdk2py` is installed in the Python environment used to launch the script.
- Ollama is running locally at `http://127.0.0.1:11434` for the Ollama stack's non-dry-run execution.
- Model `qwen3.5:2b` is already pulled in Ollama for the Ollama stack's non-dry-run execution.
- `HF_TOKEN` is set or `--hf-token` is passed for the Hugging Face stack's non-dry-run execution.
- The selected Hugging Face model supports text-only chat completions for planner/actor calls and image inputs for perception calls.
- The installed `unitree_sdk2py` package exposes Go2 `vui` controls, but not a Go2 audio/TTS streaming client.

## Safety

The executable action vocabulary is intentionally narrow and clamps `move` commands to small velocities and short durations.
With `--dry-run`, the stack still queries the video source and the configured model backend, but no `SportClient` motion command is sent to the robot.
With `--mock-ollama`, the stack uses deterministic local fallback outputs for perception, planning, and action selection.
With `--mock-hf`, the Hugging Face stack uses deterministic local fallback outputs for perception, planning, and action selection.
