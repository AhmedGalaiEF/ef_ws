# ai_control (placeholder)

Multi-model Ollama pipeline for text-in / text-out robot control, with a
confirm-before-execute gate on every tool call.

## Pieces

- `config.py` -- model tags / host / robot connection settings.
- `ollama_client.py` -- thin stdlib-only wrapper around `/api/chat`.
- `router.py` -- lightweight model, classifies each message into an intent
  (`chat`, `navigation`, `endeffector`, `gesture`, `speaker`, `vision_query`).
- `vision.py` -- vision-language model, only called for `vision_query` intents;
  describes the latest camera frame.
- `thinker.py` -- larger reasoning model, drafts the reply text and at most
  one proposed tool call as JSON.
- `tools.py` -- tool registry (navigation / endeffector / gesture / speaker)
  mapped onto `RobotBackend` methods, plus dispatch.
- `robot_backend.py` -- `MockRobotBackend` (default, just logs actions) and
  `RealRobotBackend` (wraps `sdk_lib.G1`, only imported with `--robot`).
- `cli.py` -- the REPL: routes, optionally pulls vision context, thinks,
  prints the response, and asks for `y/N` confirmation before dispatching
  any proposed tool call.

## Setup

```bash
ollama serve &
ollama pull qwen2.5:0.5b   # router
ollama pull qwen2.5:7b     # thinker
ollama pull llava:7b       # vision
```

## Run

From `dev/`:

```bash
python3 -m ai_control.cli                 # mock backend, no hardware needed
python3 -m ai_control.cli --robot --iface eth0 --domain-id 0
```

Override model tags with `--router-model` / `--thinker-model` / `--vision-model`,
or the Ollama host with `--host`.

## Status

This is a placeholder scaffold: routing is single-label JSON classification,
the thinker's tool-calling is manual JSON-in-prompt (not Ollama's native
function-calling), and there's no streaming, retries, or conversation
persistence yet. Swap pieces out as the real requirements firm up.
