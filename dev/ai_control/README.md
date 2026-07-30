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

Install from this directory if you want the `ai-control` command:

```bash
cd ~/EF/ef_ws/dev/ai_control
python3 -m pip install -e .
ai-control
```

You can also run without installing:

```bash
cd ~/EF/ef_ws/dev
python3 -m ai_control.cli                 # mock backend, no hardware needed
python3 -m ai_control.cli --robot --iface eth0 --domain-id 0
```

Running `python3 cli.py` from this `ai_control/` directory also works.

Override model tags with `--router-model` / `--thinker-model` / `--vision-model`,
or the Ollama host with `--host`.

## SLAM / Named-Point Commands

The CLI recognizes the `nav_bot.py` text commands before sending anything to
Ollama. These commands are confirmed like other robot actions and then published
to `/model_api/navbot_command` when `--robot` is enabled:

```text
start mapping
stop mapping
relocate
save current point as kitchen
go to kitchen
list points
clear points
stop navigation
resume navigation
stop slam
navigation status
```

Start the nav bot separately so that something is subscribed to the command
topic, for example:

```bash
cd ~/EF/ef_ws
python3 g1/modules/scripts/ollama_ai/nav_bot.py --iface eth0 --domain-id 0
```

## Status

This is a placeholder scaffold: routing is single-label JSON classification,
the thinker's tool-calling is manual JSON-in-prompt (not Ollama's native
function-calling), and there's no streaming, retries, or conversation
persistence yet. Swap pieces out as the real requirements firm up.
