#!/usr/bin/env python3
"""Console scene executor: runs a confirmed, ordered scene on the G1.

Drives NavState/Speaker/KnowledgeRetriever/OllamaClient (nav_bot.py) and
MotionPlayer (chatbot_with_tactile_dex3.py) directly in-process, plus one
sdk_client.Robot instance for hands/vision/state. No ROS node is created --
those classes are plain Python objects that happen to live in ROS2 node
files. See ../../../g1/docs (or the repo plan) for the full design.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

SCRIPT_DIR = Path(__file__).resolve().parent
G1_DIR = SCRIPT_DIR.parents[2] if SCRIPT_DIR.name == "ollama_ai" else SCRIPT_DIR.parent
if not (G1_DIR / "WBC").exists():
    G1_DIR = Path("/home/unitree/EF/ef_ws_clean/ef_ws/g1")
MODULES_DIR = G1_DIR / "modules"
SCRIPTS_DIR = MODULES_DIR / "scripts"
OLLAMA_AI_DIR = SCRIPTS_DIR / "ollama_ai"
WBC_DIR = G1_DIR / "WBC"
for _path in (SCRIPT_DIR, MODULES_DIR, SCRIPTS_DIR, OLLAMA_AI_DIR, WBC_DIR):
    if _path.exists() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from nav_bot import (  # noqa: E402
    DEFAULT_SYSTEM_PROMPT,
    KNOWLEDGE_SYSTEM_PROMPT,
    KnowledgeRetriever,
    NavState,
    OllamaClient,
    Speaker,
)
from chatbot_with_tactile_dex3 import (  # noqa: E402
    HL_ACTIONS,
    MotionPlayer,
    THINK_SEQUENCE,
)

DEFAULT_KNOWLEDGE_FILE = OLLAMA_AI_DIR / "robot_modules_knowledge.sample.json"
DEFAULT_POSE_FILE = WBC_DIR / "saved_ik_pose_cli_v3_poses.json"
DEFAULT_POINTS_FILE = OLLAMA_AI_DIR / "nav_points.json"

SCENE_SYSTEM_PROMPT = """You turn a human-written scene description for a Unitree G1 humanoid robot \
into an ordered JSON list of steps. Respond with ONLY a JSON object, no prose, no markdown fences:

{"understanding": "<one sentence confirming what you understood>", "steps": [ ... ]}

Each step is an object with a "type" and its args. Valid step types:
- {"type": "announce", "text": "...", "max_words": <int, optional>}
- {"type": "navigate", "point": "<point name>"}
- {"type": "move", "vx": <m/s forward, optional>, "vy": <m/s left, optional>, "vyaw": <rad/s turn, optional>}
- {"type": "gesture", "name": "<one of: %(gestures)s>"}
- {"type": "think_gesture"}
- {"type": "listen", "prompt": "<what the robot is waiting to hear>", "save_as": "<variable name>"}
- {"type": "rag_answer", "query": "<question, or {variable} to reuse a captured listen>", "max_words": <int, optional>}
- {"type": "vision_detect", "candidates": ["object a", "object b"]}
- {"type": "grasp", "object": "<object name, or {variable}>"}
- {"type": "hand_open", "hand": "left|right|both"}
- {"type": "hand_close", "hand": "left|right|both"}
- {"type": "release_arms"}
- {"type": "stop"}

Use "navigate" for going to a named point, and "move" for a brief untargeted nudge like "step back" or \
"take a step forward" (small vx/vy/vyaw, no point name). Use {variable_name} inside a string arg to \
refer to a value an earlier "listen" step captured with matching "save_as". Keep step order faithful \
to the scene description. Do not invent steps that were not implied by the description.
""" % {"gestures": ", ".join(sorted(HL_ACTIONS))}


class SceneContext:
    def __init__(self, args: argparse.Namespace, logger: logging.Logger) -> None:
        self.args = args
        self.logger = logger
        self.base_no_speech = bool(args.no_speech)
        self.toggles: dict[str, bool] = {"voice": True, "nav": True, "ll": True}
        self.vars: dict[str, str] = {}
        self.default_velocity: tuple[float, float, float] = (0.2, 0.0, 0.0)

        from sdk_client import Robot  # local import: only needed once we actually build a robot

        self.robot = Robot(
            iface=args.iface,
            domain_id=args.domain_id,
            safety_boot=False,
            recover_dev_mode_on_init=False,
            auto_start_sensors=True,
            ollama_url=args.ollama_url,
            chat_model=args.model,
        )
        self.nav = NavState(args)
        self.speaker = Speaker(args, logger)
        self.motion = MotionPlayer(args, logger)

        knowledge_paths = [Path(item).expanduser() for item in args.knowledge_file]
        existing = [path for path in knowledge_paths if path.exists()]
        missing = [str(path) for path in knowledge_paths if not path.exists()]
        if missing:
            logger.warning("Knowledge file(s) not found: %s", ", ".join(missing))
        self.retriever = KnowledgeRetriever(existing) if existing else None
        self.ollama = OllamaClient(args)

        self.sync_toggles()

    def sync_toggles(self) -> None:
        self.args.no_speech = self.base_no_speech or not self.toggles["voice"]
        self.motion.enabled = bool(self.args.enable_motion) and self.toggles["ll"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "knowledge_file",
        nargs="*",
        default=[str(DEFAULT_KNOWLEDGE_FILE)],
        help="Structured JSON knowledge file(s) for rag_answer steps.",
    )
    parser.add_argument("--iface", default=os.environ.get("G1_IFACE", "eth0"))
    parser.add_argument("--domain-id", type=int, default=int(os.environ.get("G1_DOMAIN_ID", "0")))

    nav_group = parser.add_argument_group("navigation")
    nav_group.add_argument("--map-path", default="/home/unitree/test.pcd")
    nav_group.add_argument("--points-file", default=str(DEFAULT_POINTS_FILE))
    nav_group.add_argument("--slam-type", default="indoor")

    speech_group = parser.add_argument_group("speech")
    speech_group.add_argument("--volume", type=int, default=None)
    speech_group.add_argument("--tts-language", default=None)
    speech_group.add_argument("--no-speech", action="store_true", help="Disable speech entirely, regardless of the voice toggle.")

    ollama_group = parser.add_argument_group("ollama / RAG")
    ollama_group.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    ollama_group.add_argument("--model", default="qwen3.5:9b")
    ollama_group.add_argument("--temperature", type=float, default=0.2)
    ollama_group.add_argument("--timeout", type=float, default=45.0)
    ollama_group.add_argument("--num-predict", type=int, default=220)
    ollama_group.add_argument("--num-ctx", type=int, default=4096)
    ollama_group.add_argument("--keep-alive", default="15m")
    ollama_group.add_argument("--knowledge-top-k", type=int, default=4)
    ollama_group.add_argument("--knowledge-min-score", type=float, default=0.06)
    ollama_group.add_argument("--knowledge-max-chars", type=int, default=2600)

    motion_group = parser.add_argument_group("motion / gestures")
    motion_group.add_argument("--disable-motion", dest="enable_motion", action="store_false", default=True)
    motion_group.add_argument("--pose-file", default=str(DEFAULT_POSE_FILE))
    motion_group.add_argument("--motion-speed", type=float, default=0.3)
    motion_group.add_argument("--motion-kp", type=float, default=30.0)
    motion_group.add_argument("--motion-kd", type=float, default=1.5)
    motion_group.add_argument("--pose-timeout-s", type=float, default=11.0)
    motion_group.add_argument("--sequence-gap", type=float, default=0.25)
    motion_group.add_argument("--thanks-hold-s", type=float, default=7.0)
    motion_group.add_argument("--post-sequence-hold-s", type=float, default=4.0)
    motion_group.add_argument("--release-after-sequence", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def build_context(args: argparse.Namespace, logger: logging.Logger) -> SceneContext:
    return SceneContext(args, logger)


# --------------------------------------------------------------------------
# Step dispatch
# --------------------------------------------------------------------------

class StepError(RuntimeError):
    pass


def resolve_vars(text: Any, variables: dict[str, str]) -> Any:
    if not isinstance(text, str) or "{" not in text:
        return text
    result = text
    for name, value in variables.items():
        result = result.replace("{" + name + "}", str(value))
    return result


def _resolve_step_strings(step: dict[str, Any], variables: dict[str, str]) -> dict[str, Any]:
    return {key: resolve_vars(value, variables) for key, value in step.items()}


def _truncate_words(text: str, max_words: int | None) -> str:
    if not max_words:
        return text
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(".,;:") + "..."


def describe_step(step: dict[str, Any]) -> str:
    kind = step.get("type", "?")
    if kind == "announce":
        return f"announce: \"{step.get('text', '')}\""
    if kind == "navigate":
        return f"navigate to point '{step.get('point', '')}'"
    if kind == "move":
        return f"move (vx={step.get('vx', 0.0)}, vy={step.get('vy', 0.0)}, vyaw={step.get('vyaw', 0.0)})"
    if kind == "gesture":
        return f"play gesture '{step.get('name', '')}'"
    if kind == "think_gesture":
        return "play the 'thinking' pose sequence"
    if kind == "listen":
        return f"listen: {step.get('prompt', '(waiting for speech)')!r} -> save as '{step.get('save_as', '-')}'"
    if kind == "rag_answer":
        return f"answer (RAG) query={step.get('query', '')!r}"
    if kind == "vision_detect":
        return f"look for objects: {', '.join(step.get('candidates', []))}"
    if kind == "grasp":
        return f"[STUB] grasp object '{step.get('object', '')}'"
    if kind == "hand_open":
        return f"open {step.get('hand', 'right')} hand"
    if kind == "hand_close":
        return f"close {step.get('hand', 'right')} hand"
    if kind == "release_arms":
        return "release arm authority"
    if kind == "stop":
        return "stop locomotion"
    return f"unknown step type '{kind}'"


def step_announce(ctx: SceneContext, step: dict[str, Any]) -> str:
    text = _truncate_words(str(step.get("text", "")).strip(), step.get("max_words"))
    if not text:
        return "nothing to say"
    if not ctx.toggles["voice"]:
        ctx.logger.info("[voice disabled] would say: %s", text)
        return f"[voice disabled] would say: {text}"
    ctx.speaker.say(text)
    return f"said: {text}"


def step_navigate(ctx: SceneContext, step: dict[str, Any]) -> str:
    point = str(step.get("point", "")).strip()
    if not point:
        raise StepError("navigate step is missing a point name")
    if not ctx.toggles["nav"]:
        ctx.logger.info("[navigation disabled] would navigate to '%s'", point)
        return f"[navigation disabled] would navigate to '{point}'"
    result = ctx.nav.go_to_point(point, auto_relocate=True)
    if not result.get("ok"):
        raise StepError(f"navigation to '{point}' failed: {result.get('raw')}")
    return f"navigated to '{point}': {result.get('raw')}"


def step_move(ctx: SceneContext, step: dict[str, Any]) -> str:
    vx = float(step.get("vx", 0.0))
    vy = float(step.get("vy", 0.0))
    vyaw = float(step.get("vyaw", 0.0))
    if not ctx.toggles["ll"]:
        ctx.logger.info("[LL disabled] would move vx=%.2f vy=%.2f vyaw=%.2f", vx, vy, vyaw)
        return f"[LL disabled] would move vx={vx:.2f} vy={vy:.2f} vyaw={vyaw:.2f}"
    ok = ctx.motion.move_for_user(vx, vy, vyaw)
    if not ok:
        raise StepError(f"move vx={vx:.2f} vy={vy:.2f} vyaw={vyaw:.2f} failed (see log for detail)")
    return f"moved vx={vx:.2f} vy={vy:.2f} vyaw={vyaw:.2f}"


def step_gesture(ctx: SceneContext, step: dict[str, Any]) -> str:
    name = str(step.get("name", "")).strip().lower()
    if name not in HL_ACTIONS:
        raise StepError(f"unsupported gesture '{name}'. Known gestures: {', '.join(sorted(HL_ACTIONS))}")
    if not ctx.toggles["ll"]:
        ctx.logger.info("[LL disabled] would play gesture '%s'", name)
        return f"[LL disabled] would play gesture '{name}'"
    ok = ctx.motion.play_hl_action(name)
    if not ok:
        raise StepError(f"gesture '{name}' failed (see log for detail)")
    return f"played gesture '{name}'"


def step_think_gesture(ctx: SceneContext, step: dict[str, Any]) -> str:
    if not ctx.toggles["ll"]:
        ctx.logger.info("[LL disabled] would play thinking gesture")
        return "[LL disabled] would play thinking gesture"
    thread = ctx.motion.play_async(THINK_SEQUENCE)
    if thread is not None:
        thread.join(timeout=max(5.0, float(ctx.args.pose_timeout_s) * 2))
    return "played thinking gesture"


def step_listen(ctx: SceneContext, step: dict[str, Any]) -> str:
    prompt = str(step.get("prompt", "")).strip() or "(listening)"
    text = input(f"    [LISTEN] {prompt}\n    > ").strip()
    save_as = step.get("save_as")
    if save_as:
        ctx.vars[str(save_as)] = text
    return f"heard: {text!r}" + (f" (saved as '{save_as}')" if save_as else "")


def step_rag_answer(ctx: SceneContext, step: dict[str, Any]) -> str:
    query = str(step.get("query", "")).strip()
    if not query:
        raise StepError("rag_answer step has no query (did an earlier 'listen' step fail to capture one?)")
    if ctx.retriever is not None:
        context = ctx.retriever.format_context(
            query,
            top_k=ctx.args.knowledge_top_k,
            min_score=ctx.args.knowledge_min_score,
            max_chars=ctx.args.knowledge_max_chars,
        )
        system_prompt = KNOWLEDGE_SYSTEM_PROMPT
    else:
        context = ""
        system_prompt = DEFAULT_SYSTEM_PROMPT
        ctx.logger.warning("No knowledge file loaded; answering without RAG context.")
    messages = [{"role": "system", "content": system_prompt}]
    if context:
        messages.append({"role": "system", "content": f"Knowledge context:\n{context}"})
    messages.append({"role": "user", "content": query})
    reply = ctx.ollama.chat(messages)
    reply = _truncate_words(reply, step.get("max_words"))
    if ctx.toggles["voice"]:
        ctx.speaker.say(reply)
    else:
        ctx.logger.info("[voice disabled] would say: %s", reply)
    return f"answered: {reply}"


def step_vision_detect(ctx: SceneContext, step: dict[str, Any]) -> str:
    candidates = step.get("candidates") or []
    if not candidates:
        raise StepError("vision_detect step has no candidates")
    seen: list[str] = []
    scores: dict[str, float] = {}
    for candidate in candidates:
        name = str(candidate).strip()
        try:
            confidence = float(ctx.robot.detect(name, confidence_threshold=0.0))
        except Exception as exc:
            ctx.logger.warning("detect(%r) failed: %s", name, exc)
            confidence = 0.0
        scores[name] = confidence
        if confidence >= 0.5:
            seen.append(name)
    ctx.vars["last_vision_scores"] = json.dumps(scores)
    if seen:
        ctx.vars["seen_objects"] = ", ".join(seen)
    return f"vision scores: {scores}"


def step_grasp(ctx: SceneContext, step: dict[str, Any]) -> str:
    obj = str(step.get("object", "")).strip() or "(unspecified object)"
    message = (
        f"[STUB] grasp requested for '{obj}' -- recognition_app.py is not wired up yet. "
        "See GrippingClient in navbot_with_gripping_v2.py for the intended future integration."
    )
    ctx.logger.warning(message)
    return message


def step_hand_open(ctx: SceneContext, step: dict[str, Any]) -> str:
    hand = str(step.get("hand", "right")).strip().lower()
    if not ctx.toggles["ll"]:
        ctx.logger.info("[LL disabled] would open %s hand", hand)
        return f"[LL disabled] would open {hand} hand"
    ctx.robot.hand_open(hand)
    return f"opened {hand} hand"


def step_hand_close(ctx: SceneContext, step: dict[str, Any]) -> str:
    hand = str(step.get("hand", "right")).strip().lower()
    if not ctx.toggles["ll"]:
        ctx.logger.info("[LL disabled] would close %s hand", hand)
        return f"[LL disabled] would close {hand} hand"
    ctx.robot.hand_close(hand)
    return f"closed {hand} hand"


def step_release_arms(ctx: SceneContext, step: dict[str, Any]) -> str:
    if not ctx.toggles["ll"]:
        ctx.logger.info("[LL disabled] would release arms")
        return "[LL disabled] would release arms"
    ok = ctx.motion.release_arms_for_user()
    return "released arms" if ok else "release arms returned a non-zero code (see log)"


def step_stop(ctx: SceneContext, step: dict[str, Any]) -> str:
    ctx.robot.stop()
    return "stopped locomotion"


STEP_HANDLERS: dict[str, Callable[[SceneContext, dict[str, Any]], str]] = {
    "announce": step_announce,
    "navigate": step_navigate,
    "move": step_move,
    "gesture": step_gesture,
    "think_gesture": step_think_gesture,
    "listen": step_listen,
    "rag_answer": step_rag_answer,
    "vision_detect": step_vision_detect,
    "grasp": step_grasp,
    "hand_open": step_hand_open,
    "hand_close": step_hand_close,
    "release_arms": step_release_arms,
    "stop": step_stop,
}


def dispatch_step(ctx: SceneContext, step: dict[str, Any]) -> str:
    kind = str(step.get("type", "")).strip()
    handler = STEP_HANDLERS.get(kind)
    if handler is None:
        raise StepError(f"unknown step type '{kind}'")
    resolved = _resolve_step_strings(step, ctx.vars)
    return handler(ctx, resolved)


# --------------------------------------------------------------------------
# Scene lifecycle: plan -> confirm -> run (Enter-gated) -> safety exit
# --------------------------------------------------------------------------

def extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise StepError(f"No JSON object found in model output: {text!r}")
    return json.loads(text[start : end + 1])


def generate_scene_plan(ctx: SceneContext, description: str) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SCENE_SYSTEM_PROMPT},
        {"role": "user", "content": description},
    ]
    raw = ctx.ollama.chat(messages)
    parsed = extract_json_object(raw)
    steps = parsed.get("steps")
    if not isinstance(steps, list) or not steps:
        raise StepError(f"Model did not return a non-empty 'steps' list: {raw!r}")
    for step in steps:
        if not isinstance(step, dict) or "type" not in step:
            raise StepError(f"Malformed step in model output: {step!r}")
    return {"understanding": str(parsed.get("understanding", "")).strip(), "steps": steps}


def print_plan(plan: dict[str, Any]) -> None:
    understanding = plan.get("understanding")
    if understanding:
        print(f"Understanding: {understanding}")
    print("Plan:")
    for index, step in enumerate(plan["steps"], start=1):
        print(f"  {index}. {describe_step(step)}")


def safety_exit_sequence(ctx: SceneContext) -> None:
    print("\n[SAFETY EXIT] Ctrl+C received -- stopping the robot safely.")
    actions: list[tuple[str, Callable[[], Any]]] = [
        ("pause navigation", lambda: ctx.nav.pause_nav()),
        ("stop gesture/motion sequence", lambda: ctx.motion.stop_sequence()),
        ("release arm authority", lambda: ctx.motion.release_arms_for_user()),
        ("open both hands", lambda: ctx.robot.hand_open("both")),
        ("stop locomotion", lambda: ctx.robot.stop()),
        ("announce stopping", lambda: ctx.speaker.say("Stopping now for safety.")),
    ]
    for label, action in actions:
        try:
            action()
            print(f"  [ok] {label}")
        except Exception as exc:
            print(f"  [FAILED] {label}: {exc}")
    print("[SAFETY EXIT] complete.")


def run_scene(ctx: SceneContext, plan: dict[str, Any]) -> None:
    print_plan(plan)
    answer = input("Type 'start' to begin, anything else to cancel: ").strip().lower()
    if answer != "start":
        print("Scene cancelled.")
        return

    steps = plan["steps"]
    total = len(steps)
    for index, step in enumerate(steps, start=1):
        resolved = _resolve_step_strings(step, ctx.vars)
        print(f"\nNext ({index}/{total}): {describe_step(resolved)}")
        input("Press Enter to run this step (Ctrl+C = safety exit)... ")
        try:
            result = dispatch_step(ctx, step)
            print(f"  -> {result}")
        except StepError as exc:
            print(f"  ! step failed: {exc}")
        except Exception as exc:
            print(f"  ! unexpected error: {exc}")
    print("\nScene complete.")


# --------------------------------------------------------------------------
# Interactive console commands (mapping, points, settings, robot state)
# --------------------------------------------------------------------------

HELP_TEXT = """\
Console commands:
  /map start|stop|relocate       start/stop SLAM mapping, or relocate against the saved map
  /point save <name>             save the robot's current pose under <name>
  /point clear                   delete all saved points
  /point list                    list saved point names
  /goto <name>                   navigate to a saved point
  /say <text>                    speak <text> now
  /gesture <name>                play an HL gesture (known: %(gestures)s)
  /state                         print a battery/temperature/lowstate snapshot
  /set volume <0-100>            change TTS volume
  /set velocity <vx> <vy> <vyaw> change the default locomotion velocity
  /toggle voice|nav|ll           flip a permission toggle on/off
  /scene <path-or-text>          parse and (after confirmation) run a scene
  /help                          show this message
  /exit                          quit
Anything not starting with '/' is treated as a scene description, same as '/scene <text>'.
""" % {"gestures": ", ".join(sorted(HL_ACTIONS))}


def cmd_state(ctx: SceneContext) -> None:
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "robot_status.py"),
        "--json",
        "--iface",
        str(ctx.args.iface),
        "--domain-id",
        str(ctx.args.domain_id),
    ]
    try:
        proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10.0)
    except Exception as exc:
        print(f"  ! could not run robot_status.py: {exc}")
        return
    try:
        parsed = json.loads(proc.stdout)
        print(json.dumps(parsed, indent=2, sort_keys=True))
    except Exception:
        print(proc.stdout.strip() or f"robot_status.py exited with code {proc.returncode}")


def cmd_map(ctx: SceneContext, action: str) -> None:
    if not ctx.toggles["nav"]:
        print("  [navigation disabled] skipping.")
        return
    ops = {"start": ctx.nav.start_mapping, "stop": ctx.nav.stop_mapping, "relocate": ctx.nav.relocate}
    op = ops.get(action)
    if op is None:
        print(f"  ! unknown /map action '{action}'. Use start, stop, or relocate.")
        return
    print(f"  -> {op()}")


def cmd_point(ctx: SceneContext, action: str, name: str = "") -> None:
    if action == "save":
        if not name:
            print("  ! usage: /point save <name>")
            return
        print(f"  -> {ctx.nav.add_current_point(name)}")
    elif action == "clear":
        print(f"  -> {ctx.nav.clear_points()}")
    elif action == "list":
        print("  points: " + (", ".join(sorted(ctx.nav.points)) or "(none)"))
    else:
        print(f"  ! unknown /point action '{action}'. Use save, clear, or list.")


def cmd_goto(ctx: SceneContext, name: str) -> None:
    if not name:
        print("  ! usage: /goto <name>")
        return
    print(f"  -> {step_navigate(ctx, {'point': name})}")


def cmd_say(ctx: SceneContext, text: str) -> None:
    if not text:
        print("  ! usage: /say <text>")
        return
    print(f"  -> {step_announce(ctx, {'text': text})}")


def cmd_gesture(ctx: SceneContext, name: str) -> None:
    if not name:
        print("  ! usage: /gesture <name>")
        return
    try:
        print(f"  -> {step_gesture(ctx, {'name': name})}")
    except StepError as exc:
        print(f"  ! {exc}")


def cmd_set(ctx: SceneContext, args: list[str]) -> None:
    if not args:
        print("  ! usage: /set volume <n>  |  /set velocity <vx> <vy> <vyaw>")
        return
    what = args[0]
    if what == "volume" and len(args) == 2:
        ctx.args.volume = int(args[1])
        print(f"  -> volume set to {ctx.args.volume}")
    elif what == "velocity" and len(args) == 4:
        ctx.default_velocity = (float(args[1]), float(args[2]), float(args[3]))
        print(f"  -> default velocity set to {ctx.default_velocity}")
    else:
        print("  ! usage: /set volume <n>  |  /set velocity <vx> <vy> <vyaw>")


def cmd_toggle(ctx: SceneContext, which: str) -> None:
    if which not in ctx.toggles:
        print(f"  ! unknown toggle '{which}'. Use voice, nav, or ll.")
        return
    ctx.toggles[which] = not ctx.toggles[which]
    ctx.sync_toggles()
    print(f"  -> {which} is now {'ON' if ctx.toggles[which] else 'OFF'}")


def handle_scene_input(ctx: SceneContext, description: str) -> None:
    candidate_path = Path(description).expanduser()
    if candidate_path.is_file():
        description = candidate_path.read_text(encoding="utf-8")
    print("Thinking about this scene...")
    try:
        plan = generate_scene_plan(ctx, description)
    except StepError as exc:
        print(f"  ! could not parse a plan: {exc}")
        return
    run_scene(ctx, plan)


def handle_command(ctx: SceneContext, line: str) -> bool:
    """Returns False if the REPL should exit."""
    parts = line[1:].split()
    if not parts:
        return True
    name, rest = parts[0].lower(), parts[1:]
    if name in ("exit", "quit"):
        return False
    if name == "help":
        print(HELP_TEXT)
    elif name == "state":
        cmd_state(ctx)
    elif name == "map" and rest:
        cmd_map(ctx, rest[0])
    elif name == "point" and rest:
        cmd_point(ctx, rest[0], rest[1] if len(rest) > 1 else "")
    elif name == "goto":
        cmd_goto(ctx, " ".join(rest))
    elif name == "say":
        cmd_say(ctx, " ".join(rest))
    elif name == "gesture":
        cmd_gesture(ctx, " ".join(rest))
    elif name == "set":
        cmd_set(ctx, rest)
    elif name == "toggle" and rest:
        cmd_toggle(ctx, rest[0])
    elif name == "scene":
        handle_scene_input(ctx, " ".join(rest))
    else:
        print(f"  ! unknown command '/{name}'. Type /help for the list.")
    return True


def print_banner(ctx: SceneContext) -> None:
    print("scene_executor -- console robot control")
    print(f"  iface={ctx.args.iface} domain_id={ctx.args.domain_id} model={ctx.args.model}")
    print(f"  toggles: voice={ctx.toggles['voice']} nav={ctx.toggles['nav']} ll={ctx.toggles['ll']}")
    print("Type /help for commands, or paste a scene description to plan one. Ctrl+C = safety exit.\n")


def repl(ctx: SceneContext) -> None:
    print_banner(ctx)
    while True:
        try:
            line = input("scene> ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line.startswith("/"):
            if not handle_command(ctx, line):
                break
        else:
            handle_scene_input(ctx, line)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("scene_executor")
    ctx = build_context(args, logger)
    try:
        repl(ctx)
    except KeyboardInterrupt:
        safety_exit_sequence(ctx)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
