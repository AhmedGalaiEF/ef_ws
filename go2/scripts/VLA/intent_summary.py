from __future__ import annotations

from typing import Any, Dict, List


def build_intent_statement(
    planner_output: Dict[str, Any],
    actor_output: Dict[str, Any],
) -> str:
    reasoning = str(planner_output.get("reasoning_brief", "") or "").strip()
    commands = actor_output.get("commands", [])
    action_phrase = summarize_commands(commands)

    if reasoning and action_phrase:
        return f"Go2 wants to {action_phrase}. {reasoning}"
    if action_phrase:
        return f"Go2 wants to {action_phrase}."
    if reasoning:
        return f"Go2 is holding position. {reasoning}"
    return "Go2 is holding position and waiting for the next safe action."


def summarize_commands(commands: List[Dict[str, Any]]) -> str:
    if not isinstance(commands, list) or not commands:
        return "hold position"

    phrases = []
    for command in commands[:2]:
        name = str(command.get("name", "stop_move"))
        args = dict(command.get("args", {}) or {})
        if name == "move":
            phrases.append(_summarize_move(args))
        elif name == "stop_move":
            phrases.append("stop and hold position")
        elif name == "stand_up":
            phrases.append("stand up")
        elif name == "stand_down":
            phrases.append("lower its posture")
        elif name == "balance_stand":
            phrases.append("steady itself in place")
        elif name == "recovery":
            phrases.append("recover its stance")
        elif name == "hello":
            phrases.append("wave hello")
        elif name == "stretch":
            phrases.append("stretch")
        elif name == "content":
            phrases.append("show a content pose")
        elif name == "pose_on":
            phrases.append("enable pose mode")
        elif name == "pose_off":
            phrases.append("disable pose mode")
        elif name == "dance1":
            phrases.append("perform dance one")
        elif name == "dance2":
            phrases.append("perform dance two")
        elif name == "static_walk":
            phrases.append("switch to static walk")
        elif name == "trot_run":
            phrases.append("switch to trot run")
        elif name == "walk_upright_on":
            phrases.append("enable upright walking")
        elif name == "walk_upright_off":
            phrases.append("disable upright walking")
        elif name == "classic_walk_on":
            phrases.append("enable classic walk")
        elif name == "classic_walk_off":
            phrases.append("disable classic walk")
        elif name == "switch_avoid_mode":
            phrases.append("switch avoid mode")
        elif name == "speed_level":
            phrases.append(f"set speed level to {int(args.get('level', 1) or 1)}")
        elif name == "damp":
            phrases.append("enter damp mode")
        elif name == "free_walk":
            phrases.append("walk freely")
        elif name == "sit":
            phrases.append("sit down")
        elif name == "rise_sit":
            phrases.append("rise from sitting")

    if not phrases:
        return "hold position"
    if len(phrases) == 1:
        return phrases[0]
    return f"{phrases[0]}, then {phrases[1]}"


def _summarize_move(args: Dict[str, Any]) -> str:
    vx = float(args.get("vx", 0.0) or 0.0)
    vy = float(args.get("vy", 0.0) or 0.0)
    vyaw = float(args.get("vyaw", 0.0) or 0.0)

    parts = []
    if abs(vx) >= 0.05:
        parts.append("move forward" if vx > 0 else "move backward")
    if abs(vy) >= 0.05:
        parts.append("shift left" if vy > 0 else "shift right")
    if abs(vyaw) >= 0.05:
        parts.append("turn left" if vyaw > 0 else "turn right")

    if not parts:
        return "make a small adjustment in place"
    if len(parts) == 1:
        return parts[0]
    return ", and ".join(parts)
