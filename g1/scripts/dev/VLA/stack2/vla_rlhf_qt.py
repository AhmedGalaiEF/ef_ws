#!/usr/bin/env python3
"""
vla_rlhf_qt.py  —  G1 VLA with RLHF feedback loop

Flow
====
1. User enters a task prompt.
2. Planner LLM (Ollama or Anthropic) generates a JSON step-by-step plan.
3. Each step is executed via ef_client.Robot (or simulated in dry-run mode).
4. After each step the user rates it: "Wrong Action" | "Correct Action".
5. After all steps the user rates the outcome: "Task Incomplete" | "Task Complete".
6. Per-session feedback is logged to rlhf_log.jsonl beside this file.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError as exc:
    raise SystemExit("PySide6 is required.  Install with: pip install PySide6") from exc

# ---------------------------------------------------------------------------
# Path setup — makes ef_client importable
# ---------------------------------------------------------------------------

_THIS = Path(__file__).resolve()
_DEV_DIR = _THIS.parents[2]          # .../scripts/dev
_SCRIPTS_ROOT = _DEV_DIR.parent      # .../scripts

for _p in (str(_DEV_DIR), str(_SCRIPTS_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RLHF_LOG_PATH = _THIS.parent / "rlhf_log.jsonl"

# ---------------------------------------------------------------------------
# Planner system prompt
# ---------------------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = """\
You are an autonomous robot task planner for a Unitree G1 humanoid robot.

AVAILABLE ACTIONS
=================

Locomotion
----------
  balanced_stand   Stand in balanced posture.
                   Params: {mode: 0}
  stop             Stop all motion immediately.
  sleep            Pause execution.
                   Params: {duration_s: float}
  walk             Continuous walking velocity command.
                   Params: {vx: float, vy: float, vyaw: float}  (m/s, rad/s)
  run              Continuous running velocity command.
                   Params: {vx: float, vy: float, vyaw: float}
  walk_for         Walk a precise distance with feedback control.
                   Params: {distance: float (m), max_vx: float, timeout: float}
  run_for          Run a precise distance with feedback control.
                   Params: {distance: float (m), max_vx: float, timeout: float}
  turn_for         Turn in-place by angle (positive = CCW).
                   Params: {angle_deg: float, max_vyaw: float, timeout: float}

Upper Body / Dexterity
-----------------------
  rotate_joint     Move a named joint to a target angle.
                   Params: {joint_name: str, angle_deg: float,
                            arm: "left"|"right" (for bilateral joints),
                            duration: float (s), hold: float (s)}
                   Joint names: shoulder_pitch, shoulder_roll, shoulder_yaw,
                                elbow, wrist_pitch, wrist_roll, wrist_yaw,
                                waist_yaw

Navigation / SLAM
-----------------
  slam_start       Begin SLAM mapping.
                   Params: {save_folder: "./maps"}
  slam_stop        Stop SLAM and save map.
                   Params: {save_folder: "./maps"}
  slam_nav_pose    Navigate to an absolute (x, y, yaw) pose.
                   Params: {x: float, y: float, yaw: float, obs_avoid: bool}
  slam_nav_path    Follow a list of waypoints.
                   Params: {points: [[x,y] or [x,y,yaw], ...], obs_avoid: bool}

Sensing
-------
  get_state        Read current IMU, position, gait mode, and sensor timestamps.

Communication
-------------
  say              Speak text through the robot's speaker via TTS.
                   Params: {text: "..."}
  headlight        Set headlight colour and intensity.
                   Params: {args: {color: "white"|"red"|"green"|"blue"|"yellow"|"off",
                                   intensity: 0–100}}

PLANNING RULES
==============
1. Return ONLY a valid JSON array — no markdown, no commentary, no extra text.
2. Each element is an action object with a required "type" key plus relevant params.
3. Add a "description" string key to every step for human readability.
4. Always begin locomotion plans with {"type": "balanced_stand"}.
5. Prefer short, safe plans: 2–8 steps unless the task clearly requires more.
6. Use conservative parameters (slow speeds, short distances, generous timeouts).
7. If the task calls for acknowledgement, use "say" at the start or end.
8. If lighting feedback is appropriate, change headlight colour to signal state.

OUTPUT FORMAT EXAMPLE
=====================
[
  {"type": "balanced_stand", "description": "Adopt balanced posture"},
  {"type": "headlight", "args": {"color": "green", "intensity": 80},
   "description": "Signal task start with green light"},
  {"type": "say", "text": "Starting task", "description": "Announce start"},
  {"type": "walk_for", "distance": 0.5, "timeout": 12,
   "description": "Walk forward 0.5 m"},
  {"type": "headlight", "args": {"color": "white", "intensity": 60},
   "description": "Restore default lighting"},
  {"type": "stop", "description": "Halt and await feedback"}
]
"""

# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _call_ollama(base_url: str, model: str, system: str, user: str,
                 timeout: float = 120.0) -> str:
    import urllib.request
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 1024, "num_ctx": 2048},
    }
    data = json.dumps(payload).encode()
    url = base_url.rstrip("/") + "/api/chat"
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode())
    return body["message"]["content"]


def _call_anthropic(api_key: str, model: str, system: str, user: str,
                    timeout: float = 120.0) -> str:
    import anthropic  # pip install anthropic
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text


def _call_llm(backend: str, model: str, system: str, user: str,
              ollama_url: str, anthropic_key: str, timeout: float = 120.0) -> str:
    if backend == "ollama":
        return _call_ollama(ollama_url, model, system, user, timeout)
    if backend == "anthropic":
        return _call_anthropic(anthropic_key, model, system, user, timeout)
    raise ValueError(f"Unknown LLM backend: {backend!r}")


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    """Pull the first JSON array out of raw model output."""
    import re
    text = text.strip()
    # Fenced code block
    m = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text, re.IGNORECASE)
    if m:
        return json.loads(m.group(1))
    # Bare array
    start = text.find("[")
    if start < 0:
        raise ValueError("No JSON array found in model output")
    depth = 0
    end = -1
    for i, ch in enumerate(text[start:], start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end < 0:
        raise ValueError("Unbalanced JSON brackets in model output")
    return json.loads(text[start : end + 1])


# ---------------------------------------------------------------------------
# Action dispatcher — thin wrapper around ef_client.Robot
# ---------------------------------------------------------------------------

def _dispatch_action(robot: Any, action: dict[str, Any]) -> Any:
    t = str(action.get("type", "")).strip().lower()

    if t == "balanced_stand":
        robot.balanced_stand(mode=int(action.get("mode", 0)))
        return {"mode": action.get("mode", 0)}
    if t == "stop":
        robot.stop()
        return None
    if t == "sleep":
        d = max(0.0, float(action.get("duration_s", 0.5)))
        time.sleep(d)
        return {"slept_s": d}
    if t == "walk":
        return int(robot.walk(
            vx=float(action.get("vx", 0.0)),
            vy=float(action.get("vy", 0.0)),
            vyaw=float(action.get("vyaw", 0.0)),
        ))
    if t == "run":
        return int(robot.run(
            vx=float(action.get("vx", 0.0)),
            vy=float(action.get("vy", 0.0)),
            vyaw=float(action.get("vyaw", 0.0)),
        ))
    if t == "walk_for":
        return bool(robot.walk_for(
            distance=float(action["distance"]),
            max_vx=float(action.get("max_vx", 0.25)),
            timeout=float(action.get("timeout", 20.0)),
        ))
    if t == "run_for":
        return bool(robot.run_for(
            distance=float(action["distance"]),
            max_vx=float(action.get("max_vx", 0.45)),
            timeout=float(action.get("timeout", 15.0)),
        ))
    if t == "turn_for":
        return bool(robot.turn_for(
            angle_deg=float(action["angle_deg"]),
            max_vyaw=float(action.get("max_vyaw", 0.8)),
            timeout=float(action.get("timeout", 10.0)),
        ))
    if t == "rotate_joint":
        return int(robot.rotate_joint(
            joint_name=str(action["joint_name"]),
            angle_deg=float(action["angle_deg"]),
            arm=action.get("arm"),
            duration=float(action.get("duration", 1.0)),
            hold=float(action.get("hold", 0.0)),
        ))
    if t == "slam_start":
        proc = robot.start_slam(save_folder=str(action.get("save_folder", "./maps")))
        return {"pid": int(proc.pid)}
    if t == "slam_stop":
        robot.stop_slam(save_folder=str(action.get("save_folder", "./maps")))
        return None
    if t == "slam_nav_pose":
        rc = int(robot.slam_nav_pose(
            x=float(action["x"]),
            y=float(action["y"]),
            yaw=float(action.get("yaw", 0.0)),
            obs_avoid=bool(action.get("obs_avoid", False)),
        ))
        return {"rc": rc, "ok": rc == 0}
    if t == "slam_nav_path":
        ok = bool(robot.slam_nav_path(
            points=action["points"],
            obs_avoid=bool(action.get("obs_avoid", True)),
        ))
        return {"ok": ok}
    if t == "say":
        robot.say(str(action["text"]))
        return {"text": action["text"]}
    if t == "headlight":
        rc = int(robot.headlight(
            args=action.get("args"),
            duration=action.get("duration"),
        ))
        return {"rc": rc}
    if t == "get_state":
        return robot.get_robot_state()

    raise ValueError(f"Unsupported action type: {t!r}")


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    index: int
    action: dict[str, Any]
    ok: bool
    return_value: Any
    error: str | None
    duration_s: float
    rlhf_correct: bool | None = None   # set after user rates the step


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class PlanWorker(QtCore.QObject):
    status_changed    = QtCore.Signal(str)
    plan_generated    = QtCore.Signal(list)         # list[dict] — the plan
    step_started      = QtCore.Signal(int, dict)    # (idx, action)
    step_finished     = QtCore.Signal(int, dict, object, str)  # (idx, action, result, error)
    rlhf_step_needed  = QtCore.Signal(int, dict)   # block until submit_step_feedback
    rlhf_task_needed  = QtCore.Signal()             # block until submit_task_feedback
    finished          = QtCore.Signal(list)         # list[dict] — serialised results

    def __init__(
        self,
        backend: str,
        model: str,
        ollama_url: str,
        anthropic_key: str,
        iface: str,
        domain_id: int,
        dry_run: bool,
        system_prompt: str,
        user_prompt: str,
    ) -> None:
        super().__init__()
        self._backend       = backend
        self._model         = model
        self._ollama_url    = ollama_url
        self._anthropic_key = anthropic_key
        self._iface         = iface
        self._domain_id     = domain_id
        self._dry_run       = dry_run
        self._system_prompt = system_prompt
        self._user_prompt   = user_prompt

        self._stop_ev        = threading.Event()
        self._step_fb_ev     = threading.Event()
        self._step_fb_value: bool = True
        self._task_fb_ev     = threading.Event()
        self._task_complete: bool = False
        self._results: list[StepResult] = []

    # ---- main entry point (called from worker thread) ----

    @QtCore.Slot()
    def run(self) -> None:
        try:
            self._run_impl()
        except Exception as exc:
            self.status_changed.emit(f"Error: {exc}")
        finally:
            self.finished.emit([
                {
                    "index":       r.index,
                    "type":        r.action.get("type"),
                    "description": r.action.get("description", ""),
                    "ok":          r.ok,
                    "rlhf_correct": r.rlhf_correct,
                    "duration_s":  round(r.duration_s, 3),
                    "error":       r.error,
                }
                for r in self._results
            ])

    def _run_impl(self) -> None:
        # 1. Generate plan
        self.status_changed.emit("Generating plan…")
        raw = _call_llm(
            backend      = self._backend,
            model        = self._model,
            system       = self._system_prompt,
            user         = self._user_prompt,
            ollama_url   = self._ollama_url,
            anthropic_key= self._anthropic_key,
        )
        try:
            plan: list[dict[str, Any]] = _extract_json_array(raw)
        except Exception as exc:
            raise RuntimeError(
                f"Planner returned invalid JSON: {exc}\n\nRaw output:\n{raw}"
            ) from exc

        self.plan_generated.emit(plan)
        self.status_changed.emit(f"Plan ready ({len(plan)} step(s)). Executing…")

        # 2. Initialise robot
        robot = None
        if not self._dry_run:
            from ef_client import Robot  # noqa: PLC0415
            robot = Robot(
                iface=self._iface,
                domain_id=self._domain_id,
                safety_boot=True,
                auto_start_sensors=True,
            )

        # 3. Execute each step, collect per-step RLHF
        try:
            for idx, action in enumerate(plan):
                if self._stop_ev.is_set():
                    break

                self.step_started.emit(idx, action)
                self.status_changed.emit(
                    f"Step {idx + 1}/{len(plan)}: {action.get('type', '?')} — "
                    f"{action.get('description', '')}"
                )

                t0 = time.time()
                return_value = None
                error: str | None = None
                ok = True

                if self._dry_run:
                    time.sleep(0.4)
                    return_value = {"dry_run": True}
                else:
                    try:
                        return_value = _dispatch_action(robot, action)
                    except Exception as exc:
                        ok = False
                        error = str(exc)

                duration_s = time.time() - t0

                self.step_finished.emit(idx, action, return_value, error or "")
                self.status_changed.emit(
                    f"Step {idx + 1} finished. Waiting for RLHF feedback…"
                )

                # Block until user rates this step
                self._step_fb_ev.clear()
                self.rlhf_step_needed.emit(idx, action)
                self._step_fb_ev.wait()

                self._results.append(StepResult(
                    index=idx,
                    action=action,
                    ok=ok,
                    return_value=return_value,
                    error=error,
                    duration_s=duration_s,
                    rlhf_correct=self._step_fb_value,
                ))

                if self._stop_ev.is_set():
                    break
        finally:
            if robot is not None:
                try:
                    robot.stop()
                except Exception:
                    pass

        # 4. Task-level RLHF
        self.status_changed.emit("All steps done. Rate the overall task outcome…")
        self._task_fb_ev.clear()
        self.rlhf_task_needed.emit()
        self._task_fb_ev.wait()

        # 5. Persist feedback
        self._save_log()
        outcome = "Task Complete" if self._task_complete else "Task Incomplete"
        self.status_changed.emit(f"Done — {outcome}. Feedback saved.")

    def _save_log(self) -> None:
        entry = {
            "ts":            time.time(),
            "prompt":        self._user_prompt,
            "task_complete": self._task_complete,
            "steps": [
                {
                    "index":        r.index,
                    "type":         r.action.get("type"),
                    "description":  r.action.get("description", ""),
                    "ok":           r.ok,
                    "rlhf_correct": r.rlhf_correct,
                    "duration_s":   round(r.duration_s, 3),
                    "error":        r.error,
                }
                for r in self._results
            ],
        }
        RLHF_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with RLHF_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=True) + "\n")

    # ---- slots called from the main thread ----

    @QtCore.Slot(bool)
    def submit_step_feedback(self, correct: bool) -> None:
        self._step_fb_value = correct
        self._step_fb_ev.set()

    @QtCore.Slot(bool)
    def submit_task_feedback(self, complete: bool) -> None:
        self._task_complete = complete
        self._task_fb_ev.set()

    @QtCore.Slot()
    def stop(self) -> None:
        self._stop_ev.set()
        self._step_fb_ev.set()
        self._task_fb_ev.set()


# ---------------------------------------------------------------------------
# Per-step RLHF widget
# ---------------------------------------------------------------------------

class StepFeedbackWidget(QtWidgets.QFrame):
    correct_clicked = QtCore.Signal()
    wrong_clicked   = QtCore.Signal()

    _STYLE_CORRECT  = "background:#2a6b2a;color:white;font-weight:bold;padding:4px 12px;"
    _STYLE_WRONG    = "background:#6b2a2a;color:white;font-weight:bold;padding:4px 12px;"
    _STYLE_INACTIVE = "color:grey;padding:4px 12px;"

    def __init__(self, step_index: int, action: dict[str, Any], parent=None) -> None:
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setLineWidth(1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        desc = action.get("description") or action.get("type", "?")
        action_type = action.get("type", "?")
        label = QtWidgets.QLabel(
            f"<b>Step {step_index + 1}</b> &nbsp;<code>{action_type}</code> — {desc}", self
        )
        label.setWordWrap(True)
        layout.addWidget(label)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(6)
        self._correct_btn = QtWidgets.QPushButton("Correct Action", self)
        self._wrong_btn   = QtWidgets.QPushButton("Wrong Action",   self)
        self._correct_btn.setStyleSheet(self._STYLE_CORRECT)
        self._wrong_btn.setStyleSheet(self._STYLE_WRONG)
        self._correct_btn.clicked.connect(self.correct_clicked)
        self._wrong_btn.clicked.connect(self.wrong_clicked)
        btn_row.addWidget(self._correct_btn)
        btn_row.addWidget(self._wrong_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self._result_label = QtWidgets.QLabel("", self)
        layout.addWidget(self._result_label)

    def lock(self, correct: bool) -> None:
        """Disable buttons and show the recorded choice."""
        self._correct_btn.setEnabled(False)
        self._wrong_btn.setEnabled(False)
        if correct:
            self._correct_btn.setStyleSheet(self._STYLE_CORRECT)
            self._wrong_btn.setStyleSheet(self._STYLE_INACTIVE)
            self._result_label.setText("Rated: Correct ✓")
        else:
            self._wrong_btn.setStyleSheet(self._STYLE_WRONG)
            self._correct_btn.setStyleSheet(self._STYLE_INACTIVE)
            self._result_label.setText("Rated: Wrong ✗")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _list_ollama_models() -> list[str]:
    import subprocess
    try:
        r = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        lines = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
        return [ln.split()[0] for ln in lines[1:] if ln.split()]
    except Exception:
        return []


ANTHROPIC_MODELS = [
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
]


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("G1 VLA — RLHF Feedback")
        self.resize(1420, 900)

        self._worker: PlanWorker | None = None
        self._worker_thread: QtCore.QThread | None = None
        self._active_step_idx: int = -1
        self._step_widgets: list[StepFeedbackWidget] = []

        self._build_ui()
        QtWidgets.QApplication.instance().aboutToQuit.connect(self._shutdown)

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setSpacing(10)

        # ---- LEFT panel (configuration + prompt) ----
        left = QtWidgets.QVBoxLayout()
        left.setSpacing(6)
        root.addLayout(left, 0)

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignRight)
        form.setHorizontalSpacing(8)

        self.iface_edit = QtWidgets.QLineEdit("eth0", self)
        self.domain_spin = QtWidgets.QSpinBox(self)
        self.domain_spin.setRange(0, 255)
        self.dry_run_cb = QtWidgets.QCheckBox("Dry Run (no hardware)", self)
        self.dry_run_cb.setChecked(True)
        form.addRow("Interface",  self.iface_edit)
        form.addRow("Domain ID",  self.domain_spin)
        form.addRow("",           self.dry_run_cb)

        sep = QtWidgets.QFrame(self)
        sep.setFrameShape(QtWidgets.QFrame.HLine)

        self.backend_combo = QtWidgets.QComboBox(self)
        self.backend_combo.addItems(["ollama", "anthropic"])
        self.backend_combo.currentTextChanged.connect(self._on_backend_changed)
        form.addRow("Backend", self.backend_combo)

        self.model_combo = QtWidgets.QComboBox(self)
        self.model_combo.setEditable(True)
        self.model_combo.setMinimumWidth(180)
        form.addRow("Model", self.model_combo)

        self.ollama_url_edit = QtWidgets.QLineEdit("http://127.0.0.1:11434", self)
        form.addRow("Ollama URL", self.ollama_url_edit)

        self.anthropic_key_edit = QtWidgets.QLineEdit(
            os.environ.get("ANTHROPIC_API_KEY", ""), self
        )
        self.anthropic_key_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.anthropic_key_edit.setPlaceholderText("sk-ant-…")
        form.addRow("Anthropic Key", self.anthropic_key_edit)

        refresh_btn = QtWidgets.QPushButton("Refresh Ollama Models", self)
        refresh_btn.clicked.connect(self._refresh_models)
        form.addRow("", refresh_btn)

        left.addLayout(form)
        left.addWidget(sep)

        left.addWidget(QtWidgets.QLabel("System Prompt", self))
        self.system_prompt_edit = QtWidgets.QPlainTextEdit(self)
        self.system_prompt_edit.setPlainText(PLANNER_SYSTEM_PROMPT)
        self.system_prompt_edit.setMaximumHeight(160)
        left.addWidget(self.system_prompt_edit)

        left.addWidget(QtWidgets.QLabel("Task", self))
        self.task_edit = QtWidgets.QPlainTextEdit(self)
        self.task_edit.setPlaceholderText(
            "Describe what the robot should do…\n\n"
            "Examples:\n"
            "  • Walk forward 1 meter, then turn right 90 degrees.\n"
            "  • Wave your right arm and say hello.\n"
            "  • Flash the headlight blue and walk in a small circle."
        )
        self.task_edit.setMaximumHeight(120)
        left.addWidget(self.task_edit)

        btn_row = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Plan & Execute", self)
        self.start_btn.setStyleSheet("font-weight:bold;padding:6px 16px;")
        self.stop_btn = QtWidgets.QPushButton("Stop", self)
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        left.addLayout(btn_row)

        self.status_label = QtWidgets.QLabel("Idle", self)
        self.status_label.setWordWrap(True)
        left.addWidget(self.status_label)
        left.addStretch(1)

        # ---- RIGHT panel (tabs) ----
        tabs = QtWidgets.QTabWidget(self)
        root.addWidget(tabs, 1)

        # Tab 0 — Plan (raw JSON)
        plan_widget = QtWidgets.QWidget()
        pl = QtWidgets.QVBoxLayout(plan_widget)
        self.plan_view = QtWidgets.QPlainTextEdit(self)
        self.plan_view.setReadOnly(True)
        self.plan_view.setFont(QtGui.QFont("Monospace", 10))
        pl.addWidget(self.plan_view)
        tabs.addTab(plan_widget, "Plan JSON")

        # Tab 1 — Execution log
        exec_widget = QtWidgets.QWidget()
        el = QtWidgets.QVBoxLayout(exec_widget)
        self.exec_log = QtWidgets.QPlainTextEdit(self)
        self.exec_log.setReadOnly(True)
        self.exec_log.setFont(QtGui.QFont("Monospace", 10))
        el.addWidget(self.exec_log)
        tabs.addTab(exec_widget, "Execution Log")

        # Tab 2 — RLHF feedback
        rlhf_outer = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(rlhf_outer)
        rl.setContentsMargins(6, 6, 6, 6)
        rl.setSpacing(6)

        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        rl.addWidget(scroll, 1)

        self._rlhf_inner = QtWidgets.QWidget()
        self._rlhf_vbox = QtWidgets.QVBoxLayout(self._rlhf_inner)
        self._rlhf_vbox.setSpacing(6)
        self._rlhf_vbox.addStretch(1)   # stretch pushed to the bottom
        scroll.setWidget(self._rlhf_inner)

        # Task-level outcome panel (hidden until all steps done)
        self._task_panel = QtWidgets.QFrame(rlhf_outer)
        self._task_panel.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self._task_panel.setVisible(False)
        tp = QtWidgets.QVBoxLayout(self._task_panel)
        tp.addWidget(QtWidgets.QLabel(
            "<b>All steps completed. Was the overall task successful?</b>", self._task_panel
        ))
        task_btn_row = QtWidgets.QHBoxLayout()
        self._task_complete_btn   = QtWidgets.QPushButton("Task Complete",   self._task_panel)
        self._task_incomplete_btn = QtWidgets.QPushButton("Task Incomplete", self._task_panel)
        self._task_complete_btn.setStyleSheet(
            "background:#2a6b2a;color:white;font-weight:bold;padding:6px 20px;"
        )
        self._task_incomplete_btn.setStyleSheet(
            "background:#6b2a2a;color:white;font-weight:bold;padding:6px 20px;"
        )
        self._task_complete_btn.clicked.connect(lambda: self._submit_task_feedback(True))
        self._task_incomplete_btn.clicked.connect(lambda: self._submit_task_feedback(False))
        task_btn_row.addWidget(self._task_complete_btn)
        task_btn_row.addWidget(self._task_incomplete_btn)
        task_btn_row.addStretch(1)
        tp.addLayout(task_btn_row)
        rl.addWidget(self._task_panel)

        tabs.addTab(rlhf_outer, "RLHF Feedback")
        tabs.setCurrentIndex(2)   # open on the feedback tab by default

        # Populate model list
        self._on_backend_changed("ollama")

    # ---------------------------------------------------------------- slots

    def _on_backend_changed(self, backend: str) -> None:
        self.model_combo.clear()
        if backend == "ollama":
            models = _list_ollama_models()
            self.model_combo.addItems(models or ["llama3.2"])
        else:
            self.model_combo.addItems(ANTHROPIC_MODELS)

    def _refresh_models(self) -> None:
        self._on_backend_changed(self.backend_combo.currentText())

    def _start(self) -> None:
        if self._worker_thread is not None:
            return
        task = self.task_edit.toPlainText().strip()
        if not task:
            QtWidgets.QMessageBox.warning(self, "No Task", "Enter a task prompt first.")
            return
        model = self.model_combo.currentText().strip()
        if not model:
            QtWidgets.QMessageBox.warning(self, "No Model", "Select or enter a model name.")
            return

        # Clear previous session
        for w in self._step_widgets:
            w.setParent(None)
        self._step_widgets.clear()
        self._task_panel.setVisible(False)
        self._active_step_idx = -1
        self.exec_log.clear()
        self.plan_view.clear()

        self._worker = PlanWorker(
            backend       = self.backend_combo.currentText(),
            model         = model,
            ollama_url    = self.ollama_url_edit.text().strip(),
            anthropic_key = self.anthropic_key_edit.text().strip(),
            iface         = self.iface_edit.text().strip(),
            domain_id     = int(self.domain_spin.value()),
            dry_run       = self.dry_run_cb.isChecked(),
            system_prompt = (
                self.system_prompt_edit.toPlainText().strip()
                or PLANNER_SYSTEM_PROMPT
            ),
            user_prompt   = task,
        )

        self._worker_thread = QtCore.QThread(self)
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.status_changed.connect(self._on_status)
        self._worker.plan_generated.connect(self._on_plan_generated)
        self._worker.step_started.connect(self._on_step_started)
        self._worker.step_finished.connect(self._on_step_finished)
        self._worker.rlhf_step_needed.connect(self._on_rlhf_step_needed)
        self._worker.rlhf_task_needed.connect(self._on_rlhf_task_needed)
        self._worker.finished.connect(self._on_finished)
        self._worker_thread.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def _stop(self) -> None:
        if self._worker:
            self._worker.stop()
        self.status_label.setText("Stopping…")

    @QtCore.Slot(str)
    def _on_status(self, msg: str) -> None:
        self.status_label.setText(msg)

    @QtCore.Slot(list)
    def _on_plan_generated(self, plan: list) -> None:
        self.plan_view.setPlainText(json.dumps(plan, indent=2))
        self._log(f"Plan generated — {len(plan)} step(s)\n")

    @QtCore.Slot(int, dict)
    def _on_step_started(self, idx: int, action: dict) -> None:
        desc = action.get("description") or action.get("type", "?")
        self._log(f"→ Step {idx + 1}  [{action.get('type', '?')}]  {desc}")

    @QtCore.Slot(int, dict, object, str)
    def _on_step_finished(self, idx: int, action: dict, result: Any, error: str) -> None:
        if error:
            self._log(f"  ✗ Error: {error}")
        else:
            self._log(f"  ✓ Result: {json.dumps(result)}")

    @QtCore.Slot(int, dict)
    def _on_rlhf_step_needed(self, idx: int, action: dict) -> None:
        self._active_step_idx = idx
        widget = StepFeedbackWidget(idx, action, self._rlhf_inner)
        widget.correct_clicked.connect(lambda: self._submit_step_feedback(True))
        widget.wrong_clicked.connect(lambda: self._submit_step_feedback(False))
        # Insert before the bottom stretch
        self._rlhf_vbox.insertWidget(len(self._step_widgets), widget)
        self._step_widgets.append(widget)

    def _submit_step_feedback(self, correct: bool) -> None:
        idx = self._active_step_idx
        if idx < 0 or idx >= len(self._step_widgets):
            return
        self._step_widgets[idx].lock(correct)
        label = "Correct ✓" if correct else "Wrong ✗"
        self._log(f"  RLHF step {idx + 1}: {label}")
        self._active_step_idx = -1
        if self._worker:
            self._worker.submit_step_feedback(correct)

    @QtCore.Slot()
    def _on_rlhf_task_needed(self) -> None:
        self._task_panel.setVisible(True)

    def _submit_task_feedback(self, complete: bool) -> None:
        self._task_panel.setVisible(False)
        label = "Task Complete ✓" if complete else "Task Incomplete ✗"
        self._log(f"\nTask outcome: {label}")
        if self._worker:
            self._worker.submit_task_feedback(complete)

    @QtCore.Slot(list)
    def _on_finished(self, results: list) -> None:
        if self._worker_thread:
            self._worker_thread.quit()
            self._worker_thread.wait(3000)
        self._worker_thread = None
        self._worker = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    # ---------------------------------------------------------------- misc

    def _log(self, line: str) -> None:
        cursor = self.exec_log.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(line + "\n")
        self.exec_log.setTextCursor(cursor)
        self.exec_log.ensureCursorVisible()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._shutdown()
        super().closeEvent(event)

    def _shutdown(self) -> None:
        if self._worker:
            self._worker.stop()
        if self._worker_thread:
            self._worker_thread.quit()
            self._worker_thread.wait(4000)
        self._worker_thread = None
        self._worker = None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
