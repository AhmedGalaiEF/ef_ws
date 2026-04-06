#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import cv2
import numpy as np

from hf_vla.hf_client import HuggingFaceChatClient
from intent_summary import build_intent_statement
from local_speech import LocalSpeechAnnouncer
from openai_vla.openai_client import OpenAIChatClient
from ollama_vla.agents import ActorAgent, PerceptionAgent, PerceptionWorker, PlannerAgent, VLAController
from ollama_vla.config import DEFAULT_PLANNER_SYSTEM_PROMPT, RuntimeConfig
from ollama_vla.ollama_client import OllamaChatClient
from ollama_vla.sport_actor import SportCommandExecutor
from ollama_vla.video_source import Go2VideoSource
from sdk_safety import available_interfaces, init_channel_autodetect

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except Exception as exc:  # noqa: BLE001
    raise SystemExit(
        "PySide6 is required to run this app. Install it before launching vla_control_qt.py."
    ) from exc


HF_MODELS = [
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "meta-llama/Llama-3.2-90B-Vision-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-2-2b-it",
    "CohereLabs/aya-vision-32b:cohere",
    "zai-org/GLM-4.5V",
]

CODEX_MODELS = [
    "gpt-5.2-codex",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o-mini",
]


@dataclass
class ModelChoice:
    backend: str
    model: str

    def label(self) -> str:
        return f"{self.backend}:{self.model}"


@dataclass
class ControlConfig:
    iface: str
    dry_run: bool
    planner_choice: ModelChoice
    perception_choice: ModelChoice
    planner_period_sec: float
    perception_period_sec: float
    planner_prompt: str
    ask_prompt_each_step: bool
    say_intent: bool
    ollama_url: str
    hf_api_url: str
    hf_token: str
    openai_api_url: str
    openai_api_key: str


def list_ollama_models() -> List[str]:
    try:
        result = subprocess.run(
            ["ollama", "list"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) <= 1:
        return []

    models: List[str] = []
    for line in lines[1:]:
        parts = line.split()
        if parts:
            models.append(parts[0])
    return models


def make_client(
    choice: ModelChoice,
    purpose: str,
    ollama_url: str,
    hf_api_url: str,
    hf_token: str,
    openai_api_url: str,
    openai_api_key: str,
):
    if choice.backend == "ollama":
        options = {
            "temperature": 0.1,
            "num_predict": 96 if purpose == "planner" else 160,
            "num_ctx": 1024 if purpose == "planner" else 1536,
        }
        return OllamaChatClient(
            base_url=ollama_url,
            model=choice.model,
            timeout_sec=90.0,
            default_options=options,
            keep_alive="10m",
        )

    if choice.backend == "huggingface":
        return HuggingFaceChatClient(
            api_url=hf_api_url,
            model=choice.model,
            api_token=hf_token,
            timeout_sec=90.0,
            temperature=0.1,
        )

    return OpenAIChatClient(
        api_url=openai_api_url,
        model=choice.model,
        api_key=openai_api_key,
        timeout_sec=90.0,
        temperature=0.1,
    )


class PromptDialog(QtWidgets.QDialog):
    def __init__(self, current_prompt: str, placeholder_prompt: str, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Planner Prompt")
        self.resize(800, 500)

        layout = QtWidgets.QVBoxLayout(self)
        self.editor = QtWidgets.QPlainTextEdit(self)
        self.editor.setPlaceholderText(placeholder_prompt)
        self.editor.setPlainText(current_prompt)
        layout.addWidget(self.editor)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def prompt_text(self) -> str:
        return self.editor.toPlainText().strip()


class ControlWorker(QtCore.QObject):
    outputs_ready = QtCore.Signal(dict)
    status_message = QtCore.Signal(str)
    prompt_requested = QtCore.Signal(str, str)
    finished = QtCore.Signal()

    def __init__(self, config: ControlConfig, video_source: Go2VideoSource):
        super().__init__()
        self._config = config
        self._video_source = video_source
        self._stop = threading.Event()
        self._prompt_event = threading.Event()
        self._prompt_value = config.planner_prompt
        self._say_intent = config.say_intent
        self._announcer = LocalSpeechAnnouncer(enabled=config.say_intent)

    @QtCore.Slot()
    def run(self) -> None:
        runtime = RuntimeConfig(
            iface=self._config.iface,
            perception_period_sec=self._config.perception_period_sec,
            planner_period_sec=self._config.planner_period_sec,
            dry_run=self._config.dry_run,
        )
        runtime.planner_system_prompt = self._config.planner_prompt

        planner_client = make_client(
            self._config.planner_choice,
            "planner",
            self._config.ollama_url,
            self._config.hf_api_url,
            self._config.hf_token,
            self._config.openai_api_url,
            self._config.openai_api_key,
        )
        perception_client = make_client(
            self._config.perception_choice,
            "perception",
            self._config.ollama_url,
            self._config.hf_api_url,
            self._config.hf_token,
            self._config.openai_api_url,
            self._config.openai_api_key,
        )

        perception_agent = PerceptionAgent(perception_client, runtime.perception_system_prompt)
        planner_agent = PlannerAgent(planner_client, runtime.planner_system_prompt)
        actor_agent = ActorAgent(planner_client, runtime.actor_system_prompt, runtime)
        perception_worker = PerceptionWorker(
            video_source=self._video_source,
            agent=perception_agent,
            period_sec=runtime.perception_period_sec,
            initial_prompt=runtime.initial_perception_prompt,
        )
        executor = SportCommandExecutor(timeout_sec=runtime.sport_timeout_sec, dry_run=runtime.dry_run)
        executor.start()
        controller = VLAController(planner=planner_agent, actor=actor_agent, perception_worker=perception_worker)
        perception_worker.start()

        try:
            while not self._stop.is_set():
                if self._config.ask_prompt_each_step:
                    self._prompt_event.clear()
                    self.prompt_requested.emit(self._prompt_value, DEFAULT_PLANNER_SYSTEM_PROMPT)
                    self._prompt_event.wait()
                    if self._stop.is_set():
                        break
                    planner_agent.set_system_prompt(self._prompt_value)

                step = controller.step()
                if self._say_intent:
                    self._announcer.announce(
                        build_intent_statement(step.planner_output, step.actor_output)
                    )
                executed = executor.execute_many(step.actor_output.get("commands", []))
                self.outputs_ready.emit(
                    {
                        "perception": step.perception.data if step.perception else None,
                        "perception_error": step.perception_error,
                        "video_error": self._video_source.latest_error(),
                        "planner_output": step.planner_output,
                        "actor_output": step.actor_output,
                        "executed": [e.__dict__ for e in executed],
                    }
                )
                self._wait(runtime.planner_period_sec)
        except Exception as exc:  # noqa: BLE001
            self.status_message.emit(str(exc))
        finally:
            self._stop.set()
            perception_worker.stop()
            self._announcer.close()
            if not runtime.dry_run:
                try:
                    executor.execute({"name": "stop_move", "args": {}, "duration_sec": 0.0})
                except Exception:
                    pass
            self.finished.emit()

    def _wait(self, duration_sec: float) -> None:
        deadline = time.time() + max(0.0, duration_sec)
        while not self._stop.is_set():
            remaining = deadline - time.time()
            if remaining <= 0.0:
                return
            self._stop.wait(min(0.05, remaining))

    @QtCore.Slot()
    def stop(self) -> None:
        self._stop.set()
        self._prompt_event.set()

    @QtCore.Slot(str)
    def set_planner_prompt(self, prompt: str) -> None:
        self._prompt_value = prompt or self._prompt_value
        self._prompt_event.set()

    @QtCore.Slot(bool)
    def set_say_intent(self, enabled: bool) -> None:
        self._say_intent = enabled


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VLA Control")
        self.resize(1500, 950)

        self._video_source: Go2VideoSource | None = None
        self._worker_thread: QtCore.QThread | None = None
        self._worker: ControlWorker | None = None
        self._last_preview_error = ""

        self._ollama_models = list_ollama_models()
        self._build_ui()
        self._ensure_video_source()
        self._camera_timer = QtCore.QTimer(self)
        self._camera_timer.timeout.connect(self._update_preview)
        self._camera_timer.start(33)
        QtWidgets.QApplication.instance().aboutToQuit.connect(self._shutdown)

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        controls = QtWidgets.QVBoxLayout()
        root.addLayout(controls, 0)

        self.iface_combo = QtWidgets.QComboBox(self)
        self.iface_combo.setEditable(True)
        for iface in available_interfaces():
            self.iface_combo.addItem(iface)
        default_iface = "enp0s31f6"
        index = self.iface_combo.findText(default_iface)
        if index >= 0:
            self.iface_combo.setCurrentIndex(index)
        else:
            self.iface_combo.setEditText(default_iface)
        self.iface_combo.currentTextChanged.connect(self._restart_video_source)
        self.dry_run_checkbox = QtWidgets.QCheckBox("Dry Run", self)
        self.dry_run_checkbox.setChecked(True)
        self.say_intent_toggle = QtWidgets.QPushButton("Say Intent", self)
        self.say_intent_toggle.setCheckable(True)
        self.say_intent_toggle.setChecked(True)
        self.ask_prompt_checkbox = QtWidgets.QCheckBox("Ask Before Each Planner Inference", self)
        self.ask_prompt_checkbox.setChecked(True)
        self.ollama_url_edit = QtWidgets.QLineEdit("http://127.0.0.1:11434", self)
        self.hf_url_edit = QtWidgets.QLineEdit("https://router.huggingface.co/v1/chat/completions", self)
        self.hf_token_edit = QtWidgets.QLineEdit(os.environ.get("HF_TOKEN", ""), self)
        self.hf_token_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.openai_url_edit = QtWidgets.QLineEdit("https://api.openai.com/v1/chat/completions", self)
        self.openai_key_edit = QtWidgets.QLineEdit(os.environ.get("OPENAI_API_KEY", ""), self)
        self.openai_key_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.perception_period_spin = QtWidgets.QDoubleSpinBox(self)
        self.perception_period_spin.setRange(0.2, 30.0)
        self.perception_period_spin.setValue(3.0)
        self.planner_period_spin = QtWidgets.QDoubleSpinBox(self)
        self.planner_period_spin.setRange(0.2, 30.0)
        self.planner_period_spin.setValue(4.0)

        controls.addWidget(QtWidgets.QLabel("Interface"))
        controls.addWidget(self.iface_combo)
        controls.addWidget(self.dry_run_checkbox)
        controls.addWidget(self.say_intent_toggle)
        controls.addWidget(self.ask_prompt_checkbox)
        controls.addWidget(QtWidgets.QLabel("Perception Period (s)"))
        controls.addWidget(self.perception_period_spin)
        controls.addWidget(QtWidgets.QLabel("Planner Period (s)"))
        controls.addWidget(self.planner_period_spin)
        controls.addWidget(QtWidgets.QLabel("Ollama URL"))
        controls.addWidget(self.ollama_url_edit)
        controls.addWidget(QtWidgets.QLabel("HF API URL"))
        controls.addWidget(self.hf_url_edit)
        controls.addWidget(QtWidgets.QLabel("HF Token"))
        controls.addWidget(self.hf_token_edit)
        controls.addWidget(QtWidgets.QLabel("OpenAI/Codex API URL"))
        controls.addWidget(self.openai_url_edit)
        controls.addWidget(QtWidgets.QLabel("OpenAI/Codex API Key"))
        controls.addWidget(self.openai_key_edit)

        self.refresh_models_button = QtWidgets.QPushButton("Refresh Ollama Models", self)
        self.refresh_models_button.clicked.connect(self._refresh_models)
        controls.addWidget(self.refresh_models_button)

        self.planner_backend_combo = QtWidgets.QComboBox(self)
        self.planner_backend_combo.addItems(["ollama", "huggingface", "codex"])
        self.planner_model_combo = QtWidgets.QComboBox(self)
        self.perception_backend_combo = QtWidgets.QComboBox(self)
        self.perception_backend_combo.addItems(["ollama", "huggingface", "codex"])
        self.perception_model_combo = QtWidgets.QComboBox(self)

        self.planner_backend_combo.currentTextChanged.connect(self._reload_model_combos)
        self.perception_backend_combo.currentTextChanged.connect(self._reload_model_combos)
        self._reload_model_combos()

        controls.addWidget(QtWidgets.QLabel("Planner Backend"))
        controls.addWidget(self.planner_backend_combo)
        controls.addWidget(QtWidgets.QLabel("Planner Model"))
        controls.addWidget(self.planner_model_combo)
        controls.addWidget(QtWidgets.QLabel("Perception Backend"))
        controls.addWidget(self.perception_backend_combo)
        controls.addWidget(QtWidgets.QLabel("Perception Model"))
        controls.addWidget(self.perception_model_combo)
        controls.addWidget(QtWidgets.QLabel("Planner Prompt"))
        self.prompt_editor = QtWidgets.QPlainTextEdit(self)
        self.prompt_editor.setPlaceholderText(DEFAULT_PLANNER_SYSTEM_PROMPT)
        self.prompt_editor.setPlainText(DEFAULT_PLANNER_SYSTEM_PROMPT)
        controls.addWidget(self.prompt_editor, 1)

        button_row = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("Start", self)
        self.stop_button = QtWidgets.QPushButton("Stop", self)
        self.stop_button.setEnabled(False)
        self.start_button.clicked.connect(self._start_control)
        self.stop_button.clicked.connect(self._stop_control)
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.stop_button)
        controls.addLayout(button_row)

        self.status_label = QtWidgets.QLabel("Idle", self)
        controls.addWidget(self.status_label)

        right = QtWidgets.QVBoxLayout()
        root.addLayout(right, 1)

        self.preview_label = QtWidgets.QLabel("Camera feed inactive", self)
        self.preview_label.setMinimumSize(960, 540)
        self.preview_label.setAlignment(QtCore.Qt.AlignCenter)
        self.preview_label.setStyleSheet("background:#111;color:#ddd;border:1px solid #444;")
        right.addWidget(self.preview_label, 2)

        tabs = QtWidgets.QTabWidget(self)
        right.addWidget(tabs, 1)

        self.perception_view = self._make_output_view()
        self.planner_view = self._make_output_view()
        self.actor_view = self._make_output_view()
        self.executed_view = self._make_output_view()
        self.log_view = self._make_output_view()
        tabs.addTab(self.perception_view, "Perception")
        tabs.addTab(self.planner_view, "Planner")
        tabs.addTab(self.actor_view, "Actor")
        tabs.addTab(self.executed_view, "Executed")
        tabs.addTab(self.log_view, "Logs")

    def _make_output_view(self) -> QtWidgets.QPlainTextEdit:
        widget = QtWidgets.QPlainTextEdit(self)
        widget.setReadOnly(True)
        return widget

    def _refresh_models(self) -> None:
        self._ollama_models = list_ollama_models()
        self._reload_model_combos()

    def _reload_model_combos(self) -> None:
        self._populate_model_combo(self.planner_model_combo, self.planner_backend_combo.currentText(), "qwen3.5:0.8b")
        self._populate_model_combo(self.perception_model_combo, self.perception_backend_combo.currentText(), "qwen3-vl:2b")

    def _populate_model_combo(self, combo: QtWidgets.QComboBox, backend: str, preferred: str) -> None:
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        if backend == "ollama":
            models = self._ollama_models
        elif backend == "huggingface":
            models = HF_MODELS
        else:
            models = CODEX_MODELS
        combo.addItems(models)
        target = current or preferred
        index = combo.findText(target)
        if index >= 0:
            combo.setCurrentIndex(index)
        combo.blockSignals(False)

    def _ensure_video_source(self) -> None:
        if self._video_source is not None:
            return
        try:
            init_channel_autodetect(self.iface_combo.currentText().strip())
            self._video_source = Go2VideoSource(timeout_sec=3.0, fps=30.0)
            self._video_source.start()
            self.status_label.setText("Camera ready")
        except Exception as exc:  # noqa: BLE001
            self._append_log(f"Camera init failed: {exc}")
            self.status_label.setText("Camera init failed")
            self._video_source = None

    def _restart_video_source(self) -> None:
        if self._worker_thread is not None:
            return
        if self._video_source is not None:
            self._video_source.stop()
            self._video_source = None
        self._ensure_video_source()

    def _update_preview(self) -> None:
        if self._video_source is None:
            self.preview_label.setText("Camera feed inactive")
            return
        frame = self._video_source.latest()
        if frame is None:
            error = self._video_source.latest_error()
            self.preview_label.setText(error or "Waiting for camera feed...")
            return
        try:
            array = np.frombuffer(frame.jpeg_bytes, dtype=np.uint8)
            decoded = cv2.imdecode(array, cv2.IMREAD_COLOR)
        except Exception as exc:  # noqa: BLE001
            self.preview_label.setText("Failed to decode camera frame")
            message = f"Preview decode error: {exc}"
            if message != self._last_preview_error:
                self._append_log(message)
                self._last_preview_error = message
            return
        if decoded is None:
            self.preview_label.setText("Failed to decode camera frame")
            return
        rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        image = QtGui.QImage(
            rgb.data,
            rgb.shape[1],
            rgb.shape[0],
            rgb.strides[0],
            QtGui.QImage.Format_RGB888,
        ).copy()
        pixmap = QtGui.QPixmap.fromImage(image)
        self.preview_label.setText("")
        self._last_preview_error = ""
        self.preview_label.setPixmap(
            pixmap.scaled(
                self.preview_label.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        )

    def _start_control(self) -> None:
        if self._worker_thread is not None:
            return

        self._ensure_video_source()
        if self._video_source is None:
            self._append_log("Cannot start control loop without a camera source.")
            return
        config = ControlConfig(
            iface=self.iface_combo.currentText().strip(),
            dry_run=self.dry_run_checkbox.isChecked(),
            planner_choice=ModelChoice(self.planner_backend_combo.currentText(), self.planner_model_combo.currentText()),
            perception_choice=ModelChoice(self.perception_backend_combo.currentText(), self.perception_model_combo.currentText()),
            planner_period_sec=self.planner_period_spin.value(),
            perception_period_sec=self.perception_period_spin.value(),
            planner_prompt=self.prompt_editor.toPlainText().strip() or DEFAULT_PLANNER_SYSTEM_PROMPT,
            ask_prompt_each_step=self.ask_prompt_checkbox.isChecked(),
            say_intent=self.say_intent_toggle.isChecked(),
            ollama_url=self.ollama_url_edit.text().strip(),
            hf_api_url=self.hf_url_edit.text().strip(),
            hf_token=self.hf_token_edit.text().strip(),
            openai_api_url=self.openai_url_edit.text().strip(),
            openai_api_key=self.openai_key_edit.text().strip(),
        )

        self._worker_thread = QtCore.QThread(self)
        self._worker = ControlWorker(config, self._video_source)
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.outputs_ready.connect(self._handle_outputs)
        self._worker.status_message.connect(self._append_log)
        self._worker.prompt_requested.connect(self._show_prompt_dialog)
        self.say_intent_toggle.toggled.connect(self._worker.set_say_intent)
        self._worker_thread.start()

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Running")

    def _stop_control(self) -> None:
        if self._worker is not None:
            self._worker.stop()
        self.status_label.setText("Stopping")

    def _on_worker_finished(self) -> None:
        if self._worker_thread is not None:
            self._worker_thread.quit()
            self._worker_thread.wait(1000)
        self._worker_thread = None
        self._worker = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Idle")

    def _handle_outputs(self, payload: Dict[str, Any]) -> None:
        self.perception_view.setPlainText(json.dumps(payload.get("perception"), indent=2))
        if payload.get("perception_error"):
            self._append_log(f"Perception error: {payload['perception_error']}")
        if payload.get("video_error"):
            self._append_log(f"Video error: {payload['video_error']}")
        self.planner_view.setPlainText(json.dumps(payload.get("planner_output"), indent=2))
        self.actor_view.setPlainText(json.dumps(payload.get("actor_output"), indent=2))
        self.executed_view.setPlainText(json.dumps(payload.get("executed"), indent=2))

    def _append_log(self, message: str) -> None:
        existing = self.log_view.toPlainText().strip()
        lines = [line for line in [existing, message] if line]
        self.log_view.setPlainText("\n".join(lines[-200:]))

    def _show_prompt_dialog(self, current_prompt: str, placeholder_prompt: str) -> None:
        dialog = PromptDialog(current_prompt, placeholder_prompt, self)
        if dialog.exec() == QtWidgets.QDialog.Accepted:
            prompt = dialog.prompt_text() or current_prompt
            self.prompt_editor.setPlainText(prompt)
        else:
            prompt = self.prompt_editor.toPlainText().strip() or current_prompt
        if self._worker is not None:
            self._worker.set_planner_prompt(prompt)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._shutdown()
        super().closeEvent(event)

    def _shutdown(self) -> None:
        self._stop_control()
        if self._worker is not None:
            self._worker.stop()
        if self._worker_thread is not None:
            self._worker_thread.quit()
            self._worker_thread.wait(3000)
        self._worker_thread = None
        self._worker = None
        if self._video_source is not None:
            self._video_source.stop()
            self._video_source = None


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
