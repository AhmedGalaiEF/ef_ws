from __future__ import annotations

import os


class PlannerPromptEditor:
    def __init__(self, enabled: bool = False):
        self._enabled = enabled

    def maybe_update(self, current_prompt: str) -> str:
        if not self._enabled:
            return current_prompt
        if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            try:
                return self._prompt_gui(current_prompt)
            except Exception:
                pass
        return self._prompt_terminal(current_prompt)

    def _prompt_gui(self, current_prompt: str) -> str:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        should_edit = messagebox.askyesno(
            "Planner Prompt",
            "Edit planner prompt before the next inference?",
            parent=root,
        )
        if not should_edit:
            root.destroy()
            return current_prompt

        selection = {"prompt": current_prompt}
        window = tk.Toplevel(root)
        window.title("Planner Prompt")
        window.geometry("900x600")

        text = tk.Text(window, wrap="word")
        text.pack(fill="both", expand=True, padx=12, pady=12)
        text.insert("1.0", current_prompt)

        button_row = tk.Frame(window)
        button_row.pack(fill="x", padx=12, pady=(0, 12))

        def _use_prompt() -> None:
            updated = text.get("1.0", "end-1c").strip()
            selection["prompt"] = updated or current_prompt
            window.destroy()

        def _keep_prompt() -> None:
            window.destroy()

        tk.Button(button_row, text="Use Prompt", command=_use_prompt).pack(side="right")
        tk.Button(button_row, text="Keep Current", command=_keep_prompt).pack(side="right", padx=(0, 8))

        window.protocol("WM_DELETE_WINDOW", _keep_prompt)
        window.transient(root)
        window.grab_set()
        root.wait_window(window)
        root.destroy()
        return selection["prompt"]

    def _prompt_terminal(self, current_prompt: str) -> str:
        answer = input("Edit planner prompt before next inference? [y/N]: ").strip().lower()
        if answer not in {"y", "yes"}:
            return current_prompt

        print("Enter the planner prompt. Finish with a line containing only '.'")
        print("Current prompt:")
        print(current_prompt)
        lines = []
        while True:
            line = input()
            if line == ".":
                break
            lines.append(line)
        updated = "\n".join(lines).strip()
        return updated or current_prompt
