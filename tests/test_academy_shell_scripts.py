from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    ROOT / "g1" / "academy" / "setup_python_cyclonedds_unitree_sdk.sh",
    ROOT / "g1" / "academy" / "solved" / "reset_users.sh",
    ROOT / "g1" / "academy" / "solved" / "setup_jupyterhub.sh",
]


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.name)
def test_academy_script_has_valid_bash_syntax(script: Path) -> None:
    result = subprocess.run(
        ["bash", "-n", str(script)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.name)
def test_academy_script_help_is_safe_without_root(script: Path) -> None:
    result = subprocess.run(
        ["bash", str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Usage:" in result.stdout


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.name)
def test_academy_script_rejects_unknown_arguments(script: Path) -> None:
    result = subprocess.run(
        ["bash", str(script), "--not-a-real-option"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "Unknown argument" in result.stderr


def test_user_reset_requires_explicit_confirmation_when_run_as_root() -> None:
    if os.geteuid() != 0:
        pytest.skip("destructive-operation guard is evaluated after the root check")

    script = ROOT / "g1" / "academy" / "solved" / "reset_users.sh"
    result = subprocess.run(
        ["bash", str(script)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Refusing destructive account reset" in result.stderr
