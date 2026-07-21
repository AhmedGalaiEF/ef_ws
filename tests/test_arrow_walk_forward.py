from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from dev.arrow_walk_forward import infer_forward_speed


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["topic", "message_json"])
        writer.writeheader()
        writer.writerows(rows)


def test_infer_forward_speed_uses_median_of_active_wirelesscontroller_samples(tmp_path: Path) -> None:
    csv_path = tmp_path / "walk_forward.csv"
    write_rows(
        csv_path,
        [
            {"topic": "/wirelesscontroller", "message_json": json.dumps({"ly": 0.30})},
            {"topic": "/wirelesscontroller", "message_json": json.dumps({"ly": 0.90})},
            {"topic": "/wirelesscontroller", "message_json": json.dumps({"ly": 0.60})},
            {"topic": "/wirelesscontroller", "message_json": json.dumps({"ly": 0.01})},
            {"topic": "/other", "message_json": json.dumps({"ly": 1.50})},
        ],
    )

    assert infer_forward_speed(csv_path) == pytest.approx(0.60)


def test_infer_forward_speed_raises_when_no_active_samples_exist(tmp_path: Path) -> None:
    csv_path = tmp_path / "walk_forward.csv"
    write_rows(
        csv_path,
        [
            {"topic": "/wirelesscontroller", "message_json": json.dumps({"ly": 0.0})},
            {"topic": "/wirelesscontroller", "message_json": json.dumps({"ly": 0.03})},
        ],
    )

    with pytest.raises(RuntimeError, match="No active /wirelesscontroller ly samples found"):
        infer_forward_speed(csv_path)
