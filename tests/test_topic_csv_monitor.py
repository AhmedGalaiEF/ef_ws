from __future__ import annotations

import argparse
import array
from pathlib import Path

import pytest

rclpy = pytest.importorskip("rclpy")

from dev.topic_csv_monitor import compact_value, load_topics, normalize_topic, parse_args, selected_topics


def test_normalize_topic_adds_leading_slash_and_preserves_existing_one() -> None:
    assert normalize_topic("wirelesscontroller") == "/wirelesscontroller"
    assert normalize_topic("/wirelesscontroller") == "/wirelesscontroller"
    assert normalize_topic("  ") == ""


def test_load_topics_ignores_comments_and_blank_lines(tmp_path: Path) -> None:
    topics_file = tmp_path / "topics.txt"
    topics_file.write_text("\n# comment\nwirelesscontroller\n/slam_info\n", encoding="utf-8")

    assert load_topics(topics_file) == {"/wirelesscontroller", "/slam_info"}


def test_selected_topics_prefers_explicit_topic_arguments() -> None:
    args = argparse.Namespace(
        all=False,
        topic=["wirelesscontroller", " /slam_info "],
        topics_file=None,
    )

    assert selected_topics(args) == {"/wirelesscontroller", "/slam_info"}


def test_selected_topics_returns_none_when_all_topics_requested() -> None:
    args = argparse.Namespace(all=True, topic=[], topics_file=None)

    assert selected_topics(args) is None


def test_compact_value_summarizes_long_arrays_and_strings() -> None:
    value = {
        "payload": array.array("B", [1, 2, 3, 4]),
        "text": "abcdef",
    }

    summary = {
        key: compact_value(item, max_sequence=2, max_string=3, depth=4)
        for key, item in value.items()
    }

    assert summary["payload"] == {
        "__array_typecode__": "B",
        "__array_len__": 4,
        "preview": [1, 2],
    }
    assert summary["text"] == {
        "__string_len__": 6,
        "preview": "abc",
    }


def test_parse_args_rejects_invalid_monitor_limits() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--discovery-period", "0"])
    with pytest.raises(SystemExit):
        parse_args(["--max-sequence", "-1"])
