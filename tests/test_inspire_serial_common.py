from __future__ import annotations

import argparse

import pytest

from g1.Inspire_hands.inspire_serial_common import (
    SerialHand,
    add_serial_connection_args,
)


def test_serial_argument_validation_rejects_invalid_values() -> None:
    parser = argparse.ArgumentParser()
    add_serial_connection_args(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["--baudrate", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--timeout-s", "nan"])


def test_serial_hand_validates_constructor_values() -> None:
    with pytest.raises(ValueError, match="baudrate"):
        SerialHand("/dev/null", baudrate=0)
    with pytest.raises(ValueError, match="write_delay_s"):
        SerialHand("/dev/null", write_delay_s=float("nan"))
