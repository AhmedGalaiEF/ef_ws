from __future__ import annotations

import argparse

import pytest

from go2.scripts.go2_sport_client import parse_args, positive_finite_float


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf"])
def test_positive_finite_float_rejects_invalid_timeout(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        positive_finite_float(value)


def test_parse_args_validates_robot_timeout() -> None:
    args = parse_args(["--timeout", "2.5", "--domain-id", "3"])
    assert args.timeout == pytest.approx(2.5)
    assert args.domain_id == 3

    with pytest.raises(SystemExit):
        parse_args(["--timeout", "nan"])
