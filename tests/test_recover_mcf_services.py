from __future__ import annotations

import pytest

from go2.scripts.recover_mcf_services import _nonnegative_finite_float, _nonnegative_int


def test_recovery_argument_parsers_reject_invalid_values() -> None:
    assert _nonnegative_int("3") == 3
    assert _nonnegative_finite_float("0.5") == pytest.approx(0.5)

    with pytest.raises(Exception):
        _nonnegative_int("-1")
    with pytest.raises(Exception):
        _nonnegative_finite_float("nan")
