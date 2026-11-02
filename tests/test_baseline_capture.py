"""The legacy replay that produced ``tests/baseline`` is deterministic."""
import numpy as np
import pytest

from pypalc.core.settings import load_setup
from tests.legacy.capture import CASES, case_path, run_legacy


@pytest.mark.parametrize("case", CASES)
def test_capture_is_deterministic(case):
    setup = load_setup(case_path(case))
    first = run_legacy(setup)
    second = run_legacy(setup)
    assert sorted(first) == sorted(second)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key], err_msg=key)
