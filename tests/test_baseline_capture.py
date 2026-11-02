"""The legacy replay that produced ``tests/baseline`` is deterministic.

``test_legacy_matches_baseline`` separates environment drift (numpy, scipy,
platform) from refactoring regressions: if it fails, the legacy code itself
no longer reproduces the recorded baseline.
"""
import numpy as np
import pytest

from pypalc.core.settings import load_setup
from tests.legacy.capture import CASES, baseline_path, case_path, run_legacy


@pytest.mark.parametrize("case", CASES)
def test_capture_is_deterministic(case):
    setup = load_setup(case_path(case))
    first = run_legacy(setup)
    second = run_legacy(setup)
    assert sorted(first) == sorted(second)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key], err_msg=key)


@pytest.mark.parametrize("case", CASES)
def test_legacy_matches_baseline(case):
    result = run_legacy(load_setup(case_path(case)))
    with np.load(baseline_path(case)) as baseline:
        assert sorted(result) == sorted(baseline.files)
        for key in baseline.files:
            np.testing.assert_array_equal(result[key], baseline[key],
                                          err_msg=key)
