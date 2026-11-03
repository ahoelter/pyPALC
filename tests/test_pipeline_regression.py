"""The pipeline reproduces the results recorded from the legacy code.

The comparison uses the baseline tolerance of :mod:`tests.baseline_cases`
(BLAS builds differ in the last digits).
"""
import numpy as np
import pytest

from pypalc.core.pipeline import result_arrays, run
from pypalc.core.settings import load_setup
from tests.baseline_cases import (CASES, assert_matches_baseline,
                                   baseline_path, case_path)


@pytest.mark.parametrize("case", CASES)
def test_matches_baseline(case):
    result = result_arrays(run(load_setup(case_path(case))))
    with np.load(baseline_path(case)) as baseline:
        assert sorted(result) == sorted(baseline.files)
        for key in baseline.files:
            assert_matches_baseline(result[key], baseline[key], err_msg=key)
