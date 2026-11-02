"""The pipeline reproduces the results recorded from the legacy code."""
import numpy as np
import pytest

from pypalc.core.pipeline import result_arrays, run
from pypalc.core.settings import load_setup
from tests.legacy.capture import CASES, baseline_path, case_path


@pytest.mark.parametrize("case", CASES)
def test_matches_baseline(case):
    result = result_arrays(run(load_setup(case_path(case))))
    with np.load(baseline_path(case)) as baseline:
        assert sorted(result) == sorted(baseline.files)
        for key in baseline.files:
            np.testing.assert_allclose(result[key], baseline[key],
                                       rtol=1e-9, atol=1e-9, err_msg=key)
