"""The golden-master cases: setup files and their recorded baselines.

Every case is a setup file ``tests/cases/<case>.yaml`` with the results
``tests/baseline/<case>.npz`` recorded from the legacy pyPALC code (the
replay ``tests/legacy/capture.py``, removed after commit ``8535b95``; its
last change is commit ``d3c9b2f``). Only a bug-fix commit regenerates a
baseline.

A result computed now is compared with a stored baseline at the tolerance
:data:`BASELINE_RTOL` / :data:`BASELINE_ATOL`, not exactly: numpy and scipy
builds with different BLAS libraries (MKL, OpenBLAS) differ in the last
digits (relative differences around 1e-12). Comparisons of two results
computed in the same process stay exact.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CASES_DIR = REPO / "tests" / "cases"
BASELINE_DIR = REPO / "tests" / "baseline"

#: Relative and absolute tolerance of a comparison with a stored baseline.
BASELINE_RTOL = 1e-9
BASELINE_ATOL = 1e-9

#: Cases recorded in ``tests/baseline``.
CASES = (
    "simple", "gap_without", "gap_hard_margin", "gap_soft_margin", "palc1",
    "palc3", "discrete_angles", "fixed_angle", "measured", "linear_weighting",
    "log_weighting", "target_slope", "gap_soft_margin_target_slope",
)


def case_path(case: str) -> Path:
    """Path of the setup file of a baseline case.

    Parameters
    ----------
    case : str
        Name of the case, one of :data:`CASES`.

    Returns
    -------
    Path
        ``tests/cases/<case>.yaml``.
    """
    return CASES_DIR / ("%s.yaml" % case)


def baseline_path(case: str) -> Path:
    """Path of the recorded baseline of a case.

    Parameters
    ----------
    case : str
        Name of the case, one of :data:`CASES`.

    Returns
    -------
    Path
        ``tests/baseline/<case>.npz``.
    """
    return BASELINE_DIR / ("%s.npz" % case)


def assert_matches_baseline(actual, expected, err_msg: str = "") -> None:
    """Assert that a computed array equals a stored baseline array.

    Shape and non-float values must be equal; floats may differ by
    :data:`BASELINE_RTOL` / :data:`BASELINE_ATOL` (BLAS builds differ in
    the last digits).

    Parameters
    ----------
    actual : array_like
        The value computed by the test.
    expected : array_like
        The value from ``tests/baseline/<case>.npz``.
    err_msg : str, optional
        Message shown on failure, e.g. the baseline key.

    Raises
    ------
    AssertionError
        If the values differ.
    """
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape, (err_msg, actual.shape,
                                            expected.shape)
    if actual.dtype.kind in "fc" or expected.dtype.kind in "fc":
        np.testing.assert_allclose(actual, expected, rtol=BASELINE_RTOL,
                                   atol=BASELINE_ATOL, err_msg=err_msg)
    else:
        np.testing.assert_array_equal(actual, expected, err_msg=err_msg)
