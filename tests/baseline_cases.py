"""The golden-master cases: setup files and their recorded baselines.

Every case is a setup file ``tests/cases/<case>.yaml`` with the results
``tests/baseline/<case>.npz`` recorded from the legacy pyPALC code (the
replay ``tests/legacy/capture.py``, removed after commit ``8535b95``; its
last change is commit ``d3c9b2f``). Only a bug-fix commit regenerates a
baseline.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CASES_DIR = REPO / "tests" / "cases"
BASELINE_DIR = REPO / "tests" / "baseline"

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
