"""Behavior of the pipeline beyond the baseline cases."""
import copy
import dataclasses

import numpy as np
import pytest

from pypalc.core.errors import VenueError
from pypalc.core.pipeline import (build_venue, check_line, fixed_angle_limits,
                                  initial_target_slope, result_arrays, run)
from pypalc.core.settings import (AudienceLine, Setup, VenueSettings,
                                  load_setup)
from pypalc.core.venue import PAL
from tests.legacy.capture import case_path


def _empty_lines():
    return PAL(pal_index=np.array([0]).astype(int))


def test_fixed_angle_is_clamped():
    base = load_setup(case_path("simple"))
    setup = dataclasses.replace(base, algorithm=dataclasses.replace(
        base.algorithm, fixed_top_angle_deg=90.0))
    venue = build_venue(setup.venue, setup.array)
    max_deg, min_deg = fixed_angle_limits(venue, setup)
    assert min_deg < max_deg < 90.0
    result = run(setup, venue)
    assert result.optimized.tilt_angles_deg[0] == round(max_deg, 2)


def test_fixed_angle_below_minimum_is_clamped():
    base = load_setup(case_path("simple"))
    setup = dataclasses.replace(base, algorithm=dataclasses.replace(
        base.algorithm, fixed_top_angle_deg=-90.0))
    venue = build_venue(setup.venue, setup.array)
    _, min_deg = fixed_angle_limits(venue, setup)
    result = run(setup, venue)
    assert result.optimized.tilt_angles_deg[0] == round(min_deg, 2)


def test_check_line_rejects_negative_slope():
    with pytest.raises(VenueError):
        check_line(_empty_lines(), AudienceLine((2.0, 3.0), (10.0, 1.0)))


def test_check_line_rejects_points():
    with pytest.raises(VenueError):
        check_line(_empty_lines(), AudienceLine((2.0, 3.0), (2.0, 3.0)))


def test_check_line_rejects_too_many_points():
    with pytest.raises(VenueError, match="2000"):
        check_line(_empty_lines(),
                   AudienceLine((0.0, 0.0), (100.0, 0.0), 30.0))


def test_check_line_accepts_valid_line():
    check_line(_empty_lines(), AudienceLine((2.0, 0.0), (10.0, 1.0), 5.0))


def test_build_venue_without_lines_raises():
    with pytest.raises(VenueError):
        build_venue(VenueSettings(), Setup().array)


def test_build_venue_rejects_overlapping_lines():
    line = AudienceLine((2.0, 0.0), (10.0, 1.0), 5.0)
    with pytest.raises(VenueError):
        build_venue(VenueSettings(audience_lines=(line, line)), Setup().array)


def test_run_does_not_mutate_setup():
    setup = load_setup(case_path("gap_hard_margin"))
    before = copy.deepcopy(setup)
    run(setup)
    assert setup == before


def test_run_does_not_mutate_venue():
    setup = load_setup(case_path("fixed_angle"))
    venue = build_venue(setup.venue, setup.array)
    first = result_arrays(run(setup, venue))
    second = result_arrays(run(setup, venue))
    for key in first:
        np.testing.assert_array_equal(first[key], second[key], err_msg=key)


def test_initial_target_slope_lies_on_the_distance_grid():
    setup = load_setup(case_path("target_slope"))
    venue = build_venue(setup.venue, setup.array)
    target = initial_target_slope(setup, venue, hinges=1)
    assert len(target.points) == 3
    without = dataclasses.replace(setup, algorithm=dataclasses.replace(
        setup.algorithm, weighting="without"), target_slope=None)
    grid = result_arrays(run(without, venue))["spl_over_distance_x"]
    distances = [p.distance for p in target.points]
    assert distances == sorted(distances)
    assert all(d in grid for d in distances)
    assert target.reference_distance in grid
    assert distances[0] <= target.reference_distance <= distances[-1]
