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
from pypalc.core.venue import AudienceLines
from tests.legacy.capture import case_path


def _empty_lines():
    return AudienceLines(line_end_indices=np.array([0]).astype(int))


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


def test_legacy_classes_do_not_share_default_lists():
    from pypalc.core.palc import ArrayGeometry, PalcConfig
    from pypalc.core.sound_field import SoundFieldConfig
    from pypalc.core.target_slope import TargetSlopeState
    from pypalc.core.venue import PalcVenue
    from pypalc.gui.ranges import PlotRanges

    def attrs(cls, names):
        a, b = cls(), cls()
        for name in names:
            assert getattr(a, name) is not getattr(b, name), (cls, name)

    attrs(AudienceLines, ["line_start_x", "points_x", "line_end_indices",
                          "plot_x", "cut_lines"])
    attrs(PalcVenue, ["segments", "audience_segments", "points_x"])
    attrs(PalcConfig, ["discrete_tilt_angles", "weighting_factors",
                       "aperture_angles", "tilt_angles",
                       "hard_margin_last_angle", "gap_weights",
                       "weighting_weights"])
    attrs(SoundFieldConfig, ["measured_directivity_data",
                             "measured_angles_deg", "frequencies",
                             "plot_measured_directivity_db",
                             "plot_measured_angles_deg",
                             "discrete_tilt_angles", "aperture_angles"])
    attrs(ArrayGeometry, ["tilt_angles_deg"])
    attrs(TargetSlopeState, ["region_levels_db"])
    attrs(PlotRanges, ["venue_x", "venue_y", "histogram_x", "spl_over_distance_y"])
    assert PalcConfig().aperture_angles == [0.0524]
    assert TargetSlopeState().region_levels_db == [0, 0]


def test_hard_margin_with_discrete_angles():
    angles = (0.0, 0.5, 1.0, 2.0, 3.5, 5.0, 7.0, 10.0)
    base = load_setup(case_path("gap_hard_margin"))
    setup = dataclasses.replace(base, array=dataclasses.replace(
        base.array, discrete_tilt_angles_deg=angles))
    result = run(setup)
    # the first cabinet of every continuous audience line starts from the
    # last angle of the previous one, also with discrete angles
    for angle in result.optimized.inter_cabinet_angles_deg[1:]:
        assert angle in angles


def test_default_aperture_is_not_special():
    # the first run starts from the configured aperture angle, so the GUI
    # default 3.0 deg behaves like any neighboring value
    base = load_setup(case_path("simple"))
    results = []
    for aperture in (3.0, 3.0000001):
        setup = dataclasses.replace(base, loudspeaker=dataclasses.replace(
            base.loudspeaker, aperture_angle_deg=aperture))
        results.append(run(setup).geometry.tilt_angles_deg)
    np.testing.assert_allclose(results[0], results[1], rtol=0, atol=1e-4)
