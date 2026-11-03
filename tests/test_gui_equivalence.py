"""The GUI computes the recorded baseline numbers.

Every case of ``tests/cases`` is entered into a fresh :class:`PalcApp` the
way a user does it, page by page, through the widgets and the button
callbacks, in the click order of the legacy replay that recorded the
baselines (see :mod:`tests.baseline_cases`). For target slope the target
points and the reference point are dragged by writing the target sources,
as Bokeh does when a point is dragged in the browser. Afterwards the
sources the GUI shows (and the session's last result, ``state.result``) must
equal ``tests/baseline/<case>.npz`` within the baseline tolerance of
:mod:`tests.baseline_cases` (BLAS builds differ in the last digits). That
the GUI computes exactly what the pipeline computes in the same process is
pinned by ``tests/test_apply_setup.py`` (hand path equals loaded session,
loaded session equals headless run).
"""
import base64
import contextlib
import io
from pathlib import Path

import numpy as np
import pytest
from bokeh.document import Document

from pypalc.core.settings import load_setup
from pypalc.gui.app import PalcApp
from tests.baseline_cases import (CASES, assert_matches_baseline,
                                   baseline_path, case_path)

_VARIANTS = {"palc1": "PALC 1", "palc2": "PALC 2", "palc3": "PALC 3"}
_GAP_HANDLING = {"without": "Without", "hard_margin": "Hard Margin",
                 "soft_margin": "Soft Margin"}
_WEIGHTING = {"without": "Without", "linear": "Linear Spacing",
              "logarithmic": "Logarithmic Spacing",
              "target_slope": "Target Slope"}
_DIRECTIVITY = {"circular_piston": "Circular Piston",
                "line_piston": "Line Piston",
                "combined_circular_line": "Combined Circular/Line",
                "measured": "Measured Loudspeaker Data"}
_REFERENCE = {"straight": "Straight", "progressive": "Progressive",
              "arc": "Arc", "user_defined": "User Defined"}


def _fmt(value):
    """Text a user types for a number: ``2.0`` -> ``'2'``."""
    text = repr(float(value))
    return text[:-2] if text.endswith(".0") else text


def _enter_text(widget, value, as_int=False):
    """Type a number into a text input unless it already shows it."""
    try:
        if float(widget.value) == float(value):
            return
    except ValueError:
        pass
    widget.value = str(int(value)) if as_int else _fmt(value)


def _measured_csv_base64(setup):
    md = setup.loudspeaker.measured_directivity
    raw = Path(md.path).read_bytes() if md.path is not None \
        else md.csv.encode("utf-8")
    return base64.b64encode(raw).decode("ascii")


def enter(app, setup):
    """Enter a setup into the app page by page, without pressing "PALC Run".
    """
    venue, loudspeaker, array, algorithm = (
        app.pages[1], app.pages[2], app.pages[3], app.pages[4])
    # page 1
    for line in setup.venue.audience_lines:
        venue.start_x_input.value = repr(float(line.start[0]))
        venue.end_x_input.value = repr(float(line.stop[0]))
        venue.points_per_meter_input.value = repr(float(line.points_per_meter))
        venue.start_y_input.value = repr(float(line.start[1]))
        venue.end_y_input.value = repr(float(line.stop[1]))
        venue.save_line()
    app.check_venue()
    # page 2
    ls, ar, ref, alg = (setup.loudspeaker, setup.array,
                        setup.reference_array, setup.algorithm)
    _enter_text(loudspeaker.aperture_input, ls.aperture_angle_deg)
    _enter_text(loudspeaker.cabinet_height_input, ls.cabinet_height)
    loudspeaker.directivity_select.value = _DIRECTIVITY[ls.directivity]
    if ls.directivity == 'measured':
        loudspeaker.directivity_upload.value = _measured_csv_base64(setup)
    # page 3
    _enter_text(array.cabinet_count_input, ar.cabinet_count, as_int=True)
    _enter_text(array.top_x_input, ar.top_position[0])
    _enter_text(array.top_y_input, ar.top_position[1])
    _enter_text(array.cabinet_gap_input, ar.cabinet_gap)
    if ar.discrete_tilt_angles_deg is not None:
        array.use_discrete_angles_select.value = 'Yes'
        array.discrete_angles_input.value = ','.join(
            _fmt(a) for a in ar.discrete_tilt_angles_deg)
    array.reference_type_select.value = _REFERENCE[ref.type]
    if ref.type != 'user_defined':
        _enter_text(array.reference_start_input, ref.top_angle_deg)
        if ref.type != 'straight':
            _enter_text(array.reference_step_stop_input, ref.angle_param_deg)
        if ref.use_discrete_angles and \
                not array.reference_discrete_angles_select.disabled:
            array.reference_discrete_angles_select.value = 'Yes'
    else:
        array.reference_user_angles_input.value = ','.join(
            _fmt(a) for a in ref.tilt_angles_deg)
    # page 4
    algorithm.palc_variant_select.value = _VARIANTS[alg.variant]
    algorithm.gap_handling_select.value = _GAP_HANDLING[alg.gap_handling]
    if not algorithm.soft_margin_slider.disabled:
        algorithm.soft_margin_slider.value = float(alg.soft_margin_strength)
    algorithm.tolerance_slider.value = float(alg.tolerance)
    algorithm.frequency_range_slider.value = tuple(
        float(f) for f in alg.frequency_range_hz)
    if alg.fixed_top_angle_deg is not None:
        assert not algorithm.use_fixed_angle_select.disabled
        algorithm.use_fixed_angle_select.value = 'Yes'
        algorithm.fixed_angle_input.value = _fmt(alg.fixed_top_angle_deg)
    algorithm.weighting_select.value = _WEIGHTING[alg.weighting]
    if alg.weighting in ('linear', 'logarithmic'):
        algorithm.weighting_nu_slider.value = float(np.cbrt(alg.weighting_nu))
        assert app.state.config.weighting_nu == alg.weighting_nu
    if alg.weighting == 'target_slope':
        ts = setup.target_slope
        algorithm.hinges_select.value = str(len(ts.points) - 2)
        # drag the target points and the reference point
        d = algorithm.target_source.data
        algorithm.target_selected('indices', [], [0])
        algorithm.target_source.data = dict(
            distance_m=[p.distance for p in ts.points],
            level_db=[p.level_db for p in ts.points],
            x_m=list(d['x_m']), y_m=list(d['y_m']))
        r = algorithm.target_reference_source.data
        algorithm.target_selected('indices', [], [0])
        algorithm.target_reference_source.data = dict(
            distance_m=[ts.reference_distance], level_db=list(r['level_db']),
            x_m=list(r['x_m']), y_m=list(r['y_m']))


def drive(app, setup):
    """Enter a setup into the app, press "PALC Run" and return the shown
    results."""
    enter(app, setup)
    algorithm, results = app.pages[4], app.pages[5]
    alg = setup.algorithm
    algorithm.start_toggle.active = True
    assert app.config_error_info.text == '', app.config_error_info.text
    assert algorithm.start_toggle.active is False
    st = app.state

    def a(v):
        return np.array(v, dtype=float)

    out = {
        'tilt_angles_deg': a(algorithm.results_source.data['tilt_angle_deg']),
        'inter_cabinet_angles_deg':
            a(algorithm.results_source.data['inter_cabinet_angle_deg']),
        'reference_tilt_angles_deg': a(st.result.reference.tilt_angles_deg),
        'frequencies_hz': a(st.result.optimized.evaluation.frequencies),
        'spl_db': a(st.result.optimized.evaluation.spl_db),
        'reference_spl_db': a(st.result.reference.evaluation.spl_db),
        'spl_over_distance_x':
            a(algorithm.spl_over_distance_source.data['distance_m']),
        'spl_over_distance_db':
            a(algorithm.spl_over_distance_source.data['spl_db']),
        'reference_spl_over_distance_db':
            a(algorithm.reference_spl_over_distance_source.data[
                'reference_spl_db']),
        'homogeneity_db':
            a(results.homogeneity_source.data['homogeneity_db']),
        'homogeneity_low_db':
            a(results.homogeneity_source.data['homogeneity_low_db']),
        'homogeneity_high_db':
            a(results.homogeneity_source.data['homogeneity_high_db']),
        'reference_homogeneity_db':
            a(results.homogeneity_source.data['reference_homogeneity_db']),
        'histogram_steps': a(results.histogram_source.data['spl_db']),
        'histogram_tops': a(results.histogram_source.data['count']),
        'reference_histogram_steps':
            a(results.reference_histogram_source.data['spl_db']),
        'reference_histogram_tops':
            a(results.reference_histogram_source.data['count']),
        'venue_x': a(results.venue_points_source.data['x_m']),
        'venue_y': a(results.venue_points_source.data['y_m']),
    }
    if alg.weighting == 'target_slope':
        out['target_x'] = a(algorithm.target_source.data['distance_m'])
        out['target_db'] = a(algorithm.target_source.data['level_db'])
        out['weighting_weights'] = a(st.result.weighting_weights)
    return out


@pytest.mark.parametrize("case", CASES)
def test_gui_matches_baseline(case):
    setup = load_setup(case_path(case))
    app = PalcApp(Document())
    app.show()
    with contextlib.redirect_stdout(io.StringIO()):
        gui = drive(app, setup)
    with np.load(baseline_path(case)) as baseline:
        assert sorted(gui) == sorted(baseline.files)
        for key in baseline.files:
            assert_matches_baseline(gui[key], baseline[key], err_msg=key)
