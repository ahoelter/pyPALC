"""The GUI's widgets react to user input as in the legacy GUI.

A fixed sequence of 100 user interactions (typing into inputs, choosing
options, clicking buttons, tapping lines and points, dragging the target
slope, running PALC, switching pages) is replayed on a fresh
:class:`PalcApp`. After every step the state of about 70 widgets, sources
and figures (visible, disabled, value, options, label, text, active, data,
ranges, frame height, ...) and the number of models in the document are
compared with ``tests/gui_cascade_golden.json``.

How the golden file was made
----------------------------
The golden state is the state of the legacy GUI (the original ``main.py``
before the split into :mod:`pypalc.gui`), recorded by driving the legacy
module-level callbacks through the same steps. Only the accepted
differences listed below replace legacy values; every other value is the
legacy one.

Accepted differences to the legacy GUI
--------------------------------------
* ``psi_bad`` (``2.aperture_input.value``): an invalid aperture angle now
  resets the input to the default ``'3.0'`` instead of ``'0.0524'``
  (bug fix 702f6a6).
* ``save_same`` (``1.venue_error_info.text``): saving a line that lies on
  an already drawn line now shows "Error: The audience line lies on an
  already drawn audience line"; the legacy GUI rejected the line without
  a text.
* ``run_ts`` and later steps (a target slope run with hard margin gap
  handling, whose optimization raises an IndexError): the legacy GUI
  crashed in the callback with ``UnboundLocalError`` after it had already
  written the new tilt angles, rays, cabinets and sound field into their
  sources. The GUI now shows "Error: PALC calculation failed", releases
  the start toggle and keeps the results of the previous run in all
  sources (``4.results_source``, ``5.sound_field_source``,
  ``app.cabinet_patches_source``, ``app.rays_source``,
  ``app.hit_points_source``, ``app.config_error_info.text`` and the
  recorded exception differ; see :func:`test_failed_run_keeps_results`).
* The rewording of the "more than 2000 points" text is not reached by
  these steps.
* ``document.model_count`` on page 4 (steps ``page4`` and ``page4b`` and
  the steps up to the next page change) is 8 higher: the "Save Setup"
  button, the "Load Setup" title and file input, the source of the setup
  file download and their JavaScript callbacks (Task 7). Nothing else of
  the golden file changed; the new widgets are not among :data:`MODELS`
  (``tests/test_apply_setup.py`` covers them). Refreshing the range of a
  fixed angle when the array position changes (Task 7) changes no
  recorded value.

The values are rounded to 9 decimals. Besides the widgets, the values of
the PALC configuration that the configuration widgets store
(:data:`CONFIG_KEYS`) are compared. Values whose JSON text is longer
than :data:`MAX_INLINE` characters are stored as a hash of that text.
The golden file stores the first state completely and for every later step
only the values that changed.

Running ``python -m tests.test_gui_cascade OUT.json`` writes the states of
the current GUI in the same format, which helps to inspect a failure.
"""
import base64
import contextlib
import functools
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from bokeh.document import Document

from pypalc.gui.app import PalcApp

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "tests" / "gui_cascade_golden.json"
#: Longer values (as JSON text) are compared by a hash.
MAX_INLINE = 1500

#: The recorded models: name in the golden file -> attribute path
#: ("<page index>.<attribute>" or "app.<attribute>").
MODELS = (
    "1.start_x_input", "1.start_y_input", "1.end_x_input", "1.end_y_input",
    "1.points_per_meter_input", "1.save_line_button",
    "1.remove_line_button", "1.insert_line_button",
    "1.remove_selected_line_button", "1.check_venue_button",
    "1.venue_error_info", "1.lines_source", "1.non_audience_source",
    "1.sound_field_only_source", "1.tapped_line_source",
    "2.aperture_input", "2.cabinet_height_input", "2.directivity_select",
    "2.directivity_upload", "2.directivity_source",
    "3.cabinet_count_input", "3.top_x_input", "3.top_y_input",
    "3.cabinet_gap_input", "3.use_discrete_angles_select",
    "3.discrete_angles_input", "3.reference_type_select",
    "3.reference_discrete_angles_select", "3.reference_start_input",
    "3.reference_step_stop_input", "3.reference_user_angles_input",
    "3.reference_angles_source",
    "4.palc_variant_select", "4.gap_handling_select",
    "4.soft_margin_slider", "4.tolerance_slider",
    "4.frequency_range_slider", "4.weighting_select", "4.hinges_select",
    "4.weighting_nu_slider", "4.nu_plus_button", "4.nu_minus_button",
    "4.nu_step_select", "4.use_fixed_angle_select", "4.fixed_angle_input",
    "4.fixed_angle_text", "4.start_toggle", "4.clear_results_button",
    "4.results_source", "4.spl_over_distance_source",
    "4.reference_spl_over_distance_source", "4.target_source",
    "4.target_reference_source", "4.target_region_source",
    "4.spl_over_distance_figure",
    "5.sound_field_source", "5.venue_points_source",
    "5.homogeneity_source", "5.histogram_source", "5.beam_source",
    "5.beam_figure", "5.reference_beam_figure", "5.venue_points_figure",
    "5.beam_frequency_select", "5.beam_checkbox",
    "app.config_error_info", "app.page_select", "app.venue_axes_select",
    "app.venue_figure", "app.cabinet_patches_source", "app.rays_source",
    "app.hit_points_source",
)
PROPS = ('visible', 'disabled', 'value', 'options', 'label', 'button_type',
         'text', 'active', 'start', 'end', 'step', 'title', 'data',
         'frame_height', 'x_range', 'y_range', 'min_border')

#: Values of the PALC configuration the configuration widgets store.
CONFIG_KEYS = ('aperture_angles', 'cabinet_count', 'cabinet_gap',
               'cabinet_height', 'directivity', 'discrete_tilt_angles',
               'fixed_angle', 'frequency_range_hz', 'gap_handling',
               'palc_variant', 'soft_margin_strength', 'tolerance', 'top_x',
               'top_y', 'use_discrete_angles', 'use_fixed_angle',
               'use_weighting', 'weighting_nu')


def _model(app, path):
    owner, name = path.split('.')
    return getattr(app if owner == 'app' else app.pages[int(owner)], name)


def _norm(v):
    """JSON-able value, floats rounded to 9 decimals, models by type."""
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    if isinstance(v, np.ndarray):
        return _norm(v.tolist())
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    if isinstance(v, float):
        return round(v, 9)
    if hasattr(v, 'start') and hasattr(v, 'end') and \
            type(v).__name__.endswith('Range1d'):
        return [type(v).__name__, _norm(v.start), _norm(v.end)]
    if hasattr(v, 'properties'):
        return type(v).__name__
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return round(float(v), 9)
    return v


def compact(value):
    """The value, or a hash of it if its JSON text is long."""
    text = json.dumps(value, sort_keys=True)
    if len(text) <= MAX_INLINE:
        return value
    return {"sha1": hashlib.sha1(text.encode()).hexdigest(),
            "size": len(text)}


def snapshot(app, error):
    """State of the recorded models after a step."""
    state = {}
    for path in MODELS:
        model = _model(app, path)
        for prop in PROPS:
            if prop in model.properties():
                state[path + '.' + prop] = _norm(getattr(model, prop))
        if 'selected' in model.properties():
            state[path + '.selected'] = _norm(model.selected.indices)
    roots = app.doc.roots
    state['document.model_count'] = \
        len(list(roots[0].references())) if roots else 0
    config = app.state.config
    for key in CONFIG_KEYS:
        state['config.' + key] = _norm(getattr(config, key))
    state['error'] = error
    state = json.loads(json.dumps(state, default=str))
    return {key: compact(value) for key, value in state.items()}


def _steps(app):
    """The interactions: list of (name, function)."""
    venue, loudspeaker, array, algorithm, results = (
        app.pages[1], app.pages[2], app.pages[3], app.pages[4], app.pages[5])
    csv = base64.b64encode((ROOT / 'tests' / 'cases' / 'measured.csv')
                           .read_bytes()).decode()

    def setv(path, value, attr='value'):
        return lambda: setattr(_model(app, path), attr, value)

    def select(source, indices):
        return lambda: setattr(source.selected, 'indices', indices)

    def drag():
        data = algorithm.target_source.data
        algorithm.target_source.data = dict(
            distance_m=[5.0, 12.0, 18.0, 25.0],
            level_db=[0.0, -2.0, -4.0, -8.0],
            x_m=list(data['x_m']), y_m=list(data['y_m']))

    return [
        ('page1', setv('app.page_select', 1, 'active')),
        ('sx', setv('1.start_x_input', '2')),
        ('sy', setv('1.start_y_input', '0')),
        ('ex', setv('1.end_x_input', '10')),
        ('ey', setv('1.end_y_input', '1')),
        ('save', venue.save_line),
        ('save_same', venue.save_line),
        ('sx2', setv('1.start_x_input', '10')),
        ('sy2', setv('1.start_y_input', '1.5')),
        ('ex2', setv('1.end_x_input', '20')),
        ('ey2', setv('1.end_y_input', '4')),
        ('ppm_bad', setv('1.points_per_meter_input', 'abc')),
        ('save2', venue.save_line),
        ('ppm', setv('1.points_per_meter_input', '5')),
        ('sx3', setv('1.start_x_input', '20')),
        ('sy3', setv('1.start_y_input', '5')),
        ('ex3', setv('1.end_x_input', '30')),
        ('ey3', setv('1.end_y_input', '8')),
        ('save3', venue.save_line),
        ('axes1', setv('app.venue_axes_select', 1, 'active')),
        ('axes2', setv('app.venue_axes_select', 2, 'active')),
        ('axes0', setv('app.venue_axes_select', 0, 'active')),
        ('tap', select(venue.lines_source, [1])),
        ('sxi', setv('1.start_x_input', '5')),
        ('sxi2', setv('1.end_x_input', '9')),
        ('syi', setv('1.start_y_input', '-1')),
        ('eyi', setv('1.end_y_input', '-0.5')),
        ('insert', venue.insert_line),
        ('untap', select(venue.lines_source, [])),
        ('tap0', select(venue.lines_source, [1])),
        ('remove_sel', venue.remove_selected_line),
        ('untap2', select(venue.lines_source, [])),
        ('start_nocheck', setv('4.start_toggle', True, 'active')),
        ('check', app.check_venue),
        ('page2', setv('app.page_select', 2, 'active')),
        ('psi_bad', setv('2.aperture_input', 'x')),
        ('psi', setv('2.aperture_input', '4')),
        ('dir_line', setv('2.directivity_select', 'Line Piston')),
        ('dir_meas', setv('2.directivity_select',
                          'Measured Loudspeaker Data')),
        ('upload', setv('2.directivity_upload', csv)),
        ('dir_circ', setv('2.directivity_select', 'Circular Piston')),
        ('page3', setv('app.page_select', 3, 'active')),
        ('N', setv('3.cabinet_count_input', '8')),
        ('N_bad', setv('3.cabinet_count_input', '-3')),
        ('N2', setv('3.cabinet_count_input', '10')),
        ('yh_low', setv('3.top_y_input', '-50')),
        ('yh', setv('3.top_y_input', '6')),
        ('dang', setv('3.use_discrete_angles_select', 'Yes')),
        ('dang_vals', setv('3.discrete_angles_input',
                           '0,1,2,3,4,5,6,7,8,9,10')),
        ('dang_bad', setv('3.discrete_angles_input', 'a,b')),
        ('dang_vals2', setv('3.discrete_angles_input', '0,1,2,4,6,8,10')),
        ('ref_prog', setv('3.reference_type_select', 'Progressive')),
        ('ref_arc', setv('3.reference_type_select', 'Arc')),
        ('ref_step', setv('3.reference_step_stop_input', '3')),
        ('ref_dang', setv('3.reference_discrete_angles_select', 'Yes')),
        ('ref_user', setv('3.reference_type_select', 'User Defined')),
        ('ref_userv', setv('3.reference_user_angles_input', '1,2,3')),
        ('ref_userv2', setv('3.reference_user_angles_input',
                            '1,2,3,4,5,6,7,8,9,10')),
        ('ref_str', setv('3.reference_type_select', 'Straight')),
        ('dang_no', setv('3.use_discrete_angles_select', 'No')),
        ('page4', setv('app.page_select', 4, 'active')),
        ('gh_hm', setv('4.gap_handling_select', 'Hard Margin')),
        ('gh_sm', setv('4.gap_handling_select', 'Soft Margin')),
        ('sm', setv('4.soft_margin_slider', 1.5)),
        ('w_lin', setv('4.weighting_select', 'Linear Spacing')),
        ('stepsz', setv('4.nu_step_select', '0.1')),
        ('plus', algorithm.increase_nu),
        ('plus2', algorithm.increase_nu),
        ('minus', algorithm.decrease_nu),
        ('nu', setv('4.weighting_nu_slider', 1.3)),
        ('w_log', setv('4.weighting_select', 'Logarithmic Spacing')),
        ('fix', setv('4.use_fixed_angle_select', 'Yes')),
        ('fixv', setv('4.fixed_angle_input', '-100')),
        ('fixv2', setv('4.fixed_angle_input', 'zz')),
        ('fixv3', setv('4.fixed_angle_input', '2')),
        ('fix_no', setv('4.use_fixed_angle_select', 'No')),
        ('tol', setv('4.tolerance_slider', 0.5)),
        ('fr', setv('4.frequency_range_slider', (200.0, 8000.0))),
        ('w_none', setv('4.weighting_select', 'Without')),
        ('gh_none', setv('4.gap_handling_select', 'Without')),
        ('run', setv('4.start_toggle', True, 'active')),
        ('page5', setv('app.page_select', 5, 'active')),
        ('beam0', setv('5.beam_checkbox', [0], 'active')),
        ('beam01', setv('5.beam_checkbox', [0, 1], 'active')),
        ('fbeam', setv('5.beam_frequency_select', '2000')),
        ('beam1', setv('5.beam_checkbox', [1], 'active')),
        ('sfptap', select(results.venue_points_source, [2, 5])),
        ('sfpuntap', select(results.venue_points_source, [])),
        ('page4b', setv('app.page_select', 4, 'active')),
        ('w_ts', setv('4.weighting_select', 'Target Slope')),
        ('gh_hm2', setv('4.gap_handling_select', 'Hard Margin')),
        ('hinges', setv('4.hinges_select', '2')),
        ('tsel', lambda: algorithm.target_selected('indices', [], [0])),
        ('drag', drag),
        ('run_ts', setv('4.start_toggle', True, 'active')),
        ('clear', app.clear_result_figure),
        ('page1b', setv('app.page_select', 1, 'active')),
        ('remove_last', venue.remove_line),
        ('check2', app.check_venue),
        ('page0', setv('app.page_select', 0, 'active')),
    ]


def replay():
    """Replay the steps on a new app; yield (step name, state)."""
    app = PalcApp(Document())
    app.show()
    for name, step in _steps(app):
        error = None
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                step()
        except Exception as exc:  # recorded like in the golden file
            error = type(exc).__name__ + ':' + str(exc)[:100]
        yield name, snapshot(app, error)


@functools.lru_cache(maxsize=1)
def load_golden():
    """The golden states: list of (step name, state)."""
    steps = json.loads(GOLDEN.read_text())
    states, state = [], {}
    for name, changes in steps:
        state = dict(state, **changes)
        states.append((name, state))
    return states


def write_states(states, path):
    """Write states like the golden file (first state, then changes)."""
    out, previous = [], {}
    for name, state in states:
        assert set(state) >= set(previous)
        out.append([name, {k: v for k, v in state.items()
                           if k not in previous or previous[k] != v}])
        previous = state
    Path(path).write_text(json.dumps(out, indent=0, sort_keys=True) + '\n')


@pytest.fixture(scope="module")
def replayed():
    return list(replay())


def test_replays_all_steps(replayed):
    assert [name for name, _ in replayed] == \
        [name for name, _ in load_golden()]


@pytest.mark.parametrize("index", range(100))
def test_state_after_step(replayed, index):
    golden_name, golden = load_golden()[index]
    name, state = replayed[index]
    assert name == golden_name
    differences = sorted(key for key in set(state) | set(golden)
                         if state.get(key) != golden.get(key))
    assert not differences, (name, {key: (golden.get(key), state.get(key))
                                    for key in differences[:5]})


def test_failed_run_keeps_results(replayed):
    """A failing target slope run shows its error and changes nothing else.
    """
    states = dict(replayed)
    before, after = states['drag'], states['run_ts']
    assert after['app.config_error_info.text'] == \
        'Error: PALC calculation failed'
    assert after['4.start_toggle.active'] is False
    assert after['error'] is None
    changed = {key for key in after if after[key] != before[key]}
    assert changed == {'app.config_error_info.text'}


if __name__ == '__main__':
    write_states(list(replay()), sys.argv[1])
