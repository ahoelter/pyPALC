"""Saving and loading setup files in the GUI.

"Load Setup" (:func:`pypalc.gui.setup_io.apply_setup`) must leave a session
exactly as a user leaves it who enters the setup by hand, page by page, and
clicks "PALC Run" (the hand path of ``tests/test_gui_equivalence.py``,
clicked on page 4). Compared are all widgets, sources and figures of the
pages (:func:`state_of`) and the values the PALC configuration stores, for
every baseline case. Two differences are accepted (:data:`NOT_COMPARED`):

* ``app.page_select.active``: load switches to page 5 (the results).
* ``2.directivity_upload.value``: a browser file input cannot be filled by
  the server, so after a load the directivity upload shows no file. The
  data is restored in the session (``state.measured_directivity_csv``).
* ``4.load_setup_input.value``: the uploaded setup file itself.
* ``4.setup_file_source.data``: the text of the last "Save Setup" (the
  tests save from the shared loaded sessions).
* ``3.reference_angles_source.data`` of a new session holds each column as
  a nested list (one table row with all cabinets); load writes the
  computed table, as every change of the reference array does. The angles
  are compared after flattening (:func:`_flat_reference_table`).

Target slope: after a run the GUI snaps the shown target points to the
distance grid of the optimized array (see :mod:`pypalc.core.pipeline`), so
the target slope shown after a run is not the one that was run. For these
cases the round trip compares the setup of the run (``result.setup``) with
the loaded setup, and the shown setup with the one shown after the hand
path.
"""
import base64
import contextlib
import csv
import dataclasses
import functools
import io
import logging
from pathlib import Path

import numpy as np
import pytest
from bokeh.document import Document
from bokeh.events import ButtonClick
from bokeh.model import Model

from pypalc.cli import main
from pypalc.core import pipeline
from pypalc.core.errors import ConvergenceError, SetupError, VenueError
from pypalc.core.palc import clamp_fixed_angle
from pypalc.core.pipeline import fixed_angle_limits, result_arrays, run
from pypalc.core.settings import (AudienceLine, MeasuredDirectivity,
                                  dump_setup, load_setup, load_setup_text)
from pypalc.gui.app import PalcApp
from pypalc.gui.setup_io import apply_setup, setup_from_app
from tests.baseline_cases import CASES
from tests.test_gui_cascade import CONFIG_KEYS, PROPS, _norm, compact
from tests.test_gui_equivalence import drive, enter

ROOT = Path(__file__).resolve().parent.parent
TARGET_SLOPE_CASES = ("target_slope", "gap_soft_margin_target_slope")
#: Values that differ on purpose between load and the hand path.
NOT_COMPARED = ("app.page_select.active", "2.directivity_upload.value",
                "4.load_setup_input.value", "4.setup_file_source.data")
PATH_KEY = "loudspeaker.measured_directivity.path"


def case_setup(case):
    """The setup of a case, with measured directivity embedded as csv."""
    setup = load_setup(ROOT / "tests" / "cases" / ("%s.yaml" % case))
    md = setup.loudspeaker.measured_directivity
    if md is not None and md.path is not None:
        text = Path(md.path).read_bytes().decode("utf-8")
        setup = dataclasses.replace(setup, loudspeaker=dataclasses.replace(
            setup.loudspeaker, measured_directivity=MeasuredDirectivity(
                csv=text)))
    return setup


def case_text(case):
    """YAML text of a case as a GUI user would load it."""
    return dump_setup(case_setup(case))


def quiet(function, *args):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args)


def new_app():
    app = PalcApp(Document())
    app.show()
    return app


def load(app, text):
    """Upload a setup file through the "Load Setup" input."""
    quiet(setattr, app.pages[4].load_setup_input, "value",
          base64.b64encode(text.encode("utf-8")).decode("ascii"))


def models_of(app):
    """Bokeh models of the app and its pages: name -> model."""
    owners = [("app", app)] + [(str(i), p) for i, p in app.pages.items()]
    return {"%s.%s" % (owner, name): value
            for owner, obj in owners for name, value in vars(obj).items()
            if isinstance(value, Model)}


def state_of(app):
    """Widget, source and figure state of the app and session values."""
    state = {}
    for path, model in sorted(models_of(app).items()):
        for prop in PROPS + ('width',):
            if prop in model.properties():
                state[path + '.' + prop] = compact(_norm(getattr(model,
                                                                 prop)))
        if 'selected' in model.properties():
            state[path + '.selected'] = _norm(model.selected.indices)
    st = app.state
    for key in CONFIG_KEYS:
        state['config.' + key] = _norm(getattr(st.config, key))
    state['state.audience_lines'] = list(st.audience_lines)
    state['state.measured_directivity_csv'] = st.measured_directivity_csv
    state['state.fixed_angle_range'] = _norm(st.optimized_array.fixed_angle)
    state['state.reference_tilt_angles'] = _norm(
        st.reference_array.tilt_angles_deg)
    for name in ('initialized', 'distances', 'levels_db', 'reference_distance', 'region_distances', 'region_levels_db'):
        state['state.target.' + name] = compact(_norm(getattr(st.target,
                                                              name)))
    for key in NOT_COMPARED:
        state.pop(key)
    key = '3.reference_angles_source.data'
    state[key] = _flat_reference_table(state[key])
    return state


def _flat_reference_table(data):
    """The reference table with the start-up nesting removed."""
    return {name: column[0] if len(column) == 1 and
            isinstance(column[0], list) else column
            for name, column in data.items()}


def differences(a, b):
    keys = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    return {k: (a.get(k), b.get(k)) for k in keys[:8]}


@functools.lru_cache(maxsize=None)
def by_hand(case):
    """A session after entering the case by hand and clicking Run."""
    app = new_app()
    app.page_select.active = 4
    quiet(drive, app, case_setup(case))
    return app


@functools.lru_cache(maxsize=None)
def loaded(case):
    """A new session after loading the case."""
    app = new_app()
    app.page_select.active = 4
    load(app, case_text(case))
    return app


@pytest.mark.parametrize("case", CASES)
def test_load_shows_results(case):
    app = loaded(case)
    assert app.config_error_info.text == ''
    assert app.page_select.active == 5
    assert app.state.result is not None
    assert app.pages[4].start_toggle.active is False
    assert app.state.applying_setup is False


@pytest.mark.parametrize("case", CASES)
def test_apply_then_extract_round_trip(case):
    setup = load_setup_text(case_text(case))
    app = loaded(case)
    assert app.state.result.setup == setup
    if case in TARGET_SLOPE_CASES:
        assert setup_from_app(app) == setup_from_app(by_hand(case))
    else:
        assert setup_from_app(app) == setup


@pytest.mark.parametrize("case", CASES)
def test_apply_setup_results_equal_headless(case):
    got = result_arrays(loaded(case).state.result)
    want = result_arrays(run(load_setup_text(case_text(case))))
    assert sorted(got) == sorted(want)
    for key in want:
        np.testing.assert_array_equal(got[key], want[key], err_msg=key)


@pytest.mark.parametrize("case", CASES)
def test_apply_setup_matches_hand_path(case):
    hand, applied = state_of(by_hand(case)), state_of(loaded(case))
    assert not differences(hand, applied)


def dirty_app():
    """A session in which a lot was entered before a setup is loaded."""
    app = new_app()
    app.page_select.active = 4
    quiet(drive, app, case_setup("gap_soft_margin_target_slope"))
    loudspeaker, array = app.pages[2], app.pages[3]
    loudspeaker.directivity_select.value = "Measured Loudspeaker Data"
    loudspeaker.directivity_upload.value = base64.b64encode(
        (ROOT / "tests" / "cases" / "measured.csv").read_bytes()).decode()
    array.use_discrete_angles_select.value = 'Yes'
    array.reference_type_select.value = 'Arc'
    array.reference_step_stop_input.value = '2'
    array.reference_type_select.value = 'User Defined'
    app.pages[1].lines_source.selected.indices = [0]
    app.pages[5].venue_points_source.selected.indices = [1]
    app.pages[4].tabs.active = 0
    return app


@pytest.mark.parametrize("case", CASES)
def test_apply_setup_replaces_previous_session(case):
    app = dirty_app()
    load(app, case_text(case))
    got, want = state_of(app), state_of(loaded(case))
    assert not differences(got, want)


def test_apply_setup_rejects_path_directivity():
    app = new_app()
    app.page_select.active = 4
    quiet(drive, app, case_setup("simple"))
    before, setup_before = state_of(app), setup_from_app(app)
    text = (ROOT / "tests" / "cases" / "measured.yaml").read_text()
    load(app, text)
    error = app.config_error_info.text
    assert error.startswith(
        "Error: setup file invalid at %s: " % PATH_KEY), error
    assert "Save Setup" in error and "python -m pypalc run" in error
    assert setup_from_app(app) == setup_before
    after = state_of(app)
    after.pop('app.config_error_info.text')
    before.pop('app.config_error_info.text')
    assert not differences(before, after)


def test_apply_setup_raises_for_path_directivity():
    app = new_app()
    state, before = app.state, state_of(app)
    setup = load_setup(ROOT / "tests" / "cases" / "measured.yaml")
    with pytest.raises(SetupError) as info:
        apply_setup(app, setup)
    assert info.value.key == PATH_KEY
    assert app.state is state
    assert app.state.applying_setup is False
    assert not differences(before, state_of(app))


def assert_load_rejected(text, key):
    app = new_app()
    app.page_select.active = 4
    quiet(drive, app, case_setup("fixed_angle"))
    before, setup_before = state_of(app), setup_from_app(app)
    load(app, text)
    error = app.config_error_info.text
    assert error.startswith("Error: setup file invalid at %s: " % key), error
    assert setup_from_app(app) == setup_before
    after = state_of(app)
    after.pop('app.config_error_info.text')
    before.pop('app.config_error_info.text')
    assert not differences(before, after)


def test_load_rejects_invalid_yaml():
    assert_load_rejected("algorithm: {variant: palc9}\n",
                         "algorithm.variant")


def test_load_rejects_unusable_venue():
    setup = case_setup("simple")
    line = setup.venue.audience_lines[0]
    lines = (line, AudienceLine(start=line.stop, stop=line.start))
    assert_load_rejected(dump_setup(dataclasses.replace(
        setup, venue=dataclasses.replace(setup.venue,
                                         audience_lines=lines))),
        "venue.audience_lines")


def test_load_rejects_setup_without_lines():
    assert_load_rejected("version: 1\n", "venue.audience_lines")


def _setup_with_reversed_line():
    setup = case_setup("simple")
    line = setup.venue.audience_lines[0]
    lines = (line, AudienceLine(start=line.stop, stop=line.start))
    return dataclasses.replace(setup, venue=dataclasses.replace(
        setup.venue, audience_lines=lines))


def test_apply_setup_keeps_cause_of_rejection():
    with pytest.raises(SetupError) as info:
        apply_setup(new_app(), _setup_with_reversed_line())
    assert info.value.key == "venue.audience_lines"
    assert isinstance(info.value.__cause__, VenueError)


def test_load_logs_rejection_with_traceback_at_debug_level(caplog):
    app = new_app()
    with caplog.at_level(logging.DEBUG, logger="pypalc.gui.page_algorithm"):
        load(app, dump_setup(_setup_with_reversed_line()))
    assert app.config_error_info.text.startswith(
        "Error: setup file invalid at venue.audience_lines: ")
    debug = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert len(debug) == 1 and debug[0].exc_info is not None
    assert isinstance(debug[0].exc_info[1].__cause__, VenueError)


def test_load_rejects_empty_discrete_angles():
    setup = case_setup("discrete_angles")
    assert_load_rejected(dump_setup(dataclasses.replace(
        setup, array=dataclasses.replace(setup.array,
                                         discrete_tilt_angles_deg=()))),
        "array.discrete_tilt_angles_deg")


def test_load_rejects_frequency_range_outside_slider():
    setup = case_setup("simple")
    assert_load_rejected(dump_setup(dataclasses.replace(
        setup, algorithm=dataclasses.replace(
            setup.algorithm, frequency_range_hz=(-100.0, 50000.0)))),
        "algorithm.frequency_range_hz")


def test_load_rejects_non_utf8_file():
    app = new_app()
    before = state_of(app)
    quiet(setattr, app.pages[4].load_setup_input, "value",
          base64.b64encode(b"\xff\xfe\x00bad").decode("ascii"))
    assert app.config_error_info.text.startswith(
        "Error: setup file invalid at file: ")
    after = state_of(app)
    after.pop('app.config_error_info.text')
    before.pop('app.config_error_info.text')
    assert not differences(before, after)


def test_failed_run_after_load_shows_error(monkeypatch):
    def fail(*args, **kwargs):
        raise ConvergenceError('Error: No convergence reached')

    app = new_app()
    monkeypatch.setattr(pipeline, "run", fail)
    load(app, case_text("simple"))
    assert app.config_error_info.text == 'Error: No convergence reached'
    assert app.page_select.active == 4
    assert app.state.result is None
    assert app.state.applying_setup is False
    assert setup_from_app(app) == load_setup_text(case_text("simple"))


def _failing_text(case):
    """A case that loads but cannot be calculated (no convergence)."""
    setup = case_setup(case)
    return dump_setup(dataclasses.replace(setup, loudspeaker=dataclasses.replace(
        setup.loudspeaker, aperture_angle_deg=-5.0)))


@pytest.mark.parametrize("previous, failing", [
    ("simple", "simple"), ("target_slope", "simple"),
    ("simple", "target_slope")])
def test_failed_load_clears_previous_results(previous, failing):
    """After a failed load the session shows what a new session shows
    after the same failed load: no results of the previous setup."""
    app = new_app()
    app.page_select.active = 4
    load(app, case_text(previous))
    assert app.state.result is not None
    load(app, _failing_text(failing))
    assert app.config_error_info.text == 'Error: No convergence reached'
    assert app.state.result is None
    fresh = new_app()
    fresh.page_select.active = 4
    load(fresh, _failing_text(failing))
    assert not differences(state_of(app), state_of(fresh))
    assert len(app.pages[4].results_source.data['tilt_angle_deg']) == 0
    # the target slope still snaps when dragged after the clearing
    assert app.pages[4].snap_target in \
        app.pages[4].target_source._callbacks['data']


def test_failed_run_keeps_previous_results(monkeypatch):
    app = new_app()
    app.page_select.active = 4
    load(app, case_text("simple"))
    results = dict(app.pages[4].results_source.data)
    previous = app.state.result

    def fail(*args, **kwargs):
        raise ConvergenceError('Error: No convergence reached')

    monkeypatch.setattr(pipeline, "run", fail)
    app.pages[4].start_toggle.active = True
    assert app.config_error_info.text == 'Error: No convergence reached'
    assert app.state.result is previous
    np.testing.assert_array_equal(
        app.pages[4].results_source.data['tilt_angle_deg'],
        results['tilt_angle_deg'])


def saved_text(app):
    """Click "Save Setup" and return the text the browser downloads."""
    algorithm = app.pages[4]
    quiet(algorithm.save_setup_button._trigger_event,
          ButtonClick(algorithm.save_setup_button))
    texts = algorithm.setup_file_source.data['text']
    assert len(texts) == 1
    return texts[0]


def test_save_writes_yaml_to_download_source():
    app = new_app()
    quiet(enter, app, case_setup("measured"))
    text = saved_text(app)
    assert load_setup_text(text) == setup_from_app(app)
    assert load_setup_text(text).loudspeaker.measured_directivity.csv == \
        (ROOT / "tests" / "cases" / "measured.csv").read_bytes().decode()


def test_save_twice_downloads_twice():
    app = new_app()
    algorithm = app.pages[4]
    changes = []
    algorithm.setup_file_source.on_change(
        'data', lambda attr, old, new: changes.append(list(new['text'])))
    saved_text(app)
    saved_text(app)
    assert [c for c in changes if c] == [[dump_setup(setup_from_app(app))]] * 2


def test_save_shows_error_for_invalid_input():
    app = new_app()
    app.pages[2].directivity_select.value = "Measured Loudspeaker Data"
    quiet(app.pages[4].save_setup_button._trigger_event,
          ButtonClick(app.pages[4].save_setup_button))
    assert app.config_error_info.text.startswith(
        "Error: setup not saved, invalid input at "
        "loudspeaker.measured_directivity: ")
    assert app.pages[4].setup_file_source.data['text'] == []


@pytest.mark.parametrize("case", CASES)
def test_saved_setup_runs_headless(case, tmp_path):
    app = loaded(case)
    path = tmp_path / "pypalc_setup.yaml"
    path.write_text(saved_text(app), encoding="utf-8")
    assert quiet(main, ["run", str(path), "--out", str(tmp_path)]) == 0
    with open(str(tmp_path / "tilt_angles.csv"), newline="") as handle:
        rows = list(csv.reader(handle))
    angles = [float(row[1]) for row in rows[1:]]
    assert angles == list(app.pages[4].results_source.data[
        'tilt_angle_deg'])


def test_target_slope_restored():
    app = loaded("target_slope")
    setup = load_setup_text(case_text("target_slope"))
    algorithm = app.pages[4]
    points = setup.target_slope.points
    assert algorithm.hinges_select.value == str(len(points) - 2)
    assert algorithm.hinges_select.visible
    assert app.state.result.setup.target_slope == setup.target_slope
    # shown on the distance grid of the run, levels as in the file
    grid = np.array(algorithm.spl_over_distance_source.data['distance_m'])
    shown = algorithm.target_source.data
    assert list(shown['level_db']) == [p.level_db for p in points]
    assert list(shown['distance_m']) == [
        grid[np.argmin(np.abs(grid - p.distance))] for p in points]


# --------------------------------------------------------------------------
# Run and fixed angle (carried over from the review of the GUI's run)
# --------------------------------------------------------------------------

def test_failed_display_keeps_previous_result(monkeypatch):
    app = new_app()
    app.page_select.active = 4
    quiet(drive, app, case_setup("simple"))
    previous = app.state.result

    def fail(result):
        raise ValueError("display failed")

    monkeypatch.setattr(app, "_show_result", fail)
    app.pages[4].start_toggle.active = True
    assert app.config_error_info.text == 'Error: Could not store the output'
    assert app.state.result is previous


def test_fixed_angle_follows_top_position():
    app = new_app()
    quiet(enter, app, case_setup("fixed_angle"))
    algorithm, array = app.pages[4], app.pages[3]
    array.top_y_input.value = '20'
    max_deg, min_deg = fixed_angle_limits(app.state.venue,
                                          setup_from_app(app))
    assert algorithm.fixed_angle_text.text == 'Max: %s, Min: %s' % (
        np.round(max_deg, decimals=2), np.round(min_deg, decimals=2))
    shown = float(algorithm.fixed_angle_input.value)
    assert shown == app.state.config.fixed_angle
    assert shown == clamp_fixed_angle(-4.93, max_deg, min_deg) != -4.93
    assert setup_from_app(app).algorithm.fixed_top_angle_deg == shown
    assert clamp_fixed_angle(shown, max_deg, min_deg) == shown


# --------------------------------------------------------------------------
# Saving and loading a target slope again and again
# --------------------------------------------------------------------------

def dense(case, points_per_meter):
    """A case with all audience lines discretized more densely."""
    setup = case_setup(case)
    lines = tuple(dataclasses.replace(line, points_per_meter=float(
        points_per_meter)) for line in setup.venue.audience_lines)
    return dataclasses.replace(setup, venue=dataclasses.replace(
        setup.venue, audience_lines=lines))


def load_and_save(text):
    """Load a setup file into a new session; the session and its save."""
    app = new_app()
    load(app, text)
    assert app.config_error_info.text == '', app.config_error_info.text
    return app, saved_text(app)


@pytest.mark.parametrize("points_per_meter", [None, 20, 40])
@pytest.mark.parametrize("case", TARGET_SLOPE_CASES)
def test_repeated_save_and_load_is_stable(case, points_per_meter):
    setup = case_setup(case) if points_per_meter is None else \
        dense(case, points_per_meter)
    _, a = load_and_save(dump_setup(setup))
    app_a, b = load_and_save(a)
    app_b, c = load_and_save(b)
    assert load_setup_text(b) == load_setup_text(a)
    assert load_setup_text(c) == load_setup_text(b)
    got = result_arrays(app_b.state.result)
    want = result_arrays(app_a.state.result)
    for key in want:
        np.testing.assert_array_equal(got[key], want[key], err_msg=key)


def test_save_after_dragging_writes_dragged_points():
    app = new_app()
    load(app, case_text("target_slope"))
    algorithm = app.pages[4]
    ts = app.state.result.setup.target_slope
    grid = list(algorithm.spl_over_distance_source.data['distance_m'])
    moved = (ts.points[0], dataclasses.replace(ts.points[1],
                                               distance=grid[20],
                                               level_db=-2.0),
             ts.points[2])
    quiet(algorithm.drag_target, moved, ts.reference_distance)
    saved = load_setup_text(saved_text(app)).target_slope
    assert saved == setup_from_app(app).target_slope
    assert saved != ts
    assert saved.points[1].distance == grid[20]
    assert saved.points[1].level_db == -2.0
