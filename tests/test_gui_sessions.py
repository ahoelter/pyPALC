"""Every browser session gets its own GUI state."""
import contextlib
import importlib
import io
import pkgutil
from pathlib import Path

import numpy as np
import panel as pn
import pytest
from bokeh.document import Document
from bokeh.model import Model
from panel.viewable import Viewable

import pypalc.gui
from pypalc.core import pipeline
from pypalc.core.pipeline import result_arrays, run
from pypalc.core.settings import load_setup
from pypalc.gui.app import PalcApp
from tests.baseline_cases import assert_matches_baseline, baseline_path

ROOT = Path(__file__).resolve().parent.parent


def _add_line(app, start, stop, points_per_meter=5):
    venue = app.pages[1]
    venue.start_x_input.value = repr(float(start[0]))
    venue.end_x_input.value = repr(float(stop[0]))
    venue.points_per_meter_input.value = repr(float(points_per_meter))
    venue.start_y_input.value = repr(float(start[1]))
    venue.end_y_input.value = repr(float(stop[1]))
    venue.save_line()


def test_sessions_do_not_share_state():
    a = PalcApp(Document())
    b = PalcApp(Document())
    _add_line(a, (2, 0), (10, 1))
    assert len(a.state.drawn_lines.line_start_x) == 1
    assert len(b.state.drawn_lines.line_start_x) == 0
    assert len(b.state.drawn_lines.points_x) == 0
    assert b.pages[1].lines_source.data['lines_x'] == []
    assert a.pages[1].lines_source is not b.pages[1].lines_source
    assert a.state.config is not b.state.config
    assert a.venue_figure is not b.venue_figure


def test_no_module_level_panel_objects():
    modules = [pypalc.gui] + [
        importlib.import_module('pypalc.gui.' + info.name)
        for info in pkgutil.iter_modules(pypalc.gui.__path__)]
    assert len(modules) > 5
    for module in modules:
        for name, value in vars(module).items():
            assert not isinstance(value, (Model, Viewable)), (
                module.__name__, name)
    for _ in range(2):
        PalcApp(Document()).view()
    assert pn.state.cache == {}


def test_run_calculation_matches_pipeline():
    setup = load_setup(ROOT / 'tests' / 'cases' / 'simple.yaml')
    a = PalcApp(Document())
    for line in setup.venue.audience_lines:
        _add_line(a, line.start, line.stop, line.points_per_meter)
    a.check_venue()
    with contextlib.redirect_stdout(io.StringIO()):
        a.run_calculation()
    assert a.config_error_info.object == ''
    data = a.pages[4].results_source.data
    # baseline tolerance: BLAS builds differ in the last digits
    with np.load(baseline_path('simple')) as baseline:
        assert_matches_baseline(np.array(data['tilt_angle_deg'], dtype=float),
                                baseline['tilt_angles_deg'])
        assert_matches_baseline(np.array(data['inter_cabinet_angle_deg'],
                                         dtype=float),
                                baseline['inter_cabinet_angles_deg'])


def _app_with_venue(case='simple'):
    setup = load_setup(ROOT / 'tests' / 'cases' / ('%s.yaml' % case))
    app = PalcApp(Document())
    for line in setup.venue.audience_lines:
        _add_line(app, line.start, line.stop, line.points_per_meter)
    app.check_venue()
    return app, setup


def test_run_calculation_stores_result():
    app, setup = _app_with_venue()
    assert app.state.result is None
    with contextlib.redirect_stdout(io.StringIO()):
        app.run_calculation()
    gui = result_arrays(app.state.result)
    headless = result_arrays(run(setup))
    assert sorted(gui) == sorted(headless)
    for key in headless:
        np.testing.assert_array_equal(gui[key], headless[key], err_msg=key)


def test_failed_calculation_shows_error_and_keeps_results():
    app, _ = _app_with_venue()
    with contextlib.redirect_stdout(io.StringIO()):
        app.run_calculation()
    result = app.state.result
    angles = dict(app.pages[4].results_source.data)
    array = app.pages[3]
    array.use_discrete_angles_select.value = 'Yes'
    array.discrete_angles_input.value = 'a,b'
    algorithm = app.pages[4]
    algorithm.start_toggle.value = True
    assert algorithm.start_toggle.value is False
    assert app.config_error_info.object.startswith(
        'Error: array.discrete_tilt_angles_deg')
    assert app.state.result is result
    assert algorithm.results_source.data == angles


def test_unexpected_error_releases_start_toggle(monkeypatch, caplog):
    app, _ = _app_with_venue()

    def fail(setup, venue=None):
        raise RuntimeError('unexpected')

    monkeypatch.setattr(pipeline, 'run', fail)
    algorithm = app.pages[4]
    with pytest.raises(RuntimeError):
        algorithm.start_toggle.value = True
    assert algorithm.start_toggle.value is False
    assert any(record.exc_info and record.levelname == 'ERROR'
               for record in caplog.records)
