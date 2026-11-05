"""The running look of "26. PALC Run": while a calculation runs (started
by the toggle or by "Load Setup") the toggle reads "Running..." and is dark
red; a second start is ignored; the look is reset in every case."""
import contextlib
import io

from bokeh.document import Document

from pypalc.core import pipeline
from pypalc.core.errors import ConvergenceError
from pypalc.core.settings import dump_setup, load_setup
from pypalc.gui.app import PalcApp
from pypalc.gui.page_algorithm import RUN_LABEL, RUNNING_LABEL
from pypalc.gui.style import RUNNING
from tests.baseline_cases import case_path
from tests.gui_driver import read, upload, write
from tests.test_gui_equivalence import enter


def _look(toggle):
    return read(toggle, 'label'), toggle.sx.get('backgroundColor')


def _entered(case='simple'):
    app = PalcApp(Document())
    with contextlib.redirect_stdout(io.StringIO()):
        enter(app, load_setup(case_path(case)))
    return app


def _watch_runs(monkeypatch, toggle, during=None):
    """Patch :func:`pipeline.run` to record the look of the toggle at each
    call (and to call ``during`` inside the first one)."""
    looks = []
    run = pipeline.run

    def watched(*args, **kwargs):
        looks.append(_look(toggle))
        if during is not None and len(looks) == 1:
            during()
        return run(*args, **kwargs)
    monkeypatch.setattr(pipeline, 'run', watched)
    return looks


def _assert_idle(app):
    toggle = app.pages[4].start_toggle
    label, color = _look(toggle)
    assert label == RUN_LABEL == '26. PALC Run'
    assert color != RUNNING
    assert read(toggle, 'active') is False
    assert app.pages[4].running is False


def test_running_look_during_run(monkeypatch):
    app = _entered()
    toggle = app.pages[4].start_toggle
    _assert_idle(app)
    looks = _watch_runs(monkeypatch, toggle)
    write(toggle, 'active', True)
    assert looks == [(RUNNING_LABEL, RUNNING)]
    assert RUNNING_LABEL == 'Running...' and RUNNING == '#8b0000'
    assert app.state.result is not None
    _assert_idle(app)


def test_run_ignored_while_running(monkeypatch):
    app = _entered()
    toggle = app.pages[4].start_toggle

    def click_twice():
        # the browser toggles the value on each click
        write(toggle, 'active', False)
        write(toggle, 'active', True)
    looks = _watch_runs(monkeypatch, toggle, click_twice)
    write(toggle, 'active', True)
    assert len(looks) == 1
    _assert_idle(app)


def test_running_look_resets_after_error(monkeypatch):
    app = _entered()
    toggle = app.pages[4].start_toggle
    looks = []

    def fail(*args, **kwargs):
        looks.append(_look(toggle))
        raise ConvergenceError('Error: No convergence reached')
    monkeypatch.setattr(pipeline, 'run', fail)
    write(toggle, 'active', True)
    assert looks == [(RUNNING_LABEL, RUNNING)]
    assert read(app.config_error_info, 'text') == \
        'Error: No convergence reached'
    _assert_idle(app)


def test_running_look_during_load(monkeypatch):
    app = PalcApp(Document())
    toggle = app.pages[4].start_toggle
    looks = _watch_runs(monkeypatch, toggle,
                        lambda: write(toggle, 'active', True))
    text = dump_setup(load_setup(case_path('simple')))
    with contextlib.redirect_stdout(io.StringIO()):
        upload(app.pages[4].load_setup_input, text.encode('utf-8'),
               'pypalc_setup.yaml')
    # the click on "PALC Run" during the load started no calculation
    assert looks == [(RUNNING_LABEL, RUNNING)]
    assert read(app.config_error_info, 'text') == ''
    assert read(app.page_select, 'active') == 5
    _assert_idle(app)


def test_load_ignored_while_running(monkeypatch):
    app = _entered()
    toggle = app.pages[4].start_toggle
    load = app.pages[4].load_setup_input
    text = dump_setup(load_setup(case_path('target_slope')))
    looks = _watch_runs(monkeypatch, toggle, lambda: upload(
        load, text.encode('utf-8'), 'pypalc_setup.yaml'))
    write(toggle, 'active', True)
    assert len(looks) == 1
    assert app.state.result.setup.target_slope is None
    # emptied, so that selecting the file again loads it
    assert load.value is None
    _assert_idle(app)
