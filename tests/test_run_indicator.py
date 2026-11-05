"""The running look of "26. PALC Run": while a calculation runs (started
by the toggle or by "Load Setup") the toggle reads "Running..." and is dark
red; a second start is ignored; the look is reset in every case.

While it runs the toggle is also disabled: the browser does not block key
presses (Space, Enter) on the focused toggle otherwise, and the server
would handle them after the run, starting a second calculation."""
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
    # enabled again (the venue of the cases is PALC compatible)
    assert read(toggle, 'disabled') is False


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


def test_run_ignored_while_load_pending(monkeypatch):
    """Served, the load runs later (:meth:`PalcApp.later`); a click on
    "PALC Run" before it ran, while no setup is applied yet, is ignored."""
    app = _entered()
    toggle = app.pages[4].start_toggle
    looks = _watch_runs(monkeypatch, toggle)
    pending = []
    monkeypatch.setattr(app, 'later', pending.append)
    text = dump_setup(load_setup(case_path('target_slope')))
    upload(app.pages[4].load_setup_input, text.encode('utf-8'),
           'pypalc_setup.yaml')
    assert len(pending) == 1
    assert app.pages[4].running
    assert not app.state.applying_setup
    assert _look(toggle) == (RUNNING_LABEL, RUNNING)
    write(toggle, 'active', True)
    assert len(pending) == 1
    assert looks == []
    with contextlib.redirect_stdout(io.StringIO()):
        pending.pop()()
    assert read(app.config_error_info, 'text') == ''
    # the load alone: the initial run and the target slope run
    assert len(looks) == 2
    assert app.state.result.setup.target_slope is not None
    assert not pending
    _assert_idle(app)


def test_run_toggle_disabled_while_running(monkeypatch):
    """Disabled during the run, also when the venue is checked during it
    (as Load Setup does); enabled again afterwards."""
    app = _entered()
    toggle = app.pages[4].start_toggle
    assert read(toggle, 'disabled') is False
    disabled = []

    def check_venue():
        disabled.append(read(toggle, 'disabled'))
        app.set_venue_checked(True)
        disabled.append(read(toggle, 'disabled'))
    _watch_runs(monkeypatch, toggle, check_venue)
    write(toggle, 'active', True)
    assert disabled == [True, True]
    _assert_idle(app)


def test_run_toggle_disabled_while_loading(monkeypatch):
    """Disabled during a load and while the load is pending."""
    app = PalcApp(Document())
    toggle = app.pages[4].start_toggle
    assert read(toggle, 'disabled') is True  # no venue yet
    disabled = []
    _watch_runs(monkeypatch, toggle,
                lambda: disabled.append(read(toggle, 'disabled')))
    pending = []
    monkeypatch.setattr(app, 'later', pending.append)
    text = dump_setup(load_setup(case_path('simple')))
    upload(app.pages[4].load_setup_input, text.encode('utf-8'),
           'pypalc_setup.yaml')
    assert read(toggle, 'disabled') is True
    with contextlib.redirect_stdout(io.StringIO()):
        pending.pop()()
    assert disabled == [True]
    _assert_idle(app)


def test_run_toggle_stays_disabled_without_venue():
    """After a run, the toggle is enabled only if the venue is PALC
    compatible."""
    app = PalcApp(Document())
    page = app.pages[4]
    page.set_running(True)
    page.enable_run(True)
    assert read(page.start_toggle, 'disabled') is True
    page.enable_run(False)
    page.set_running(False)
    assert read(page.start_toggle, 'disabled') is True
    page.enable_run(True)
    assert read(page.start_toggle, 'disabled') is False
