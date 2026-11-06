"""Callbacks of the GUI run while the session holds the document lock
(:func:`pypalc.gui.widgets.locked`).

Panel runs a browser event (a click, Enter) without the document lock
when a value change of a widget is still waiting for its debounce; a
callback that changes a Bokeh model then fails and the typed value or the
click is lost. Every callback of the app that Panel calls is therefore
registered through :func:`~pypalc.gui.widgets.watch` or
:func:`~pypalc.gui.widgets.on_click`.
"""
import inspect
import re
from pathlib import Path
from types import SimpleNamespace

import param
import pytest
from bokeh.document import Document
from bokeh.server.session import ServerSession
from panel.io.state import set_curdoc

from pypalc.gui.app import PalcApp
from pypalc.gui.widgets import LOCKED_MARK, locked, on_click, watch
from tests.test_shell import rendered

GUI = Path(__file__).resolve().parent.parent / 'pypalc' / 'gui'


class FakeDocument:
    """A served document whose session holds its lock or not."""

    def __init__(self, pending_writes):
        session = SimpleNamespace(_pending_writes=pending_writes)
        self.session_context = SimpleNamespace(session=session)
        self.next_tick = []

    @property
    def session(self):
        return self.session_context.session

    def add_next_tick_callback(self, callback):
        self.next_tick.append(callback)

    def run_next_ticks(self):
        """Run the next tick callbacks as Bokeh does: in order, each with
        the lock held."""
        while self.next_tick:
            callback = self.next_tick.pop(0)
            self.session._pending_writes = []
            try:
                callback()
            finally:
                self.session._pending_writes = None


def _watcher_function(watcher):
    """The function a watcher calls (the method of a ``param.depends``
    watcher)."""
    fn = watcher.fn
    return getattr(fn, 'keywords', {}).get('function', fn)


def _app_objects(app):
    """The Parameterized objects of a session: the attributes of the app
    and its pages and every object of the view and the page layouts."""
    found = {}

    def add(obj):
        if isinstance(obj, param.Parameterized):
            found.setdefault(id(obj), obj)
    for owner in [app, *app.pages.values()]:
        for value in vars(owner).values():
            add(value)
    for root in [app.view(), *(page.layout() for page in app.pages.values())]:
        for obj in root.select():
            add(obj)
    return list(found.values())


def test_every_callback_of_the_app_runs_locked():
    app = rendered(PalcApp(Document()))
    unlocked, registered = [], 0
    for obj in _app_objects(app):
        for name, whats in obj.param.watchers.items():
            for watcher in (w for ws in whats.values() for w in ws):
                fn = _watcher_function(watcher)
                if not getattr(fn, '__module__', '').startswith('pypalc'):
                    continue  # Panel's own watchers
                registered += 1
                if not getattr(fn, LOCKED_MARK, False):
                    unlocked.append('%s.%s: %s' % (type(obj).__name__, name,
                                                   fn))
    assert unlocked == []
    # the walk finds the registrations: the pages register more than 40
    assert registered > 40


def test_download_buttons_make_their_file_locked():
    algorithm = PalcApp(Document()).pages[4]
    for button in (algorithm.download_angles_button,
                   algorithm.download_results_button,
                   algorithm.save_setup_button):
        functions = [_watcher_function(w)
                     for w in button.param.watchers['_clicks']['value']]
        # Panel's own _transfer watcher is replaced by a locked one
        assert button._transfer not in functions
        assert [getattr(f, LOCKED_MARK, False) for f in functions
                if getattr(f, '__module__', '').startswith('pypalc')] == [True]


def test_gui_registers_callbacks_only_through_the_helpers():
    pattern = re.compile(r'\.param\.watch\(|\.on_click\(')
    found = []
    for path in sorted(GUI.glob('*.py')):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if pattern.search(line):
                found.append('%s:%d' % (path.name, number))
    # only the helpers themselves
    assert [f.split(':')[0] for f in found] == ['widgets.py', 'widgets.py']


def test_locked_runs_at_once_without_a_session():
    calls = []
    run = locked(lambda *args, **kwargs: calls.append((args, kwargs)))
    run(1, key=2)
    assert calls == [((1,), {'key': 2})]
    assert getattr(run, LOCKED_MARK)


def test_locked_runs_at_once_while_the_lock_is_held():
    doc, calls = FakeDocument(pending_writes=[]), []
    with set_curdoc(doc):
        locked(calls.append)('event')
    assert calls == ['event'] and doc.next_tick == []


def test_locked_runs_at_once_for_an_unknown_session():
    doc, calls = FakeDocument(pending_writes=None), []
    del doc.session._pending_writes
    with set_curdoc(doc):
        locked(calls.append)('event')
    assert calls == ['event'] and doc.next_tick == []


def test_locked_schedules_the_call_without_the_lock_in_order():
    doc, calls = FakeDocument(pending_writes=None), []

    def change(event):
        calls.append(('change', event))
        # a callback that sets a widget: its watcher runs at once, inside
        nested('nested')

    nested = locked(lambda event: calls.append(('nested', event)))
    with set_curdoc(doc):
        # the waiting value change, then the click (Panel's event path)
        locked(change)('typed')
        locked(lambda event: calls.append(('click', event)))('clicked')
        assert calls == []
        doc.run_next_ticks()
    assert calls == [('change', 'typed'), ('nested', 'nested'),
                     ('click', 'clicked')]


def test_watch_and_on_click_register_locked_callbacks():
    widget = param.Parameterized()
    button = SimpleNamespace(on_click=lambda callback: callback)
    watcher = watch(widget, lambda event: None, 'name')
    assert getattr(watcher.fn, LOCKED_MARK)
    assert watcher.parameter_names == ('name',)
    assert getattr(on_click(button, lambda event: None), LOCKED_MARK)


@pytest.mark.parametrize('text', [
    '_pending_writes = None',
    '_pending_writes should be non-None when we have a document lock'])
def test_bokeh_marks_the_lock_as_locked_expects(text):
    # locked() reads ServerSession._pending_writes (bokeh 3.10.0)
    assert text in inspect.getsource(ServerSession)
