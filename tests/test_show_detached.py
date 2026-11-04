"""``PalcApp.show`` changes no model while the document is cleared.

Between ``doc.clear()`` and ``doc.add_root()`` the models of the previous
page are outside the document. A change to such a model sends no event to
the browser, and when the page switch came from the browser, Bokeh 3 sends
the models the browser had before the clear only by reference when they
are added again; the browser keeps the old value (the rays on page 1, fixed
in ea30381). This test records every property change of a model that was
in the document before ``doc.clear()`` until the next ``doc.add_root()``.

Changes are observed through ``bokeh.model.Model.trigger``, which every
property change (assignment and in-place change of a container, e.g.
``data.update``) calls, also for models outside a document; document
callbacks would not see changes of detached models.
"""
import contextlib
import io

import pytest
from bokeh.document import Document
from bokeh.model import Model

from pypalc.core.settings import load_setup
from pypalc.gui.app import PalcApp
from tests.baseline_cases import case_path
from tests.test_apply_setup import case_text, load
from tests.test_gui_equivalence import drive

#: Order of the page switches (0: Help). Page 1 clears the rays of a
#: result, which the browser has while page 4 is shown (the case of
#: ea30381), so the order starts with page 4 and then page 1.
PAGE_ORDER = [4, 1, 2, 3, 4, 5, 0, 4, 1, 5, 3, 2, 1]


class DetachedChanges:
    """Records the changes of models that were in the document before
    ``doc.clear()``, made before the next ``doc.add_root()``.

    Parameters
    ----------
    doc : Document
        The document of the app.
    monkeypatch : pytest.MonkeyPatch
        Patches ``doc.clear``, ``doc.add_root`` and ``Model.trigger``.
    """

    def __init__(self, doc, monkeypatch):
        self.detached = None  # ids of the cleared models, while cleared
        self.windows = 0
        self.changes_in_windows = 0
        self.changes = []
        clear, add_root = doc.clear, doc.add_root
        trigger = Model.trigger

        def recording_clear():
            ids = {m.id for m in doc.models}
            clear()
            self.detached = ids
            self.windows += 1

        def recording_add_root(*args, **kwargs):
            self.detached = None
            return add_root(*args, **kwargs)

        def recording_trigger(model, attr, *args, **kwargs):
            if self.detached is not None:
                self.changes_in_windows += 1
                if model.id in self.detached:
                    self.changes.append('%s(%s).%s' % (
                        type(model).__name__, model.id, attr))
            return trigger(model, attr, *args, **kwargs)

        monkeypatch.setattr(doc, 'clear', recording_clear)
        monkeypatch.setattr(doc, 'add_root', recording_add_root)
        monkeypatch.setattr(Model, 'trigger', recording_trigger)

    def switch_pages(self, app, pages=PAGE_ORDER):
        """Show the pages one after the other through the page selection.

        Parameters
        ----------
        app : PalcApp
            The app.
        pages : list of int, optional
            Indices of the pages.

        Returns
        -------
        None.
        """
        for page in pages:
            windows, switched = self.windows, page != app.page_select.active
            with contextlib.redirect_stdout(io.StringIO()):
                app.page_select.active = page
            assert self.windows == windows + switched, page
            assert self.detached is None, 'page %d: no add_root' % page


def _new_app(monkeypatch):
    app = PalcApp(Document())
    recorder = DetachedChanges(app.doc, monkeypatch)
    app.show()
    return app, recorder


def _assert_no_changes(recorder):
    # the hook must have seen the changes of the models that show() creates
    # between doc.clear() and doc.add_root(), or it observes nothing
    assert recorder.windows > len(PAGE_ORDER) // 2
    assert recorder.changes_in_windows > 0
    assert recorder.changes == [], recorder.changes


def test_page_switches_of_a_new_session(monkeypatch):
    app, recorder = _new_app(monkeypatch)
    recorder.switch_pages(app)
    _assert_no_changes(recorder)


@pytest.mark.parametrize("case", ["simple", "target_slope"])
def test_page_switches_after_a_run(monkeypatch, case):
    app, recorder = _new_app(monkeypatch)
    with contextlib.redirect_stdout(io.StringIO()):
        drive(app, load_setup(case_path(case)))
    recorder.switch_pages(app)
    _assert_no_changes(recorder)


def test_page_switches_after_loading_a_target_slope_setup(monkeypatch):
    app, recorder = _new_app(monkeypatch)
    load(app, case_text('gap_soft_margin_target_slope'))
    assert app.page_select.active == 5
    recorder.switch_pages(app)
    _assert_no_changes(recorder)
