""""Load Setup" on page 1: the same loading as the input of page 4."""
import contextlib
import io

import panel_material_ui as pmui
from bokeh.document import Document

from pypalc.core import pipeline
from pypalc.core.settings import dump_setup, load_setup
from pypalc.gui.app import PalcApp
from pypalc.gui.downloads import angles_csv
from tests.baseline_cases import case_path
from tests.gui_driver import read, upload
from tests.test_shell import rendered


def _text(case='simple'):
    return dump_setup(load_setup(case_path(case))).encode('utf-8')


def _load(input_, data):
    with contextlib.redirect_stdout(io.StringIO()):
        upload(input_, data, 'pypalc_setup.yaml')


def _angles(app):
    return angles_csv(app.pages[4])


def test_input_is_like_the_one_of_page_4():
    app = PalcApp(Document())
    one, four = app.pages[1].load_setup_input, app.pages[4].load_setup_input
    assert one is not four
    assert isinstance(one, pmui.FileInput)
    for name in ('label', 'accept', 'color', 'variant', 'icon'):
        assert getattr(one, name) == getattr(four, name)


def test_page_1_shows_input_and_error_pane():
    app = rendered(PalcApp(Document()))
    view = app.pages[1].layout()
    card = [c for c in view.select(pmui.Card) if c.title == 'Setup file'][0]
    assert app.pages[1].load_setup_input in card.objects
    assert app.config_error_info in card.objects


def test_load_through_page_1_equals_page_4():
    via4 = PalcApp(Document())
    _load(via4.pages[4].load_setup_input, _text())
    via1 = PalcApp(Document())
    _load(via1.pages[1].load_setup_input, _text())
    assert read(via1.page_select, 'active') == 4
    assert read(via1.config_error_info, 'text') == ''
    assert _angles(via1) == _angles(via4) and _angles(via1)
    assert via1.pages[4].running is False
    # emptied, so that the same file loads again
    assert via1.pages[1].load_setup_input.value is None
    assert via1.pages[4].load_setup_input.value is None


def test_invalid_file_through_page_1_shows_error():
    app = PalcApp(Document())
    _load(app.pages[1].load_setup_input, b'venue: [')
    assert read(app.config_error_info, 'text').startswith('Error: ')
    assert read(app.page_select, 'active') == 1
    assert app.pages[1].load_setup_input.value is None


def test_load_through_page_1_ignored_while_running(monkeypatch):
    app = PalcApp(Document())
    app.pages[4].running = True
    _load(app.pages[1].load_setup_input, _text())
    assert read(app.page_select, 'active') == 1
    assert app.pages[1].load_setup_input.value is None
    app.pages[4].running = False


def test_load_again_through_page_1(monkeypatch):
    app = PalcApp(Document())
    calls = []
    run = pipeline.run
    monkeypatch.setattr(pipeline, 'run',
                        lambda *a, **k: calls.append(1) or run(*a, **k))
    _load(app.pages[1].load_setup_input, _text())
    first = len(calls)
    _load(app.pages[1].load_setup_input, _text())
    assert first > 0 and len(calls) == 2 * first
