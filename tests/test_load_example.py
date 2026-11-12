""""Load Example" on page 1: the example setups of ``pypalc.examples``."""
import contextlib
import io
import shutil
from pathlib import Path

import panel_material_ui as pmui
import pytest
from bokeh.document import Document

import pypalc.examples
from pypalc.core.errors import SetupError
from pypalc.examples import examples
from pypalc.gui.app import PalcApp
from pypalc.gui.downloads import angles_csv
from tests.gui_driver import click, read, upload
from tests.test_shell import rendered

CASES = Path(__file__).parent / 'cases'


def _upload(input_, path):
    with contextlib.redirect_stdout(io.StringIO()):
        upload(input_, path.read_bytes(), path.name)


def _click(app):
    with contextlib.redirect_stdout(io.StringIO()):
        click(app.pages[1].load_example_button)


def test_card_has_example_select_and_button():
    app = rendered(PalcApp(Document()))
    venue = app.pages[1]
    card = [c for c in venue.layout().select(pmui.Card)
            if c.title == 'Setup file'][0]
    assert venue.example_select in card.objects
    assert venue.load_example_button in card.objects
    assert venue.example_select.label == 'Example'
    assert venue.example_select.options == {e.title: e.name
                                            for e in examples()}
    assert venue.example_select.value == examples()[0].name
    assert venue.load_example_button.label == 'Load Example'
    assert not venue.example_select.disabled
    assert not venue.load_example_button.disabled


@pytest.mark.parametrize('example', examples(), ids=lambda e: e.name)
def test_every_example_loads_on_page_1(example):
    via4 = PalcApp(Document())
    _upload(via4.pages[4].load_setup_input, example.path)
    app = PalcApp(Document())
    app.pages[1].example_select.value = example.name
    _click(app)
    assert read(app.page_select, 'active') == 4
    assert read(app.config_error_info, 'text') == ''
    assert angles_csv(app.pages[4]) == angles_csv(via4.pages[4])
    assert angles_csv(app.pages[4])
    assert app.pages[4].running is False


def test_example_with_data_file_loads(tmp_path, monkeypatch):
    for name in ('measured.yaml', 'measured.csv'):
        shutil.copy(CASES / name, tmp_path / name)
    monkeypatch.setattr(pypalc.examples, 'EXAMPLES_DIR', tmp_path)
    app = PalcApp(Document())
    venue = app.pages[1]
    assert venue.example_select.options == {'Measured': 'measured'}
    venue.example_select.value = 'measured'
    _click(app)
    assert read(app.config_error_info, 'text') == ''
    assert read(app.page_select, 'active') == 4
    assert app.pages[2].directivity_select.value == 'Measured Loudspeaker Data'


def test_example_ignored_while_running():
    app = PalcApp(Document())
    app.pages[4].running = True
    _click(app)
    assert read(app.page_select, 'active') == 1
    app.pages[4].running = False


def test_example_error_shown_on_page_1(monkeypatch):
    def fail(name):
        raise SetupError('x', 'bad')
    monkeypatch.setattr('pypalc.gui.page_algorithm.load_example', fail)
    app = PalcApp(Document())
    _click(app)
    assert app.config_error_info.object == \
        'Error: setup file invalid at x: bad'
    assert read(app.page_select, 'active') == 1
    assert app.pages[4].running is False


def test_no_example_disables_the_widgets(tmp_path, monkeypatch):
    monkeypatch.setattr(pypalc.examples, 'EXAMPLES_DIR', tmp_path)
    venue = PalcApp(Document()).pages[1]
    assert venue.example_select.disabled
    assert venue.load_example_button.disabled
