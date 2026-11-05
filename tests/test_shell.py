"""The Panel frame of the GUI: the Material page, its header with the page
tabs and logos, page switching and :meth:`PalcApp.later`.

:func:`rendered` builds the view of an app into its document, as
``panel serve`` does with ``main.py``; tests that need the models of the
shown page in the document use it.
"""
import panel_material_ui as pmui
from bokeh.document import Document

from pypalc.gui.app import PalcApp
from pypalc.gui.style import MATERIAL_THEME, SURFACE, bokeh_figure_theme
from tests.gui_driver import read, write

#: A model that only the page of the index shows.
MARKERS = {0: 'tabs', 1: 'save_line_button', 2: 'directivity_select',
           3: 'cabinet_count_input', 4: 'palc_variant_select', 5: 'tabs'}


def rendered(app: PalcApp) -> PalcApp:
    """The app with its view built into its document (as served)."""
    app.view().server_doc(app.doc)
    return app


def _shown_markers(app):
    """Indices of the pages whose marker model the content area shows."""
    objects = app.content.objects
    assert len(objects) == 1
    shown = objects[0].object.references()
    return [index for index, name in MARKERS.items()
            if getattr(app.pages[index], name) in shown]


def test_page_select_switches_content():
    app = rendered(PalcApp(Document()))
    assert read(app.page_select, 'active') == 1
    assert _shown_markers(app) == [1]
    for index in [0, 2, 3, 4, 5, 1, 5, 0]:
        write(app.page_select, 'active', index)
        assert _shown_markers(app) == [index]
        # only the shown page is in the document
        in_doc = [i for i, name in MARKERS.items()
                  if getattr(app.pages[i], name).document is app.doc]
        assert in_doc == [index]


def test_pages_keep_figure_theme():
    doc = Document()
    doc.theme = bokeh_figure_theme()  # as main.py
    app = rendered(PalcApp(doc))
    for index in [1, 2, 3, 4, 1]:
        write(app.page_select, 'active', index)
        assert app.venue_figure.background_fill_color == SURFACE, index


def test_page_tabs():
    app = PalcApp(Document())
    assert read(app.page_select, 'options') == [
        'Help', '1 · Venue', '2 · Loudspeaker', '3 · Array', '4 · Algorithm',
        '5 · Sound Field']


def test_view_is_material_page():
    app = PalcApp(Document())
    page = app.view()
    assert isinstance(page, pmui.Page)
    assert page.title == 'pyPALC'
    assert page.dark_theme and not page.theme_toggle
    assert page.theme_config == MATERIAL_THEME
    palette = MATERIAL_THEME['palette']
    assert palette['mode'] == 'dark'
    assert palette['background']['default'] == '#323030'
    assert palette['text']['primary'] == '#ffffff'
    assert palette['success']['main'] == '#5cb85c'
    assert palette['error']['main'] == '#d9534f'
    assert palette['warning']['main'] == '#f0ad4e'
    assert palette['primary']['main'] == '#428bca'
    link = MATERIAL_THEME['components']['MuiLink']['styleOverrides']['root']
    assert link['color'] == '#5cb85c'
    assert app.page_select in page.header
    assert list(page.main) == [app.content]
    logos = ''.join(str(getattr(item, 'object', '')) for item in page.header)
    assert logos.count('src="data:image/png;base64,') == 2
    assert 'href="https://www.ak.tu-berlin.de"' in logos


def test_later_runs_now_when_not_served():
    app = PalcApp(Document())
    calls = []
    app.later(lambda: calls.append(1))
    assert calls == [1]


def test_later_waits_when_served(monkeypatch):
    app = PalcApp(Document())
    monkeypatch.setattr(Document, 'session_context',
                        property(lambda doc: object()))
    calls = []
    app.later(lambda: calls.append(1))
    assert calls == []
    (callback,) = app.doc.session_callbacks
    assert callback.timeout == 120
    callback.callback()
    assert calls == [1]
