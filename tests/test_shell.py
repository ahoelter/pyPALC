"""The Panel frame of the GUI: the Material page, its header with the page
tabs and logos, page switching and :meth:`PalcApp.later`.

:func:`rendered` builds the view of an app into its document, as
``panel serve`` does with ``main.py``; tests that need the models of the
shown page in the document use it.
"""
import html
import re

import panel as pn
import panel_material_ui as pmui
import param
from bokeh.document import Document
from bokeh.models import Tabs as BokehTabs
from bokeh.models import widgets as bokeh_widgets
from bokeh.plotting import figure as BokehFigure
from panel.viewable import Viewable

from pypalc.gui.app import PalcApp
from pypalc.gui.downloads import DOWNLOAD_FILE_NAMES, RESULTS_FILE_NAME
from pypalc.gui.style import MATERIAL_THEME, SURFACE, bokeh_figure_theme
from tests.gui_driver import read, write
from tests.test_gui_reference import RENAMED_LABELS

#: A model that only the page of the index shows.
MARKERS = {0: 'tabs', 1: 'save_line_button', 2: 'directivity_select',
           3: 'cabinet_count_input', 4: 'palc_variant_select', 5: 'tabs'}


def rendered(app: PalcApp) -> PalcApp:
    """The app with its view built into its document (as served)."""
    app.view().server_doc(app.doc)
    return app


def _shows(view, model):
    """Whether a Panel view holds the model: a Panel object in it or a
    Bokeh model in one of its Bokeh panes."""
    if isinstance(model, Viewable):
        return any(obj is model for obj in view.select())
    return any(model in pane.object.references()
               for pane in view.select(pn.pane.Bokeh))


def _in_document(model, doc):
    """Whether the model (Bokeh model or Panel object) is shown in the
    document."""
    if isinstance(model, Viewable):
        return any(bokeh_model.document is doc
                   for bokeh_model, _ in model._models.values())
    return model.document is doc


def _shown_markers(app):
    """Indices of the pages whose marker model the content area shows."""
    objects = app.content.objects
    assert len(objects) == 1
    return [index for index, name in MARKERS.items()
            if _shows(objects[0], getattr(app.pages[index], name))]


def test_page_select_switches_content():
    app = rendered(PalcApp(Document()))
    assert read(app.page_select, 'active') == 1
    assert _shown_markers(app) == [1]
    for index in [0, 2, 3, 4, 5, 1, 5, 0]:
        write(app.page_select, 'active', index)
        assert _shown_markers(app) == [index]
        # only the shown page is in the document
        in_doc = [i for i, name in MARKERS.items()
                  if _in_document(getattr(app.pages[i], name), app.doc)]
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


def _attributes(app):
    """(owner, name, value) of the attributes of the app and its pages."""
    owners = [('app', app)] + [(str(index), page)
                               for index, page in app.pages.items()]
    return [(owner, name, value) for owner, obj in owners
            for name, value in vars(obj).items()]


def test_no_bokeh_widgets_left():
    """Every page is Panel: no Bokeh widget (incl. Div and Paragraph) or
    Bokeh tabs among the attributes, and no page layout is a Bokeh model
    (figures are shown in Bokeh panes)."""
    app = PalcApp(Document())
    for owner, name, value in _attributes(app):
        assert not isinstance(value, (bokeh_widgets.Widget, BokehTabs)), (
            owner, name, type(value).__name__)
    for index, page in app.pages.items():
        view = page.layout()
        assert isinstance(view, Viewable), index
        assert not isinstance(view, pn.pane.Bokeh), index


def _help_text():
    """The text of the help page, without HTML tags."""
    return html.unescape(re.sub(r'<[^>]+>', '', '\n'.join(
        pane.object for pane in PalcApp(Document()).pages[0].layout()
        .select(pn.pane.HTML))))


def test_help_mentions_new_labels():
    """The help text uses the new labels (all of RENAMED_LABELS, which
    includes every renamed label that names a step) and none of the old
    ones."""
    text = _help_text()
    for old, new in RENAMED_LABELS.items():
        assert new in text, new
        assert old not in text, old


def test_help_quotes_the_step_labels():
    """Every label of a numbered step (inputs, buttons, figure titles of
    pages 1 to 5) is quoted in the help text as the GUI shows it."""
    app = PalcApp(Document())
    text = _help_text()
    labels = set()
    for _, _, value in _attributes(app):
        if isinstance(value, param.Parameterized) and \
                isinstance(getattr(value, 'label', None), str):
            labels.add(value.label)
        elif isinstance(value, BokehFigure) and value.title is not None:
            labels.add(value.title.text)
    # the ν label shows the value of ν
    steps = {label.split(' = ')[0] for label in labels
             if re.match(r'\d+\. ', label)}
    assert len(steps) > 30
    missing = sorted(label for label in steps if label not in text)
    assert not missing, missing


def test_help_names_every_download_file():
    """The help text names every file of "Download PALC Angles" and
    "Download Results" (all of DOWNLOAD_FILE_NAMES and the zip) and that
    the zip holds five files."""
    text = _help_text()
    for name in [*DOWNLOAD_FILE_NAMES.values(), RESULTS_FILE_NAME]:
        assert name in text, name
    assert len(DOWNLOAD_FILE_NAMES) - 1 == 5
    assert 'five CSV files' in text


def test_help_tabs_name_the_pages():
    page = PalcApp(Document()).pages[0]
    view = page.layout()
    assert isinstance(page.tabs, pmui.Tabs)
    assert page.tabs in view.select(pmui.Tabs)
    assert list(page.tabs._names) == [
        'General information', '1 · Venue', '2 · Loudspeaker', '3 · Array',
        '4 · Algorithm', '5 · Sound Field']
