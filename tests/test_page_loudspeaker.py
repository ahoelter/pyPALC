"""Page 2 (loudspeaker) in Material widgets: labels, cards, the upload of
measured directivity data and its error text."""
from pathlib import Path

import numpy as np
import panel as pn
import panel_material_ui as pmui
from bokeh.document import Document

from pypalc.gui.app import PalcApp
from tests.gui_driver import read, upload, write
from tests.test_shell import rendered

MEASURED = Path(__file__).resolve().parent / 'cases' / 'measured.csv'
INPUTS = {'aperture_input': '9. Aperture angle in deg',
          'cabinet_height_input': '10. Loudspeaker height in m'}


def _cards(view):
    return [c.title for c in view.select(pmui.Card)]


def _alerts(view):
    return list(view.select(pmui.Alert))


def test_page_widgets_are_material():
    page = PalcApp(Document()).pages[2]
    for name, label in INPUTS.items():
        widget = getattr(page, name)
        assert isinstance(widget, pmui.TextInput), name
        assert read(widget, 'title') == label
    assert isinstance(page.directivity_select, pmui.Select)
    assert read(page.directivity_select, 'title') == '11. Directivity'
    assert read(page.directivity_select, 'value') == 'Circular Piston'
    assert read(page.directivity_select, 'options') == [
        'Modeled: Circular Piston', 'Modeled: Line Piston',
        'Modeled: Combined Circular/Line', 'Measured Loudspeaker Data']
    upload_input = page.directivity_upload
    assert isinstance(upload_input, pmui.FileInput)
    assert upload_input.accept == '.csv'
    assert not upload_input.visible


def test_layout_cards_and_figures():
    app = rendered(PalcApp(Document()))
    page = app.pages[2]
    write(app.page_select, 'active', 2)
    view = app.content.objects[0]
    assert view is page.layout()
    assert _cards(view) == ['Cabinet', 'Directivity', 'Figures']
    assert isinstance(page.tabs, pmui.Tabs)
    assert app.venue_axes_select in view.select(pmui.RadioButtonGroup)
    figures = [pane.object for pane in view.select(pn.pane.Bokeh)]
    assert figures == [app.venue_figure, page.directivity_figure]
    for fig in figures:
        assert fig.sizing_mode == 'stretch_width'


def test_measured_directivity_shows_the_upload():
    page = PalcApp(Document()).pages[2]
    write(page.directivity_select, 'value', 'Measured Loudspeaker Data')
    assert page.directivity_upload.visible
    write(page.directivity_select, 'value', 'Line Piston')
    assert not page.directivity_upload.visible


def test_upload_invalid_csv_shows_error():
    app = rendered(PalcApp(Document()))
    page = app.pages[2]
    write(app.page_select, 'active', 2)
    write(page.directivity_select, 'value', 'Measured Loudspeaker Data')
    upload(page.directivity_upload, b"not,a\ncsv", 'broken.csv')
    error = read(app.config_error_info, 'text')
    assert error and error.startswith('Error'), error
    alert, = _alerts(app.content.objects[0])
    assert alert.visible and alert.severity == 'error'
    assert read(alert, 'text') == error
    # a valid file afterwards is read and plotted
    text = MEASURED.read_bytes()
    upload(page.directivity_upload, text, 'measured.csv')
    assert app.state.measured_directivity_csv == text.decode('utf-8')
    assert read(app.config_error_info, 'text') == ''
    assert not alert.visible
    measured = app.state.sound_field_config.plot_measured_angles_deg
    np.testing.assert_array_equal(
        page.directivity_source.data['angle_deg'], measured)

