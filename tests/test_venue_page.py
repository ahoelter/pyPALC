"""Page 1 (venue) in Material widgets and the shared building blocks of
:mod:`pypalc.gui.widgets`."""
import panel as pn
import panel_material_ui as pmui
from bokeh.document import Document

from pypalc.gui.app import PalcApp
from pypalc.gui.style import bokeh_figure_theme
from pypalc.gui.widgets import card, message
from tests.gui_driver import click, read, write
from tests.test_shell import rendered

INPUTS = {'start_x_input': '1. Start x in m', 'end_x_input': '2. End x in m',
          'points_per_meter_input': '3. Discretization in points per m',
          'start_y_input': '4. Start y in m', 'end_y_input': '5. End y in m'}
BUTTONS = ('save_line_button', 'remove_line_button', 'insert_line_button',
           'remove_selected_line_button', 'check_venue_button')


def _cards(view):
    return [c.title for c in view.select(pmui.Card)]


def test_page_widgets_are_material():
    venue = PalcApp(Document()).pages[1]
    for name, label in INPUTS.items():
        widget = getattr(venue, name)
        assert isinstance(widget, pmui.TextInput), name
        assert read(widget, 'title') == label
    for name in BUTTONS:
        assert isinstance(getattr(venue, name), pmui.Button), name
    assert isinstance(venue.venue_error_info, pmui.Alert)


def test_layout_has_the_cards_of_the_prototype():
    app = rendered(PalcApp(Document()))
    view = app.pages[1].layout()
    assert _cards(view) == ['Audience line', 'Lines and venue',
                            'Venue slice']
    assert app.venue_axes_select in view.select(pmui.RadioButtonGroup)
    figures = [pane.object for pane in view.select(pn.pane.Bokeh)]
    assert figures == [app.venue_figure]


def test_buttons_run_their_callbacks():
    app = rendered(PalcApp(Document()))
    venue = app.pages[1]
    for widget, value in ((venue.start_x_input, '2'),
                          (venue.end_x_input, '10'),
                          (venue.end_y_input, '1')):
        write(widget, 'value', value)
    click(venue.save_line_button)
    assert len(app.state.audience_lines) == 1
    click(venue.check_venue_button)
    assert read(venue.check_venue_button, 'label') == \
        '8. Venue is PALC compatible!'
    click(venue.remove_line_button)
    assert app.state.audience_lines == []


def test_line_changes_keep_the_shown_page():
    """Ported widgets update in place: saving, inserting and removing
    lines and the venue check do not build the page again."""
    app = rendered(PalcApp(Document()))
    venue = app.pages[1]
    shown = list(app.content.objects)
    for widget, value in ((venue.start_x_input, '2'),
                          (venue.end_x_input, '10'),
                          (venue.end_y_input, '1')):
        write(widget, 'value', value)
    venue.save_line()
    write(venue.lines_source, 'selected', [0])
    for widget, value in ((venue.start_x_input, '0'),
                          (venue.end_x_input, '1'),
                          (venue.start_y_input, '-1'),
                          (venue.end_y_input, '-1')):
        write(widget, 'value', value)
    venue.insert_line()
    write(venue.lines_source, 'selected', [0])
    venue.remove_selected_line()
    app.check_venue()
    write(app.venue_axes_select, 'active', 0)
    assert app.content.objects == shown


def test_error_text_shows_as_alert():
    venue = PalcApp(Document()).pages[1]
    alert = venue.venue_error_info
    assert not alert.visible
    write(venue.start_x_input, 'value', 'abc')
    assert read(alert, 'text') == 'Error: Wrong Input. Only Numbers are allowed'
    assert alert.visible and alert.severity == 'error'
    write(venue.start_x_input, 'value', '1')
    assert read(alert, 'text') == ''
    assert not alert.visible


def test_message_severity():
    alert = message()
    for text, severity in (('Error: x', 'error'), ('Couldnt read', 'error'),
                           ('Warning: y', 'warning'), ('Note', 'info')):
        alert.object = text
        assert alert.visible and alert.severity == severity, text
    alert.object = ''
    assert not alert.visible


def test_card_look():
    box = card('Title', pn.pane.HTML('x'))
    assert isinstance(box, pmui.Card)
    assert box.title == 'Title' and not box.collapsible
    assert box.outlined and box.sizing_mode == 'stretch_width'


def test_toolbars_show_every_tool():
    """The theme's toolbar style lets a vertical toolbar grow beside a
    short frame instead of moving tools into an overflow menu."""
    theme = bokeh_figure_theme()
    sheets = theme._json['attrs']['Toolbar']['stylesheets']
    assert any('height: auto !important' in sheet for sheet in sheets)
