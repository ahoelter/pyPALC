"""Page 3 (array) in Material widgets: labels, cards, the reference array
widgets and table, and the error text."""
import panel as pn
import panel_material_ui as pmui
from bokeh.document import Document

from pypalc.gui.app import PalcApp
from tests.gui_driver import read, write
from tests.test_shell import rendered

INPUTS = {'cabinet_count_input': '12. Number of cabinets',
          'top_x_input': '13. Top x in m',
          'top_y_input': '14. Top y in m',
          'cabinet_gap_input': '15. Gap between cabinets in m',
          'discrete_angles_input': '17. Discrete tilt angles in deg',
          'reference_start_input': 'Array tilt angle in deg',
          'reference_step_stop_input': 'Inter cabinet angle in deg'}
SELECTS = {'use_discrete_angles_select': '16. Use discrete tilt angles?',
           'reference_type_select': 'Type',
           'reference_discrete_angles_select': 'Use angles of 17.?'}


def _cards(view):
    return [c.title for c in view.select(pmui.Card)]


def _shown(app):
    write(app.page_select, 'active', 3)
    return app.content.objects[0]


def test_page_widgets_are_material():
    page = PalcApp(Document()).pages[3]
    for name, label in INPUTS.items():
        widget = getattr(page, name)
        assert isinstance(widget, pmui.TextInput), name
        assert read(widget, 'title') == label, name
    for name, label in SELECTS.items():
        widget = getattr(page, name)
        assert isinstance(widget, pmui.Select), name
        assert read(widget, 'title') == label, name
    assert isinstance(page.reference_user_angles_input, pmui.TextAreaInput)
    assert read(page.reference_user_angles_input, 'title') == \
        'Tilt angles in deg'


def test_layout_cards_and_figure():
    app = rendered(PalcApp(Document()))
    page = app.pages[3]
    view = _shown(app)
    assert view is page.layout()
    assert _cards(view) == ['Cabinets', 'Discrete tilt angles',
                            'Reference array', 'Venue slice']
    assert app.venue_axes_select in view.select(pmui.RadioButtonGroup)
    figures = [pane.object for pane in view.select(pn.pane.Bokeh)]
    assert figures == [app.venue_figure]


def test_reference_labels_follow_the_type():
    page = PalcApp(Document()).pages[3]
    start, step_stop = page.reference_start_input, \
        page.reference_step_stop_input
    write(page.reference_type_select, 'value', 'Progressive')
    assert (start.label, step_stop.label) == ('Start angle in deg',
                                              'Final angle in deg')
    assert page.reference_discrete_angles_select.visible
    write(page.reference_type_select, 'value', 'Arc')
    assert (start.label, step_stop.label) == ('Start angle in deg',
                                              'Inter cabinet angle in deg')
    write(page.reference_type_select, 'value', 'User Defined')
    assert not start.visible and not step_stop.visible
    assert page.reference_user_angles_input.visible
    write(page.reference_type_select, 'value', 'Straight')
    assert (start.label, step_stop.label) == ('Array tilt angle in deg',
                                              'Inter cabinet angle in deg')
    assert step_stop.disabled


def test_reference_table_shows_the_reference_angles():
    app = rendered(PalcApp(Document()))
    page = app.pages[3]
    table, = _shown(app).select(pn.widgets.Tabulator)
    assert list(table.value['cabinet']) == [1, 2, 3, 4]
    assert list(table.value['tilt_angle_deg']) == [0, 0, 0, 0]
    write(page.reference_type_select, 'value', 'Arc')
    write(page.reference_start_input, 'value', '1')
    write(page.reference_step_stop_input, 'value', '2')
    assert list(table.value['tilt_angle_deg']) == [1, 3, 5, 7]
    write(page.cabinet_count_input, 'value', '2')
    assert list(table.value['cabinet']) == [1, 2]


def test_error_text_shows_as_alert():
    app = rendered(PalcApp(Document()))
    page = app.pages[3]
    alert, = _shown(app).select(pmui.Alert)
    assert not alert.visible
    write(page.cabinet_count_input, 'value', '-3')
    assert read(alert, 'text') == \
        'Error: only numbers are allowed. Set to default'
    assert alert.visible and alert.severity == 'error'
    write(page.top_y_input, 'value', '-50')
    assert read(alert, 'text').startswith('Warning')
    assert alert.visible and alert.severity == 'warning'
    write(page.top_y_input, 'value', '10')
    assert read(alert, 'text') == ''
    assert not alert.visible


def test_discrete_angles_enable_their_input():
    page = PalcApp(Document()).pages[3]
    write(page.use_discrete_angles_select, 'value', 'Yes')
    assert not page.discrete_angles_input.disabled
    assert not page.reference_discrete_angles_select.disabled
    assert read(page.discrete_angles_input, 'value') == '0,0.5,1,2,3.5,5,7,10'
    write(page.use_discrete_angles_select, 'value', 'No')
    assert page.discrete_angles_input.disabled
    assert read(page.discrete_angles_input, 'value') == ''
