"""Page 4 (algorithm) in Material widgets: labels, cards, the order of the
run and file buttons, the tilt angle table and the shared error text."""
import contextlib
import io

import panel as pn
import panel_material_ui as pmui
from bokeh.document import Document

from pypalc.core.settings import load_setup
from pypalc.gui.app import PalcApp
from tests.baseline_cases import case_path
from tests.gui_driver import read, write
from tests.test_gui_equivalence import drive
from tests.test_shell import rendered

SELECTS = {'palc_variant_select': '18. PALC algorithm',
           'gap_handling_select': '19. Gap handling',
           'weighting_select': '21. Weighting',
           'hinges_select': 'Hinges',
           'nu_step_select': 'Step',
           'use_fixed_angle_select': '24. Fix angle of uppermost cabinet?'}
SLIDERS = {'soft_margin_slider': '20. Strength of soft margin',
           'tolerance_slider': '23. Tolerance (abort criterion) in m',
           'frequency_range_slider': 'Frequency range in Hz'}


def _shown(app, index=4):
    write(app.page_select, 'active', index)
    return app.content.objects[0]


def test_page_widgets_are_material():
    page = PalcApp(Document()).pages[4]
    for name, label in SELECTS.items():
        widget = getattr(page, name)
        assert isinstance(widget, pmui.Select), name
        assert read(widget, 'title') == label, name
    for name, label in SLIDERS.items():
        widget = getattr(page, name)
        assert isinstance(widget, (pmui.FloatSlider, pmui.RangeSlider)), name
        assert read(widget, 'title') == label, name
    assert isinstance(page.fixed_angle_input, pmui.TextInput)
    assert read(page.fixed_angle_input, 'title') == \
        '25. Angle of uppermost cabinet in deg'
    assert isinstance(page.start_toggle, pmui.Toggle)
    assert read(page.clear_results_button, 'label') == \
        '27. Delete result visualization'
    assert isinstance(page.results_table, pn.widgets.Tabulator)
    assert isinstance(page.tabs, pmui.Tabs)


def test_nu_label_shows_nu():
    page = PalcApp(Document()).pages[4]
    assert read(page.weighting_nu_slider, 'title') == \
        '22. Weighting parameter ν = 1.00'
    write(page.weighting_select, 'value', 'Linear Spacing')
    write(page.weighting_nu_slider, 'value', 1.5)
    assert read(page.weighting_nu_slider, 'title') == \
        '22. Weighting parameter ν = 3.38'


def _card(view, title):
    (found,) = [c for c in view.select(pmui.Card) if c.title == title]
    return found


def test_layout_cards_and_order():
    app = rendered(PalcApp(Document()))
    page = app.pages[4]
    view = _shown(app)
    assert view is page.layout()
    assert [c.title for c in view.select(pmui.Card)] == [
        'PALC', 'Weighting', 'Tolerance and fixed angle', 'Run', 'Files',
        'Tilt angles', 'Results']
    run = [obj for obj in _card(view, 'Run').objects]
    assert run == [app.pages[1].check_venue_button, page.start_toggle,
                   page.clear_results_button, app.config_error_info]
    files = list(_card(view, 'Files').objects)
    assert files == [page.download_angles_button,
                     page.download_results_button, page.save_setup_button,
                     page.load_setup_input]
    assert list(_card(view, 'Tilt angles').objects) == [page.results_table]
    assert page.tabs in _card(view, 'Results').objects
    assert list(page.tabs._names) == ['Venue slice', 'SPL over distance']
    figures = [pane.object for pane in view.select(pn.pane.Bokeh)]
    assert figures == [app.venue_figure, page.spl_over_distance_figure]
    assert page.spl_over_distance_figure.sizing_mode == 'stretch_width'


def test_results_table_follows_the_results():
    app = PalcApp(Document())
    page = app.pages[4]
    assert len(page.results_table.value) == 0
    with contextlib.redirect_stdout(io.StringIO()):
        drive(app, load_setup(case_path('simple')))
    table = page.results_table.value
    data = page.results_source.data
    assert list(table['cabinet']) == [1, 2, 3, 4]
    assert list(table['tilt_angle_deg']) == list(data['tilt_angle_deg'])
    assert list(table['inter_cabinet_angle_deg']) == \
        list(data['inter_cabinet_angle_deg'])


def test_weighting_target_slope_shows_hinges_and_spl_tab():
    app = PalcApp(Document())
    page = app.pages[4]
    with contextlib.redirect_stdout(io.StringIO()):
        drive(app, load_setup(case_path('simple')))
        write(page.weighting_select, 'value', 'Target Slope')
    assert page.hinges_select.visible
    assert page.tabs.active == 1
    write(page.weighting_select, 'value', 'Without')
    assert not page.hinges_select.visible


def test_one_error_text_for_pages_2_to_4():
    """The error text is one alert, shown on pages 2, 3 and 4."""
    app = rendered(PalcApp(Document()))
    alert = app.config_error_info
    assert isinstance(alert, pmui.Alert)
    assert not hasattr(app, 'config_message')
    for index in (2, 3, 4):
        view = _shown(app, index)
        alerts = [a for a in view.select(pmui.Alert) if a is alert]
        assert len(alerts) == 1, index
    app.pages[3].cabinet_count_input.value = '-3'
    assert read(alert, 'text').startswith('Error')
    assert alert.visible and alert.severity == 'error'
    app.pages[3].cabinet_count_input.value = '5'
    assert read(alert, 'text') == '' and not alert.visible


def test_spl_over_distance_hover_only_on_lines_and_points():
    """The hover of "SPL over distance" shows values only on the lines and
    points: not on the shaded target region, which covers the whole plot
    (it would show a tooltip everywhere, with ??? for the missing fields).
    Every hovered glyph has a name and the venue coordinates."""
    from bokeh.models import HoverTool, Patch
    fig = PalcApp(Document()).pages[4].spl_over_distance_figure
    (hover,) = [t for t in fig.toolbar.tools if isinstance(t, HoverTool)]
    patches = [r for r in fig.renderers if isinstance(r.glyph, Patch)]
    assert patches and hover.renderers != 'auto'
    assert not set(patches) & set(hover.renderers)
    assert {r.name for r in hover.renderers} == {
        'PALC', 'Reference', 'Target Slope', 'Reference Point'}
    for renderer in hover.renderers:
        assert {'x_m', 'y_m'} <= set(renderer.data_source.data)
