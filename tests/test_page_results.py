"""Page 5 (sound field) in Material widgets: the frequency of the beam
plots, the beam plot selection, the tabs of the figures, the responsive
figures, the beam plots with the extent of their image as ranges and
equal scaling, and the venue point tap."""
import contextlib
import io

import numpy as np
import panel as pn
import panel_material_ui as pmui
import pytest
from bokeh.document import Document
from bokeh.models import BoxZoomTool, CustomJS, Range1d

from pypalc.core.settings import load_setup
from pypalc.gui.app import PalcApp
from pypalc.gui.ranges import VARIABLE_HEIGHT, VENUE_MAX_HEIGHT, ScaledFigure
from tests.baseline_cases import case_path
from tests.gui_driver import read, write
from tests.test_gui_equivalence import drive
from tests.test_shell import rendered

TABS = ['Frequency responses', 'Homogeneity', 'Bar charts', 'Beam plots']

# the beam plots divide by zero at the array's points (as in the reference
# steps of tests/test_gui_reference.py); these warnings are known
pytestmark = pytest.mark.filterwarnings(
    'ignore::RuntimeWarning:pypalc.core.sound_field')


@pytest.fixture(scope='module')
def ran():
    """A rendered app with case ``simple`` entered and run."""
    app = rendered(PalcApp(Document()))
    with contextlib.redirect_stdout(io.StringIO()):
        drive(app, load_setup(case_path('simple')))
    return app


def _shown(app, index=5):
    write(app.page_select, 'active', index)
    return app.content.objects[0]


def _beam_figures(page):
    return [page.beam_figure, page.reference_beam_figure]


def test_page_widgets_are_material():
    page = PalcApp(Document()).pages[5]
    select = page.beam_frequency_select
    assert isinstance(select, pmui.Select)
    assert read(select, 'title') == 'Frequency in Hz'
    assert read(select, 'options') == ['63', '125', '250', '500', '1000',
                                       '2000', '4000', '8000', '16000']
    assert select.value == '63'
    checks = page.beam_checkbox
    assert isinstance(checks, pmui.CheckButtonGroup)
    assert read(checks, 'options') == ['PALC', 'Reference']
    assert read(checks, 'active') == []
    assert checks.disabled
    assert isinstance(page.tabs, pmui.Tabs)


def test_layout_card_tabs_and_figures():
    app = rendered(PalcApp(Document()))
    page = app.pages[5]
    view = _shown(app)
    assert view is page.layout()
    (found,) = view.select(pmui.Card)
    assert page.tabs in found.objects
    assert list(page.tabs._names) == TABS
    figures = [pane.object for pane in view.select(pn.pane.Bokeh)]
    assert figures == [page.venue_points_figure, page.sound_field_figure,
                       page.reference_sound_field_figure,
                       page.homogeneity_figure, page.histogram_figure,
                       page.reference_histogram_figure, page.beam_figure,
                       page.reference_beam_figure]
    for figure in figures:
        assert figure.sizing_mode == 'stretch_width', figure.title.text
        assert figure.frame_width is None, figure.title.text
    widgets = list(view.select(pmui.Select)) + \
        list(view.select(pmui.CheckButtonGroup))
    assert page.beam_frequency_select in widgets
    assert page.beam_checkbox in widgets


def _image_extent(page):
    data = page.beam_source.data
    x, y = data['x'][0], data['y'][0]
    return [x, x + data['dw'][0]], [y, y + data['dh'][0]]


def test_beam_ranges_are_the_extent_of_the_image(ran):
    """The ranges of both beam plots are where the beam plot calculation
    starts and ends (the image), without a margin; the figures are sized
    for equal scaling as "Equal, variable height"."""
    page = ran.pages[5]
    ranges = ran.state.ranges
    write(page.beam_checkbox, 'active', [0, 1])
    x, y = _image_extent(page)
    # the image covers the venue ranges
    assert x == pytest.approx(ranges.venue_x) and \
        y == pytest.approx(ranges.venue_y)
    for figure in _beam_figures(page):
        for axis_range, extent in ((figure.x_range, x), (figure.y_range, y)):
            assert type(axis_range) is Range1d
            assert [axis_range.start, axis_range.end] == extent
        assert figure.match_aspect is False
        assert figure.sizing_mode == 'stretch_width'
        assert figure.height == VENUE_MAX_HEIGHT
        assert figure.styles == {}
    # each figure has its own ranges (linked in the browser)
    beam, reference = _beam_figures(page)
    assert beam.x_range is not reference.x_range
    panes = list(_shown(ran).select(ScaledFigure))
    assert [pane.object.object for pane in panes] == _beam_figures(page)
    for pane in panes:
        assert pane.scaling == VARIABLE_HEIGHT
        # at most the height of the venue figure (and the window limit)
        assert pane.largest_height == VENUE_MAX_HEIGHT
        assert pane.data_aspect == pytest.approx(
            (x[1] - x[0]) / (y[1] - y[0]))
    # the venue points are stretched over the venue ranges
    points = page.venue_points_figure
    assert isinstance(points.x_range, Range1d)
    assert [points.x_range.start, points.x_range.end] == ranges.venue_x
    assert [points.y_range.start, points.y_range.end] == ranges.venue_y
    assert not points.match_aspect
    write(page.beam_checkbox, 'active', [])


def test_beam_ranges_follow_a_new_image_extent(ran):
    """New ranges only for a new extent: a page switch or a new frequency
    keeps them (and a zoom)."""
    page = ran.pages[5]
    write(page.beam_checkbox, 'active', [0])
    before = page.beam_figure.x_range
    _shown(ran, 1)
    _shown(ran)
    write(page.beam_frequency_select, 'value', '125')
    assert page.beam_figure.x_range is before
    data = dict(page.beam_source.data)
    data.update(x=[-5.0], dw=[30.0])
    page.beam_source.data = data
    page.scale_beam_figures()
    assert [page.beam_figure.x_range.start,
            page.beam_figure.x_range.end] == [-5.0, 25.0]
    assert [page.reference_beam_figure.x_range.start,
            page.reference_beam_figure.x_range.end] == [-5.0, 25.0]
    write(page.beam_frequency_select, 'value', '63')
    write(page.beam_checkbox, 'active', [])
    assert [page.beam_figure.x_range.start,
            page.beam_figure.x_range.end] == _image_extent(page)[0]


def test_beam_plots_follow_the_selection(ran):
    page = ran.pages[5]
    assert not page.beam_checkbox.disabled
    write(page.beam_checkbox, 'active', [0])
    assert page.beam_figure.visible
    assert not page.reference_beam_figure.visible
    spl = np.array(page.beam_source.data['spl_db'][0])
    assert spl.ndim == 2 and spl.size > 1
    assert 50 <= spl.min() <= spl.max() <= 100
    write(page.beam_frequency_select, 'value', '2000')
    assert not np.array_equal(page.beam_source.data['spl_db'][0], spl)
    write(page.beam_checkbox, 'active', [1])
    assert not page.beam_figure.visible
    assert page.reference_beam_figure.visible
    write(page.beam_checkbox, 'active', [])


def test_tapped_venue_points_show_their_responses(ran):
    page = ran.pages[5]
    evaluation = ran.state.result.optimized.evaluation
    write(page.venue_points_source, 'selected', [2, 5])
    data = page.sound_field_source.data
    assert len(data['spl_db']) == 2
    assert np.array_equal(data['spl_db'][1], evaluation.plot_spl_db[5])
    write(page.venue_points_source, 'selected', [])
    assert len(page.sound_field_source.data['spl_db']) == \
        len(evaluation.plot_spl_db)


def test_a_new_result_resets_the_beam_selection(ran):
    page = ran.pages[5]
    write(page.beam_checkbox, 'active', [0, 1])
    ran.pages[4].start_toggle.value = True
    assert read(page.beam_checkbox, 'active') == []
    assert not page.beam_figure.visible
    assert not page.reference_beam_figure.visible


def test_beam_plots_zoom_together():
    """A zoom, pan or reset in one beam plot is copied to the other in the
    browser (``rangesupdate``, which the tools and the reset fire, and
    ``pan`` while panning); the code reads the ranges the figures have
    when it runs, so renewed ranges stay linked."""
    page = PalcApp(Document()).pages[5]
    beam, reference = _beam_figures(page)
    for source, target in ((beam, reference), (reference, beam)):
        (callback,) = source.js_event_callbacks['rangesupdate']
        assert source.js_event_callbacks['pan'] == [callback]
        assert isinstance(callback, CustomJS)
        assert callback.args == {'source': source, 'target': target}
        assert 'x_range' in callback.code and 'y_range' in callback.code
        # a box zoom keeps the equal scaling
        (box_zoom,) = [tool for tool in source.tools
                       if isinstance(tool, BoxZoomTool)]
        assert box_zoom.match_aspect
    # the venue points figure is not linked (it is stretched)
    assert 'rangesupdate' not in page.venue_points_figure.js_event_callbacks
