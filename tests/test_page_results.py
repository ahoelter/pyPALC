"""Page 5 (sound field) in Material widgets: the frequency of the beam
plots, the beam plot selection, the tabs of the figures, the responsive
figures with equal scaling of the beam plots and the venue point tap."""
import contextlib
import io

import numpy as np
import panel as pn
import panel_material_ui as pmui
import pytest
from bokeh.document import Document
from bokeh.models import BoxZoomTool, CustomJS, DataRange1d, Range1d

from pypalc.core.settings import load_setup
from pypalc.gui.app import PalcApp
from pypalc.gui.ranges import VENUE_MAX_HEIGHT, venue_bounds
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


def test_beam_figures_are_equally_scaled(ran):
    page = ran.pages[5]
    ranges = ran.state.ranges
    width = ranges.venue_x[1] - ranges.venue_x[0]
    height = ranges.venue_y[1] - ranges.venue_y[0]
    for figure in _beam_figures(page):
        bounds = venue_bounds(figure)
        assert bounds is not None and not bounds.visible
        assert list(bounds.data_source.data['x']) == list(ranges.venue_x)
        assert list(bounds.data_source.data['y']) == list(ranges.venue_y)
        assert figure.match_aspect is True and figure.aspect_scale == 1
        for axis_range in (figure.x_range, figure.y_range):
            assert isinstance(axis_range, DataRange1d)
            assert axis_range.renderers == [bounds]
            assert axis_range.range_padding == 0
        # height follows the width up to the maximum (as "Equal, variable
        # height" of the venue figure)
        assert figure.aspect_ratio == pytest.approx(width / height)
        assert figure.max_height == VENUE_MAX_HEIGHT
        assert figure.height is None
    # each figure has its own ranges (BokehJS computes them per figure)
    beam, reference = _beam_figures(page)
    assert beam.x_range is not reference.x_range
    # the venue points are stretched over the venue ranges
    points = page.venue_points_figure
    assert isinstance(points.x_range, Range1d)
    assert [points.x_range.start, points.x_range.end] == ranges.venue_x
    assert [points.y_range.start, points.y_range.end] == ranges.venue_y
    assert not points.match_aspect


def test_showing_page_5_renews_the_beam_ranges(ran):
    page = ran.pages[5]
    _shown(ran, 1)
    before = [(f.x_range, f.y_range) for f in _beam_figures(page)]
    _shown(ran)
    after = [(f.x_range, f.y_range) for f in _beam_figures(page)]
    for (x0, y0), (x1, y1) in zip(before, after):
        assert x1 is not x0 and y1 is not y0
        assert isinstance(x1, DataRange1d) and isinstance(y1, DataRange1d)


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
    ``pan`` while panning; not the automatic computation of the ranges);
    the code reads the ranges the figures have when it runs, so renewed
    ranges stay linked."""
    page = PalcApp(Document()).pages[5]
    beam, reference = _beam_figures(page)
    for source, target in ((beam, reference), (reference, beam)):
        (callback,) = source.js_event_callbacks['rangesupdate']
        assert source.js_event_callbacks['pan'] == [callback]
        assert isinstance(callback, CustomJS)
        assert callback.args == {'source': source, 'target': target}
        assert 'have_updated_interactively' in callback.code
        assert 'x_range' in callback.code and 'y_range' in callback.code
        # a box zoom keeps the equal scaling
        (box_zoom,) = [tool for tool in source.tools
                       if isinstance(tool, BoxZoomTool)]
        assert box_zoom.match_aspect
    # the venue points figure is not linked (it is stretched)
    assert 'rangesupdate' not in page.venue_points_figure.js_event_callbacks
