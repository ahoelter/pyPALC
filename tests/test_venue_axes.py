"""Axis modes of the responsive venue figure.

The venue figure follows the width of the window. Its three axis modes
(``app.venue_axes_select``):

* 0, "Equal, variable height": equal scaling; the height follows the width
  (aspect ratio of the venue) up to :data:`~pypalc.gui.ranges.VENUE_MAX_HEIGHT`.
* 1, "Equal, fixed height": equal scaling at the fixed height; the whole
  venue stays visible and one range widens to fill the figure.
* 2, "Stretched": the venue ranges fill the figure, as before the port.

Equal scaling is done in the browser: both ranges are ``DataRange1d`` that
cover the venue ranges (the data of the hidden bounds renderer) and the
figure matches the aspect (``match_aspect``, ``aspect_scale`` 1), so Bokeh
widens one range until one meter has the same length on both axes, for
every size of the figure. The browser check (``tools/browser_check.py``,
checks ``equal_axes`` and ``axis_labels``) measures that.
"""
import panel_material_ui as pmui
import pytest
from bokeh.document import Document
from bokeh.models import DataRange1d, Range1d

from pypalc.core.settings import load_setup
from pypalc.gui.app import PalcApp
from pypalc.gui.ranges import (VARIABLE_MAX_HEIGHT, VENUE_BOUNDS,
                               VENUE_MAX_HEIGHT)
from tests.gui_driver import read, write
from tests.gui_steps import ROOT

MODES = ['Equal, variable height', 'Equal, fixed height', 'Stretched']


@pytest.fixture
def app():
    """An app with the audience lines of case ``simple`` drawn."""
    app = PalcApp(Document())
    setup = load_setup(ROOT / 'tests' / 'cases' / 'simple.yaml')
    app.pages[1].draw_lines(setup.venue.audience_lines)
    return app


def _bounds(app):
    return app.venue_figure.select_one({'name': VENUE_BOUNDS})


def test_axes_select_is_material_button_group():
    app = PalcApp(Document())
    select = app.venue_axes_select
    assert isinstance(select, pmui.RadioButtonGroup)
    assert read(select, 'options') == MODES
    assert read(select, 'active') == 2


@pytest.mark.parametrize('mode', [0, 1])
def test_equal_modes_match_aspect(app, mode):
    write(app.venue_axes_select, 'active', mode)
    figure, ranges = app.venue_figure, app.state.ranges
    assert figure.match_aspect is True
    assert figure.aspect_scale == 1
    bounds = _bounds(app)
    assert bounds is not None and not bounds.visible
    assert list(bounds.data_source.data['x']) == list(ranges.venue_x)
    assert list(bounds.data_source.data['y']) == list(ranges.venue_y)
    for axis_range in (figure.x_range, figure.y_range):
        assert isinstance(axis_range, DataRange1d)
        # the venue ranges, widened in the browser to equal scaling
        assert axis_range.renderers == [bounds]
        assert axis_range.range_padding == 0
    assert figure.sizing_mode == 'stretch_width'


def test_stretched_fills_figure_with_venue_ranges(app):
    write(app.venue_axes_select, 'active', 0)
    write(app.venue_axes_select, 'active', 2)
    figure, ranges = app.venue_figure, app.state.ranges
    assert figure.match_aspect is False
    assert isinstance(figure.x_range, Range1d)
    assert isinstance(figure.y_range, Range1d)
    assert [figure.x_range.start, figure.x_range.end] == ranges.venue_x
    assert [figure.y_range.start, figure.y_range.end] == ranges.venue_y
    assert figure.sizing_mode == 'stretch_width'
    assert figure.height == VENUE_MAX_HEIGHT
    assert figure.aspect_ratio is None and figure.max_height is None


def test_variable_height_follows_width_up_to_maximum(app):
    write(app.venue_axes_select, 'active', 0)
    figure, ranges = app.venue_figure, app.state.ranges
    width = ranges.venue_x[1] - ranges.venue_x[0]
    height = ranges.venue_y[1] - ranges.venue_y[0]
    assert figure.aspect_ratio == pytest.approx(width / height)
    assert figure.max_height == VENUE_MAX_HEIGHT
    assert figure.height is None
    # without a percentage, so that the card around the figure is sized to
    # the capped height (Bokeh's own rule is min(max_height, 100%)); at
    # most VENUE_MAX_HEIGHT, less on low windows
    assert figure.styles['max-height'] == VARIABLE_MAX_HEIGHT == \
        'min(500px, max(200px, calc(100vh - 210px)))'
    assert figure.styles['width'] == '100%'


def test_fixed_height(app):
    write(app.venue_axes_select, 'active', 0)
    write(app.venue_axes_select, 'active', 1)
    figure = app.venue_figure
    assert figure.height == VENUE_MAX_HEIGHT
    assert figure.aspect_ratio is None and figure.max_height is None
    assert figure.styles == {}


def test_new_line_keeps_axis_mode(app):
    """Saving a line refits the ranges in the selected mode."""
    write(app.venue_axes_select, 'active', 0)
    venue = app.pages[1]
    for widget, value in ((venue.start_x_input, '20'),
                          (venue.start_y_input, '3'),
                          (venue.end_x_input, '30'),
                          (venue.end_y_input, '9')):
        write(widget, 'value', value)
    venue.save_line()
    ranges = app.state.ranges
    assert ranges.venue_x[1] == 32 and ranges.venue_y[1] == 11
    assert list(_bounds(app).data_source.data['x']) == list(ranges.venue_x)
    assert isinstance(app.venue_figure.x_range, DataRange1d)
    assert app.venue_figure.aspect_ratio == pytest.approx(
        (ranges.venue_x[1] - ranges.venue_x[0])
        / (ranges.venue_y[1] - ranges.venue_y[0]))


@pytest.mark.parametrize('mode', [0, 1])
def test_showing_a_venue_page_renews_equal_ranges(app, mode):
    """BokehJS keeps the bounds of every view of a figure in a DataRange1d;
    a page that draws the figure anew gets new ranges, so that the new
    view alone sets the scaling."""
    write(app.venue_axes_select, 'active', mode)
    for page in (4, 2, 3, 1):
        x_range, y_range = app.venue_figure.x_range, app.venue_figure.y_range
        write(app.page_select, 'active', page)
        assert app.venue_figure.x_range is not x_range, page
        assert app.venue_figure.y_range is not y_range, page
        assert isinstance(app.venue_figure.x_range, DataRange1d)
        assert app.venue_figure.match_aspect


def test_showing_a_page_keeps_stretched_ranges(app):
    x_range = app.venue_figure.x_range
    for page in (4, 2, 3, 5, 0, 1):
        write(app.page_select, 'active', page)
        assert app.venue_figure.x_range is x_range, page


@pytest.mark.parametrize('mode', [0, 1])
def test_ranges_are_renewed_after_the_page_is_swapped(app, mode):
    """The new ranges are set after the content shows the new page, so the
    view of the page that is left cannot write its bounds into them."""
    write(app.venue_axes_select, 'active', mode)
    events = []
    app.content.param.watch(lambda event: events.append('swap'), 'objects')
    scale = app.scale_venue_axes

    def logged_scale():
        events.append('ranges')
        scale()
    app.scale_venue_axes = logged_scale
    for page in (4, 2, 3, 1):
        events.clear()
        write(app.page_select, 'active', page)
        assert events == ['swap', 'ranges'], page


@pytest.mark.parametrize('mode', [0, 1])
def test_new_equal_ranges_are_laid_out(app, mode):
    """BokehJS lays out a shown figure with new ranges only if a layout
    property changes (else its axes lose their labels): every renewal of
    the equal ranges changes the left border, between None and the same
    border."""
    figure = app.venue_figure
    write(app.venue_axes_select, 'active', mode)
    borders = []
    for _ in range(3):
        before = figure.min_border_left
        app.scale_venue_axes()
        assert figure.min_border_left != before
        borders.append(figure.min_border_left)
    assert set(borders) == {None, figure.min_border}
