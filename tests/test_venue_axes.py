"""Axis modes of the responsive venue figure.

The venue figure follows the width of the window. Its three axis modes
(``app.venue_axes_select``):

* 0, "Equal, variable height": equal scaling; the figure is as wide as in
  "Stretched" and its height follows, up to the window limit (it ends
  inside the window; then it is narrower).
* 1, "Equal, fixed height": equal scaling; the figure is as high as in
  "Stretched" (less on low windows) and its width follows, up to the
  available width (then it is lower).
* 2, "Stretched": the venue ranges fill the figure, as before the port.

A figure narrower than its card is centered in it.

In every mode the ranges are the data extent (the venue ranges with their
margin), never widened. Equal scaling is done in the browser by the
containers of the figure (:class:`~pypalc.gui.ranges.ScaledFigure`, one per
page), which size the figure so that its frame has the aspect of the
ranges. In the equal modes both axes share one ticker that gives them the
same tick spacing (:func:`~pypalc.gui.ranges.equal_ticker`); "Stretched"
has Bokeh's default tickers. The browser check (``tools/browser_check.py``,
checks ``equal_axes`` and ``axis_labels``) measures that.
"""
import json
import math
import shutil
import subprocess

import panel_material_ui as pmui
import pytest
from bokeh.document import Document
from bokeh.models import BasicTicker, BoxZoomTool, CustomJSTicker, Range1d
from bokeh.plotting import figure as new_figure

from pypalc.core.settings import load_setup
from pypalc.gui.app import PalcApp
from pypalc.gui.ranges import (FIXED_HEIGHT, MIN_MAX_HEIGHT, STRETCHED,
                               VARIABLE_HEIGHT, VENUE_MAX_HEIGHT,
                               WINDOW_MARGIN, ScaledFigure, DESIRED_TICKS,
                               MIN_TICK_PX, MINOR_TICKS, equal_ticker,
                               fixed_height, set_data_ranges, window_limit)
from tests.gui_driver import read, write
from tests.gui_steps import ROOT
from tests.test_shell import rendered

MODES = ['Equal, variable height', 'Equal, fixed height', 'Stretched']
SCALINGS = [VARIABLE_HEIGHT, FIXED_HEIGHT, STRETCHED]


@pytest.fixture
def app():
    """A rendered app with the audience lines of case ``simple`` drawn and
    every page built."""
    app = rendered(PalcApp(Document()))
    setup = load_setup(ROOT / 'tests' / 'cases' / 'simple.yaml')
    app.pages[1].draw_lines(setup.venue.audience_lines)
    for page in app.pages.values():
        page.layout()
    return app


def _box_zoom(figure):
    (tool,) = [tool for tool in figure.tools if isinstance(tool, BoxZoomTool)]
    return tool


def test_axes_select_is_material_button_group():
    app = PalcApp(Document())
    select = app.venue_axes_select
    assert isinstance(select, pmui.RadioButtonGroup)
    assert read(select, 'options') == MODES
    assert read(select, 'active') == 2


def test_every_venue_page_shows_the_figure_in_a_scaled_container(app):
    panes = app.venue_panes
    assert len(panes) == 4
    for pane in panes:
        assert isinstance(pane, ScaledFigure)
        assert pane.object.object is app.venue_figure
        assert pane.sizing_mode == 'stretch_width'
        # only the window limits "Equal, variable height"
        assert pane.largest_height is None
    for index in (1, 2, 3, 4):
        layout = app.pages[index].layout()
        assert sum(pane in layout.select(ScaledFigure)
                   for pane in panes) == 1, index


@pytest.mark.parametrize('mode', [0, 1, 2])
def test_ranges_are_the_data_extent_in_every_mode(app, mode):
    """The ranges are the venue ranges (data extent and its margin), not
    widened for equal scaling: the figure is sized instead."""
    write(app.venue_axes_select, 'active', mode)
    figure, ranges = app.venue_figure, app.state.ranges
    # the lines and the array (x 0 to 20, y 0 to 3) with a margin of 2 m
    assert ranges.venue_x == [-2, 22] and ranges.venue_y == [-2, 5]
    for axis_range, venue in ((figure.x_range, ranges.venue_x),
                              (figure.y_range, ranges.venue_y)):
        assert type(axis_range) is Range1d
        assert [axis_range.start, axis_range.end] == venue
    assert figure.match_aspect is False
    # the figure itself has the size of "Stretched"; the browser sizes it
    assert figure.sizing_mode == 'stretch_width'
    assert figure.height == VENUE_MAX_HEIGHT
    assert figure.aspect_ratio is None and figure.max_height is None
    assert figure.frame_width is None and figure.frame_height is None
    assert figure.styles == {}
    for pane in app.venue_panes:
        assert pane.scaling == SCALINGS[mode]
        assert pane.data_aspect == pytest.approx(24 / 7)
    # a box zoom keeps the equal scaling; in "Stretched" it is free
    assert _box_zoom(figure).match_aspect is (mode != 2)


def test_main_area_keeps_the_scrollbar_space():
    """The width of the pages, from which "Equal, variable height" sets the
    height, does not change when the main area shows a scrollbar."""
    view = PalcApp(Document()).view()
    assert view.sx == {'& .main-content': {'scrollbarGutter': 'stable'}}


def test_height_limits():
    """"Equal, variable height" is limited by the window only; "Equal,
    fixed height" has the height of "Stretched", less on low windows."""
    assert (VENUE_MAX_HEIGHT, MIN_MAX_HEIGHT, WINDOW_MARGIN) == \
        (500, 200, 210)
    assert window_limit(994) == 784  # 1920 x 1080
    assert window_limit(714) == 504  # 1280 x 800
    assert window_limit(682) == 472  # 1366 x 768
    assert window_limit(300) == 200
    assert fixed_height(994) == 500
    assert fixed_height(714) == 500
    assert fixed_height(682) == 472
    assert fixed_height(300) == 200


def test_scaled_figure_code_has_the_constants():
    code = ScaledFigure._esm
    for name in ('STRETCHED', 'VARIABLE_HEIGHT', 'FIXED_HEIGHT',
                 'VENUE_MAX_HEIGHT', 'MIN_MAX_HEIGHT', 'WINDOW_MARGIN'):
        assert name not in code
    for text in ("'stretched'", "'variable_height'", 'Math.max(200,',
                 'innerHeight - 210', 'Math.min(limit, model.largest_height)',
                 'h = Math.min(500, limit) - border_y',
                 "style.marginLeft = style.marginRight = width ? 'auto'"):
        assert text in code, text


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
    assert ranges.venue_x == [-2, 32] and ranges.venue_y == [-2, 11]
    figure = app.venue_figure
    assert [figure.x_range.start, figure.x_range.end] == [-2, 32]
    assert [figure.y_range.start, figure.y_range.end] == [-2, 11]
    for pane in app.venue_panes:
        assert pane.scaling == VARIABLE_HEIGHT
        assert pane.data_aspect == pytest.approx(34 / 13)


@pytest.mark.parametrize('mode', [0, 1, 2])
def test_showing_a_page_keeps_the_ranges(app, mode):
    """A page switch keeps the ranges, so a zoom stays: in the equal modes
    a zoom keeps the aspect of the ranges, which is the aspect of the frame
    on every page."""
    write(app.venue_axes_select, 'active', mode)
    x_range, y_range = app.venue_figure.x_range, app.venue_figure.y_range
    for page in (4, 2, 3, 5, 0, 1):
        write(app.page_select, 'active', page)
        assert app.venue_figure.x_range is x_range, page
        assert app.venue_figure.y_range is y_range, page


def test_a_mode_change_gives_new_ranges(app):
    """A new mode resets a zoom (new ranges over the data extent)."""
    for mode in (0, 1, 2, 0):
        x_range = app.venue_figure.x_range
        write(app.venue_axes_select, 'active', mode)
        assert app.venue_figure.x_range is not x_range
        assert [app.venue_figure.x_range.start,
                app.venue_figure.x_range.end] == app.state.ranges.venue_x


def assert_equal_ticks(figure):
    """Both axes of an equally scaled figure share one ticker
    (:func:`~pypalc.gui.ranges.equal_ticker` of this figure); the grids
    follow the axes."""
    (x_axis,), (y_axis,) = figure.xaxis, figure.yaxis
    assert isinstance(x_axis.ticker, CustomJSTicker)
    assert x_axis.ticker is y_axis.ticker
    assert x_axis.ticker.args == {'plot': figure}
    for grid in figure.grid:
        assert grid.ticker is None


def assert_default_ticks(figure):
    """Bokeh's default tickers: one BasicTicker per axis."""
    (x_axis,), (y_axis,) = figure.xaxis, figure.yaxis
    for axis in (x_axis, y_axis):
        assert type(axis.ticker) is BasicTicker
        assert axis.ticker.desired_num_ticks == 6
        assert axis.ticker.num_minor_ticks == 5
    assert x_axis.ticker is not y_axis.ticker


@pytest.mark.parametrize('mode', [0, 1, 2])
def test_equal_modes_share_the_tick_spacing(app, mode):
    """In the equal modes both axes have the same tick spacing (one
    ticker); "Stretched" has Bokeh's default tickers."""
    write(app.venue_axes_select, 'active', mode)
    if mode == 2:
        assert_default_ticks(app.venue_figure)
    else:
        assert_equal_ticks(app.venue_figure)


def test_mode_changes_swap_the_tickers(app):
    """Every mode change gives the tickers of the new mode, in both
    directions; a change between the equal modes or new venue ranges keep
    the shared ticker."""
    figure = app.venue_figure
    for mode in (0, 2, 1, 0, 2, 2, 1):
        write(app.venue_axes_select, 'active', mode)
        if mode == 2:
            assert_default_ticks(figure)
        else:
            assert_equal_ticks(figure)
    ticker = figure.xaxis[0].ticker
    app.update_venue_ranges()
    assert figure.xaxis[0].ticker is ticker


def _browser_ticks(ticker, x, y, size):
    """Major and minor ticks the browser code of ``ticker`` gives on both
    axes of a figure with the ranges ``x``, ``y`` and the frame size
    ``size`` (``None``: before the first layout), run with node."""
    script = """
    const [major_code, minor_code, x, y, size] = %s;
    // BokehJS throws on reading the frame size before the first layout
    const frame = i => {
      if (!size) throw new Error('inner_width is unset');
      return size[i];
    };
    const plot = {x_range: {start: x[0], end: x[1]},
                  y_range: {start: y[0], end: y[1]},
                  get inner_width() { return frame(0); },
                  get inner_height() { return frame(1); }};
    const out = {};
    for (const [name, range] of [['x', plot.x_range], ['y', plot.y_range]]) {
      const cb_data = {start: range.start, end: range.end, range: range,
                       cross_loc: NaN};
      const major = new Function('cb_data', 'plot', major_code)(
        cb_data, plot);
      const minor = new Function('cb_data', 'plot', minor_code)(
        Object.assign({major_ticks: major}, cb_data), plot);
      out[name] = [major, minor];
    }
    console.log(JSON.stringify(out));
    """ % json.dumps([ticker.major_code, ticker.minor_code, x, y, size])
    done = subprocess.run(['node', '-e', script], capture_output=True,
                          text=True, check=True)
    return json.loads(done.stdout)


def _spacing(ticks):
    steps = {round(b - a, 9) for a, b in zip(ticks, ticks[1:])}
    assert len(steps) == 1, ticks
    return steps.pop()


@pytest.mark.skipif(shutil.which('node') is None, reason='needs node')
@pytest.mark.parametrize('x, y, size, major, minor', [
    # the longer axis gets the interval of Bokeh's BasicTicker (about six
    # ticks); the shorter one has two ticks with it
    ([-2, 22], [-2, 5], (1000, 292), 5, 1),
    ([-2, 5], [-2, 22], (292, 1000), 5, 1),
    # the shorter axis would get one tick: a smaller interval, as long as
    # the ticks stay at least MIN_TICK_PX apart
    ([0, 30], [0, 4], (1200, 160), 2, 0.4),
    ([0.5, 30.5], [0.5, 4.5], (1200, 160), 2, 0.4),
    ([0, 100], [0, 3], (1200, 36), 5, 1),
    # before the first layout (no frame size): Bokeh's interval
    ([0, 30], [0, 4], None, 5, 1),
    # small and zoomed ranges
    ([0.1, 0.4], [0.2, 0.35], (600, 300), 0.05, 0.01),
])
def test_equal_ticks_in_the_browser(x, y, size, major, minor):
    """The browser code gives both axes the same major and minor tick
    spacing, multiples of it within the ranges."""
    figure = new_figure()
    set_data_ranges(figure, x, y, equal=True)
    ticks = _browser_ticks(figure.xaxis[0].ticker, x, y, size)
    for name, extent in (('x', x), ('y', y)):
        majors, minors = ticks[name]
        assert majors and minors
        for values, spacing in ((majors, major), (minors, minor)):
            for value in values:
                assert extent[0] - 1e-9 <= value <= extent[1] + 1e-9
                assert value / spacing == pytest.approx(
                    round(value / spacing), abs=1e-6)
            if len(values) > 1:
                assert _spacing(values) == pytest.approx(spacing)
        # all multiples of the spacing in the range
        assert len(majors) == math.floor(extent[1] / major + 1e-9) - \
            math.ceil(extent[0] / major - 1e-9) + 1


def test_equal_ticker_is_new_per_figure():
    """Each figure gets its own ticker, which reads the ranges the figure
    has when the ticks are computed (new ranges need no new ticker)."""
    first, second = new_figure(), new_figure()
    assert equal_ticker(first) is not equal_ticker(second)
    assert equal_ticker(first).args['plot'] is first
    set_data_ranges(first, [0, 1], [0, 1], equal=True)
    ticker = first.xaxis[0].ticker
    set_data_ranges(first, [0, 2], [0, 1], equal=True)
    assert first.xaxis[0].ticker is ticker
    set_data_ranges(first, [0, 2], [0, 1], equal=False)
    assert_default_ticks(first)


def test_equal_ticker_code_has_the_constants():
    """The density is Bokeh's: the numbers of ticks of its BasicTicker."""
    default = BasicTicker()
    assert (DESIRED_TICKS, MINOR_TICKS, MIN_TICK_PX) == \
        (default.desired_num_ticks, default.num_minor_ticks, 40)
    ticker = equal_ticker(new_figure())
    for code in (ticker.major_code, ticker.minor_code):
        for name in ('DESIRED_TICKS', 'MIN_TICK_PX', 'MINOR_TICKS'):
            assert name not in code
        assert 'Math.abs(6 - long' in code and 'next * ppm >= 40' in code
    assert ticker.minor_code.endswith('interval / 5);')
