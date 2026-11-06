"""Axis modes of the responsive venue figure.

The venue figure follows the width of the window. Its three axis modes
(``app.venue_axes_select``):

* 0, "Equal, variable height": equal scaling; the figure is as wide as in
  "Stretched" and its height follows, up to the window limit (it ends
  inside the window; then it is narrower).
* 1, "Equal, fixed height": equal scaling; the figure is as high as in
  "Stretched" (less on low windows) and its width follows, up to the
  available width (then it is lower).

A figure narrower than its card is centered in it.
* 2, "Stretched": the venue ranges fill the figure, as before the port.

In every mode the ranges are the data extent (the venue ranges with their
margin), never widened. Equal scaling is done in the browser by the
containers of the figure (:class:`~pypalc.gui.ranges.ScaledFigure`, one per
page), which size the figure so that its frame has the aspect of the
ranges. The browser check (``tools/browser_check.py``, checks
``equal_axes`` and ``axis_labels``) measures that.
"""
import panel_material_ui as pmui
import pytest
from bokeh.document import Document
from bokeh.models import BoxZoomTool, Range1d

from pypalc.core.settings import load_setup
from pypalc.gui.app import PalcApp
from pypalc.gui.ranges import (FIXED_HEIGHT, MIN_MAX_HEIGHT, STRETCHED,
                               VARIABLE_HEIGHT, VENUE_MAX_HEIGHT,
                               WINDOW_MARGIN, ScaledFigure, fixed_height,
                               window_limit)
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
