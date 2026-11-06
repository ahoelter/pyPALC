"""Ranges and axis scaling of the Bokeh figures.

:class:`PlotRanges` (formerly ``Plot_ranges`` in ``fun/PALC_classes.py``)
and the helpers :func:`show_rays` and :func:`scale_venue_figure` (formerly
``venue_slice2CDS`` and ``get_p_axis_rng`` in ``fun/gui_helper.py``), the
latter with the axis modes of the responsive venue figure, and
:class:`ScaledFigure`, which sizes a figure in the browser so that one
meter has the same length on both axes (venue figure and beam plots).

author: Arne Hoelter
"""
from __future__ import annotations

import numpy as np
import param
from bokeh.models import BoxZoomTool, Range1d
from panel.custom import Child, JSComponent

#: Height in px of the venue figure in the axis modes "Stretched" and
#: "Equal, fixed height" (the latter lower if the window is too low, see
#: :func:`fixed_height`, or too narrow for its width), and the largest
#: height of the beam plots.
VENUE_MAX_HEIGHT = 500
#: Space in px of the window above and below an equally scaled figure
#: (header, page tabs, the tabs of pages 2 and 4, card margins): it is at
#: most the window height less this space (:func:`window_limit`), so that
#: it ends inside the window.
WINDOW_MARGIN = 210
#: Smallest value of that limit in px, for very low windows.
MIN_MAX_HEIGHT = 200
#: Scalings of :class:`ScaledFigure`: the figure fills the width of its
#: container at the height :data:`VENUE_MAX_HEIGHT` with the data
#: stretched ("Stretched"), or it is sized for equal scaling with the width
#: of its container ("Equal, variable height") or with the height
#: :data:`VENUE_MAX_HEIGHT` ("Equal, fixed height").
STRETCHED = 'stretched'
VARIABLE_HEIGHT = 'variable_height'
FIXED_HEIGHT = 'fixed_height'
#: The scaling of each axis mode of the venue figure, by its index
#: (``app.venue_axes_select.active``).
AXIS_SCALINGS = (VARIABLE_HEIGHT, FIXED_HEIGHT, STRETCHED)


def window_limit(window_height):
    """
    Largest height in px of an equally scaled figure in a window of the
    given height, so that it ends inside the window (what
    :class:`ScaledFigure` computes in the browser): the limit of "Equal,
    variable height".

    Parameters
    ----------
    window_height : float
        Inner height of the browser window in px.

    Returns
    -------
    float
        The window height less :data:`WINDOW_MARGIN`, at least
        :data:`MIN_MAX_HEIGHT`.

    """
    return max(MIN_MAX_HEIGHT, window_height - WINDOW_MARGIN)


def fixed_height(window_height):
    """
    Height in px of a figure in "Equal, fixed height" in a window of the
    given height, if the window is wide enough.

    Parameters
    ----------
    window_height : float
        Inner height of the browser window in px.

    Returns
    -------
    float
        :data:`VENUE_MAX_HEIGHT`, at most :func:`window_limit`.

    """
    return min(VENUE_MAX_HEIGHT, window_limit(window_height))


#: Browser code of :class:`ScaledFigure`. ``fit`` measures the borders of
#: the figure around its frame (axes, title, toolbar, color bar) in its
#: last layout, computes the frame size for equal scaling and sets the
#: outer size of the figure to it plus the borders (inline styles of its
#: element, which win over the sizes of its model, in whole px, as Bokeh
#: measures them); it then lays the figure out again at once, so that
#: Bokeh's frame is the outer size less the borders. It runs when the
#: scaling or the aspect changes, after each layout (the borders may change
#: with the tick labels; a size that is right is not set again) and when
#: the container or the window is resized.
_SCALED_FIGURE_JS = """
export function render({ model, data, el, view }) {
  el.append(model.get_child('object'));
  let layouts = 0, calm = null;
  function fit() {
    const figure = view.get_child_view(data.object);
    if (figure == null || figure.frame == null) return;
    const style = figure.el.style;
    const scaling = model.scaling, aspect = model.data_aspect;
    let width = '', height = '';
    if (scaling !== 'STRETCHED' && aspect > 0) {
      const available = el.clientWidth;
      const outer = figure.bbox, frame = figure.frame.bbox;
      if (!(available > 0 && frame.width > 0 && frame.height > 0)) return;
      // the borders around the frame in the last layout, in px
      const border_x = outer.width - frame.width;
      const border_y = outer.height - frame.height;
      // the figure ends inside the window
      let limit = Math.max(MIN_MAX_HEIGHT,
                           window.innerHeight - WINDOW_MARGIN);
      let w, h;  // size of the frame, w / h = aspect
      if (scaling === 'VARIABLE_HEIGHT') {
        if (model.largest_height != null)
          limit = Math.min(limit, model.largest_height);
        w = available - border_x;
        h = w / aspect;
        if (h + border_y > limit) {
          h = limit - border_y;
          w = h * aspect;
        }
      } else {
        h = Math.min(VENUE_MAX_HEIGHT, limit) - border_y;
        w = h * aspect;
        if (w + border_x > available) {
          w = available - border_x;
          h = w / aspect;
        }
      }
      width = Math.round(Math.max(w, 1) + border_x) + 'px';
      height = Math.round(Math.max(h, 1) + border_y) + 'px';
    }
    if (style.width !== width || style.height !== height) {
      style.width = width;
      style.height = height;
      // a figure narrower than its container is centered in it
      style.marginLeft = style.marginRight = width ? 'auto' : '';
    }
    // lay the figure out again if its size in Bokeh's last layout is not
    // its size now; stop if that never settles
    const now = figure.el.getBoundingClientRect();
    if (figure.bbox.width === Math.floor(now.width) &&
        figure.bbox.height === Math.floor(now.height)) return;
    if (++layouts > 20) return;
    clearTimeout(calm);
    calm = setTimeout(() => { layouts = 0; }, 1000);
    figure.compute_layout();
  }
  model.on(['scaling', 'data_aspect', 'largest_height'], () => {
    layouts = 0;
    fit();
  });
  model.on('after_layout', fit);
  model.on('resize', fit);
  window.addEventListener('resize', fit);
  model.on('remove', () => window.removeEventListener('resize', fit));
}
"""
for _name in ('STRETCHED', 'VARIABLE_HEIGHT', 'FIXED_HEIGHT'):
    _SCALED_FIGURE_JS = _SCALED_FIGURE_JS.replace(
        "'%s'" % _name, repr(globals()[_name]))
for _name in ('VENUE_MAX_HEIGHT', 'MIN_MAX_HEIGHT', 'WINDOW_MARGIN'):
    _SCALED_FIGURE_JS = _SCALED_FIGURE_JS.replace(
        _name, str(globals()[_name]))


class ScaledFigure(JSComponent):
    """
    A Bokeh figure in a container that follows the width of the page,
    sized in the browser for its axis scaling (:data:`STRETCHED`,
    :data:`VARIABLE_HEIGHT`, :data:`FIXED_HEIGHT`).

    The figure itself has the size of "Stretched": the width of the
    container and the height :data:`VENUE_MAX_HEIGHT`. In the equal
    scalings its ranges are the data extent (``Range1d``) and the browser
    sizes the figure so that its frame has the aspect ``data_aspect`` (x
    span / y span of the ranges), so one meter has the same length on both
    axes:

    * :data:`VARIABLE_HEIGHT`: the width of the container; the height
      follows. If the figure would run below the window (higher than
      :func:`window_limit`, or than ``largest_height`` if given), it gets
      that height and less width.
    * :data:`FIXED_HEIGHT`: the height :data:`VENUE_MAX_HEIGHT` (less on
      low windows, :func:`fixed_height`); the width follows. If the figure
      would be wider than the container, it gets the width of the
      container and less height.

    A figure narrower than its container is centered in it.

    Only the browser knows the width of the container, the height of the
    window and the space the axes, the title and the toolbar take around
    the frame, so the size is computed there (:data:`_SCALED_FIGURE_JS`):
    after each layout the borders around the frame are measured, and the
    outer size is set to the frame size plus these borders. A zoom keeps
    the aspect of the ranges (the box zoom matches the aspect of the frame,
    :func:`set_data_ranges`), so the scaling stays equal while zoomed and
    after a page switch.

    Parameters
    ----------
    object : panel.pane.Bokeh
        Pane of the figure.
    scaling : str
        :data:`STRETCHED`, :data:`VARIABLE_HEIGHT` or :data:`FIXED_HEIGHT`.
    data_aspect : float
        Width / height of the data extent in m / m (the ranges of the
        figure).
    largest_height : int or None
        Largest height in px in :data:`VARIABLE_HEIGHT` besides the window
        limit (the beam plots: :data:`VENUE_MAX_HEIGHT`); None: only the
        window limit.
    """

    object = Child(doc="Pane of the figure.")

    scaling = param.Selector(default=STRETCHED, objects=list(AXIS_SCALINGS),
                             doc="Axis scaling of the figure.")

    data_aspect = param.Number(default=1.0, bounds=(0, None),
                               doc="Width / height of the data extent.")

    largest_height = param.Integer(default=None, allow_None=True,
                                   bounds=(1, None),
                                   doc="Largest height in variable height.")

    _esm = _SCALED_FIGURE_JS


def set_data_ranges(figure, x, y, equal):
    """
    Gives a figure new ranges over the data extent (a zoom is reset).

    Parameters
    ----------
    figure : figure [out]
        The figure.
    x, y : list of float
        Start and end of the x- and y-range.
    equal : bool
        If True, the figure is equally scaled (:class:`ScaledFigure`): its
        box zoom keeps the aspect of the frame, so a zoom keeps the equal
        scaling. If False, the box zoom draws any box.

    Returns
    -------
    None.

    """
    figure.update(x_range=Range1d(x[0], x[1]), y_range=Range1d(y[0], y[1]))
    for tool in figure.tools:
        if isinstance(tool, BoxZoomTool):
            tool.match_aspect = equal


def data_aspect(x, y):
    """
    Width / height of a data extent.

    Parameters
    ----------
    x, y : list of float
        Start and end of the x- and y-range.

    Returns
    -------
    float
        ``(x[1] - x[0]) / (y[1] - y[0])``; 1 for an empty y-range.

    """
    height = y[1] - y[0]
    return (x[1] - x[0]) / height if height else 1.0


class PlotRanges():
    """ Class to create objects that contain ranges of bokeh figures. """
    def __init__(self, venue_x=None, venue_y=None, histogram_x=None, \
                 histogram_top=None, homogeneity_x=None, homogeneity_y=None, sound_field_y=None, \
                 spl_over_distance_y=None):
        # None stands for a new list with the default value (no shared mutable defaults)
        venue_x = [-0.5,2.5] if venue_x is None else venue_x
        venue_y = [-0.5,2.5] if venue_y is None else venue_y
        histogram_x = [0,0] if histogram_x is None else histogram_x
        histogram_top = [0,0] if histogram_top is None else histogram_top
        homogeneity_y = [0,1] if homogeneity_y is None else homogeneity_y
        sound_field_y = [0,1] if sound_field_y is None else sound_field_y
        spl_over_distance_y = [0.3,15] if spl_over_distance_y is None else spl_over_distance_y
        self.venue_x, self.venue_y          = venue_x, venue_y
        self.histogram_x, self.histogram_top = histogram_x, histogram_top
        self.homogeneity_y, self.sound_field_y        = homogeneity_y, sound_field_y
        self.spl_over_distance_y             = spl_over_distance_y
    
    def update_range(self, plot_axis, overlap, *plot_data):
        """
        Updates the range of bokeh figures.

        Parameters
        ----------
        plot_axis : str
            Figure and axis information.
        overlap : list
            Overlap to max and min value on considered axis.
        *plot_data : list
            List with all data points to consider.

        Returns
        -------
        new_range : obj
            Range1d object to be used on bokeh figures.

        """
        # number of inputs
        shape = len(plot_data)
        # initialize list outputs
        axmin, axmax = [], []
        # get min and max values of input array
        for n in range(shape):
            axmin.append(np.amin(plot_data[n]))
            axmax.append(np.amax(plot_data[n]))
        # write min and max value of previously detected min and max to attribute
        setattr(self, plot_axis, [min(axmin)+overlap[0], max(axmax)+overlap[1]])
        # write new range to bokeh Range1d object and return
        if plot_axis in ['histogram_top']:
            new_range = Range1d(-1, getattr(self, plot_axis)[1])
        else:
            new_range = Range1d(getattr(self, plot_axis)[0], getattr(self, plot_axis)[1])
        return new_range


def show_rays(rays_source, hit_points_source, \
                    ray_plot, optimized_array):
    """
    Stores venue slice data in a ColumnDataSource to plot the data in the
    venue slice plot (:attr:`pypalc.gui.app.PalcApp.venue_figure`).

    Parameters
    ----------
    rays_source : ColumnDataSource [out]
        First CDS to visualize the rays of the LSA cabinets to the venue.
    hit_points_source : ColumnDataSource [out]
        Second CDS to visualize the LSA cabinets.
    ray_plot : obj [in]
        Contains the plotting data.
    optimized_array : obj [in]
        Contains the data of the optimized LSA array by PALC.

    Returns
    -------
    None.

    """
    if optimized_array != []:
        rays_source.data = dict(center_rays_x      = ray_plot.center_rays_x, \
                                     center_rays_y       = ray_plot.center_rays_y, \
                                     start_rays_x  = ray_plot.start_rays_x, \
                                     start_rays_y  = ray_plot.start_rays_y, \
                                     stop_rays_x  = ray_plot.stop_rays_x, \
                                     stop_rays_y  = ray_plot.stop_rays_y)
        hit_points_source.data = dict(hit_points_x        = optimized_array.hit_points[:,0], \
                                     hit_points_y         = optimized_array.hit_points[:,1], \
                                     hit_points_start_x   = optimized_array.hit_points_start[:,0], \
                                     hit_points_start_y   = optimized_array.hit_points_start[:,1], \
                                     hit_points_stop_x    = optimized_array.hit_points_stop[:,0], \
                                     hit_points_stop_y    = optimized_array.hit_points_stop[:,1])
    else:
        hit_points_source.data = dict(hit_points_x=[], hit_points_y=[], \
                                      hit_points_start_x=[], hit_points_start_y=[], \
                                      hit_points_stop_x=[], hit_points_stop_y=[])
        rays_source.data = dict(x_start_stop=[], y_start_stop=[], \
                                      x_fin_unitn_start_stop=[], \
                                      y_fin_unitn_start_stop=[], \
                                      x_start_unitn_start=[], \
                                      y_start_unitn_start=[], \
                                      x_stop_unitn_stop=[], \
                                      y_stop_unitn_stop=[], \
                                      center_rays_x=[], center_rays_y=[], \
                                      start_rays_x=[], start_rays_y=[], \
                                      stop_rays_x=[], stop_rays_y=[])


def scale_venue_figure(venue_figure, panes, ranges, axes_select):
    """
    Sets the ranges and the axis scaling of the venue slice figure
    (:attr:`pypalc.gui.app.PalcApp.venue_figure`) as selected in
    ``axes_select``.

    In every axis mode the ranges are new ``Range1d`` over the venue ranges
    (``ranges.venue_x``, ``ranges.venue_y``, the data extent with its
    margin), never widened. Axis modes (``axes_select.active``, see
    :data:`AXIS_SCALINGS` and :class:`ScaledFigure`):

    * 0, "Equal, variable height": equal scaling; the figure is as wide as
      in "Stretched" and its height follows, up to the window limit
      (:func:`window_limit`, so that it ends inside the window; then it is
      narrower).
    * 1, "Equal, fixed height": equal scaling; the figure is as high as in
      "Stretched" (:data:`VENUE_MAX_HEIGHT`, less on low windows,
      :func:`fixed_height`) and its width follows, up to the width of its
      container (then it is lower).
    * 2, "Stretched": the venue ranges fill the figure (the width of its
      container, height :data:`VENUE_MAX_HEIGHT`).

    Parameters
    ----------
    venue_figure : figure [out]
        Venue slice figure.
    panes : list of ScaledFigure [out]
        The containers that show the figure (one per page).
    ranges : PlotRanges [in]
        Contains the venue ranges.
    axes_select : panel_material_ui.RadioButtonGroup [in]
        Selection of the axis mode; ``active`` is its index.

    Returns
    -------
    None.

    """
    scaling = AXIS_SCALINGS[axes_select.active]
    set_data_ranges(venue_figure, ranges.venue_x, ranges.venue_y,
                     equal=scaling != STRETCHED)
    aspect = data_aspect(ranges.venue_x, ranges.venue_y)
    for pane in panes:
        pane.param.update(scaling=scaling, data_aspect=aspect)
