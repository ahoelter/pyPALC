"""Ranges and axis scaling of the Bokeh figures.

:class:`PlotRanges` (formerly ``Plot_ranges`` in ``fun/PALC_classes.py``)
and the helpers :func:`show_rays` and :func:`scale_venue_figure` (formerly
``venue_slice2CDS`` and ``get_p_axis_rng`` in ``fun/gui_helper.py``), the
latter with the size and the axis modes of the responsive venue figure,
and :func:`set_figure_ranges` with the equal scaling of the beam plots
(:func:`scale_equally`).

author: Arne Hoelter
"""
from __future__ import annotations

import numpy as np
from bokeh.models import ColumnDataSource, DataRange1d, Range1d

#: Height in px of the venue figure in the axis modes "Equal, fixed height"
#: and "Stretched", and its maximum height in "Equal, variable height"
#: (low enough that the figure ends inside a 1280 x 800 window).
VENUE_MAX_HEIGHT = 500
#: Space in px of the window above and below an equally scaled figure with
#: a height that follows its width (header, page tabs, the tabs of pages 2
#: and 4, card margins): its maximum height is also the window height less
#: this space (:data:`VARIABLE_MAX_HEIGHT`), so that it ends inside smaller
#: windows such as 1366 x 768 and 1280 x 720 too.
WINDOW_MARGIN = 210
#: Smallest maximum height in px of such a figure on very low windows.
MIN_MAX_HEIGHT = 200
#: CSS maximum height of such a figure: :data:`VENUE_MAX_HEIGHT`, at most
#: the window height less :data:`WINDOW_MARGIN`, at least
#: :data:`MIN_MAX_HEIGHT`.
VARIABLE_MAX_HEIGHT = 'min(%dpx, max(%dpx, calc(100vh - %dpx)))' % (
    VENUE_MAX_HEIGHT, MIN_MAX_HEIGHT, WINDOW_MARGIN)
#: Name of the hidden renderer of the venue figure whose data are the venue
#: ranges; the ranges of the equal axis modes cover it.
VENUE_BOUNDS = 'venue_bounds'


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


def add_venue_bounds(figure):
    """
    Adds the hidden renderer :data:`VENUE_BOUNDS` to a figure, whose data
    are the venue ranges that the equal axis modes cover
    (:func:`scale_equally`).

    Parameters
    ----------
    figure : figure [out]
        A figure that is shown with equal scaling.

    Returns
    -------
    GlyphRenderer
        The hidden renderer.

    """
    return figure.scatter('x', 'y',
                          source=ColumnDataSource(data=dict(x=[], y=[])),
                          name=VENUE_BOUNDS, visible=False)


def venue_bounds(figure):
    """
    The renderer :data:`VENUE_BOUNDS` of a figure (:func:`add_venue_bounds`).

    Searched among the figure's own renderers: ``select_one`` would also
    search the figures that callbacks of the figure refer to (the linked
    beam plots).

    Parameters
    ----------
    figure : figure [in]
        A figure with the renderer.

    Returns
    -------
    GlyphRenderer

    """
    (bounds,) = [renderer for renderer in figure.renderers
                 if renderer.name == VENUE_BOUNDS]
    return bounds


def scale_equally(figure, ranges, variable_height):
    """
    Shows the venue ranges in a figure with equal scaling: one meter has
    the same length on both axes, for every size of the figure.

    The figure follows the width of its container. Equal scaling is done
    by the browser for the actual size of the figure: both ranges are new
    ``DataRange1d`` over the renderer :data:`VENUE_BOUNDS` (whose data are
    set to ``ranges.venue_x``, ``ranges.venue_y``), and the figure matches
    the aspect (``match_aspect``), so Bokeh widens one range until one
    meter has the same length on both axes. The left border alternates
    between None and ``min_border`` (the same border), so that the browser
    lays out a shown figure with its new ranges.

    Parameters
    ----------
    figure : figure [out]
        A figure with the renderer :data:`VENUE_BOUNDS`
        (:func:`add_venue_bounds`).
    ranges : PlotRanges [in]
        Contains the venue ranges.
    variable_height : bool
        If True, the height follows the width with the aspect ratio of the
        venue ranges, up to :data:`VARIABLE_MAX_HEIGHT`
        (:data:`VENUE_MAX_HEIGHT`, less on low windows); at that height it
        behaves as with False. If False, the height is
        :data:`VENUE_MAX_HEIGHT`; the whole venue stays visible and one
        range widens to fill the figure.

    Returns
    -------
    None.

    """
    bounds = venue_bounds(figure)
    bounds.data_source.data = dict(x=list(ranges.venue_x),
                                   y=list(ranges.venue_y))
    sizes = dict(sizing_mode='stretch_width', height=VENUE_MAX_HEIGHT,
                 aspect_ratio=None, max_height=None, styles={},
                 frame_width=None, frame_height=None)
    if variable_height:
        width = ranges.venue_x[1] - ranges.venue_x[0]
        height = ranges.venue_y[1] - ranges.venue_y[0]
        # The CSS aspect ratio sets the height from the width. The inline
        # styles keep the full width when the maximum height is reached
        # (instead of narrowing the figure to the aspect ratio), stop
        # Panel's layout from giving the figure a fixed flex height, and
        # give the maximum height without a percentage: Bokeh writes it as
        # min(max_height, 100%), which the card ignores when it sizes
        # itself to the figure, growing to the uncapped height. The
        # maximum also depends on the window height, so that the figure
        # ends inside low windows; the browser lays the figure out again
        # when the window is resized.
        sizes.update(height=None, aspect_ratio=width / height,
                     max_height=VENUE_MAX_HEIGHT,
                     styles={'flex': '0 0 auto', 'width': '100%',
                             'max-height': VARIABLE_MAX_HEIGHT})
    figure.update(
        x_range=DataRange1d(renderers=[bounds], range_padding=0),
        y_range=DataRange1d(renderers=[bounds], range_padding=0),
        match_aspect=True, aspect_scale=1, **sizes)
    # BokehJS does not lay out a shown figure again when it gets new ranges;
    # its axes keep the size they had without ticks (the new ranges start
    # at NaN) and lose their labels, unless the size of the figure changes.
    # A changed border is laid out: the left border alternates between None
    # and min_border, which is the same border.
    figure.min_border_left = figure.min_border \
        if figure.min_border_left is None else None


def scale_venue_figure(venue_figure, ranges, axes_select):
    """
    Sets the ranges, the scaling and the size of the venue slice figure
    (:attr:`pypalc.gui.app.PalcApp.venue_figure`) as selected in
    ``axes_select``.

    The figure always follows the width of its container. The venue ranges
    (``ranges.venue_x``, ``ranges.venue_y``) are the data of the hidden
    renderer :data:`VENUE_BOUNDS` of the figure. Axis modes
    (``axes_select.active``):

    * 0, "Equal, variable height": equal scaling; the height follows the
      width with the aspect ratio of the venue ranges, up to
      :data:`VARIABLE_MAX_HEIGHT` (:data:`VENUE_MAX_HEIGHT`, less on low
      windows, so that the figure ends inside the window); at that height
      it behaves as mode 1.
    * 1, "Equal, fixed height": equal scaling at the height
      :data:`VENUE_MAX_HEIGHT`; the whole venue stays visible and one
      range widens to fill the figure.
    * 2, "Stretched": the venue ranges fill the figure (height
      :data:`VENUE_MAX_HEIGHT`).

    Equal scaling: see :func:`scale_equally`.

    Parameters
    ----------
    venue_figure : figure [out]
        Venue slice figure with the renderer :data:`VENUE_BOUNDS`.
    ranges : PlotRanges [in]
        Contains the venue ranges.
    axes_select : panel_material_ui.RadioButtonGroup [in]
        Selection of the axis mode; ``active`` is its index.

    Returns
    -------
    None.

    """
    if axes_select.active == 2:  # stretched
        bounds = venue_bounds(venue_figure)
        bounds.data_source.data = dict(x=list(ranges.venue_x),
                                       y=list(ranges.venue_y))
        venue_figure.update(
            x_range=Range1d(ranges.venue_x[0], ranges.venue_x[1]),
            y_range=Range1d(ranges.venue_y[0], ranges.venue_y[1]),
            match_aspect=False, sizing_mode='stretch_width',
            height=VENUE_MAX_HEIGHT, aspect_ratio=None, max_height=None,
            styles={}, frame_width=None, frame_height=None)
        return
    scale_equally(venue_figure, ranges,
                  variable_height=axes_select.active == 0)


def set_figure_ranges(ranges, figures, equal, x_range, y_range):
    """
    Sets the x- and y-ranges of figures (formerly ``set_ranges`` in
    ``main.py``).

    Parameters
    ----------
    ranges : PlotRanges [in]
        Contains the venue ranges.
    figures : list of figure [out]
        Figures to set the ranges of.
    equal : list of bool
        For each figure: if True, it shows the venue ranges with equal
        scaling and a height that follows its width
        (:func:`scale_equally`, the figure needs the renderer
        :data:`VENUE_BOUNDS`); each such figure gets ranges of its own.
        If False, it gets ``x_range`` and ``y_range`` (all such figures the
        same range objects).
    x_range, y_range : Range1d
        The ranges of the figures without equal scaling.

    Returns
    -------
    None.

    """
    for figure, scaled in zip(figures, equal):
        if scaled:
            scale_equally(figure, ranges, variable_height=True)
        else:
            figure.x_range = x_range
            figure.y_range = y_range
