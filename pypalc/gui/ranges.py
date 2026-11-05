"""Ranges and axis scaling of the Bokeh figures.

:class:`PlotRanges` (formerly ``Plot_ranges`` in ``fun/PALC_classes.py``)
and the helpers :func:`show_rays` and :func:`scale_venue_figure` (formerly
``venue_slice2CDS`` and ``get_p_axis_rng`` in ``fun/gui_helper.py``), the
latter with the size and the axis modes of the responsive venue figure.

author: Arne Hoelter
"""
from __future__ import annotations

import numpy as np
from bokeh.models import DataRange1d, Range1d

#: Height in px of the venue figure in the axis modes "Equal, fixed height"
#: and "Stretched", and its maximum height in "Equal, variable height"
#: (low enough that the figure ends inside a 1280 x 800 window).
VENUE_MAX_HEIGHT = 500
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
      :data:`VENUE_MAX_HEIGHT`; at that height the x range follows the
      width (as in mode 1).
    * 1, "Equal, fixed height": equal scaling at the height
      :data:`VENUE_MAX_HEIGHT`; the x range follows the width.
    * 2, "Stretched": the venue ranges fill the figure (height
      :data:`VENUE_MAX_HEIGHT`).

    Equal scaling is done by the browser for the actual size of the figure:
    both ranges are ``DataRange1d`` over the bounds renderer, and the
    figure matches the aspect (``match_aspect``), so Bokeh widens one range
    until one meter has the same length on both axes.

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
    bounds = venue_figure.select_one({'name': VENUE_BOUNDS})
    bounds.data_source.data = dict(x=list(ranges.venue_x),
                                   y=list(ranges.venue_y))
    sizes = dict(sizing_mode='stretch_width', height=VENUE_MAX_HEIGHT,
                 aspect_ratio=None, max_height=None, styles={},
                 frame_width=None, frame_height=None)
    if axes_select.active == 2:  # stretched
        venue_figure.update(
            x_range=Range1d(ranges.venue_x[0], ranges.venue_x[1]),
            y_range=Range1d(ranges.venue_y[0], ranges.venue_y[1]),
            match_aspect=False, **sizes)
        return
    if axes_select.active == 0:  # equal, variable height
        width = ranges.venue_x[1] - ranges.venue_x[0]
        height = ranges.venue_y[1] - ranges.venue_y[0]
        # The CSS aspect ratio sets the height from the width. The inline
        # styles keep the full width when the maximum height is reached
        # (instead of narrowing the figure to the aspect ratio), stop
        # Panel's layout from giving the figure a fixed flex height, and
        # give the maximum height in px: Bokeh writes it as
        # min(max_height, 100%), which the card ignores when it sizes
        # itself to the figure, growing to the uncapped height.
        sizes.update(height=None, aspect_ratio=width / height,
                     max_height=VENUE_MAX_HEIGHT,
                     styles={'flex': '0 0 auto', 'width': '100%',
                             'max-height': '%dpx' % VENUE_MAX_HEIGHT})
    venue_figure.update(
        x_range=DataRange1d(renderers=[bounds], range_padding=0),
        y_range=DataRange1d(renderers=[bounds], range_padding=0),
        match_aspect=True, aspect_scale=1, **sizes)


def set_figure_ranges(ranges, figures, sizes, x_range, y_range):
    """
    Sets the x- and y-ranges of figures (formerly ``set_ranges`` in
    ``main.py``). All figures get the same range objects.

    Parameters
    ----------
    ranges : PlotRanges [in]
        Contains the ranging data; ``venue_x`` and ``venue_y`` give the aspect ratio.
    figures : list of figure [out]
        Figures to set the ranges of.
    sizes : list of bool
        For each figure: if True, the frame height is set so that the axes
        are equally scaled.
    x_range, y_range : Range1d
        The new ranges.

    Returns
    -------
    None.

    """
    for figure, size in zip(figures, sizes):
        figure.x_range = x_range
        figure.y_range = y_range
        if size:
            figure.frame_height = int((ranges.venue_y[1]-ranges.venue_y[0]) * \
                 figure.frame_width / (ranges.venue_x[1]-ranges.venue_x[0]))
