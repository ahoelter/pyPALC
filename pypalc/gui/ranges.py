"""Ranges and axis scaling of the Bokeh figures.

:class:`PlotRanges` (formerly ``Plot_ranges`` in ``fun/PALC_classes.py``)
and the helpers :func:`show_rays` and :func:`scale_venue_figure` (formerly
``venue_slice2CDS`` and ``get_p_axis_rng`` in ``fun/gui_helper.py``).

author: Arne Hoelter
"""
from __future__ import annotations

import numpy as np
from bokeh.models import Range1d


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
    Sets the range and the scaling of the venue slice plot :any:`venue_figure`. The
    scaling depends on the user selection in CheckBoxGroup widget
    :any:`axes_select` and called by :any:`scale_p_axis`.

    Parameters
    ----------
    venue_figure : figure [out]
        Venue slice figure.
    ranges : obj [in] 
        Contains the ranging data.
    axes_select : obj [in]
        CheckBoxGroup widget to select scaling type of :any:`venue_figure`.

    Returns
    -------
    None.

    """
    venue_figure.frame_height=478
    if axes_select.active == 2: # stretched is active
        venue_figure.x_range = Range1d(ranges.venue_x[0], ranges.venue_x[1])
        venue_figure.y_range = Range1d(ranges.venue_y[0], ranges.venue_y[1])
        venue_figure.match_aspect=False
    elif axes_select.active == 1: # equal axis const. height is active
        scaling_factor = venue_figure.frame_height/venue_figure.frame_width
        axmin = min([ranges.venue_x[0],ranges.venue_y[0]])
        axmax = max([ranges.venue_x[1], ranges.venue_y[1]])
        if ranges.venue_x[1] >= ranges.venue_y[1]:
            venue_figure.x_range = Range1d(axmin, axmax)
            venue_figure.y_range = Range1d(axmin*scaling_factor, axmax*scaling_factor)
        else:
            venue_figure.y_range = Range1d(axmin, axmax)
            venue_figure.x_range = Range1d(axmin/scaling_factor, axmax/scaling_factor)
        venue_figure.match_aspect=True
    elif axes_select.active == 0: # equal axis variable height is active
        venue_figure.x_range = Range1d(ranges.venue_x[0], ranges.venue_x[1])
        venue_figure.y_range = Range1d(ranges.venue_y[0], ranges.venue_y[1])        
        venue_figure.frame_height = int(venue_figure.frame_width*((ranges.venue_y[1]-ranges.venue_y[0]) / \
                                            (ranges.venue_x[1]-ranges.venue_x[0])))
        venue_figure.match_aspect=True


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
