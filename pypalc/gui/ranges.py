"""Ranges and axis scaling of the Bokeh figures.

``Plot_ranges`` (formerly in ``fun/PALC_classes.py``) and the helpers
``venue_slice2CDS`` and ``get_p_axis_rng`` (formerly in
``fun/gui_helper.py``).

author: Arne Hoelter
"""
from __future__ import annotations

import numpy as np
from bokeh.models import Range1d


class Plot_ranges():
    """ Class to create objects that contain ranges of bokeh figures. """
    def __init__(self, p_x=None, p_y=None, bar_range=None, \
                 bartop=None, h_x=None, h_y=None, sfp_y=None, \
                 SPLoverX_y=None):
        # None stands for a new list with the default value (no shared mutable defaults)
        p_x = [-0.5,2.5] if p_x is None else p_x
        p_y = [-0.5,2.5] if p_y is None else p_y
        bar_range = [0,0] if bar_range is None else bar_range
        bartop = [0,0] if bartop is None else bartop
        h_y = [0,1] if h_y is None else h_y
        sfp_y = [0,1] if sfp_y is None else sfp_y
        SPLoverX_y = [0.3,15] if SPLoverX_y is None else SPLoverX_y
        self.p_x, self.p_y          = p_x, p_y
        self.bar_range, self.bartop = bar_range, bartop
        self.h_y, self.sfp_y        = h_y, sfp_y
        self.SPLoverX_y             = SPLoverX_y
    
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
        if plot_axis in ['bartop']:
            new_range = Range1d(-1, getattr(self, plot_axis)[1])
        else:
            new_range = Range1d(getattr(self, plot_axis)[0], getattr(self, plot_axis)[1])
        return new_range


def venue_slice2CDS(plt_venue_slice_1, plt_venue_slice_2, \
                    PALC_plots, Opt_arr):
    """
    Stores venue slice data in a ColumnDataSource to plot the data in the
    venue slice plot any:`p`.

    Parameters
    ----------
    plt_venue_slice_1 : ColumnDataSource [out]
        First CDS to visualize the rays of the LSA cabinets to the venue.
    plt_venue_slice_2 : ColumnDataSource [out]
        Second CDS to visualize the LSA cabinets.
    PALC_plots : obj [in]
        Contains the plotting data.
    Opt_arr : obj [in]
        Contains the data of the optimized LSA array by PALC.

    Returns
    -------
    None.

    """
    if Opt_arr != []:
        plt_venue_slice_1.data = dict(center_rays_x      = PALC_plots.center_rays_x, \
                                     center_rays_y       = PALC_plots.center_rays_y, \
                                     start_rays_x  = PALC_plots.start_rays_x, \
                                     start_rays_y  = PALC_plots.start_rays_y, \
                                     stop_rays_x  = PALC_plots.stop_rays_x, \
                                     stop_rays_y  = PALC_plots.stop_rays_y)
        plt_venue_slice_2.data = dict(seg_pos_x        = Opt_arr.hit_points[:,0], \
                                     seg_pos_y         = Opt_arr.hit_points[:,1], \
                                     seg_pos_start_x   = Opt_arr.hit_points_start[:,0], \
                                     seg_pos_start_y   = Opt_arr.hit_points_start[:,1], \
                                     seg_pos_stop_x    = Opt_arr.hit_points_stop[:,0], \
                                     seg_pos_stop_y    = Opt_arr.hit_points_stop[:,1])
    else:
        plt_venue_slice_2.data = dict(seg_pos_x=[], seg_pos_y=[], \
                                      seg_pos_start_x=[], seg_pos_start_y=[], \
                                      seg_pos_stop_x=[], seg_pos_stop_y=[])
        plt_venue_slice_1.data = dict(x_start_stop=[], y_start_stop=[], \
                                      x_fin_unitn_start_stop=[], \
                                      y_fin_unitn_start_stop=[], \
                                      x_start_unitn_start=[], \
                                      y_start_unitn_start=[], \
                                      x_stop_unitn_stop=[], \
                                      y_stop_unitn_stop=[], \
                                      center_rays_x=[], center_rays_y=[], \
                                      start_rays_x=[], start_rays_y=[], \
                                      stop_rays_x=[], stop_rays_y=[])


def get_p_axis_rng(p, plt_ranges, p_equal_axis):
    """
    Sets the range and the scaling of the venue slice plot :any:`p`. The
    scaling depends on the user selection in CheckBoxGroup widget
    :any:`p_equal_axis` and called by :any:`scale_p_axis`.

    Parameters
    ----------
    p : fig [out]
        Venue slice figure.
    plt_ranges : obj [in] 
        Contains the ranging data.
    p_equal_axis : obj [in]
        CheckBoxGroup widget to select scaling type of :any:`p`.

    Returns
    -------
    None.

    """
    p.frame_height=478
    if p_equal_axis.active == 2: # stretched is active
        p.x_range = Range1d(plt_ranges.p_x[0], plt_ranges.p_x[1])
        p.y_range = Range1d(plt_ranges.p_y[0], plt_ranges.p_y[1])
        p.match_aspect=False
    elif p_equal_axis.active == 1: # equal axis const. height is active
        scaling_factor = p.frame_height/p.frame_width
        axmin = min([plt_ranges.p_x[0],plt_ranges.p_y[0]])
        axmax = max([plt_ranges.p_x[1], plt_ranges.p_y[1]])
        if plt_ranges.p_x[1] >= plt_ranges.p_y[1]:
            p.x_range = Range1d(axmin, axmax)
            p.y_range = Range1d(axmin*scaling_factor, axmax*scaling_factor)
        else:
            p.y_range = Range1d(axmin, axmax)
            p.x_range = Range1d(axmin/scaling_factor, axmax/scaling_factor)
        p.match_aspect=True
    elif p_equal_axis.active == 0: # equal axis variable height is active
        p.x_range = Range1d(plt_ranges.p_x[0], plt_ranges.p_x[1])
        p.y_range = Range1d(plt_ranges.p_y[0], plt_ranges.p_y[1])        
        p.frame_height = int(p.frame_width*((plt_ranges.p_y[1]-plt_ranges.p_y[0]) / \
                                            (plt_ranges.p_x[1]-plt_ranges.p_x[0])))
        p.match_aspect=True


def set_figure_ranges(plt_ranges, figs, sizes, x_range, y_range):
    """
    Sets the x- and y-ranges of figures (formerly ``set_ranges`` in
    ``main.py``). All figures get the same range objects.

    Parameters
    ----------
    plt_ranges : Plot_ranges [in]
        Contains the ranging data; ``p_x`` and ``p_y`` give the aspect ratio.
    figs : list of figure [out]
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
    for fig, size in zip(figs, sizes):
        fig.x_range = x_range
        fig.y_range = y_range
        if size:
            fig.frame_height = int((plt_ranges.p_y[1]-plt_ranges.p_y[0]) * \
                 fig.frame_width / (plt_ranges.p_x[1]-plt_ranges.p_x[0]))
