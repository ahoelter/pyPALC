#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Arne Hoelter

Functions that help to organize the :doc:`main` and display in the GUI. The
functions do not directly correspond to a specific page in the GUI. They
are typically called by several functions / callbacks of the :doc:`main` script.
"""

from bokeh.models import Range1d

def update_text(text_obj, text):
    """
    Set text in str-format to an object. The text will be displayed in the GUI.
    The text is usually an extra info or an error message, raised by an except
    case.

    Parameters
    ----------
    text_obj : obj [out]
        Object that contains text to display in the GUI.
    text : str
        Text to display in the GUI.

    Returns
    -------
    None.

    """
    setattr(text_obj,'text', text)

     
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
        plt_venue_slice_1.data = dict(x_c_n_unitn      = PALC_plots.x_c_n_unitn, \
                                     y_c_n_unitn       = PALC_plots.y_c_n_unitn, \
                                     x_c_n_unitn_psi1  = PALC_plots.x_c_n_unitn_psi1, \
                                     y_c_n_unitn_psi1  = PALC_plots.y_c_n_unitn_psi1, \
                                     x_c_n_unitn_psi2  = PALC_plots.x_c_n_unitn_psi2, \
                                     y_c_n_unitn_psi2  = PALC_plots.y_c_n_unitn_psi2)
        plt_venue_slice_2.data = dict(seg_pos_x        = Opt_arr.seg_pos[:,0], \
                                     seg_pos_y         = Opt_arr.seg_pos[:,1], \
                                     seg_pos_start_x   = Opt_arr.seg_pos_start[:,0], \
                                     seg_pos_start_y   = Opt_arr.seg_pos_start[:,1], \
                                     seg_pos_stop_x    = Opt_arr.seg_pos_stop[:,0], \
                                     seg_pos_stop_y    = Opt_arr.seg_pos_stop[:,1])
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
                                      x_c_n_unitn=[], y_c_n_unitn=[], \
                                      x_c_n_unitn_psi1=[], y_c_n_unitn_psi1=[], \
                                      x_c_n_unitn_psi2=[], y_c_n_unitn_psi2=[])
            

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
        
