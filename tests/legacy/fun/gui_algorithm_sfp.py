#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Arne Hoelter

The module contains functions used in callbacks that belong to the PALC algorithm
computation and the sound field prediction by the CDPS-model.
"""

import numpy as np

import calcPALC
import PALC_classes
import PALC_opt as opt
from PALC_functions import LSA_visualization
from sfp_functions import calcSFP, calcSPLoverX


def optimize_PALC(PALC_config, SFP_config, PALC_pal, PALC_plots, \
                  Opt_arr, created_pal, Tech_res, SPLoverX, Opt_w, \
                  max_loops=80):
    """
    Runs the while loop of target slope optimization. Will return the optimized
    weighting factors in :any:`Opt_w`. Called by :any:`start_calc`. Calls mainly
    methods of :py:class:`PALC_classes.Opt_weight`, :py:mod:`PALC_opt` and
    :py:mod:`sfp_functions`.

    Parameters
    ----------
    PALC_config : obj [in]
        Configuration of the PALC algorithm.
    SFP_config : obj [in]
        Configuration of the sound field prediction.
    PALC_pal : obj [in]
        Data of the venue slice.
    PALC_plots : obj [in, out]
        Data to plot the venue slice information.
    Opt_arr : obj [out]
        Data of the optimized LSA.
    created_pal : obj [in]
        Venue slice that was drawn by the user.
    Tech_res : obj [out]
        Technical Measure results depending on the sound field prediction.
    SPLoverX : obj [out]
        Data of the Sound Pressure Level over Distance.
    Opt_w : obj [out]
        Optimized weighting data to run the target slope optimization.
    max_loops : int, optional
        Maximum of loops to compute. The default is 40.

    Returns
    -------
    None.

    """
    loop = 0
    while int(loop) < int(max_loops):
        calcPALC.calcPALC(PALC_plots, PALC_pal, PALC_config, Opt_arr)
        x_patches, y_patches = LSA_visualization(PALC_plots, PALC_config, Opt_arr.gamma_n)
        PALC_plots.get_plot_array(Opt_arr)
        x_S, y_S = calcSFP(Opt_arr.gamma_tilt_deg, created_pal, SFP_config, \
                            Tech_res)#, dir_meas_LSA, dir_meas_degree)
        calcSPLoverX(PALC_plots, created_pal, Tech_res.p_SPL, SPLoverX, SFP_config.freq_range)
        
        # get difference between reference and PALC at optimization points
        # indices of optimization region
        opt_ind = opt.get_opt_region(SPLoverX, Opt_w)
        
        # shift Opt_w SPL at x_ref on SPLoverX
        opt.shift2ref(Opt_w, SPLoverX, opt_ind, [])
        # indices that belong to the loudspeakers
        ls_ind, ls_opt_ind = opt.ls2pal(PALC_config.N, Opt_arr, created_pal, opt_ind=opt_ind)
        # map values on LS
        Opt_w.diffLS, Opt_w.diffgradLS = opt.diff_on_ls(PALC_config.N, opt_ind, \
                                                        ls_ind, ls_opt_ind, \
                                                        Opt_w, SPLoverX)
        Opt_w.diffLS = opt.smooth_1D_list(Opt_w.diffLS)
        # opt_ind: pal points that belong to opt region
        # diff_opt: difference in opt_region
        # ls_opt_ind: opt region pal points sorted to loudspeaker
        # ls_ind: pal points sorted to loudspeaker 
        # removed opt_ind behind, change mean of SPL_interp
        svdiff, svgrad = opt.calc_SingleValueDiff(np.array([SPLoverX.SPL_interp, \
                                                            SPLoverX.SPL_grad]), \
                                                  np.array([Opt_w.SPL_interp, \
                                                            Opt_w.SPL_grad]), \
                                                  mtype='quantiles', ef=True)
        svtot = svdiff * svgrad
        out = Opt_w.shift2psval(PALC_config, svtot, Opt_arr.num_iter)
        if out:
            calcPALC.calcPALC(PALC_plots, PALC_pal, PALC_config, Opt_arr)
            break
        # update weighting
        opt.opt_weighting(PALC_config, Opt_w, loop)
        loop += 1
    return opt_ind
        
 
def get_fixed_angle_borders(last_pal_sec, PALC_config):
    """
    Calculates the average / mid and maximum tilt angle for the heighest LSA cabinet.
    When functionality of fixing the heighest LSA cabinet on a specific tilt
    angle, the borders of the maximum and minimum tilt angle can thus be found.
    Called by :any:`use_fixed_angle`.

    Parameters
    ----------
    last_pal_sec : list
        Last PAL section, i.e., PALC_pal.pal[-1].
    PALC_config : obj [in]
        Configuration of the PALC algorithm.

    Returns
    -------
    gamma_tilt_deg : list
        Contains average (first entry) and maximum (seconed entry) tilt angle
        of the heighest LSA cabinet.

    """
    gamma_tilt_deg = []
    for n in [0,-1]:
        dist_hy = np.sqrt((last_pal_sec[n,1]-(PALC_config.y_H-PALC_config.Lambda_y))**2 + \
                          (last_pal_sec[n,0]-PALC_config.x_H)**2)
        dist_an = np.abs(last_pal_sec[n,0]-PALC_config.x_H)
        if PALC_config.y_H < last_pal_sec[n,1]:
            gamma_tilt_deg.append(np.round(-np.arccos(dist_an /dist_hy) *(180/np.pi), decimals=2))
        else:
            gamma_tilt_deg.append(np.round(np.arccos(dist_an /dist_hy) *(180/np.pi), decimals=2))
    return gamma_tilt_deg

        
def round_fixed_angle(PALC_config, Opt_arr, gui_fixed_first_angle):
    """
    Rounds fixed tilt angle of heighest LSA cabinet to min or max value if user
    input was out of possible range. Min and max is stored in :any:`Opt_arr`.
    :any:`PALC_config` and :any:`gui_fixed_first_angle` is updated. Called by
    :any:`get_value`.

    Parameters
    ----------
    PALC_config : obj [in, out]
        PALC configuration object.
    Opt_arr : obj [in]
        Optimized array data object.
    gui_fixed_first_angle : obj [out]
        TextInput widget that handles the fixed tilt angle of the heighest LSA cabinet.

    Returns
    -------
    None.

    """
    if PALC_config.fixed_angle > Opt_arr.fixed_angle[0] and \
        gui_fixed_first_angle.disabled == False:
        PALC_config.fixed_angle = np.round(Opt_arr.fixed_angle[0], decimals=2)
        gui_fixed_first_angle.value = str(np.round(Opt_arr.fixed_angle[0], \
                                                   decimals=2))
    elif PALC_config.fixed_angle < Opt_arr.fixed_angle[2] and \
        gui_fixed_first_angle.disabled == False:
        PALC_config.fixed_angle = np.round(Opt_arr.fixed_angle[2], decimals=2)
        gui_fixed_first_angle.value = str(np.round(Opt_arr.fixed_angle[2], \
                                                   decimals=2))
            

def set_beamplot_visibility(gui_which_beam, pBeam, pBeam_ref):
    """
    Sets beamplot figures to visible or invisible depending on user input in
    :any:`gui_which_beam` widget. Called by :any:`get_beamplot`.

    Parameters
    ----------
    gui_which_beam : obj [in]
        CheckBoxGroup widget that handles which beamplot is shown.
    pBeam : fig [out]
        Beamplot figure of PALC results.
    pBeam_ref : fig [out]
        Beamplot figure of reference array.

    Returns
    -------
    None.

    """
    if np.any(np.array(gui_which_beam.active) == 0):
        pBeam.visible = True
    else:
        pBeam.visible = False
    if np.any(np.array(gui_which_beam.active) == 1):
        pBeam_ref.visible = True
    else:
        pBeam_ref.visible = False

       
def choose_gap_handling(gui_gap_handling, gui_strength_smarg):
    """
    Enables the Slider widget :any:`gui_strength_smarg` if soft margin is
    chosen as the gap handling approach. Otherwise disables the slider.
    Called by :any:`get_value`

    Parameters
    ----------
    gui_gap_handling : obj [in]
        Select widget of gap handling approach.
    gui_strength_smarg : obj [out]
        Slider widget of strength of soft margin gap handling approach.

    Returns
    -------
    None.

    """
    if gui_gap_handling.value == "Without":
        gui_strength_smarg.disabled = True
        gui_strength_smarg.value = 1.0
    elif gui_gap_handling.value == "Hard Margin":
        gui_strength_smarg.disabled = True
        gui_strength_smarg.value = 1.0
    elif gui_gap_handling.value == "Soft Margin":
        gui_strength_smarg.disabled = False
        
       
def update_gh_w_select(gui_weighting, gui_gap_handling):
    """
    Ensures that hard margin gap handling approach and target slope weighting
    optimization cannot be enabled at same time. Called by :any:`get_value` and
    :any:`weighting_select`.

    Parameters
    ----------
    gui_weighting : obj [in, out]
        Select widget of weighting approach.
    gui_gap_handling : obj [in, out]
        Select widget of gap handling approach.

    Returns
    -------
    None.

    """
    if gui_weighting.value == 'Target Slope':
        gui_gap_handling.options = ['Without', 'Soft Margin']
    elif gui_gap_handling.value == 'Hard Margin':
        gui_weighting.options = ['Without', 'Linear Spacing', \
                                  'Logarithmic Spacing']
    else:
        gui_gap_handling.options = ['Without', 'Hard Margin', 'Soft Margin']
        gui_weighting.options = ['Without', 'Linear Spacing', \
                                 'Logarithmic Spacing', 'Target Slope']
            
## Function to show weighting possibilities
def set_weighting_in(gui_weighting, weighting_plus, weighting_minus, \
                     weighting_step_size, gui_weighting_nu):
    """
    If linear or logarithmic spaced weighting is chosen in :any:`gui_weighting`,
    enables the weighting Buttons and Sliders :any:`weighting_plus`,
    :any:`weighting_minus`, :any:`weighting_step_size` and :any:`gui_weighting_nu`.
    Triggered by :any:`weighting_select`.

    Parameters
    ----------
    gui_weighting : obj [in]
        Widget to select weighting approach.
    weighting_plus : obj [out]
        Widget to increase weighting.
    weighting_minus : obj [out]
        Widget to decrease weighting.
    weighting_step_size : obj [out]
        Widget to adjust weighting step size.DESCRIPTION
    gui_weighting_nu : obj [out]
        Widget to define nu for weighting strength.

    Returns
    -------
    None.

    """
    if gui_weighting.value in ['Without', 'Target Slope']:
        weighting_plus.disabled      = True
        weighting_minus.disabled     = True
        weighting_step_size.disabled = True
        gui_weighting_nu.disabled    = True
        gui_weighting_nu.value       = 1.00
    elif gui_weighting.value in ['Linear Spacing', 'Logarithmic Spacing']:
        weighting_plus.disabled      = False
        weighting_minus.disabled     = False
        weighting_step_size.disabled = False
        gui_weighting_nu.disabled    = False