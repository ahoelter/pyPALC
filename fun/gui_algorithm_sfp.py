#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Arne Hoelter

The module contains functions used in callbacks that belong to the PALC algorithm
computation and the sound field prediction by the CDPS-model.
"""

import numpy as np

from pypalc.core.palc import clamp_fixed_angle
from pypalc.core.sound_field import calcBeamplot


def round_fixed_angle(PALC_config, Opt_arr, gui_fixed_first_angle):
    """
    Rounds fixed tilt angle of heighest LSA cabinet to min or max value if user
    input was out of possible range. Min and max is stored in :any:`Opt_arr`.
    :any:`PALC_config` and :any:`gui_fixed_first_angle` is updated. Called by
    :any:`get_value`. Calls :any:`clamp_fixed_angle`.

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
    max_deg, min_deg = Opt_arr.fixed_angle[0], Opt_arr.fixed_angle[2]
    if gui_fixed_first_angle.disabled == False and \
       (PALC_config.fixed_angle > max_deg or PALC_config.fixed_angle < min_deg):
        PALC_config.fixed_angle = clamp_fixed_angle(PALC_config.fixed_angle,
                                                    max_deg, min_deg)
        gui_fixed_first_angle.value = str(PALC_config.fixed_angle)


def getBeamplotRes(pBeam, mapper, PALC_config, SFP_config, gamma_n, plt_ranges, f, ind_f):
    """
    Computes the beamplot of a certain frequency regarding the simulated
    venue slice. Calls :any:`calcBeamplot`. Called by :any:`get_beamplot`.

    Parameters
    ----------
    pBeam : fig [out]
        Bokeh beamplot figure.
    mapper : LinearColorMapper [out]
        Object that maps the Colors of the beamplot. Also used for the Colorbar.
    PALC_config : obj [in]
        Configuration of PALC algorithm.
    SFP_config : obj [in]
        Configuration of sound field prediction.
    gamma_n : 1D-array
        Set of LSA tilt angles in radian.
    plt_ranges : obj [out]
        Contains ranging information of the bokeh figures.
    f : float
        Considered frequency bin.
    ind_f : int
        Index of the frequency bin.

    Returns
    -------
    p_SPL : ndarray
        Sound pressure levels mapped on the mesh.

    """
    # Calculation
    p_SPL, X, Y = calcBeamplot(PALC_config, gamma_n, plt_ranges, f, \
                               np.array(SFP_config.dir_meas[:,ind_f]), \
                               np.array(SFP_config.dir_meas_deg[:,1]))
    high_ind = np.argwhere(p_SPL > 100)
    low_ind = np.argwhere(p_SPL < 50)
    for n in range(np.shape(high_ind)[0]):
        p_SPL[high_ind[n,0], high_ind[n,1]] = 100
    for n in range(np.shape(low_ind)[0]):
        p_SPL[low_ind[n,0], low_ind[n,1]] = 50
    mapper.low = np.amin(p_SPL)
    mapper.high = np.amax(p_SPL)   
    return p_SPL


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
