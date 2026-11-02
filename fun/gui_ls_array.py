#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Arne Hoelter

Module to support GUI interaction on the pages of loudspeaker and array
configuration.
"""

import base64

from pypalc.core.geometry import reference_tilt_angles_deg
from pypalc.core.sound_field import parse_measured_directivity

# labels of gui_ref_array -> reference array types of pypalc.core
_REF_TYPES = {'Straight': 'straight', 'Progressive': 'progressive',
              'Arc': 'arc', 'User Defined': 'user_defined'}


def ref_array_angles(PALC_config, Ref_arr, gui_ref_array, \
                     gui_ref_start, gui_ref_step_stop, \
                     gui_ref_discrete_angles, gui_ref_userdef):
    """
    Calculates the reference LSA tilt angles depending on user input. Called
    by :any:`get_ref_array_angles` and :any:`get_value`. Reads the widgets
    and calls :any:`reference_tilt_angles_deg`.

    Parameters
    ----------
    PALC_config : obj [in]
        Configuration of the PALC algorithm.
    Ref_arr : obj [out]
        Contains information of the reference array to use in SFP.
    gui_ref_array : obj [in]
        Select widget that handles the type of the reference array.
    gui_ref_start : obj [in]
        TextInput widget that contains the angle of the highest LSA cabinet in
        degree.
    gui_ref_step_stop : obj [in]
        TextInput widget that contains the intercabinet angle or the angle of
        the last LSA cabinet in degree.
    gui_ref_discrete_angles : obj [in]
        Select widget if discrete tilt angles shall be used.
    gui_ref_userdef : obj [in]
        TextAreaInput widget with user defined LSA tilt angles in degree.

    Returns
    -------
    None.

    """
    ref_type = _REF_TYPES[gui_ref_array.value]
    if ref_type == 'user_defined':
        # split up the input tilt angles of the TextInput widget
        user_angles = [float(s) for s in gui_ref_userdef.value.split(',')]
        start, step_stop = 0, 0
    else:
        user_angles = ()
        start = float(gui_ref_start.value)
        step_stop = float(gui_ref_step_stop.value) \
            if ref_type != 'straight' else 0
    Ref_arr.gamma_tilt_deg = reference_tilt_angles_deg(
        ref_type, PALC_config.N, PALC_config.gamma_LSA, start, step_stop,
        str(gui_ref_discrete_angles.value) == "Yes", user_angles)


def read_dir_data(SFP_config, new):
    """
    Reads (measured, complex) directivity data from an uploaded .csv-file.
    Called by :any:`upload_directivity`. Decodes the upload and calls
    :any:`parse_measured_directivity`.

    Parameters
    ----------
    SFP_config : obj [out]
        Sound field prediction configuration data.
    new : str
        csv-data to read. Must be decoded by base64.

    Returns
    -------
    None.

    """
    parse_measured_directivity(base64.b64decode(new).decode('utf-8'),
                               SFP_config)


def arrange_ref_arr_in(gui_ref_array, gui_ref_start, gui_ref_step_stop, \
                       gui_ref_discrete_angles, gui_ref_userdef):
    """
    Arrange / Rearrange the visibility or disability of the widgets that
    belong to the configuration of the reference LSA. Called by
    :any:`reference_array_setting`.

    Parameters
    ----------
    gui_ref_array : obj [in, out]
        Select widget that handles the type of the reference array.
    gui_ref_start : obj [in, out]
        TextInput widget that contains the angle of the highest LSA cabinet in
        degree.
    gui_ref_step_stop : obj [in, out]
        TextInput widget that contains the intercabinet angle or the angle of
        the last LSA cabinet in degree.
    gui_ref_discrete_angles : obj [in, out]
        Select widget if discrete tilt angles shall be used.
    gui_ref_userdef : obj [in, out]
        TextAreaInput widget with user defined LSA tilt angles in degree.

    Returns
    -------
    None.

    """
    if gui_ref_array.value in ['Straight', 'Progressive', 'Arc']:
        gui_ref_step_stop.visible, gui_ref_start.visible = True, True
        gui_ref_userdef.visible = False
        if gui_ref_array.value in ['Straight']:
            gui_ref_step_stop.value, gui_ref_step_stop.disabled = "0", True
            gui_ref_start.title= "Reference Array: Array Tilt Angle in deg"
            gui_ref_discrete_angles.visible = False
            gui_ref_step_stop.title = "Reference Array: Inter Cabinet Angle in deg"
        elif gui_ref_array.value in ['Progressive']:
            gui_ref_step_stop.title = "Reference Array: Final Angle in deg"
            gui_ref_start.title= "Reference Array: Start Angle in deg"
            gui_ref_step_stop.disabled, gui_ref_discrete_angles.visible = False, True
        elif gui_ref_array.value in ['Arc']:
            gui_ref_step_stop.title = "Reference Array: Inter Cabinet Angle in deg"
            gui_ref_start.title= "Reference Array: Start Angle in deg"
            gui_ref_step_stop.disabled, gui_ref_discrete_angles.visible = False, True
    elif gui_ref_array.value in ['User Defined']:
        gui_ref_step_stop.visible, gui_ref_start.visible = False, False
        gui_ref_discrete_angles.visible, gui_ref_userdef.visible = False, True
