#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Arne Hoelter

Module to support GUI interaction on the pages of loudspeaker and array
configuration.
"""

import numpy as np
import base64

from PALC_functions import calc_progressive_array, calc_arc_array, repmat
from sfp_functions  import get_freq_vec

def ref_array_angles(PALC_config, Ref_arr, gui_ref_array, \
                     gui_ref_start, gui_ref_step_stop, \
                     gui_ref_discrete_angles, gui_ref_userdef):
    """
    Calculates the reference LSA tilt angles depending on user input. Called
    by :any:`get_ref_array_angles` and :any:`get_value`. Depending on the
    array type, the function calls :any:`calc_progressive_array` or 
    :any:`calc_arc_array`.

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
    if gui_ref_array.value in ['Straight']:
        Ref_arr.gamma_tilt_deg = np.ones(PALC_config.N) * float(gui_ref_start.value) 
    elif gui_ref_array.value in ['Progressive']:
        Ref_arr.gamma_tilt_deg = calc_progressive_array(PALC_config.N, PALC_config.gamma_LSA, \
                                                        float(gui_ref_start.value), \
                                                        float(gui_ref_step_stop.value), \
                                                        str(gui_ref_discrete_angles.value))
    elif gui_ref_array.value in ['Arc']:
        Ref_arr.gamma_tilt_deg = calc_arc_array(PALC_config.N, PALC_config.gamma_LSA, \
                                                float(gui_ref_start.value), \
                                                float(gui_ref_step_stop.value), \
                                                str(gui_ref_discrete_angles.value))
    elif gui_ref_array.value in ['User Defined']:
        # split up the input tilt angles of the TextInput widget
        Ref_arr.gamma_tilt_deg = np.array([float(s) for s in gui_ref_userdef.value.split(',')])
        # check if too less or many tilt angles are given by the user
        diff2N = np.shape(Ref_arr.gamma_tilt_deg)[0] - PALC_config.N
        if diff2N < 0:
            Ref_arr.gamma_tilt_deg = np.append(Ref_arr.gamma_tilt_deg, \
                                                np.ones(np.abs(diff2N))*Ref_arr.gamma_tilt_deg[-1])
        elif diff2N > 0:
            Ref_arr.gamma_tilt_deg = Ref_arr.gamma_tilt_deg[:PALC_config.N]
            
            
def read_dir_data(SFP_config, new):
    """
    Reads (measured, complex) directivity data from an .csv-file. Called by
    :any:`upload_directivity`.

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
    # get data data frame
    new_df = base64.b64decode(new).decode('utf-8')
    counter = sum('\n' in s for s in new_df)+1
    new_df = str(np.char.replace(np.array(new_df).astype(str),'\n',','))
    new_df = new_df.split(',')
    directivity = np.char.replace(np.array(new_df),'i','j').astype(np.complex).reshape([int(counter),int(np.shape(new_df)[0]/counter)])

    # get an array of corresponding degree and frequency and delete these "header" and "index" from the directivity array
    # and write it once for whole data in dictionary and second just for the frequencies to be plotted in another dictionary
    # initialize the considered frequency bins
    SFP_config.f = get_freq_vec(N_freq=120, step_freq=1/12, freq_range=[20,20000])          
    # cut the degree and frequency vector out of the directivity array
    degree = np.real(directivity[1:,0]).astype(float)
    SFP_config.plot_beta_meas = list(degree)      
    frequency = np.real(directivity[0,1:]).astype(float)
    directivity = np.delete(np.delete(directivity,0,axis=0),0,axis=1)

    # interpolate directivity to the frequency bins used in bokeh app 
    directivity_f = np.array([])
    for n in range(np.shape(directivity)[0]):
        directivity_f = np.append(directivity_f, np.interp(SFP_config.f,frequency,directivity[n,:]))
    directivity_f = np.reshape(directivity_f,(np.shape(degree)[0],np.shape(SFP_config.f)[0]))    
    
    # get the index of the three frequencies to be plotted
    SFP_config.get_plot_dir_amp(directivity_f, [100, 1000, 10000], meas=True) 
    
    # get frequency and degree in shape to upload in ColumnDataSource
    #f = repmat(SFP_config.f, np.shape(directivity_f)[0], 1)
    degree = repmat(degree, np.shape(directivity_f)[1]-1, 1).transpose()
    
    # amplitude and phase calculation of directivity
    amplitude = 20*np.log10(np.abs(directivity_f))
    #phase = np.angle(directivity_f)

    # write data in ColumnDataSource
    for key, val in [['dir_meas', directivity_f], ['dir_meas_deg', degree], ['dir_meas_amp', amplitude]]:
        setattr(SFP_config, key, val)
            

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