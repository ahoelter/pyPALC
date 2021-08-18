#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Arne Hoelter, Audio Communication Group, TU Berlin

The main script of the pyPALC application. This is a **Loudspeaker Array Optimization
Software** to find the tilt angles of the *Loudspeaker Cabinets*.
The application works with a bokeh server that can be called
by typing "bokeh serve --show pyPALC" in the terminal on the directory that
contains the "pyPALC" folder.

The script is structured as follows:
    1. Import all necessary modules
    2. Definition of objects for calculation and ColumnDataSources for visualization
    3. Definition of callbacks
    4. Definition of widgets
    5. Connection of widgets with callbacks
    6. Setting up all figures, tables and tabs
    7. Defining the start page (venue creation)
"""
######################## IMPORT PACKAGES ######################################
import sys
import os
abs_path = os.path.dirname(os.path.abspath(__file__))
appname = os.path.basename(abs_path)
sys.path.append(f'{abs_path}/fun/') # add functions to path

# import PALC functions
import calcPALC
import checkPAL
import PALC_classes
from PALC_functions import calc_diff_tilt_angles, LSA_visualization
from PALC_opt import get_opt_region, get_weight_links, shift2ref, shift_ref_on_zero
from sfp_functions import calcSFP, calcHomogeneity, calc_directivity 
from sfp_functions import init_dir_plot, calcSPLoverX, calcHistogram, getBeamplotRes
from gui_help_text import help_text
from gui_helper import update_text, venue_slice2CDS, get_p_axis_rng
from gui_ls_array import ref_array_angles, read_dir_data, arrange_ref_arr_in
from gui_algorithm_sfp import optimize_PALC, round_fixed_angle
from gui_algorithm_sfp import get_fixed_angle_borders, set_beamplot_visibility
from gui_algorithm_sfp import update_gh_w_select, set_weighting_in, choose_gap_handling
from gui_venue import check_all_draw_conditions

# Import python
import numpy as np
from colorcet import rainbow
import time

# Import bokeh
from bokeh.io import curdoc
from bokeh.layouts import column, row, gridplot
from bokeh.models import ColumnDataSource, Range1d, CustomJS, ColorBar
from bokeh.models import LinearColorMapper, FuncTickFormatter
from bokeh.models.widgets import Slider, TextInput, Button, Toggle, FileInput
from bokeh.models.widgets import RadioButtonGroup, Select, Paragraph, DataTable
from bokeh.models.widgets import TableColumn, Div, Panel, Tabs, TextAreaInput
from bokeh.models.widgets import RadioGroup, CheckboxGroup, RangeSlider, Paragraph
from bokeh.models.tools import PointDrawTool
from bokeh.plotting import figure
from bokeh.transform import linear_cmap
from bokeh.palettes import Cividis256


#%%########## Initialize all necessary objects ###############################
created_pal    = PALC_classes.PAL(pal_index = np.array([0]).astype(int))
"""Audience lines drawn by user"""
PALC_pal       = PALC_classes.PALC_compatible_PAL()
"""PALC compatible venue slice (with audience lines)"""
PALC_config    = PALC_classes.PALC_configuration()
"""PALC configuration"""
SFP_config     = PALC_classes.SFP_configuration()
"""Sound Field Prediction configuration (CDPS-Model)"""
PALC_plots     = PALC_classes.PALC_plotting()
"""Plot data of PALC results"""
PALC_plots_ref = PALC_classes.PALC_plotting()
"""Plot data of the reference array"""
plt_ranges     = PALC_classes.Plot_ranges()
"""Ranges of the plots"""
Opt_arr        = PALC_classes.Array_opt()
"""Data of the optimized array"""
Ref_arr        = PALC_classes.Array_opt(gamma_tilt_deg=np.zeros(4))
"""Data of the reference array"""
Tech_res       = PALC_classes.Technical_meas_results()
"""Results of the technical measures of the PALC optimized array"""
Tech_res_ref   = PALC_classes.Technical_meas_results()
"""Results of the technical measures of the reference array"""
Opt_w          = PALC_classes.Opt_weight()
"""Data needed for the target slope optimization of PALC (SPL over distance)"""
SPLoverX       = PALC_classes.Opt_weight(SPL_interp = [0])
"""Result data of SPL over distance computation"""
SPLoverX_ref   = PALC_classes.Opt_weight(SPL_interp = [0])
"""Result data of SPL over distance computation of reference LSA"""
############ Set up dictionaries for user interaction with the GUI ############
x_init = np.array([0, 0]).astype(float)
chg_N = np.array([0])
# Variables for Plotting
guipal     = ColumnDataSource(data=dict(x=x_init, y=x_init))
"""Contains data points that are given by user. Visualized in venue slice plot."""
guilineplt = ColumnDataSource(data=dict(xlineplt=[], ylineplt=[]))
"""Contains the drawn audience lines based on user input. After compatibilization
   to the PALC algorithm, shows the lines used for PALC and SFP."""
guinalplt  = ColumnDataSource(data=dict(xnalplt=[], ynalplt=[]))
"""Contains non audience lines, e.g., gaps between audience line."""
guisfpplt  = ColumnDataSource(data=dict(xsfpplt=[], ysfpplt=[]))
"""Contains audience lines that are not used for PALC calcution but for SFP."""
plt_venue_slice_2 = ColumnDataSource(data=dict(seg_pos_x=[], seg_pos_y=[], \
                                               seg_pos_start_x=[], seg_pos_start_y=[], \
                                               seg_pos_stop_x=[], seg_pos_stop_y=[]))
"""Contains the rays from LSA cabinets to the venue slice in venue slice plot."""
plt_venue_slice_1 = ColumnDataSource(data=dict(x_start_stop=[], y_start_stop=[], \
                                               x_fin_unitn_start_stop=[], \
                                               y_fin_unitn_start_stop=[], \
                                               x_start_unitn_start=[], \
                                               y_start_unitn_start=[], \
                                               x_stop_unitn_stop=[], \
                                               y_stop_unitn_stop=[], \
                                               x_c_n_unitn=[], y_c_n_unitn=[], \
                                               x_c_n_unitn_psi1=[], \
                                               y_c_n_unitn_psi1=[], \
                                               x_c_n_unitn_psi2=[], \
                                               y_c_n_unitn_psi2=[]))
"""Contains data to plot the LSA cabinets in venue slice plot."""
results_dict = ColumnDataSource(data=dict(a_num_LS=np.linspace(1,PALC_config.N,num=0), \
                                          b_gamma_tilt_deg=np.array([]), \
                                          c_gamma_tilt_deg_diff=np.array([])))
"""Contains the tilt angles of the LSA cabinet computed by PALC."""
gamma_ref_dict = ColumnDataSource(data=dict(gamma_ref=np.array([np.zeros(PALC_config.N)]), \
                                            LS=np.array([np.linspace(1,4,num=PALC_config.N)])))
"""Contains the tilt angles of the reference LSA."""
# Set up dicts for plotting
p_SPL    = list(zip([0,0]))
x_vert   = list(zip([0,0]))
SFP_dict = ColumnDataSource(data=dict(p_SPL=p_SPL, f=p_SPL, \
                                      foo=list(range(np.shape(p_SPL)[0])), \
                                      p_SPL_ref=p_SPL))
"""Contains the sound pressure levels of PALC and reference computed by the CDPS model."""
SFPv_dict = ColumnDataSource(data=dict(x_vert=x_vert, y_vert=x_vert, \
                                       foo=list(range(np.shape(x_vert)[0]))))
"""Contains the discrete venue slice points needed for visualization of :any:`SFP_dict`."""    
Homogeneity_dict = ColumnDataSource(data=dict(H=x_vert, f=x_vert, \
                                              H_dist_high=x_vert, \
                                              H_dist_low=x_vert, \
                                              H_str=x_vert))
"""Contains the computed homogeneity."""
Bar_dict     = ColumnDataSource(data=dict(top=x_vert, x=x_vert))
"""Contains the values of the bar plot of the PALC results."""
Bar_dict_ref = ColumnDataSource(data=dict(top_ref=x_vert, x_ref=x_vert))
"""Contains the values of the bar plot of the reference results."""
directivity_dict_plt = ColumnDataSource(data=dict(amplitude_100=p_SPL, \
                                                  amplitude_1k=p_SPL, \
                                                  amplitude_10k=p_SPL, \
                                                  degree=p_SPL))
"""Contains the directivity of 100 Hz, 1 kHz and 10 kHz used in SFP (CDPS-model).
   Plotted on the second page."""
SPLoverX_dict = ColumnDataSource(data=dict(SPL=p_SPL, x=x_vert, x_v=x_vert, \
                                           y_v=x_vert))
"""Contains the SPL values over distance between LSA reference point and the
   receiver positions normalized on 0 dB (if target slope, optimization region is used)."""
SPLoverX_ref_dict = ColumnDataSource(data=dict(SPL=p_SPL, x=x_vert, x_v=x_vert, \
                                           y_v=x_vert))
"""Contains the SPL values over distance between LSA reference point and the
   receiver positions normalized on 0 dB of reference LSA."""    
Opt_SPLoverX_dict = ColumnDataSource(data=dict(SPL=p_SPL, x=x_vert, x_v=[0,0], \
                                               y_v=[0,0]))
Opt_refpoint_dict = ColumnDataSource(data=dict(x_ref=[10],SPL=[0],x_v=[0],y_v=[0]))
"""Contains SPL and distance values on the hinges of the target slope."""
SPLoverX_optreg = ColumnDataSource(data=dict(x=[0,0,0,0], y=[0,0,0,0]))
"""Contains edges of the drawn polygon which indicates the optimization region"""
Beamplot_dict = ColumnDataSource(data=dict(SPL=[np.zeros([100,100])], \
                                           SPL_ref=[np.zeros([100,100])], \
                                           x=[-2], y=[-2], dw=[4], dh=[4]))
"""Contains the SPL, start points, height and widtch of the beamplot (PALC and reference)."""
plot_patches = ColumnDataSource(data=dict(x_list=[], y_list=[], \
                                          x_list_ref=[], y_list_ref=[]))
"""Contains the values to plot the LSA cabinets with bokeh's patch function."""
# dict to store tapped audience line to insert previous line or delete selected line
audience_line_tap_dict = ColumnDataSource(data=dict(tapped=[]))
"""Contains the tapped line of the venue slice plot."""

# initialize the directivity plot
c0, alpha, beta_deg, beta, f, omega, dir_meas, dir_meas_deg, H_post = init_dir_plot()
for n in range(np.shape(f)[0]):
    H_post[:,n] = calc_directivity(PALC_config.directivity, alpha, \
                                   PALC_config.Lambda_y, beta, omega[n], c0, \
                                   f, dir_meas[:,n], dir_meas_deg[:,1], n)
setattr(SFP_config, 'f', f)
SFP_config.get_plot_dir_amp(H_post, [100, 1000, 10000])    
# write data in ColumnDataSource that is for directivity plotting
directivity_dict_plt.data = dict(amplitude_100=SFP_config.plot_dir_amp[0], \
                                 amplitude_1k=SFP_config.plot_dir_amp[1], \
                                 amplitude_10k=SFP_config.plot_dir_amp[2], \
                                 degree=list(beta_deg))   

#%%########## CALLBACKS SPECIFICALLY USED IN PAGE 1. PAL ######################    

def update_point(attr, old, new):
    """
    Function that saves the input data to draw audience lines.
    Reads the input of the start and end points and stores them in :any:`guipal`.
    After storing, updates the range of the Venue Slice Plot. Triggered by
    :any:`start_point_x`, :any:`start_point_y`, :any:`end_point_x` and
    :any:`end_point_y`.
    
    Parameters
    ----------
    attr : str
        Value attribute returned from widget.
    old : float
        Old value of changed attribute.
    new : float
        New value of changed attribute.

    Returns
    -------
    None.

    """
    # transform points into floats
    try:
        x = np.array([float(start_point_x.value), float(end_point_x.value)])
        y = np.array([float(start_point_y.value), float(end_point_y.value)])
        update_text(pal_error_info,'')
    except:
        x = np.array([0,0]).astype(float)
        y = np.array([0,0]).astype(float)
        update_text(pal_error_info, 'Error: Wrong Input. Only Numbers are allowed')
        return    
    # write the values in the dictionary
    guipal.data = dict(x=x, y=y)

    
def save_line():
    """
    Function to draw audience lines based on the data stored in :any:`guipal`.
    Triggered by :any:`save_line_button`
    Checks the defined conditions defined in :py:mod:`checkPAL`. These are:
        
        1. Stop values are bigger or equal to the start values
        2. The slope of the lines must be zero or positive
        3. No point of intersection shall be in the range of a line
        
    Stores the drawn lines in :any:`guilineplt`.

    Returns
    -------
    None.

    """    
    # load the necessary data input
    [x_start, x_stop] = guipal.data['x']
    [y_start, y_stop] = guipal.data['y']

    # distance between start and end point of audience line
    diff = np.sqrt((x_stop - x_start)**2 + (y_stop - y_start)**2)
    ################# check conditions for adding a audience line ############################
    condition1, condition2, condition3, slope, b = \
        check_all_draw_conditions(x_start, x_stop, y_start, y_stop, \
                                  created_pal, pal_error_info)        
    # Check if all conditions are fullfilled, add the line, save the line in the dictionary and set up the plot range again
    if condition1 and condition2 and condition3:
        update_text(pal_error_info,'')
        try:
            l_d = float(line_discretization.value)
        except:
            l_d = 1
            update_text(pal_error_info,'Couldnt process discretization input. Use default: 5 points per m')
        # Create the new audience line
        created_pal.append_line(x_start, x_stop, y_start, y_stop, slope, b, diff*l_d)
        if np.shape(created_pal.xline)[0] > 2000:
            remove_line()
            update_text(pal_error_info,'Couldnt append line. Maximum discretized points are 2000. Please reduce the discretization.')
            return
        # set ranges
        x_range = plt_ranges.update_range('p_x', [-2,2], created_pal.xline_start, \
                                          created_pal.xline_stop, PALC_config.x_H)
        y_range = plt_ranges.update_range('p_y', [-2,2], created_pal.yline_start, \
                                          created_pal.yline_stop, PALC_config.y_H)
        # write ranges to plots
        scale_p_axis(call=False)
        set_ranges(figs=[pBeam, pBeam_ref, pSFPv], sizes=[True, True, False], \
                   ranges=[x_range, y_range])
        pSPLoverX.x_range = x_range        
        # save data into the dictionries
        guilineplt.data = dict(xlineplt=created_pal.plot_x, ylineplt=created_pal.plot_y)
        change_start_pal_button(False)
        steps()


def remove_line(): 
    """
    Removes the last created line.
    Triggered by :any:`remove_line_button`.

    Returns
    -------
    None.

    """
    # remove the last created line
    if np.shape(created_pal.pal_index)[0] > 1:
        created_pal.remove_last_line()
    if np.shape(created_pal.pal_index)[0] > 1:
        # set ranges
        x_range = plt_ranges.update_range('p_x', [-2,2], created_pal.xline_start, \
                                          created_pal.xline_stop, PALC_config.x_H)
        y_range = plt_ranges.update_range('p_y', [-2,2], created_pal.yline_start, \
                                          created_pal.yline_stop, PALC_config.y_H)
        # write ranges to plots
        scale_p_axis(call=False)
        set_ranges(figs=[pBeam, pBeam_ref, pSFPv], sizes=[True, True, False], \
                   ranges=[x_range, y_range])
        pSPLoverX.x_range = x_range
          
    # write the output into the dictionaries
    guilineplt.data = dict(xlineplt=created_pal.plot_x, ylineplt=created_pal.plot_y)
    change_start_pal_button(False)


def create_prev_line():  
    """
    Creates a line in front of the selected line (only possible if a line is
    selected). Gets the tapped line from :any:`audience_line_tap_dict`, checks
    the draw conditions and inserts the line with ... Stores the data in
    :any:`guilineplt`.
    Triggered by :any:`save_prev_line_button`.

    Returns
    -------
    None.

    """    
    # load the necessary data out of the dictionaries
    # distance between start and end point of audience line
    [x_start, x_stop] = guipal.data['x']
    [y_start, y_stop] = guipal.data['y']
    sel_ind = audience_line_tap_dict.data['tapped'][0]
    # get the discretization of the line
    diff = np.sqrt((x_stop - x_start)**2 + (y_stop - y_start)**2)
    try:
        l_d = float(line_discretization.value)
    except:
        l_d = 1
        update_text(pal_error_info, 'Couldnt process discretization input. Use default: 5 points per m')
    # check if draw conditions are fulfilled
    condition1, condition2, condition3, slope, b = \
        check_all_draw_conditions(x_start, x_stop, y_start, y_stop, \
                                  created_pal, pal_error_info)
    # Check if all conditions are fullfilled, add the line, save the line in the dictionary and set up the plot range again
    if condition1 and condition2 and condition3:
        update_text(pal_error_info, '')  
        # Insert new audience line      
        created_pal.insert_previous_line(sel_ind, x_start, x_stop, y_start, \
                                         y_stop, slope, b, diff*l_d)
        if np.shape(created_pal.xline)[0] > 2000:
            remove_sel_line()
            update_text(pal_error_info,'Couldnt append line. Maximum discretized points are 2000. Please reduce the discretization.')
            return
        # write the output into the dictionaries
        guilineplt.data = dict(xlineplt=created_pal.plot_x, ylineplt=created_pal.plot_y)
        change_start_pal_button(False)
        steps()
    

def remove_sel_line():
    """
    Removes the selected line. Tapped line is stored in :any:`audience_line_tap_dict`.
    The index of the lines behind the removed line are then shifted by -1.
    Triggered by :any:`remove_sel_line_button`.

    Returns
    -------
    None.

    """
    # load the necessary data out of the dictionaries
    sel_ind = audience_line_tap_dict.data['tapped'][0]    
    # remove selected line
    created_pal.remove_selected_line(sel_ind)       
    # write the output into the dictionaries and change buttons
    guilineplt.data = dict(xlineplt=created_pal.plot_x, ylineplt=created_pal.plot_y)
    change_start_pal_button(False)
    steps()


def audience_line_tap(attr, old, new):
    """
    Controls the visibility of line saving and removing buttons / widgets depending
    on user selection of audience lines. Updates :any:`audience_line_tap_dict`
    if a line is selected (or not).
    Triggered if user taps on a line in figure :any:`p`.

    Parameters
    ----------
    attr : str
        Attribute name (here index).
    old : int
        Index of previously tapped line. [] if no line is tapped.
    new : int
        Index of actually tapped line. [] if no line is tapped..

    Returns
    -------
    None.

    """
    if new != []:
        save_prev_line_button.visible = True
        remove_sel_line_button.visible = True
        save_line_button.visible = False
        remove_line_button.visible = False
    else:
        save_prev_line_button.visible = False
        remove_sel_line_button.visible = False
        save_line_button.visible = True
        remove_line_button.visible = True
    audience_line_tap_dict.data = dict(tapped=new)

   
def check_PAL():
    """
    Creates a PALC compatible venue slice suggestions based on the lines drawn
    by the user. Updates :any:`guinalplt` and :any:`guisfpplt`. Calls checkPAL
    functions and PALC_pal methods to create the venue. :any:`PALC_pal` may then
    be used for PALC computation. Resets initialization of :any:`Opt_w`.
    Triggered by :any:`check_PAL_button`.

    Returns
    -------
    None.

    """
    try:
        # initialize objects
        suggested_nal = PALC_classes.PAL(ind_iscut=[])
        suggested_sfp = PALC_classes.PAL(ind_iscut=[])
        # compute checking
        checkPAL.suggestPAL(created_pal, suggested_nal, suggested_sfp)
        # plot the suggested lines
        guinalplt.data = dict(xnalplt=suggested_nal.plot_x, ynalplt=suggested_nal.plot_y)
        guisfpplt.data = dict(xsfpplt=suggested_sfp.plot_x, ysfpplt=suggested_sfp.plot_y)
        # initialize PALC compatible pal lines and create them
        PALC_pal.pal, PALC_pal.pal_no_nal = [], []
        checkPAL.get_PALC_compatible_PAL(PALC_pal, created_pal, suggested_nal, suggested_sfp)
        PALC_pal.create_pal_without_nal(suggested_nal)
        change_start_pal_button(True)
        update_text(config_error_info,'')
        steps()
        if gui_weighting.value in ['Target Slope']:
            gui_weighting.value = 'Without'
            Opt_w.init = False
            weighting_select(0,0,0)
            #np.shape(created_pal.xline)[0]
        weight_links = np.arange(np.shape(created_pal.xline_start)[0]+1)
        gui_opt_weight_link.options = [str(n) for n in list(weight_links)]
        gui_opt_weight_link.value = '0'
    except:
        update_text(config_error_info,'Error: Could not create PAL proposal')
          

def change_start_pal_button(has_pal):
    """
    If :any:`check_PAL` terminated succesfully, the "Run Calcution" Button is
    enabled.

    Parameters
    ----------
    has_pal : bool
        If True: PALC computation is enabled, otherwise disabled.

    Returns
    -------
    None.

    """
    if has_pal and PALC_pal.pal != []:
        gui_start.disabled = False
        check_PAL_button.button_type = "success"
        check_PAL_button.label = "8. Venue is PALC compatible!"
        p.legend.visible = False
        gui_use_fixed_angles.disabled = False
    else:
        gui_start.disabled = True
        check_PAL_button.button_type = "warning"
        check_PAL_button.label = "8. Get PALC Venue Proposal"
        p.legend.visible = False
        gui_use_fixed_angles.disabled = True
        gui_use_fixed_angles.value = 'No'
        gui_fixed_first_angle.disabled = True
        PALC_config.use_fixed_angle = False

#%%### CALLBACKS SPECIFICALLY USED IN PAGE 2. LOUDSPEAKER CONFIGURATION #######    

def upload_directivity(attr, old, new):
    """
    Reads the directivity and stores the directivity of the plotted frequencies
    in :any:`directivity_dict_plt`.
    Uploaded csv-data must be formatted as: frequency bins in lines and angle
    in degree in rows. Imaginary unit can be "j" or "i".
    Triggered if :any:`button_input` receives a .csv-file.

    Parameters
    ----------
    attr : str
        Changed attribute.
    old : str
        Old input.
    new : str
        New input. csv-data to read.

    Returns
    -------
    None.

    """
    try:
        read_dir_data(SFP_config, new)
        directivity_dict_plt.data = dict(amplitude_100=SFP_config.plot_dir_amp_meas[0], \
                                         amplitude_1k=SFP_config.plot_dir_amp_meas[1], \
                                         amplitude_10k=SFP_config.plot_dir_amp_meas[2], \
                                         degree=SFP_config.plot_beta_meas)
    except:
        update_text(config_error_info,'Error: Could not read Measured Loudspeaker Data')
        
#%%#### CALLBACKS SPECIFICALLY USED IN PAGE 3. ARRAY CONFIGURATION ############

## Callback to disable and enable field of discrete tilt angles and 
# shows a default example
def choose_dangles(attr, old, new):
    """
    If discrete angles are used, text input field to insert a set of discrete
    tilt angles is enabled (and filled with a default set).
    Triggered by :any:`gui_use_dangles`.

    Parameters
    ----------
    attr : str
        Changed attribute (value can be 'Yes' or 'No').
    old : str
        Old value of the widget gui_use_dangles.
    new : TYPE
        New value of the widget gui_use_dangles.

    Returns
    -------
    None.

    """
    if gui_use_dangles.value == 'Yes':
        gui_discrete_angles.value = '0,0.5,1,2,3.5,5,7,10'
        gui_discrete_angles.disabled = False
        PALC_config.store_config('use_gamma_LSA', True)
        gui_ref_discrete_angles.disabled = False
    elif gui_use_dangles.value == 'No':
        gui_discrete_angles.value = ''
        gui_discrete_angles.disabled = True
        PALC_config.store_config('use_gamma_LSA', False)
        gui_ref_discrete_angles.disabled = True
        gui_ref_discrete_angles.value = "No"
    get_discrete_angles(attr, old, new)


def get_discrete_angles(attr, old, new):
    """
    Reads a set of discrete tilt angles. Enabled if :any:`gui_use_dangles`
    is set to 'Yes'. Triggered by :any:`gui_discrete_angles`.

    Parameters
    ----------
    attr : str
        Changed attribute.
    old : str
        Old set of discrete tilt angles.
    new : str
        New set of discrete tilt angles with ',' as seperator and '.' as decimal
        seperator.

    Returns
    -------
    None.

    """
    # check if function was called because discrete angles were disabled
    if PALC_config.use_gamma_LSA == False:
        update_text(config_error_info,'')
        PALC_config.gamma_LSA = []
        return  
    # convert input and store in PALC_config object
    try:
        PALC_config.gamma_LSA = np.array([float(s) for s in gui_discrete_angles.value.split(',')])*(np.pi/180)
#        discrete_angles_dict.data = dict(discrete_angles=PALC_config.gamma_LSA)
        update_text(config_error_info,'')
    except:
        update_text(config_error_info,'Error: Could not process the input')

      
def reference_array_setting(attr, old, new):
    """
    Saves the current setting of the reference LSA. Set widgets visible / invisible
    or enables / disables them if needed / not needed. Triggered by
    :any:`gui_ref_array`.

    Parameters
    ----------
    attr : str
        Changed attribute (value).
    old : str
        Old reference array setting.
    new : str
        New reference array setting.

    Returns
    -------
    None.

    """
    arrange_ref_arr_in(gui_ref_array, gui_ref_start, gui_ref_step_stop, \
                       gui_ref_discrete_angles, gui_ref_userdef)
    get_ref_array_angles()

     
def get_ref_array_angles(chg_n=False):
    """
    Saves and calculates the reference array tilt angles. Is triggered by
    :any:`reference_array_setting`, :any:`gui_ref_start`, :any:`gui_ref_step_stop`,
    :any:`gui_ref_discrete_angles` and :any:`gui_ref_userdef`.

    Parameters
    ----------
    chg_n : bool, optional
        True if number of LSA cabinets has changed. The default is False.

    Returns
    -------
    None.

    """
    try:
        ref_array_angles(PALC_config, Ref_arr, gui_ref_array, \
                         gui_ref_start, gui_ref_step_stop, \
                         gui_ref_discrete_angles, gui_ref_userdef)
        # write to dict to show in DataTable in degree
        gamma_ref_dict.data = dict(gamma_ref=np.round(Ref_arr.gamma_tilt_deg, decimals=2), \
                                   LS=np.linspace(1, PALC_config.N, PALC_config.N))
        # convert to radian
        Ref_arr.gamma_n = Ref_arr.gamma_tilt_deg * (np.pi / 180)
        if chg_n == False:
            update_text(config_error_info,'')
    except:
        update_text(config_error_info,'Error: Could not process the input')
        gui_ref_start.value = '0'
        gui_ref_step_stop.value = '0'
        
#%%##### CALLBACKS SPECIFICALLY USED IN PAGE 4. ALGORITHM #####################

## Callback to select the weighting method (f.e. -3 dB loss per distance doubling)
def weighting_select(attr, old, new):
    """
    Arranges the GUI depending on chosen weighting approach. Runs init PALC
    calculation by triggering :any:`start_calc`, if 'Target Slope' is chosen.
    Triggered by :any:`gui_weighting`.

    Parameters
    ----------
    attr : str
        Changed attribute (value).
    old : str
        Old weighting approach.
    new : str
        New weighting approach.

    Returns
    -------
    None.

    """
    set_weighting_in(gui_weighting, weighting_plus, weighting_minus, \
                     weighting_step_size, gui_weighting_nu)
    if gui_weighting.value in ['Target Slope']:
        start_calc(True)
        Opt_w.init = True
        SPLoverX_tabs.active = 1 #SPLoverX_tab2
        gui_opt_weight_link.visible=True
        gui_weighting.width = 135
    else:
            gui_opt_weight_link.visible=False
            gui_weighting.width = 255
    PALC_config.store_config('use_weighting', gui_weighting.value)
    update_gh_w_select(gui_weighting, gui_gap_handling)
    
    
def update_weighting_step_size(attr, old, new):
    """
    Updates weighting step size to update :any:`gui_weighting_nu` by clicking
    on :any:`weighting_plus` or :any:`weighting_minus`. Triggered by
    :any:`weighting_step_size`.

    Parameters
    ----------
    attr : str
        Changed attribute (value).
    old : str
        Old step size.
    new : str
        New step size.

    Returns
    -------
    None.

    """
    gui_weighting_nu.step = float(weighting_step_size.value)
    

def adjust_weighting_plus():
    """
    Increases :any:`gui_weighting_nu` by :any:`weighting_step_size`.

    Returns
    -------
    None.

    """
    gui_weighting_nu.value += float(weighting_step_size.value)

   
def adjust_weighting_minus():
    """
    Decreases :any:`gui_weighting_nu` by :any:`weighting_step_size`.

    Returns
    -------
    None.

    """
    gui_weighting_nu.value -= float(weighting_step_size.value)


def store_weighting_nu(attr, old, new):
    """
    Stores actual value to calculate weighting factors (nu). Triggered by
    :any:`gui_weighting_nu`.

    Parameters
    ----------
    attr : str
        Changed attribute (value).
    old : float
        Old nu.
    new : float
        New nu.

    Returns
    -------
    None.

    """
    PALC_config.weighting_nu = np.round(gui_weighting_nu.value**3, decimals=2)


def use_fixed_angle(attr, old, new):
    """
    Enables or disables the the possibility to use a fixed angle for the
    heighest LSA cabinet. Runs an init of PALC to calculate the possible range
    of the angle of the fixed loudspeaker. Calls :any:`start_calc` and
    :any:`get_fixed_angle_borders`.

    Parameters
    ----------
    attr : str
        Changed attribute (value).
    old : str
        Old value if fixed angle is used.
    new : str
        New value if fixed angle is used.

    Returns
    -------
    None.

    """
    if gui_use_fixed_angles.value == 'Yes' and PALC_pal.pal != []:
        gamma_tilt_deg = get_fixed_angle_borders(PALC_pal.pal[-1], PALC_config)
        Opt_arr.fixed_angle = [gamma_tilt_deg[0]-1, gamma_tilt_deg[1], gamma_tilt_deg[1]-10]
        gui_fixed_first_angle.disabled = False
        if Opt_arr.fixed_angle[1] < Opt_arr.fixed_angle[0]:
            gui_fixed_first_angle.value = str(Opt_arr.fixed_angle[1])
        else:
            gui_fixed_first_angle.value = str(Opt_arr.fixed_angle[0]-1)
        update_text(fixed_angle_text, 'Max: '+str(np.round(Opt_arr.fixed_angle[0], decimals=2))+ \
                    ', Min: '+str(np.round(Opt_arr.fixed_angle[2], decimals=2)))
        PALC_config.use_fixed_angle = True
    elif gui_use_fixed_angles.value == 'No':
        gui_fixed_first_angle.disabled = True
        Opt_arr.fixed_angle = [0,0,0]
        PALC_config.use_fixed_angle = False
        

def start_calc(arg):
    """
    Main routine of the application. Runs the PALC algorithm with given setting
    venue slice by the user. Always calls :any:`get_initial_weights`,
    :py:meth:`calcPALC` or :any:`optimize_PALC`, respectively, :any:`get_plot_array`,
    :any:`venue_slice2CDS`, :any:`calc_diff_tilt_angles`,
    :any:`LSA_visualization`, :any:`calcSFP`, :any:`calcSPLoverX`,
    :any:`calcHomogeneity`, :any:`calcHistogram`. Triggered by :any:`gui_start`
    and for initialization by :any:`weighting_select` and :any:`use_fixed_angle`.
    Saves all output in belonging :py:class:`bokeh.models.ColumnDataSource` to
    visualize the output and sets the ranges of the figures.

    Parameters
    ----------
    arg : bool
        If false, return nothing.

    Returns
    -------
    None.

    """
    if not arg:
        return
    if PALC_pal.pal == []:
        gui_start.active = False
        update_text(config_error_info,'Error: Please CREATE and GET PALC compatible venue proposal on page 1.')
        return
    # Start the PALC algorithm
    PALC_config.get_initial_weights()
#    PALC_plots     = PALC_classes.PALC_plotting()
    ###################### RUN THE CALCUlATION ####################################

    try:
        # weighting optimization here    
        if PALC_config.use_weighting == 'Target Slope' and Opt_w.init:
            x, SPL = SPLoverX_dict.data['x'], SPLoverX_dict.data['SPL']
            Opt_w.calc_init()
            # move reference
            #Opt_w.maxonref(0)
            Opt_SPLoverX_dict.data.update(dict(SPL=list(Opt_w.SPL)))
            # start optimization, last input is max loops (default=40)
            opt_ind = optimize_PALC(PALC_config, SFP_config, PALC_pal, PALC_plots, \
                                    Opt_arr, created_pal, Tech_res, SPLoverX, Opt_w, \
                                    200)
        else:
            calcPALC.calcPALC(PALC_plots, PALC_pal, PALC_config, Opt_arr)        
    except:
        gui_start.active = False
        update_text(config_error_info,'Error: PALC calculation failed')
    ###############################################################################
    # reset some possibly changed values...
    for key, val in [['last_angle_hm', []],['x_H', gui_x_H.value], \
                     ['y_H', gui_y_H.value], ['psi_n', gui_psi_n.value]]:
        PALC_config.store_config(key, val)
    # case no convergence reached and calcPALC returned by except-case
    if Opt_arr.num_iter == 500 or np.shape(Opt_arr.gamma_tilt_deg)[0] == 0:
        update_text(config_error_info,'Error: No convergence reached')
        gui_start.active = False
        return
    else:
        update_text(config_error_info,'')
    # save / copy results to plot in ColumnDataSourcee
    try:    
        x_patches, y_patches = LSA_visualization(PALC_plots, PALC_config, Opt_arr.gamma_n)
        plot_patches.data.update(dict(x_list=x_patches, y_list=y_patches, \
                                      x_list_ref=np.zeros(PALC_config.N), \
                                      y_list_ref=np.zeros(PALC_config.N)))
    except:
        gui_start.active = False
        update_text(config_error_info,'Error: Could not store the output')
        return         
    try:
        PALC_plots.get_plot_array(Opt_arr)
        venue_slice2CDS(plt_venue_slice_1, plt_venue_slice_2, \
                        PALC_plots, Opt_arr)
    except:
        venue_slice2CDS(plt_venue_slice_1, plt_venue_slice_2, \
                        [], [])
        update_text(config_error_info,'Error: Could not store the output')
        gui_start.active = False
        return       
    # Prepare the data output (results) and save them in a dictionary
    gamma_tilt_deg_diff, gamma_tilt_deg_round = calc_diff_tilt_angles(Opt_arr.gamma_tilt_deg)
    setattr(Opt_arr, 'gamma_tilt_deg_diff', gamma_tilt_deg_diff)
    results_dict.data = dict(a_num_LS=np.linspace(1,PALC_config.N,num=PALC_config.N), \
                             b_gamma_tilt_deg=gamma_tilt_deg_round, \
                             c_gamma_tilt_deg_diff=gamma_tilt_deg_diff)
    ############## Calculate the Sound Field Prediction and put the results in the dictionaries ######################## 
    try:
        t1 = time.time()
        x_S, y_S = calcSFP(Opt_arr.gamma_tilt_deg, created_pal, SFP_config, \
                           Tech_res)#, dir_meas_LSA, dir_meas_degree)
        t2 = time.time()
        dt = t2-t1
        print('elapsed time for SFP in s: ',dt)
    except:
        gui_start.active = False
        update_text(config_error_info,'Error: SFP calculation failed')
        return

    ################## Calculation for SPL over distance ###########################################################
    calcSPLoverX(PALC_plots, created_pal, Tech_res.p_SPL, SPLoverX, SFP_config.freq_range)
    ####### Calculate the Homogeneity and Bar Chart #############
    calcHomogeneity(Tech_res, x_S, y_S, created_pal.xline, created_pal.yline) 
    calcHistogram(Tech_res)
    ################# Calculation of reference array #############################################################
    try:
        x_S, y_S = calcSFP(Ref_arr.gamma_tilt_deg, created_pal, SFP_config, \
                           Tech_res_ref)#, dir_meas_LSA, dir_meas_degree)
    except:
        gui_start.active = False
        update_text(config_error_info,'Error: SFP calculation of reference array failed')

    x_patches_ref, y_patches_ref = LSA_visualization(PALC_plots_ref, PALC_config, \
                                                     Ref_arr.gamma_n)
    calcSPLoverX(PALC_plots_ref, created_pal, Tech_res_ref.p_SPL, SPLoverX_ref, SFP_config.freq_range)
    calcHomogeneity(Tech_res_ref, x_S, y_S, created_pal.xline, created_pal.yline)
    calcHistogram(Tech_res_ref)
    ############# Plot and save to plot dictionaries ##########################
    # Sound Field Prediction
    SFPv_dict.data = dict(x_vert=created_pal.xline, y_vert=created_pal.yline, \
                          foo=list(np.arange(np.shape(created_pal.xline)[0])))
    pSFPv.x('x_vert','y_vert', source=SFPv_dict, \
            color=linear_cmap('foo', rainbow[::-1], 0, np.shape(Tech_res.p_SPL)[0]-1))
    SFP_dict.data = dict(p_SPL=Tech_res.plot_p_SPL, f=Tech_res.plot_f, \
                         foo=list(np.arange(np.shape(Tech_res.plot_p_SPL)[0])), \
                         p_SPL_ref=Tech_res_ref.plot_p_SPL)
    pSFP.multi_line('f','p_SPL', source=SFP_dict, color=linear_cmap(field_name='foo', \
                    palette=rainbow[::-1], low=0, high=np.shape(Tech_res.p_SPL)[0]-1), \
                    line_alpha=0.9)
    pSFPref.multi_line('f','p_SPL_ref', source=SFP_dict, color=linear_cmap(field_name='foo', \
                       palette=rainbow[::-1], low=0, high=np.shape(Tech_res_ref.p_SPL)[0]-1), \
                       line_alpha=0.9)
    if not Opt_w.init:
        shift_ref_on_zero(Opt_w, SPLoverX, SPLoverX_ref)
        SPLoverX_dict.data.update(dict(SPL=SPLoverX.SPL, x=SPLoverX.x, \
                                   x_v=created_pal.xline, y_v=created_pal.yline))
        SPLoverX_ref_dict.data.update(dict(SPL=SPLoverX_ref.SPL, x=SPLoverX_ref.x, \
                                   x_v=created_pal.xline, y_v=created_pal.yline))
        Opt_SPLoverX_dict.data.update(dict(SPL=[SPLoverX.SPL[0], SPLoverX.SPL[-1]], \
                                           x  =[SPLoverX.x[0], SPLoverX.x[-1]], \
                                           x_v=[created_pal.xline[0],created_pal.xline[-1]], \
                                           y_v=[created_pal.yline[0],created_pal.yline[-1]]))
        Opt_refpoint_dict.data.update(dict(x_ref=[Opt_w.x_interp[int(len(Opt_w.x_interp)/2)]], \
                                           SPL  =[Opt_w.SPL_interp[int(len(Opt_w.SPL_interp)/2)]], \
                                           x_v  =[created_pal.xline[int(len(created_pal.xline)/2)]], \
                                           y_v  =[created_pal.yline[int(len(created_pal.yline)/2)]]))
    elif PALC_config.use_weighting == 'Target Slope' and Opt_w.init:
        shift2ref(Opt_w, SPLoverX, opt_ind, SPLoverX_ref)
        get_opt_region(SPLoverX, Opt_w)
        SPLoverX_dict.data.update(dict(SPL=SPLoverX.SPL, x=SPLoverX.x, \
                                       x_v=created_pal.xline, y_v=created_pal.yline))
        Opt_SPLoverX_dict.data.update(dict(SPL=Opt_w.SPL))
        Opt_refpoint_dict.data.update(dict(SPL=[Opt_w.SPL_interp[Opt_w.ref_ind]]))
        SPLoverX_ref_dict.data.update(dict(SPL=SPLoverX_ref.SPL))
    else:
        SPLoverX_dict.data.update(dict(SPL=SPLoverX.SPL, x=SPLoverX.x, \
                                       x_v=created_pal.xline, y_v=created_pal.yline))
        SPLoverX_ref_dict.data.update(dict(SPL=SPLoverX_ref.SPL, x=SPLoverX_ref.x, \
                                   x_v=created_pal.xline, y_v=created_pal.yline))
    # Homogeneity
    Homogeneity_dict.data = dict(H=Tech_res.H, f=Tech_res.f, \
                                 H_dist_high=Tech_res.H_dist[1], \
                                 H_dist_low=Tech_res.H_dist[0], \
                                 H_str=Tech_res_ref.H)
    # Histogram
    Bar_dict.data     = dict(top=Tech_res.Hist_tops, x=Tech_res.Hist_steps)
    Bar_dict_ref.data = dict(top_ref=Tech_res_ref.Hist_tops, \
                             x_ref=Tech_res_ref.Hist_steps)
    # Set Ranges for SFP, Venue Slice, Homogeneity and Bar Chart
    # Bar Chart
    bar_range     = plt_ranges.update_range('bar_range', [-1,1], \
                                            Tech_res.Hist_steps, \
                                            Tech_res_ref.Hist_steps)
    bar_top_range = plt_ranges.update_range('bartop', [-1,2], \
                                            Tech_res.Hist_tops, \
                                            Tech_res_ref.Hist_tops)
    pBar.x_range, pBar.y_range       = bar_range, bar_top_range
    pBarref.x_range, pBarref.y_range = bar_range, bar_top_range   
    # Homogeneity
    hom_y_range  = plt_ranges.update_range('h_y', [-2,2], Tech_res.H, \
                                           Tech_res_ref.H)
    pHom.y_range = hom_y_range                                        
    # SFP
    psfp_y_range = plt_ranges.update_range('sfp_y', [-5,5], Tech_res.p_SPL, \
                                           Tech_res_ref.p_SPL)
    pSFP.y_range, pSFPref.y_range = psfp_y_range, psfp_y_range
    # SPLoverX
    SPLoverX_y_range = plt_ranges.update_range('SPLoverX_y', [-1,1], Opt_w.SPL_interp, \
                                               SPLoverX.SPL_interp, SPLoverX.SPL, \
                                               SPLoverX_ref.SPL)
    pSPLoverX.y_range = SPLoverX_y_range
    update_text(config_error_info,'')
    gui_which_beam.active, gui_which_beam.disabled = [], False
    # Array visualization
    plot_patches.data.update(dict(x_list=x_patches, y_list=y_patches, \
                                  x_list_ref=x_patches_ref, \
                                  y_list_ref=y_patches_ref))
    gui_start.active = False
    steps()


def clear_result_fig():
    """Clears the venue slice plot. Calls :any:`venue_slice2CDS`. Triggered by
       :any:`gui_clear_result_fig` and in some cases by :any:`steps`. """
    venue_slice2CDS(plt_venue_slice_1, plt_venue_slice_2, \
                    [], [])
        

#%%### CALLBACKS USED ON MORE THAN ONE PAGE OR IN 5. SOUND FIELD PREDICTION ###

def get_value(attr, old, new):
    """
    This function saves the user input of many widgets into the config objects
    PALC_config and SFP_config by calling :any:`store_config`. The widgets that
    trigger this function are:
        
        * :any:`gui_x_H` -  Highest front point of the LSA on the x-axis
        * :any:`gui_y_H` -  Highest front point of the LSA on the y-axis
        * :any:`gui_Lambda_y` - Height of the loudspeakers
        * :any:`gui_Lambda_gap` - Distance between adjacent LSA cabinets
        * :any:`gui_PALC_alg` -  Algorithm type (PALC1, PALC2 or PALC3)
        * :any:`gui_N_LS` - Number of LSA cabinets
        * :any:`gui_psi_n` - Initialization aperture angle of LSA cabinets
        * :any:`gui_use_dangles` - Selection it continuous or discrete tilt angles
          shall be used (does not trigger this function directly)
        * :any:`gui_gap_handling` - Gap handling approach (Without, Soft or Hard Margin)
        * :any:`gui_strength_smarg` - Strength of the soft margin gap handling approach
        * :any:`gui_directivity` - Loudspeaker directivity (Circular Piston, Line
          Piston, Combined Circular and Line Piston or Measured Loudspeaker Data)
        * :any:`gui_fixed_first_angle` - If the highest LSA cabinet shall be
          mounted with a fixed tilt angle, this widget takes the fixed angle in
          degree.
          
    If entered value cannot be stored, the default values will be used. Calls also
    :any:`get_ref_array_angles` if the value of :any:`gui_use_dangles` or :any:`gui_N_LS`
    has changed. Always calls :any:`round_fixed_angle`, :any:`choose_gap_handling`,
    :any:`update_gh_w_select` and :any:`init_dir_plot`. Updates ranges of figures.

    Parameters
    ----------
    attr : str
        Changed attribute (value).
    old : either str, int, float
        Old value of changed attribute of triggered widget.
    new : either str, int, float
        New value of changed attribute of triggered widget.

    Returns
    -------
    None.

    """
    # make dictionary of possible widgets to evaluate
    widget_dict = {'constraint':gui_PALC_alg.value, \
                   'use_gamma_LSA':gui_use_dangles.value, \
                   'gap_handling':gui_gap_handling.value, \
                   'strength_sm':gui_strength_smarg.value, \
                   'Lambda_y':gui_Lambda_y.value, \
                   'Lambda_gap':gui_Lambda_gap.value, \
                   'directivity':gui_directivity.value, \
                   'fixed_angle':gui_fixed_first_angle.value, \
                   'x_H':gui_x_H.value, 'y_H':gui_y_H.value, \
                   'N':gui_N_LS.value, \
                   'psi_n':gui_psi_n.value, 'tolerance':gui_tolerance.value, \
                   'freq_range':gui_freq_range.value}
    # iterate over widget_dict to detect changed value and store in object
    for key, val in widget_dict.items():
        if new == val:
            error_info, failed_key = PALC_config.store_config(key, val)
            # if wrong user input --> insert default value
            if failed_key in ['Lambda_y']    : setattr(gui_Lambda_y,attr,'0.2')
            if failed_key in ['N']           : setattr(gui_N_LS,attr,'4') 
            if failed_key in ['Lambda_gap']  : setattr(gui_Lambda_gap,attr,'0')
            if failed_key in ['tolerance']   : setattr(gui_tolerance,attr,'0.1')
            if failed_key in ['fixed_angle'] : setattr(gui_fixed_first_angle,attr,'')
            if failed_key in ['x_H']         : setattr(gui_x_H,attr,'0')
            if failed_key in ['y_H']         : setattr(gui_y_H,attr,'2')
            if failed_key in ['too low']     : setattr(gui_y_H,attr,str(PALC_config.y_H))
            if failed_key in ['psi_n']       : setattr(gui_psi_n,attr,'0.0524')
            # Set attribute in SFP_config if possible
            if hasattr(SFP_config, key) and failed_key in [None, 'too low']: 
                SFP_config.store_config(key,val)
            update_text(config_error_info,str(error_info))
            if key in ['N', 'use_gamma_LSA']:
                get_ref_array_angles(chg_n=True)
    # round fixed angle of heighest LS
    round_fixed_angle(PALC_config, Opt_arr, gui_fixed_first_angle)
    # set chosen gap_handling
    choose_gap_handling(gui_gap_handling, gui_strength_smarg)
    # set options if hard margin or target slope is chosen
    update_gh_w_select(gui_weighting, gui_gap_handling)

 # For the selected Directivity, directly provide the plot for three frequencies          
    if PALC_config.directivity in ['Circular Piston','Line Piston','Combined Circular/Line']:
        button_input.visible = False
        # init of diretivity variables
        c0, alpha, beta_deg, beta, f, omega, dir_meas, dir_meas_deg, H_post = init_dir_plot() 
        for n in range(np.shape(f)[0]):
            H_post[:,n] = calc_directivity(PALC_config.directivity, alpha, \
                                           PALC_config.Lambda_y, beta, omega[n], \
                                           c0, f, dir_meas[:,n], dir_meas_deg[:,1], n)
        setattr(SFP_config, 'f', f)
        SFP_config.get_plot_dir_amp(H_post, [100, 1000, 10000])    
    # write data in ColumnDataSource that is for directivity plotting
        directivity_dict_plt.data = dict(amplitude_100=SFP_config.plot_dir_amp[0], \
                                         amplitude_1k=SFP_config.plot_dir_amp[1], \
                                         amplitude_10k=SFP_config.plot_dir_amp[2], \
                                         degree=list(beta_deg))   
    # else Measured Loudspeaker Data is selected
    else:
        button_input.visible = True
        directivity_dict_plt.data = dict(amplitude_100=SFP_config.plot_dir_amp_meas[0], \
                                         amplitude_1k=SFP_config.plot_dir_amp_meas[1], \
                                         amplitude_10k=SFP_config.plot_dir_amp_meas[2], \
                                         degree=SFP_config.plot_beta_meas) 
    # set ranges
    if created_pal.xline != []:
        x_range = plt_ranges.update_range('p_x', [-2,2], created_pal.xline_start, \
                                          created_pal.xline_stop, PALC_config.x_H)
        y_range = plt_ranges.update_range('p_y', [-2,2], created_pal.yline_start, \
                                          created_pal.yline_stop, PALC_config.y_H)
        # write ranges to plots
        scale_p_axis(call=False)
        set_ranges(figs=[pBeam, pBeam_ref, pSFPv], sizes=[True, True, False], \
                   ranges=[x_range, y_range])
        pSPLoverX.x_range = x_range
        
        
def set_ranges(**fig_rng_sz):
    """
    Sets the x- and y-ranges of figures and plots.

    Parameters
    ----------
    **fig_rng_sz : lists
        Must contain three argument called figs = [figure], sizes = [bool],
        ranges = [x_range, y_range].

    Returns
    -------
    None.

    """
    for ind, figs in enumerate(fig_rng_sz['figs']):
        figs.x_range = fig_rng_sz['ranges'][0]
        figs.y_range = fig_rng_sz['ranges'][1]
        if fig_rng_sz['sizes'][ind] == True:
            figs.frame_height = int((plt_ranges.p_y[1]-plt_ranges.p_y[0]) * \
                 figs.frame_width / (plt_ranges.p_x[1]-plt_ranges.p_x[0]))
        
        
def scale_p_axis(call=True):
    """
    Changes the sizing mode of the venue slice. Calls therefore
    :any:`get_p_axis_rng`. Triggered by :any:`p_equal_axis`.

    Parameters
    ----------
    call : bool, optional
        If True, calls :any:`steps` to update the page. The default is True.

    Returns
    -------
    None.

    """
    get_p_axis_rng(p, plt_ranges, p_equal_axis)
    if call:
        steps()
        

def get_beamplot(attr, old, new):
    """
    Calculates the beamplot of PALC results and / or reference array and saves
    the beamplot data in a dictionary for visualization. Triggered by
    :any:`gui_f_beam` or :any:`gui_which_beam`. Calls :any:`set_beamplot_visibility`
    and :any:`getBeamplotRes`.

    Parameters
    ----------
    attr : str
        Changed attribute (Either value or active).
    old : either float or int
        Old attribute.
    new : either float or int
        New attribute.

    Returns
    -------
    None.

    """
    try:
        # visibility of the plots
        set_beamplot_visibility(gui_which_beam, pBeam, pBeam_ref)
        # if frequency is changed, make string to list for if request below
        if isinstance(old, str):
            old = [int(old)]
        if isinstance(new, str):
            new = [int(new)]
        # Beamplot Calculation    
        f = float(gui_f_beam.value)
        # Get indice of frequency for the case of Measured Loudspeaker Data
        ind_f = (np.abs(f-SFP_config.f).argmin())
        for n in gui_which_beam.active:
            if n == 0 and 0 not in old and (0 in new or int(gui_f_beam.value) in new):
                p_SPL = getBeamplotRes(pBeam, mapper, PALC_config, SFP_config, \
                                       Opt_arr.gamma_n, plt_ranges, f, ind_f)
                Beamplot_dict.data.update(dict(SPL=[p_SPL], \
                                               x=[plt_ranges.p_x[0]], \
                                               y=[plt_ranges.p_y[0]], \
                                               dw=[plt_ranges.p_x[1]-plt_ranges.p_x[0]], \
                                               dh=[plt_ranges.p_y[1]-plt_ranges.p_y[0]]))
            elif n == 1 and 1 not in old and (1 in new or int(gui_f_beam.value) in new):
                p_SPL_ref = getBeamplotRes(pBeam_ref, mapper_ref, PALC_config, \
                                           SFP_config, Ref_arr.gamma_n, plt_ranges, f, ind_f)
                Beamplot_dict.data.update(dict(SPL_ref=[p_SPL_ref], \
                                               x=[plt_ranges.p_x[0]], \
                                               y=[plt_ranges.p_y[0]], \
                                               dw=[plt_ranges.p_x[1]-plt_ranges.p_x[0]], \
                                               dh=[plt_ranges.p_y[1]-plt_ranges.p_y[0]]))
    except:
        pass


## Function for the TapTool that can be used on the venue slice plot in page 4 (sound field prediction).
def tap_handle(attr, old, new):
    """
    Handles the tap tool that can be used in the sound field prediction on page
    four. Highlites the tapped line or venue point in :any:`pSFPv` and
    :any:`pSFP`.

    Parameters
    ----------
    attr : str
        Changed attribute (index).
    old : int
        Old tapped index.
    new : TYPE
        New tapped index.

    Returns
    -------
    None.

    """
    # Initialize and Preparation
    f, p_SPL, p_SPL_ref = [], [], []        
    # hide the unchosen lines or show them again (if nothing select or reset)
    if np.shape(new)[0] >= 1:
        for n in range(np.shape(new)[0]):
            f.append(Tech_res.plot_f[new[n]])
            p_SPL.append(Tech_res.plot_p_SPL[new[n]])
            p_SPL_ref.append(Tech_res_ref.plot_p_SPL[new[n]])
        SFP_dict.data = dict(f=f, p_SPL=p_SPL, \
                             foo=list(np.arange(np.shape(p_SPL)[0])), \
                             p_SPL_ref=p_SPL_ref)        
    else:
        SFP_dict.data = dict(f=Tech_res.plot_f, p_SPL=Tech_res.plot_p_SPL, \
                             foo=list(np.arange(np.shape(Tech_res.plot_p_SPL)[0])), \
                             p_SPL_ref=Tech_res_ref.plot_p_SPL)
            
## Function to let the user draw the optimal SPL over X
def optimal_SPLoverX(attr, old, new):
    """
    Triggered if user draws in :any:`pSPLoverX`. Sets is2update attribute of
    :any:Opt_w` to zero. :any:`Opt_SPLoverX_dict` is automatically updated.
    """
    Opt_w.is2update = 0
    pass
    # Nothing to do, updating the CDS works automatically
    # Afterwards def update_draw(attr, old, new) is called

def chg_weight_links(attr, old, new):
    """
    Sets the number of hinges for the target slope weighting optimization.
    Calls :any:`get_weight_links`. Triggered by :any:`gui_opt_weight_link`.

    Parameters
    ----------
    attr : str
        Changed attribute (value).
    old : float
        Old number of hinges in target slope.
    new : float
        New number of hinges in target slope.

    Returns
    -------
    None.

    """
    ind = get_weight_links(int(new), SPLoverX)
    Opt_SPLoverX_dict.data.update(dict(SPL=np.array(SPLoverX.SPL)[ind], \
                                           x=SPLoverX.x[ind], \
                                           x_v=created_pal.xline[ind], \
                                           y_v=created_pal.yline[ind]))


def update_draw(attr, old, new):
    """
    Updates :any:`Opt_w`, :any:`Opt_SPLoverX_dict` and :any:`SPLoverX_optreg`,
    when :any:`optimal_SPLoverX` function was triggered. This happens if the
    target slope weighting is chosen and the user changes the target slope.
    Triggered by data change of :any:`Opt_SPLoverX_dict`.

    Parameters
    ----------
    attr : str
        Changed attribute (data).
    old : float
        Float point of old data point.
    new : float
        Float point of new data point (drawn by user in figure :any:`pSPLoverX`.

    Returns
    -------
    None.

    """
    # change values to smallest and highest real x values or move to nearest
    # grid point on x axis
    x         = SPLoverX_dict.data['x']
    x_v, y_v  = SPLoverX_dict.data['x_v'], SPLoverX_dict.data['y_v']
    Opt_w.CDS2obj(Opt_SPLoverX_dict, ['x', 'SPL', 'x_v', 'y_v'])
    Opt_w.CDS2obj(Opt_refpoint_dict, ['x_ref'])
    for ind, val in enumerate(Opt_w.x):
        arg = np.argmin(np.abs(np.array(x)-val))
        Opt_w.x[ind] = x[arg]
        Opt_w.x_v[ind], Opt_w.y_v[ind] = x_v[arg], y_v[arg]
    # sort CDS so that smaller x value is on indice 0
    Opt_w.resort_opt_region()           
    Opt_w.x_interp = list(x[np.argmin(np.abs(x-Opt_w.x[0])):np.argmin(np.abs(x-Opt_w.x[-1]))+1])
    Opt_w.interpolate_SPL()
    # find nearest index for x_ref
    ref_ind = np.argmin(np.abs(Opt_w.x_ref[0] - np.array(Opt_w.x_interp)))
    Opt_w.x_ref, Opt_w.ref_ind = Opt_w.x_interp[ref_ind], ref_ind   
    xv_ind = np.argmin(np.abs(np.array(x)-Opt_w.x_ref))
    #Opt_w.SPL -= Opt_w.SPL_interp[ref_ind]
    if Opt_w.is2update < 4:
        Opt_w.is2update += 1
        Opt_SPLoverX_dict.data.update(dict(x=Opt_w.x, SPL=Opt_w.SPL, \
                                           x_v=Opt_w.x_v, y_v=Opt_w.y_v))
        Opt_refpoint_dict.data.update(dict(x_ref=[Opt_w.x_ref], SPL=[Opt_w.SPL_interp[ref_ind]], \
                                           x_v=[x_v[xv_ind]], y_v=[y_v[xv_ind]]))#SPL=[Opt_w.SPL_interp[ref_ind]], \
    SPLoverX_optreg.data.update(dict(x=[Opt_w.x[0],Opt_w.x[0],Opt_w.x[-1],Opt_w.x[-1]], \
                                     y=[Opt_w.SPL[0]+100,Opt_w.SPL[-1]-100, \
                                        Opt_w.SPL[-1]-100,Opt_w.SPL[0]+100]))


def steps():
    """
    Function to place the widgets on the pages. Can also be called to update the
    a page after a user input, e.g., to make sure that range of a plot is surely
    updated.
    Clears the page with :py:class:`bokeh.io.curdoc().clear()`.The arrangement
    of the widgets is done with :py:class:`bokeh.layouts.row`,
    :py:class:`bokeh.layouts.column` and :py:class:`bokeh.layouts.gridplot`.
    The layout is then added to the root by :py:class:`bokeh.io.curdoc().add_root()`.
    Note: Tab bars could also be used, but would need to have all widgets
    available in curdoc() the whole time.

    Returns
    -------
    None.

    """
    curdoc().clear()
    # shows the select part in the document
    if gui_step.active == 0:
        row0 = row([column(aklogo,width=100), column(tublogo,width=170),\
                    column(aklogo2,width=270), column(gui_step,width=690)])
        row1 = row([headline], width=1230)
        row2 = row([help_tabs])
        figall = gridplot([[row0], [row1], [row2]], toolbar_location='right', \
                          merge_tools=False)      
        curdoc().add_root(figall)
    
    elif gui_step.active == 1:
        clear_result_fig()
        col0 = column([xAchse, start_point_x, end_point_x, strich1, \
                       line_discretization, pal_error_info], width=270)
        col1 = column([yAchse, start_point_y, end_point_y, strich6, \
                       save_line_button, remove_line_button, \
                       save_prev_line_button, remove_sel_line_button, \
                       check_PAL_button], width=270)
        col2 = column([row([p_equal_axis_dist, p_equal_axis]), p], width=690)
        row0 = row([column(aklogo, width=100), column(tublogo, width=170), \
                    column(aklogo2, width=270), column(gui_step, width=690)])
        row1 = row([headline], width=1230)
        row2 = row([col0, col1, col2])
        figall = gridplot([[row0],[row1],[row2]], toolbar_location='right', \
                          merge_tools=False)
        curdoc().add_root(figall)
        
    elif gui_step.active == 2:
        col0 = column([gui_psi_n, gui_Lambda_y], width=270 )
        col1 = column([gui_directivity, button_input, config_error_info], width=270)
        col2 = column([row([p_equal_axis_dist, p_equal_axis]), dir_tabs], width=690)
        row0 = row([column(aklogo, width=100), column(tublogo, width=170), \
                    column(aklogo2, width=270), column(gui_step, width=690)])
        row1 = row([headline], width=1230)
        row2 = row([col0, col1, col2])
        figall = gridplot([[row0],[row1],[row2]], toolbar_location='right', \
                          merge_tools=False)
        curdoc().add_root(figall)
    
    elif gui_step.active == 3:
        col0 = column([gui_N_LS, gui_x_H, gui_y_H, empty_s, strich1, \
                       gui_ref_array, gui_ref_discrete_angles, \
                       gui_ref_userdef], width=270)
        col1 = column([gui_Lambda_gap, gui_use_dangles, gui_discrete_angles, \
                       config_error_info, strich6, gui_ref_start, \
                       gui_ref_step_stop, ref_arr_table], width=270)
        col2 = column([row([p_equal_axis_dist, p_equal_axis]), p], width=690)
        row0 = row([column(aklogo,width=100), column(tublogo,width=170), \
                    column(aklogo2,width=270), column(gui_step,width=690)])
        row1 = row([headline], width=1230)
        row2 = row([col0, col1, col2])
        figall = gridplot([[row0],[row1],[row2]], toolbar_location='right', \
                          merge_tools=False)
        curdoc().add_root(figall)
              
    elif gui_step.active == 4:
        col0 = column([gui_PALC_alg, gui_gap_handling, gui_strength_smarg, \
                       strich5, row([gui_weighting, gui_opt_weight_link]), gui_weighting_nu, weighting_title, \
                       row([weighting_plus, weighting_minus, weighting_step_size]), \
                       strich6, gui_tolerance, gui_use_fixed_angles, \
                       gui_fixed_first_angle, fixed_angle_text], width=270)
        col1 = column([check_PAL_button, gui_start, gui_clear_result_fig, \
                       gui_download_result, config_error_info, results_table, gui_download_SPLoverX], \
                      width=270)
        col2 = column([row([p_equal_axis_dist, p_equal_axis]), SPLoverX_tabs], \
                      width=690)
        row0 = row([column(aklogo,width=100), column(tublogo,width=170), \
                    column(aklogo2,width=270), column(gui_step,width=690)])
        row1 = row([headline], width=1230)
        row2 = row([col0, col1, col2])
        figall = gridplot([[row0],[row1],[row2]], toolbar_location='right', \
                          merge_tools=False)
        curdoc().add_root(figall)
        
    elif gui_step.active == 5:
        row0 = row([column(aklogo,width=100), column(tublogo,width=170), \
                    column(aklogo2,width=270), column(gui_step,width=690)])
        row1 = row([headline], width=1230)
        row2 = row([res_tabs])
        figall = gridplot([[row0],[row1],[row2]], toolbar_location='right', \
                          merge_tools=False)
        curdoc().add_root(figall)
        

        
#%%##################### WIDGETS ##############################################
# PAGE 0: General Information and Help
help_tabs = help_text()

# PAGE 1: Draw The Polygonal Audience Line
start_point_x = TextInput(value="0", width=255, \
                          title="1. Start Position (x) of Audience Line in m")
"""TextInput widget of start point of audience lines in x-direction. Triggers 
   :any:`update_point`."""
start_point_y = TextInput(value="0", width=255, \
                          title="4. Start Position (y) of Audience Line in m")
"""TextInput widget of start point of audience lines in y-direction. Triggers
   :any:`update_point`."""
end_point_x = TextInput(value="0", width=255, \
                        title="2. End Position (x) of Audience Line in m")
"""TextInput widget of end point of audience lines in x-direction. Triggers
   :any:`update_point`."""
end_point_y = TextInput(value="0", width=255, \
                        title="5. End Position (y) of Audience Line in m")
"""TextInput widget of end point of audience lines in y-direction. Triggers
   :any:`update_point`."""
line_discretization = TextInput(value="1", width=255, \
                                title="3. Discretization: Number of Points per m")
"""TextInput widget to define the discretization in points per m of audience lines"""
save_line_button = Button(label="6. Save Current Audience Line", \
                          button_type="success", width=255)
"""Button widget to save audience lines. Triggers :any:`save_line`"""
remove_line_button = Button(label="7. Remove Last Audience Line", \
                            button_type="danger", width=255)
"""Button widget to remove audience lines. Triggers :any:`remove_line`"""
save_prev_line_button = Button(label="6. Add Line before Selected Line", \
                               button_type="success", visible=False, width=255)
"""Button widget to save a audience line before the selected line. Triggers
   :any:`create_prev_line`"""
remove_sel_line_button = Button(label="7. Remove Selected Audience Line", \
                                button_type="danger", visible=False, width=255)
"""Button widget to remove a audience line in front of the selected line. Triggers
   :any:`remove_sel_line`"""
check_PAL_button = Button(label="8. Get PALC Venue Proposal", \
                          button_type="warning", width=255)
"""Button widget to check the created venue slice. Triggers :any:`check_PAL`"""

# PAGE 2: Loudspeaker Configuration
gui_psi_n = TextInput(value="3.0", title="9. PALC Aperture Angle in deg", \
                      width=255, height=50)
"""TextInput widget to define the initial splay / aperture angle of the LSA
   cabinets. Triggers :any:`get_value`."""
gui_Lambda_y = TextInput(value="0.2", title="10. Loudspeaker Height in m", \
                         width=255)
"""TextInput widget to define the height of the used LSA cabinets. Triggers
   :any:`get_value`."""
gui_directivity = Select(title="11. Loudspeaker Directivity", value="Circular Piston", \
                         options=[("Circular Piston","Modeled: Circular Piston"), \
                                  ("Line Piston","Modeled: Line Piston"), \
                                  ("Combined Circular/Line", "Modeled: Combined Circular/Line"), \
                                  "Measured Loudspeaker Data"], width=255, height=50)
"""Select widget to choose a directivity of the LSA cabinets used in the Sound
   Field Prediction. Can be either Circular Piston, Line Piston, Comined Circular
   and Line Piston (x-over at 1.5 kHz) or Measured Loudspeaker Data. Triggers
   :any:`get_value`."""
button_input = FileInput(accept=".csv", visible=False)
"""FileInput widget to upload measured (complex) directivity data. Triggers
   :any:`upload_directivity`."""

# PAGE 3: Array Configuration
gui_N_LS = TextInput(value="4", title="12. Number of Loudspeaker Cabinets", \
                     width=255, height=50)
"""TextInput widget to enter the number of LS cabinets in LSA. Triggers
   :any:`get_value`."""
gui_x_H = TextInput(value="0", title="13. Uppermost Cabinet: x in m", \
                    width=255, height=50)
"""TextInput widget to enter the mounting point in x-direction (front position
   of the heighest LS cabinet. Triggers :any:`get_value`."""
gui_y_H = TextInput(value="2", title="14. Uppermost Cabinet: y in m", \
                    width=255, height=50)
"""TextInput widget to enter the mounting height of the LSA (top front of the
   highest LS cabinet. Triggers :any:`get_value`"""
gui_Lambda_gap = TextInput(value="0", title="15. Gap between Cabinets in m", \
                           width=255, height=50)
"""TextInput to define gaps between the LS cabinets. Triggers :any:`get_value`"""
gui_use_dangles = Select(title="16. Use Discrete Tilt Angles?", value="No", \
                         options=["No", "Yes"], width=255, height=50)
"""Select widget to choose if discrete tilt angles shall be used. Triggers
   :any:`choose_dangles`."""
gui_discrete_angles = TextInput(value='', disabled=True, width=255, height=50, \
                                title="17. Discrete Tilt Angles in deg (',': separator)")
"""TextInput widget to enter the discrete tilt angles. Triggers :any:`get_discrete_angles`."""
gui_ref_array = Select(title="Reference Array: Type", value="Straight", \
                       options=["Straight", "Progressive", "Arc", "User Defined"], \
                           width=255, height=50)
"""Select widget to choose the type of the reference LSA. Triggers
   :any:`reference_array_setting`"""
gui_ref_discrete_angles = Select(title="Reference Array: Use Angles of (17.)?", \
                                 value="No", options=["Yes", "No"], \
                                 disabled=True, visible=False, width=255, height=50)
"""Select widget to choose if discrete tilt angles of :any:`gui_discrete_angles`
   shall be used. Triggers :any:`get_ref_array_angles`."""
gui_ref_start = TextInput(value="0", width=255, height=50, \
                          title="Reference Array: Array Tilt Angle in deg")
"""TextInput widget to enter the angle of the heighest LSA cabinet in reference
   array.  Triggers :any:`get_ref_array_angles`."""
gui_ref_step_stop = TextInput(value="0", disabled=True, width=255, height=50, \
                              title="Reference Array: Inter Cabinet Angle in deg")
"""TextInput widget to enter the inter cabinet angle of the reference array.
   Triggers :any:`get_ref_array_angles`."""
gui_ref_userdef = TextAreaInput(value="", visible=False, cols=2, rows=3, width=255, height=70, \
                                title="User Defined Tilt Angles in deg (',': separator)")
"""TextAreaInput widget to enter user defined tilt angles of the reference array.
   Triggers :any:`get_ref_array_angles`."""

# PAGE 4: Algorithm Selection and Start Calculation
gui_PALC_alg = Select(title="18. PALC-Algorithm", value="PALC 2", \
                      options=["PALC 1", "PALC 2", "PALC 3"], width=255, height=50)
"""Select widget to choose the type of the algorithm. Either PALC1 (Splay angles
   are equal), PALC2 (Splay angles depend on distance to receiver), PALC3
   Similar to PALC2, tan of splay angles is used). Triggers :any:`get_value`."""    
gui_gap_handling = Select(title="19. Gap Handling Approach", value="Without", \
                          options=["Without", "Hard Margin", "Soft Margin"], \
                          width=255, height=50)
"""Select widget to choose the gap handling approach. Either Without (No gap
   handling), Hard Margin (based on assumed splay angles, gaps are totally avoided),
   Soft Margin (gaps are lower weighted). Triggers :any:`get_value`."""
gui_strength_smarg = Slider(title="20. Strength of Soft Margin", value=1.0, \
                            start=0.1, end=1.9, step=0.05, disabled=True, width=255)
"""Slider widget to set the strength of soft margin approach, if chosen in 
   :any:`gui_gap_handling`. Triggers :any:`get_value`."""
gui_weighting = Select(title="21. Weighting", value="Without", \
                       options=["Without", "Linear Spacing", "Logarithmic Spacing", "Target Slope"], \
                       width=255, height=50)
"""Select widget to choose the weighting approach. Triggers :any:`weighting_select`."""
gui_opt_weight_link = Select(title="Hinges in Target", value = "0", \
                             options=["0", "1", "2"], width=105, height=50, visible=False)
"""Select widget to choose number of hinges in target slope when 'Target Slope'
   is chosen in :any:`gui_weighting`. The number of choosable hinges depends on
   the number of audience lines."""
# cubic spaced slider: when used, function store_weighting_nu must be adapted (input with the power of 3)
gui_weighting_nu = Slider(title="Weighting Parameter ν", value=np.cbrt(1.00), \
                          start=np.cbrt(0.01), end=np.cbrt(10.00), step=0.01, disabled=True, width=255, \
                          format=FuncTickFormatter(code="return Math.pow(tick,3).toFixed(2)"))
"""Slider widget to set the strength of the weighting if 'Linear Spacing' or
   'Logarithmic Spacing' is chosen in :any:`gui_weighting`. Triggers
   :any:`store_weighting_nu`."""
# linear spaced slider of the weighting
# gui_weighting_nu = Slider(title="Weighting Parameter ν", value=1.00, \
#                           start=0.01, end=10.00, step=0.01, disabled=True, width=255)
weighting_title = Paragraph(text="22. Adjustment of ν and Step Size", width=255, height=20)
weighting_plus = Button(label="+", button_type="success", disabled=True, width=70)
"""Button widget that increases the weightings strength by the value of
   :any:`weighting_step_size`."""
weighting_minus = Button(label="-", button_type="danger", disabled=True, width=70)
"""Button widget that decreases the weightings strength by the value of
   :any:`weighting_step_size`."""
weighting_step_size = Select(value='0.05', options=['0.01', '0.05', '0.10', '0.50'], \
                             disabled=True, width=80, height=30)
"""Select widget to set the step size of :any:`weighting_plus` and
   :any:`weighting_minus`. Triggers :any:`update_weighting_step_size`."""
gui_tolerance = Slider(title="23. Tolerance (Abort Criterion) in m", \
                       value=1.0, start=0.05, end=5, step=0.05, width=255)
"""Slider widget to set the tolerance, which is the break out condition of the
   PALC computation. It is the distance of covered region to the first audience
   position in m. Triggers :any:`get_value`."""
gui_use_fixed_angles = Select(title="24. Fix Angle of Uppermost Cabinet?", \
                              value="No", options=["No", "Yes"], disabled=True, \
                                  width=255, height=50)
"""Select widget to choose if the heighest LSA cabinet shall be fixed on a
   specific angle."""
gui_fixed_first_angle = TextInput(title="25. Angle of Uppermost Cabinet in deg", \
                                  value="0.0", disabled=True, width=255)
"""TextInput widget to enter the fixed angle of the heighest LSA cabinet.
   Triggers :any:`get_value`."""
fixed_angle_text = Paragraph(text='Max: ' + str(Opt_arr.fixed_angle[0]) + ', Min: ' + str(Opt_arr.fixed_angle[2]))

gui_start = Toggle(label="26. PALC Run", button_type="success", \
                   disabled=True, width=255, active=False)
"""Toggle widget to start the PALC and SFP computation."""
gui_clear_result_fig = Button(label="27. Delete Result Visualization", \
                              button_type="danger", width=255)
"""Button widget to clear the plotted venue slice with LSA and rays. Triggers
   :any:`clear_result_fig`"""
gui_download_result = Button(label="Download PALC Angles", button_type="success", width=255)
"""Button widget to download the PALC tilt angles in .csv format"""
gui_freq_range = RangeSlider(title="Frequency Range", width=690, \
                             start=0, end=20000, value=(200,8000), step=20)
gui_download_SPLoverX = Button(label="Download SPL", button_type="success", width=255)

# PAGE 5: Results Visualization
gui_f_beam = Select(title="Frequency in Hz", value="63", width=200, \
                    options=["63", "125", "250", "500", "1000", "2000", "4000", "8000", "16000"])
"""Select widget to choose the plotted frequency in the beamplot. Triggers
   :any:`get_beamplot`."""
gui_which_beam = CheckboxGroup(labels=["PALC", "Reference"], active=[], \
                               orientation='horizontal', height=11, inline=True, \
                               disabled=True)
"""CheckboxGroup widget to choose wheter PALC and/or Reference results shall
   be shown as a beamplot. Triggers :any:`get_beamplot`."""

# HEADER and GENERAL: AK Logo, RadioGroup steps, errors and warnings, etc.
# AK Logo
x_range, y_range = (10,30), (10,30)
aklogo = figure(x_range=x_range, y_range=y_range, plot_width=80, plot_height=80, min_border=0)
aklogo.toolbar.logo, aklogo.axis.visible        = 'grey', False
aklogo.outline_line_color, aklogo.grid.visible = None, False
aklogo.toolbar_location, aklogo.toolbar.logo = None, 'grey'
IMPATH = os.path.join(f"{appname}/static/logo_grau-schwarz.png")
aklogo.image_url(url=[IMPATH], x=x_range[0], y=y_range[1]-0.3, \
                 w=x_range[1]-(x_range[0]+1), h=y_range[1]-(y_range[0]+0.6))
aklogo2 = Div(text='<b>A Software Tool by the <br><big><a href="https://www.ak.tu-berlin.de" target="_blank">Audio Communication Group</a></big></b>',width=270, height=60)
tublogo = figure(x_range=x_range, y_range=y_range, plot_width=int(80*1.786), \
                 plot_height=80, min_border=0)
tublogo.toolbar.logo, tublogo.axis.visible        = 'grey', False
tublogo.outline_line_color, tublogo.grid.visible = None, False
tublogo.toolbar_location, tublogo.toolbar.logo = None, 'grey'
IMPATHtub = os.path.join(f"{appname}/static/TUB.png")
tublogo.image_url(url=[IMPATHtub], x=x_range[0]+0.5, y=y_range[1]-0.3, \
                  w=x_range[1]-(x_range[0]+1), h=y_range[1]-(y_range[0]+0.6))
gui_step = RadioButtonGroup(labels=["Help","1. PAL", "2. Loudspeaker Configuration", \
                                    "3. Array Configuration", "4. Algorithm", \
                                    "5. Sound Field Prediction"], \
                            active=1, button_type="primary", width=680)
"""RadioButtonGroup widget to switch between the different pages."""
p_equal_axis = RadioGroup(labels=['Equal (variable figure height)', \
                                  'Equal (const. figure height)', 'Stretched'], \
                          active=2, height=15, margin=5, orientation='horizontal', \
                          inline=True)
"""RadioGroup widget to set the scaling of the venue slice plot. Triggers
   :any:`scale_p_axis`."""
p_equal_axis_dist = Div(text='      <b> Venue Slice Axes: </b>', height=15, width=150)
p_which_beamplot = Div(text='      <b> Beam Plot of: </b>', height=10, width=120)

pal_error_info = Div(text="", width=250, height=300)
config_error_info = Paragraph(text="", width=250, height=50)
strich1 = Paragraph(text="______________________________________")
strich5 = Paragraph(text="___________________________________")
strich6 = Paragraph(text="___________________________________")
headline = Paragraph(text="_____________________________________________________________________________________________________________________________________________________________________________")
#strichSFP = Paragraph(text="________________________________________________________________________________________________________________________________________________________________________________________")
empty_s   = Paragraph(text="", height=50)
yAchse = Div(text="<b>y-Axis</b>")
xAchse = Div(text="<b>x-Axis</b>")

#%%######### Callbacks of the Widgets #########################################
# PAGE 1: Draw The Polygonal Audience Line
start_point_x.on_change("value", update_point)
start_point_y.on_change("value", update_point)
end_point_x.on_change("value", update_point)
end_point_y.on_change("value", update_point)
save_line_button.on_click(save_line)
remove_line_button.on_click(remove_line)
save_prev_line_button.on_click(create_prev_line)
remove_sel_line_button.on_click(remove_sel_line)
check_PAL_button.on_click(check_PAL)

# PAGE 2: Loudspeaker Configuration
gui_psi_n.on_change("value", get_value)
gui_Lambda_y.on_change("value", get_value)
gui_directivity.on_change("value", get_value)
button_input.on_change("value", upload_directivity)

# PAGE 3: Array Configuration
gui_x_H.on_change("value", get_value)
gui_y_H.on_change("value", get_value)
gui_N_LS.on_change("value", get_value)
gui_Lambda_gap.on_change("value", get_value)
gui_use_dangles.on_change("value", choose_dangles)
gui_discrete_angles.on_change("value", get_discrete_angles)
gui_discrete_angles.on_change("value", lambda attr, old, new: get_ref_array_angles())
gui_ref_array.on_change("value", reference_array_setting)
gui_ref_start.on_change("value", lambda attr, old, new: get_ref_array_angles())
gui_ref_step_stop.on_change("value", lambda attr, old, new: get_ref_array_angles())
gui_ref_discrete_angles.on_change("value", lambda attr, old, new: get_ref_array_angles())
gui_ref_userdef.on_change("value", lambda attr, old, new: get_ref_array_angles())

# PAGE 4: Algorithm Selection and Start Calculation
gui_PALC_alg.on_change("value", get_value)
gui_gap_handling.on_change("value", get_value)
gui_strength_smarg.on_change("value", get_value)
gui_weighting.on_change("value", weighting_select)
gui_opt_weight_link.on_change("value", chg_weight_links)
gui_weighting_nu.on_change("value", store_weighting_nu)
weighting_plus.on_click(adjust_weighting_plus)
weighting_minus.on_click(adjust_weighting_minus)
weighting_step_size.on_change("value", update_weighting_step_size)
gui_tolerance.on_change("value", get_value)
gui_use_fixed_angles.on_change("value", use_fixed_angle)
gui_fixed_first_angle.on_change("value", get_value)
gui_start.on_click(start_calc)
gui_clear_result_fig.on_click(clear_result_fig)
Opt_SPLoverX_dict.on_change('data', update_draw)
Opt_refpoint_dict.on_change('data', update_draw)
gui_download_result.js_on_click(CustomJS(args=dict(source=results_dict), \
                            code=open(os.path.join(os.path.dirname(__file__), \
                                                   "ext/download.js")).read()))
gui_freq_range.on_change("value", get_value)
gui_download_SPLoverX.js_on_click(CustomJS(args=dict(source=SPLoverX_dict), \
                            code=open(os.path.join(os.path.dirname(__file__), \
                                                   "ext/download.js")).read()))
gui_download_SPLoverX.js_on_click(CustomJS(args=dict(source=Opt_SPLoverX_dict), \
                            code=open(os.path.join(os.path.dirname(__file__), \
                                                   "ext/download.js")).read()))
gui_download_SPLoverX.js_on_click(CustomJS(args=dict(source=Opt_refpoint_dict), \
                            code=open(os.path.join(os.path.dirname(__file__), \
                                                   "ext/download.js")).read()))
gui_download_SPLoverX.js_on_click(CustomJS(args=dict(source=SPLoverX_ref_dict), \
                            code=open(os.path.join(os.path.dirname(__file__), \
                                                   "ext/download.js")).read()))
gui_download_SPLoverX.js_on_click(CustomJS(args=dict(source=Homogeneity_dict), \
                            code=open(os.path.join(os.path.dirname(__file__), \
                                                   "ext/download.js")).read()))

# PAGE 5: Results Visualization
gui_f_beam.on_change("value", get_beamplot)
gui_which_beam.on_change("active", get_beamplot)

# HEADER: Switch between the Pages
gui_step.on_change('active', lambda attr, old, new: steps())
p_equal_axis.on_change('active', lambda attr, old, new: scale_p_axis())


#%%################### PLOTTING ###############################################
# frequency axis ticker
f_ticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
f_ticks_override = {20: '0.02', 50: '0.05', 100: '0.1', 200: '0.2', 500: '0.5', \
                    1000: '1', 2000: '2', 5000: '5', 10000: '10', 20000: '20'}
###### Venue Slice: p (PAGE 1,2,3,4) ######
ptools = "box_zoom,pan,wheel_zoom,reset,tap,hover,save"
p = figure(title="", x_axis_label="x in m", y_axis_label="y in m", \
           toolbar_sticky=False, toolbar_location="right", tools=ptools, \
           frame_height=478, frame_width=637)
"""Venue Slice Plot"""
p.toolbar.logo = None
# Venue Slice
p.xaxis.axis_label_text_font_style = "normal"
p.yaxis.axis_label_text_font_style = "normal"
#p.x(2,2, color='#323030')
p.x('x','y', source=guipal, line_color="lightskyblue") #blue 
pal_line = p.multi_line('xlineplt', 'ylineplt', source=guilineplt, \
                        legend_label='PALC and SFP', line_dash='solid', \
                        line_color="lightskyblue", line_width=2, \
                        selection_color="crimson", nonselection_line_alpha=0.5)#lightskyblue
nal_line = p.multi_line('xnalplt', 'ynalplt', source=guinalplt, \
                        legend_label='PALC (non-audience line)', \
                        line_dash='dashed', line_color="lime", line_width=2, \
                        nonselection_line_alpha=0.5) # firebrick, lime
sfp_line = p.multi_line('xsfpplt', 'ysfpplt', source=guisfpplt, \
                        legend_label='only SFP', line_dash='solid', \
                        line_color="orange", line_width=2, \
                        nonselection_line_alpha=0.5) # orange
p.patches('x_list','y_list', source=plot_patches, color="white") # white

 # Array and its Rays
p.multi_line('x_c_n_unitn', 'y_c_n_unitn', source=plt_venue_slice_1, \
             color="white", line_dash="solid") # black
p.multi_line('x_c_n_unitn_psi1', 'y_c_n_unitn_psi1', source=plt_venue_slice_1, \
             color="red", line_dash="dashed") # red
p.multi_line('x_c_n_unitn_psi2', 'y_c_n_unitn_psi2', source=plt_venue_slice_1, \
             color="green", line_dash="dashed") # green
p.x('seg_pos_x', 'seg_pos_y', source=plt_venue_slice_2, color="white") # black
p.x('seg_pos_start_x', 'seg_pos_start_y', source=plt_venue_slice_2, color="red") # red
p.x('seg_pos_stop_x', 'seg_pos_stop_y', source=plt_venue_slice_2, color="green") # green
# TapTool Callback for user interaction and Legend
guilineplt.selected.on_change('indices', audience_line_tap)
# Set legend attributes
p.legend.location, p.legend.orientation, p.legend.margin = 'top_center', 'horizontal', 0
p.legend.background_fill_alpha, p.legend.spacing, p.legend.visible = 0.5, 15, False
# export stuff for poster
#p.background_fill_color, p.border_fill_color = None, None
#p.output_backend = "svg"

# p.ygrid.minor_grid_line_alpha, p.xgrid.minor_grid_line_alpha = 0,0

###### Directivity (PAGE 2) ######
pdirtools = "box_zoom,pan,wheel_zoom,reset,hover,save"
pdirtips = [("Index","$index"),("Amplitude in dB","$y"),("Angle in deg","$x")]
pdir = figure(title="Loudspeaker Directivity for Selected Frequencies", \
              toolbar_sticky=False, tools=pdirtools, tooltips=pdirtips, \
              x_axis_label="Angle in deg", y_axis_label="Amplitude in dB", \
              x_range=Range1d(-92,92), height=550, width=690)
"""Directivity Plot of 100 Hz, 1 kHz and 10 kHz to check used directivity in
   SFP and / or uploaded measured loudspeaker data."""
pdir.toolbar.logo = None
pdir.line('degree', 'amplitude_100', source=directivity_dict_plt, \
          legend_label='100 Hz', color='deepskyblue') # black
pdir.line('degree', 'amplitude_1k', source=directivity_dict_plt, \
          legend_label='1 kHz', color='lime') # red
pdir.line('degree', 'amplitude_10k', source=directivity_dict_plt, \
          legend_label='10 kHz', color='orange') #orange
pdir.legend.title = "Frequencies"
pdir.legend.location = "bottom_right"
pdir.legend.background_fill_alpha = 0.5
pdir.xaxis.ticker = [-90, -60, -30, 0, 30, 60, 90]

###### SPL over distance: pSPLoverX (PAGE 4) ######
pSPLoverXtips = [("Title", "$name"),("Index","$index"),("SPL in dB", "$y"), \
                 ("Distance", "$x"), ("(x,y) of Venue Slice in m", "@x_v, @y_v")]
pSPLoverXtools = "box_zoom,pan,wheel_zoom,reset,hover,save, tap"
pSPLoverX = figure(title="SPL over Distance (maximum SPL normalized to 0 dB)", \
                   x_axis_label="Distance LSA to Receiver in m", \
                   y_axis_label="SPL change in dB", toolbar_sticky=False, \
                   tools=pSPLoverXtools,tooltips=pSPLoverXtips, \
                   y_range=(-15,0.3), height=550, width=690) #SPL over Distance (maximum SPL normalized to 0 dB), width=690
"""Sound Pressure over Distance Plot. If Target Slope weighting is chosen, the
   target slope can be adjusted within this figure."""
pSPLoverX.toolbar.logo = None
pSPLoverX.line('x', 'SPL', source=SPLoverX_dict, color="lightskyblue", name="PALC", \
               legend_label='PALC (optimized)') #lightskyblue
# pSPLoverX.circle('x', 'SPL', source=SPLoverX_dict, line_color="white", \
#                  fill_color="black", line_width=2, name="PALC", \
#                  legend_label='PALC (optimized)')
pSPLoverX.line('x', 'SPL', source=SPLoverX_ref_dict, color="orange", name="Reference", \
               legend_label='Reference LSA') #orange
Opt2 = pSPLoverX.line('x', 'SPL', source=Opt_SPLoverX_dict, color="lime", \
                      name="Reference", legend_label='Target Slope') #lime
Opt1 = pSPLoverX.circle('x', 'SPL', source=Opt_SPLoverX_dict, line_color="yellow", \
                        fill_color="black", line_width=2, name="Reference", \
                        legend_label='Target Slope')
Opt_ref = pSPLoverX.circle('x_ref','SPL', source=Opt_refpoint_dict, color="darkred", \
                           fill_color="red", size=10, name="Reference Point", \
                           legend_label="Reference Point")
pSPLoverX.patch('x', 'y', source=SPLoverX_optreg, alpha=0.2, line_width=0.1)
pSPLoverX.legend.location, pSPLoverX.legend.orientation = 'top_right', 'horizontal'
pSPLoverX.legend.margin, pSPLoverX.legend.glyph_width = 0, 30
pSPLoverX.legend.background_fill_alpha, pSPLoverX.legend.spacing = 0.5, 15

Opt_SPLoverX_dict.selected.on_change('indices', optimal_SPLoverX)
Opt_refpoint_dict.selected.on_change('indices', optimal_SPLoverX)
optSPLtool = PointDrawTool(renderers=[Opt1, Opt2, Opt_ref], add=False)
pSPLoverX.add_tools(optSPLtool)
pSPLoverX.toolbar.active_tap = optSPLtool

# export stuff for poster
pSPLoverX.background_fill_color, pSPLoverX.border_fill_color = None, None
pSPLoverX.output_backend = "svg"

####### Venue Slice with Colorbar and TapTool: pSFPv (PAGE 5) ######
pSFPvtooltips = [("Index", "$index"),("(x,y)", "(@x_vert, @y_vert)"),]
pSFPvtools = "box_zoom,pan,wheel_zoom,reset,tap,hover,save"
pSFPv = figure(title="28. Venue Slice Positions", x_axis_label="x in m", \
               y_axis_label="y in m", tools=pSFPvtools, tooltips=pSFPvtooltips, \
               width=1220, height=300)
"""Venue Slice Plot of SFP, shows the discrete points of the venue slice in
   color of the frequency responses in :any:`pSFP`."""
pSFPv.toolbar.logo = None
pSFPv.patches('x_list','y_list', source=plot_patches, color="white") #black
# TapTool for user interaction
SFPv_dict.selected.on_change('indices', tap_handle)

###### Sound Field Prediction PALC: pSFP (PAGE 5) ######
pSFPtools = "box_zoom,pan,wheel_zoom,reset,tap,hover,save"
pSFPtooltips = [("Index", "$index"),("Frequency in Hz", "$x"),("SPL in dB", "$y")]
pSFP = figure(title="29. SPL at all Audience Positions: PALC", \
              x_axis_type="log", x_range=(15,24000), y_range=(0, 100), \
              x_axis_label="f in kHz", y_axis_label="SPL in dB", \
              tools=pSFPtools, toolbar_sticky=False, \
              tooltips=pSFPtooltips, width=610) #29. SPL at all Audience Positions: PALC
"""Frequency Response Plot of PALC results of each discrete point on venue
   slice. Color of the responses belongs to the discrete points in figure
   :any:`pSFPv`."""
pSFP.toolbar.logo = None
pSFP.xaxis.ticker = f_ticks
pSFP.xaxis.major_label_overrides = f_ticks_override

# export stuff for poster
#pSFP.background_fill_color, pSFP.border_fill_color = None, None
#pSFP.output_backend = "svg"

###### Sound Field Prediction straight array: pSFPref (PAGE 5) ######
pSFPreftools = "box_zoom,pan,wheel_zoom,reset,tap,hover,save"
pSFPreftooltips = [("Index", "$index"),("Frequency in Hz", "$x"),("SPL in dB", "$y")]
pSFPref = figure(title="30. SPL at all Audience Positions: Reference Array", \
                 x_axis_type="log",x_range=(15,24000), y_range=(0, 100), \
                 x_axis_label="f in kHz", y_axis_label="SPL in dB", \
                 tools=pSFPreftools, toolbar_sticky=False, \
                 tooltips=pSFPreftooltips, width=610) #30. SPL at all Audience Positions: Reference Array"
pSFPref.toolbar.logo = None
pSFPref.xaxis.ticker = f_ticks
pSFPref.xaxis.major_label_overrides = f_ticks_override

# export stuff for poster
#pSFPref.background_fill_color, pSFPref.border_fill_color = None, None
#pSFPref.output_backend = "svg"

###### Homogeneity PALC and reference array: pHom (PAGE 5) ######
pHomtools = "box_zoom,pan,wheel_zoom,reset,hover,save"
pHomtooltips = [("Frequency in Hz", "$x"),("SPL in dB", "$y")]
pHom = figure(title="31. Homogeneity", x_axis_type="log", \
              x_axis_label="f in kHz", y_axis_label="H(f) in dB", \
              tools=pHomtools, tooltips=pHomtooltips, x_range=(15,24000), \
              width=1220) #31. Homogeneity
"""Frequency Response Plot of reference array of each discrete point on venue
   slice. Color of the responses belongs to the discrete points in figure
   :any:`pSFPv`."""
pHom.toolbar.logo = None
pHom.xaxis.ticker = f_ticks
pHom.xaxis.major_label_overrides = f_ticks_override
pHom.line('f','H', source=Homogeneity_dict, color="lightskyblue", \
          legend_label="PALC") # lightskyblue
pHom.line('f','H_dist_high', source=Homogeneity_dict, color="navajowhite", \
          line_dash="dashed") #navajowhite
pHom.line('f','H_dist_low', source=Homogeneity_dict, color="navajowhite", \
          line_dash="dashed") # navajowhite
pHom.line('f', 'H_str', source=Homogeneity_dict, color="orange", \
          legend_label="Reference Array") #orange
pHom.legend.location = "top_left"

# export stuff for poster
#pHom.background_fill_color, pHom.border_fill_color = None, None
#pHom.output_backend = "svg"

###### Beamplot PALC: pBeam (PAGE 5) ######
pBeamtips = [("SPL in dB", "@SPL"),("(x in m, y in m)", "($x, $y)"),]
pBeamtools = "box_zoom,pan,wheel_zoom,reset,hover,save"
pBeam = figure(title="34. Beam Plot PALC: SPL in dB", x_axis_label="x in m", \
               y_axis_label="y in m", frame_width=1110, match_aspect=True, \
               visible=False, tools=pBeamtools, tooltips=pBeamtips) #34. Beam Plot PALC: SPL in dB
"""Beamplot of PALC results"""
pBeam.toolbar.logo = None
pBeam.x_range.range_padding = pBeam.y_range.range_padding = 0
mapper = LinearColorMapper(palette=Cividis256, low=50, high=100)
color_bar = ColorBar(color_mapper=mapper, location=(0,0), \
                     title="", title_text_font_style="normal")
pBeam.add_layout(color_bar, 'right')
pBeam.image(image='SPL',x='x', y='y', dw='dw', dh='dh', \
            source=Beamplot_dict, palette=Cividis256)
pBeam.multi_line('xlineplt', 'ylineplt', source=guilineplt, \
                 line_dash='solid', line_color="black", line_width=4, \
                 selection_color="darkred")
pBeam.patches('x_list','y_list', source=plot_patches, color="black")

pBeamreftips = [("SPL in dB", "@SPL_ref"),("(x in m, y in m)", "($x, $y)"),]
pBeamreftools = "box_zoom,pan,wheel_zoom,reset,hover,save"
pBeam_ref = figure(title="34. Beam Plot Reference: SPL in dB", \
                   x_axis_label="x in m", y_axis_label="y in m", \
                   frame_width=1110, match_aspect=True, visible=False, \
                   tools=pBeamreftools, tooltips=pBeamreftips) #34. Beam Plot Reference: SPL in dB
"""Beamplot of reference array"""
pBeam_ref.toolbar.logo = None
pBeam_ref.x_range.range_padding = pBeam.y_range.range_padding = 0
mapper_ref = LinearColorMapper(palette=Cividis256, low=50, high=100)
color_bar_ref = ColorBar(color_mapper=mapper_ref, location=(0,0), \
                     title_standoff=10, title="", title_text_font_style="normal")
pBeam_ref.add_layout(color_bar_ref, 'right')
pBeam_ref.image(image='SPL_ref',x='x', y='y', dw='dw', dh='dh', \
                source=Beamplot_dict, palette=Cividis256)
pBeam_ref.multi_line('xlineplt', 'ylineplt', source=guilineplt, \
                     line_dash='solid', line_color="black", line_width=4, \
                     selection_color="darkred")
pBeam_ref.patches('x_list_ref','y_list_ref', source=plot_patches, color="black")
pBeam.xaxis.axis_label_text_font_style = "normal"

###### Bar Plots: pBar, pBarref (PAGE 5) ######
# Bar Chart PALC: pBar
pBar = figure(title="32. Bar Chart: PALC", x_axis_label="SPL in dB", \
              y_axis_label="Number of Positions", width=610)
"""Barplot of PALC results. Number of Positions that contain SPL values on
   y-axis."""
pBar.vbar(x='x', top='top', source=Bar_dict, bottom=0, width=0.9, color="firebrick")
pBar.toolbar.logo = None
# Bar Chart straight array: pBarsref
pBarref = figure(title="33. Bar Chart: Reference Array", x_axis_label="SPL in dB", \
                 y_axis_label="Number of positions", width=610)
"""Barplot of reference array. Number of Positions that contain SPL values on
   y-axis."""
pBarref.vbar(x='x_ref', top='top_ref', source=Bar_dict_ref, bottom=0, \
             width=0.9, color="firebrick")
pBarref.toolbar.logo = None

###### Tab-Plots: Venue Slice - Directivity (PAGE 2) and Venue Slice - SPLoverX (PAGE 4) ######
# Venue Slice Plot and LS Directivity (PAGE 2)
dir_tab1 = Panel(child=p, title="Venue Slice")
dir_tab2 = Panel(child=pdir, title="Loudspeaker Directivity")
dir_tabs = Tabs(tabs=[dir_tab1, dir_tab2])
# Venue Slice Plot and SPL over distance (PAGE 4)
SPLoverX_tab1 = Panel(child=p, title="Venue Slice")
SPLoverX_tab2 = Panel(child=column([gui_freq_range, pSPLoverX]), title="SPL over Distance")
SPLoverX_tabs = Tabs(tabs=[SPLoverX_tab1, SPLoverX_tab2])
# Result Tabs (PAGE 5)
res_tab1 = Panel(child=column([pSFPv, row([pSFP, pSFPref])]), \
                 title="Frequency Responses at Audience Positions")
res_tab2 = Panel(child=pHom, title="Homogeneity")
res_tab3 = Panel(child=row([pBar, pBarref]), title="Bar Chart")
res_tab4 = Panel(child=column([row([gui_f_beam, column([p_which_beamplot, gui_which_beam])]), \
                               pBeam, pBeam_ref]), title="Beam Plots")
res_tabs = Tabs(tabs=[res_tab1, res_tab2, res_tab3, res_tab4])

###### Set up values before showing on Data Table
PALC_config.store_config('N', gui_N_LS.value)
###### DATA TABLE of Resulting PALC Calculcation (PAGE 4) ###### 
results_columns = [TableColumn(field="a_num_LS", title="LS", width=20), 
                   TableColumn(field="b_gamma_tilt_deg", title="Total Angle", width=80),  
                   TableColumn(field="c_gamma_tilt_deg_diff", title="ICA", width=80)] 
results_table = DataTable(source=results_dict, columns=results_columns, \
                          index_position=None, width=250, height=700, \
                          fit_columns=False)

###### DATA TABLE of Reference Array
ref_arr_columns = [TableColumn(field="LS", title="LS", width=20), \
                   TableColumn(field="gamma_ref", title="Tilt Angle in deg", width=140)]
ref_arr_table   = DataTable(source=gamma_ref_dict, columns=ref_arr_columns, \
                            index_position=None, width=170, fit_columns=False)

#%%######### set up the bokeh document in browser for the first run ###########
col0 = column([xAchse, start_point_x, end_point_x, strich1, line_discretization, pal_error_info], width=270)
col1 = column([yAchse, start_point_y, end_point_y, strich6, save_line_button, remove_line_button, \
               save_prev_line_button, remove_sel_line_button, check_PAL_button], width=270)
col2 = column([row([p_equal_axis_dist, p_equal_axis]), p], width=690)
row0 = row([column(aklogo,width=100), column(tublogo,width=170), column(aklogo2,width=270), column(gui_step,width=690)])
row1 = row([headline], width=1230)
row2 = row([col0, col1, col2])
figall = gridplot([[row0],[row1],[row2]], toolbar_location='right', merge_tools=False)
curdoc().add_root(figall)

########################## END MAIN ##########################################
