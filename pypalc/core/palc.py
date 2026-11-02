"""The PALC algorithm.

Configuration and result classes of the PALC algorithm, the main routine
:func:`calc_angles` and the gap handling wrapper :func:`calcPALC`
(formerly ``fun/calcPALC.py``, ``calc_angles`` from
``fun/PALC_functions.py``, the configuration and array classes from
``fun/PALC_classes.py`` and ``get_fixed_angle_borders`` from
``fun/gui_algorithm_sfp.py``).

author: Arne Hoelter
"""
from __future__ import annotations

import logging
import time
from copy import copy

import numpy as np

from .geometry import (LSA_visualization, calc_gamma, calc_intersection_PAL,
                       source_pos)

logger = logging.getLogger(__name__)


class PALC_configuration():
    """ Class to create a configuration object of the PALC algorithm. """
    def __init__(self, Lambda_gap=0, Lambda_y=0.2, x_H=0, y_H=2, constraint='PALC 2', use_gamma_LSA=False, gamma_LSA=None, \
                 use_weighting='Without', weighting_nu=1, weighting_factors=None, N=int(4), gap_handling='Without', strength_sm=1, \
                 last_angle_hm=None, gap_weights=None, weighting_weights=None, use_fixed_angle=False, fixed_angle=0, \
                 directivity = 'Circular Piston', psi_n=None, gamma_n=None, tolerance=1.0, freq_range=(200,8000)):
        # None stands for a new list with the default value (no shared mutable defaults)
        gamma_LSA, weighting_factors, last_angle_hm, gap_weights, weighting_weights = \
            [[] if val is None else val for val in
             (gamma_LSA, weighting_factors, last_angle_hm, gap_weights, weighting_weights)]
        psi_n = [0.0524] if psi_n is None else psi_n
        gamma_n = [0] if gamma_n is None else gamma_n
        self.x_H, self.y_H , self.Lambda_gap, self.Lambda_y = x_H, y_H, Lambda_gap, Lambda_y
        self.tolerance, self.psi_n, self.gamma_n            = tolerance, psi_n, gamma_n
        self.constraint, self.N, self.directivity           = constraint, N, directivity
        self.use_weighting, self.weighting_nu               = use_weighting, weighting_nu
        self.weighting_factors, self.weighting_weights      = weighting_factors, weighting_weights
        self.gap_weights, self. gap_handling                = gap_weights, gap_handling
        self.strength_sm, self.last_angle_hm                = strength_sm, last_angle_hm
        self.use_fixed_angle, self.fixed_angle              = use_fixed_angle, fixed_angle
        self.use_gamma_LSA, self.gamma_LSA                  = use_gamma_LSA, gamma_LSA
        self.freq_range                                     = freq_range


    def store_config(self, *new_input):
        """
        Writes configuration input to the attributes of the object.

        Parameters
        ----------
        *new_input : list
            First entry contains a list of attributes to update / store.
            Second entry contains the list with corresponding data / values.

        Returns
        -------
        str
            Error message if storage failed.
        str
            Info message if storage failed.

        """
        # convert to PALC compatible dtypes
        if new_input[0] in ['x_H', 'y_H', 'fixed_angle', 'Lambda_gap', 'Lambda_y', 'tolerance']:
            try:
                val = float(new_input[1])
            except (TypeError, ValueError):
                return 'Error: only numbers are allowed. Set to Default', new_input[0]
        elif new_input[0] in ['N', 'psi_n']:
            try:
                if new_input[0] in ['N']: 
                    val = int(new_input[1])
                if new_input[0] in ['psi_n']:
                    val = self.N
                    self.psi_n = [float(new_input[1])*np.pi/180]
                self.psi_n = np.linspace(self.psi_n[0], self.psi_n[0], num=val)
                self.gamma_n = np.linspace(-0.0428, 0.7147, num=val)
                self.weighting_factors = np.ones(val)
                # more than one LS necessary and maximum of 30
                if val == 0: return 'Error: number of LS must be bigger than 0. Set to default', new_input[0]
                if val > 30: return 'Error: maximum amount of LS is 30. Set to default', new_input[0]
                if new_input[0] in ['psi_n']: return '', None
            except (TypeError, ValueError):
                return 'Error: only numbers are allowed. Set to default', new_input[0]
        elif new_input[0] in ['strength_sm', 'fixed_angle']:
            val = float(new_input[1])
        
        if hasattr(self, new_input[0]) and 'val' in locals():
            setattr(self, new_input[0], val)
        elif hasattr(self, new_input[0]):
            setattr(self, new_input[0], new_input[1])
        else:            
            logger.error('PALC_config has no attribute %s', new_input[0])
        
        # array must be smaller than y_H
        if self.N * (self.Lambda_y + self.Lambda_gap) > self.y_H:
            self.y_H = np.round(self.N * (self.Lambda_y + self.Lambda_gap), \
                                decimals=2)
            return 'Warning: loudspeaker array is too long for the mounting height. Changed (14.)', 'too low'
        
        return '', None
    
    def get_initial_weights(self):
        """
        Initializes the weighting factors before starting the PALC algorithm.

        Returns
        -------
        None.

        """
        # Reset gap_weights
        self.gap_weights = np.ones(self.N)
        # Set weights depending on initial weighting_nu
        if self.use_weighting in ['Linear Spacing']:
            self.weighting_weights = np.linspace(1, self.weighting_nu, num=self.N)
        elif self.use_weighting in ['Logarithmic Spacing']:
            self.weighting_weights = np.geomspace(1, self.weighting_nu, num=self.N)
        else:
            self.weighting_weights = np.ones(self.N)
        # Set initial weighting factors
        self.weighting_factors = self.weighting_weights * self.gap_weights


class PALC_plotting():
    """ Class to create objects that contain data for visualization in bokeh
        figures. """
    def __init__(self):
        self.x_c_n, self.y_c_n, self.x_start, self.x_stop             = [], [], [], []
        self.y_start, self.y_stop, self.x_fin_unitn, self.y_fin_unitn = [], [], [], []
        self.x_start_b, x_stop_b, y_start_b, y_stop_b                 = [], [], [], []
        
        self.y_stop_unitn_stop, self.x_c_n_unitn, self.y_c_n_unitn    = [], [], []
        self.x_c_n_unitn_psi1, self.y_c_n_unitn_psi1                  = [], []
        self.x_c_n_unitn_psi2, self.y_c_n_unitn_psi2                  = [], []

    def update_plot_array(self, **new_input):
        """
        Updates the data of attributes.

        Parameters
        ----------
        **new_input : dict
            Key = value pair that shall be stored.

        Returns
        -------
        None.

        """
        for key, val in new_input.items():
            if hasattr(self, key):
                setattr(self, key, val)
                
    def get_plot_array(self, Opt_arr):
        """
        Includes the results of the PALC optimization to plot them in a
        bokeh figure.

        Parameters
        ----------
        Opt_arr : obj [in]
            Data of the optimized array by PALC.

        Returns
        -------
        None.

        """
        setattr(self, 'x_c_n_unitn', list(zip(self.x_c_n, self.x_fin_unitn)))
        setattr(self, 'y_c_n_unitn', list(zip(self.y_c_n, self.y_fin_unitn)))
        setattr(self, 'x_c_n_unitn_psi1', list(zip(self.x_c_n, Opt_arr.x_fin_unitn_psi_1)))
        setattr(self, 'y_c_n_unitn_psi1', list(zip(self.y_c_n, Opt_arr.y_fin_unitn_psi_1)))
        setattr(self, 'x_c_n_unitn_psi2', list(zip(self.x_c_n, Opt_arr.x_fin_unitn_psi_2)))
        setattr(self, 'y_c_n_unitn_psi2', list(zip(self.y_c_n, Opt_arr.y_fin_unitn_psi_2)))


class Array_opt():
    """ Class to create an object with optimized array data. """
    def __init__(self, gamma_tilt_deg=None):
        if gamma_tilt_deg is None:
            gamma_tilt_deg = []
        self.gamma_n, self.gamma_tilt_deg    = [], gamma_tilt_deg
        self.gamma_tilt_deg_diff, self.psi_n = [], []
        self.Gamma_aud, self.thr_dist        = [], []
        self.num_iter                        = []
        # deg_opt
        self.lin_eq_n_psi_1, self.lin_eq_n_psi_2            = [], []
        self.x_fin_unitn_psi_1, self.x_fin_unitn_psi_2      = [], []
        self.y_fin_unitn_psi_1, self.y_fin_unitn_psi_2      = [], []
        self.seg_pos, self.seg_pos_start, self.seg_pos_stop = [], [], []
        # fixed first angle (max, mid, min)
        self.fixed_angle = [0,0,0]
        if self.gamma_tilt_deg != [] : self.gamma_n = self.gamma_tilt_deg*np.pi/180


    def update_opt_array(self, x_c_n=None, y_c_n=None, **new_input):
        """
        Updates the optimized array object.

        Parameters
        ----------
        x_c_n : float, optional
            Center of rays at receiver positions on x-axis. The default is None.
        y_c_n : float, optional
            Center of rays at receiver positions on y-axis. The default is None.
        **new_input : dict
            Key = value pair that shall be stored.

        Returns
        -------
        None.

        """
        for key, val in new_input.items():
            if hasattr(self, key):
                setattr(self, key, val)
        if np.all(x_c_n != None) and 'psi_n' in new_input.keys():
            self.lin_eq_n_psi_1 = y_c_n - np.tan(-self.gamma_n + self.psi_n) * x_c_n
            self.x_fin_unitn_psi_1 = new_input['x_a_t_n'][:,0]
            self.y_fin_unitn_psi_1 = np.tan(-self.gamma_n + self.psi_n) \
                                     * self.x_fin_unitn_psi_1 + self.lin_eq_n_psi_1
            self.lin_eq_n_psi_2 = y_c_n - np.tan(-self.gamma_n - self.psi_n) * x_c_n
            self.x_fin_unitn_psi_2 = new_input['x_a_b_n'][:,0]
            self.y_fin_unitn_psi_2 = np.tan(-self.gamma_n - self.psi_n) \
                                     * self.x_fin_unitn_psi_2 + self.lin_eq_n_psi_2
            self.seg_pos = new_input['x_a_c_n']
            self.seg_pos_start = new_input['x_a_t_n']
            self.seg_pos_stop = new_input['x_a_b_n']


def calc_angles(PALC_plots, pal, psi_n, gamma_n, N, PALC_config, pal_e, Opt_arr):
    """
    Main routine of the PALC computation and is calles by :py:meth:`calcPALC`.
    Computes the tilt angles of the LSA cabinets.

    Parameters
    ----------
    PALC_plots : obj [out]
        Contains plotting data of the PALC results.
    pal : list
        All audience lines to be considered in PALC (audience lines,
        non-audience lines). Each element contains a audience line as a ndarray.
    psi_n : float
        Init splay angle for first PALC loop.
    gamma_n : ndarray
        Init tilt angles for the first iteration in radian.
    N : int
        Number of LSA cabinets.
    PALC_config : obj [in]
        Contains PALC Configuration.
    pal_e : ndarray
        Audience lines without non-audience lines.
    Opt_arr : obj [out]
        Result of the PALC algorithm. Optimized LSA data.

    Returns
    -------
    None.

    """
#    try:
    # aud_l: length of each audience line segment
    # aud_angle: epsilon
    # aud_pos_seg
    # aud_length: total length of the audience line
    aud_l = np.empty([np.shape(pal)[0]],float)
    aud_angle = np.empty([np.shape(pal)[0]],float)
    aud_pos_seg = np.empty([np.shape(pal)[0]+1,3],float)
    for n in range(np.shape(pal)[0]):
        aud_l[n] = np.sqrt(np.sum((pal[n][np.shape(pal[n])[0]-1,:]-pal[n][0,:])**2))
        aud_angle[n] = (pal[n][1,1]-pal[n][0,1]) / (pal[n][1,0]-pal[n][0,0])
        aud_pos_seg[n,:] = pal[n][0,:]
    #aud_pos_seg[np.shape(pal)[0],:] = pal[np.shape(pal)[0]-1][np.shape(pal[np.shape(pal)[0]-1])[0]-1,:]
    aud_pos_seg[-1,:] = pal[-1][-1,:]
    aud_pos_seg = np.flipud(aud_pos_seg)
    aud_l = np.flipud(aud_l)
    aud_angle = np.arctan(np.flipud(aud_angle))-np.pi
    aud_length = aud_l.sum()
    aud_length_copy = aud_length
    
    N_PALC = 0
    Gamma_aud = 0
    num_iter = 0
    fig_vs = 0
    psi_N = np.ones([N],float) * np.pi
    N_opt_deg_flag = 0
    d_array = np.zeros(N,float)
    aud_overlap = 0 # for discrete tilt angles: overlap on audience lines: to correct the error 'Gamma_aud'
    error = 0 # just to test
    
    #### begin of the while-loop
    while ( ((np.abs(Gamma_aud) > PALC_config.tolerance or (N_PALC != N) ) and (num_iter <= 500) ) or (num_iter < 15) ):
    #while num_iter == 0:
        # return if number of iteration reaches the maximum (declared above)
        aud_overlap = 0 # for discrete tilt angles: overlap on audience lines: to correct the error 'Gamma_aud'
        if num_iter == 500:
            Opt_arr.num_iter = 500
            return
            
        k = 0
        seg_rem = 0
        x_a_c_n = np.zeros([N,3],float)
        x_a_t_n = np.zeros([N,3],float)
        x_a_b_n = np.zeros([N,3],float)
        Gamma_n = np.zeros([N,3],float)
        
        ####### BEGIN ITERATION STEPS #######
        if num_iter != 0 and N_PALC != N:
            if N_opt_deg_flag == 0:
                psi_n[:] = psi_n[:] * (1 - 0.1 * N / N_PALC)
            else:
                psi_n[:] = psi_n[:] * (1 - 0.1 * N / N_PALC)
        elif num_iter != 0 and N_PALC == N:
            N_opt_deg_flag = 1
            if num_iter <= 15:
                psi_n[:] = psi_n[:] * (1 - np.sign(Gamma_aud) * 0.1 / num_iter)
            elif num_iter > 15:
                if np.abs(Gamma_aud) <= 0.2:
                    psi_n[:] = psi_n[:] * (1 - np.sign(Gamma_aud) * 0.00015)
                elif np.abs(Gamma_aud) > 0.2 and np.abs(Gamma_aud) <= 1.5:
                    psi_n[:] = psi_n[:] * (1 - np.sign(Gamma_aud) * 0.002)
                elif np.abs(Gamma_aud) > 1.5 and np.abs(Gamma_aud) <= 4:
                    psi_n[:] = psi_n[:] * (1 - np.sign(Gamma_aud) * 0.01)
                elif np.abs(Gamma_aud) > 4 and np.abs(Gamma_aud) <= 8:
                    psi_n[:] = psi_n[:] * (1 - np.sign(Gamma_aud) * 0.03)
                elif np.abs(Gamma_aud) > 8:
                    psi_n[:] = psi_n[:] * (1 - np.sign(Gamma_aud) * 0.05)
            
        
        N_PALC = 0
        
        ###### BEGIN FOR LOOP, one loop for every loudspeaker cabinet
        
        # use m instead of i (matlab code)
        for m in range(N):
            
            if m == 0:
                # (top position) x_PAL_k, Initialisierung, that highest LS cabinet beams on highest audience positions
                if not PALC_config.use_fixed_angle or PALC_config.last_angle_hm: # not not
                    x_a_t_n[m,:] = aud_pos_seg[k,:]
                    # aud_l are the distances between x_PAL_k and x_PAL_k+1
                    # seg_rem = |x_a_t_n - x_PAL_k-1|                      
                    seg_rem = aud_l[k]  
                    # analytical Calculation of GammaTiltDeg_angle: 
                    gamma_n[m] = calc_gamma(gamma_n, psi_n, \
                       PALC_config.x_H, PALC_config.y_H, x_a_t_n, PALC_config.Lambda_y, m)

                ############ FIXED FIRST ANGLE ############################
                elif PALC_config.use_fixed_angle:
                    # es muss
                    # gamma_n eingelesen werden - fertig
                    # 2. checken ob es einen schnittpunkt des obersten strahls mit pal gibt (strahl über gamma_n und psi_n)
                    # mit intersect funktion: x max wert von pal raussuchen und den strahl bis x max +1 laufen lassen
                    # 3. wenn nicht: pal vertikal einfügen sodass schnittpunkt existiert
                    # 4. wenn nicht: pal in aud_pos_seg einfügen
                    # 5. x_a_t_n berechnen 
                    if num_iter > 0 and np.any(point_exists1) == 0:
                        del pal[-1]
                        aud_pos_seg = np.delete(aud_pos_seg, [0,1,2]).reshape(int((np.shape(aud_pos_seg)[0]-1)),3)
                        aud_length -= aud_l[0]
                        aud_l = np.delete(aud_l, 0)
                        aud_angle = np.delete(aud_angle, 0)
                    elif num_iter > 0:
                        aud_length = aud_length_copy
                    # get the fixed angle
                    gamma_n[m] = PALC_config.fixed_angle * np.pi / 180

                    # top direction of rays of the highest LS cabinet
                    direction = gamma_n[m] - psi_n[m]  
                    hypo = 1/np.cos(direction)   
                    direction_y = float(hypo * np.sin(direction))
                    # source position of the highest LS cabinet
                    source_x = [PALC_config.x_H - (np.sin(gamma_n[m]) * PALC_config.Lambda_y)/2]

                    source_y = [PALC_config.y_H - (np.cos(gamma_n[m]) * PALC_config.Lambda_y)/2]
                    # find maximum value on x-axis of pal and end point of ray

                    xmax = max(o.max() for o in pal)
                    end_x = np.array([xmax + 1])
                    end_y = source_y - direction_y * (end_x - source_x)
                    # find the intersection point if exists
                    xy_s1 = np.empty([np.shape(pal)[0],2])
                    point_exists1 = np.empty(np.shape(pal)[0])
                    for o in range(np.shape(pal)[0]):

                        xy_s1[o,:], lambda_1, lambda_2, point_exists1[o] = \
                        calc_intersection_PAL(np.array([source_x[0], source_y[0], 0]), np.array([end_x[0], end_y[0], 0]), pal[o][0,:], pal[o][-1,:])

                        if point_exists1[o] == 1 and pal[o][0,0]<=np.round(xy_s1[o,0], decimals=3)<=pal[o][-1,0] and \
                           pal[o][0,1]<=np.round(xy_s1[o,1], decimals=3)<=pal[o][-1,1]:

                            point_exists1[o] = int(1)
                        else:

                            point_exists1[o] = int(0)
                    # add a audience line if intersection point doesnt exist
                    if np.any(point_exists1) == 0:
                        pal.append(np.array([pal[-1][-1,0],pal[-1][-1,1], 0, xmax, source_y[0] - direction_y * (xmax - source_x[0]), 0]).reshape(2,3))
                        aud_l = np.append(np.sqrt(np.sum((pal[-1][-1,:]-pal[-1][0,:])**2)), aud_l)
                        aud_angle = np.append(np.arctan((pal[-1][1,1]-pal[-1][0,1]) / (pal[-1][1,0]-pal[-1][0,0]))-np.pi, aud_angle)
                        aud_length += aud_l[0]
                        aud_pos_seg = np.append([pal[-1][-1,:]], aud_pos_seg, axis=0)

                        x_a_t_n[m,:] = aud_pos_seg[k,:] 
                        seg_rem = aud_l[k]
                    # case if the highest LS cabinet covers the last audience line
                    elif point_exists1[-1] == 1:
                        x_a_t_n[m,:-1] = xy_s1[np.argmax(point_exists1),:]
                        seg_rem = aud_l[k] - np.sqrt(np.sum((pal[-1][np.shape(pal[-1])[0]-1,:-1]-xy_s1[np.argmax(point_exists1),:])**2))
                        aud_length -= (aud_l[k] - seg_rem)
                        aud_pos_seg[k,:-1] = xy_s1[np.argmax(point_exists1),:]
                    # if the highest LS cabinet hits pal 0,1,...n-1: no convergence
                    else:
                        return
                    
  
                # Discrete angles if hard margin is used for possibly second computation
                if PALC_config.use_gamma_LSA and np.shape(PALC_config.gamma_LSA)[0] > 1 \
                   and PALC_config.gap_handling in ['Hard Margin'] and PALC_config.last_angle_hm:
                    val = np.abs(PALC_config.gamma_LSA - (gamma_n[m] - PALC_config.last_angle_hm))
                    ind = np.unravel_index(np.argmin(val, axis=None), val.shape)
                    gamma_n[m] = PALC_config.gamma_LSA[ind[0]] + PALC_config.last_angle_hm
            elif m > 0:
                x_a_t_n[m,:] = x_a_b_n[m-1,:]
                gamma_n[m] = calc_gamma(gamma_n, psi_n, x_start, y_start, x_a_t_n, PALC_config.Lambda_y, m)
 
                ################## DISCRETE TILT ANGLES ###################
                if PALC_config.use_gamma_LSA and np.shape(PALC_config.gamma_LSA)[0] > 1:
                    ######### Calculate The Tilt Angles By Usage Of The Set Of Discrete Tilt Angles #######
                    val = np.abs(PALC_config.gamma_LSA - (gamma_n[m] - gamma_n[m-1]))
                    ind = np.unravel_index(np.argmin(val, axis=None), val.shape)
                    gamma_n[m] = PALC_config.gamma_LSA[ind[0]] + gamma_n[m-1]
                    
                if (PALC_config.use_gamma_LSA and np.shape(PALC_config.gamma_LSA)[0] > 1) or gamma_n[m]-gamma_n[m-1] < 0:
                    # case in continuous tilt angles: intercabinet angle is smaller than zero
                    if gamma_n[m]-gamma_n[m-1] < 0: gamma_n[m] = float(gamma_n[m-1])
                    ######### Shift Necessary Variables And Update k If Necessary ###########
                    direction = gamma_n[m] - psi_n[m]  # angle of direction to x_a_t_n of m-th loudspeaker
                    hypo = 1/np.cos(direction)    # length if side adjacent is equal to one
                    direction_y = hypo * np.sin(direction) # length in y direction if side adjacent (x direction)  is equal to one
                    x_start, y_start, x_stop, y_stop, x_c_n, y_c_n, x_S, y_S = source_pos(gamma_n, PALC_config) # source positions
                    xmax = max(o.max() for o in pal)
                    end_x = np.array([xmax + 1]) # max length in x direction
                    end_y = y_c_n[m] - direction_y * (end_x - x_c_n[m]) #  max depth of y direction source_y - (end_x - source_x) / np.tan((np.pi/2) - (gamma_n[m]-psi_n[m]))
                    xy_s1 = np.empty([np.shape(pal)[0],2])
                    point_exists2 = False
                    # check in which pal segment is the intersection point
                    for o in range(np.shape(pal)[0]):
                        xy_s1[0,:], lambda_1, lambda_2, point_exists2 = \
                        calc_intersection_PAL(np.array([x_c_n[m], y_c_n[m], 0]), \
                                              np.array([end_x[0], end_y[0], 0]), pal[o][0,:], pal[o][-1,:])
                        if point_exists2 == True and pal[o][0,0]<=np.round(xy_s1[0,0], decimals=3)<=pal[o][-1,0] and \
                           pal[o][0,1]<=np.round(xy_s1[0,1], decimals=3)<=pal[o][-1,1]:
                            # check if current pal[o] is equal to current k
                            if o == (np.shape(pal)[0]-1-k):
                                x_a_t_n[m,:] = [xy_s1[0,0], xy_s1[0,1], 0] # new top position on audience line
                                seg_rem = aud_l[k] - np.sqrt(np.sum((pal[-k-1][-1,:-1]-xy_s1[0,:])**2)) # corrected segment remaining
                                # calc total overlap on audience line to correct Gamma_aud
                                if np.sum(x_a_t_n[m,:]-x_a_b_n[m-1,:]) >= 0:
                                    aud_overlap += np.sqrt(np.sum((x_a_t_n[m,:]-x_a_b_n[m-1,:])**2))
                                elif np.sum(x_a_t_n[m,:]-x_a_b_n[m-1,:]) < 0:
                                    aud_overlap -= np.sqrt(np.sum((x_a_t_n[m,:]-x_a_b_n[m-1,:])**2))
                            # case if current pal[o] is on the pal segment above or rather on k-1
                            elif o == (np.shape(pal)[0]-k):
                                aud_overlap += np.sqrt(np.sum((x_a_t_n[m,:]-pal[-1-k][-1,:])**2)) # old k-th segment overlap
                                k -= 1 # update k
                                epsilon = -aud_angle[k] - (np.pi / 2) # angle of new pal segment
                                x_a_t_n[m,:] = [xy_s1[0,0], xy_s1[0,1], 0] # new top position on audience line
                                seg_rem = aud_l[k] - np.sqrt(np.sum((pal[-k-1][-1,:-1]-xy_s1[0,:])**2)) # corrected segment remaining
                                aud_overlap += np.sqrt(np.sum((x_a_t_n[m,:]-pal[-1-k][0,:])**2)) # new k-th segment overlap
                            # case if current pal[o] is on the pal segment below or rather on k+1
                            elif o == (np.shape(pal)[0]-2-k):
                                aud_overlap -= np.sqrt(np.sum((x_a_t_n[m,:]-pal[-1-k][0,:])**2)) # old k-th segment overlap
                                k += 1 # update k
                                epsilon = -aud_angle[k] - (np.pi / 2) # angle of new pal segment
                                x_a_t_n[m,:] = [xy_s1[0,0], xy_s1[0,1], 0] # new top position on audience line
                                seg_rem = aud_l[k] - np.sqrt(np.sum((pal[-k-1][-1,:-1]-xy_s1[0,:])**2)) # corrected segment remaining
                                aud_overlap -= np.sqrt(np.sum((x_a_t_n[m,:]-pal[-1-k][-1,:])**2)) # new k-th segment overlap
                        
                    if point_exists2 == False:
                            logger.warning('Error in discrete tilt angles')
                            return
   
            ###############################################################################        
            # STEP 2: CALCULATE THE SOURCE POSITIONS  
            x_start, y_start, x_stop, y_stop, x_c_n, y_c_n, x_S, y_S = source_pos(gamma_n, PALC_config)
                
            ###############################################################################
            # STEP 3
            
            # center position of audience segment - epsilon for eq (15) JAES
            epsilon = -aud_angle[k] - (np.pi / 2)
            # eq (15) JAES
            Xi_1 = np.sqrt(np.sum((x_a_t_n[m,:] - [x_c_n[m], y_c_n[m], 0])**2)) * (np.sin(psi_n[m]) \
                           / np.cos(epsilon - gamma_n[m]))
            if seg_rem >= Xi_1: # case (i) eq (16) in JAES
                seg_rem = seg_rem - Xi_1
                # eq (17) JAES (aud_angle = epsilon_k)
                x_a_c_n[m,:] = x_a_t_n[m,:] + np.array([np.cos(aud_angle[k]), np.sin(aud_angle[k]), 0]) * Xi_1
                psi_n_tilde = 0
                # eq (18) JAES
                Gamma_n[m,0] = Xi_1
            elif seg_rem < Xi_1:
                Gamma_n[m,0] = seg_rem # first part of eq (23) JAES
                d = np.sqrt(np.sum((x_a_t_n[m,:] - [x_c_n[m], y_c_n[m], 0])**2)) # distance for eq (20) JAES
                # eq (20) JAES
                psi_n_tilde = np.arctan((seg_rem / d * np.cos(epsilon - gamma_n[m]) - \
                                         np.sin(psi_n[m])) / (seg_rem / d * np.sin(epsilon \
                                               - gamma_n[m]) - np.cos(psi_n[m])))
            ########################## STEP IV ###########################################
                if k < np.shape(aud_l)[0]-1:
                    k += 1
                    epsilon = -aud_angle[k] - (np.pi / 2)
                else:
                    N_PALC = m
                    Gamma_n[m,0] = 0
                    break
            ####################### CONTINUE WITH STEP III ###############################
                d = np.sqrt(np.sum((aud_pos_seg[k,:] - [x_c_n[m], y_c_n[m], 0])**2))
                # eq (21) JAES
                Xi_tilde = d * np.sin(psi_n_tilde) / np.cos(-gamma_n[m] + epsilon)
                # eq (22) JAES
                x_a_c_n[m,:] = aud_pos_seg[k,:] + np.array([np.cos(aud_angle[k]), np.sin(aud_angle[k]), 0]) * Xi_tilde
                seg_rem = aud_l[k] - Xi_tilde
                # second part of eq (23) JAES
                Gamma_n[m,0] = Gamma_n[m,0] + Xi_tilde
            ######################## STEP V #############################################
    
            d = np.sqrt(np.sum((x_a_c_n[m,:] - [x_c_n[m], y_c_n[m], 0])**2))
            # stop position of audience segment
            # calculate psi, that it fits for PALC1, PALC2 or PALC3 ### CONSTRAINTS ###
            if m == 0:
                d_array[m] = np.sqrt(np.sum((x_a_c_n[m,:] - [x_c_n[m], y_c_n[m], 0])**2))
            elif m > 0:
                d_array[m] = np.sqrt(np.sum((x_a_c_n[m,:] - [x_c_n[m], y_c_n[m], 0])**2))
                if PALC_config.constraint in ['PALC 1']:
                    # for PALC1: psi = const.
                    psi_n[m] = (psi_n[m-1] * PALC_config.weighting_factors[m-1]) / PALC_config.weighting_factors[m]
                elif PALC_config.constraint in ['PALC 2']:
                    # psi * d = const. with weighting factor PALC_config[7]
                    psi_n[m] = (d_array[m-1] * psi_n[m-1] * PALC_config.weighting_factors[m-1]) \
                    / (d_array[m] * PALC_config.weighting_factors[m])
                elif PALC_config.constraint in ['PALC 3']:
                    # tan(psi) * d = const.
                    psi_n[m] = np.arctan((d_array[m-1] * np.tan(psi_n[m-1]) * PALC_config.weighting_factors[m-1]) \
                    / (d_array[m] * PALC_config.weighting_factors[m]))    
            # eq (24) JAES
            Xi_2 = d * np.sin(psi_n[m]) / np.cos(epsilon - gamma_n[m] - psi_n[m])
            
            if seg_rem >= Xi_2: # case (ii) eq (30) JAES
                # eq (31) JAES
                x_a_b_n[m,:] = x_a_c_n[m,:] + np.array([np.cos(aud_angle[k]), np.sin(aud_angle[k]), 0]) * Xi_2
                seg_rem = seg_rem - Xi_2
                psi_n_tilde = 0
                Gamma_n[m,1] = Xi_2 # eq (32) JAES
            elif seg_rem < Xi_2: # case (i) eq (23) JAES
                Gamma_n[m,1] = seg_rem # first part of eq (29) JAES
                # eq (26) JAES
                psi_n_tilde = np.arctan(seg_rem / d * np.cos(epsilon - gamma_n[m]) / \
                                         (1 - seg_rem / d * np.sin(epsilon - gamma_n[m])))
            ############################## STEP VI ########################################     
                if k < np.shape(aud_l)[0]-1:
                    k += 1
                    epsilon = -aud_angle[k] - (np.pi / 2)
                else:
                    N_PALC = m
                    Gamma_n[m,1] = 0
                    break 
            ########################### CONTINUE WITH STEP V #############################
                d = np.sqrt(np.sum((aud_pos_seg[k,:] - [x_c_n[m], y_c_n[m], 0])**2))
                # eq (27) JAES
                Xi_tilde = d * np.sin(psi_n[m] - psi_n_tilde) / np.cos(epsilon - gamma_n[m] - psi_n[m])
                seg_rem = aud_l[k] - Xi_tilde
                # eq (28) JAES
                x_a_b_n[m,:] = aud_pos_seg[k,:] + np.array([np.cos(aud_angle[k]), np.sin(aud_angle[k]), 0]) * Xi_tilde
                # second part of eq (29) JAES

                Gamma_n[m,1] = Gamma_n[m,1] + Xi_tilde
        ####################### END OF FOR-LOOP ####################################### 

        if N_PALC == 0:
            N_PALC = N
            psi_N = psi_n
        
        ######### PLOT ################################################################
            
        lin_eq_n = y_c_n - np.tan(-gamma_n[:]) * x_c_n
        x_fin_unitn = x_a_c_n[:,0]
        y_fin_unitn = np.tan(-gamma_n[:]) * x_fin_unitn + lin_eq_n
        if psi_n.ndim == 2:
            psi_n = psi_n[:,0]
        
        # update optimized LSA
        Opt_arr.update_opt_array(x_c_n=x_c_n, y_c_n=y_c_n, x_a_c_n=x_a_c_n, \
                                 x_a_t_n=x_a_t_n, x_a_b_n=x_a_b_n, \
                                 gamma_n=gamma_n, psi_n=psi_n)

        ############################### Soft Margin Approach for gap handling #########################
        if PALC_config.gap_handling in ['Soft Margin']:            
            # First step: Find the added non audience line with pal and pal_e (indice)
            ind_na_line = np.array([]).astype(int)
            PALC_config.gap_weights = np.ones(PALC_config.N)
            m = 0
            o = 0
            for n in range(np.shape(pal)[0]):
                if PALC_config.use_fixed_angle and np.any(point_exists1) == 0 and n == np.shape(pal)[0]-1:
                    o=1
                if pal_e[n-m-o][0,0] != pal[n][0,0] or pal_e[n-m-o][0,1] != pal[n][0,1]:
                    m += 1
                    ind_na_line = np.append(ind_na_line, n)
            # Second step: check which loudspeaker hits which line on which point and 
            #              add the weighting
            for n in range(np.shape(pal)[0]):
                if np.any(n==ind_na_line):
                    for m in range(N):
                        if pal[n][0,0] <= Opt_arr.seg_pos[m,0] <= pal[n][-1,0] and \
                           pal[n][0,1] <= Opt_arr.seg_pos[m,1] <= pal[n][-1,1]:
                               # compute the length of non-audience line and
                               # distance of deg_opt to next audience line
                            dist_na_line = np.sqrt((pal[n][-1,0]-pal[n][0,0])**2 + \
                                                      (pal[n][-1,1]-pal[n][0,1])**2)
                            dist_high = np.sqrt((pal[n][-1,0]-Opt_arr.seg_pos[m,0])**2 + \
                                                   (pal[n][-1,1]-Opt_arr.seg_pos[m,1])**2)
                            dist_low = np.sqrt((Opt_arr.seg_pos[m,0]-pal[n][0,0])**2 + \
                                                  (Opt_arr.seg_pos[m,1]-pal[n][0,1])**2)
                            if n == np.shape(pal)[0]-1:
                                dist_near = dist_low
                            else:
                                dist_near = np.amin([dist_high, dist_low])
                            PALC_config.gap_weights[m] = 1 - PALC_config.strength_sm*(dist_near/dist_na_line)
                                               
        ###### weighting factors calculation with special case if fixed angles, soft margin and weighting
        if PALC_config.use_fixed_angle and PALC_config.gap_handling in ['Soft Margin'] \
           and PALC_config.weighting_nu <= 1.02 and not PALC_config.use_weighting in ['Without']:
            elsecase = False
            for n in range(PALC_config.N):
                if PALC_config.gap_weights[n] < 1 and not elsecase:
                    PALC_config.weighting_factors[n] = PALC_config.weighting_weights[n]
                else:
                    elsecase = True
                    PALC_config.weighting_factors[n] = PALC_config.gap_weights[n] * PALC_config.weighting_weights[n]
            del elsecase
        else:
            PALC_config.weighting_factors = PALC_config.gap_weights * PALC_config.weighting_weights
            
            # For future: may add some weighting for intersect points near to non-audience lines
        ###############################################################################################
        # plot function! not optimized for plotting during calculation -> one plot at the end
        
        ######### update output + variables ###########################################
        
        gamma_tilt_deg = gamma_n / np.pi * 180
        error = np.sqrt(np.sum((aud_pos_seg[-1,:]-x_a_b_n[-1,:])**2))
        Gamma_aud = np.sum(Gamma_n[:,:]) - aud_length - aud_overlap

        thr_dist = np.sqrt(np.sum((x_a_c_n[:,:] - np.transpose(np.array([x_c_n[:], y_c_n[:], x_c_n[:]*0])))**2,1))
        num_iter += 1
        logger.debug('error is: %s', Gamma_aud)
        # if difference is to big, reset psi_n
        if np.abs(Gamma_aud) > 50:
            psi_n = PALC_config.psi_n

    PALC_plots.update_plot_array(x_c_n=x_c_n, y_c_n=y_c_n, x_fin_unitn=x_fin_unitn, y_fin_unitn=y_fin_unitn)
    Opt_arr.update_opt_array(gamma_n=gamma_n, psi_n=psi_n, num_iter=num_iter, \
                             Gamma_aud=Gamma_aud, thr_dist = thr_dist, \
                             gamma_tilt_deg=gamma_tilt_deg)
  
    # set new gamma_n
    setattr(PALC_config,'gamma_n',gamma_n)
    return


def calcPALC(PALC_plots, PALC_pal, PALC_config, Opt_arr):
    """
    Handles the different PALC approaches. Offers the functionality of the
    Hard Margin Gap Handling Approaches. Without and Soft Margin Gap Handling
    Approach call directily :any:`calc_angles`.

    Parameters
    ----------
    PALC_plots : obj [in, out]
        Contains plotting information.
    PALC_pal : obj [in]
        Contains all venue information.
    PALC_config : obj [in]
        Contains PALC configuration data.
    Opt_arr : obj [in, out]
        Contains the data of the optimized LSA.

    Returns
    -------
    None.

    """
    #### again: initialization ####
    N       = int(PALC_config.N)
    gamma_n = np.array(PALC_config.gamma_n)
    psi_n   = np.array(PALC_config.psi_n)

    pal_e = list(PALC_pal.pal_no_nal)
    pal = list(PALC_pal.pal)

    # no Gap Handling or Soft Margin
    if PALC_config.gap_handling in ['Without', 'Soft Margin']:
        t1 = time.time()
        calc_angles(PALC_plots, pal, psi_n, gamma_n, N, PALC_config, pal_e, Opt_arr)
        t2 = time.time()
        dt = t2 - t1
        logger.debug('elapsed time in s: %s', dt)
        return 
    # Gap Handling: Hard Margin Approach
    elif PALC_config.gap_handling in ['Hard Margin']:
        calc_angles(PALC_plots, pal, psi_n, gamma_n, N, PALC_config, pal_e, Opt_arr)
        # create pal_h that contains lists with the continuous audience lines
        pal_h = []
        pal_h.append([])
        m = 0
        for n in range(np.shape(pal_e)[0]):
            if pal_e[n][0,0] != pal[n+m][0,0] or pal_e[n][0,1] != pal[n+m][0,1]:
                pal_h.append([])
                m += 1
            if pal_e[n][0,0] == pal[n+m][0,0] and pal_e[n][0,1] == pal[n+m][0,1]:
                pal_h[m].append(pal[n+m])
        # check which loudspeaker cabinets belong to which continuous audience line
        ls2seg = np.empty(N).astype(int)
        seg_dist_high = np.empty(np.shape(pal_h)[0])
        seg_dist_low = np.empty(np.shape(pal_h)[0])
        seg_dist = np.empty(np.shape(pal_h)[0])
        for m in range(N):
            # go from the highest audience line to the lowest
            for n in range(np.shape(pal_h)[0]-1,-1,-1):
                seg_dist_high[n] = np.abs(np.sqrt((pal_h[n][-1][-1,0]-Opt_arr.seg_pos[m,0])**2 \
                              + (pal_h[n][-1][-1,1]-Opt_arr.seg_pos[m,1])**2))
                seg_dist_low[n] = np.abs(np.sqrt((pal_h[n][0][0,0]-Opt_arr.seg_pos[m,0])**2 \
                            + (pal_h[n][0][0,1]-Opt_arr.seg_pos[m,1])**2))
                seg_dist[n] = np.minimum(seg_dist_high[n], seg_dist_low[n])
            
            # get the indice of the segment to which the LS belongs
            ls2seg[m] = int(np.argmin(seg_dist))

        # Run the calculation for each continuous audience line
        for n in range(np.shape(pal_h)[0]-1,-1,-1):
            N_margin = 0
            for m in range(N):
                if n == ls2seg[m]:
                    N_margin += 1

            if n == np.shape(pal_h)[0]-1:
                gamma_n = np.linspace(-0.0428, 0.7147, num=N_margin)
                psi_n = np.linspace(0.00524, 0.00524, num=N_margin)
                calc_angles(PALC_plots, pal_h[n], psi_n, gamma_n, N_margin, PALC_config, pal_e, Opt_arr)
                x_patches, y_patches = LSA_visualization(PALC_plots, PALC_config, Opt_arr.gamma_n, \
                                                          N_margin=np.shape(Opt_arr.gamma_tilt_deg)[0])

                Opt_arr_hm = copy(Opt_arr)
                PALC_plots_hm = copy(PALC_plots)
            else:

                PALC_config.last_angle_hm = Opt_arr_hm.gamma_tilt_deg[-1] * np.pi / 180
                PALC_config.gamma_n = gamma_n = np.linspace(-0.0428, 0.7147, num=N_margin)
                PALC_config.psi_n = psi_n = np.linspace(0.00524, 0.00524, num=N_margin)

                PALC_config.x_H = PALC_plots_hm.x_stop[-1] - np.sin(PALC_config.last_angle_hm) * PALC_config.Lambda_gap
                PALC_config.y_H = PALC_plots_hm.y_stop[-1] - np.cos(PALC_config.last_angle_hm) * PALC_config.Lambda_gap

                calc_angles(PALC_plots_hm, pal_h[n], psi_n, gamma_n, N_margin, PALC_config, pal_e, Opt_arr_hm)
                x_patches_hm, y_patches_hm = LSA_visualization(PALC_plots_hm, PALC_config, Opt_arr_hm.gamma_n, \
                                                          N_margin=np.shape(Opt_arr_hm.gamma_tilt_deg)[0])
                for attr in ['x_c_n', 'y_c_n','x_start', 'y_start', 'x_stop', \
                              'y_stop', 'x_fin_unitn', 'y_fin_unitn']:
                    setattr(PALC_plots, attr, np.append(getattr(PALC_plots, attr), getattr(PALC_plots_hm, attr)))
                for attr in ['seg_pos', 'seg_pos_start', 'seg_pos_stop', 'x_fin_unitn_psi_1', \
                              'y_fin_unitn_psi_1', 'x_fin_unitn_psi_2', 'y_fin_unitn_psi_2', \
                              'gamma_n', 'gamma_tilt_deg', 'thr_dist']:
                    setattr(Opt_arr, attr, np.append(getattr(Opt_arr, attr), getattr(Opt_arr_hm, attr)))
                    if attr in ['seg_pos', 'seg_pos_start', 'seg_pos_stop']:
                        setattr(Opt_arr, attr, getattr(Opt_arr, attr).reshape(int(N),3))
        return


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


def fixed_angle_borders(last_pal_sec, PALC_config):
    """
    Allowed range of the fixed tilt angle of the highest LSA cabinet, as the
    GUI computes it when "fix angle of uppermost cabinet" is enabled.

    Parameters
    ----------
    last_pal_sec : ndarray
        Last PAL section, i.e., PALC_pal.pal[-1].
    PALC_config : obj [in]
        Configuration of the PALC algorithm.

    Returns
    -------
    list of float
        Maximum, mid and minimum tilt angle in degree (the legacy
        ``Opt_arr.fixed_angle``).

    """
    gamma_tilt_deg = get_fixed_angle_borders(last_pal_sec, PALC_config)
    return [gamma_tilt_deg[0]-1, gamma_tilt_deg[1], gamma_tilt_deg[1]-10]


def clamp_fixed_angle(fixed_angle, max_deg, min_deg):
    """
    Clamps the fixed tilt angle of the highest LSA cabinet into its allowed
    range, rounded to two decimals like ``round_fixed_angle`` in the GUI.

    Parameters
    ----------
    fixed_angle : float
        Requested tilt angle in degree.
    max_deg : float
        Maximum tilt angle in degree.
    min_deg : float
        Minimum tilt angle in degree.

    Returns
    -------
    float
        The angle itself if it is within the range, otherwise the rounded
        border it exceeds.

    """
    if fixed_angle > max_deg:
        return np.round(max_deg, decimals=2)
    elif fixed_angle < min_deg:
        return np.round(min_deg, decimals=2)
    return fixed_angle
