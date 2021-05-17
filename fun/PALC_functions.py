#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Arne Hoelter

This module contains function that are used in :py:meth:`calcPALC` and
:any:`calc_angles`.
"""

import numpy as np
from copy import copy

##    function is contained in the numpy.matlib module
#     for further information, see: https://github.com/numpy/numpy/blob/v1.13.0/numpy/matlib.py#L310-L358 
#
def repmat(a, m, n): 
    """ Contained in numpy.matlib module,
        see https://github.com/numpy/numpy/blob/v1.13.0/numpy/matlib.py#L310-L358 """
    a = np.asanyarray(a)
    ndim = a.ndim
    if ndim == 0:
        origrows, origcols = (1, 1)
    elif ndim == 1:
        origrows, origcols = (1, a.shape[0])
    else:
        origrows, origcols = a.shape
    rows = origrows * m
    cols = origcols * n
    c = a.reshape(1, a.size).repeat(m, 0).reshape(rows, origcols).repeat(n, 0)
    return c.reshape(rows, cols)

def list_sort(seq):
    """
    Sorts a sequence of lists. Actually not used.

    Parameters
    ----------
    seq : list
        List to sort.

    Returns
    -------
    list
        Sorted list.

    """
    return sorted(range(len(seq)), key=seq.__getitem__)


def calc_diff_tilt_angles(gamma_tilt_deg):
    """
    Calculates the intercabinet tilt angles. Called by :any:`start_calc`.

    Parameters
    ----------
    gamma_tilt_deg : ndarray
        Computed absolute tilt angles in degree.

    Returns
    -------
    ndarray
        Rounded intercabinet tilt angles and absolute tilt angles in degree.

    """
    gamma_tilt_deg_diff = np.zeros(np.shape(gamma_tilt_deg)[0],float)
    gamma_tilt_deg_diff[0] = gamma_tilt_deg[0]
    for n in range(1, np.shape(gamma_tilt_deg)[0]):
        gamma_tilt_deg_diff[n] =  gamma_tilt_deg[n] - gamma_tilt_deg[n-1]
    return np.round(gamma_tilt_deg_diff, decimals=2) , np.round(gamma_tilt_deg, decimals=2)


def calc_progressive_array(N, gamma_LSA=[], gamma_0 = 0, gamma_f = 38, use_dangles = "No"):
    """
    Calculates the absolute tilt angles of a progressive LSA curving. Called by
    :any:`ref_array_angles`.

    Parameters
    ----------
    N : int
        Number of loudspeaker array cabinets.
    gamma_LSA : list, optional
        Possible discrete intercabinet tilt angles, if use_dangles is set to
        'Yes'. The default is [].
    gamma_0 : float, optional
        Tilt angle of the highest (first) LSA cabinet in degree.
        The default is 0.
    gamma_f : float, optional
        Angle of the last tilt angle in degree. The default is 38.
    use_dangles : str, optional
        If discrete tilt angles shall be used: 'Yes'. The default is 'No'.

    Returns
    -------
    gamma_deg : ndarray
        Array with the progressive LSA curving in degree.

    """
    gamma_deg = np.zeros(N)   
    # calculate delta_gamma with given terminal angle gamma_f
    delta_gamma = (gamma_f-gamma_0) / ((N/2)*(N-1))
    #delta_gamma = 7/N
    # calculate cabinet tilt angle for all LSA cabinets
    for n in range(N):
        gamma_deg[n] = gamma_0 + ((n+1)/2)*(n)*delta_gamma
        if np.shape(gamma_LSA)[0] > 1 and n > 0 and use_dangles == "Yes":
            val = np.abs(gamma_LSA / (np.pi / 180) - (gamma_deg[n] - gamma_deg[n-1]))
            ind = np.unravel_index(np.argmin(val, axis=None), val.shape)
            gamma_deg[n] = gamma_LSA[ind[0]] / (np.pi / 180) + gamma_deg[n-1]
    return gamma_deg


def calc_arc_array(N, gamma_LSA=[], gamma_0 = 0, gamma_delta = 1, use_dangles = "No"):
    """
    Calculates the absolute tilt angles of an arc LSA curving. Called by
    :any:`ref_array_angles`.

    Parameters
    ----------
    N : int
        Number of loudspeaker array cabinets.
    gamma_LSA : list, optional
        Possible discrete intercabinet tilt angles, if use_dangles is set to
        'Yes'. The default is [].
    gamma_0 : float, optional
        Tilt angle of the highest (first) LSA cabinet in degree.
        The default is 0.
    gamma_delta : float, optional
        Angle of the intercabinet angles in degree. The default is 1.
    use_dangles : str, optional
        If discrete tilt angles shall be used: 'Yes'. The default is 'No'.

    Returns
    -------
    gamma_deg : ndarray
        Array with the progressive LSA curving in degree.

    """
    gamma_deg = np.zeros(N)
    if np.shape(gamma_LSA)[0] > 1 and use_dangles == "Yes":
        gamma_delta = gamma_LSA[np.argmin(np.abs((gamma_LSA/(np.pi/180))-gamma_delta))]*(180/np.pi)
    for n in range(N):
        gamma_deg[n] = gamma_0 + n * gamma_delta
    return gamma_deg
    

def calc_gamma(gamma_n, psi_n, x_start, y_start, x_a_t_n, Lambda_y, m=0):
    """
    Calculates a specific loudspeaker tilt angle of a LSA cabinet. Called by
    :any:`calc_angles`.

    Parameters
    ----------
    gamma_n : ndarray
        Set of LSA tilt angles.
    psi_n : float
        Splay / aperture angle of the loudspeaker cabinets or of the m-th LS cabinet.
    x_start : float
        Highest point of the LSA cabinet in x-coordinate.
    y_start : float
        Highest point of the LSA cabinet in y-coordinate.
    x_a_t_n : float
        Top position of covered audience line by m-th LSA cabinet (x- and y-coordinate).
    Lambda_y : float
        Height of m-th LSA cabinet.
    m : int, optional
        Variable which LSA cabinet is optimized. The default is 0.

    Returns
    -------
    float
        Tilt angle of m-th LSA cabinet in degree.

    """
    if m == 0:
        dx_an = x_a_t_n[m,0] - x_start
        dy_an = x_a_t_n[m,1] - y_start
    else:
        dx_an = x_a_t_n[m,0] - x_start[m]
        dy_an = x_a_t_n[m,1] - y_start[m]
    A_an = dx_an * np.sin(psi_n[m]) - dy_an * np.cos(psi_n[m])
    B_an = -dx_an * np.cos(psi_n[m]) - dy_an * np.sin(psi_n[m])
    C_an = (-Lambda_y / 2) * np.cos(psi_n[m])
    if A_an > 0:
        gamma_n[m] = np.arccos(-C_an / (np.sqrt(A_an**2 + B_an**2))) - np.arctan(-B_an / A_an)
    else:
        gamma_n[m] = -np.arccos(C_an / (np.sqrt(A_an**2 + B_an**2))) - np.arctan(-B_an / A_an)  
    return gamma_n[m]


def source_pos(gamma_n, PALC_config):
    """
    Calculates the positions of the LSA cabinets for a given tilt angles.
    Called by :any:`calc_angles`.

    Parameters
    ----------
    gamma_n : ndarray
        Set of LSA tilt angles.
    PALC_config : obj
        Configuration of the PALC-algorithm.

    Returns
    -------
    x_start : ndarray
        Highest points of LSA cabinets in x-coordinates.
    y_start : ndarray
        Highest points of LSA cabinets in y-coordinates.
    x_stop : ndarray
        Lowest points of LSA cabinets in x-coordinates.
    y_stop : ndarray
        Lowest points of LSA cabinets in y-coordinates.
    x_src : ndarray
        Center points of LSA cabinets in x-coordinates.
    y_src : ndarray
        Center points of LSA cabinets in y-coordinates.
    x_S : ndarray
        Center of the whole LSA in x-coordinates.
    y_S : ndarray
        Center of the whole LSA in y-coordinates.

    """
    # initialize the information of the PALC_config list
    Lambda_y = PALC_config.Lambda_y
    Lambda_gap = PALC_config.Lambda_gap
    x_H = PALC_config.x_H
    y_H = PALC_config.y_H
    gamma_n = np.reshape(gamma_n,[np.shape(gamma_n)[0],1])
    
    # calculate the source positions
    X = -(Lambda_y + Lambda_gap) * np.sin(gamma_n)
    Y = -(Lambda_y + Lambda_gap) * np.cos(gamma_n)
    X_start = repmat(X, 1, np.shape(X)[0]) * np.triu(np.ones((np.shape(X)[0],np.shape(X)[0]),int),0)
    Y_start = repmat(Y, 1, np.shape(Y)[0]) * np.triu(np.ones((np.shape(Y)[0],np.shape(Y)[0]),int),0)
    x_start = np.transpose(np.sum(X_start,0)) + x_H
    x_start = np.append([x_H],[x_start])
    x_start = np.delete(x_start,[np.shape(x_start)[0]-1])
    y_start = np.transpose(np.sum(Y_start,0)) + y_H
    y_start = np.append([y_H],[y_start])
    y_start = np.delete(y_start,[np.shape(y_start)[0]-1])
    x_stop = x_start - Lambda_y * np.sin(gamma_n[:,0])
    y_stop = y_start - Lambda_y * np.cos(gamma_n[:,0])
    x_src = (x_start + x_stop) / 2
    y_src = (y_start + y_stop) / 2
    x_S = (x_start[0] + x_stop[np.shape(x_stop)[0]-1]) / 2
    y_S = (y_start[0] + y_stop[np.shape(x_stop)[0]-1]) / 2
    
    return x_start, y_start, x_stop, y_stop, x_src, y_src, x_S, y_S

 
def calc_intersection_PAL(xy_1_a, xy_1_b, xy_2_a, xy_2_b):
    """
    Calculates intersection points of the polygonal audience lines
    This function is necessary for the handling of gaps in the audience lines and
    is used to create the non-audience lines.
    Calculation of the intersection of two adjacent sections of the polygonal
    audience line based on the intersection of two linear functions in vector
    representation. Called by :any:`calc_angles`, :any:`pal_draw_condition_3`
    and :any:`suggestPAL`.
    
    
        * vec{OP_1} = vec{OA_1} + lambda_1 * ( vec{OB_1} - vec{OA_1} )
        * vec{OP_2} = vec{OA_2} + lambda_2 * ( vec{OB_2} - vec{OA_2} )

        * vec{OA_1} = [ xy_1_a(1) xy_1_a(2) ]
        * vec{OB_1} = [ xy_1_b(1) xy_1_b(2) ]
        * vec{OA_2} = [ xy_2_a(1) xy_2_a(2) ]
        * vec{OB_2} = [ xy_2_b(1) xy_2_b(2) ]

    Parameters
    ----------
    xy_1_a : list
        Start position (x, y) of the pal section 1.
    xy_1_b : list
        Stop position (x, y) of the pal section 1.
    xy_2_a : list
        Start position (x, y) of the pal section 2.
    xy_2_b : list
        Stop position (x, y) of the pal section 2.

    Returns
    -------
    xy_s : list
        Intersection (x, y) of pal section 1 and pal section 2.
    lambda_1 : float
        Vector parameter 1 for the intersection.
    lambda_2 : float
        Vector parameter 2 for the intersection.
    point_exist : bool
        True, if a point of intersection exists. Otherwise False.

    """
    # initialize
    xy_s_1 = np.zeros(2)
    xy_s_2 = np.zeros(2)
    xy_s = np.zeros(2)
    if xy_1_a[0] != xy_1_b[0] and xy_2_a[0] != xy_2_b[0]:
        slope_1 = (xy_1_a[1] - xy_1_b[1]) / (xy_1_a[0] - xy_1_b[0])
        slope_2 = (xy_2_a[1] - xy_2_b[1]) / (xy_2_a[0] - xy_2_b[0])
    elif xy_1_a[0] == xy_1_b[0] and xy_2_a[0] != xy_2_b[0]:
        slope_1 = 5000
        slope_2 = (xy_2_a[1] - xy_2_b[1]) / (xy_2_a[0] - xy_2_b[0])
    elif xy_1_a[0] != xy_1_b[0] and xy_2_a[0] == xy_2_b[0]:
        slope_1 = (xy_1_a[1] - xy_1_b[1]) / (xy_1_a[0] - xy_1_b[0])
        slope_2 = 4900
    else:
        slope_1 = 5000
        slope_2 = 4900
    # check if the slope of both lines is infinity or slope of both lines is equal
    if (xy_1_a[0] == xy_1_b[0] and xy_2_a[0] == xy_2_b[0]) or \
    (np.abs(slope_1 - slope_2) <= 10**(-5)):
        point_exist = False
        lambda_1 = 0
        lambda_2 = 0
        xy_s = [0,0]
    else:
        # calculate vector parameters lambda_1 and lambda_2 for the
        # intersection position
        lambda_1 = ((xy_2_a[0] - xy_1_a[0]) * (xy_2_b[1] - xy_2_a[1]) + \
                    (xy_2_b[0] - xy_2_a[0]) * (xy_1_a[1] - xy_2_a[1])) / \
                   ((xy_1_b[0] - xy_1_a[0]) * (xy_2_b[1] - xy_2_a[1]) + \
                    (xy_2_b[0] - xy_2_a[0]) * (xy_1_a[1] - xy_1_b[1]))
        
        lambda_2 = ((xy_1_b[0] - xy_1_a[0]) * (xy_1_a[1] - xy_2_a[1]) + \
                    (xy_2_a[0] - xy_1_a[0]) * (xy_1_b[1] - xy_1_a[1])) / \
                   ((xy_1_b[0] - xy_1_a[0]) * (xy_2_b[1] - xy_2_a[1]) + \
                    (xy_2_b[0] - xy_2_a[0]) * (xy_1_a[1] - xy_1_b[1]))    
        
        # calculate intersection position based on the computed lambda_1 for pal section 1
        xy_s_1[0] = xy_1_a[0] + lambda_1 * (xy_1_b[0] - xy_1_a[0])
        xy_s_1[1] = xy_1_a[1] + lambda_1 * (xy_1_b[1] - xy_1_a[1])
        # calculate intersction position based on the computed lambda_2 for pal section 2
        xy_s_2[0] = xy_2_a[0] + lambda_2 * (xy_2_b[0] - xy_2_a[0])
        xy_s_2[1] = xy_2_a[1] + lambda_2 * (xy_2_b[1] - xy_2_a[1])
        # check of calculation: do the computed lambda_1 and lambda_2 result in the 
        # same intersection position -- x-coordinates
        if np.abs(xy_s_1[0] - xy_s_2[0]) < 10**(-8):
            xy_s[0] = xy_s_1[0]
        else:
            xy_s[0] = 0
        # check of calculation: do the computed lambda_1 and lambda_2 result in the 
        # same intersection position -- y-coordinates
        if np.abs(xy_s_1[1] - xy_s_2[1]) < 10**(-8):
            xy_s[1] = xy_s_1[1]
        else:
            xy_s[1] = 0
        point_exist = True
    
    return xy_s, lambda_1, lambda_2, point_exist


def LSA_visualization(PALC_plots, PALC_config, gamma_n, N_margin=[]):
    """
    Calculates the positions of the LSA cabinet edges to visualize them in a
    bokeh figure as patches. Called by :any:`start_calc`.

    Parameters
    ----------
    PALC_plots : obj [out]
        Contains plotting data of the PALC results.
    PALC_config : obj [in]
        Contains the PALC configuration.
    gamma_n : ndarray
        Absolute tilt angles of the LSA cabinets in degree.
    N_margin : int, optional
        If not all LSA cabinets shall be visualized. The default is [].

    Returns
    -------
    x_patches : list
        x-coordinates of the LSA patches.
    y_patches : list
        y-coordinates of the LSA patches.

    """
    if N_margin == []:
        N = PALC_config.N
    else:
        N = N_margin
    # Array visualization
    # assign variables
    x_start, x_stop, y_start, y_stop = np.zeros(N), np.zeros(N), \
                                       np.zeros(N), np.zeros(N)
    x_start[0], y_start[0] = PALC_config.x_H, PALC_config.y_H
    x_stop[0] = x_start[0] - PALC_config.Lambda_y * np.sin(gamma_n[0])
    y_stop[0] = y_start[0] - PALC_config.Lambda_y * np.cos(gamma_n[0])
    for n in range(1,N):
        x_start[n] = x_stop[n-1] - PALC_config.Lambda_gap * np.sin(gamma_n[n-1])
        y_start[n] = y_stop[n-1] - PALC_config.Lambda_gap * np.cos(gamma_n[n-1])
        x_stop[n] = x_start[n] - PALC_config.Lambda_y * np.sin(gamma_n[n])
        y_stop[n] = y_start[n] - PALC_config.Lambda_y * np.cos(gamma_n[n])
        
    PALC_plots.x_start, PALC_plots.x_stop = x_start, x_stop
    PALC_plots.y_start, PALC_plots.y_stop = y_start, y_stop
    PALC_plots.x_start_b = list(np.array(PALC_plots.x_start) - 1*PALC_config.Lambda_y * (np.cos(gamma_n) + np.sin(gamma_n)*0.25))
    PALC_plots.x_stop_b = list(np.array(PALC_plots.x_stop) - 1*PALC_config.Lambda_y * (np.cos(gamma_n) - np.sin(gamma_n)*0.25))
    PALC_plots.y_start_b = list(np.array(PALC_plots.y_start) + 1*PALC_config.Lambda_y * (np.sin(gamma_n) - np.cos(gamma_n)*0.25))
    PALC_plots.y_stop_b = list(np.array(PALC_plots.y_stop) + 1*PALC_config.Lambda_y * (np.sin(gamma_n) + np.cos(gamma_n)*0.25))

    x_patches, y_patches = [], []
    for n in range(np.shape(PALC_plots.x_start)[0]):
        x_patches.append([PALC_plots.x_start[n], PALC_plots.x_stop[n], PALC_plots.x_stop_b[n], PALC_plots.x_start_b[n]])
        y_patches.append([PALC_plots.y_start[n], PALC_plots.y_stop[n], PALC_plots.y_stop_b[n], PALC_plots.y_start_b[n]])
    return x_patches, y_patches


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
                    psi_n[:] = psi_n[:] * (1 - np.sign(Gamma_aud) * 0.05)
                elif np.abs(Gamma_aud) > 8:
                    psi_n[:] = psi_n[:] * (1 - np.sign(Gamma_aud) * 0.1)
            
        
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
                    val = np.abs(PALC_config.gamma_LSA - (gamma_n[m] - PALC_config.laste_angle_hm))
                    ind = np.unravel_index(np.argmin(val, axis=None), val.shape)
                    gamma_n[m] = PALC_config.gamma_LSA[ind[0]] + PALC_config.last_angle_hm
            elif m > 0:
                x_a_t_n[m,:] = x_a_b_n[m-1,:]
                gamma_n[m] = calc_gamma(gamma_n, psi_n, x_start, y_start, x_a_t_n, PALC_config.Lambda_y, m)

                ################## DISCRETE TILT ANGLES ###################
                if  PALC_config.use_gamma_LSA and np.shape(PALC_config.gamma_LSA)[0] > 1:
                    ######### Calculate The Tilt Angles By Usage Of The Set Of Discrete Tilt Angles #######
                    val = np.abs(PALC_config.gamma_LSA - (gamma_n[m] - gamma_n[m-1]))
                    ind = np.unravel_index(np.argmin(val, axis=None), val.shape)
                    gamma_n[m] = PALC_config.gamma_LSA[ind[0]] + gamma_n[m-1]

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
                            print('Error in discrete tilt angles')
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
                                               
        ###### Iterative optimization of weighting_nu may be done here!!!
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

    #PALC_plots  = PALC_classes.PALC_plotting()
    PALC_plots.update_plot_array(x_c_n=x_c_n, y_c_n=y_c_n, x_fin_unitn=x_fin_unitn, y_fin_unitn=y_fin_unitn)
    Opt_arr.update_opt_array(gamma_n=gamma_n, psi_n=psi_n, num_iter=num_iter, \
                             Gamma_aud=Gamma_aud, thr_dist = thr_dist, \
                             gamma_tilt_deg=gamma_tilt_deg)
  
    # set new gamma_n
    setattr(PALC_config,'gamma_n',gamma_n)
    return
