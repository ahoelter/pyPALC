"""Geometry of the loudspeaker array and of the audience lines.

Source positions of the cabinets, intersections of audience lines, the
cabinet outlines for plotting and the curvings of the reference array
(formerly ``fun/PALC_functions.py`` without ``calc_angles``, plus the
computation of ``ref_array_angles`` from ``fun/gui_ls_array.py``).

author: Arne Hoelter
"""
from __future__ import annotations

import numpy as np


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
    representation. Called by :any:`calc_angles`, :any:`draw_condition_3`
    and :any:`suggest_venue`.
    
    
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


def reference_tilt_angles_deg(ref_type, N, gamma_LSA, start_deg, step_stop_deg,
                              use_dangles, user_angles_deg=()):
    """
    Calculates the tilt angles of the reference LSA (formerly the computation
    in ``ref_array_angles`` of ``fun/gui_ls_array.py``). Depending on the
    array type, calls :any:`calc_progressive_array` or :any:`calc_arc_array`.

    Parameters
    ----------
    ref_type : str
        'straight', 'progressive', 'arc' or 'user_defined'.
    N : int
        Number of LSA cabinets.
    gamma_LSA : list or ndarray
        Discrete intercabinet tilt angles in radian ([] if not used).
    start_deg : float
        Tilt angle of the highest LSA cabinet in degree (straight,
        progressive, arc).
    step_stop_deg : float
        Final angle (progressive) or intercabinet angle (arc) in degree.
    use_dangles : bool
        If the discrete tilt angles shall be used (progressive, arc).
    user_angles_deg : sequence of float, optional
        User defined tilt angles in degree. Missing angles are filled with the
        last one, surplus ones are dropped. The default is ().

    Returns
    -------
    ndarray
        Tilt angles of the reference LSA in degree.

    Raises
    ------
    ValueError
        If the type is unknown or no user defined angles are given.

    """
    use_dangles = "Yes" if use_dangles else "No"
    if ref_type == 'straight':
        return np.ones(N) * float(start_deg)
    elif ref_type == 'progressive':
        return calc_progressive_array(N, gamma_LSA, float(start_deg),
                                      float(step_stop_deg), use_dangles)
    elif ref_type == 'arc':
        return calc_arc_array(N, gamma_LSA, float(start_deg),
                              float(step_stop_deg), use_dangles)
    elif ref_type == 'user_defined':
        gamma_tilt_deg = np.array([float(s) for s in user_angles_deg])
        if np.shape(gamma_tilt_deg)[0] == 0:
            raise ValueError('no user defined tilt angles given')
        # check if too less or many tilt angles are given by the user
        diff2N = np.shape(gamma_tilt_deg)[0] - N
        if diff2N < 0:
            gamma_tilt_deg = np.append(gamma_tilt_deg,
                                       np.ones(np.abs(diff2N))*gamma_tilt_deg[-1])
        elif diff2N > 0:
            gamma_tilt_deg = gamma_tilt_deg[:N]
        return gamma_tilt_deg
    raise ValueError('unknown reference array type %r' % ref_type)
