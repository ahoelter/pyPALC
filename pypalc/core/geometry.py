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


def inter_cabinet_angles(tilt_angles_deg):
    """
    Calculates the intercabinet tilt angles. Called by :any:`start_calc`.

    Parameters
    ----------
    tilt_angles_deg : ndarray
        Computed absolute tilt angles in degree.

    Returns
    -------
    ndarray
        Rounded intercabinet tilt angles and absolute tilt angles in degree.

    """
    inter_cabinet_angles_deg = np.zeros(np.shape(tilt_angles_deg)[0],float)
    inter_cabinet_angles_deg[0] = tilt_angles_deg[0]
    for n in range(1, np.shape(tilt_angles_deg)[0]):
        inter_cabinet_angles_deg[n] =  tilt_angles_deg[n] - tilt_angles_deg[n-1]
    return np.round(inter_cabinet_angles_deg, decimals=2) , np.round(tilt_angles_deg, decimals=2)


def progressive_tilt_angles_deg(cabinet_count, discrete_tilt_angles=[], top_angle_deg = 0, final_angle_deg = 38, use_discrete_angles = "No"):
    """
    Calculates the absolute tilt angles of a progressive LSA curving. Called by
    :any:`ref_array_angles`.

    Parameters
    ----------
    cabinet_count : int
        Number of loudspeaker array cabinets.
    discrete_tilt_angles : list, optional
        Possible discrete intercabinet tilt angles, if use_discrete_angles is set to
        'Yes'. The default is [].
    top_angle_deg : float, optional
        Tilt angle of the highest (first) LSA cabinet in degree.
        The default is 0.
    final_angle_deg : float, optional
        Angle of the last tilt angle in degree. The default is 38.
    use_discrete_angles : str, optional
        If discrete tilt angles shall be used: 'Yes'. The default is 'No'.

    Returns
    -------
    tilt_angles_deg : ndarray
        Array with the progressive LSA curving in degree.

    """
    tilt_angles_deg = np.zeros(cabinet_count)   
    # calculate angle_step_deg with given terminal angle final_angle_deg
    angle_step_deg = (final_angle_deg-top_angle_deg) / ((cabinet_count/2)*(cabinet_count-1))
    #angle_step_deg = 7/cabinet_count
    # calculate cabinet tilt angle for all LSA cabinets
    for n in range(cabinet_count):
        tilt_angles_deg[n] = top_angle_deg + ((n+1)/2)*(n)*angle_step_deg
        if np.shape(discrete_tilt_angles)[0] > 1 and n > 0 and use_discrete_angles == "Yes":
            val = np.abs(discrete_tilt_angles / (np.pi / 180) - (tilt_angles_deg[n] - tilt_angles_deg[n-1]))
            ind = np.unravel_index(np.argmin(val, axis=None), val.shape)
            tilt_angles_deg[n] = discrete_tilt_angles[ind[0]] / (np.pi / 180) + tilt_angles_deg[n-1]
    return tilt_angles_deg


def arc_tilt_angles_deg(cabinet_count, discrete_tilt_angles=[], top_angle_deg = 0, inter_cabinet_angle_deg = 1, use_discrete_angles = "No"):
    """
    Calculates the absolute tilt angles of an arc LSA curving. Called by
    :any:`ref_array_angles`.

    Parameters
    ----------
    cabinet_count : int
        Number of loudspeaker array cabinets.
    discrete_tilt_angles : list, optional
        Possible discrete intercabinet tilt angles, if use_discrete_angles is set to
        'Yes'. The default is [].
    top_angle_deg : float, optional
        Tilt angle of the highest (first) LSA cabinet in degree.
        The default is 0.
    inter_cabinet_angle_deg : float, optional
        Angle of the intercabinet angles in degree. The default is 1.
    use_discrete_angles : str, optional
        If discrete tilt angles shall be used: 'Yes'. The default is 'No'.

    Returns
    -------
    tilt_angles_deg : ndarray
        Array with the progressive LSA curving in degree.

    """
    tilt_angles_deg = np.zeros(cabinet_count)
    if np.shape(discrete_tilt_angles)[0] > 1 and use_discrete_angles == "Yes":
        inter_cabinet_angle_deg = discrete_tilt_angles[np.argmin(np.abs((discrete_tilt_angles/(np.pi/180))-inter_cabinet_angle_deg))]*(180/np.pi)
    for n in range(cabinet_count):
        tilt_angles_deg[n] = top_angle_deg + n * inter_cabinet_angle_deg
    return tilt_angles_deg


def cabinet_tilt_angle(tilt_angles, aperture_angles, x_start, y_start, hit_points_start, cabinet_height, m=0):
    """
    Calculates a specific loudspeaker tilt angle of a LSA cabinet. Called by
    :any:`calc_angles`.

    Parameters
    ----------
    tilt_angles : ndarray
        Set of LSA tilt angles.
    aperture_angles : float
        Splay / aperture angle of the loudspeaker cabinets or of the m-th LS cabinet.
    x_start : float
        Highest point of the LSA cabinet in x-coordinate.
    y_start : float
        Highest point of the LSA cabinet in y-coordinate.
    hit_points_start : float
        Top position of covered audience line by m-th LSA cabinet (x- and y-coordinate).
    cabinet_height : float
        Height of m-th LSA cabinet.
    m : int, optional
        Variable which LSA cabinet is optimized. The default is 0.

    Returns
    -------
    float
        Tilt angle of m-th LSA cabinet in degree.

    """
    if m == 0:
        dx_an = hit_points_start[m,0] - x_start
        dy_an = hit_points_start[m,1] - y_start
    else:
        dx_an = hit_points_start[m,0] - x_start[m]
        dy_an = hit_points_start[m,1] - y_start[m]
    A_an = dx_an * np.sin(aperture_angles[m]) - dy_an * np.cos(aperture_angles[m])
    B_an = -dx_an * np.cos(aperture_angles[m]) - dy_an * np.sin(aperture_angles[m])
    C_an = (-cabinet_height / 2) * np.cos(aperture_angles[m])
    if A_an > 0:
        tilt_angles[m] = np.arccos(-C_an / (np.sqrt(A_an**2 + B_an**2))) - np.arctan(-B_an / A_an)
    else:
        tilt_angles[m] = -np.arccos(C_an / (np.sqrt(A_an**2 + B_an**2))) - np.arctan(-B_an / A_an)  
    return tilt_angles[m]


def source_positions(tilt_angles, config):
    """
    Calculates the positions of the LSA cabinets for a given tilt angles.
    Called by :any:`calc_angles`.

    Parameters
    ----------
    tilt_angles : ndarray
        Set of LSA tilt angles.
    config : obj
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
    cabinet_centers_x : ndarray
        Center points of LSA cabinets in x-coordinates.
    cabinet_centers_y : ndarray
        Center points of LSA cabinets in y-coordinates.
    array_center_x : ndarray
        Center of the whole LSA in x-coordinates.
    array_center_y : ndarray
        Center of the whole LSA in y-coordinates.

    """
    # initialize the information of the config list
    cabinet_height = config.Lambda_y
    cabinet_gap = config.Lambda_gap
    top_x = config.x_H
    top_y = config.y_H
    tilt_angles = np.reshape(tilt_angles,[np.shape(tilt_angles)[0],1])
    
    # calculate the source positions
    X = -(cabinet_height + cabinet_gap) * np.sin(tilt_angles)
    Y = -(cabinet_height + cabinet_gap) * np.cos(tilt_angles)
    X_start = repmat(X, 1, np.shape(X)[0]) * np.triu(np.ones((np.shape(X)[0],np.shape(X)[0]),int),0)
    Y_start = repmat(Y, 1, np.shape(Y)[0]) * np.triu(np.ones((np.shape(Y)[0],np.shape(Y)[0]),int),0)
    x_start = np.transpose(np.sum(X_start,0)) + top_x
    x_start = np.append([top_x],[x_start])
    x_start = np.delete(x_start,[np.shape(x_start)[0]-1])
    y_start = np.transpose(np.sum(Y_start,0)) + top_y
    y_start = np.append([top_y],[y_start])
    y_start = np.delete(y_start,[np.shape(y_start)[0]-1])
    x_stop = x_start - cabinet_height * np.sin(tilt_angles[:,0])
    y_stop = y_start - cabinet_height * np.cos(tilt_angles[:,0])
    cabinet_centers_x = (x_start + x_stop) / 2
    cabinet_centers_y = (y_start + y_stop) / 2
    array_center_x = (x_start[0] + x_stop[np.shape(x_stop)[0]-1]) / 2
    array_center_y = (y_start[0] + y_stop[np.shape(x_stop)[0]-1]) / 2
    
    return x_start, y_start, x_stop, y_stop, cabinet_centers_x, cabinet_centers_y, array_center_x, array_center_y


def line_intersection(xy_1_a, xy_1_b, xy_2_a, xy_2_b):
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
    point_exists : bool
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
        point_exists = False
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
        point_exists = True
    
    return xy_s, lambda_1, lambda_2, point_exists


def cabinet_outlines(ray_plot, config, tilt_angles, margin_cabinet_count=[]):
    """
    Calculates the positions of the LSA cabinet edges to visualize them in a
    bokeh figure as patches. Called by :any:`start_calc`.

    Parameters
    ----------
    ray_plot : obj [out]
        Contains plotting data of the PALC results.
    config : obj [in]
        Contains the PALC configuration.
    tilt_angles : ndarray
        Absolute tilt angles of the LSA cabinets in degree.
    margin_cabinet_count : int, optional
        If not all LSA cabinets shall be visualized. The default is [].

    Returns
    -------
    x_patches : list
        x-coordinates of the LSA patches.
    y_patches : list
        y-coordinates of the LSA patches.

    """
    if margin_cabinet_count == []:
        cabinet_count = config.N
    else:
        cabinet_count = margin_cabinet_count
    # Array visualization
    # assign variables
    x_start, x_stop, y_start, y_stop = np.zeros(cabinet_count), np.zeros(cabinet_count), \
                                       np.zeros(cabinet_count), np.zeros(cabinet_count)
    x_start[0], y_start[0] = config.x_H, config.y_H
    x_stop[0] = x_start[0] - config.Lambda_y * np.sin(tilt_angles[0])
    y_stop[0] = y_start[0] - config.Lambda_y * np.cos(tilt_angles[0])
    for n in range(1,cabinet_count):
        x_start[n] = x_stop[n-1] - config.Lambda_gap * np.sin(tilt_angles[n-1])
        y_start[n] = y_stop[n-1] - config.Lambda_gap * np.cos(tilt_angles[n-1])
        x_stop[n] = x_start[n] - config.Lambda_y * np.sin(tilt_angles[n])
        y_stop[n] = y_start[n] - config.Lambda_y * np.cos(tilt_angles[n])
        
    ray_plot.x_start, ray_plot.x_stop = x_start, x_stop
    ray_plot.y_start, ray_plot.y_stop = y_start, y_stop
    ray_plot.x_start_b = list(np.array(ray_plot.x_start) - 1*config.Lambda_y * (np.cos(tilt_angles) + np.sin(tilt_angles)*0.25))
    ray_plot.x_stop_b = list(np.array(ray_plot.x_stop) - 1*config.Lambda_y * (np.cos(tilt_angles) - np.sin(tilt_angles)*0.25))
    ray_plot.y_start_b = list(np.array(ray_plot.y_start) + 1*config.Lambda_y * (np.sin(tilt_angles) - np.cos(tilt_angles)*0.25))
    ray_plot.y_stop_b = list(np.array(ray_plot.y_stop) + 1*config.Lambda_y * (np.sin(tilt_angles) + np.cos(tilt_angles)*0.25))

    x_patches, y_patches = [], []
    for n in range(np.shape(ray_plot.x_start)[0]):
        x_patches.append([ray_plot.x_start[n], ray_plot.x_stop[n], ray_plot.x_stop_b[n], ray_plot.x_start_b[n]])
        y_patches.append([ray_plot.y_start[n], ray_plot.y_stop[n], ray_plot.y_stop_b[n], ray_plot.y_start_b[n]])
    return x_patches, y_patches


def reference_tilt_angles_deg(ref_type, cabinet_count, discrete_tilt_angles, start_deg, step_stop_deg,
                              use_discrete_angles, user_angles_deg=()):
    """
    Calculates the tilt angles of the reference LSA (formerly the computation
    in ``ref_array_angles`` of ``fun/gui_ls_array.py``). Depending on the
    array type, calls :any:`progressive_tilt_angles_deg` or :any:`arc_tilt_angles_deg`.

    Parameters
    ----------
    ref_type : str
        'straight', 'progressive', 'arc' or 'user_defined'.
    cabinet_count : int
        Number of LSA cabinets.
    discrete_tilt_angles : list or ndarray
        Discrete intercabinet tilt angles in radian ([] if not used).
    start_deg : float
        Tilt angle of the highest LSA cabinet in degree (straight,
        progressive, arc).
    step_stop_deg : float
        Final angle (progressive) or intercabinet angle (arc) in degree.
    use_discrete_angles : bool
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
    use_discrete_angles = "Yes" if use_discrete_angles else "No"
    if ref_type == 'straight':
        return np.ones(cabinet_count) * float(start_deg)
    elif ref_type == 'progressive':
        return progressive_tilt_angles_deg(cabinet_count, discrete_tilt_angles, float(start_deg),
                                      float(step_stop_deg), use_discrete_angles)
    elif ref_type == 'arc':
        return arc_tilt_angles_deg(cabinet_count, discrete_tilt_angles, float(start_deg),
                              float(step_stop_deg), use_discrete_angles)
    elif ref_type == 'user_defined':
        tilt_angles_deg = np.array([float(s) for s in user_angles_deg])
        if np.shape(tilt_angles_deg)[0] == 0:
            raise ValueError('no user defined tilt angles given')
        # check if too less or many tilt angles are given by the user
        surplus = np.shape(tilt_angles_deg)[0] - cabinet_count
        if surplus < 0:
            tilt_angles_deg = np.append(tilt_angles_deg,
                                       np.ones(np.abs(surplus))*tilt_angles_deg[-1])
        elif surplus > 0:
            tilt_angles_deg = tilt_angles_deg[:cabinet_count]
        return tilt_angles_deg
    raise ValueError('unknown reference array type %r' % ref_type)
