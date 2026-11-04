"""The PALC algorithm.

Configuration and result classes of the PALC algorithm, the main routine
:func:`compute_tilt_angles` and the gap handling wrapper :func:`run_palc`
(formerly ``fun/run_palc.py``, ``compute_tilt_angles`` from
``fun/PALC_functions.py``, the configuration and array classes from
``fun/PALC_classes.py`` and ``fixed_angle_mid_max`` from
``fun/gui_algorithm_sfp.py``).

author: Arne Hoelter
"""
from __future__ import annotations

import logging
import time
from copy import copy

import numpy as np

from .geometry import (cabinet_outlines, cabinet_tilt_angle, line_intersection,
                       source_positions)

logger = logging.getLogger(__name__)


class PalcConfig():
    """ Class to create a configuration object of the PALC algorithm. """
    def __init__(self, cabinet_gap=0, cabinet_height=0.2, top_x=0, top_y=2, palc_variant='PALC 2', use_discrete_angles=False, discrete_tilt_angles=None, \
                 use_weighting='Without', weighting_nu=1, weighting_factors=None, cabinet_count=int(4), gap_handling='Without', soft_margin_strength=1, \
                 hard_margin_last_angle=None, gap_weights=None, weighting_weights=None, use_fixed_angle=False, fixed_angle=0, \
                 directivity = 'Circular Piston', aperture_angles=None, tilt_angles=None, tolerance=1.0, frequency_range_hz=(200,8000)):
        # None stands for a new list with the default value (no shared mutable defaults)
        discrete_tilt_angles, weighting_factors, hard_margin_last_angle, gap_weights, weighting_weights = \
            [[] if val is None else val for val in
             (discrete_tilt_angles, weighting_factors, hard_margin_last_angle, gap_weights, weighting_weights)]
        aperture_angles = [0.0524] if aperture_angles is None else aperture_angles
        tilt_angles = [0] if tilt_angles is None else tilt_angles
        self.top_x, self.top_y , self.cabinet_gap, self.cabinet_height = top_x, top_y, cabinet_gap, cabinet_height
        self.tolerance, self.aperture_angles, self.tilt_angles            = tolerance, aperture_angles, tilt_angles
        self.palc_variant, self.cabinet_count, self.directivity           = palc_variant, cabinet_count, directivity
        self.use_weighting, self.weighting_nu               = use_weighting, weighting_nu
        self.weighting_factors, self.weighting_weights      = weighting_factors, weighting_weights
        self.gap_weights, self. gap_handling                = gap_weights, gap_handling
        self.soft_margin_strength, self.hard_margin_last_angle                = soft_margin_strength, hard_margin_last_angle
        self.use_fixed_angle, self.fixed_angle              = use_fixed_angle, fixed_angle
        self.use_discrete_angles, self.discrete_tilt_angles                  = use_discrete_angles, discrete_tilt_angles
        self.frequency_range_hz                                     = frequency_range_hz


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
        if new_input[0] in ['top_x', 'top_y', 'fixed_angle', 'cabinet_gap', 'cabinet_height', 'tolerance']:
            try:
                val = float(new_input[1])
            except (TypeError, ValueError):
                return 'Error: only numbers are allowed. Set to Default', new_input[0]
        elif new_input[0] in ['cabinet_count', 'aperture_angles']:
            try:
                if new_input[0] in ['cabinet_count']: 
                    val = int(new_input[1])
                if new_input[0] in ['aperture_angles']:
                    val = self.cabinet_count
                    self.aperture_angles = [float(new_input[1])*np.pi/180]
                self.aperture_angles = np.linspace(self.aperture_angles[0], self.aperture_angles[0], num=val)
                self.tilt_angles = np.linspace(-0.0428, 0.7147, num=val)
                self.weighting_factors = np.ones(val)
                # more than one LS necessary and maximum of 30
                if val == 0: return 'Error: number of LS must be bigger than 0. Set to default', new_input[0]
                if val > 30: return 'Error: maximum amount of LS is 30. Set to default', new_input[0]
                if new_input[0] in ['aperture_angles']: return '', None
            except (TypeError, ValueError):
                return 'Error: only numbers are allowed. Set to default', new_input[0]
        elif new_input[0] in ['soft_margin_strength', 'fixed_angle']:
            val = float(new_input[1])
        
        if hasattr(self, new_input[0]) and 'val' in locals():
            setattr(self, new_input[0], val)
        elif hasattr(self, new_input[0]):
            setattr(self, new_input[0], new_input[1])
        else:            
            logger.error('PalcConfig has no attribute %s', new_input[0])
        
        # array must be smaller than top_y
        if self.cabinet_count * (self.cabinet_height + self.cabinet_gap) > self.top_y:
            self.top_y = np.round(self.cabinet_count * (self.cabinet_height + self.cabinet_gap), \
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
        self.gap_weights = np.ones(self.cabinet_count)
        # Set weights depending on initial weighting_nu
        if self.use_weighting in ['Linear Spacing']:
            self.weighting_weights = np.linspace(1, self.weighting_nu, num=self.cabinet_count)
        elif self.use_weighting in ['Logarithmic Spacing']:
            self.weighting_weights = np.geomspace(1, self.weighting_nu, num=self.cabinet_count)
        else:
            self.weighting_weights = np.ones(self.cabinet_count)
        # Set initial weighting factors
        self.weighting_factors = self.weighting_weights * self.gap_weights


class RayPlotData():
    """ Class to create objects that contain data for visualization in bokeh
        figures. """
    def __init__(self):
        self.cabinet_centers_x, self.cabinet_centers_y, self.x_start, self.x_stop             = [], [], [], []
        self.y_start, self.y_stop, self.center_ray_ends_x, self.center_ray_ends_y = [], [], [], []
        self.x_start_b = []
        
        self.y_stop_unitn_stop, self.center_rays_x, self.center_rays_y    = [], [], []
        self.start_rays_x, self.start_rays_y                  = [], []
        self.stop_rays_x, self.stop_rays_y                  = [], []

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
                
    def get_plot_array(self, optimized_array, segments=None):
        """
        Includes the results of the PALC optimization to plot them in a
        bokeh figure.

        Each ray (start, center, stop) goes from the cabinet center along
        its direction. PALC computes its end where the ray is assumed to
        hit the venue (for the start ray: the hit point of the stop ray of
        the cabinet above). With discrete tilt angles or a soft margin the
        ray does not pass through that point, so with ``segments`` the ray
        ends where it first crosses a venue line instead.

        Parameters
        ----------
        optimized_array : obj [in]
            Data of the optimized array by PALC.
        segments : list of ndarray, optional
            Venue lines the rays end on (audience and non-audience lines,
            each with its points in the rows). The default is None: the
            rays end at the points computed by PALC.

        Returns
        -------
        None.

        """
        ends = {'center': (self.center_ray_ends_x, self.center_ray_ends_y),
                'start': (optimized_array.start_ray_ends_x,
                          optimized_array.start_ray_ends_y),
                'stop': (optimized_array.stop_ray_ends_x,
                         optimized_array.stop_ray_ends_y)}
        for name, (ends_x, ends_y) in ends.items():
            if segments:
                ends_x, ends_y = zip(*[
                    ray_end_on_venue(x0, y0, x1, y1, segments)
                    for x0, y0, x1, y1 in zip(self.cabinet_centers_x,
                                              self.cabinet_centers_y,
                                              ends_x, ends_y)])
            setattr(self, name + '_rays_x',
                    list(zip(self.cabinet_centers_x, ends_x)))
            setattr(self, name + '_rays_y',
                    list(zip(self.cabinet_centers_y, ends_y)))


def ray_end_on_venue(x0, y0, x1, y1, segments):
    """
    End point of a ray on the venue lines.

    The ray starts at ``(x0, y0)`` and passes through ``(x1, y1)``. Its end
    is the first point, seen from the start, where it crosses one of the
    venue lines.

    Parameters
    ----------
    x0, y0 : float
        Start of the ray (cabinet center).
    x1, y1 : float
        A point on the ray that gives its direction.
    segments : list of ndarray
        Venue lines; each is a straight line from its first to its last
        point (rows of x, y, ...).

    Returns
    -------
    tuple of float
        End point ``(x, y)``. A ray that crosses no venue line (it passes
        over or under the venue) ends where it passes the farthest point of
        the venue in x; ``(x1, y1)`` if it does not get there either.

    """
    direction = np.array([x1 - x0, y1 - y0], dtype=float)
    nearest = None
    for segment in segments:
        a = np.asarray(segment[0][:2], dtype=float)
        line = np.asarray(segment[-1][:2], dtype=float) - a
        denominator = direction[0] * line[1] - direction[1] * line[0]
        if denominator == 0:
            continue  # parallel
        w = a - [x0, y0]
        along_ray = (w[0] * line[1] - w[1] * line[0]) / denominator
        along_line = (w[0] * direction[1] - w[1] * direction[0]) / denominator
        if along_ray > 0 and -1e-9 <= along_line <= 1 + 1e-9 \
                and (nearest is None or along_ray < nearest):
            nearest = along_ray
    if nearest is None:
        farthest_x = max(max(segment[0][0], segment[-1][0])
                         for segment in segments)
        if direction[0] <= 0 or farthest_x <= x0:
            return x1, y1
        nearest = (farthest_x - x0) / direction[0]
    return x0 + nearest * direction[0], y0 + nearest * direction[1]


class ArrayGeometry():
    """ Class to create an object with optimized array data. """
    def __init__(self, tilt_angles_deg=None):
        if tilt_angles_deg is None:
            tilt_angles_deg = []
        self.tilt_angles, self.tilt_angles_deg    = [], tilt_angles_deg
        self.inter_cabinet_angles_deg, self.aperture_angles = [], []
        self.coverage_error, self.hit_distances        = [], []
        self.iteration_count                        = []
        # deg_opt
        self.start_ray_intercepts, self.stop_ray_intercepts            = [], []
        self.start_ray_ends_x, self.stop_ray_ends_x      = [], []
        self.start_ray_ends_y, self.stop_ray_ends_y      = [], []
        self.hit_points, self.hit_points_start, self.hit_points_stop = [], [], []
        # fixed first angle (max, mid, min)
        self.fixed_angle = [0,0,0]
        if len(self.tilt_angles_deg) > 0:
            self.tilt_angles = self.tilt_angles_deg*np.pi/180


    def update_opt_array(self, cabinet_centers_x=None, cabinet_centers_y=None, **new_input):
        """
        Updates the optimized array object.

        Parameters
        ----------
        cabinet_centers_x : float, optional
            Center of rays at receiver positions on x-axis. The default is None.
        cabinet_centers_y : float, optional
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
        if np.all(cabinet_centers_x != None) and 'aperture_angles' in new_input.keys():
            self.start_ray_intercepts = cabinet_centers_y - np.tan(-self.tilt_angles + self.aperture_angles) * cabinet_centers_x
            self.start_ray_ends_x = new_input['hit_points_start'][:,0]
            self.start_ray_ends_y = np.tan(-self.tilt_angles + self.aperture_angles) \
                                     * self.start_ray_ends_x + self.start_ray_intercepts
            self.stop_ray_intercepts = cabinet_centers_y - np.tan(-self.tilt_angles - self.aperture_angles) * cabinet_centers_x
            self.stop_ray_ends_x = new_input['hit_points_stop'][:,0]
            self.stop_ray_ends_y = np.tan(-self.tilt_angles - self.aperture_angles) \
                                     * self.stop_ray_ends_x + self.stop_ray_intercepts
            self.hit_points = new_input['hit_points']
            self.hit_points_start = new_input['hit_points_start']
            self.hit_points_stop = new_input['hit_points_stop']


def compute_tilt_angles(ray_plot, segments, aperture_angles, tilt_angles, cabinet_count, config, audience_segments, optimized_array):
    """
    Main routine of the PALC computation and is calles by :py:meth:`run_palc`.
    Computes the tilt angles of the LSA cabinets.

    Parameters
    ----------
    ray_plot : obj [out]
        Contains plotting data of the PALC results.
    segments : list
        All audience lines to be considered in PALC (audience lines,
        non-audience lines). Each element contains a audience line as a ndarray.
    aperture_angles : float
        Init splay angle for first PALC loop.
    tilt_angles : ndarray
        Init tilt angles for the first iteration in radian.
    cabinet_count : int
        Number of LSA cabinets.
    config : obj [in]
        Contains PALC Configuration.
    audience_segments : ndarray
        Audience lines without non-audience lines.
    optimized_array : obj [out]
        Result of the PALC algorithm. Optimized LSA data.

    Returns
    -------
    None.

    """
#    try:
    # segment_lengths: length of each audience line segment
    # segment_angles: epsilon
    # segment_points
    # audience_length: total length of the audience line
    segment_lengths = np.empty([len(segments)],float)
    segment_angles = np.empty([len(segments)],float)
    segment_points = np.empty([len(segments)+1,3],float)
    for n in range(len(segments)):
        segment_lengths[n] = np.sqrt(np.sum((segments[n][np.shape(segments[n])[0]-1,:]-segments[n][0,:])**2))
        segment_angles[n] = (segments[n][1,1]-segments[n][0,1]) / (segments[n][1,0]-segments[n][0,0])
        segment_points[n,:] = segments[n][0,:]
    #segment_points[np.shape(pal)[0],:] = pal[np.shape(pal)[0]-1][np.shape(pal[np.shape(pal)[0]-1])[0]-1,:]
    segment_points[-1,:] = segments[-1][-1,:]
    segment_points = np.flipud(segment_points)
    segment_lengths = np.flipud(segment_lengths)
    segment_angles = np.arctan(np.flipud(segment_angles))-np.pi
    audience_length = segment_lengths.sum()
    audience_length_copy = audience_length
    
    covered_cabinet_count = 0
    coverage_error = 0
    iteration_count = 0
    all_cabinets_covered = 0
    d_array = np.zeros(cabinet_count,float)
    audience_overlap = 0 # for discrete tilt angles: overlap on audience lines: to correct the error 'coverage_error'
    error = 0 # just to test
    
    #### begin of the while-loop
    while ( ((np.abs(coverage_error) > config.tolerance or (covered_cabinet_count != cabinet_count) ) and (iteration_count <= 500) ) or (iteration_count < 15) ):
    #while iteration_count == 0:
        # return if number of iteration reaches the maximum (declared above)
        audience_overlap = 0 # for discrete tilt angles: overlap on audience lines: to correct the error 'coverage_error'
        if iteration_count == 500:
            optimized_array.iteration_count = 500
            return
            
        k = 0
        seg_rem = 0
        hit_points = np.zeros([cabinet_count,3],float)
        hit_points_start = np.zeros([cabinet_count,3],float)
        hit_points_stop = np.zeros([cabinet_count,3],float)
        Gamma_n = np.zeros([cabinet_count,3],float)
        
        ####### BEGIN ITERATION STEPS #######
        if iteration_count != 0 and covered_cabinet_count != cabinet_count:
            if all_cabinets_covered == 0:
                aperture_angles[:] = aperture_angles[:] * (1 - 0.1 * cabinet_count / covered_cabinet_count)
            else:
                aperture_angles[:] = aperture_angles[:] * (1 - 0.1 * cabinet_count / covered_cabinet_count)
        elif iteration_count != 0 and covered_cabinet_count == cabinet_count:
            all_cabinets_covered = 1
            if iteration_count <= 15:
                aperture_angles[:] = aperture_angles[:] * (1 - np.sign(coverage_error) * 0.1 / iteration_count)
            elif iteration_count > 15:
                if np.abs(coverage_error) <= 0.2:
                    aperture_angles[:] = aperture_angles[:] * (1 - np.sign(coverage_error) * 0.00015)
                elif np.abs(coverage_error) > 0.2 and np.abs(coverage_error) <= 1.5:
                    aperture_angles[:] = aperture_angles[:] * (1 - np.sign(coverage_error) * 0.002)
                elif np.abs(coverage_error) > 1.5 and np.abs(coverage_error) <= 4:
                    aperture_angles[:] = aperture_angles[:] * (1 - np.sign(coverage_error) * 0.01)
                elif np.abs(coverage_error) > 4 and np.abs(coverage_error) <= 8:
                    aperture_angles[:] = aperture_angles[:] * (1 - np.sign(coverage_error) * 0.03)
                elif np.abs(coverage_error) > 8:
                    aperture_angles[:] = aperture_angles[:] * (1 - np.sign(coverage_error) * 0.05)
            
        
        covered_cabinet_count = 0
        
        ###### BEGIN FOR LOOP, one loop for every loudspeaker cabinet
        
        # use m instead of i (matlab code)
        for m in range(cabinet_count):
            
            if m == 0:
                # (top position) x_PAL_k, Initialisierung, that highest LS cabinet beams on highest audience positions
                if not config.use_fixed_angle or config.hard_margin_last_angle: # not not
                    hit_points_start[m,:] = segment_points[k,:]
                    # segment_lengths are the distances between x_PAL_k and x_PAL_k+1
                    # seg_rem = |hit_points_start - x_PAL_k-1|                      
                    seg_rem = segment_lengths[k]  
                    # analytical Calculation of GammaTiltDeg_angle: 
                    tilt_angles[m] = cabinet_tilt_angle(tilt_angles, aperture_angles, \
                       config.top_x, config.top_y, hit_points_start, config.cabinet_height, m)

                ############ FIXED FIRST ANGLE ############################
                elif config.use_fixed_angle:
                    # es muss
                    # tilt_angles eingelesen werden - fertig
                    # 2. checken ob es einen schnittpunkt des obersten strahls mit pal gibt (strahl über tilt_angles und aperture_angles)
                    # mit intersect funktion: x max wert von pal raussuchen und den strahl bis x max +1 laufen lassen
                    # 3. wenn nicht: pal vertikal einfügen sodass schnittpunkt existiert
                    # 4. wenn nicht: pal in segment_points einfügen
                    # 5. hit_points_start berechnen 
                    if iteration_count > 0 and np.any(point_exists1) == 0:
                        del segments[-1]
                        segment_points = np.delete(segment_points, [0,1,2]).reshape(int((np.shape(segment_points)[0]-1)),3)
                        audience_length -= segment_lengths[0]
                        segment_lengths = np.delete(segment_lengths, 0)
                        segment_angles = np.delete(segment_angles, 0)
                    elif iteration_count > 0:
                        audience_length = audience_length_copy
                    # get the fixed angle
                    tilt_angles[m] = config.fixed_angle * np.pi / 180

                    # top direction of rays of the highest LS cabinet
                    direction = tilt_angles[m] - aperture_angles[m]  
                    hypo = 1/np.cos(direction)   
                    direction_y = float(hypo * np.sin(direction))
                    # source position of the highest LS cabinet
                    source_x = [config.top_x - (np.sin(tilt_angles[m]) * config.cabinet_height)/2]

                    source_y = [config.top_y - (np.cos(tilt_angles[m]) * config.cabinet_height)/2]
                    # find maximum value on x-axis of pal and end point of ray

                    xmax = max(o.max() for o in segments)
                    end_x = np.array([xmax + 1])
                    end_y = source_y - direction_y * (end_x - source_x)
                    # find the intersection point if exists
                    xy_s1 = np.empty([len(segments),2])
                    point_exists1 = np.empty(len(segments))
                    for o in range(len(segments)):

                        xy_s1[o,:], lambda_1, lambda_2, point_exists1[o] = \
                        line_intersection(np.array([source_x[0], source_y[0], 0]), np.array([end_x[0], end_y[0], 0]), segments[o][0,:], segments[o][-1,:])

                        if point_exists1[o] == 1 and segments[o][0,0]<=np.round(xy_s1[o,0], decimals=3)<=segments[o][-1,0] and \
                           segments[o][0,1]<=np.round(xy_s1[o,1], decimals=3)<=segments[o][-1,1]:

                            point_exists1[o] = int(1)
                        else:

                            point_exists1[o] = int(0)
                    # add a audience line if intersection point doesnt exist
                    if np.any(point_exists1) == 0:
                        segments.append(np.array([segments[-1][-1,0],segments[-1][-1,1], 0, xmax, source_y[0] - direction_y * (xmax - source_x[0]), 0]).reshape(2,3))
                        segment_lengths = np.append(np.sqrt(np.sum((segments[-1][-1,:]-segments[-1][0,:])**2)), segment_lengths)
                        segment_angles = np.append(np.arctan((segments[-1][1,1]-segments[-1][0,1]) / (segments[-1][1,0]-segments[-1][0,0]))-np.pi, segment_angles)
                        audience_length += segment_lengths[0]
                        segment_points = np.append([segments[-1][-1,:]], segment_points, axis=0)

                        hit_points_start[m,:] = segment_points[k,:] 
                        seg_rem = segment_lengths[k]
                    # case if the highest LS cabinet covers the last audience line
                    elif point_exists1[-1] == 1:
                        hit_points_start[m,:-1] = xy_s1[np.argmax(point_exists1),:]
                        seg_rem = segment_lengths[k] - np.sqrt(np.sum((segments[-1][np.shape(segments[-1])[0]-1,:-1]-xy_s1[np.argmax(point_exists1),:])**2))
                        audience_length -= (segment_lengths[k] - seg_rem)
                        segment_points[k,:-1] = xy_s1[np.argmax(point_exists1),:]
                    # if the highest LS cabinet hits pal 0,1,...n-1: no convergence
                    else:
                        return
                    
  
                # Discrete angles if hard margin is used for possibly second computation
                if config.use_discrete_angles and np.shape(config.discrete_tilt_angles)[0] > 1 \
                   and config.gap_handling in ['Hard Margin'] and config.hard_margin_last_angle:
                    val = np.abs(config.discrete_tilt_angles - (tilt_angles[m] - config.hard_margin_last_angle))
                    ind = np.unravel_index(np.argmin(val, axis=None), val.shape)
                    tilt_angles[m] = config.discrete_tilt_angles[ind[0]] + config.hard_margin_last_angle
            elif m > 0:
                hit_points_start[m,:] = hit_points_stop[m-1,:]
                tilt_angles[m] = cabinet_tilt_angle(tilt_angles, aperture_angles, x_start, y_start, hit_points_start, config.cabinet_height, m)
 
                ################## DISCRETE TILT ANGLES ###################
                if config.use_discrete_angles and np.shape(config.discrete_tilt_angles)[0] > 1:
                    ######### Calculate The Tilt Angles By Usage Of The Set Of Discrete Tilt Angles #######
                    val = np.abs(config.discrete_tilt_angles - (tilt_angles[m] - tilt_angles[m-1]))
                    ind = np.unravel_index(np.argmin(val, axis=None), val.shape)
                    tilt_angles[m] = config.discrete_tilt_angles[ind[0]] + tilt_angles[m-1]
                    
                if (config.use_discrete_angles and np.shape(config.discrete_tilt_angles)[0] > 1) or tilt_angles[m]-tilt_angles[m-1] < 0:
                    # case in continuous tilt angles: intercabinet angle is smaller than zero
                    if tilt_angles[m]-tilt_angles[m-1] < 0: tilt_angles[m] = float(tilt_angles[m-1])
                    ######### Shift Necessary Variables And Update k If Necessary ###########
                    direction = tilt_angles[m] - aperture_angles[m]  # angle of direction to hit_points_start of m-th loudspeaker
                    hypo = 1/np.cos(direction)    # length if side adjacent is equal to one
                    direction_y = hypo * np.sin(direction) # length in y direction if side adjacent (x direction)  is equal to one
                    x_start, y_start, x_stop, y_stop, cabinet_centers_x, cabinet_centers_y, x_S, y_S = source_positions(tilt_angles, config) # source positions
                    xmax = max(o.max() for o in segments)
                    end_x = np.array([xmax + 1]) # max length in x direction
                    end_y = cabinet_centers_y[m] - direction_y * (end_x - cabinet_centers_x[m]) #  max depth of y direction source_y - (end_x - source_x) / np.tan((np.pi/2) - (tilt_angles[m]-aperture_angles[m]))
                    xy_s1 = np.empty([len(segments),2])
                    point_exists2 = False
                    # check in which pal segment is the intersection point
                    for o in range(len(segments)):
                        xy_s1[0,:], lambda_1, lambda_2, point_exists2 = \
                        line_intersection(np.array([cabinet_centers_x[m], cabinet_centers_y[m], 0]), \
                                              np.array([end_x[0], end_y[0], 0]), segments[o][0,:], segments[o][-1,:])
                        if point_exists2 == True and segments[o][0,0]<=np.round(xy_s1[0,0], decimals=3)<=segments[o][-1,0] and \
                           segments[o][0,1]<=np.round(xy_s1[0,1], decimals=3)<=segments[o][-1,1]:
                            # check if current pal[o] is equal to current k
                            if o == (len(segments)-1-k):
                                hit_points_start[m,:] = [xy_s1[0,0], xy_s1[0,1], 0] # new top position on audience line
                                seg_rem = segment_lengths[k] - np.sqrt(np.sum((segments[-k-1][-1,:-1]-xy_s1[0,:])**2)) # corrected segment remaining
                                # calc total overlap on audience line to correct coverage_error
                                if np.sum(hit_points_start[m,:]-hit_points_stop[m-1,:]) >= 0:
                                    audience_overlap += np.sqrt(np.sum((hit_points_start[m,:]-hit_points_stop[m-1,:])**2))
                                elif np.sum(hit_points_start[m,:]-hit_points_stop[m-1,:]) < 0:
                                    audience_overlap -= np.sqrt(np.sum((hit_points_start[m,:]-hit_points_stop[m-1,:])**2))
                            # case if current pal[o] is on the pal segment above or rather on k-1
                            elif o == (len(segments)-k):
                                audience_overlap += np.sqrt(np.sum((hit_points_start[m,:]-segments[-1-k][-1,:])**2)) # old k-th segment overlap
                                k -= 1 # update k
                                epsilon = -segment_angles[k] - (np.pi / 2) # angle of new pal segment
                                hit_points_start[m,:] = [xy_s1[0,0], xy_s1[0,1], 0] # new top position on audience line
                                seg_rem = segment_lengths[k] - np.sqrt(np.sum((segments[-k-1][-1,:-1]-xy_s1[0,:])**2)) # corrected segment remaining
                                audience_overlap += np.sqrt(np.sum((hit_points_start[m,:]-segments[-1-k][0,:])**2)) # new k-th segment overlap
                            # case if current pal[o] is on the pal segment below or rather on k+1
                            elif o == (len(segments)-2-k):
                                audience_overlap -= np.sqrt(np.sum((hit_points_start[m,:]-segments[-1-k][0,:])**2)) # old k-th segment overlap
                                k += 1 # update k
                                epsilon = -segment_angles[k] - (np.pi / 2) # angle of new pal segment
                                hit_points_start[m,:] = [xy_s1[0,0], xy_s1[0,1], 0] # new top position on audience line
                                seg_rem = segment_lengths[k] - np.sqrt(np.sum((segments[-k-1][-1,:-1]-xy_s1[0,:])**2)) # corrected segment remaining
                                audience_overlap -= np.sqrt(np.sum((hit_points_start[m,:]-segments[-1-k][-1,:])**2)) # new k-th segment overlap
                        
                    if point_exists2 == False:
                            logger.warning('Error in discrete tilt angles')
                            return
   
            ###############################################################################        
            # STEP 2: CALCULATE THE SOURCE POSITIONS  
            x_start, y_start, x_stop, y_stop, cabinet_centers_x, cabinet_centers_y, x_S, y_S = source_positions(tilt_angles, config)
                
            ###############################################################################
            # STEP 3
            
            # center position of audience segment - epsilon for eq (15) JAES
            epsilon = -segment_angles[k] - (np.pi / 2)
            # eq (15) JAES
            Xi_1 = np.sqrt(np.sum((hit_points_start[m,:] - [cabinet_centers_x[m], cabinet_centers_y[m], 0])**2)) * (np.sin(aperture_angles[m]) \
                           / np.cos(epsilon - tilt_angles[m]))
            if seg_rem >= Xi_1: # case (i) eq (16) in JAES
                seg_rem = seg_rem - Xi_1
                # eq (17) JAES (segment_angles = epsilon_k)
                hit_points[m,:] = hit_points_start[m,:] + np.array([np.cos(segment_angles[k]), np.sin(segment_angles[k]), 0]) * Xi_1
                psi_n_tilde = 0
                # eq (18) JAES
                Gamma_n[m,0] = Xi_1
            elif seg_rem < Xi_1:
                Gamma_n[m,0] = seg_rem # first part of eq (23) JAES
                d = np.sqrt(np.sum((hit_points_start[m,:] - [cabinet_centers_x[m], cabinet_centers_y[m], 0])**2)) # distance for eq (20) JAES
                # eq (20) JAES
                psi_n_tilde = np.arctan((seg_rem / d * np.cos(epsilon - tilt_angles[m]) - \
                                         np.sin(aperture_angles[m])) / (seg_rem / d * np.sin(epsilon \
                                               - tilt_angles[m]) - np.cos(aperture_angles[m])))
            ########################## STEP IV ###########################################
                if k < np.shape(segment_lengths)[0]-1:
                    k += 1
                    epsilon = -segment_angles[k] - (np.pi / 2)
                else:
                    covered_cabinet_count = m
                    Gamma_n[m,0] = 0
                    break
            ####################### CONTINUE WITH STEP III ###############################
                d = np.sqrt(np.sum((segment_points[k,:] - [cabinet_centers_x[m], cabinet_centers_y[m], 0])**2))
                # eq (21) JAES
                Xi_tilde = d * np.sin(psi_n_tilde) / np.cos(-tilt_angles[m] + epsilon)
                # eq (22) JAES
                hit_points[m,:] = segment_points[k,:] + np.array([np.cos(segment_angles[k]), np.sin(segment_angles[k]), 0]) * Xi_tilde
                seg_rem = segment_lengths[k] - Xi_tilde
                # second part of eq (23) JAES
                Gamma_n[m,0] = Gamma_n[m,0] + Xi_tilde
            ######################## STEP V #############################################
    
            d = np.sqrt(np.sum((hit_points[m,:] - [cabinet_centers_x[m], cabinet_centers_y[m], 0])**2))
            # stop position of audience segment
            # calculate psi, that it fits for PALC1, PALC2 or PALC3 ### CONSTRAINTS ###
            if m == 0:
                d_array[m] = np.sqrt(np.sum((hit_points[m,:] - [cabinet_centers_x[m], cabinet_centers_y[m], 0])**2))
            elif m > 0:
                d_array[m] = np.sqrt(np.sum((hit_points[m,:] - [cabinet_centers_x[m], cabinet_centers_y[m], 0])**2))
                if config.palc_variant in ['PALC 1']:
                    # for PALC1: psi = const.
                    aperture_angles[m] = (aperture_angles[m-1] * config.weighting_factors[m-1]) / config.weighting_factors[m]
                elif config.palc_variant in ['PALC 2']:
                    # psi * d = const. with weighting factor config[7]
                    aperture_angles[m] = (d_array[m-1] * aperture_angles[m-1] * config.weighting_factors[m-1]) \
                    / (d_array[m] * config.weighting_factors[m])
                elif config.palc_variant in ['PALC 3']:
                    # tan(psi) * d = const.
                    aperture_angles[m] = np.arctan((d_array[m-1] * np.tan(aperture_angles[m-1]) * config.weighting_factors[m-1]) \
                    / (d_array[m] * config.weighting_factors[m]))    
            # eq (24) JAES
            Xi_2 = d * np.sin(aperture_angles[m]) / np.cos(epsilon - tilt_angles[m] - aperture_angles[m])
            
            if seg_rem >= Xi_2: # case (ii) eq (30) JAES
                # eq (31) JAES
                hit_points_stop[m,:] = hit_points[m,:] + np.array([np.cos(segment_angles[k]), np.sin(segment_angles[k]), 0]) * Xi_2
                seg_rem = seg_rem - Xi_2
                psi_n_tilde = 0
                Gamma_n[m,1] = Xi_2 # eq (32) JAES
            elif seg_rem < Xi_2: # case (i) eq (23) JAES
                Gamma_n[m,1] = seg_rem # first part of eq (29) JAES
                # eq (26) JAES
                psi_n_tilde = np.arctan(seg_rem / d * np.cos(epsilon - tilt_angles[m]) / \
                                         (1 - seg_rem / d * np.sin(epsilon - tilt_angles[m])))
            ############################## STEP VI ########################################     
                if k < np.shape(segment_lengths)[0]-1:
                    k += 1
                    epsilon = -segment_angles[k] - (np.pi / 2)
                else:
                    covered_cabinet_count = m
                    Gamma_n[m,1] = 0
                    break 
            ########################### CONTINUE WITH STEP V #############################
                d = np.sqrt(np.sum((segment_points[k,:] - [cabinet_centers_x[m], cabinet_centers_y[m], 0])**2))
                # eq (27) JAES
                Xi_tilde = d * np.sin(aperture_angles[m] - psi_n_tilde) / np.cos(epsilon - tilt_angles[m] - aperture_angles[m])
                seg_rem = segment_lengths[k] - Xi_tilde
                # eq (28) JAES
                hit_points_stop[m,:] = segment_points[k,:] + np.array([np.cos(segment_angles[k]), np.sin(segment_angles[k]), 0]) * Xi_tilde
                # second part of eq (29) JAES

                Gamma_n[m,1] = Gamma_n[m,1] + Xi_tilde
        ####################### END OF FOR-LOOP ####################################### 

        if covered_cabinet_count == 0:
            covered_cabinet_count = cabinet_count
        
        ######### PLOT ################################################################
            
        lin_eq_n = cabinet_centers_y - np.tan(-tilt_angles[:]) * cabinet_centers_x
        center_ray_ends_x = hit_points[:,0]
        center_ray_ends_y = np.tan(-tilt_angles[:]) * center_ray_ends_x + lin_eq_n
        if aperture_angles.ndim == 2:
            aperture_angles = aperture_angles[:,0]
        
        # update optimized LSA
        optimized_array.update_opt_array(cabinet_centers_x=cabinet_centers_x, cabinet_centers_y=cabinet_centers_y, hit_points=hit_points, \
                                 hit_points_start=hit_points_start, hit_points_stop=hit_points_stop, \
                                 tilt_angles=tilt_angles, aperture_angles=aperture_angles)

        ############################### Soft Margin Approach for gap handling #########################
        if config.gap_handling in ['Soft Margin']:            
            # First step: Find the added non audience line with pal and audience_segments (indice)
            ind_na_line = np.array([]).astype(int)
            config.gap_weights = np.ones(config.cabinet_count)
            m = 0
            o = 0
            for n in range(len(segments)):
                if config.use_fixed_angle and np.any(point_exists1) == 0 and n == len(segments)-1:
                    o=1
                if audience_segments[n-m-o][0,0] != segments[n][0,0] or audience_segments[n-m-o][0,1] != segments[n][0,1]:
                    m += 1
                    ind_na_line = np.append(ind_na_line, n)
            # Second step: check which loudspeaker hits which line on which point and 
            #              add the weighting
            for n in range(len(segments)):
                if np.any(n==ind_na_line):
                    for m in range(cabinet_count):
                        if segments[n][0,0] <= optimized_array.hit_points[m,0] <= segments[n][-1,0] and \
                           segments[n][0,1] <= optimized_array.hit_points[m,1] <= segments[n][-1,1]:
                               # compute the length of non-audience line and
                               # distance of deg_opt to next audience line
                            dist_na_line = np.sqrt((segments[n][-1,0]-segments[n][0,0])**2 + \
                                                      (segments[n][-1,1]-segments[n][0,1])**2)
                            dist_high = np.sqrt((segments[n][-1,0]-optimized_array.hit_points[m,0])**2 + \
                                                   (segments[n][-1,1]-optimized_array.hit_points[m,1])**2)
                            dist_low = np.sqrt((optimized_array.hit_points[m,0]-segments[n][0,0])**2 + \
                                                  (optimized_array.hit_points[m,1]-segments[n][0,1])**2)
                            if n == len(segments)-1:
                                dist_near = dist_low
                            else:
                                dist_near = np.amin([dist_high, dist_low])
                            config.gap_weights[m] = 1 - config.soft_margin_strength*(dist_near/dist_na_line)
                                               
        ###### weighting factors calculation with special case if fixed angles, soft margin and weighting
        if config.use_fixed_angle and config.gap_handling in ['Soft Margin'] \
           and config.weighting_nu <= 1.02 and not config.use_weighting in ['Without']:
            elsecase = False
            for n in range(config.cabinet_count):
                if config.gap_weights[n] < 1 and not elsecase:
                    config.weighting_factors[n] = config.weighting_weights[n]
                else:
                    elsecase = True
                    config.weighting_factors[n] = config.gap_weights[n] * config.weighting_weights[n]
            del elsecase
        else:
            config.weighting_factors = config.gap_weights * config.weighting_weights
            
            # For future: may add some weighting for intersect points near to non-audience lines
        ###############################################################################################
        # plot function! not optimized for plotting during calculation -> one plot at the end
        
        ######### update output + variables ###########################################
        
        tilt_angles_deg = tilt_angles / np.pi * 180
        error = np.sqrt(np.sum((segment_points[-1,:]-hit_points_stop[-1,:])**2))
        coverage_error = np.sum(Gamma_n[:,:]) - audience_length - audience_overlap

        hit_distances = np.sqrt(np.sum((hit_points[:,:] - np.transpose(np.array([cabinet_centers_x[:], cabinet_centers_y[:], cabinet_centers_x[:]*0])))**2,1))
        iteration_count += 1
        logger.debug('error is: %s', coverage_error)
        # if difference is to big, reset aperture_angles
        if np.abs(coverage_error) > 50:
            aperture_angles = config.aperture_angles

    ray_plot.update_plot_array(cabinet_centers_x=cabinet_centers_x, cabinet_centers_y=cabinet_centers_y, center_ray_ends_x=center_ray_ends_x, center_ray_ends_y=center_ray_ends_y)
    optimized_array.update_opt_array(tilt_angles=tilt_angles, aperture_angles=aperture_angles, iteration_count=iteration_count, \
                             coverage_error=coverage_error, hit_distances = hit_distances, \
                             tilt_angles_deg=tilt_angles_deg)
  
    # set new tilt_angles
    setattr(config,'tilt_angles',tilt_angles)
    return


def run_palc(ray_plot, palc_venue, config, optimized_array):
    """
    Handles the different PALC approaches. Offers the functionality of the
    Hard Margin Gap Handling Approaches. Without and Soft Margin Gap Handling
    Approach call directily :any:`compute_tilt_angles`.

    Parameters
    ----------
    ray_plot : obj [in, out]
        Contains plotting information.
    palc_venue : obj [in]
        Contains all venue information.
    config : obj [in]
        Contains PALC configuration data.
    optimized_array : obj [in, out]
        Contains the data of the optimized LSA.

    Returns
    -------
    None.

    """
    #### again: initialization ####
    cabinet_count       = int(config.cabinet_count)
    tilt_angles = np.array(config.tilt_angles)
    aperture_angles   = np.array(config.aperture_angles)

    audience_segments = list(palc_venue.audience_segments)
    segments = list(palc_venue.segments)

    # no Gap Handling or Soft Margin
    if config.gap_handling in ['Without', 'Soft Margin']:
        t1 = time.time()
        compute_tilt_angles(ray_plot, segments, aperture_angles, tilt_angles, cabinet_count, config, audience_segments, optimized_array)
        t2 = time.time()
        dt = t2 - t1
        logger.debug('elapsed time in s: %s', dt)
        return 
    # Gap Handling: Hard Margin Approach
    elif config.gap_handling in ['Hard Margin']:
        compute_tilt_angles(ray_plot, segments, aperture_angles, tilt_angles, cabinet_count, config, audience_segments, optimized_array)
        # create continuous_segments that contains lists with the continuous audience lines
        continuous_segments = []
        continuous_segments.append([])
        m = 0
        for n in range(len(audience_segments)):
            if audience_segments[n][0,0] != segments[n+m][0,0] or audience_segments[n][0,1] != segments[n+m][0,1]:
                continuous_segments.append([])
                m += 1
            if audience_segments[n][0,0] == segments[n+m][0,0] and audience_segments[n][0,1] == segments[n+m][0,1]:
                continuous_segments[m].append(segments[n+m])
        # check which loudspeaker cabinets belong to which continuous audience line
        cabinet_to_group = np.empty(cabinet_count).astype(int)
        group_distances_high = np.empty(np.shape(continuous_segments)[0])
        group_distances_low = np.empty(np.shape(continuous_segments)[0])
        group_distances = np.empty(np.shape(continuous_segments)[0])
        for m in range(cabinet_count):
            # go from the highest audience line to the lowest
            for n in range(np.shape(continuous_segments)[0]-1,-1,-1):
                group_distances_high[n] = np.abs(np.sqrt((continuous_segments[n][-1][-1,0]-optimized_array.hit_points[m,0])**2 \
                              + (continuous_segments[n][-1][-1,1]-optimized_array.hit_points[m,1])**2))
                group_distances_low[n] = np.abs(np.sqrt((continuous_segments[n][0][0,0]-optimized_array.hit_points[m,0])**2 \
                            + (continuous_segments[n][0][0,1]-optimized_array.hit_points[m,1])**2))
                group_distances[n] = np.minimum(group_distances_high[n], group_distances_low[n])
            
            # get the indice of the segment to which the LS belongs
            cabinet_to_group[m] = int(np.argmin(group_distances))

        # Run the calculation for each continuous audience line
        for n in range(np.shape(continuous_segments)[0]-1,-1,-1):
            margin_cabinet_count = 0
            for m in range(cabinet_count):
                if n == cabinet_to_group[m]:
                    margin_cabinet_count += 1

            if n == np.shape(continuous_segments)[0]-1:
                tilt_angles = np.linspace(-0.0428, 0.7147, num=margin_cabinet_count)
                aperture_angles = np.linspace(0.00524, 0.00524, num=margin_cabinet_count)
                compute_tilt_angles(ray_plot, continuous_segments[n], aperture_angles, tilt_angles, margin_cabinet_count, config, audience_segments, optimized_array)
                patches_x, patches_y = cabinet_outlines(ray_plot, config, optimized_array.tilt_angles, \
                                                          margin_cabinet_count=np.shape(optimized_array.tilt_angles_deg)[0])

                margin_array = copy(optimized_array)
                margin_ray_plot = copy(ray_plot)
            else:

                config.hard_margin_last_angle = margin_array.tilt_angles_deg[-1] * np.pi / 180
                config.tilt_angles = tilt_angles = np.linspace(-0.0428, 0.7147, num=margin_cabinet_count)
                config.aperture_angles = aperture_angles = np.linspace(0.00524, 0.00524, num=margin_cabinet_count)

                config.top_x = margin_ray_plot.x_stop[-1] - np.sin(config.hard_margin_last_angle) * config.cabinet_gap
                config.top_y = margin_ray_plot.y_stop[-1] - np.cos(config.hard_margin_last_angle) * config.cabinet_gap

                compute_tilt_angles(margin_ray_plot, continuous_segments[n], aperture_angles, tilt_angles, margin_cabinet_count, config, audience_segments, margin_array)
                margin_patches_x, margin_patches_y = cabinet_outlines(margin_ray_plot, config, margin_array.tilt_angles, \
                                                          margin_cabinet_count=np.shape(margin_array.tilt_angles_deg)[0])
                for attr in ['cabinet_centers_x', 'cabinet_centers_y','x_start', 'y_start', 'x_stop', \
                              'y_stop', 'center_ray_ends_x', 'center_ray_ends_y']:
                    setattr(ray_plot, attr, np.append(getattr(ray_plot, attr), getattr(margin_ray_plot, attr)))
                for attr in ['hit_points', 'hit_points_start', 'hit_points_stop', 'start_ray_ends_x', \
                              'start_ray_ends_y', 'stop_ray_ends_x', 'stop_ray_ends_y', \
                              'tilt_angles', 'tilt_angles_deg', 'hit_distances']:
                    setattr(optimized_array, attr, np.append(getattr(optimized_array, attr), getattr(margin_array, attr)))
                    if attr in ['hit_points', 'hit_points_start', 'hit_points_stop']:
                        setattr(optimized_array, attr, getattr(optimized_array, attr).reshape(int(cabinet_count),3))
        return


def fixed_angle_mid_max(last_segment, config):
    """
    Calculates the average / mid and maximum tilt angle for the heighest LSA cabinet.
    When functionality of fixing the heighest LSA cabinet on a specific tilt
    angle, the borders of the maximum and minimum tilt angle can thus be found.
    Called by :any:`use_fixed_angle`.

    Parameters
    ----------
    last_segment : list
        Last segment of the venue, i.e., palc_venue.segments[-1].
    config : obj [in]
        Configuration of the PALC algorithm.

    Returns
    -------
    tilt_angles_deg : list
        Contains average (first entry) and maximum (seconed entry) tilt angle
        of the heighest LSA cabinet.

    """
    tilt_angles_deg = []
    for n in [0,-1]:
        dist_hy = np.sqrt((last_segment[n,1]-(config.top_y-config.cabinet_height))**2 + \
                          (last_segment[n,0]-config.top_x)**2)
        dist_an = np.abs(last_segment[n,0]-config.top_x)
        if config.top_y < last_segment[n,1]:
            tilt_angles_deg.append(np.round(-np.arccos(dist_an /dist_hy) *(180/np.pi), decimals=2))
        else:
            tilt_angles_deg.append(np.round(np.arccos(dist_an /dist_hy) *(180/np.pi), decimals=2))
    return tilt_angles_deg


def fixed_angle_borders(last_segment, config):
    """
    Allowed range of the fixed tilt angle of the highest LSA cabinet, as the
    GUI computes it when "fix angle of uppermost cabinet" is enabled.

    Parameters
    ----------
    last_segment : ndarray
        Last segment of the venue, i.e., palc_venue.segments[-1].
    config : obj [in]
        Configuration of the PALC algorithm.

    Returns
    -------
    list of float
        Maximum, mid and minimum tilt angle in degree (the legacy
        ``optimized_array.fixed_angle``).

    """
    tilt_angles_deg = fixed_angle_mid_max(last_segment, config)
    return [tilt_angles_deg[0]-1, tilt_angles_deg[1], tilt_angles_deg[1]-10]


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
