"""Target slope optimization of the PALC weighting.

The state of the target slope (:class:`TargetSlopeState`), the helpers of the
optimization and its loop :func:`optimize_target_slope` (formerly
``fun/PALC_opt.py``, ``TargetSlopeState`` from ``fun/PALC_classes.py`` and
``optimize_target_slope`` from ``fun/gui_algorithm_sfp.py``).

author: Arne Hoelter
"""
from __future__ import annotations

import logging

import numpy as np
from numpy import array as nar
from scipy.special import erf
from scipy.stats.mstats import mquantiles

from .geometry import cabinet_outlines
from .palc import run_palc
from .sound_field import predict_sound_field, compute_spl_over_distance

logger = logging.getLogger(__name__)


class TargetSlopeState():
    """ Class to optimize PALC by the target slope optimization. """
    def __init__(self, region_levels_db=None):
        self.distances, self.levels_db   = [], []
        self.region_distances      = [0,0]
        self.region_levels_db    = [0,0] if region_levels_db is None else region_levels_db
        self.initialized          = False
        self.points_x, self.points_y = [0,0], [0,0]
        self.cabinet_differences        = []
        self.previous_differences        = {'n-1':100, 'lowest':[],'same':0}
        self.previous_weights          = {'n-1':[], 'lowest':[]}
        self.update_count     = 0
        self.region_gradients      = []
        self.region_indices       = 0
        self.cabinet_gradient_differences    = []
        self.reference_distance         = []
        self.reference_index       = []
        
    def reset_optimization(self):
        """
        Initializes the target slope optimization before starting the
        optimization.

        Returns
        -------
        None.

        """
        self.cabinet_differences        = []
        self.previous_differences      = {'n-1':100, 'lowest':[]}
        self.previous_weights          = {'n-1':[], 'lowest':[]}
        
    def read_source(self, CDS, keys):
        """
        Stores data from a ColumnDataSource in the object when the keys are
        identic.

        Parameters
        ----------
        CDS : ColumnDataSource
            ColumnDataSource that contain the data.
        keys : str
            Key to write into the object.

        Returns
        -------
        None.

        """
        for key in keys:
            setattr(self, key, CDS.data[key])
        
        
    def normalize_to(self, ref=0):
        """
        Normalizes all data on a given reference value.

        Parameters
        ----------
        ref : float, optional
            Value on which data shall be normalized. The default is 0.

        Returns
        -------
        None.

        """
        shift           = np.amax(self.region_levels_db)-ref
        self.levels_db        = self.levels_db - shift
        self.region_levels_db = self.region_levels_db - shift 


    def store_previous(self, config, diff_opt, iteration_count):
        """
        Shifts the actual value to a list that stores the past value. These are
        used to check if the actual loop improved that the difference between
        target and computation result.

        Parameters
        ----------
        config : obj [in, out]
            Contains the configuration of the PALC algorithm.
        diff_opt : float
            Single value of the optimized PALC computation.
        iteration_count : int
            Iteration count of the PALC run of this loop
            (``optimized_array.iteration_count`` after :func:`run_palc`);
            500 means that PALC did not converge, and the optimization
            stops.

        Returns
        -------
        bool
            True, if the while optimization loop has to stop: 10 loops were
            worse than the loop before, the difference stayed the same
            5 times in a row, or ``iteration_count`` is 500. The weights of
            the loop with the lowest difference are then restored in
            ``config``. Otherwise False.

        """
        if np.round(diff_opt, decimals=8) == np.round(self.previous_differences['n-1'], decimals=8):
            self.previous_differences['same'] += 1
        else:
            self.previous_differences['same'] = 0
        if diff_opt > self.previous_differences['n-1']:
            self.previous_differences['lowest'].append(self.previous_differences['n-1'])
            self.previous_weights['lowest'].append(self.previous_weights['n-1'])
        self.previous_differences['n-1'] = diff_opt
        self.previous_weights['n-1'] = np.array(config.weighting_weights[:])
        if len(self.previous_weights['lowest']) == 10 or self.previous_differences['same'] == 5 or iteration_count==500:
            low_ind = np.argmin(self.previous_differences['lowest'])
            config.weighting_weights = self.previous_weights['lowest'][low_ind][:]
            return True
        return False
            

    def sort_points(self):
        """
        Resorts the target slope hinge points, so that the x-axis / SPL pairs
        begin with lowest x-axis values.

        Returns
        -------
        None.

        """
        sort_key = np.argsort(self.distances)
        for key in ['distances', 'levels_db', 'points_x', 'points_y']:
            setattr(self, key, list(np.array(getattr(self,key))[sort_key]))
            
    def interpolate_levels(self):
        """
        Interpolates the drawn target slope line on the discrete distance
        points, that are used von SPL over distance calculation of the
        computed results.

        Returns
        -------
        None.

        """
        self.region_levels_db, ap, ind = [], [], [0]
        for n in range(1,len(self.distances)):
            ind.append(np.argwhere(self.distances[n] == self.region_distances)[0,0])
            ap.append(list((self.levels_db[n]-self.levels_db[n-1])/(self.distances[n]-self.distances[n-1])* \
                                   (np.array(self.region_distances[ind[n-1]:ind[n]])- \
                                    self.distances[n-1])+self.levels_db[n-1]))
        ap.append([self.levels_db[-1]])
        list(map(self.region_levels_db.extend, ap))


def hinge_indices(hinge_count, spl_over_distance):
    """
    Init function to return indices of the hinges / links in the SPL over distance figure.
    Necessary for the movable target slope line. Works also if user has moved
    the hinge / link. Called by :any:`chg_weight_links`.

    Parameters
    ----------
    hinge_count : int
        Number of hinges / links in the target slope line.
    spl_over_distance : obj [in]
        Contains computed SPL over distance data including distance between
        center of LSA and receiver positions.

    Returns
    -------
    ind : list
        Indices (int) where the hinges / links are placed in the SPL over distance figure.

    """
    point_count = len(spl_over_distance.distances)
    ind = [0] * (hinge_count+2)
    for n in range(hinge_count+2):
        ind[n] = n * int(point_count/(hinge_count+1))
        if n > 0: ind[n] -= 1
    return ind


def optimization_region(spl_over_distance, target):
    """
    Find the indices of the user defined optimization region in
    the SPL over distance figure. Called by :func:`pypalc.core.pipeline.run`
    and :any:`optimize_target_slope`.

    Parameters
    ----------
    spl_over_distance : obj [in, out]
        Contains SPL over distance data of the PALC computed results. Attribute
        region_levels_db is normalized to the maximum in the optimization
        region.
    target : obj [in]
        Contains SPL over distance data of the optimized results and the target
        slope.

    Returns
    -------
    ind : list
        Indices (int) of the optimization region in center of LSA to receiver
        position distance of the attribute x of :any:`spl_over_distance`.

    """
    ind = []
    for val in list(target.region_distances):
        ind.append(np.argmin(np.abs(np.round(list(spl_over_distance.distances),2) - np.round(val,2))))#[0,0])
    spl_over_distance.region_levels_db = spl_over_distance.levels_db[ind]
    target.region_distances      = spl_over_distance.distances[ind]
    target.distances             = [spl_over_distance.distances[ind[0]], spl_over_distance.distances[ind[-1]]]
    target.region_gradients      = np.gradient(np.array(target.region_levels_db))
    spl_over_distance.region_gradients   = np.gradient(np.array(spl_over_distance.region_levels_db))
    spl_over_distance.region_indices    = ind
    return ind


def region_difference(comp, opt):
    """
    Returns difference between two arrays of same size. Actually unused.

    Parameters
    ----------
    comp : ndarray
        Computed results.
    opt : ndarray
        Target values.

    Returns
    -------
    diff : ndarray
        Difference between computed and target.

    """
    return comp - opt


def cabinet_points(cabinet_count, optimized_array, drawn_lines, **kwargs):
    """
    Maps LSA cabinets to discrete venue points depending on the assumed splay
    angle. Called by :any:`optimize_target_slope`.

    Parameters
    ----------
    cabinet_count : int
        Number of LSA cabinets.
    optimized_array : obj [in]
        Contains results of the PALC computation.
    drawn_lines : obj [in]
        Venue information that was drawn by the user.
    **kwargs : dict, optional
        'tol' : Tolerance of discrete points to find (default = 10⁻⁵ m).
        'region_indices' : Indices of optimization region in pal / venue slice.

    Returns
    -------
    cabinet_points : list
        Indices of audience line points sorted to the LSA cabinets.
    cabinet_region_point_indices : list
        Indices in optimization region sorted to the LSA cabinets.

    """
    # tolerance (optional input) default is ndiscretization / np.sqrt(2)
    discr = np.sqrt((drawn_lines.points_x[1]-drawn_lines.points_x[0])**2 + \
                    (drawn_lines.points_y[1]-drawn_lines.points_y[0])**2) / np.sqrt(2)
    if 'tol' not in kwargs.keys(): kwargs['tol'] = discr#0.00001
    tol = kwargs['tol']
    # list to store index of pal for each ls
    cabinet_points, cabinet_region_point_indices = [], []
    for n in range(cabinet_count):
        ind = [[],[]]
        # check x-values
        for m,val in enumerate(drawn_lines.points_x):
            if val >= optimized_array.hit_points_stop[n][0]-tol and val <= optimized_array.hit_points_start[n][0]+tol:
                ind[0].append(m)
        # check y-values
        for m,val in enumerate(drawn_lines.points_y):
            if val >= optimized_array.hit_points_stop[n][1]-tol and val <= optimized_array.hit_points_start[n][1]+tol:
                ind[1].append(m)
        # check for matching index
        cabinet_points.append(list(set(ind[0]) & set(ind[1])))
    if 'region_indices' in kwargs.keys():
        for n in range(cabinet_count):
            cabinet_region_point_indices.append(list(set(cabinet_points[n]) & set(kwargs['region_indices'])))
        return cabinet_points, cabinet_region_point_indices
    return cabinet_points


def differences_per_cabinet(cabinet_count, region_indices, cabinet_point_indices, cabinet_region_indices, target, spl_over_distance, **kwargs):
    """
    Maps diference in optimization region to LSA cabinets. Called by
    :any:`optimize_target_slope`.

    Parameters
    ----------
    cabinet_count : int
        Number of LSA cabinets.
    region_indices : list
        Indices of optimization region.
    cabinet_point_indices : list
        Indices of venue slice mapped on LSA cabinets.
    cabinet_region_indices : list
        Indices of optimization region mapped on LSA cabinets.
    spl_over_distance : obj [in, out]
        Object with computed PALC results in optimization region.
    target : obj [in, out]
        Object with information of Target Slope in optimization.
    **kwargs : dict, optional
        Optional input to set single value difference

    Returns
    -------
    cabinet_differences : list
        Mean average of difference between target slope and PALC computed
        results regarding each LSA cabinet. If a LSA cabinet does not hit any
        point in the optimization region, the value is set to 100.

    """
    if 'mtype' not in kwargs.keys(): kwargs['mtype'] = 'mean'
    if 'lsmap' not in kwargs.keys(): kwargs['lsmap'] = 'in_coverage'
    # write computed and optimized interpolated SPL values to variable
    opt       = np.array(target.region_levels_db)
    comp      = np.array(spl_over_distance.region_levels_db)# removed region_indices
    grad_opt  = np.array(target.region_gradients)
    grad_comp = np.array(spl_over_distance.region_gradients)# removed region_indices
    # compute maximum neighbouring points of each loudspeaker impact point
    neighbours = neighbour_counts(cabinet_count, cabinet_region_indices, region_indices)
    # first list in diff_comp_opt is computed results and secont optization target
    cabinet_differences, cabinet_gradient_differences, diff_comp_opt_LS, diff_grad = [], [], [[],[]], [[],[]]
    for n in range(cabinet_count):
        cabinet_differences.append([])
        cabinet_gradient_differences.append([])
        for m,val in enumerate(region_indices):
            if val in cabinet_region_indices[n]:
                diff_comp_opt_LS[0].append(comp[m])
                diff_comp_opt_LS[1].append(opt[m])
                diff_grad[0].append(grad_comp[m])
                diff_grad[1].append(grad_opt[m])
        if len(cabinet_region_indices[n]) >= 1:
            frontend         = range_neighbours(comp, diff_comp_opt_LS)
            diff_comp_opt_LS = insert_neighbours(comp, opt, diff_comp_opt_LS, \
                                                  neighbours[n,:], frontend)
            diff_grad        = insert_neighbours(grad_comp, grad_opt, diff_grad, \
                                                  neighbours[n,:], frontend)
            cabinet_differences[n], cabinet_gradient_differences[n] = single_value_difference(nar([nar(diff_comp_opt_LS[0]), \
                                                                 nar(diff_grad[0])]), \
                                                            nar([nar(diff_comp_opt_LS[1]), \
                                                                 nar(diff_grad[1])]), \
                                                            mtype=kwargs['mtype'], ef=True)
        else:
            cabinet_differences[n], cabinet_gradient_differences[n] = 100, 100
    
    return cabinet_differences, cabinet_gradient_differences


def neighbour_counts(cabinet_count, cabinet_region_indices, region_indices):
    """
    Computes the number of neighbours on which it could be smoothed. Called
    by :any:`differences_per_cabinet`.

    Parameters
    ----------
    cabinet_count : int
        Number of LSA cabinets.
    cabinet_region_indices : list
        List of LSA cabinets mapped to audience positions.
    region_indices : list
        Indices of optimization region.

    Returns
    -------
    nebs: ndarray
        Number of possible neighbouring points for each cabinet.

    """
    neighbours = np.zeros((cabinet_count,2),dtype=int)
    for n in range(cabinet_count):
        if cabinet_region_indices[n] != []:
            neighbours[n,0] = min(cabinet_region_indices[n]) - min(region_indices)
            neighbours[n,1] = max(region_indices) - max(cabinet_region_indices[n])
    return neighbours


def range_neighbours(comp, diff_comp_opt_LS):
    """
    Computes the front and behind position to insert neighbouring points.

    Parameters
    ----------
    comp : list or 1D-array
        Computed PALC results in optimization region
    diff_comp_opt_LS : list
        Contains the computed and target values of actual LSA cabinet.

    Returns
    -------
    list
        List with two entries. Front and behind position of neighbouring points.

    """
    
    front = np.argwhere(comp == diff_comp_opt_LS[0][0])[0,0]
    end   = np.argwhere(comp == diff_comp_opt_LS[0][-1])[0,0]
    return [front, end]


def insert_neighbours(comp, opt, diff_comp_opt_LS, neighbours, frontend, to_ins=5):
    """
    Insert neighbouring points of the LSA cabinet to for difference of
    computed and target values. Called by :any:`differences_per_cabinet`.

    Parameters
    ----------
    comp : list or 1D-array
        Computed PALC results in optimization region.
    opt : list or 1D-array
        Target Slope of optimization.
    diff_comp_opt_LS : list
        Contains the computed and target values of actual LSA cabinet.
    neighbours : list
        Maximum of points that can be added in front and end of diff_comp_opt_LS.
    frontend : list
        Index of front and behind position to insert neighbouring points.
    to_ins : float, optional
        Depending on discretized audience positions, percentage of the total
        amount of audience positions to add in front and end of
        diff_comp_opt_LS. The default is 5.

    Returns
    -------
    diff_comp_opt_LS : list
        Contains the computed and target values of actual LSA cabinet.

    """   
    # Number of points (default is 5 percent)
    nbs = int((to_ins/100) * len(comp))
    # compute possible range
    if nbs > neighbours[0]:
        fr_rng = neighbours[0]
    else:
        fr_rng = nbs
    if nbs > neighbours[1]:
        end_rng = neighbours[1]
    else:
        end_rng = nbs
    # insert in front and end (first find start index)
    front = frontend[0]
    end   = frontend[1]
    for n in range(1,fr_rng+1):
        if front-n > 0: #if len(comp) > end+n:
            diff_comp_opt_LS[0].insert(0,comp[front-n])
            diff_comp_opt_LS[1].insert(0,opt[front-n])
    for n in range(1,end_rng+1):
        if len(comp) > end+n:
            diff_comp_opt_LS[0].append(comp[end+n])
            diff_comp_opt_LS[1].append(opt[end+n])
    return diff_comp_opt_LS


def smooth_list(data, nbs_w=0.2, nums=1):
    """
    Smoothes a 1D array with the neighbouring entry.

    Parameters
    ----------
    data : 1D-array
        Array to be smoothed.
    nbs_w : float, optional
        Weight of the neighbouring point, means
        new[n] = old[n-1]*nbs_w + old[n]*(1-2*nbs_w) + old[n+1]*nbs_w.
        The default is 0.2.
    nums : int
        Number of neighbouring data points to use

    Returns
    -------
    1D-array
        Smoothed array.

    """
    s_data = np.zeros(len(data))
    for n in range(0,len(data)):
        add = 0
        # if data[n] < 0.1:
        #     s_data[n] = data[n]
        #     continue
        for row in range(-nums,nums+1):
            if n+row < 0 or n+row >= len(data):
                continue
            else:
                if n<nums and row==0:
                    s_data[n] += data[n+row]*(1-(nums+add)*nbs_w)
                elif n>=len(data)-nums and row==0:
                    s_data[n] += data[n+row]*(1-add*nbs_w)
                elif row==0:
                    s_data[n] += data[n+row]*(1-2*nums*nbs_w)
                else:
                    s_data[n] += data[n+row]*nbs_w
                    add += 1
    return list(s_data)


def update_weighting(config, target, loop):
    """
    Updates the weighting factors depending on differences at the LSA cabinets
    and weighting factors of the previous loop of target slope optimization.
    The break out condition depends equally on all positions in the
    optimization region and an additional (and optional) function to compute
    a single difference value. . Called by :any:`optimize_target_slope`.

    Parameters
    ----------
    config : obj [in, out]
        Configuration of PALC algorithm. The attribute weighting_weights will
        be updated for the next PALC computation.
    target : obj [in]
        Contains information on the optimization target.
    loop : int
        Number of actual loop of target slope optimization.

    Returns
    -------
    None.

    """

    w    = config.weighting_weights
    news = np.ones(config.cabinet_count)
    config.weighting_weights[0] = 1.0
    
    grad_coeff = gaussian([-.4,.4,0.01], .15)
    grad_coeff[1,:] = mirror(0.8+grad_coeff[1,:]*.2,[-.4,.4,0.01])
    in_opt     = False

    for n in range(1,config.cabinet_count):
        if target.cabinet_differences[n] == 100:
            if not in_opt:
                coeff = 1.2
            else:
                coeff = 0.6
            gcoeff = 1.0
        else:
            in_opt = True
            coeff  = sv_gaussian(target.cabinet_differences[n], 1.2, .65)
            gcoeff = sv_gaussian(target.cabinet_gradient_differences[n], .15, .8)
        news[n] = news[n-1]*coeff*gcoeff
        config.weighting_weights[n] = news[n] * w[n]
    return


def single_value_difference(comp, opt, mtype='mean', ef=True):
    """
    Computes a single value difference of the actual PALC results. Called by
    :any:`optimize_target_slope` and :any:`differences_per_cabinet`.

    Parameters
    ----------
    comp : list or 1D-array
        Computed PALC results in optimization region.
    opt : list or 1D-array
        Target slope of optimization.
    mtype : str, optional
        Type of summing up the differences. Possible types are ... 'mean'.
        Default is 'mean_abs'.
    ef : bool, optional
        If True, use an error function, otherwise False. Default is True. 
    
    Returns
    -------
    svdiff : float
        Single value of difference in optimization region.

    """
    # check if gradient is submitted
    if np.array(comp).ndim == 2 and np.array(opt).ndim == 2:
        comp_grad, opt_grad = comp[1,:], opt[1,:]
        comp_val , opt_val  = comp[0,:], opt[0,:]
        diff_grad = np.array(comp_grad) - np.array(opt_grad)
        twoD = True
    else:
        comp_val, opt_val = comp, opt
        twoD = False
    
    # difference between computed and optimum
    diff_opt = np.array(comp_val) - np.array(opt_val)
    # compute error function
    if ef and len(diff_opt) >=10:
        r        = np.arange(0, len(diff_opt)) # range number of points in opt region
        r_in     = 0.2 * len(diff_opt)    # start of using difference
        r_out    = 0.8 * len(diff_opt)    # end of using difference
        l        = 0.1 * len(diff_opt)   # transient start of difference usage
        coeffs   = (erf((np.sqrt(np.pi)/l)*(r-r_in)) - erf((np.sqrt(np.pi)/l)*(r-r_out))) / 2
        diff_opt = coeffs * diff_opt
        #diff_grad = coeffs * diff_grad # not used
    # different types to compute a single value quality criterion
    if mtype == 'quantiles':
        svdiff = mquantiles(diff_opt, [0.1, 0.9], alphap=0.5, betap=0.5)
        svdiff = svdiff[1] - svdiff[0]
        if twoD:
            svgrad = mquantiles(diff_grad, [0.1, 0.9], alphap=0.5, betap=0.5)
            svgrad = svgrad[1] - svgrad[0]
    elif mtype == 'mean':
        svdiff = np.mean(diff_opt)
        if twoD: svgrad = np.mean(diff_grad)
    elif mtype == 'mean_abs':
        svdiff = np.mean(np.abs(diff_opt)) 
        if twoD: svgrad = np.mean(np.abs(diff_grad))
    else:
        raise ValueError('unknown mtype %r' % mtype)
    
    if twoD: 
        return svdiff, svgrad
    else:
        return svdiff


def err_func(x_l, x_in, x_out, l):
    """
    Computes gaussian like function with error function

    Parameters
    ----------
    x_l : list
        Coordinates along function is computed. [start, stop, step_size]
    x_in : float
        Fade in of error function.
    x_out : float
        Fade out of error function.
    l : float
        Top of error function.

    Returns
    -------
    out : array
        Error function.

    """
    x        = np.arange(x_l[0], x_l[1], x_l[2]) # range number of points in opt region
    x_in     *= len(x)    # start of using difference
    x_out    *= len(x)    # end of using difference
    l        *= len(x)   # transient start of difference usage
    coeffs   = (erf((np.sqrt(np.pi)/l)*(x-x_in)) - erf((np.sqrt(np.pi)/l)*(x-x_out))) / 2
    out      = np.array([x, coeffs])
    return out


def gaussian(x_l, sig):
    """
    Compute Gaussian distribution

    Parameters
    ----------
    x_l : array or list
        Spatial points of gaussian distribution.
    sig : float
        Sigma auf gaussian distribution.

    Returns
    -------
    out : array
        Gaussian distribution.

    """
    x      = np.arange(x_l[0], x_l[1], x_l[2]) # range number of points in opt region
    coeffs = np.exp(-np.power(x, 2.) / (2 * np.power(sig, 2.)))
    out    = np.array([x,coeffs])
    return out


def sv_gaussian(x, sig, offset, mir=0):
    """
    Compute Gaussian distribution and flips the output if x is bigger than mir.

    Parameters
    ----------
    x : float
        Spatial point in  gaussian distribution.
    sig : float
        Sigma of gaussian distribution.
    offset : float
        Height of start of gaussian distribution.
    mir : float, optional
        Value to control point to flip. The default is 0.

    Returns
    -------
    coeff : float
        Value of spatial point in gaussian distribution.

    """
    coeff = offset + np.exp(-np.power(x, 2.) / (2 * np.power(sig, 2.)))*(1-offset)
    if x > mir:
        coeff = 1/coeff
    return coeff


def mirror(arr, x_l, mir=0):
    """
    Flips arr if x is bigger than mir.

    Parameters
    ----------
    arr : array
        Input array.
    x_l : array
        Information for gaussian distribution.
    mir : float, optional
        Value to control point to flip. The default is 0.

    Returns
    -------
    arr : array
        Output array.

    """
    x = np.arange(x_l[0], x_l[1], x_l[2])
    for n in range(len(x)):
        if x[n] > mir:
            arr[n] = 1/arr[n]
    return arr


def shift_to_reference(target, spl_over_distance, region_indices, reference_spl_over_distance):
    """
    Shifts SPL values on given reference.

    Parameters
    ----------
    target : obj [in]
        Contains information on the optimization target.
    spl_over_distance : obj [in]
        Contains computed SPL over distance data including distance between
        center of LSA and receiver positions.
    region_indices : list [in]
        Indice of points in optimization region.
    reference_spl_over_distance : obj [in]
        Contains computed SPL over distance data including distance between
        center of LSA and receiver positions of reference.

    Returns
    -------
    None.

    """
    # update reference index
    target.reference_index = np.argmin(np.abs(target.region_distances - target.reference_distance))
    # find index for spl_over_distance
    comp_ind = np.argmin(np.abs(spl_over_distance.distances - target.reference_distance))
    # compute shift
    shift     = target.region_levels_db[target.reference_index] - spl_over_distance.levels_db[comp_ind]
    # apply shift
    spl_over_distance.levels_db        += shift
    spl_over_distance.region_levels_db += shift
    if type(reference_spl_over_distance) is not list:
        shift_ref = target.region_levels_db[target.reference_index] - reference_spl_over_distance.levels_db[comp_ind]
        reference_spl_over_distance.levels_db        += shift_ref
        reference_spl_over_distance.region_levels_db += shift_ref
    return


def shift_to_zero(target, spl_over_distance, reference_spl_over_distance):
    """
    Shifts SPL on zero.

    Parameters
    ----------
    target : obj [in]
        Contains information on the optimization target.
    spl_over_distance : obj [in]
        Contains computed SPL over distance data including distance between
        center of LSA and receiver positions.
    reference_spl_over_distance : obj [in]
        Contains computed SPL over distance data including distance between
        center of LSA and receiver positions of reference.

    Returns
    -------
    None.

    """
    shift = spl_over_distance.levels_db[int(len(spl_over_distance.levels_db)/2)]
    shift_ref = reference_spl_over_distance.levels_db[int(len(reference_spl_over_distance.levels_db)/2)]
    spl_over_distance.levels_db     = [x - shift for x in spl_over_distance.levels_db]
    reference_spl_over_distance.levels_db = [x - shift_ref for x in reference_spl_over_distance.levels_db]


def optimize_target_slope(config, sound_field_config, palc_venue, ray_plot, \
                  optimized_array, drawn_lines, evaluation, spl_over_distance, target, \
                  max_loops=80):
    """
    Runs the while loop of target slope optimization. Will return the optimized
    weighting factors in ``config``. Called by :func:`pypalc.core.pipeline.run`.
    Calls mainly methods of :py:class:`TargetSlopeState`, this module and
    :py:mod:`pypalc.core.sound_field`.

    Every loop runs PALC and the sound field prediction with the weighting
    factors in ``config``, compares the SPL over distance with the target
    (:meth:`TargetSlopeState.store_previous`) and updates the weighting
    factors for the next loop (:func:`update_weighting`). If
    :meth:`~TargetSlopeState.store_previous` stops the optimization, PALC
    runs once more with the weighting factors it restored and the loop
    ends. After ``max_loops`` loops without such a stop the loop ends after
    the update: ``optimized_array`` holds the PALC result of the last loop,
    and ``config`` the weighting factors updated after it.

    Parameters
    ----------
    config : obj [in, out]
        Configuration of the PALC algorithm; holds the weighting factors.
    sound_field_config : obj [in]
        Configuration of the sound field prediction.
    palc_venue : obj [in]
        Data of the venue slice.
    ray_plot : obj [in, out]
        Data to plot the venue slice information.
    optimized_array : obj [out]
        Data of the optimized LSA.
    drawn_lines : obj [in]
        Venue slice that was drawn by the user.
    evaluation : obj [out]
        Technical Measure results depending on the sound field prediction.
    spl_over_distance : obj [out]
        Data of the Sound Pressure Level over Distance.
    target : obj [out]
        Optimized weighting data to run the target slope optimization.
    max_loops : int, optional
        Maximum number of loops. The default is 80; the pipeline passes
        :data:`pypalc.core.pipeline.TARGET_SLOPE_MAX_LOOPS`.

    Returns
    -------
    region_indices : list of int
        Indices of the SPL over distance points in the optimization region
        (:func:`optimization_region`) of the last loop.

    """
    loop = 0
    while int(loop) < int(max_loops):
        run_palc(ray_plot, palc_venue, config, optimized_array)
        patches_x, patches_y = cabinet_outlines(ray_plot, config, optimized_array.tilt_angles)
        ray_plot.get_plot_array(optimized_array)
        array_center_x, array_center_y = predict_sound_field(optimized_array.tilt_angles_deg, drawn_lines, sound_field_config, \
                            evaluation)
        compute_spl_over_distance(ray_plot, drawn_lines, evaluation.spl_db, spl_over_distance, sound_field_config.frequency_range_hz)
        
        # get difference between reference and PALC at optimization points
        # indices of optimization region
        region_indices = optimization_region(spl_over_distance, target)
        
        # shift target SPL at the reference distance on spl_over_distance
        shift_to_reference(target, spl_over_distance, region_indices, [])
        # indices that belong to the loudspeakers
        cabinet_point_indices, cabinet_region_indices = cabinet_points(config.cabinet_count, optimized_array, drawn_lines, region_indices=region_indices)
        # map values on LS
        target.cabinet_differences, target.cabinet_gradient_differences = differences_per_cabinet(config.cabinet_count, region_indices, \
                                                        cabinet_point_indices, cabinet_region_indices, \
                                                        target, spl_over_distance)
        target.cabinet_differences = smooth_list(target.cabinet_differences)
        # region_indices: pal points that belong to opt region
        # diff_opt: difference in opt_region
        # cabinet_region_indices: opt region pal points sorted to loudspeaker
        # cabinet_point_indices: pal points sorted to loudspeaker 
        # removed region_indices behind, change mean of region_levels_db
        svdiff, svgrad = single_value_difference(np.array([spl_over_distance.region_levels_db, \
                                                            spl_over_distance.region_gradients]), \
                                                  np.array([target.region_levels_db, \
                                                            target.region_gradients]), \
                                                  mtype='quantiles', ef=True)
        svtot = svdiff * svgrad
        out = target.store_previous(config, svtot, optimized_array.iteration_count)
        if out:
            run_palc(ray_plot, palc_venue, config, optimized_array)
            break
        # update weighting
        update_weighting(config, target, loop)
        loop += 1
    return region_indices


def snap_to_distance_grid(target, distances, points_x, points_y, shown_target,
                          reference_distance):
    """
    Moves the target slope points and the reference point to the nearest
    points of the SPL over distance grid and interpolates the target slope on
    the grid between its first and last point. This is the computation of the
    GUI callback ``update_draw``.

    Parameters
    ----------
    target : TargetSlopeState [out]
        Target slope state. The attributes distances, levels_db, points_x,
        points_y, region_distances, region_levels_db, reference_distance and
        reference_index are written.
    distances : ndarray
        Distance grid: distances between the center of the LSA and the
        venue points.
    points_x, points_y : ndarray
        Venue points belonging to the distances.
    shown_target : dict
        Target slope points with the keys 'distance_m', 'level_db', 'x_m'
        and 'y_m' (lists of equal length; the columns of the target source
        of the GUI and of ``target_slope.csv``). Like the ColumnDataSource of
        the GUI, the lists 'distance_m', 'x_m' and 'y_m' are snapped in
        place.
    reference_distance : list of float
        One-element list with the distance of the reference point.

    Returns
    -------
    None.

    """
    target.distances = shown_target['distance_m']
    target.levels_db = shown_target['level_db']
    target.points_x, target.points_y = shown_target['x_m'], shown_target['y_m']
    target.reference_distance = reference_distance
    for ind, val in enumerate(target.distances):
        arg = np.argmin(np.abs(np.array(distances)-val))
        target.distances[ind] = distances[arg]
        target.points_x[ind], target.points_y[ind] = points_x[arg], points_y[arg]
    # sort so that smaller x value is on indice 0
    target.sort_points()
    target.region_distances = list(distances[np.argmin(np.abs(distances-target.distances[0])):np.argmin(np.abs(distances-target.distances[-1]))+1])
    target.interpolate_levels()
    # find nearest index for the reference distance
    reference_index = np.argmin(np.abs(target.reference_distance[0] - np.array(target.region_distances)))
    target.reference_distance, target.reference_index = target.region_distances[reference_index], reference_index
