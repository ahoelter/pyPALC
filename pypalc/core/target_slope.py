"""Target slope optimization of the PALC weighting.

The state of the target slope (:class:`Opt_weight`), the helpers of the
optimization and its loop :func:`optimize_PALC` (formerly
``fun/PALC_opt.py``, ``Opt_weight`` from ``fun/PALC_classes.py`` and
``optimize_PALC`` from ``fun/gui_algorithm_sfp.py``).

author: Arne Hoelter
"""
from __future__ import annotations

import logging

import numpy as np
from numpy import array as nar
from scipy.special import erf
from scipy.stats.mstats import mquantiles

from .geometry import LSA_visualization
from .palc import calcPALC
from .sound_field import calcSFP, calcSPLoverX

logger = logging.getLogger(__name__)


class Opt_weight():
    """ Class to optimize PALC by the target slope optimization. """
    def __init__(self, SPL_interp=None):
        self.x, self.SPL   = [], []
        self.x_interp      = [0,0]
        self.SPL_interp    = [0,0] if SPL_interp is None else SPL_interp
        self.init          = False
        self.x_v, self.y_v = [0,0], [0,0]
        self.diffLS        = []
        self.diffps        = {'n-1':100, 'lowest':[],'same':0}
        self.w_ps          = {'n-1':[], 'lowest':[]}
        self.is2update     = 0
        self.SPL_grad      = []
        self.opt_ind       = 0
        self.diffgradLS    = []
        self.x_ref         = []
        self.ref_ind       = []
        
    def calc_init(self):
        """
        Initializes the target slope optimization before starting the
        optimization.

        Returns
        -------
        None.

        """
        self.diffLS        = []
        self.diffps      = {'n-1':100, 'lowest':[]}
        self.w_ps          = {'n-1':[], 'lowest':[]}
        
    def CDS2obj(self, CDS, keys):
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
        
        
    def maxonref(self, ref=0):
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
        shift           = np.amax(self.SPL_interp)-ref
        self.SPL        = self.SPL - shift
        self.SPL_interp = self.SPL_interp - shift 


    def shift2psval(self, PALC_config, diff_opt, num_iter):
        """
        Shifts the actual value to a list that stores the past value. These are
        used to check if the actual loop improved that the difference between
        target and computation result.

        Parameters
        ----------
        PALC_config : obj [in, out]
            Contains the configuration of the PALC algorithm.
        diff_opt : float
            Single value of the optimized PALC computation.

        Returns
        -------
        bool
            True, if diff_opt was 5 times higher than past value. Then break
            of while optimization loop. Otherwise False.

        """
        if np.round(diff_opt, decimals=8) == np.round(self.diffps['n-1'], decimals=8):
            self.diffps['same'] += 1
        else:
            self.diffps['same'] = 0
        if diff_opt > self.diffps['n-1']:
            self.diffps['lowest'].append(self.diffps['n-1'])
            self.w_ps['lowest'].append(self.w_ps['n-1'])
        self.diffps['n-1'] = diff_opt
        self.w_ps['n-1'] = np.array(PALC_config.weighting_weights[:])
        if len(self.w_ps['lowest']) == 10 or self.diffps['same'] == 5 or num_iter==500:
            low_ind = np.argmin(self.diffps['lowest'])
            PALC_config.weighting_weights = self.w_ps['lowest'][low_ind][:]
            return True
        return False
            

    def resort_opt_region(self):
        """
        Resorts the target slope hinge points, so that the x-axis / SPL pairs
        begin with lowest x-axis values.

        Returns
        -------
        None.

        """
        sort_key = np.argsort(self.x)
        for key in ['x', 'SPL', 'x_v', 'y_v']:
            setattr(self, key, list(np.array(getattr(self,key))[sort_key]))
            
    def interpolate_SPL(self):
        """
        Interpolates the drawn target slope line on the discrete distance
        points, that are used von SPL over distance calculation of the
        computed results.

        Returns
        -------
        None.

        """
        self.SPL_interp, ap, ind = [], [], [0]
        for n in range(1,len(self.x)):
            ind.append(np.argwhere(self.x[n] == self.x_interp)[0,0])
            ap.append(list((self.SPL[n]-self.SPL[n-1])/(self.x[n]-self.x[n-1])* \
                                   (np.array(self.x_interp[ind[n-1]:ind[n]])- \
                                    self.x[n-1])+self.SPL[n-1]))
        ap.append([self.SPL[-1]])
        list(map(self.SPL_interp.extend, ap))


def get_weight_links(n_links, SPLoverX):
    """
    Init function to return indices of the hinges / links in :any:`pSPLoverX`.
    Necessary for the movable target slope line. Works also if user has moved
    the hinge / link. Called by :any:`chg_weight_links`.

    Parameters
    ----------
    n_links : int
        Number of hinges / links in the target slope line.
    SPLoverX : obj [in]
        Contains computed SPL over distance data including distance between
        center of LSA and receiver positions.

    Returns
    -------
    ind : list
        Indices (int) where the hinges / links are placed in :any:`pSPLoverX`.

    """
    tot_ind = len(SPLoverX.x)
    ind = [0] * (n_links+2)
    for n in range(n_links+2):
        ind[n] = n * int(tot_ind/(n_links+1))
        if n > 0: ind[n] -= 1
    return ind


def get_opt_region(SPLoverX, Opt_w):
    """
    Find the indices of the user defined optimization region in
    :any:`pSPLoverX`. :any:`start_calc` and :any:`optimize_PALC`.

    Parameters
    ----------
    SPLoverX : obj [in, out]
        Contains SPL over distance data of the PALC computed results. Attribute
        SPL_interp is normalized to the maximum in the optimization
        region.
    Opt_w : obj [in]
        Contains SPL over distance data of the optimized results and the target
        slope.

    Returns
    -------
    ind : list
        Indices (int) of the optimization region in center of LSA to receiver
        position distance of the attribute x of :any:`SPLoverX`.

    """
    ind = []
    for val in list(Opt_w.x_interp):
        ind.append(np.argmin(np.abs(np.round(list(SPLoverX.x),2) - np.round(val,2))))#[0,0])
    #SPLoverX.SPL_interp = SPLoverX.SPL - np.amax(SPLoverX.SPL[ind])
    SPLoverX.SPL_interp = SPLoverX.SPL[ind]
    Opt_w.x_interp      = SPLoverX.x[ind]
    Opt_w.x             = [SPLoverX.x[ind[0]], SPLoverX.x[ind[-1]]]
    Opt_w.SPL_grad      = np.gradient(np.array(Opt_w.SPL_interp))
    SPLoverX.SPL_grad   = np.gradient(np.array(SPLoverX.SPL_interp))
    SPLoverX.opt_ind    = ind
    return ind


def get_diff_optreg(comp, opt):
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


def ls2pal(N, Opt_arr, created_pal, **kwargs):
    """
    Maps LSA cabinets to discrete venue points depending on the assumed splay
    angle. Called by :any:`optimize_PALC`.

    Parameters
    ----------
    N : int
        Number of LSA cabinets.
    Opt_arr : obj [in]
        Contains results of the PALC computation.
    created_pal : obj [in]
        Venue information that was drawn by the user.
    **kwargs : dict, optional
        'tol' : Tolerance of discrete points to find (default = 10⁻⁵ m).
        'opt_ind' : Indices of optimization region in pal / venue slice.

    Returns
    -------
    ls2pal : list
        Indices of audience line points sorted to the LSA cabinets.
    ls2pal_opt : list
        Indices in optimization region sorted to the LSA cabinets.

    """
    # tolerance (optional input) default is ndiscretization / np.sqrt(2)
    discr = np.sqrt((created_pal.xline[1]-created_pal.xline[0])**2 + \
                    (created_pal.yline[1]-created_pal.yline[0])**2) / np.sqrt(2)
    if 'tol' not in kwargs.keys(): kwargs['tol'] = discr#0.00001
    tol = kwargs['tol']
    # list to store index of pal for each ls
    ls2pal, ls2pal_opt = [], []
    for n in range(N):
        ind = [[],[]]
        # check x-values
        for m,val in enumerate(created_pal.xline):
            if val >= Opt_arr.seg_pos_stop[n][0]-tol and val <= Opt_arr.seg_pos_start[n][0]+tol:
                ind[0].append(m)
        # check y-values
        for m,val in enumerate(created_pal.yline):
            if val >= Opt_arr.seg_pos_stop[n][1]-tol and val <= Opt_arr.seg_pos_start[n][1]+tol:
                ind[1].append(m)
        # check for matching index
        ls2pal.append(list(set(ind[0]) & set(ind[1])))
    if 'opt_ind' in kwargs.keys():
        for n in range(N):
            ls2pal_opt.append(list(set(ls2pal[n]) & set(kwargs['opt_ind'])))
        return ls2pal, ls2pal_opt
    return ls2pal


def diff_on_ls(N, opt_ind, ls_ind, ls_opt_ind, Opt_w, SPLoverX, **kwargs):
    """
    Maps diference in optimization region to LSA cabinets. Called by
    :any:`optimize_PALC`.

    Parameters
    ----------
    N : int
        Number of LSA cabinets.
    opt_ind : list
        Indices of optimization region.
    ls_ind : list
        Indices of venue slice mapped on LSA cabinets.
    ls_opt_ind : list
        Indices of optimization region mapped on LSA cabinets.
    SPLoverX : obj [in, out]
        Object with computed PALC results in optimization region.
    Opt_w : obj [in, out]
        Object with information of Target Slope in optimization.
    created_pal : obj [in]
        Venue information that was drawn by the user.
    **kwargs : dict, optional
        Optional input to set single value difference

    Returns
    -------
    diffLS : list
        Mean average of difference between target slope and PALC computed
        results regarding each LSA cabinet. If a LSA cabinet does not hit any
        point in the optimization region, the value is set to 100.

    """
    if 'mtype' not in kwargs.keys(): kwargs['mtype'] = 'mean'
    if 'lsmap' not in kwargs.keys(): kwargs['lsmap'] = 'in_coverage'
    # write computed and optimized interpolated SPL values to variable
    opt       = np.array(Opt_w.SPL_interp)
    comp      = np.array(SPLoverX.SPL_interp)# removed opt_ind
    grad_opt  = np.array(Opt_w.SPL_grad)
    grad_comp = np.array(SPLoverX.SPL_grad)# removed opt_ind
    # compute maximum neighbouring points of each loudspeaker impact point
    nebs = get_nbs_neighbours(N, ls_opt_ind, opt_ind)
    # first list in diff_comp_opt is computed results and secont optization target
    diffLS, diffgradLS, diff_comp_opt_LS, diff_grad = [], [], [[],[]], [[],[]]
    for n in range(N):
        diffLS.append([])
        diffgradLS.append([])
        for m,val in enumerate(opt_ind):
            if val in ls_opt_ind[n]:
                diff_comp_opt_LS[0].append(comp[m])
                diff_comp_opt_LS[1].append(opt[m])
                diff_grad[0].append(grad_comp[m])
                diff_grad[1].append(grad_opt[m])
        if len(ls_opt_ind[n]) >= 1:
            frontend         = range_neighbours(comp, diff_comp_opt_LS)
            diff_comp_opt_LS = insert_neighbours(comp, opt, diff_comp_opt_LS, \
                                                  nebs[n,:], frontend)
            diff_grad        = insert_neighbours(grad_comp, grad_opt, diff_grad, \
                                                  nebs[n,:], frontend)
            diffLS[n], diffgradLS[n] = calc_SingleValueDiff(nar([nar(diff_comp_opt_LS[0]), \
                                                                 nar(diff_grad[0])]), \
                                                            nar([nar(diff_comp_opt_LS[1]), \
                                                                 nar(diff_grad[1])]), \
                                                            mtype=kwargs['mtype'], ef=True)
        else:
            diffLS[n], diffgradLS[n] = 100, 100
    
    return diffLS, diffgradLS


def get_nbs_neighbours(N, ls_opt_ind, opt_ind):
    """
    Computes the number of neighbours on which it could be smoothed. Called
    by :any:`diff_on_ls`.

    Parameters
    ----------
    N : int
        Number of LSA cabinets.
    ls_opt_ind : list
        List of LSA cabinets mapped to audience positions.
    opt_ind : list
        Indices of optimization region.

    Returns
    -------
    nebs: ndarray
        Number of possible neighbouring points for each cabinet.

    """
    nebs = np.zeros((N,2),dtype=int)
    for n in range(N):
        if ls_opt_ind[n] != []:
            nebs[n,0] = min(ls_opt_ind[n]) - min(opt_ind)
            nebs[n,1] = max(opt_ind) - max(ls_opt_ind[n])
    return nebs


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


def insert_neighbours(comp, opt, diff_comp_opt_LS, nebs, frontend, to_ins=5):
    """
    Insert neighbouring points of the LSA cabinet to for difference of
    computed and target values. Called by :any:`diff_on_ls`.

    Parameters
    ----------
    comp : list or 1D-array
        Computed PALC results in optimization region.
    opt : list or 1D-array
        Target Slope of optimization.
    diff_comp_opt_LS : list
        Contains the computed and target values of actual LSA cabinet.
    nebs : list
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
    if nbs > nebs[0]:
        fr_rng = nebs[0]
    else:
        fr_rng = nbs
    if nbs > nebs[1]:
        end_rng = nebs[1]
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


def smooth_1D_list(data, nbs_w=0.2, nums=1):
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


def opt_weighting(PALC_config, Opt_w, loop):
    """
    Updates the weighting factors depending on differences at the LSA cabinets
    and weighting factors of the previous loop of target slope optimization.
    The break out condition depends equally on all positions in the
    optimization region and an additional (and optional) function to compute
    a single difference value. . Called by :any:`optimize_PALC`.

    Parameters
    ----------
    PALC_config : obj [in, out]
        Configuration of PALC algorithm. The attribute weighting_weights will
        be updated for the next PALC computation.
    Opt_w : obj [in]
        Contains information on the optimization target.
    loop : int
        Number of actual loop of target slope optimization.

    Returns
    -------
    None.

    """

    w    = PALC_config.weighting_weights
    news = np.ones(PALC_config.N)
    PALC_config.weighting_weights[0] = 1.0
    
    grad_coeff = gaussian([-.4,.4,0.01], .15)
    grad_coeff[1,:] = mirror(0.8+grad_coeff[1,:]*.2,[-.4,.4,0.01])
    in_opt     = False

    for n in range(1,PALC_config.N):
        if Opt_w.diffLS[n] == 100:
            if not in_opt:
                coeff = 1.2
            else:
                coeff = 0.6
            gcoeff = 1.0
        else:
            in_opt = True
            coeff  = sv_gaussian(Opt_w.diffLS[n], 1.2, .65)
            gcoeff = sv_gaussian(Opt_w.diffgradLS[n], .15, .8)
        news[n] = news[n-1]*coeff*gcoeff
        PALC_config.weighting_weights[n] = news[n] * w[n]
    return


def calc_SingleValueDiff(comp, opt, mtype='mean', ef=True):
    """
    Computes a single value difference of the actual PALC results. Called by
    :any:`optimize_PALC` and :any:`diff_on_ls`.

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


def shift2ref(Opt_w, SPLoverX, opt_ind, SPLoverX_ref):
    """
    Shifts SPL values on given reference.

    Parameters
    ----------
    Opt_w : obj [in]
        Contains information on the optimization target.
    SPLoverX : obj [in]
        Contains computed SPL over distance data including distance between
        center of LSA and receiver positions.
    opt_ind : list [in]
        Indice of points in optimization region.
    SPLoverX_ref : obj [in]
        Contains computed SPL over distance data including distance between
        center of LSA and receiver positions of reference.

    Returns
    -------
    None.

    """
    # update reference index
    Opt_w.ref_ind = np.argmin(np.abs(Opt_w.x_interp - Opt_w.x_ref))
    # find index for SPLoverX
    comp_ind = np.argmin(np.abs(SPLoverX.x - Opt_w.x_ref))
    # compute shift
    shift     = Opt_w.SPL_interp[Opt_w.ref_ind] - SPLoverX.SPL[comp_ind]
    # apply shift
    SPLoverX.SPL        += shift
    SPLoverX.SPL_interp += shift
    if type(SPLoverX_ref) is not list:
        shift_ref = Opt_w.SPL_interp[Opt_w.ref_ind] - SPLoverX_ref.SPL[comp_ind]
        SPLoverX_ref.SPL        += shift_ref
        SPLoverX_ref.SPL_interp += shift_ref
    return


def shift_ref_on_zero(Opt_w, SPLoverX, SPLoverX_ref):
    """
    Shifts SPL on zero.

    Parameters
    ----------
    Opt_w : obj [in]
        Contains information on the optimization target.
    SPLoverX : obj [in]
        Contains computed SPL over distance data including distance between
        center of LSA and receiver positions.
    SPLoverX_ref : obj [in]
        Contains computed SPL over distance data including distance between
        center of LSA and receiver positions of reference.

    Returns
    -------
    None.

    """
    shift = SPLoverX.SPL[int(len(SPLoverX.SPL)/2)]
    shift_ref = SPLoverX_ref.SPL[int(len(SPLoverX_ref.SPL)/2)]
    SPLoverX.SPL     = [x - shift for x in SPLoverX.SPL]
    SPLoverX_ref.SPL = [x - shift_ref for x in SPLoverX_ref.SPL]


def optimize_PALC(PALC_config, SFP_config, PALC_pal, PALC_plots, \
                  Opt_arr, created_pal, Tech_res, SPLoverX, Opt_w, \
                  max_loops=80):
    """
    Runs the while loop of target slope optimization. Will return the optimized
    weighting factors in :any:`Opt_w`. Called by :any:`start_calc`. Calls mainly
    methods of :py:class:`Opt_weight`, this module and
    :py:mod:`pypalc.core.sound_field`.

    Parameters
    ----------
    PALC_config : obj [in]
        Configuration of the PALC algorithm.
    SFP_config : obj [in]
        Configuration of the sound field prediction.
    PALC_pal : obj [in]
        Data of the venue slice.
    PALC_plots : obj [in, out]
        Data to plot the venue slice information.
    Opt_arr : obj [out]
        Data of the optimized LSA.
    created_pal : obj [in]
        Venue slice that was drawn by the user.
    Tech_res : obj [out]
        Technical Measure results depending on the sound field prediction.
    SPLoverX : obj [out]
        Data of the Sound Pressure Level over Distance.
    Opt_w : obj [out]
        Optimized weighting data to run the target slope optimization.
    max_loops : int, optional
        Maximum of loops to compute. The default is 40.

    Returns
    -------
    None.

    """
    loop = 0
    while int(loop) < int(max_loops):
        calcPALC(PALC_plots, PALC_pal, PALC_config, Opt_arr)
        x_patches, y_patches = LSA_visualization(PALC_plots, PALC_config, Opt_arr.gamma_n)
        PALC_plots.get_plot_array(Opt_arr)
        x_S, y_S = calcSFP(Opt_arr.gamma_tilt_deg, created_pal, SFP_config, \
                            Tech_res)#, dir_meas_LSA, dir_meas_degree)
        calcSPLoverX(PALC_plots, created_pal, Tech_res.p_SPL, SPLoverX, SFP_config.freq_range)
        
        # get difference between reference and PALC at optimization points
        # indices of optimization region
        opt_ind = get_opt_region(SPLoverX, Opt_w)
        
        # shift Opt_w SPL at x_ref on SPLoverX
        shift2ref(Opt_w, SPLoverX, opt_ind, [])
        # indices that belong to the loudspeakers
        ls_ind, ls_opt_ind = ls2pal(PALC_config.N, Opt_arr, created_pal, opt_ind=opt_ind)
        # map values on LS
        Opt_w.diffLS, Opt_w.diffgradLS = diff_on_ls(PALC_config.N, opt_ind, \
                                                        ls_ind, ls_opt_ind, \
                                                        Opt_w, SPLoverX)
        Opt_w.diffLS = smooth_1D_list(Opt_w.diffLS)
        # opt_ind: pal points that belong to opt region
        # diff_opt: difference in opt_region
        # ls_opt_ind: opt region pal points sorted to loudspeaker
        # ls_ind: pal points sorted to loudspeaker 
        # removed opt_ind behind, change mean of SPL_interp
        svdiff, svgrad = calc_SingleValueDiff(np.array([SPLoverX.SPL_interp, \
                                                            SPLoverX.SPL_grad]), \
                                                  np.array([Opt_w.SPL_interp, \
                                                            Opt_w.SPL_grad]), \
                                                  mtype='quantiles', ef=True)
        svtot = svdiff * svgrad
        out = Opt_w.shift2psval(PALC_config, svtot, Opt_arr.num_iter)
        if out:
            calcPALC(PALC_plots, PALC_pal, PALC_config, Opt_arr)
            break
        # update weighting
        opt_weighting(PALC_config, Opt_w, loop)
        loop += 1
    return opt_ind


def snap_to_distance_grid(Opt_w, x, x_v, y_v, target, x_ref):
    """
    Moves the target slope points and the reference point to the nearest
    points of the SPL over distance grid and interpolates the target slope on
    the grid between its first and last point. This is the computation of the
    GUI callback ``update_draw``.

    Parameters
    ----------
    Opt_w : obj [out]
        Target slope state. The attributes x, SPL, x_v, y_v, x_interp,
        SPL_interp, x_ref and ref_ind are written.
    x : ndarray
        Distance grid: distances between the center of the LSA and the
        venue points.
    x_v, y_v : ndarray
        Venue points belonging to the distances in x.
    target : dict
        Target slope points with the keys 'x', 'SPL', 'x_v' and 'y_v' (lists
        of equal length). Like the ColumnDataSource of the GUI, the lists
        'x', 'x_v' and 'y_v' are snapped in place.
    x_ref : list of float
        One-element list with the distance of the reference point.

    Returns
    -------
    None.

    """
    Opt_w.x, Opt_w.SPL = target['x'], target['SPL']
    Opt_w.x_v, Opt_w.y_v = target['x_v'], target['y_v']
    Opt_w.x_ref = x_ref
    for ind, val in enumerate(Opt_w.x):
        arg = np.argmin(np.abs(np.array(x)-val))
        Opt_w.x[ind] = x[arg]
        Opt_w.x_v[ind], Opt_w.y_v[ind] = x_v[arg], y_v[arg]
    # sort so that smaller x value is on indice 0
    Opt_w.resort_opt_region()
    Opt_w.x_interp = list(x[np.argmin(np.abs(x-Opt_w.x[0])):np.argmin(np.abs(x-Opt_w.x[-1]))+1])
    Opt_w.interpolate_SPL()
    # find nearest index for x_ref
    ref_ind = np.argmin(np.abs(Opt_w.x_ref[0] - np.array(Opt_w.x_interp)))
    Opt_w.x_ref, Opt_w.ref_ind = Opt_w.x_interp[ref_ind], ref_ind
