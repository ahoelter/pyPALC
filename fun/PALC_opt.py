#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Arne Hoelter

Contains functions to compute the target slope optimization.
"""
import numpy as np
from scipy.stats.mstats import mquantiles
from scipy.special import erf


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
    SPLoverX.SPL_interp = SPLoverX.SPL - np.amax(SPLoverX.SPL[ind])
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
    # tolerance (optional input)
    if 'tol' not in kwargs.keys(): kwargs['tol'] = 0.00001
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


def diff_on_ls(N, opt_ind, ls_ind, ls_opt_ind, comp, opt):
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
    comp : list or 1D-array
        Computed PALC results in optimization region.
    opt : list or 1D-array
        Target Slope of optimization.

    Returns
    -------
    diffLS : list
        Mean average of difference between target slope and PALC computed
        results regarding each LSA cabinet. If a LSA cabinet does not hit any
        point in the optimization region, the value is set to 100.

    """
    # first list in diff_comp_opt is computed results and secont optization target
    diffLS, diff_comp_opt_LS = [], [[],[]]
    for n in range(N):
        diffLS.append([])
        for m,val in enumerate(opt_ind):
            if val in ls_opt_ind[n]:
                #diffLS[n].append(diff_opt[m])
                diff_comp_opt_LS[0].append(comp[m])
                diff_comp_opt_LS[1].append(opt[m])
        if len(ls_opt_ind[n]) >= 1:
            diffLS[n] = calc_SingleValueDiff(diff_comp_opt_LS[0], \
                                             diff_comp_opt_LS[1], \
                                             mtype='mean')
        else:
            diffLS[n] = 100
    return diffLS


def smooth_1D_list(data, nbs=0.2):
    """
    Smoothes a 1D array with the neighbouring entry.

    Parameters
    ----------
    data : 1D-array
        Array to be smoothed.
    nbs : float, optional
        Weight of the neighbouring point, means
        new[n] = old[n-1]*nbs + old[n]*(1-2*nbs) + old[n+1]*nbs.
        The default is 0.2.

    Returns
    -------
    1D-array
        Smoothed array.

    """
    s_data = np.zeros(len(data))
    for n in range(0,len(data)):
        if data[n] < 0.1:
            s_data[n] = data[n]
            continue
        for row in range(-1,2):
            if n+row < 0 or n+row >= len(data):
                continue
            else:
                if (n==0 or n==len(data)-1) and row==0:
                    s_data[n] += data[n+row]*(1-nbs)
                elif row==0:
                    s_data[n] += data[n+row]*(1-2*nbs)
                else:
                    s_data[n] += data[n+row]*nbs
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

    w = PALC_config.weighting_weights
    news = np.ones(PALC_config.N)
    PALC_config.weighting_weights[0] = 1.0
    opt_coeff = np.genfromtxt('pyPALC_bokeh/opt_list.csv', delimiter=',')
    in_opt = False
    for n in range(1,PALC_config.N):
        if Opt_w.diffLS[n] == 100:
            #Opt_w.diffLS[n] = 0
            if not in_opt:
                coeff = 1.2
            else:
                coeff = 0.8
        else:
            in_opt = True
            coeff = opt_coeff[np.argmin(np.abs(opt_coeff[:,0]-Opt_w.diffLS[n])),1]
        news[n] = news[n-1]*coeff
        PALC_config.weighting_weights[n] = news[n] * w[n]
    return


def calc_SingleValueDiff(comp, opt, mtype='mean_abs'):
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
    
    Returns
    -------
    svdiff : float
        Single value of difference in optimization region.

    """
    # difference between computed and optimum
    diff_opt = np.array(comp) - np.array(opt)
    # different types to compute a single value quality criterion
    if mtype == 'quantiles':
        svdiff = mquantiles(diff_opt, [0.1, 0.9], alphap=0.5, betap=0.5)
        svdiff = svdiff[1] - svdiff[0]
    elif mtype == 'err_func':
        r       = np.arange(0, len(diff_opt)) # range number of points in opt region
        r_in    = int(0.2 * len(diff_opt))    # start of using difference
        r_out   = int(0.8 * len(diff_opt))    # end of using difference
        l       = int(0.1 * len(diff_opt))   # transient start of difference usage
        coeffs = (erf((np.sqrt(np.pi)/l)*(r-r_in)) - erf((np.sqrt(np.pi)/l)*(r-r_out))) / 2
        # mean
        #svdiff = np.mean(np.abs(coeffs * diff_opt))
        #quantiles
        svdiff = mquantiles(coeffs*diff_opt, [0.1, 0.9], alphap=0.5, betap=0.5)
        svdiff = svdiff[1] - svdiff[0]
    elif mtype == 'mean':
        svdiff = np.mean(diff_opt)
    elif mtype == 'mean_abs':
        svdiff = np.mean(np.abs(diff_opt))
    else:
        print('wrong input')
    
    return svdiff