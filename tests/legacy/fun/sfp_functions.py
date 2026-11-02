#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: arne Hoelter

Function to be used to compute the sound field prediction with the
Complex Directivity Point Source (CDPS)-model.
"""

import numpy as np
from PALC_functions import repmat, source_pos
import scipy.special as sp
import scipy.signal as sig
from scipy.stats.mstats import mquantiles


def get_freq_vec(N_freq, step_freq, freq_range):
    """
    Compute a linear frequency vector. Called by :any:`calcSFP` and
    :any:`init_dir_plot`.

    Parameters
    ----------
    N_freq : int
        Number of frequency bins.
    step_freq : int
        Step size of frequency vector.
    freq_range: tuple or list
        Contains start and stop frequency of frequncy vector

    Returns
    -------
    f : 1D-array
        Frequency vector.

    """
    f = 20 * (2 * np.ones(N_freq+1))**(step_freq * np.arange(N_freq+1))
    f = f[f <= freq_range[1]]
    f = f[f >= freq_range[0]]
    f = np.unique(f)
    return f


def greens_fct(x0, y0, omega, c, x, y, z):
    """
    Computes the 3D Green's function. The acoustic transfer function in air.
    Called by :any:`CalcGreenFunctions`.

    Parameters
    ----------
    x0 : float
        Source position x-axis.
    y0 : float
        Source position y-axis.
    omega : float
        Discrete radian frequency.
    c : float
        Speed of sound.
    x : 1D-array
        Receiver positions x-axis.
    y : 1D-array
        Receiver positions y-axis.
    z : 1D-array
        Receiver positions z-axis.

    Returns
    -------
    G : 1D-array
        Computed Green's function for all receiver positions.

    """
    G = np.exp( -complex(0,1) * ( omega / c * np.sqrt((x-x0)**2 + (y-y0)**2 + \
                                 z**2))) / np.sqrt((x-x0)**2 + (y-y0)**2 + z**2)
    return G


def AirAbsorptionRelaxationFrequencies(T,p,H,T0, p_r):
    """
    Calculates the relaxation frequencies for air absorption conforming to
    ISO 9613-1. Called by :any:`AirAbsorptionCoefficient`.

    Parameters
    ----------
    T : float
        Temperature in K.
    p : float
        Pressure in Pa.
    H : float
        Humidity as molar conentration in percent.
    T0 : float
        Reference temperature in K, 293.15 K.
    p_r : float
        Reference sound pressure in Pa, 101.325*10³ Pa.

    Returns
    -------
    f_rO : float
        Relaxation frequency of oxygen.
    f_rN : float
        Relaxation frequency of nitrogen.

    """
    
    f_rO = p / p_r * (24 + 4.04 * 10**4 * H * (0.02+H) / (0.391+H))
    
    f_rN = p / p_r * (T/T0)**(-0.5) * (9+280*H*np.exp(-4.17*((T/T0)**(-1/3)-1)))
    
    return f_rO, f_rN


def AirAbsorptionCoefficient(f, T, p, h):
    """
    Calculates the air absorption coefficient alpha. Called by
    :any:`calcSFP` and :any:`calcBeamplot`.
    Calls :any:`AirAbsorptionRelaxationFrequencies`. Uses the following
    constants:
        
        * p_r        = 101.325*10³ Pa as reference sound pressure
        * T0         = 293.15 K       as reference temperature in K for
                                         relaxation frequencies and alpha
        * T0_p_sat_r = 273.16 K       as reference temperature in K for p_sat_r
        * c0         = 331.45 m/s     as reference speed of sound

    Parameters
    ----------
    f : float
        Frequency in Hz.
    T : float
        Temperature in K.
    p : float
        Pressure in Pa.
    h : float
        Relative humidity in percent.

    Returns
    -------
    alpha : float
        Attenuation coefficient in dB/m.
    c : float
        Speed of sound.

    """
    p_r = 101.325*(10**3)   # reference sound pressure in Pa
    T0 = 293.15             # reference termperature in K for the relaxation frequencies and alpha
    T0_p_sat_r = 273.16     # reference termperature in K for p_sat_r
    c0 = 331.45             # reference speed of sound in m/s
    
    p_sat_r = 10**(-6.8346*(T0_p_sat_r/T)**1.261 + 4.6151)
    H = h * p_sat_r * p_r / p
    
    f_rO, f_rN = AirAbsorptionRelaxationFrequencies(T,p,H,T0, p_r)
    
    alpha = 8.686 * f**2 * ((1.84*(10**-11) * (p/p_r)**(-1) * (T/T0)**(0.5)) \
            + (T/T0)**(-5/2) * (0.01275 * np.exp(-2239.1/T) / (f_rO + f**2 / f_rO) + \
               0.1068 * np.exp(-3352/T) / (f_rN + f**2 / f_rN)))
    
    c = c0 * np.sqrt(29 * T / (1.4*(T0-20))) * np.sqrt(((700+H) / (500+H)) / (29-0.11*H))

    return alpha, c


def LinkwitzRileyCrossOver(f_c_m, fs, f):
    """
    Computes a Linkwitz-Riley-Crossover filter. The lowpass and highpass are
    of second order butterworth filter. Called by :any:`calc_directivity`.

    Parameters
    ----------
    f_c_m : float
        Cut-off frequency.
    fs : float
        Sampling frequency.
    f : list or 1D-array
        Considered frequency bins.

    Returns
    -------
    D_xo_low : 1D-array
        Complex frequency response of the Linkwitz-Riley lowpass of
        the selected BP frequencies and the selected sampling frequency.
    D_xo_high : 1D-array
        Complex frequency response of the Linkwitz-Riley highpass of
        the selected BP frequencies and the selected sampling frequency.

    """
    # transform the frequency bins into normalized radian frequency
    # Lowpass
    # lowpass coefficients
    b_xo_low, a_xo_low = sig.butter(2, f_c_m/(fs/2), btype='lowpass')
    # Complex frequency response of the second order Butterworth filter
    f_xo_low, D_xo_low = sig.freqz(b_xo_low, a_xo_low, worN=f, fs=fs)
    D_xo_low = D_xo_low * D_xo_low
    
    # Highpass
    # highpass coefficients
    b_xo_high, a_xo_high = sig.butter(2, f_c_m/(fs/2), btype='highpass')
    # Complex frequency response of the second order Butterworth filter
    f_xo_low, D_xo_high = sig.freqz(b_xo_high, a_xo_high, worN=f, fs=fs)
    D_xo_high = D_xo_high * D_xo_high
    
    return D_xo_low, D_xo_high

   
def calc_directivity(dire, alpha, Lambda_y, beta, omega, c, f, dir_meas, dir_meas_deg, n):
    """
    Compute the LSA cabinets directivity. Possible directivities are Circular
    Piston, Line Piston, Combined Circular / Line Piston, Measured
    Loudspeaker Data or Constant (Monopole). The Crossover frequency is set
    to 1.5 kHz.

    Parameters
    ----------
    dire : str
        Directivity can be either 'Circular Piston', 'Line Piston',
        'Combined Circular/Line', 'Measured Loudspeaker Data' or 'const'.
    alpha : float
        Active radiation factor. Normally 0.82 (Schultz et al.)
    Lambda_y : float
        Height of LSA cabinets.
    beta : 1D-array
        Considered radiation angles.
    omega : 1D-array
        Considered radian frequencies.
    c : float
        Speed of sound.
    f : 1D-array
        Considered frequency bin in Hz.
    dir_meas : ndarray
        Measured loudspeaker directivity.
    dir_meas_deg : ndarray
        Radiation angles of measured loudspeaker directivity.
    n : int
        Index of considered frequency bin in actual loop.

    Returns
    -------
    H_post : ndarray
        Computed loudspeaker directivity.

    """
    if dire == 'Circular Piston':
        R_circ = alpha * Lambda_y / 2
        with np.errstate(invalid='ignore', divide='ignore'):
            H_post = 2 * (sp.jv(1, omega / c * R_circ * np.sin(beta)) / \
                          (omega / c * R_circ * np.sin(beta)))
        H_post[np.isnan(H_post)] = 1
    elif dire == 'Line Piston':
        Lambda_y_line = alpha * Lambda_y
        with np.errstate(invalid='ignore', divide='ignore'):
            H_post = np.sin(omega / (2*c) * Lambda_y_line * np.sin(beta)) / \
                     (omega / (2*c) * Lambda_y_line * np.sin(beta))
        H_post[np.isnan(H_post)] = 1
    elif dire == 'Combined Circular/Line':
        # sampling frequency for filtering
        fs = 40000 
        # cut-off frequency of Linkwitz-Riley-Crossover
        f_c_m = 1500
        # Calculate filter coefficients
        D_xo_low, D_xo_high = LinkwitzRileyCrossOver(f_c_m, fs, f)
        # calculate H_post for circular piston
        R_circ = alpha * Lambda_y / 2
        with np.errstate(invalid='ignore', divide='ignore'):
            H_post_circ = 2 * (sp.jv(1, omega / c * R_circ * np.sin(beta)) / \
                               (omega / c * R_circ * np.sin(beta)))
        H_post_circ[np.isnan(H_post_circ)] = 1
        H_post_circ = H_post_circ * D_xo_low[n]
        # calculate H_post for line piston
        Lambda_y_line = alpha * Lambda_y
        with np.errstate(invalid='ignore', divide='ignore'):
            H_post_line = np.sin(omega / (2*c) * Lambda_y_line * np.sin(beta)) / \
                          (omega / (2*c) * Lambda_y_line * np.sin(beta))
        H_post_line[np.isnan(H_post_line)] = 1
        H_post_line = H_post_line * D_xo_high[n]
        # Combine H_post of circular and line piston
        H_post = H_post_circ + H_post_line
    elif dire == 'Measured Loudspeaker Data':
        H_post = np.interp(beta*180/np.pi, dir_meas_deg, dir_meas)
#        H_post_abs = np.interp(beta*180/np.pi, dir_meas_deg, np.abs(dir_meas))
#        H_post_angle = np.interp(beta*180/np.pi, dir_meas_deg, np.angle(dir_meas))
#        H_post2 = H_post_abs * np.exp(np.complex(0+1j) * H_post_angle)
    else:
        H_post = 1
    
    return H_post
    

def CalcGreenFunctions(x, y, z, x_src_l, y_src_l, alpha, dire, Lambda_y, \
                       gamma_l, c, omega, G_sen, dir_meas, dir_meas_deg, airloss_alpha, f, n):
    """
    Computes the Green's function multiplied with the loudspeaker directivity.
    Calls :any:`greens_fct` and :any:`calc_directivity`.

    Parameters
    ----------
    x : 1D-array
        Receiver positions on x-axis.
    y : 1D-array
        Receiver positions on y-axis.
    z : 1D-array
        Receiver positions on z-axis.
    x_src_l : 1D-array
        Source positions on x-axis.
    y_src_l : 1D-array
        Source positions on y-axis.
    alpha : float
        Active radiation factor. Normally 0.82 (Schultz et al.)
    dire : str
        Directivity can be either 'Circular Piston', 'Line Piston',
        'Combined Circular/Line', 'Measured Loudspeaker Data' or 'const'.
    Lambda_y : float
        Height of LSA cabinets.
    gamma_l : 1D-array
        Set of LSA tilt angles in radian.
    c : float
        Speed of sound.
    omega : float
        Radian frequency.
    G_sen : float
        Additional factor of the Green's function.
    dir_meas : ndarray
        Measured loudspeaker directivity.
    dir_meas_deg : ndarray
        Radiation angles of the measured loudspeaker directivity.
    airloss_alpha : 1D-array
        Computed air attenuation / absorption.
    f : float
        Considered frequency in Hz.
    n : int
        Index of considered frequency in actual loop.

    Returns
    -------
    G : ndarray
        Green's function multiplied with loudspeaker directivity.

    """
    
    G = greens_fct(repmat(x_src_l, np.shape(x)[0],1), repmat(y_src_l, np.shape(y)[0],1), omega, c, \
                   np.transpose(repmat(x, np.shape(x_src_l)[0], 1)), np.transpose(repmat(y, np.shape(y_src_l)[0], 1)), z)

    G = G_sen * G
    
    beta = np.arcsin((np.transpose(repmat(y, np.shape(y_src_l)[0], 1)) - repmat(y_src_l, np.shape(y)[0], 1)) \
                     * np.sqrt((np.transpose(repmat(x, np.shape(x_src_l)[0], 1)) - \
                                repmat(x_src_l, np.shape(x)[0], 1))**2 + \
                        (np.transpose(repmat(y, np.shape(y_src_l)[0], 1)) - repmat(y_src_l, np.shape(y)[0], 1))**2)**(-1)) \
                        + repmat(gamma_l, np.shape(x)[0], 1)
                        
    # air attenuation
    src_rec_dist = np.sqrt((np.transpose(repmat(x, np.shape(x_src_l)[0], 1)) - repmat(x_src_l, np.shape(x)[0], 1))**2 \
                           + (np.transpose(repmat(y, np.shape(y_src_l)[0], 1)) - repmat(y_src_l, np.shape(y)[0], 1))**2)
    
    air_att = airloss_alpha * src_rec_dist
    air_att = 10**(-air_att / 20)
    G = G * air_att
 
    H_post = calc_directivity(dire, alpha, Lambda_y, beta, omega, c, f, dir_meas, dir_meas_deg, n)

    G = G * H_post
    
    return G

 
def calcSFP(gamma_tilt_deg, created_pal, SFP_config, Tech_res):
    """
    Main function to compute 2D sound field prediction with a 3D transfer
    function. Calls :any:`AirAbsorptionCoefficient` and :any:`CalcGreenFunctions`.
    Called by :any:`start_calc` and :any:`optimize_PALC`.

    Parameters
    ----------
    gamma_tilt_deg : list
        Set of LSA tilt angles in degree.
    created_pal : obj [in]
        Contains the venue slice that shall simulated by the CDPS-model.
    SFP_config : obj [in]
        Configuration of the sound field prediction.
    Tech_res : obj [out]
        Stores the results of the sound field prediction.

    Returns
    -------
    x_S : float
        Center position of the LSA on x-axis.
    y_S : float
        Center position of the LSA on y-axis.

    """
    # general
    gamma_n = gamma_tilt_deg / 180 * np.pi
    N = SFP_config.N
    
    ########################## SIMULATION SETUP ###############################
    # reference pressure
    p0 = 2 * 10**(-5)
    # frequencies
    f = get_freq_vec(N_freq=120, step_freq=1/12, freq_range=[20,20000])
    f_xy = np.array([100, 200, 400, 800, 1000, 2000, 5000, 10000, 16000])

    # initialize variables
    omega = 2 * np.pi * f
    omega_xy = 2 * np.pi * f_xy
    D_opt_LSA = np.ones([N, np.shape(f)[0]])
    P_LSA = np.zeros([np.shape(created_pal.xline)[0],np.shape(f)[0]], dtype=complex)

    # air attenuation
    alpha, c = AirAbsorptionCoefficient(f, T=293.15, p=101.325*10**(3), h=50)

    # directivity
    # if PALC_config.directivity not in ['Measured Loudspeaker Data']:
    #     dire_meas_LSA = np.ones([np.shape(f)[0],np.shape(f)[0]])
    #     dire_meas_deg = np.ones([np.shape(f)[0],np.shape(f)[0]])
        
    ######################### SPL CALCULATION #################################
    x_start, y_start, x_stop, y_stop, x_c_n, y_c_n, x_S, y_S = source_pos(gamma_n, SFP_config)

    for n in range(np.shape(f)[0]):
        G_LSA_vert = CalcGreenFunctions(created_pal.xline, created_pal.yline, np.array([0]), \
                                        x_c_n, y_c_n, 0.82, SFP_config.directivity, \
                                        SFP_config.Lambda_y, gamma_n, c, omega[n], 1, \
                                        np.array(SFP_config.dir_meas[:,n]), \
                                        np.array(SFP_config.dir_meas_deg[:,1]), \
                                        alpha[n], f, n )

        P_LSA[:,n] = G_LSA_vert @ D_opt_LSA[:,n] # D_opt_LSA possibility to include driving functions
    p_SPL = 20 * np.log10(np.abs(P_LSA) / p0)
    Tech_res.update_tech_meas(p_SPL=p_SPL, f=f)
    return x_S, y_S


def calcBeamplot(PALC_config, gamma_n, plt_ranges, f, dire_meas_LSA, dire_meas_deg):
    """
    Calcuates the SPL values to map them on a mesh. This is called beamplot.
    Calls :any:`AirAbsorptionCoefficient` and :any:`CalcGreenFunctions`. Called
    by :any:`getBeamplotRes`. The maximum amount of points in the mesh is set
    to 1000. If the app is run locally, the amount of points may be increased.

    Parameters
    ----------
    PALC_config : obj [in]
        Configuration of the PALC algorithm.
    gamma_n : either list or 1D-array
        Set of LSA tilt angles in radian.
    plt_ranges : obj [out]
        Handles the ranges of bokeh figures.
    f : float
        Frequency to be considered in beamplot.
    dir_meas_LSA : ndarray
        Measured loudspeaker directivity.
    dir_meas_deg : ndarray
        Radiation angles of the measured loudspeaker directivity.

    Returns
    -------
    p_SPL : 2D-array
        Sound pressure level values to be mapped on mesh (X, Y).
    X : 2D-array
        Mesh of x-axis values.
    Y : 2D-array
        Mesh of y-axis values

    """
    # general
    N = PALC_config.N
   
########################## SIMULATION SETUP ###############################
    # maximum of discrete mapping points:
    max_points = 10000
    # mesh
    x_range = plt_ranges.p_x
    y_range = plt_ranges.p_y
    pts_x = max_points / (y_range[1]-y_range[0])
    pts_y = max_points / (x_range[1]-x_range[0])

    x = np.linspace(x_range[0], x_range[1], num=int(pts_x+1))
    y = np.linspace(y_range[0], y_range[1], num=int(pts_y+1))

    X, Y = np.meshgrid(x,y)
    # get vertically array
    x_vert = np.reshape(X, np.size(X))
    y_vert = np.reshape(Y, np.size(Y))
    z_vert = np.array([0])

    # reference pressure
    p0 = 2 * 10**(-5)
    # considered frequency
    omega = 2 * np.pi * f
    # initialize driving fct. and output array
    D_opt_LSA = np.ones([N, 1])
    P_LSA = np.zeros([np.shape(x_vert)[0],1], dtype=complex)    

    # air attenuation
    T = 293.15
    p = 101.325 * 10**(3)
    h = 50
    alpha, c = AirAbsorptionCoefficient(f, T, p, h)

        # directivity
    if PALC_config.directivity not in ['Measured Loudspeaker Data']:
        dire_meas_LSA = np.ones([1,1])
        dire_meas_deg = np.ones([1,1])
    
    ######################### SPL CALCULATION #################################
    x_start, y_start, x_stop, y_stop, x_c_n, y_c_n, x_S, y_S = source_pos(gamma_n, PALC_config)

    G_LSA_vert = CalcGreenFunctions(x_vert, y_vert, z_vert, x_c_n, y_c_n, 0.82,\
                                    PALC_config.directivity, PALC_config.Lambda_y, \
                                    gamma_n, c, omega, 1, dire_meas_LSA, \
                                    dire_meas_deg, alpha, f, 0 )

    P_LSA[:,0] = G_LSA_vert @ D_opt_LSA[:,0]

    p_SPL = 20 * np.log10(np.abs(P_LSA) / p0)
    
    p_SPL = np.reshape(p_SPL, np.shape(X))

    return p_SPL, X, Y


def getBeamplotRes(pBeam, mapper, PALC_config, SFP_config, gamma_n, plt_ranges, f, ind_f):
    """
    Computes the beamplot of a certain frequency regarding the simulated
    venue slice. Calls :any:`calcBeamplot`. Called by :any:`get_beamplot`.

    Parameters
    ----------
    pBeam : fig [out]
        Bokeh beamplot figure.
    mapper : LinearColorMapper [out]
        Object that maps the Colors of the beamplot. Also used for the Colorbar.
    PALC_config : obj [in]
        Configuration of PALC algorithm.
    SFP_config : obj [in]
        Configuration of sound field prediction.
    gamma_n : 1D-array
        Set of LSA tilt angles in radian.
    plt_ranges : obj [out]
        Contains ranging information of the bokeh figures.
    f : float
        Considered frequency bin.
    ind_f : int
        Index of the frequency bin.

    Returns
    -------
    p_SPL : ndarray
        Sound pressure levels mapped on the mesh.

    """
    # Calculation
    p_SPL, X, Y = calcBeamplot(PALC_config, gamma_n, plt_ranges, f, \
                               np.array(SFP_config.dir_meas[:,ind_f]), \
                               np.array(SFP_config.dir_meas_deg[:,1]))
    high_ind = np.argwhere(p_SPL > 100)
    low_ind = np.argwhere(p_SPL < 50)
    for n in range(np.shape(high_ind)[0]):
        p_SPL[high_ind[n,0], high_ind[n,1]] = 100
    for n in range(np.shape(low_ind)[0]):
        p_SPL[low_ind[n,0], low_ind[n,1]] = 50
    mapper.low = np.amin(p_SPL)
    mapper.high = np.amax(p_SPL)   
    return p_SPL
    

def calcHomogeneity(Tech_res, x_S, y_S, x_vert, y_vert, objective = '0dB'):
    """
    Computes the homogeneity of a given sound field. Computes also the values
    to be visualized in the bar chart (Number of SPL values per audience
    positions). Here the output of :any:`calcSFP`. Called by :any:`start_calc`.

    Parameters
    ----------
    Tech_res : obj [in, out]
        Contains technical measure results. Input: sound field prediction.
        Output: Homogeneity.
    x_S : float
        Center position of the LSA on x-axis.
    y_S : float
        Center position of the LSA on y-axis.
    x_vert : list
        Audience positions on x-axis.
    y_vert : list
        Audience positions on y-axis.
    objective : str, optional
        Normalization on '0dB', '3dB' or '6dB' loss over distance doubling.
        The default is '0dB'.

    Returns
    -------
    None.

    """
    p0 = 2*10**(-5) # atmospheric pressure
    # optional: Optimization on 3dB loss over distance doubling
    xy_dist = np.sqrt((x_vert-x_S)**2 + (y_vert-y_S)**2)
    if objective == '6dB':
        for n in range(np.shape(Tech_res.p_SPL)[0]):
            n_dist = np.sqrt((x_vert[n]-x_S)**2 + (y_vert[n]-y_S)**2)
            Tech_res.p_SPL[n,:] = 20*np.log10(p0*10**(Tech_res.p_SPL[n,:] / 20) \
                                              * (n_dist) / np.amin(xy_dist))
    elif objective == '3dB':
        for n in range(np.shape(Tech_res.p_SPL)[0]):
            n_dist = np.sqrt((x_vert[n]-x_S)**2 + (y_vert[n]-y_S)**2)
            Tech_res.p_SPL[n,:] = 20*np.log10(p0*10**(Tech_res.p_SPL[n,:] / 20) \
                                              * np.sqrt((n_dist) / np.amin(xy_dist)))
    
    # Calculate the quantiles
    H = np.zeros([2, np.shape(Tech_res.p_SPL)[1]])
    for n in range(np.shape(Tech_res.p_SPL)[1]):
        H[:,n] = mquantiles(Tech_res.p_SPL[:,n], [0.1, 0.9], alphap=0.5, betap=0.5)
    H = H[1,:] - H[0,:]
    H_dist_high = 20 * np.log10(np.abs(np.amax(xy_dist)/np.amin(xy_dist)))
    H_dist_high = np.linspace(H_dist_high, H_dist_high, np.shape(H)[0])
    H_dist      = [H_dist_high / 2, H_dist_high]
    # Calculate values for bar chart
    A_mean      = 20 * np.log10(np.sqrt(np.mean((p0 * 10**(Tech_res.p_SPL/20))**2,axis=1)) / p0)
    min_val     = np.floor(np.amin(A_mean))
    max_val     = np.ceil(np.amax(A_mean))
    if max_val <= min_val:
        max_val = min_val + 5
    bins_shape  = int(max_val - min_val + 1)
    hist_bins   = np.linspace(min_val, max_val, bins_shape)
    Tech_res.update_tech_meas(H=H, H_dist=H_dist, A_mean=A_mean, hist_bins=hist_bins)
    return


def calcHistogram(Tech_res):
    """
    Computes the Histogram values to be visualized in bokeh. Called by
    :any:`start_calc`.

    Parameters
    ----------
    Tech_res : obj [in, out]
        Technical Measure Results.

    Returns
    -------
    None.

    """
    A_Hist = np.histogram(Tech_res.A_mean, Tech_res.hist_bins)
    top, bin_list, x_steps = list(A_Hist[0]), list(A_Hist[1]), []
    for n in range(np.shape(bin_list)[0]-1):
        x_steps.append((bin_list[n+1] + bin_list[n]) / 2)
    Tech_res.update_tech_meas(Hist_tops=top, Hist_steps=x_steps)
    return


def calcSPLoverX(PALC_plots, created_pal, p_SPL, SPLoverX, freq_range):
    """
    Computes the SPL over distance. The distance is measured from the center
    of the LSA to the receiver positions. Called by :any:`start_calc` and
    :any:`optimize_PALC`.

    Parameters
    ----------
    PALC_plots : obj [in]
        Contains data to compute the center of the LSA.
    created_pal : obj [in]
        Contains data of the receiver positions.
    p_SPL : 1D-array
        Sound pressure level at each receiver position.
    SPLoverX : obj [out]
        Contains the SPL over distance output.
    freq_range: tuple or list [in]
        Frequency range to be considered, contains (start,stop) in Hz.

    Returns
    -------
    p_SPL_x_vert : 1D-array
        SPL value of the receiver positions.
    total_dist : 1D-array
        Distances between center of LSA and receiver positions.

    """
    p0 = 2*10**(-5) # atmospheric pressure
    f  = get_freq_vec(120, 1/12, (20,20000))
    f_ind_min = np.argmin(np.abs(f-freq_range[0]))
    f_ind_max = np.argmin(np.abs(f-freq_range[1]))
    # distance source --> receiver
    total_dist = np.sqrt(((PALC_plots.x_start[0]+PALC_plots.x_stop[-1])/2 - created_pal.xline)**2 + \
                         ((PALC_plots.y_start[0]+PALC_plots.y_stop[-1])/2 - created_pal.yline)**2)
    # SPLoverX computation
    p_sum = np.sum(p0 * 10**(p_SPL[:,f_ind_min:f_ind_max+1]/20), axis=1)
    p_SPL_x_vert = 20 * np.log10(np.abs(p_sum) / p0)
    p_SPL_x_vert = p_SPL_x_vert - max(p_SPL_x_vert)
    SPLoverX.x   = total_dist
    SPLoverX.SPL = p_SPL_x_vert
    return p_SPL_x_vert, total_dist


def calcAcousticContrast(p_SPL, p_SPL_na, x_a, y_a, x_na, y_na, p0):
    """
    Computes the acoustic Contrast. In general it shows the relation between
    radiated acoustic energy on receiver positions and non-receiver positions.
    Not visualized in bokeh, only for manual use without bokeh.

    Parameters
    ----------
    p_SPL : 2D-array
        Sound pressure level at receiver positions. First axis: receiver
        positions. Second axis: Frequency
    p_SPL_na : 2D-array
        Sound pressure level at non-receiver positions. First axis: receiver
        positions. Second axis: Frequency
    x_a : 1D-array
        Receiver positions on x-axis.
    y_a : 1D-array
        Receiver positions on y-axis.
    x_na : 1D-array
        Non-receiver positions on x-axis.
    y_na : 1D-array
        Non-receiver positions on y-axis.
    p0 : float
        Reference sound pressure.

    Returns
    -------
    L_p_a_na : 1D-array
        Acoustic contrast of all considered frequencies.

    """
    # convert to sound pressure in Pascal
    p = p0 * 10**(p_SPL/20)
    p_na = p0 * 10**(p_SPL_na/20)
    # Compute Acoustic Contrast
    L_p_a_na = np.zeros(np.shape(p_SPL)[1])
    p_mean = np.mean(np.abs(p)**2,axis=0)
    p_na_mean = np.mean(np.abs(p_na)**2,axis=0)
    L_p_a_na = 10 * np.log10(p_mean/p_na_mean)    
#    for n in range(np.shape(p_SPL)[1]):
#        L_p_a_na[n] = 10 * np.log10( (1/np.shape(p_SPL)[0] * np.sqrt(np.sum(np.abs(p[:,n])**2, axis=0))**2) / \
#                (1/np.shape(p_SPL_na)[0] * np.sqrt(np.sum(np.abs(p_na[:,n])**2, axis=0))**2))
    #L_p_a_na = L_p_a_na - np.amin(L_p_a_na)
        
    return L_p_a_na

  
def smoothSpectrum(f, X_f, r_oct):
    """
    Smoothes a frequency spectrum. Not used in bokeh.

    Parameters
    ----------
    f : 1D-array
        Frequency bins.
    X_f : 1D-array
        Spectrum to be smoothed.
    r_oct : float
        Resolution to smooth.

    Returns
    -------
    X_f_out : 1D-array
        Smoothed spectrum.

    """
    X_f_out = np.zeros(np.shape(X_f))
    for n in range(np.shape(f)[0]):
        # standard deviation
        sigma = f[n] / r_oct / np.pi
        # Gaussian window with the center frequnecy f[n] an dstandard deviation
        w = np.exp( -(f-f[n])**2 / (2*sigma**2) )
        w = w / np.sum(w, axis=0)
        X_f_out[n] = np.sum(w * X_f)
    
    return X_f_out
    
   
## Function only to initialize the Circular Piston Directivity Plot
    
def init_dir_plot():
    """
    Initializes the directivity plot in bokeh :any:`pdir`. Avoids error message
    that no data is presend to plot. Calls :any:`get_freq_vec`. Called by
    :py:doc:`main`.

    Returns
    -------
    c0 : float
        Speed of sound.
    alpha : float
        Active radiating factor. Set to 0.82 (Schult et al.)
    beta_deg : 1D-array
        Radiation angles in degree.
    beta : 1D-array
        Radiation angles in radian.
    f : 1D-array
        Frequency bins in Hz.
    omega : 1D-array
        Frequency bins in radian.
    dir_meas : 2D-array
        Measured loudspeaker data.
    dir_meas_deg : 2D-array
        Angle of measured loudspeaker data.
    H_post : 2D-array
        Directivity.

    """
    c0  = 331.45             # reference speed of sound in m/s
    alpha = 0.82
    beta_deg = np.linspace(-180,180,181)
    beta = beta_deg * np.pi / 180
    # frequencies
    f = get_freq_vec(N_freq=120, step_freq=1/12, freq_range=[20,20000])
    omega = 2 * np.pi * f
    dir_meas = np.ones([np.shape(f)[0],np.shape(f)[0]])
    dir_meas_deg = np.ones([np.shape(f)[0],np.shape(f)[0]])
    H_post = np.zeros([np.shape(beta)[0],np.shape(f)[0]], dtype=complex)
    return c0, alpha, beta_deg, beta, f, omega, dir_meas, dir_meas_deg, H_post
    
  
    