"""Sound field prediction with the CDPS model.

Configuration and result classes of the sound field prediction, the
Complex Directivity Point Source (CDPS) model, loudspeaker directivities and
the parser of measured directivity data (formerly ``fun/sfp_functions.py``,
``SoundFieldConfig`` and ``ArrayEvaluation`` from
``fun/PALC_classes.py`` and ``read_dir_data`` from ``fun/gui_ls_array.py``).

author: Arne Hoelter
"""
from __future__ import annotations

import logging

import numpy as np
import scipy.signal as sig
import scipy.special as sp
from scipy.stats.mstats import mquantiles

from .geometry import repmat, source_positions
from .palc import PalcConfig

logger = logging.getLogger(__name__)


class SoundFieldConfig(PalcConfig):
    """ Child class of PalcConfig, contains data that are additionally
        needed for the sound field prediction. """
    def __init__(self, directivity='Circular Piston', cabinet_count=1, cabinet_gap=0, \
                 cabinet_height=0.2, top_x=0, top_y=2, measured_directivity_data=None, \
                 measured_angles_deg=None, \
                 measured_amplitude_db=None, tilt_angles=None, frequencies=None, plot_directivity_db=None, \
                 plot_measured_directivity_db=None, plot_measured_angles_deg=None, \
                 frequency_range_hz=(200,8000)):
        # None stands for a new object with the default value (no shared mutable defaults)
        measured_directivity_data = np.zeros([2,120]) if measured_directivity_data is None else measured_directivity_data
        measured_angles_deg = np.zeros([2,120]) if measured_angles_deg is None else measured_angles_deg
        measured_amplitude_db, frequencies, plot_directivity_db = [[] if val is None else val for val in
                                         (measured_amplitude_db, frequencies, plot_directivity_db)]
        plot_measured_directivity_db = [[0],[0],[0]] if plot_measured_directivity_db is None else plot_measured_directivity_db
        plot_measured_angles_deg = [0] if plot_measured_angles_deg is None else plot_measured_angles_deg
        self.measured_directivity_data, self.frequencies                     = measured_directivity_data, frequencies 
        self.measured_angles_deg, self.measured_amplitude_db      = measured_angles_deg, measured_amplitude_db
        self.plot_directivity_db, self.plot_measured_directivity_db = plot_directivity_db, plot_measured_directivity_db
        self.plot_measured_angles_deg                       = plot_measured_angles_deg
        self.frequency_range_hz                           = frequency_range_hz
        super(SoundFieldConfig, self).__init__()
        
    def set_plot_directivity(self, H_post, plot_frequencies, measured=False):
        """
        Calculates the amplitude of the directivity to plot in :any:`pdir`.

        Parameters
        ----------
        H_post : ndarray
            (Complex) directivity data.
        plot_frequencies : int
            frequency to plot.
        measured : bool, optional
            Must be True if measured loudspeaker data are used. The default
            is False.

        Returns
        -------
        None.

        """
        # get index
        self.plot_directivity_db, ind_f = [], []
        if measured: self.plot_measured_directivity_db = []
        # get amplitude values
        amplitude = 20*np.log10(np.abs(H_post))
        # get amplitudes for frequencies to plot
        for ind, n in enumerate(plot_frequencies):
            ind_f.append((np.abs(self.frequencies-n)).argmin())
            if measured:
                self.plot_measured_directivity_db.append(list(amplitude[:,ind_f[ind]].transpose()))
            else:
                self.plot_directivity_db.append(list(amplitude[:,ind_f[ind]].transpose()))


class ArrayEvaluation():
    """ Class to create objects that contain the results of the technical
        measures based on the sound field prediction. """
    def __init__(self):
        self.frequencies, self.spl_db                      = [], []
        self.homogeneity_db                                = []
        self.homogeneity_low_db, self.homogeneity_high_db  = [], []
        self.mean_spl_db, self.histogram_bins              = [], []
        self.histogram_tops, self.histogram_steps          = [], []
        self.beam_spl_db                                   = []
        
        #for plotting
        self.plot_frequencies, self.plot_spl_db = [], []
        
    
    def update_evaluation(self, **new_input):
        """
        Updates the technical measures. 

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
        if 'frequencies' in new_input.keys() and 'spl_db' in new_input.keys():
            self.plot_spl_db, self.plot_frequencies = [], []
            for n in range(np.shape(new_input['spl_db'])[0]):
                self.plot_spl_db.append(list(self.spl_db[n,:]))
                self.plot_frequencies.append(list(self.frequencies))


def frequency_vector(frequency_count, octave_step, frequency_range_hz):
    """
    Compute a linear frequency vector. Called by :any:`predict_sound_field` and
    :any:`initial_directivity_plot`.

    Parameters
    ----------
    frequency_count : int
        Number of frequency bins.
    octave_step : int
        Step size of frequency vector.
    frequency_range_hz: tuple or list
        Contains start and stop frequency of frequncy vector

    Returns
    -------
    frequencies : 1D-array
        Frequency vector.

    """
    frequencies = 20 * (2 * np.ones(frequency_count+1))**(octave_step * np.arange(frequency_count+1))
    frequencies = frequencies[frequencies <= frequency_range_hz[1]]
    frequencies = frequencies[frequencies >= frequency_range_hz[0]]
    frequencies = np.unique(frequencies)
    return frequencies


def greens_function(x0, y0, omega, c, x, y, z):
    """
    Computes the 3D Green's function. The acoustic transfer function in air.
    Called by :any:`green_functions`.

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


def air_absorption_relaxation_frequencies(T,p,humidity_molar,T0, p_r):
    """
    Calculates the relaxation frequencies for air absorption conforming to
    ISO 9613-1. Called by :any:`air_absorption_coefficient`.

    Parameters
    ----------
    T : float
        Temperature in K.
    p : float
        Pressure in Pa.
    humidity_molar : float
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
    
    f_rO = p / p_r * (24 + 4.04 * 10**4 * humidity_molar * (0.02+humidity_molar) / (0.391+humidity_molar))
    
    f_rN = p / p_r * (T/T0)**(-0.5) * (9+280*humidity_molar*np.exp(-4.17*((T/T0)**(-1/3)-1)))
    
    return f_rO, f_rN


def air_absorption_coefficient(frequency, T, p, h):
    """
    Calculates the air absorption coefficient alpha. Called by
    :any:`predict_sound_field` and :any:`beam_plot_spl`.
    Calls :any:`air_absorption_relaxation_frequencies`. Uses the following
    constants:
        
        * p_r        = 101.325*10³ Pa as reference sound pressure
        * T0         = 293.15 K       as reference temperature in K for
                                         relaxation frequencies and alpha
        * T0_p_sat_r = 273.16 K       as reference temperature in K for p_sat_r
        * c0         = 331.45 m/s     as reference speed of sound

    Parameters
    ----------
    frequency : float
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
    humidity_molar = h * p_sat_r * p_r / p
    
    f_rO, f_rN = air_absorption_relaxation_frequencies(T,p,humidity_molar,T0, p_r)
    
    alpha = 8.686 * frequency**2 * ((1.84*(10**-11) * (p/p_r)**(-1) * (T/T0)**(0.5)) \
            + (T/T0)**(-5/2) * (0.01275 * np.exp(-2239.1/T) / (f_rO + frequency**2 / f_rO) + \
               0.1068 * np.exp(-3352/T) / (f_rN + frequency**2 / f_rN)))
    
    c = c0 * np.sqrt(29 * T / (1.4*(T0-20))) * np.sqrt(((700+humidity_molar) / (500+humidity_molar)) / (29-0.11*humidity_molar))

    return alpha, c


def linkwitz_riley_crossover(f_c_m, fs, frequencies):
    """
    Computes a Linkwitz-Riley-Crossover filter. The lowpass and highpass are
    of second order butterworth filter. Called by :any:`cabinet_directivity`.

    Parameters
    ----------
    f_c_m : float
        Cut-off frequency.
    fs : float
        Sampling frequency.
    frequencies : list or 1D-array
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
    f_xo_low, D_xo_low = sig.freqz(b_xo_low, a_xo_low, worN=frequencies, fs=fs)
    D_xo_low = D_xo_low * D_xo_low
    
    # Highpass
    # highpass coefficients
    b_xo_high, a_xo_high = sig.butter(2, f_c_m/(fs/2), btype='highpass')
    # Complex frequency response of the second order Butterworth filter
    f_xo_low, D_xo_high = sig.freqz(b_xo_high, a_xo_high, worN=frequencies, fs=fs)
    D_xo_high = D_xo_high * D_xo_high
    
    return D_xo_low, D_xo_high


def cabinet_directivity(directivity, alpha, cabinet_height, beta, omega, c, frequencies, measured_directivity_data, measured_angles_deg, n):
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
    cabinet_height : float
        Height of LSA cabinets.
    beta : 1D-array
        Considered radiation angles.
    omega : 1D-array
        Considered radian frequencies.
    c : float
        Speed of sound.
    frequencies : 1D-array
        Considered frequency bin in Hz.
    measured_directivity_data : ndarray
        Measured loudspeaker directivity.
    measured_angles_deg : ndarray
        Radiation angles of measured loudspeaker directivity.
    n : int
        Index of considered frequency bin in actual loop.

    Returns
    -------
    H_post : ndarray
        Computed loudspeaker directivity.

    """
    if directivity == 'Circular Piston':
        R_circ = alpha * cabinet_height / 2
        with np.errstate(invalid='ignore', divide='ignore'):
            H_post = 2 * (sp.jv(1, omega / c * R_circ * np.sin(beta)) / \
                          (omega / c * R_circ * np.sin(beta)))
        H_post[np.isnan(H_post)] = 1
    elif directivity == 'Line Piston':
        Lambda_y_line = alpha * cabinet_height
        with np.errstate(invalid='ignore', divide='ignore'):
            H_post = np.sin(omega / (2*c) * Lambda_y_line * np.sin(beta)) / \
                     (omega / (2*c) * Lambda_y_line * np.sin(beta))
        H_post[np.isnan(H_post)] = 1
    elif directivity == 'Combined Circular/Line':
        # sampling frequency for filtering
        fs = 40000 
        # cut-off frequency of Linkwitz-Riley-Crossover
        f_c_m = 1500
        # Calculate filter coefficients
        D_xo_low, D_xo_high = linkwitz_riley_crossover(f_c_m, fs, frequencies)
        # calculate H_post for circular piston
        R_circ = alpha * cabinet_height / 2
        with np.errstate(invalid='ignore', divide='ignore'):
            H_post_circ = 2 * (sp.jv(1, omega / c * R_circ * np.sin(beta)) / \
                               (omega / c * R_circ * np.sin(beta)))
        H_post_circ[np.isnan(H_post_circ)] = 1
        H_post_circ = H_post_circ * D_xo_low[n]
        # calculate H_post for line piston
        Lambda_y_line = alpha * cabinet_height
        with np.errstate(invalid='ignore', divide='ignore'):
            H_post_line = np.sin(omega / (2*c) * Lambda_y_line * np.sin(beta)) / \
                          (omega / (2*c) * Lambda_y_line * np.sin(beta))
        H_post_line[np.isnan(H_post_line)] = 1
        H_post_line = H_post_line * D_xo_high[n]
        # Combine H_post of circular and line piston
        H_post = H_post_circ + H_post_line
    elif directivity == 'Measured Loudspeaker Data':
        H_post = np.interp(beta*180/np.pi, measured_angles_deg, measured_directivity_data)
#        H_post_abs = np.interp(beta*180/np.pi, measured_angles_deg, np.abs(measured_directivity_data))
#        H_post_angle = np.interp(beta*180/np.pi, measured_angles_deg, np.angle(measured_directivity_data))
#        H_post2 = H_post_abs * np.exp(complex(0+1j) * H_post_angle)
    else:
        H_post = 1
    
    return H_post


def green_functions(x, y, z, x_src_l, y_src_l, alpha, directivity, cabinet_height, \
                       gamma_l, c, omega, G_sen, measured_directivity_data, measured_angles_deg, airloss_alpha, frequencies, n):
    """
    Computes the Green's function multiplied with the loudspeaker directivity.
    Calls :any:`greens_function` and :any:`cabinet_directivity`.

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
    cabinet_height : float
        Height of LSA cabinets.
    gamma_l : 1D-array
        Set of LSA tilt angles in radian.
    c : float
        Speed of sound.
    omega : float
        Radian frequency.
    G_sen : float
        Additional factor of the Green's function.
    measured_directivity_data : ndarray
        Measured loudspeaker directivity.
    measured_angles_deg : ndarray
        Radiation angles of the measured loudspeaker directivity.
    airloss_alpha : 1D-array
        Computed air attenuation / absorption.
    frequencies : float
        Considered frequency in Hz.
    n : int
        Index of considered frequency in actual loop.

    Returns
    -------
    G : ndarray
        Green's function multiplied with loudspeaker directivity.

    """
    
    G = greens_function(repmat(x_src_l, np.shape(x)[0],1), repmat(y_src_l, np.shape(y)[0],1), omega, c, \
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
 
    H_post = cabinet_directivity(directivity, alpha, cabinet_height, beta, omega, c, frequencies, measured_directivity_data, measured_angles_deg, n)

    G = G * H_post
    
    return G


def predict_sound_field(tilt_angles_deg, drawn_lines, sound_field_config, evaluation):
    """
    Main function to compute 2D sound field prediction with a 3D transfer
    function. Calls :any:`air_absorption_coefficient` and :any:`green_functions`.
    Called by :any:`start_calc` and :any:`optimize_target_slope`.

    Parameters
    ----------
    tilt_angles_deg : list
        Set of LSA tilt angles in degree.
    drawn_lines : obj [in]
        Contains the venue slice that shall simulated by the CDPS-model.
    sound_field_config : obj [in]
        Configuration of the sound field prediction.
    evaluation : obj [out]
        Stores the results of the sound field prediction.

    Returns
    -------
    array_center_x : float
        Center position of the LSA on x-axis.
    array_center_y : float
        Center position of the LSA on y-axis.

    """
    # general
    tilt_angles = tilt_angles_deg / 180 * np.pi
    cabinet_count = sound_field_config.cabinet_count
    
    ########################## SIMULATION SETUP ###############################
    # reference pressure
    p0 = 2 * 10**(-5)
    # frequencies
    frequencies = frequency_vector(frequency_count=120, octave_step=1/12, frequency_range_hz=[20,20000])
    f_xy = np.array([100, 200, 400, 800, 1000, 2000, 5000, 10000, 16000])

    # initialize variables
    omega = 2 * np.pi * frequencies
    omega_xy = 2 * np.pi * f_xy
    driving_functions = np.ones([cabinet_count, np.shape(frequencies)[0]])
    pressure = np.zeros([np.shape(drawn_lines.points_x)[0],np.shape(frequencies)[0]], dtype=complex)

    # air attenuation
    alpha, c = air_absorption_coefficient(frequencies, T=293.15, p=101.325*10**(3), h=50)

    # directivity
    # if config.directivity not in ['Measured Loudspeaker Data']:
    #     measured_directivity_data = np.ones([np.shape(frequencies)[0],np.shape(frequencies)[0]])
    #     measured_angles_deg = np.ones([np.shape(frequencies)[0],np.shape(frequencies)[0]])
        
    ######################### SPL CALCULATION #################################
    x_start, y_start, x_stop, y_stop, cabinet_centers_x, cabinet_centers_y, array_center_x, array_center_y = source_positions(tilt_angles, sound_field_config)

    for n in range(np.shape(frequencies)[0]):
        transfer_functions = green_functions(drawn_lines.points_x, drawn_lines.points_y, np.array([0]), \
                                        cabinet_centers_x, cabinet_centers_y, 0.82, sound_field_config.directivity, \
                                        sound_field_config.cabinet_height, tilt_angles, c, omega[n], 1, \
                                        np.array(sound_field_config.measured_directivity_data[:,n]), \
                                        np.array(sound_field_config.measured_angles_deg[:,1]), \
                                        alpha[n], frequencies, n )

        pressure[:,n] = transfer_functions @ driving_functions[:,n] # driving_functions possibility to include driving functions
    spl_db = 20 * np.log10(np.abs(pressure) / p0)
    evaluation.update_evaluation(spl_db=spl_db, frequencies=frequencies)
    return array_center_x, array_center_y


def beam_plot_spl(config, tilt_angles, ranges, frequency, measured_directivity_data, measured_angles_deg):
    """
    Calcuates the SPL values to map them on a mesh. This is called beamplot.
    Calls :any:`air_absorption_coefficient` and :any:`green_functions`. Called
    by :any:`getBeamplotRes`. The maximum amount of points in the mesh is set
    to 1000. If the app is run locally, the amount of points may be increased.

    Parameters
    ----------
    config : obj [in]
        Configuration of the PALC algorithm.
    tilt_angles : either list or 1D-array
        Set of LSA tilt angles in radian.
    ranges : obj [out]
        Handles the ranges of bokeh figures.
    frequency : float
        Frequency to be considered in beamplot.
    measured_directivity_data : ndarray
        Measured loudspeaker directivity.
    measured_angles_deg : ndarray
        Radiation angles of the measured loudspeaker directivity.

    Returns
    -------
    spl_db : 2D-array
        Sound pressure level values to be mapped on mesh (X, Y).
    X : 2D-array
        Mesh of x-axis values.
    Y : 2D-array
        Mesh of y-axis values

    """
    # general
    cabinet_count = config.cabinet_count
   
########################## SIMULATION SETUP ###############################
    # maximum of discrete mapping points:
    max_points = 10000
    # mesh
    x_range = ranges.p_x
    y_range = ranges.p_y
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
    omega = 2 * np.pi * frequency
    # initialize driving fct. and output array
    driving_functions = np.ones([cabinet_count, 1])
    pressure = np.zeros([np.shape(x_vert)[0],1], dtype=complex)    

    # air attenuation
    T = 293.15
    p = 101.325 * 10**(3)
    h = 50
    alpha, c = air_absorption_coefficient(frequency, T, p, h)

        # directivity
    if config.directivity not in ['Measured Loudspeaker Data']:
        measured_directivity_data = np.ones([1,1])
        measured_angles_deg = np.ones([1,1])
    
    ######################### SPL CALCULATION #################################
    x_start, y_start, x_stop, y_stop, cabinet_centers_x, cabinet_centers_y, array_center_x, array_center_y = source_positions(tilt_angles, config)

    transfer_functions = green_functions(x_vert, y_vert, z_vert, cabinet_centers_x, cabinet_centers_y, 0.82,\
                                    config.directivity, config.cabinet_height, \
                                    tilt_angles, c, omega, 1, measured_directivity_data, \
                                    measured_angles_deg, alpha, frequency, 0 )

    pressure[:,0] = transfer_functions @ driving_functions[:,0]

    spl_db = 20 * np.log10(np.abs(pressure) / p0)
    
    spl_db = np.reshape(spl_db, np.shape(X))

    return spl_db, X, Y


def compute_homogeneity(evaluation, array_center_x, array_center_y, points_x, points_y, objective = '0dB'):
    """
    Computes the homogeneity of a given sound field. Computes also the values
    to be visualized in the bar chart (Number of SPL values per audience
    positions). Here the output of :any:`predict_sound_field`. Called by :any:`start_calc`.

    Parameters
    ----------
    evaluation : obj [in, out]
        Contains technical measure results. Input: sound field prediction.
        Output: Homogeneity.
    array_center_x : float
        Center position of the LSA on x-axis.
    array_center_y : float
        Center position of the LSA on y-axis.
    points_x : list
        Audience positions on x-axis.
    points_y : list
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
    xy_dist = np.sqrt((points_x-array_center_x)**2 + (points_y-array_center_y)**2)
    if objective == '6dB':
        for n in range(np.shape(evaluation.spl_db)[0]):
            n_dist = np.sqrt((points_x[n]-array_center_x)**2 + (points_y[n]-array_center_y)**2)
            evaluation.spl_db[n,:] = 20*np.log10(p0*10**(evaluation.spl_db[n,:] / 20) \
                                              * (n_dist) / np.amin(xy_dist))
    elif objective == '3dB':
        for n in range(np.shape(evaluation.spl_db)[0]):
            n_dist = np.sqrt((points_x[n]-array_center_x)**2 + (points_y[n]-array_center_y)**2)
            evaluation.spl_db[n,:] = 20*np.log10(p0*10**(evaluation.spl_db[n,:] / 20) \
                                              * np.sqrt((n_dist) / np.amin(xy_dist)))
    
    # Calculate the quantiles
    homogeneity_db = np.zeros([2, np.shape(evaluation.spl_db)[1]])
    for n in range(np.shape(evaluation.spl_db)[1]):
        homogeneity_db[:,n] = mquantiles(evaluation.spl_db[:,n], [0.1, 0.9], alphap=0.5, betap=0.5)
    homogeneity_db = homogeneity_db[1,:] - homogeneity_db[0,:]
    homogeneity_high_db = 20 * np.log10(np.abs(np.amax(xy_dist)/np.amin(xy_dist)))
    homogeneity_high_db = np.linspace(homogeneity_high_db, homogeneity_high_db, np.shape(homogeneity_db)[0])
    homogeneity_low_db = homogeneity_high_db / 2
    # Calculate values for bar chart
    mean_spl_db      = 20 * np.log10(np.sqrt(np.mean((p0 * 10**(evaluation.spl_db/20))**2,axis=1)) / p0)
    min_val     = np.floor(np.amin(mean_spl_db))
    max_val     = np.ceil(np.amax(mean_spl_db))
    if max_val <= min_val:
        max_val = min_val + 5
    bins_shape  = int(max_val - min_val + 1)
    histogram_bins   = np.linspace(min_val, max_val, bins_shape)
    evaluation.update_evaluation(homogeneity_db=homogeneity_db,
                                 homogeneity_low_db=homogeneity_low_db,
                                 homogeneity_high_db=homogeneity_high_db,
                                 mean_spl_db=mean_spl_db,
                                 histogram_bins=histogram_bins)
    return


def compute_histogram(evaluation):
    """
    Computes the Histogram values to be visualized in bokeh. Called by
    :any:`start_calc`.

    Parameters
    ----------
    evaluation : obj [in, out]
        Technical Measure Results.

    Returns
    -------
    None.

    """
    A_Hist = np.histogram(evaluation.mean_spl_db, evaluation.histogram_bins)
    top, bin_list, x_steps = list(A_Hist[0]), list(A_Hist[1]), []
    for n in range(np.shape(bin_list)[0]-1):
        x_steps.append((bin_list[n+1] + bin_list[n]) / 2)
    evaluation.update_evaluation(histogram_tops=top, histogram_steps=x_steps)
    return


def compute_spl_over_distance(ray_plot, drawn_lines, spl_db, spl_over_distance, frequency_range_hz):
    """
    Computes the SPL over distance. The distance is measured from the center
    of the LSA to the receiver positions. Called by :any:`start_calc` and
    :any:`optimize_target_slope`.

    Parameters
    ----------
    ray_plot : obj [in]
        Contains data to compute the center of the LSA.
    drawn_lines : obj [in]
        Contains data of the receiver positions.
    spl_db : 1D-array
        Sound pressure level at each receiver position.
    spl_over_distance : obj [out]
        Contains the SPL over distance output.
    frequency_range_hz: tuple or list [in]
        Frequency range to be considered, contains (start,stop) in Hz.

    Returns
    -------
    levels_db : 1D-array
        SPL value of the receiver positions.
    distances : 1D-array
        Distances between center of LSA and receiver positions.

    """
    p0 = 2*10**(-5) # atmospheric pressure
    frequencies  = frequency_vector(120, 1/12, (20,20000))
    f_ind_min = np.argmin(np.abs(frequencies-frequency_range_hz[0]))
    f_ind_max = np.argmin(np.abs(frequencies-frequency_range_hz[1]))
    # distance source --> receiver
    distances = np.sqrt(((ray_plot.x_start[0]+ray_plot.x_stop[-1])/2 - drawn_lines.points_x)**2 + \
                         ((ray_plot.y_start[0]+ray_plot.y_stop[-1])/2 - drawn_lines.points_y)**2)
    # spl_over_distance computation
    p_sum = np.sum(p0 * 10**(spl_db[:,f_ind_min:f_ind_max+1]/20), axis=1)
    levels_db = 20 * np.log10(np.abs(p_sum) / p0)
    levels_db = levels_db - max(levels_db)
    spl_over_distance.distances   = distances
    spl_over_distance.levels_db = levels_db
    return levels_db, distances


def acoustic_contrast(spl_db, non_audience_spl_db, x_a, y_a, x_na, y_na, p0):
    """
    Computes the acoustic Contrast. In general it shows the relation between
    radiated acoustic energy on receiver positions and non-receiver positions.
    Not visualized in bokeh, only for manual use without bokeh.

    Parameters
    ----------
    spl_db : 2D-array
        Sound pressure level at receiver positions. First axis: receiver
        positions. Second axis: Frequency
    non_audience_spl_db : 2D-array
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
    p = p0 * 10**(spl_db/20)
    p_na = p0 * 10**(non_audience_spl_db/20)
    # Compute Acoustic Contrast
    L_p_a_na = np.zeros(np.shape(spl_db)[1])
    p_mean = np.mean(np.abs(p)**2,axis=0)
    p_na_mean = np.mean(np.abs(p_na)**2,axis=0)
    L_p_a_na = 10 * np.log10(p_mean/p_na_mean)    
#    for n in range(np.shape(spl_db)[1]):
#        L_p_a_na[n] = 10 * np.log10( (1/np.shape(spl_db)[0] * np.sqrt(np.sum(np.abs(p[:,n])**2, axis=0))**2) / \
#                (1/np.shape(non_audience_spl_db)[0] * np.sqrt(np.sum(np.abs(p_na[:,n])**2, axis=0))**2))
    #L_p_a_na = L_p_a_na - np.amin(L_p_a_na)
        
    return L_p_a_na


def smooth_spectrum(frequencies, X_f, r_oct):
    """
    Smoothes a frequency spectrum. Not used in bokeh.

    Parameters
    ----------
    frequencies : 1D-array
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
    for n in range(np.shape(frequencies)[0]):
        # standard deviation
        sigma = frequencies[n] / r_oct / np.pi
        # Gaussian window with the center frequnecy frequencies[n] an dstandard deviation
        w = np.exp( -(frequencies-frequencies[n])**2 / (2*sigma**2) )
        w = w / np.sum(w, axis=0)
        X_f_out[n] = np.sum(w * X_f)
    
    return X_f_out


def initial_directivity_plot():
    """
    Initializes the directivity plot in bokeh :any:`pdir`. Avoids error message
    that no data is presend to plot. Calls :any:`frequency_vector`. Called by
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
    frequencies : 1D-array
        Frequency bins in Hz.
    omega : 1D-array
        Frequency bins in radian.
    measured_directivity_data : 2D-array
        Measured loudspeaker data.
    measured_angles_deg : 2D-array
        Angle of measured loudspeaker data.
    H_post : 2D-array
        Directivity.

    """
    c0  = 331.45             # reference speed of sound in m/s
    alpha = 0.82
    beta_deg = np.linspace(-180,180,181)
    beta = beta_deg * np.pi / 180
    # frequencies
    frequencies = frequency_vector(frequency_count=120, octave_step=1/12, frequency_range_hz=[20,20000])
    omega = 2 * np.pi * frequencies
    measured_directivity_data = np.ones([np.shape(frequencies)[0],np.shape(frequencies)[0]])
    measured_angles_deg = np.ones([np.shape(frequencies)[0],np.shape(frequencies)[0]])
    H_post = np.zeros([np.shape(beta)[0],np.shape(frequencies)[0]], dtype=complex)
    return c0, alpha, beta_deg, beta, frequencies, omega, measured_directivity_data, measured_angles_deg, H_post


def parse_measured_directivity(csv_text, sound_field_config):
    """
    Reads (measured, complex) directivity data from the text of a .csv-file
    and stores it in the sound field prediction configuration. The GUI
    wrapper ``read_dir_data`` decodes the uploaded file and calls this
    function.

    The first row holds the frequencies in Hz, the first column the
    radiation angles in degree; the other entries are the complex
    directivity. The imaginary unit can be "j" or "i".

    Parameters
    ----------
    csv_text : str
        csv-data to read (decoded text).
    sound_field_config : obj [out]
        Sound field prediction configuration data. The attributes frequencies,
        plot_measured_angles_deg, plot_measured_directivity_db, measured_directivity_data, measured_angles_deg and
        measured_amplitude_db are written.

    Returns
    -------
    None.

    """
    # get data data frame
    new_df = csv_text
    counter = sum('\n' in s for s in new_df)+1
    new_df = str(np.char.replace(np.array(new_df).astype(str),'\n',','))
    new_df = new_df.split(',')
    directivity = np.char.replace(np.array(new_df),'i','j').astype(complex).reshape([int(counter),int(np.shape(new_df)[0]/counter)])

    # get an array of corresponding degree and frequency and delete these "header" and "index" from the directivity array
    # and write it once for whole data in dictionary and second just for the frequencies to be plotted in another dictionary
    # initialize the considered frequency bins
    sound_field_config.frequencies = frequency_vector(frequency_count=120, octave_step=1/12, frequency_range_hz=[20,20000])          
    # cut the degree and frequency vector out of the directivity array
    degree = np.real(directivity[1:,0]).astype(float)
    sound_field_config.plot_measured_angles_deg = list(degree)      
    frequency = np.real(directivity[0,1:]).astype(float)
    directivity = np.delete(np.delete(directivity,0,axis=0),0,axis=1)

    # interpolate directivity to the frequency bins used in bokeh app 
    directivity_f = np.array([])
    for n in range(np.shape(directivity)[0]):
        directivity_f = np.append(directivity_f, np.interp(sound_field_config.frequencies,frequency,directivity[n,:]))
    directivity_f = np.reshape(directivity_f,(np.shape(degree)[0],np.shape(sound_field_config.frequencies)[0]))    
    
    # get the index of the three frequencies to be plotted
    sound_field_config.set_plot_directivity(directivity_f, [100, 1000, 10000], measured=True) 
    
    # get frequency and degree in shape to upload in ColumnDataSource
    #frequencies = repmat(sound_field_config.frequencies, np.shape(directivity_f)[0], 1)
    degree = repmat(degree, np.shape(directivity_f)[1]-1, 1).transpose()
    
    # amplitude and phase calculation of directivity
    amplitude = 20*np.log10(np.abs(directivity_f))
    #phase = np.angle(directivity_f)

    # write data in ColumnDataSource
    for key, val in [['measured_directivity_data', directivity_f], ['measured_angles_deg', degree], ['measured_amplitude_db', amplitude]]:
        setattr(sound_field_config, key, val)
