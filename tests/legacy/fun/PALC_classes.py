#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Arne Hoelter

Definitions of all classes and their methods are in this module. The constructed
objects suit for the following aspects:
    
    * PALC positions, i.e., audience lines, non-audience line and audience
      lines that are only used in the sound field prediction
    * PALC configuration, i.e., loudspeaker data, array data and algorithm
      configuration
    * Results of PALC calculation and sound field prediction by the CDPS-model
    * Data to visualize the results
    * Weighting optimization by a target slope
"""

import numpy as np
from bokeh.models import Range1d

class PAL:
    """ Class that handles the drawn lines by the user """
    def __init__(self, xline_start=[], xline_stop=[], yline_start=[], yline_stop=[], \
                 slope=[], b=[], xline=[], yline=[], pal_index=[], \
                 plot_x=[], plot_y=[], ind_iscut=[]):
        self.xline_start, self.xline_stop  = xline_start, xline_stop
        self.yline_start, self.yline_stop  = yline_start, yline_stop
        self.slope, self.b, self.pal_index = slope, b, pal_index
        self.xline, self.yline             = xline, yline
        self.plot_x, self.plot_y           = plot_x, plot_y
        self.ind_iscut                     = ind_iscut
        
    def append_line(self, x_start=[], x_stop=[], y_start=[], y_stop=[], slope_new=[], \
                    b_new=[], eval_points=0):
        """
        Appends a line to the drawn audience lines.

        Parameters
        ----------
        x_start : float, optional
            Start point on the x-axis. The default is [].
        x_stop : float, optional
            Stop point on the x-axis. The default is [].
        y_start : float, optional
            Start point on the y-axis. The default is [].
        y_stop : float, optional
            Stop point on the y-axis. The default is [].
        slope_new : float, optional
            Slope of the new line. The default is [].
        b_new : float, optional
            Intersection point on the y-axis of the new line. The default is [].
        eval_points : float, optional
            Number of data points of the new line. The default is 0.

        Returns
        -------
        None.

        """
        self.xline_start = np.append(self.xline_start, x_start)
        self.xline_stop  = np.append(self.xline_stop, x_stop)
        self.yline_start = np.append(self.yline_start, y_start)
        self.yline_stop  = np.append(self.yline_stop, y_stop)
        self.slope       = np.append(self.slope, slope_new)
        self.b           = np.append(self.b, b_new)
        self.plot_x      = list(zip(self.xline_start, self.xline_stop))
        self.plot_y      = list(zip(self.yline_start, self.yline_stop))
        if np.all(eval_points):
            self.pal_index   = np.append(self.pal_index, [np.shape(self.xline)[0] - 1 + eval_points]).astype(int) 
            self.xline       = np.append(self.xline, np.linspace(x_start, x_stop, int(eval_points)))
            self.yline       = np.append(self.yline, np.linspace(y_start, y_stop, int(eval_points)))
        
    def remove_last_line(self):
        """
        Removes the last drawn line.

        Returns
        -------
        None.

        """
        self.xline_start = self.xline_start[:-1]
        self.xline_stop  = self.xline_stop[:-1]
        self.yline_start = self.yline_start[:-1]
        self.yline_stop  = self.yline_stop[:-1]
        self.slope       = self.slope[:-1]
        self.b           = self.b[:-1]
        self.plot_x      = self.plot_x[:-1]
        self.plot_y      = self.plot_y[:-1]
        if np.shape(self.pal_index)[0] > 2:
            self.xline = self.xline[:self.pal_index[-2]+1]
            self.yline = self.yline[:self.pal_index[-2]+1]
        else:
            self.xline = self.xline[:self.pal_index[-2]]
            self.yline = self.yline[:self.pal_index[-2]]
        self.pal_index = self.pal_index[:-1].astype(int)

        
    def insert_previous_line(self, sel_ind=[], x_start=[], x_stop=[], y_start=[], \
                             y_stop=[], slope_new=[], b_new=[], eval_points=0):
        """
        Inserts a line previous of the selected line.

        Parameters
        ----------
        sel_ind : int, optional
            Selected index, where the line shall be inserted previously.
            The default is [].
        x_start : float, optional
            Start point on the x-axis. The default is [].
        x_stop : float, optional
            Stop point on the x-axis. The default is [].
        y_start : float, optional
            Start point on the y-axis. The default is [].
        y_stop : float, optional
            Stop point on the y-axis. The default is [].
        slope_new : float, optional
            Slope of the new line. The default is [].
        b_new : float, optional
            Intersection point on the y-axis of the new line. The default is [].
        eval_points : float, optional
            Number of data points of the new line. The default is 0.

        Returns
        -------
        None.

        """
        self.xline_start = np.insert(self.xline_start, sel_ind, x_start)
        self.xline_stop  = np.insert(self.xline_stop, sel_ind, x_stop)
        self.yline_start = np.insert(self.yline_start, sel_ind, y_start)
        self.yline_stop  = np.insert(self.yline_stop, sel_ind, y_stop)
        self.slope       = np.insert(self.slope, sel_ind, slope_new)
        self.b           = np.insert(self.b, sel_ind, b_new)
        self.plot_x      = list(zip(self.xline_start, self.xline_stop))
        self.plot_y      = list(zip(self.yline_start, self.yline_stop))
        if np.all(eval_points):
            if sel_ind != 0:
                self.xline = np.insert(self.xline, self.pal_index[sel_ind]+1, np.linspace(x_start, x_stop, int(eval_points)))
                self.yline = np.insert(self.yline, self.pal_index[sel_ind]+1, np.linspace(y_start, y_stop, int(eval_points)))
                self.pal_index = np.insert(self.pal_index, sel_ind+1, self.pal_index[sel_ind] + eval_points).astype(int) # insert the new value
            else:
                self.xline = np.insert(self.xline, self.pal_index[sel_ind], np.linspace(x_start, x_stop, int(eval_points)))
                self.yline = np.insert(self.yline, self.pal_index[sel_ind], np.linspace(y_start, y_stop, int(eval_points)))
                self.pal_index = np.insert(self.pal_index, sel_ind+1, self.pal_index[sel_ind] + eval_points - 1).astype(int) # insert the new value
            # for pal_ind: 2. add number of discretization points to the shifted values
            for n in range(sel_ind+2, np.shape(self.pal_index)[0]):
                self.pal_index[n] = (self.pal_index[n] + eval_points).astype(int)
            
            
    def remove_selected_line(self, sel_ind=[]):
        """
        Removes the selected line.

        Parameters
        ----------
        sel_ind : int, optional
            Index of the selected line. The default is [].

        Returns
        -------
        None.

        """
        self.xline_start = np.delete(self.xline_start, sel_ind)
        self.xline_stop  = np.delete(self.xline_stop, sel_ind)
        self.yline_start = np.delete(self.yline_start, sel_ind)
        self.yline_stop  = np.delete(self.yline_stop, sel_ind)
        self.slope       = np.delete(self.slope, sel_ind)
        self.b           = np.delete(self.b, sel_ind)
        if sel_ind != []: del self.plot_x[sel_ind], self.plot_y[sel_ind]
        if sel_ind != 0:
            self.xline = np.delete(self.xline, np.s_[self.pal_index[sel_ind]+1:self.pal_index[sel_ind+1]+1])
            self.yline = np.delete(self.yline, np.s_[self.pal_index[sel_ind]+1:self.pal_index[sel_ind+1]+1])
        else:
            self.xline = np.delete(self.xline, np.s_[self.pal_index[sel_ind]:self.pal_index[sel_ind+1]+1])
            self.yline = np.delete(self.yline, np.s_[self.pal_index[sel_ind]:self.pal_index[sel_ind+1]+1])
        for n in range(sel_ind+2, np.shape(self.pal_index)[0]):
            if sel_ind != 0:
                self.pal_index[n] = int(self.pal_index[n] - (self.pal_index[n-1] - self.pal_index[n-2]))
            else:
                self.pal_index[n] = int(self.pal_index[n] - (self.pal_index[n-1] - self.pal_index[n-2]) - 1)
        self.pal_index = np.delete(self.pal_index, sel_ind+1)
        
class PALC_compatible_PAL(PAL):
    """ Class that contains data of the drawn lines in a compatible format to
        compute the PALC algorithm. The lines depend on an object of the class
        :py:class:`PAL`."""
    def __init__(self, xline_start=[], xline_stop=[], yline_start=[], yline_stop=[], slope=[], \
                 b=[], xline=[], yline=[], pal_index=[], plot_x=[], plot_y=[], pal=[], \
                 pal_no_nal=[]):
        PAL.__init__(self, xline_start=[], xline_stop=[], yline_start=[], yline_stop=[], \
                     slope=[], b=[], xline=[], yline=[], pal_index=[], plot_x=[], plot_y=[])
        self.pal        = pal
        self.pal_no_nal = pal_no_nal
        
    def append_pal(self, fir_arr_x, sec_arr_x, fir_arr_y, sec_arr_y, fir_ind, sec_ind, m):
        """
        Appends a polygonal audience line.

        Parameters
        ----------
        fir_arr_x : ndarray
            First array in x-direction.
        sec_arr_x : ndarray
            Second array in x-direction.
        fir_arr_y : ndarray
            First array in y-direction.
        sec_arr_y : ndarray
            Second array in y-direction.
        fir_ind : int
            Index of first array.
        sec_ind : int
            Index of second array.
        m : int
            Index in PALC compatible array.

        Returns
        -------
        None.

        """
        self.pal.append(np.zeros([sec_ind - fir_ind, 3]))
        self.pal[m][:,0] = np.append(fir_arr_x, sec_arr_x)
        self.pal[m][:,1] = np.append(fir_arr_y, sec_arr_y)
        
    def create_pal_without_nal(self, s_nal):
        """
        Creates polygonal audience lines without non-audience lines.

        Parameters
        ----------
        s_nal : ndarray
            Array of non-audience lines.

        Returns
        -------
        None.

        """
        if self.pal == []:
            return
        else:
            # create list of non-audience lines
            nal = []
            for n in range (np.shape(s_nal.xline_start)[0]):
                nal.append(np.array([[s_nal.xline_start[n], s_nal.yline_start[n], 0], \
                                     [s_nal.xline_stop[n], s_nal.yline_stop[n], 0]]))
            # create list of pal without non-audience lines
            isnal = []
            for m in range(len(self.pal)):
                for n in range(len(nal)):
                    if np.all(self.pal[m] == nal[n]):
                        isnal.append(m)
                if m not in isnal:
                    self.pal_no_nal.append(self.pal[m])
                        
            

class PALC_configuration():
    """ Class to create a configuration object of the PALC algorithm. """
    def __init__(self, Lambda_gap=0, Lambda_y=0.2, x_H=0, y_H=2, constraint='PALC 2', use_gamma_LSA=False, gamma_LSA=[], \
                 use_weighting='Without', weighting_nu=1, weighting_factors=[], N=int(4), gap_handling='Without', strength_sm=1, \
                 last_angle_hm=[], gap_weights=[], weighting_weights=[], use_fixed_angle=False, fixed_angle=0, \
                 directivity = 'Circular Piston', psi_n=[0.0524], gamma_n=[0], tolerance=1.0, freq_range=(200,8000)):
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
            except:
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
            except:
                return 'Error: only numbers are allowed. Set to default', new_input[0]
        elif new_input[0] in ['strength_sm', 'fixed_angle']:
            val = float(new_input[1])
        
        if hasattr(self, new_input[0]) and 'val' in locals():
            setattr(self, new_input[0], val)
        elif hasattr(self, new_input[0]):
            setattr(self, new_input[0], new_input[1])
        else:            
            print('Error: PALC_config has no attribute ' + new_input[0])
        
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
        
        
class SFP_configuration(PALC_configuration):
    """ Child class of PALC_configuration, contains data that are additionally
        needed for the sound field prediction. """
    def __init__(self, directivity='Circular Piston', N=1, Lambda_gap=0, \
                 Lambda_y=0.2, x_H=0, y_H=2, dir_meas=np.zeros([2,120]), \
                 dir_meas_deg=np.zeros([2,120]), \
                 dir_meas_amp=[], gamma_n=[0], f=[], plot_dir_amp=[], \
                 plot_dir_amp_meas=[[0],[0],[0]], plot_beta_meas=[0], \
                 freq_range=(200,8000)):
        self.dir_meas, self.f                     = dir_meas, f 
        self.dir_meas_deg, self.dir_meas_amp      = dir_meas_deg, dir_meas_amp
        self.plot_dir_amp, self.plot_dir_amp_meas = plot_dir_amp, plot_dir_amp_meas
        self.plot_beta_meas                       = plot_beta_meas
        self.freq_range                           = freq_range
        super(SFP_configuration, self).__init__()
        
    def get_plot_dir_amp(self, H_post, f2plot, meas=False):
        """
        Calculates the amplitude of the directivity to plot in :any:`pdir`.

        Parameters
        ----------
        H_post : ndarray
            (Complex) directivity data.
        f2plot : int
            frequency to plot.
        meas : bool, optional
            Must be True if measured loudspeaker data are used. The default
            is False.

        Returns
        -------
        None.

        """
        # get index
        self.plot_dir_amp, ind_f = [], []
        if meas: self.plot_dir_amp_meas = []
        # get amplitude values
        amplitude = 20*np.log10(np.abs(H_post))
        # get amplitudes for frequencies to plot
        for ind, n in enumerate(f2plot):
            ind_f.append((np.abs(self.f-n)).argmin())
            if meas:
                self.plot_dir_amp_meas.append(list(amplitude[:,ind_f[ind]].transpose()))
            else:
                self.plot_dir_amp.append(list(amplitude[:,ind_f[ind]].transpose()))

        
        
class Plot_ranges():
    """ Class to create objects that contain ranges of bokeh figures. """
    def __init__(self, p_x=[-0.5,2.5],p_y=[-0.5,2.5], bar_range=[0,0], \
                 bartop=[0,0], h_x=[0,1], h_y=[0,1], sfp_y=[0,1], \
                 SPLoverX_y=[0.3,15]):
        self.p_x, self.p_y          = p_x, p_y
        self.bar_range, self.bartop = bar_range, bartop
        self.h_y, self.sfp_y        = h_y, sfp_y
        self.SPLoverX_y             = SPLoverX_y
    
    def update_range(self, plot_axis, overlap, *plot_data):
        """
        Updates the range of bokeh figures.

        Parameters
        ----------
        plot_axis : str
            Figure and axis information.
        overlap : list
            Overlap to max and min value on considered axis.
        *plot_data : list
            List with all data points to consider.

        Returns
        -------
        new_range : obj
            Range1d object to be used on bokeh figures.

        """
        # number of inputs
        shape = len(plot_data)
        # initialize list outputs
        axmin, axmax = [], []
        # get min and max values of input array
        for n in range(shape):
            axmin.append(np.amin(plot_data[n]))
            axmax.append(np.amax(plot_data[n]))
        # write min and max value of previously detected min and max to attribute
        setattr(self, plot_axis, [min(axmin)+overlap[0], max(axmax)+overlap[1]])
        # write new range to bokeh Range1d object and return
        if plot_axis in ['bartop']:
            new_range = Range1d(-1, getattr(self, plot_axis)[1])
        else:
            new_range = Range1d(getattr(self, plot_axis)[0], getattr(self, plot_axis)[1])
        return new_range
    
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
    def __init__(self, gamma_tilt_deg=[]):
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
            
class Technical_meas_results():
    """ Class to create objects that contain the results of the technical
        measures based on the sound field prediction. """
    def __init__(self):
        self.f, self.p_SPL              = [], []
        self.H, self.H_dist             = [], []
        self.A_mean, self.hist_bins     = [], []
        self.Hist_tops, self.Hist_steps = [], []
        self.p_SPL_beam                 = []
        
        #for plotting
        self.plot_f, self.plot_p_SPL = [], []
        
    
    def update_tech_meas(self, **new_input):
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
        if 'f' in new_input.keys() and 'p_SPL' in new_input.keys():
            self.plot_p_SPL, self.plot_f = [], []
            for n in range(np.shape(new_input['p_SPL'])[0]):
                self.plot_p_SPL.append(list(self.p_SPL[n,:]))
                self.plot_f.append(list(self.f))

class Opt_weight():
    """ Class to optimize PALC by the target slope optimization. """
    def __init__(self, SPL_interp=[0,0]):
        self.x, self.SPL   = [], []
        self.x_interp      = [0,0]
        self.SPL_interp    = SPL_interp
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
