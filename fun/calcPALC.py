#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Arne Hoelter

The calcPALC module contains the calcPALC method.
"""

import numpy as np
from PALC_functions import calc_angles, LSA_visualization
from copy import copy
import time


def calcPALC(PALC_plots, PALC_pal, PALC_config, Opt_arr):
    """
    Handles the different PALC approaches. Offers the functionality of the
    Hard Margin Gap Handling Approaches. Without and Soft Margin Gap Handling
    Approach call directily :any:`calc_angles`.

    Parameters
    ----------
    PALC_plots : obj [in, out]
        Contains plotting information.
    PALC_pal : obj [in]
        Contains all venue information.
    PALC_config : obj [in]
        Contains PALC configuration data.
    Opt_arr : obj [in, out]
        Contains the data of the optimized LSA.

    Returns
    -------
    None.

    """
    #### again: initialization ####
    N       = int(PALC_config.N)
    gamma_n = np.array(PALC_config.gamma_n)
    psi_n   = np.array(PALC_config.psi_n)

    pal_e = list(PALC_pal.pal_no_nal)
    pal = list(PALC_pal.pal)

    # no Gap Handling or Soft Margin
    if PALC_config.gap_handling in ['Without', 'Soft Margin']:
        t1 = time.time()
        calc_angles(PALC_plots, pal, psi_n, gamma_n, N, PALC_config, pal_e, Opt_arr)
        t2 = time.time()
        dt = t2 - t1
        print('elapsed time in s: ',dt)
        return 
    # Gap Handling: Hard Margin Approach
    elif PALC_config.gap_handling in ['Hard Margin']:
        calc_angles(PALC_plots, pal, psi_n, gamma_n, N, PALC_config, pal_e, Opt_arr)
        # create pal_h that contains lists with the continuous audience lines
        pal_h = []
        pal_h.append([])
        m = 0
        for n in range(np.shape(pal_e)[0]):
            if pal_e[n][0,0] != pal[n+m][0,0] or pal_e[n][0,1] != pal[n+m][0,1]:
                pal_h.append([])
                m += 1
            if pal_e[n][0,0] == pal[n+m][0,0] and pal_e[n][0,1] == pal[n+m][0,1]:
                pal_h[m].append(pal[n+m])
        # check which loudspeaker cabinets belong to which continuous audience line
        ls2seg = np.empty(N).astype(int)
        seg_dist_high = np.empty(np.shape(pal_h)[0])
        seg_dist_low = np.empty(np.shape(pal_h)[0])
        seg_dist = np.empty(np.shape(pal_h)[0])
        for m in range(N):
            # go from the highest audience line to the lowest
            for n in range(np.shape(pal_h)[0]-1,-1,-1):
                seg_dist_high[n] = np.abs(np.sqrt((pal_h[n][-1][-1,0]-Opt_arr.seg_pos[m,0])**2 \
                              + (pal_h[n][-1][-1,1]-Opt_arr.seg_pos[m,1])**2))
                seg_dist_low[n] = np.abs(np.sqrt((pal_h[n][0][0,0]-Opt_arr.seg_pos[m,0])**2 \
                            + (pal_h[n][0][0,1]-Opt_arr.seg_pos[m,1])**2))
                seg_dist[n] = np.minimum(seg_dist_high[n], seg_dist_low[n])
            
            # get the indice of the segment to which the LS belongs
            ls2seg[m] = int(np.argmin(seg_dist))

        # Run the calculation for each continuous audience line
        for n in range(np.shape(pal_h)[0]-1,-1,-1):
            N_margin = 0
            for m in range(N):
                if n == ls2seg[m]:
                    N_margin += 1

            if n == np.shape(pal_h)[0]-1:
                gamma_n = np.linspace(-0.0428, 0.7147, num=N_margin)
                psi_n = np.linspace(0.00524, 0.00524, num=N_margin)
                calc_angles(PALC_plots, pal_h[n], psi_n, gamma_n, N_margin, PALC_config, pal_e, Opt_arr)
                x_patches, y_patches = LSA_visualization(PALC_plots, PALC_config, Opt_arr.gamma_n, \
                                                          N_margin=np.shape(Opt_arr.gamma_tilt_deg)[0])

                Opt_arr_hm = copy(Opt_arr)
                PALC_plots_hm = copy(PALC_plots)
            else:

                PALC_config.last_angle_hm = Opt_arr_hm.gamma_tilt_deg[-1] * np.pi / 180
                PALC_config.gamma_n = gamma_n = np.linspace(-0.0428, 0.7147, num=N_margin)
                PALC_config.psi_n = psi_n = np.linspace(0.00524, 0.00524, num=N_margin)

                PALC_config.x_H = PALC_plots_hm.x_stop[-1] - np.sin(PALC_config.last_angle_hm) * PALC_config.Lambda_gap
                PALC_config.y_H = PALC_plots_hm.y_stop[-1] - np.cos(PALC_config.last_angle_hm) * PALC_config.Lambda_gap

                calc_angles(PALC_plots_hm, pal_h[n], psi_n, gamma_n, N_margin, PALC_config, pal_e, Opt_arr_hm)
                x_patches_hm, y_patches_hm = LSA_visualization(PALC_plots_hm, PALC_config, Opt_arr_hm.gamma_n, \
                                                          N_margin=np.shape(Opt_arr_hm.gamma_tilt_deg)[0])
                for attr in ['x_c_n', 'y_c_n','x_start', 'y_start', 'x_stop', \
                              'y_stop', 'x_fin_unitn', 'y_fin_unitn']:
                    setattr(PALC_plots, attr, np.append(getattr(PALC_plots, attr), getattr(PALC_plots_hm, attr)))
                for attr in ['seg_pos', 'seg_pos_start', 'seg_pos_stop', 'x_fin_unitn_psi_1', \
                              'y_fin_unitn_psi_1', 'x_fin_unitn_psi_2', 'y_fin_unitn_psi_2', \
                              'gamma_n', 'gamma_tilt_deg', 'thr_dist']:
                    setattr(Opt_arr, attr, np.append(getattr(Opt_arr, attr), getattr(Opt_arr_hm, attr)))
                    if attr in ['seg_pos', 'seg_pos_start', 'seg_pos_stop']:
                        setattr(Opt_arr, attr, getattr(Opt_arr, attr).reshape(int(N),3))
        return