#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Arne Hoelter

The checkPAL module checks conditions of user drawn audience lines. Non-audience
lines can be inserted. The venue slice will be transformed into a PALC
compatible format. This guarantees also the compatibility for the sound field
prediction (SFP).
"""

import numpy as np
from PALC_functions import calc_intersection_PAL


def get_slope_b(x_start, x_stop, y_start, y_stop):
    """
    Computes the slope and intersection point on the y-axis of line specified by
    start and stop point. The format of the line is: y = slope * x + b

    Parameters
    ----------
    x_start : float
        Start point on x-axis.
    x_stop : float
        Stop point on x-axis.
    y_start : float
        Start point on y-axis.
    y_stop : float
        Stop point on y-axis.

    Returns
    -------
    slope : float
        Slope of the drawn line.
    b : float
        Intersection point with y-axis.

    """
    if x_start != x_stop:
        slope = (y_start - y_stop) / (x_start - x_stop)
        b = (y_start - ((y_start - y_stop) / (x_start - x_stop)) * x_start)
    else: # vertical line: slope and b are infinity
        slope = 5000
        b = 5000
    return slope, b

def pal_draw_condition_1(x_start, x_stop, y_start, y_stop):
    """
    First draw condition of polygonal audience lines. Stop values are bigger
    or equal to the start values

    Parameters
    ----------
    x_start : float
        Start point on x-axis.
    x_stop : float
        Stop point on x-axis.
    y_start : float
        Start point on y-axis.
    y_stop : float
        Stop point on y-axis.

    Returns
    -------
    bool
        True if condition has passed, otherwise False.

    """
    if (0 <= x_start <= x_stop and 0 <= y_start <= y_stop) and (x_start != x_stop or y_start != y_stop):
        return True
    else:
        return False
    
    
def pal_draw_condition_2(slope):
    """
    Second draw condition of polygonal audience lines. The slope of the
    lines must be zero or positive

    Parameters
    ----------
    slope : float
        Slope of the polygonal audience line.

    Returns
    -------
    condition_2 : bool
        True if condition has passed, otherwise False.

    """
    if slope < 0:
        return False
    else:
        return True


def pal_draw_condition_3(x_start, x_stop, y_start, y_stop, slope, b, created_pal):
    """
    Third draw condition of polygonal audience lines. No overlay of audience lines.
    Also to avoid double clicking at line creation

    Parameters
    ----------
    x_start : float
        Start point on x-axis.
    x_stop : float
        Stop point on x-axis.
    y_start : float
        Start point on y-axis.
    y_stop : float
        Stop point on y-axis.
    slope : float
        Slope of drawn line.
    b : float
        Intersection point on y-axis of drawn line.
    created_pal : obj [in]
        Contains data of the drawn lines.

    Returns
    -------
    bool
        True if condition has passed, otherwise False.

    """
    for n in range(np.shape(created_pal.xline_start)[0]):
        xy_s2,_,_,px = calc_intersection_PAL([x_start, y_start], [x_stop, y_stop], \
                                      [created_pal.xline_start[n], created_pal.yline_start[n]], [created_pal.xline_stop[n], created_pal.yline_stop[n]])
        # check if overlay exists
        if not px and b == created_pal.b[n]:
            # check if line is before or behind of tested line
            if x_start >= created_pal.xline_stop[n] or x_stop <= created_pal.xline_start[n]:
                pass
            else:
                return False
        else:
            pass
            #condition_3 = True        
    return True

def which_pal_jump(created_pal, mPAL):
    """
    Detects lines that will be totally avoided in the PALC computation. This
    must be considered in the PALC compatible PAL format.

    Parameters
    ----------
    created_pal : obj [in]
        Contains the drawn polygonal audience lines.
    mPAL : int
        Number of audience lines to be considered.

    Returns
    -------
    jump : int
        Index of lines that are totally avoided in PALC computation.

    """
    # initialize jump index
    jump = np.array([])
    # check which lines to be jumped for the cases [3,7,8,9]
    for n in range(mPAL-1):
        for ni in range(n+1,mPAL):
            if created_pal.xline_start[n] >= created_pal.xline_start[ni] and \
            not created_pal.xline_stop[n] == created_pal.xline_start[ni]:
                jump = np.append(jump, n)
    # check which lines to be jumped for the cases [1,4,5,7]
    n = 0
    while n < mPAL-1:
        for m in range(n,mPAL):
            if np.any(m == jump):
                n += 1
            else:
                break
        ni = n+1
        n2 = 0
        while ni < mPAL:
            if created_pal.yline_stop[n] >= created_pal.yline_stop[ni] and \
            (created_pal.xline_stop[n] != created_pal.xline_start[ni] and \
             created_pal.yline_stop[n] != created_pal.yline_start[ni]):
                jump = np.append(jump, ni)
                n2 += 1
            ni += 1
        n += 1 + n2
    # sort the index of lines to be jumped and delete possibly multiple entries
    jump = np.array(list(set(np.sort(jump)))).astype(int)
        
    return jump


def suggestPAL(created_pal, suggested_nal, suggested_sfp):
    """
    Suggests a venue slice depending on drawn audience lines. Creates a suggestion
    for non-audience lines and lines only considered for SFP. Therefore considers
    all possible cases explained in AES PAPER REF.

    Parameters
    ----------
    created_pal : obj [in]
        Contains data of the drawn audience lines by the user.
    suggested_nal : obj [out]
        Contains data of the non-audience line.
    suggested_sfp : obj [out]
        Contains data of audience lines that are only used for SFP.

    Returns
    -------
    None.

    """
    # we consider 11 conditions, 9 condition: x1 <=> x2 * y1 <=> y2 + 2 special cases for x1 > x2 and y1 >= y2
    # get pal data to unpack
    xline_start = created_pal.xline_start
    xline_stop  = created_pal.xline_stop
    yline_start = created_pal.yline_start
    yline_stop  = created_pal.yline_stop
  
    mPAL = np.shape(xline_start)[0] # number of pal lines to be considered
    if mPAL < 2: return suggested_nal, suggested_sfp
    # check cases, jump has the index of totally avoided lines in PALC
    jump = which_pal_jump(created_pal, mPAL)
    # add jump lines to suggested_sfp only lines
    suggested_sfp.append_line(xline_start[jump], xline_stop[jump], yline_start[jump], yline_stop[jump])
    suggested_sfp.pal_index = np.append(suggested_sfp.pal_index, jump).astype(int)
    m = 0
    while m < mPAL-1:
        nj = 0
        # cases when to leave while loop and which lines to be jumped
        for n in range(m, mPAL-1):
            if np.any(n+1 == jump):
                nj += 1
            else:
                break
        if m+1+nj >= mPAL:
            break
        # same handling for cases: [1,4], [3,5,6], [9], special handling case [7,8] and no handling in case 2
        if xline_stop[m] <= xline_start[m+1+nj] and yline_stop[m] > yline_start[m+1+nj]: # case 1 and 4
            xy_s = calc_intersection_PAL([xline_stop[m], yline_stop[m]], [xline_stop[m+1+nj], yline_stop[m]], \
                                         [xline_start[m+1+nj], yline_start[m+1+nj]], [xline_stop[m+1+nj], yline_stop[m+1+nj]])[0]
            xy_s2 = calc_intersection_PAL([xline_start[m], yline_start[m]], [xline_stop[m], yline_stop[m]], \
                                         [xline_start[m+1+nj], yline_start[m+1+nj]], [xline_stop[m+1+nj], yline_stop[m+1+nj]])[0]
            if xy_s2[0] == xline_stop[m] and xy_s2[1] == yline_stop[m] \
               and xline_start[m+1+nj] == xy_s2[0] == xline_stop[m+1+nj]: # special case: line m+1 is vertical at stop pos of line m
                suggested_sfp.append_line(xline_start[m+1+nj], xy_s2[0], yline_start[m+1+nj], xy_s2[1])
                suggested_sfp.pal_index = np.append(suggested_sfp.pal_index, m+1+nj).astype(int)
                suggested_sfp.ind_iscut.append(['after start', m+1+nj])
                m += 1+nj
                continue
                
            suggested_nal.append_line(xline_stop[m], xy_s[0], yline_stop[m], xy_s[1])
            suggested_sfp.append_line(xline_start[m+1+nj], xy_s[0], yline_start[m+1+nj], xy_s[1])
            
            suggested_nal.pal_index = np.append(suggested_nal.pal_index, m).astype(int)
            suggested_sfp.pal_index = np.append(suggested_sfp.pal_index, m+1+nj).astype(int)
            suggested_sfp.ind_iscut.append(['after start', m+1+nj])

        elif (xline_stop[m] < xline_start[m+1+nj] and yline_stop[m] <= yline_start[m+1+nj]) or \
             (xline_stop[m] == xline_start[m+1+nj] and yline_stop[m] < yline_start[m+1+nj]): # case 3, 5, 6
            suggested_nal.append_line(xline_stop[m], xline_start[m+1+nj], yline_stop[m], yline_start[m+1+nj])

            suggested_nal.pal_index = np.append(suggested_nal.pal_index, m).astype(int)

        elif xline_stop[m] > xline_start[m+1+nj] and yline_stop[m] < yline_start[m+1+nj]: # case 9
            xy_s = calc_intersection_PAL([xline_start[m],yline_start[m]], [xline_stop[m], yline_stop[m]], \
                                         [xline_start[m+1+nj],yline_start[m]], [xline_start[m+1+nj], yline_stop[m+1+nj]])[0]
            suggested_nal.append_line(xy_s[0], xline_start[m+1+nj], xy_s[1], yline_start[m+1+nj])
            suggested_sfp.append_line(xy_s[0], xline_stop[m], xy_s[1], yline_stop[m])

            suggested_nal.pal_index = np.append(suggested_nal.pal_index, m).astype(int)
            suggested_sfp.pal_index = np.append(suggested_sfp.pal_index, m).astype(int)
            suggested_sfp.ind_iscut.append(['before end', m])

        elif (xline_stop[m] > xline_start[m+1+nj] and yline_stop[m] >= yline_start[m+1+nj]): # case 8 and 7 (special handling)
            xy_s, _, _, p_ex = calc_intersection_PAL([xline_start[m],yline_start[m]], [xline_stop[m], yline_stop[m]], \
                                         [xline_start[m+1+nj],yline_start[m+1+nj]], [xline_stop[m+1+nj], yline_stop[m+1+nj]])
            xy_s2, _, _, p_ex2 = calc_intersection_PAL([xline_stop[m],yline_stop[m]], [xline_stop[m], yline_stop[m]+2], \
                                         [xline_start[m+1+nj],yline_start[m+1+nj]], [xline_stop[m+1+nj], yline_stop[m+1+nj]])
            xy_s3, _, _, p_ex2 = calc_intersection_PAL([xline_stop[m],yline_stop[m]], [xline_stop[m+1+nj], yline_stop[m]], \
                                         [xline_start[m+1+nj],yline_start[m+1+nj]], [xline_stop[m+1+nj], yline_stop[m+1+nj]])
            xy_s4, _, _, p_ex2 = calc_intersection_PAL([xline_start[m],yline_start[m]], [xline_stop[m], yline_stop[m]], \
                                         [xline_start[m+1+nj],yline_start[m+1+nj]], [xline_start[m+1+nj], yline_start[m+1+nj]+2])
            # case 7: intersection point is on the lines:
            if (xline_start[m] <= xy_s[0] < xline_stop[m]) and (yline_start[m] <= xy_s[1] <= yline_stop[m]) \
                and (xline_start[m+1+nj] <= xy_s[0] <= xline_stop[m+1+nj]) and (yline_start[m+1+nj] < xy_s[1] <= yline_stop[m+1+nj]) and p_ex == True:
                suggested_sfp.append_line([xy_s[0], xline_start[m+1+nj]], [xline_stop[m], xy_s[0]], [xy_s[1], yline_start[m+1+nj]], [yline_stop[m], xy_s[1]])

                suggested_sfp.pal_index = np.append(suggested_sfp.pal_index, [m, m+1+nj]).astype(int)
                suggested_sfp.ind_iscut.append(['before end', m])
                suggested_sfp.ind_iscut.append(['after start', m+1+nj])
             
            # case 8: start point of line m+1 is intersec point
            elif xy_s[0] == xline_start[m+1+nj] and xy_s[1] == yline_start[m+1+nj]:
                    suggested_sfp.append_line(xline_start[m+1+nj], xline_stop[m], yline_start[m+1+nj], yline_stop[m])

                    suggested_sfp.pal_index = np.append(suggested_sfp.pal_index, m).astype(int)
                    suggested_sfp.ind_iscut.append(['before end', m])
                    
            # case 10: line m is over line m+1 
            elif (yline_stop[m] >= xy_s2[1]):              
                # special case: intersec point is stop point of line m but line m+1 stays over line m
                #               could also be handled differently, but there seems not to be a real case like this
                if np.round(xy_s[0], decimals=6) == xline_stop[m] and \
                   np.round(xy_s[1], decimals=6) == yline_stop[m] and \
                   np.round(xy_s4[1],decimals=6) <  yline_start[m+1+nj]:
                    if xline_start[m] >= xline_start[m+1+nj]:
                        m += 1+nj
                        continue
                    suggested_sfp.append_line(xline_start[m+1+nj], xy_s[0], yline_start[m+1+nj], xy_s[1])
                    suggested_sfp.pal_index = np.append(suggested_sfp.pal_index, m+1+nj).astype(int)
                    suggested_sfp.ind_iscut.append(['after start', m+1+nj])
                    m += 1+nj
                    continue
                # special case: intersec point is stop point of line m
                if np.round(xy_s[0], decimals=6) == xline_stop[m] and \
                   np.round(xy_s[1], decimals=6) == yline_stop[m]:
                    
                    suggested_sfp.append_line(xline_start[m+1+nj], xy_s3[0], yline_start[m+1+nj], xy_s3[1])
                    suggested_sfp.pal_index = np.append(suggested_sfp.pal_index, m+1+nj).astype(int)
                    suggested_sfp.ind_iscut.append(['after start', m+1+nj])
                    m += 1+nj
                    continue                
                    
                suggested_nal.append_line(xline_stop[m], xy_s3[0], yline_stop[m], xy_s3[1])
                suggested_sfp.append_line(xline_start[m+1+nj], xy_s3[0], yline_start[m+1+nj], xy_s3[1])

                suggested_nal.pal_index = np.append(suggested_nal.pal_index, m).astype(int)
                suggested_sfp.pal_index = np.append(suggested_sfp.pal_index, m+1+nj).astype(int)
                suggested_sfp.ind_iscut.append(['after start', m+1+nj])

            # case 11: line m is below line m+1
            elif (yline_stop[m] < xy_s2[1]):
                if np.round(xy_s[0], decimals=6) == xline_start[m+1+nj] and \
                   np.round(xy_s[1], decimals=6) == yline_start[m+1+nj]: # special case: intersec point is start point of line m+1
                    suggested_sfp.append_line(xy_s4[0], xline_stop[m], xy_s4[1], yline_stop[m])
                    suggested_sfp.pal_index = np.append(suggested_sfp.pal_index, m).astype(int)
                    suggested_sfp.ind_iscut.append(['before end', m])
                    m += 1+nj
                    continue

                suggested_nal.append_line(xy_s4[0], xline_start[m+1+nj], xy_s4[1], yline_start[m+1+nj])
                suggested_sfp.append_line(xy_s4[0], xline_stop[m], xy_s4[1], yline_stop[m])

                suggested_nal.pal_index = np.append(suggested_nal.pal_index, m).astype(int)
                suggested_sfp.pal_index = np.append(suggested_sfp.pal_index, m).astype(int)
                suggested_sfp.ind_iscut.append(['before end', m])
        m += 1+nj

    suggested_sfp.xline_start, suggested_sfp.xline_stop, suggested_sfp.yline_start, suggested_sfp.yline_stop \
    = suggested_sfp.xline_start[suggested_sfp.pal_index.argsort()], suggested_sfp.xline_stop[suggested_sfp.pal_index.argsort()], \
      suggested_sfp.yline_start[suggested_sfp.pal_index.argsort()], suggested_sfp.yline_stop[suggested_sfp.pal_index.argsort()]
    # plot list cannot be sorted, however it does not seem to be necessary
#    suggested_sfp.plot_x, suggested_sfp.plot_y = suggested_sfp.plot_x[suggested_sfp.pal_index.argsort()], suggested_sfp.plot_y[suggested_sfp.pal_index.argsort()]
    suggested_sfp.pal_index.sort()
    return

def get_PALC_compatible_PAL(PALC_pal, c_pal, s_nal, s_sfp):
    """
    Creates a PALC compatible PAL object that can be passed to the PALC
    computation. Needs the data that were generated in :any:`suggestPAL`.

    Parameters
    ----------
    PALC_pal : obj [out]
        Contains the object to pass to PALC computation.
    c_pal : obj [in]
        Contains data of the drawn audience lines by the user.
    s_nal : obj [in]
        Contains data of the non-audience line.
    s_sfp : obj [in]
        Contains data of audience lines that are only used for SFP.

    Returns
    -------
    None.

    """
    # get number of lines that are not included in PALC calculation at all
    pals2del = []
    for n in range(np.shape(c_pal.pal_index)[0]-1):
        for m in range(np.shape(s_sfp.pal_index)[0]):
            if c_pal.plot_x[n] == s_sfp.plot_x[m] and c_pal.plot_y[n] == s_sfp.plot_y[m]:
                pals2del.append(n)
    # remove audience lines that are not included in PALC calculation
    s_sfp_ind = np.array([])
    if pals2del != []:
        for n in s_sfp.pal_index:
            if n not in pals2del:
                s_sfp_ind = np.append(s_sfp_ind, n).astype(int)
    else:
        s_sfp_ind = np.array(s_sfp.pal_index)

    # total number of pal: shape(c_pal) - 1 + shape(s_nal) - pals2del
    m = 0
    for n in range(np.shape(c_pal.pal_index)[0] - 1):
        # case 1: non-audience line is inserted
        if n-1 in s_nal.pal_index:
            # index of inserted line in s_nal object
            ind_a = np.argwhere(s_nal.pal_index == n-1)[0,0]
            prev_line = 'nal'
            PALC_pal.append_pal(s_nal.xline_start[ind_a], s_nal.xline_stop[ind_a], s_nal.yline_start[ind_a], s_nal.yline_stop[ind_a], 0, 2, m)
            m += 1
        else:
            prev_line = 'haspal'
        # check if next line is non-audience line
        if n in s_nal.pal_index:
            ind_b       = np.argwhere(s_nal.pal_index == n)[0,0]
            next_line   = 'nal'         
        else:
            next_line = 'haspal'    
        # get sfp index for current n, if there is one
        if s_sfp_ind.size != 0: sfp_ind = np.argmin(np.abs(s_sfp_ind - n))
        # case n is a jumped audience line
        if n in pals2del:
            pass
        
        # case 2: the pal line is cut off after the start, i.e., new start point is inserted
        elif ['after start', n] in s_sfp.ind_iscut:
            cut_ind = np.argmin(np.abs(c_pal.xline[c_pal.pal_index[n]:c_pal.pal_index[int(n+1)]] - s_sfp.xline_stop[sfp_ind])) + c_pal.pal_index[n]                

            if prev_line == 'nal':
                PALC_pal.append_pal(s_nal.xline_stop[ind_a], c_pal.xline[cut_ind+1:c_pal.pal_index[n+1]+1], s_nal.yline_stop[ind_a], \
                                    c_pal.yline[cut_ind+1:c_pal.pal_index[n+1]+1], cut_ind, c_pal.pal_index[n+1]+1, m)
            elif ['before end', n-1] in s_sfp.ind_iscut:
                PALC_pal.append_pal(s_sfp.xline_start[sfp_ind-1], c_pal.xline[cut_ind+1:c_pal.pal_index[n+1]+1], s_sfp.yline_start[sfp_ind-1], \
                                    c_pal.yline[cut_ind+1:c_pal.pal_index[n+1]+1], cut_ind, c_pal.pal_index[n+1]+1, m)
            else:
                cut_ind = np.argmin(np.abs(c_pal.xline[c_pal.pal_index[n]+1:c_pal.pal_index[int(n+1)]] - s_sfp.xline_stop[sfp_ind])) + c_pal.pal_index[n]+1
                PALC_pal.append_pal(s_sfp.xline_stop[sfp_ind], c_pal.xline[cut_ind+1:c_pal.pal_index[n+1]+1], s_sfp.yline_stop[sfp_ind], \
                                    c_pal.yline[cut_ind+1:c_pal.pal_index[n+1]+1], cut_ind, c_pal.pal_index[n+1]+1, m)
            m += 1
            # if 'before end' will be called for current line:
            if ['before end', n] in s_sfp.ind_iscut:
                # get cut_ind
                cut_ind = np.argmin(np.abs(PALC_pal.pal[-1][:,0] - s_sfp.xline_start[sfp_ind+1]))
                # cut pal that is part of only sfp line
                PALC_pal.pal[-1] = PALC_pal.pal[-1][:cut_ind,:]
                # append the intersection point to the pal
                PALC_pal.pal[-1] = np.append(PALC_pal.pal[-1], [s_sfp.xline_start[sfp_ind+1], s_sfp.yline_start[sfp_ind+1], \
                                             0]).reshape(int(np.shape(PALC_pal.pal[-1])[0]+1),3)

        # case 3: the pal line is cut off before the end, i.e., new stop point is inserted
        elif ['before end', n] in s_sfp.ind_iscut:
            cut_ind = np.argmin(np.abs(c_pal.xline[c_pal.pal_index[n]:c_pal.pal_index[int(n+1)]] - s_sfp.xline_start[sfp_ind])) + c_pal.pal_index[n]
            if m == 0:
                if next_line == 'nal':
                    PALC_pal.append_pal(c_pal.xline[c_pal.pal_index[n]:cut_ind], s_nal.xline_start[ind_b], c_pal.yline[c_pal.pal_index[n]:cut_ind], \
                                        s_nal.yline_start[ind_b], c_pal.pal_index[n], cut_ind+1, m)
                else:
                    PALC_pal.append_pal(c_pal.xline[c_pal.pal_index[n]:cut_ind], s_sfp.xline_start[sfp_ind], c_pal.yline[c_pal.pal_index[n]:cut_ind], \
                                        s_sfp.yline_start[sfp_ind], c_pal.pal_index[n], cut_ind+1, m)
            else:
                if next_line == 'nal':
                    PALC_pal.append_pal(c_pal.xline[c_pal.pal_index[n]+1:cut_ind], s_nal.xline_start[ind_b], c_pal.yline[c_pal.pal_index[n]+1:cut_ind], \
                                        s_nal.yline_start[ind_b], c_pal.pal_index[n], cut_ind, m)
                else:
                    PALC_pal.append_pal(c_pal.xline[c_pal.pal_index[n]+1:cut_ind], s_sfp.xline_start[sfp_ind], c_pal.yline[c_pal.pal_index[n]+1:cut_ind], \
                                        s_sfp.yline_start[sfp_ind], c_pal.pal_index[n], cut_ind, m)
            m += 1

        # case 4: created pal line is not changed
        elif [['after start', n],['before end', n]] not in s_sfp.ind_iscut:
            if m == 0:
                PALC_pal.append_pal(c_pal.xline[c_pal.pal_index[n]:c_pal.pal_index[int(n+1)]], c_pal.xline[c_pal.pal_index[int(n+1)]], \
                                                c_pal.yline[c_pal.pal_index[n]:c_pal.pal_index[int(n+1)]], c_pal.yline[c_pal.pal_index[int(n+1)]], \
                                                c_pal.pal_index[n], c_pal.pal_index[int(n+1)]+1, m)
            else:
                PALC_pal.append_pal(c_pal.xline[c_pal.pal_index[n]+1:c_pal.pal_index[int(n+1)]], c_pal.xline[c_pal.pal_index[int(n+1)]], \
                                                c_pal.yline[c_pal.pal_index[n]+1:c_pal.pal_index[int(n+1)]], c_pal.yline[c_pal.pal_index[int(n+1)]], \
                                                c_pal.pal_index[n], c_pal.pal_index[int(n+1)], m)
            m += 1
    #return PALC_pal
                
    
    
    