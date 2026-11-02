"""Audience lines, draw conditions and the PALC compatible venue.

Formerly ``PAL`` and ``PALC_compatible_PAL`` from ``fun/PALC_classes.py``,
``fun/checkPAL.py`` and the logic of ``fun/gui_venue.py``.

author: Arne Hoelter
"""
from __future__ import annotations

import numpy as np

from .geometry import calc_intersection_PAL


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


def check_draw_conditions(x_start, x_stop, y_start, y_stop, created_pal):
    """
    Checks all draw conditions when an audience line is created
    (formerly the logic of ``check_all_draw_conditions`` in
    ``fun/gui_venue.py``).

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
    created_pal : obj [in]
        Contains all drawn lines by the user and corresponding data.

    Returns
    -------
    bool
        True if condition 1 passed, otherwise False.
    bool
        True if condition 2 passed, otherwise False.
    bool
        True if condition 3 passed, otherwise False.
    slope : float
        Slope of the drawn line.
    b : float
        Intersection point on the y-axis of the drawn line.
    message : str or None
        Error message the GUI shows if condition 1 or 2 failed, otherwise
        None. A failed condition 3 has no message.

    """
    # set condition3 default to True
    condition3 = True
    # calc slope and intersection on y-axis (b) of audience line
    slope, b = get_slope_b(x_start, x_stop, y_start, y_stop)

    # first condition (stop values are bigger or equal to the start values)
    condition1 = pal_draw_condition_1(x_start, x_stop, y_start, y_stop)
    if (x_start == x_stop and y_start == y_stop):
        return False, False, True, slope, b, \
            'Error: Wrong Input. Points are not allowed'
    elif not condition1:
        return False, False, True, slope, b, \
            'Error: Stop values must be higher or equal than the start values and only positive values are allowed'
    # second condition (the slope of the line must be zero or positive):
    # f(x)=m*x+b, first step: calculate slope (m) and b and save them
    # check if x_start and x_stop are equal (vertical line)
    if condition1:
        condition2 = pal_draw_condition_2(slope)
        if not condition2:
            return False, False, True, slope, b, \
                'Error: The slope of the drawn line must be positive'
    # avoid double clicking and and lines laying on each other
    if np.shape(created_pal.xline_start)[0] > 0:
        condition3 = pal_draw_condition_3(x_start, x_stop, y_start, y_stop,
                                          slope, b, created_pal)
    return condition1, condition2, condition3, slope, b, None
