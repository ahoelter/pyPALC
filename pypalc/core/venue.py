"""Audience lines, draw conditions and the PALC compatible venue.

:class:`AudienceLines` and :class:`PalcVenue` (formerly ``PAL`` and
``PALC_compatible_PAL`` from ``fun/PALC_classes.py``),
``fun/checkPAL.py`` and the logic of ``fun/gui_venue.py``.

author: Arne Hoelter
"""
from __future__ import annotations

import numpy as np

from .geometry import line_intersection


class AudienceLines:
    """ Class that handles the drawn lines by the user """
    def __init__(self, line_start_x=None, line_stop_x=None, line_start_y=None, \
                 line_stop_y=None, slope=None, intercept=None, points_x=None, points_y=None, \
                 line_end_indices=None, plot_x=None, plot_y=None, cut_lines=None):
        # None stands for a new empty list (no shared mutable defaults)
        line_start_x, line_stop_x, line_start_y, line_stop_y, slope, intercept, points_x, \
            points_y, line_end_indices, plot_x, plot_y, cut_lines = \
            [[] if val is None else val for val in
             (line_start_x, line_stop_x, line_start_y, line_stop_y, slope, intercept,
              points_x, points_y, line_end_indices, plot_x, plot_y, cut_lines)]
        self.line_start_x, self.line_stop_x  = line_start_x, line_stop_x
        self.line_start_y, self.line_stop_y  = line_start_y, line_stop_y
        self.slope, self.intercept, self.line_end_indices = slope, intercept, line_end_indices
        self.points_x, self.points_y             = points_x, points_y
        self.plot_x, self.plot_y           = plot_x, plot_y
        self.cut_lines                     = cut_lines
        
    def append_line(self, x_start=[], x_stop=[], y_start=[], y_stop=[], slope_new=[], \
                    intercept_new=[], eval_points=0):
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
        intercept_new : float, optional
            Intersection point on the y-axis of the new line. The default is [].
        eval_points : float, optional
            Number of data points of the new line. The default is 0.

        Returns
        -------
        None.

        """
        self.line_start_x = np.append(self.line_start_x, x_start)
        self.line_stop_x  = np.append(self.line_stop_x, x_stop)
        self.line_start_y = np.append(self.line_start_y, y_start)
        self.line_stop_y  = np.append(self.line_stop_y, y_stop)
        self.slope       = np.append(self.slope, slope_new)
        self.intercept           = np.append(self.intercept, intercept_new)
        self.plot_x      = list(zip(self.line_start_x, self.line_stop_x))
        self.plot_y      = list(zip(self.line_start_y, self.line_stop_y))
        if np.all(eval_points):
            self.line_end_indices   = np.append(self.line_end_indices, [np.shape(self.points_x)[0] - 1 + eval_points]).astype(int) 
            self.points_x       = np.append(self.points_x, np.linspace(x_start, x_stop, int(eval_points)))
            self.points_y       = np.append(self.points_y, np.linspace(y_start, y_stop, int(eval_points)))
        
    def remove_last_line(self):
        """
        Removes the last drawn line.

        Returns
        -------
        None.

        """
        self.line_start_x = self.line_start_x[:-1]
        self.line_stop_x  = self.line_stop_x[:-1]
        self.line_start_y = self.line_start_y[:-1]
        self.line_stop_y  = self.line_stop_y[:-1]
        self.slope       = self.slope[:-1]
        self.intercept           = self.intercept[:-1]
        self.plot_x      = self.plot_x[:-1]
        self.plot_y      = self.plot_y[:-1]
        if np.shape(self.line_end_indices)[0] > 2:
            self.points_x = self.points_x[:self.line_end_indices[-2]+1]
            self.points_y = self.points_y[:self.line_end_indices[-2]+1]
        else:
            self.points_x = self.points_x[:self.line_end_indices[-2]]
            self.points_y = self.points_y[:self.line_end_indices[-2]]
        self.line_end_indices = self.line_end_indices[:-1].astype(int)

        
    def insert_previous_line(self, selected_index=[], x_start=[], x_stop=[], y_start=[], \
                             y_stop=[], slope_new=[], intercept_new=[], eval_points=0):
        """
        Inserts a line previous of the selected line.

        Parameters
        ----------
        selected_index : int, optional
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
        intercept_new : float, optional
            Intersection point on the y-axis of the new line. The default is [].
        eval_points : float, optional
            Number of data points of the new line. The default is 0.

        Returns
        -------
        None.

        """
        self.line_start_x = np.insert(self.line_start_x, selected_index, x_start)
        self.line_stop_x  = np.insert(self.line_stop_x, selected_index, x_stop)
        self.line_start_y = np.insert(self.line_start_y, selected_index, y_start)
        self.line_stop_y  = np.insert(self.line_stop_y, selected_index, y_stop)
        self.slope       = np.insert(self.slope, selected_index, slope_new)
        self.intercept           = np.insert(self.intercept, selected_index, intercept_new)
        self.plot_x      = list(zip(self.line_start_x, self.line_stop_x))
        self.plot_y      = list(zip(self.line_start_y, self.line_stop_y))
        if np.all(eval_points):
            if selected_index != 0:
                self.points_x = np.insert(self.points_x, self.line_end_indices[selected_index]+1, np.linspace(x_start, x_stop, int(eval_points)))
                self.points_y = np.insert(self.points_y, self.line_end_indices[selected_index]+1, np.linspace(y_start, y_stop, int(eval_points)))
                self.line_end_indices = np.insert(self.line_end_indices, selected_index+1, self.line_end_indices[selected_index] + eval_points).astype(int) # insert the new value
            else:
                self.points_x = np.insert(self.points_x, self.line_end_indices[selected_index], np.linspace(x_start, x_stop, int(eval_points)))
                self.points_y = np.insert(self.points_y, self.line_end_indices[selected_index], np.linspace(y_start, y_stop, int(eval_points)))
                self.line_end_indices = np.insert(self.line_end_indices, selected_index+1, self.line_end_indices[selected_index] + eval_points - 1).astype(int) # insert the new value
            # for pal_ind: 2. add number of discretization points to the shifted values
            for n in range(selected_index+2, np.shape(self.line_end_indices)[0]):
                self.line_end_indices[n] = (self.line_end_indices[n] + eval_points).astype(int)
            
            
    def remove_selected_line(self, selected_index=[]):
        """
        Removes the selected line.

        Parameters
        ----------
        selected_index : int, optional
            Index of the selected line. The default is [].

        Returns
        -------
        None.

        """
        self.line_start_x = np.delete(self.line_start_x, selected_index)
        self.line_stop_x  = np.delete(self.line_stop_x, selected_index)
        self.line_start_y = np.delete(self.line_start_y, selected_index)
        self.line_stop_y  = np.delete(self.line_stop_y, selected_index)
        self.slope       = np.delete(self.slope, selected_index)
        self.intercept           = np.delete(self.intercept, selected_index)
        if selected_index != []: del self.plot_x[selected_index], self.plot_y[selected_index]
        if selected_index != 0:
            self.points_x = np.delete(self.points_x, np.s_[self.line_end_indices[selected_index]+1:self.line_end_indices[selected_index+1]+1])
            self.points_y = np.delete(self.points_y, np.s_[self.line_end_indices[selected_index]+1:self.line_end_indices[selected_index+1]+1])
        else:
            self.points_x = np.delete(self.points_x, np.s_[self.line_end_indices[selected_index]:self.line_end_indices[selected_index+1]+1])
            self.points_y = np.delete(self.points_y, np.s_[self.line_end_indices[selected_index]:self.line_end_indices[selected_index+1]+1])
        for n in range(selected_index+2, np.shape(self.line_end_indices)[0]):
            if selected_index != 0:
                self.line_end_indices[n] = int(self.line_end_indices[n] - (self.line_end_indices[n-1] - self.line_end_indices[n-2]))
            else:
                self.line_end_indices[n] = int(self.line_end_indices[n] - (self.line_end_indices[n-1] - self.line_end_indices[n-2]) - 1)
        self.line_end_indices = np.delete(self.line_end_indices, selected_index+1)


class PalcVenue(AudienceLines):
    """ Class that contains data of the drawn lines in a compatible format to
        compute the PALC algorithm. The lines depend on an object of the class
        :py:class:`AudienceLines`."""
    def __init__(self, line_start_x=None, line_stop_x=None, line_start_y=None, \
                 line_stop_y=None, slope=None, intercept=None, points_x=None, points_y=None, \
                 line_end_indices=None, plot_x=None, plot_y=None, segments=None, \
                 audience_segments=None):
        AudienceLines.__init__(self, line_start_x=[], line_stop_x=[], line_start_y=[], line_stop_y=[], \
                     slope=[], intercept=[], points_x=[], points_y=[], line_end_indices=[], plot_x=[], plot_y=[])
        self.segments        = [] if segments is None else segments
        self.audience_segments = [] if audience_segments is None else audience_segments
        
    def append_segment(self, fir_arr_x, sec_arr_x, fir_arr_y, sec_arr_y, fir_ind, sec_ind, m):
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
        self.segments.append(np.zeros([sec_ind - fir_ind, 3]))
        self.segments[m][:,0] = np.append(fir_arr_x, sec_arr_x)
        self.segments[m][:,1] = np.append(fir_arr_y, sec_arr_y)
        
    def collect_audience_segments(self, non_audience_lines):
        """
        Creates polygonal audience lines without non-audience lines.

        Parameters
        ----------
        non_audience_lines : AudienceLines
            The non-audience lines.

        Returns
        -------
        None.

        """
        if self.segments == []:
            return
        else:
            # create list of non-audience lines
            non_audience_segments = []
            for n in range (np.shape(non_audience_lines.line_start_x)[0]):
                non_audience_segments.append(np.array([[non_audience_lines.line_start_x[n], non_audience_lines.line_start_y[n], 0], \
                                     [non_audience_lines.line_stop_x[n], non_audience_lines.line_stop_y[n], 0]]))
            # create list of segments without non-audience lines
            non_audience_indices = []
            for m in range(len(self.segments)):
                for n in range(len(non_audience_segments)):
                    if np.array_equal(self.segments[m], non_audience_segments[n]):
                        non_audience_indices.append(m)
                if m not in non_audience_indices:
                    self.audience_segments.append(self.segments[m])


def slope_and_intercept(x_start, x_stop, y_start, y_stop):
    """
    Computes the slope and intersection point on the y-axis of line specified by
    start and stop point. The format of the line is: y = slope * x + intercept

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
    intercept : float
        Intersection point with y-axis.

    """
    if x_start != x_stop:
        slope = (y_start - y_stop) / (x_start - x_stop)
        intercept = (y_start - ((y_start - y_stop) / (x_start - x_stop)) * x_start)
    else: # vertical line: slope and intercept are infinity
        slope = 5000
        intercept = 5000
    return slope, intercept


def draw_condition_1(x_start, x_stop, y_start, y_stop):
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


def draw_condition_2(slope):
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


def draw_condition_3(x_start, x_stop, y_start, y_stop, slope, intercept, drawn_lines):
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
    intercept : float
        Intersection point on y-axis of drawn line.
    drawn_lines : obj [in]
        Contains data of the drawn lines.

    Returns
    -------
    bool
        True if condition has passed, otherwise False.

    """
    for n in range(np.shape(drawn_lines.line_start_x)[0]):
        xy_s2,_,_,px = line_intersection([x_start, y_start], [x_stop, y_stop], \
                                      [drawn_lines.line_start_x[n], drawn_lines.line_start_y[n]], [drawn_lines.line_stop_x[n], drawn_lines.line_stop_y[n]])
        # check if overlay exists
        if not px and intercept == drawn_lines.intercept[n]:
            # check if line is before or behind of tested line
            if x_start >= drawn_lines.line_stop_x[n] or x_stop <= drawn_lines.line_start_x[n]:
                pass
            else:
                return False
        else:
            pass
            #condition_3 = True        
    return True


def skipped_lines(drawn_lines, line_count):
    """
    Detects lines that will be totally avoided in the PALC computation. This
    must be considered in the PALC compatible PAL format.

    Parameters
    ----------
    drawn_lines : obj [in]
        Contains the drawn polygonal audience lines.
    line_count : int
        Number of audience lines to be considered.

    Returns
    -------
    jump : int
        Index of lines that are totally avoided in PALC computation.

    """
    # initialize jump index
    skipped = np.array([])
    # check which lines to be jumped for the cases [3,7,8,9]
    for n in range(line_count-1):
        for ni in range(n+1,line_count):
            if drawn_lines.line_start_x[n] >= drawn_lines.line_start_x[ni] and \
            not drawn_lines.line_stop_x[n] == drawn_lines.line_start_x[ni]:
                skipped = np.append(skipped, n)
    # check which lines to be jumped for the cases [1,4,5,7]
    n = 0
    while n < line_count-1:
        for m in range(n,line_count):
            if np.any(m == skipped):
                n += 1
            else:
                break
        ni = n+1
        n2 = 0
        while ni < line_count:
            if drawn_lines.line_stop_y[n] >= drawn_lines.line_stop_y[ni] and \
            (drawn_lines.line_stop_x[n] != drawn_lines.line_start_x[ni] and \
             drawn_lines.line_stop_y[n] != drawn_lines.line_start_y[ni]):
                skipped = np.append(skipped, ni)
                n2 += 1
            ni += 1
        n += 1 + n2
    # sort the index of lines to be jumped and delete possibly multiple entries
    skipped = np.array(list(set(np.sort(skipped)))).astype(int)
        
    return skipped


def suggest_venue(drawn_lines, non_audience_lines, sound_field_only_lines):
    """
    Suggests a venue slice depending on drawn audience lines. Creates a suggestion
    for non-audience lines and lines only considered for SFP. Therefore considers
    all possible cases explained in AES PAPER REF.

    Parameters
    ----------
    drawn_lines : obj [in]
        Contains data of the drawn audience lines by the user.
    non_audience_lines : obj [out]
        Contains data of the non-audience line.
    sound_field_only_lines : obj [out]
        Contains data of audience lines that are only used for SFP.

    Returns
    -------
    None.

    """
    # we consider 11 conditions, 9 condition: x1 <=> x2 * y1 <=> y2 + 2 special cases for x1 > x2 and y1 >= y2
    # get pal data to unpack
    line_start_x = drawn_lines.line_start_x
    line_stop_x  = drawn_lines.line_stop_x
    line_start_y = drawn_lines.line_start_y
    line_stop_y  = drawn_lines.line_stop_y
  
    line_count = np.shape(line_start_x)[0] # number of pal lines to be considered
    if line_count < 2: return non_audience_lines, sound_field_only_lines
    # check cases, jump has the index of totally avoided lines in PALC
    skipped = skipped_lines(drawn_lines, line_count)
    # add jump lines to sound_field_only_lines only lines
    sound_field_only_lines.append_line(line_start_x[skipped], line_stop_x[skipped], line_start_y[skipped], line_stop_y[skipped])
    sound_field_only_lines.line_end_indices = np.append(sound_field_only_lines.line_end_indices, skipped).astype(int)
    m = 0
    while m < line_count-1:
        nj = 0
        # cases when to leave while loop and which lines to be jumped
        for n in range(m, line_count-1):
            if np.any(n+1 == skipped):
                nj += 1
            else:
                break
        if m+1+nj >= line_count:
            break
        # same handling for cases: [1,4], [3,5,6], [9], special handling case [7,8] and no handling in case 2
        if line_stop_x[m] <= line_start_x[m+1+nj] and line_stop_y[m] > line_start_y[m+1+nj]: # case 1 and 4
            xy_s = line_intersection([line_stop_x[m], line_stop_y[m]], [line_stop_x[m+1+nj], line_stop_y[m]], \
                                         [line_start_x[m+1+nj], line_start_y[m+1+nj]], [line_stop_x[m+1+nj], line_stop_y[m+1+nj]])[0]
            xy_s2 = line_intersection([line_start_x[m], line_start_y[m]], [line_stop_x[m], line_stop_y[m]], \
                                         [line_start_x[m+1+nj], line_start_y[m+1+nj]], [line_stop_x[m+1+nj], line_stop_y[m+1+nj]])[0]
            if xy_s2[0] == line_stop_x[m] and xy_s2[1] == line_stop_y[m] \
               and line_start_x[m+1+nj] == xy_s2[0] == line_stop_x[m+1+nj]: # special case: line m+1 is vertical at stop pos of line m
                sound_field_only_lines.append_line(line_start_x[m+1+nj], xy_s2[0], line_start_y[m+1+nj], xy_s2[1])
                sound_field_only_lines.line_end_indices = np.append(sound_field_only_lines.line_end_indices, m+1+nj).astype(int)
                sound_field_only_lines.cut_lines.append(['after start', m+1+nj])
                m += 1+nj
                continue
                
            non_audience_lines.append_line(line_stop_x[m], xy_s[0], line_stop_y[m], xy_s[1])
            sound_field_only_lines.append_line(line_start_x[m+1+nj], xy_s[0], line_start_y[m+1+nj], xy_s[1])
            
            non_audience_lines.line_end_indices = np.append(non_audience_lines.line_end_indices, m).astype(int)
            sound_field_only_lines.line_end_indices = np.append(sound_field_only_lines.line_end_indices, m+1+nj).astype(int)
            sound_field_only_lines.cut_lines.append(['after start', m+1+nj])

        elif (line_stop_x[m] < line_start_x[m+1+nj] and line_stop_y[m] <= line_start_y[m+1+nj]) or \
             (line_stop_x[m] == line_start_x[m+1+nj] and line_stop_y[m] < line_start_y[m+1+nj]): # case 3, 5, 6
            non_audience_lines.append_line(line_stop_x[m], line_start_x[m+1+nj], line_stop_y[m], line_start_y[m+1+nj])

            non_audience_lines.line_end_indices = np.append(non_audience_lines.line_end_indices, m).astype(int)

        elif line_stop_x[m] > line_start_x[m+1+nj] and line_stop_y[m] < line_start_y[m+1+nj]: # case 9
            xy_s = line_intersection([line_start_x[m],line_start_y[m]], [line_stop_x[m], line_stop_y[m]], \
                                         [line_start_x[m+1+nj],line_start_y[m]], [line_start_x[m+1+nj], line_stop_y[m+1+nj]])[0]
            non_audience_lines.append_line(xy_s[0], line_start_x[m+1+nj], xy_s[1], line_start_y[m+1+nj])
            sound_field_only_lines.append_line(xy_s[0], line_stop_x[m], xy_s[1], line_stop_y[m])

            non_audience_lines.line_end_indices = np.append(non_audience_lines.line_end_indices, m).astype(int)
            sound_field_only_lines.line_end_indices = np.append(sound_field_only_lines.line_end_indices, m).astype(int)
            sound_field_only_lines.cut_lines.append(['before end', m])

        elif (line_stop_x[m] > line_start_x[m+1+nj] and line_stop_y[m] >= line_start_y[m+1+nj]): # case 8 and 7 (special handling)
            xy_s, _, _, p_ex = line_intersection([line_start_x[m],line_start_y[m]], [line_stop_x[m], line_stop_y[m]], \
                                         [line_start_x[m+1+nj],line_start_y[m+1+nj]], [line_stop_x[m+1+nj], line_stop_y[m+1+nj]])
            xy_s2, _, _, p_ex2 = line_intersection([line_stop_x[m],line_stop_y[m]], [line_stop_x[m], line_stop_y[m]+2], \
                                         [line_start_x[m+1+nj],line_start_y[m+1+nj]], [line_stop_x[m+1+nj], line_stop_y[m+1+nj]])
            xy_s3, _, _, p_ex2 = line_intersection([line_stop_x[m],line_stop_y[m]], [line_stop_x[m+1+nj], line_stop_y[m]], \
                                         [line_start_x[m+1+nj],line_start_y[m+1+nj]], [line_stop_x[m+1+nj], line_stop_y[m+1+nj]])
            xy_s4, _, _, p_ex2 = line_intersection([line_start_x[m],line_start_y[m]], [line_stop_x[m], line_stop_y[m]], \
                                         [line_start_x[m+1+nj],line_start_y[m+1+nj]], [line_start_x[m+1+nj], line_start_y[m+1+nj]+2])
            # case 7: intersection point is on the lines:
            if (line_start_x[m] <= xy_s[0] < line_stop_x[m]) and (line_start_y[m] <= xy_s[1] <= line_stop_y[m]) \
                and (line_start_x[m+1+nj] <= xy_s[0] <= line_stop_x[m+1+nj]) and (line_start_y[m+1+nj] < xy_s[1] <= line_stop_y[m+1+nj]) and p_ex == True:
                sound_field_only_lines.append_line([xy_s[0], line_start_x[m+1+nj]], [line_stop_x[m], xy_s[0]], [xy_s[1], line_start_y[m+1+nj]], [line_stop_y[m], xy_s[1]])

                sound_field_only_lines.line_end_indices = np.append(sound_field_only_lines.line_end_indices, [m, m+1+nj]).astype(int)
                sound_field_only_lines.cut_lines.append(['before end', m])
                sound_field_only_lines.cut_lines.append(['after start', m+1+nj])
             
            # case 8: start point of line m+1 is intersec point
            elif xy_s[0] == line_start_x[m+1+nj] and xy_s[1] == line_start_y[m+1+nj]:
                    sound_field_only_lines.append_line(line_start_x[m+1+nj], line_stop_x[m], line_start_y[m+1+nj], line_stop_y[m])

                    sound_field_only_lines.line_end_indices = np.append(sound_field_only_lines.line_end_indices, m).astype(int)
                    sound_field_only_lines.cut_lines.append(['before end', m])
                    
            # case 10: line m is over line m+1 
            elif (line_stop_y[m] >= xy_s2[1]):              
                # special case: intersec point is stop point of line m but line m+1 stays over line m
                #               could also be handled differently, but there seems not to be a real case like this
                if np.round(xy_s[0], decimals=6) == line_stop_x[m] and \
                   np.round(xy_s[1], decimals=6) == line_stop_y[m] and \
                   np.round(xy_s4[1],decimals=6) <  line_start_y[m+1+nj]:
                    if line_start_x[m] >= line_start_x[m+1+nj]:
                        m += 1+nj
                        continue
                    sound_field_only_lines.append_line(line_start_x[m+1+nj], xy_s[0], line_start_y[m+1+nj], xy_s[1])
                    sound_field_only_lines.line_end_indices = np.append(sound_field_only_lines.line_end_indices, m+1+nj).astype(int)
                    sound_field_only_lines.cut_lines.append(['after start', m+1+nj])
                    m += 1+nj
                    continue
                # special case: intersec point is stop point of line m
                if np.round(xy_s[0], decimals=6) == line_stop_x[m] and \
                   np.round(xy_s[1], decimals=6) == line_stop_y[m]:
                    
                    sound_field_only_lines.append_line(line_start_x[m+1+nj], xy_s3[0], line_start_y[m+1+nj], xy_s3[1])
                    sound_field_only_lines.line_end_indices = np.append(sound_field_only_lines.line_end_indices, m+1+nj).astype(int)
                    sound_field_only_lines.cut_lines.append(['after start', m+1+nj])
                    m += 1+nj
                    continue                
                    
                non_audience_lines.append_line(line_stop_x[m], xy_s3[0], line_stop_y[m], xy_s3[1])
                sound_field_only_lines.append_line(line_start_x[m+1+nj], xy_s3[0], line_start_y[m+1+nj], xy_s3[1])

                non_audience_lines.line_end_indices = np.append(non_audience_lines.line_end_indices, m).astype(int)
                sound_field_only_lines.line_end_indices = np.append(sound_field_only_lines.line_end_indices, m+1+nj).astype(int)
                sound_field_only_lines.cut_lines.append(['after start', m+1+nj])

            # case 11: line m is below line m+1
            elif (line_stop_y[m] < xy_s2[1]):
                if np.round(xy_s[0], decimals=6) == line_start_x[m+1+nj] and \
                   np.round(xy_s[1], decimals=6) == line_start_y[m+1+nj]: # special case: intersec point is start point of line m+1
                    sound_field_only_lines.append_line(xy_s4[0], line_stop_x[m], xy_s4[1], line_stop_y[m])
                    sound_field_only_lines.line_end_indices = np.append(sound_field_only_lines.line_end_indices, m).astype(int)
                    sound_field_only_lines.cut_lines.append(['before end', m])
                    m += 1+nj
                    continue

                non_audience_lines.append_line(xy_s4[0], line_start_x[m+1+nj], xy_s4[1], line_start_y[m+1+nj])
                sound_field_only_lines.append_line(xy_s4[0], line_stop_x[m], xy_s4[1], line_stop_y[m])

                non_audience_lines.line_end_indices = np.append(non_audience_lines.line_end_indices, m).astype(int)
                sound_field_only_lines.line_end_indices = np.append(sound_field_only_lines.line_end_indices, m).astype(int)
                sound_field_only_lines.cut_lines.append(['before end', m])
        m += 1+nj

    sound_field_only_lines.line_start_x, sound_field_only_lines.line_stop_x, sound_field_only_lines.line_start_y, sound_field_only_lines.line_stop_y \
    = sound_field_only_lines.line_start_x[sound_field_only_lines.line_end_indices.argsort()], sound_field_only_lines.line_stop_x[sound_field_only_lines.line_end_indices.argsort()], \
      sound_field_only_lines.line_start_y[sound_field_only_lines.line_end_indices.argsort()], sound_field_only_lines.line_stop_y[sound_field_only_lines.line_end_indices.argsort()]
    # plot list cannot be sorted, however it does not seem to be necessary
#    sound_field_only_lines.plot_x, sound_field_only_lines.plot_y = sound_field_only_lines.plot_x[sound_field_only_lines.line_end_indices.argsort()], sound_field_only_lines.plot_y[sound_field_only_lines.line_end_indices.argsort()]
    sound_field_only_lines.line_end_indices.sort()
    return


def fill_palc_venue(palc_venue, drawn_lines, non_audience_lines, sound_field_only_lines):
    """
    Fills a PalcVenue object that can be passed to the PALC
    computation. Needs the data that were generated in :any:`suggest_venue`.

    Parameters
    ----------
    palc_venue : obj [out]
        Contains the object to pass to PALC computation.
    drawn_lines : obj [in]
        Contains data of the drawn audience lines by the user.
    non_audience_lines : obj [in]
        Contains data of the non-audience line.
    sound_field_only_lines : obj [in]
        Contains data of audience lines that are only used for SFP.

    Returns
    -------
    None.

    """
    # get number of lines that are not included in PALC calculation at all
    skipped_line_indices = []
    for n in range(np.shape(drawn_lines.line_end_indices)[0]-1):
        for m in range(np.shape(sound_field_only_lines.line_end_indices)[0]):
            if drawn_lines.plot_x[n] == sound_field_only_lines.plot_x[m] and drawn_lines.plot_y[n] == sound_field_only_lines.plot_y[m]:
                skipped_line_indices.append(n)
    # remove audience lines that are not included in PALC calculation
    sound_field_only_indices = np.array([])
    if skipped_line_indices != []:
        for n in sound_field_only_lines.line_end_indices:
            if n not in skipped_line_indices:
                sound_field_only_indices = np.append(sound_field_only_indices, n).astype(int)
    else:
        sound_field_only_indices = np.array(sound_field_only_lines.line_end_indices)

    # total number of pal: shape(drawn_lines) - 1 + shape(non_audience_lines) - skipped_line_indices
    m = 0
    for n in range(np.shape(drawn_lines.line_end_indices)[0] - 1):
        # case 1: non-audience line is inserted
        if n-1 in non_audience_lines.line_end_indices:
            # index of inserted line in non_audience_lines object
            ind_a = np.argwhere(non_audience_lines.line_end_indices == n-1)[0,0]
            prev_line = 'nal'
            palc_venue.append_segment(non_audience_lines.line_start_x[ind_a], non_audience_lines.line_stop_x[ind_a], non_audience_lines.line_start_y[ind_a], non_audience_lines.line_stop_y[ind_a], 0, 2, m)
            m += 1
        else:
            prev_line = 'haspal'
        # check if next line is non-audience line
        if n in non_audience_lines.line_end_indices:
            ind_b       = np.argwhere(non_audience_lines.line_end_indices == n)[0,0]
            next_line   = 'nal'         
        else:
            next_line = 'haspal'    
        # get sfp index for current n, if there is one
        if sound_field_only_indices.size != 0: sound_field_only_index = np.argmin(np.abs(sound_field_only_indices - n))
        # case n is a jumped audience line
        if n in skipped_line_indices:
            pass
        
        # case 2: the pal line is cut off after the start, i.e., new start point is inserted
        elif ['after start', n] in sound_field_only_lines.cut_lines:
            cut_ind = np.argmin(np.abs(drawn_lines.points_x[drawn_lines.line_end_indices[n]:drawn_lines.line_end_indices[int(n+1)]] - sound_field_only_lines.line_stop_x[sound_field_only_index])) + drawn_lines.line_end_indices[n]                

            if prev_line == 'nal':
                palc_venue.append_segment(non_audience_lines.line_stop_x[ind_a], drawn_lines.points_x[cut_ind+1:drawn_lines.line_end_indices[n+1]+1], non_audience_lines.line_stop_y[ind_a], \
                                    drawn_lines.points_y[cut_ind+1:drawn_lines.line_end_indices[n+1]+1], cut_ind, drawn_lines.line_end_indices[n+1]+1, m)
            elif ['before end', n-1] in sound_field_only_lines.cut_lines:
                palc_venue.append_segment(sound_field_only_lines.line_start_x[sound_field_only_index-1], drawn_lines.points_x[cut_ind+1:drawn_lines.line_end_indices[n+1]+1], sound_field_only_lines.line_start_y[sound_field_only_index-1], \
                                    drawn_lines.points_y[cut_ind+1:drawn_lines.line_end_indices[n+1]+1], cut_ind, drawn_lines.line_end_indices[n+1]+1, m)
            else:
                cut_ind = np.argmin(np.abs(drawn_lines.points_x[drawn_lines.line_end_indices[n]+1:drawn_lines.line_end_indices[int(n+1)]] - sound_field_only_lines.line_stop_x[sound_field_only_index])) + drawn_lines.line_end_indices[n]+1
                palc_venue.append_segment(sound_field_only_lines.line_stop_x[sound_field_only_index], drawn_lines.points_x[cut_ind+1:drawn_lines.line_end_indices[n+1]+1], sound_field_only_lines.line_stop_y[sound_field_only_index], \
                                    drawn_lines.points_y[cut_ind+1:drawn_lines.line_end_indices[n+1]+1], cut_ind, drawn_lines.line_end_indices[n+1]+1, m)
            m += 1
            # if 'before end' will be called for current line:
            if ['before end', n] in sound_field_only_lines.cut_lines:
                # get cut_ind
                cut_ind = np.argmin(np.abs(palc_venue.segments[-1][:,0] - sound_field_only_lines.line_start_x[sound_field_only_index+1]))
                # cut pal that is part of only sfp line
                palc_venue.segments[-1] = palc_venue.segments[-1][:cut_ind,:]
                # append the intersection point to the pal
                palc_venue.segments[-1] = np.append(palc_venue.segments[-1], [sound_field_only_lines.line_start_x[sound_field_only_index+1], sound_field_only_lines.line_start_y[sound_field_only_index+1], \
                                             0]).reshape(int(np.shape(palc_venue.segments[-1])[0]+1),3)

        # case 3: the pal line is cut off before the end, i.e., new stop point is inserted
        elif ['before end', n] in sound_field_only_lines.cut_lines:
            cut_ind = np.argmin(np.abs(drawn_lines.points_x[drawn_lines.line_end_indices[n]:drawn_lines.line_end_indices[int(n+1)]] - sound_field_only_lines.line_start_x[sound_field_only_index])) + drawn_lines.line_end_indices[n]
            if m == 0:
                if next_line == 'nal':
                    palc_venue.append_segment(drawn_lines.points_x[drawn_lines.line_end_indices[n]:cut_ind], non_audience_lines.line_start_x[ind_b], drawn_lines.points_y[drawn_lines.line_end_indices[n]:cut_ind], \
                                        non_audience_lines.line_start_y[ind_b], drawn_lines.line_end_indices[n], cut_ind+1, m)
                else:
                    palc_venue.append_segment(drawn_lines.points_x[drawn_lines.line_end_indices[n]:cut_ind], sound_field_only_lines.line_start_x[sound_field_only_index], drawn_lines.points_y[drawn_lines.line_end_indices[n]:cut_ind], \
                                        sound_field_only_lines.line_start_y[sound_field_only_index], drawn_lines.line_end_indices[n], cut_ind+1, m)
            else:
                if next_line == 'nal':
                    palc_venue.append_segment(drawn_lines.points_x[drawn_lines.line_end_indices[n]+1:cut_ind], non_audience_lines.line_start_x[ind_b], drawn_lines.points_y[drawn_lines.line_end_indices[n]+1:cut_ind], \
                                        non_audience_lines.line_start_y[ind_b], drawn_lines.line_end_indices[n], cut_ind, m)
                else:
                    palc_venue.append_segment(drawn_lines.points_x[drawn_lines.line_end_indices[n]+1:cut_ind], sound_field_only_lines.line_start_x[sound_field_only_index], drawn_lines.points_y[drawn_lines.line_end_indices[n]+1:cut_ind], \
                                        sound_field_only_lines.line_start_y[sound_field_only_index], drawn_lines.line_end_indices[n], cut_ind, m)
            m += 1

        # case 4: created pal line is not changed
        elif [['after start', n],['before end', n]] not in sound_field_only_lines.cut_lines:
            if m == 0:
                palc_venue.append_segment(drawn_lines.points_x[drawn_lines.line_end_indices[n]:drawn_lines.line_end_indices[int(n+1)]], drawn_lines.points_x[drawn_lines.line_end_indices[int(n+1)]], \
                                                drawn_lines.points_y[drawn_lines.line_end_indices[n]:drawn_lines.line_end_indices[int(n+1)]], drawn_lines.points_y[drawn_lines.line_end_indices[int(n+1)]], \
                                                drawn_lines.line_end_indices[n], drawn_lines.line_end_indices[int(n+1)]+1, m)
            else:
                palc_venue.append_segment(drawn_lines.points_x[drawn_lines.line_end_indices[n]+1:drawn_lines.line_end_indices[int(n+1)]], drawn_lines.points_x[drawn_lines.line_end_indices[int(n+1)]], \
                                                drawn_lines.points_y[drawn_lines.line_end_indices[n]+1:drawn_lines.line_end_indices[int(n+1)]], drawn_lines.points_y[drawn_lines.line_end_indices[int(n+1)]], \
                                                drawn_lines.line_end_indices[n], drawn_lines.line_end_indices[int(n+1)], m)
            m += 1


def check_draw_conditions(x_start, x_stop, y_start, y_stop, drawn_lines):
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
    drawn_lines : obj [in]
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
    intercept : float
        Intersection point on the y-axis of the drawn line.
    message : str or None
        Error message the GUI shows if condition 1 or 2 failed, otherwise
        None. A failed condition 3 has no message.

    """
    # set condition3 default to True
    condition3 = True
    # calc slope and intersection on y-axis (intercept) of audience line
    slope, intercept = slope_and_intercept(x_start, x_stop, y_start, y_stop)

    # first condition (stop values are bigger or equal to the start values)
    condition1 = draw_condition_1(x_start, x_stop, y_start, y_stop)
    if (x_start == x_stop and y_start == y_stop):
        return False, False, True, slope, intercept, \
            'Error: Wrong Input. Points are not allowed'
    elif not condition1:
        return False, False, True, slope, intercept, \
            'Error: Stop values must be higher or equal than the start values and only positive values are allowed'
    # second condition (the slope of the line must be zero or positive):
    # f(x)=m*x+intercept, first step: calculate slope (m) and intercept and save them
    # check if x_start and x_stop are equal (vertical line)
    if condition1:
        condition2 = draw_condition_2(slope)
        if not condition2:
            return False, False, True, slope, intercept, \
                'Error: The slope of the drawn line must be positive'
    # avoid double clicking and and lines laying on each other
    if np.shape(drawn_lines.line_start_x)[0] > 0:
        condition3 = draw_condition_3(x_start, x_stop, y_start, y_stop,
                                          slope, intercept, drawn_lines)
    return condition1, condition2, condition3, slope, intercept, None
