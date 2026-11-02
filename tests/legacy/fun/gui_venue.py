#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Arne Hoelter

Helper functions in the venue slice creation.
"""

import numpy as np
import checkPAL
from gui_helper import update_text


def check_all_draw_conditions(x_start, x_stop, y_start, y_stop, \
                              created_pal, pal_error_info):
    """
    Checks all draw condition, when audience lines are created.

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
    pal_error_info : obj [out]
        Contains an error info if a condition failed - line could not be drawn.

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

    """
    # set condition3 default to True
    condition3 = True
    # calc slope and intersection on y-axis (b) of audience line    
    slope, b = checkPAL.get_slope_b(x_start, x_stop, y_start, y_stop)

    # first condition (stop values are bigger or equal to the start values)
    condition1 = checkPAL.pal_draw_condition_1(x_start, x_stop, y_start, y_stop)
    if (x_start == x_stop and y_start == y_stop):
        update_text(pal_error_info,'Error: Wrong Input. Points are not allowed')
        return False, False, True, slope, b
    elif not condition1:
        update_text(pal_error_info, \
                    'Error: Stop values must be higher or equal than the start values and only positive values are allowed')
        return False, False, True, slope, b
    # second condition (the slope of the line must be zero or positive):
    # f(x)=m*x+b, first step: calculate slope (m) and b and save them
    # check if x_start and x_stop are equal (vertical line)
    if condition1:
        condition2 = checkPAL.pal_draw_condition_2(slope)
        if not condition2:
            update_text(pal_error_info, 'Error: The slope of the drawn line must be positive')
            return False, False, True, slope, b
    # avoid double clicking and and lines laying on each other
    if np.shape(created_pal.xline_start)[0] > 0:
        condition3 = checkPAL.pal_draw_condition_3(x_start, x_stop, y_start, \
                                                   y_stop, slope, b, created_pal)
    return condition1, condition2, condition3, slope, b

