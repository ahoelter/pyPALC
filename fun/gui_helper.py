#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Arne Hoelter

Functions that help to organize the :doc:`main` and display in the GUI. The
functions do not directly correspond to a specific page in the GUI. They
are typically called by several functions / callbacks of the :doc:`main` script.
"""

from pypalc.core.venue import check_draw_conditions

def update_text(text_obj, text):
    """
    Set text in str-format to an object. The text will be displayed in the GUI.
    The text is usually an extra info or an error message, raised by an except
    case.

    Parameters
    ----------
    text_obj : obj [out]
        Object that contains text to display in the GUI.
    text : str
        Text to display in the GUI.

    Returns
    -------
    None.

    """
    setattr(text_obj,'text', text)


def check_all_draw_conditions(x_start, x_stop, y_start, y_stop, \
                              created_pal, pal_error_info):
    """
    Checks all draw condition, when audience lines are created. Calls
    :any:`check_draw_conditions` and shows its error message.

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
    condition1, condition2, condition3, slope, b, message = \
        check_draw_conditions(x_start, x_stop, y_start, y_stop, created_pal)
    if message is not None:
        update_text(pal_error_info, message)
    return condition1, condition2, condition3, slope, b
