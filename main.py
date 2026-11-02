#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Arne Hoelter, Audio Communication Group, TU Berlin

Entry point of the pyPALC application, a **Loudspeaker Array Optimization
Software** to find the tilt angles of the *Loudspeaker Cabinets*.
The application works with a bokeh server that can be called
by typing "bokeh serve --show pyPALC" in the terminal on the directory that
contains the "pyPALC" folder.

Bokeh runs this script for every browser session; each session gets its own
:class:`pypalc.gui.app.PalcApp`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # make the pypalc package importable

from bokeh.io import curdoc  # noqa: E402

from pypalc.gui.app import PalcApp  # noqa: E402

PalcApp(curdoc()).show()
