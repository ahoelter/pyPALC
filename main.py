#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Arne Hoelter, Audio Communication Group, TU Berlin

Entry point of the pyPALC application, a **Loudspeaker Array Optimization
Software** to find the tilt angles of the *Loudspeaker Cabinets*.
The application is a Panel app; start it by typing
"panel serve pyPALC --show" in the terminal in the directory that contains
the "pyPALC" folder.

Panel runs this script for every browser session; each session gets its own
:class:`pypalc.gui.app.PalcApp`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # make the pypalc package importable

import panel as pn  # noqa: E402

from pypalc.gui.app import PalcApp  # noqa: E402
from pypalc.gui.style import bokeh_figure_theme  # noqa: E402

# tabulator: Panel loads the JavaScript of a widget only if the first page
# shows one; the tables of pages 3 and 4 are on later pages
pn.extension('tabulator')
app = PalcApp()
app.doc.theme = bokeh_figure_theme()
app.view().servable(title="pyPALC")
