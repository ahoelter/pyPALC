"""Bokeh GUI of pyPALC.

:class:`pypalc.gui.app.PalcApp` is the GUI of one browser session.
"""
from pathlib import Path

from pypalc.core.errors import PalcError

#: Directory of the Bokeh app (the one with ``main.py``).
APP_DIR = Path(__file__).resolve().parents[2]
#: Name of the app; Bokeh serves its static files under ``<name>/static``.
APP_NAME = APP_DIR.name
#: JavaScript of the CSV download buttons.
DOWNLOAD_JS = APP_DIR / 'ext' / 'download.js'

#: Exceptions a GUI callback catches to show an error text instead of
#: failing: the errors of :mod:`pypalc.core` and those the legacy
#: calculation code raises on unusable input.
CALLBACK_ERRORS = (PalcError, ArithmeticError, AttributeError, LookupError,
                   TypeError, ValueError)
