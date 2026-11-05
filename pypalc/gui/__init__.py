"""GUI of pyPALC (Panel and panel-material-ui, with Bokeh figures).

:class:`pypalc.gui.app.PalcApp` is the GUI of one browser session.
"""
import functools
from pathlib import Path

from pypalc.core.errors import PalcError

#: Directory of the app (the one with ``main.py``).
APP_DIR = Path(__file__).resolve().parents[2]
#: JavaScript of the CSV download buttons.
DOWNLOAD_JS = APP_DIR / 'ext' / 'download.js'
#: JavaScript that downloads a text (the setup file of "Save Setup").
DOWNLOAD_TEXT_JS = APP_DIR / 'ext' / 'download_text.js'

#: Exceptions a GUI callback catches to show an error text instead of
#: failing: the errors of :mod:`pypalc.core` and those the legacy
#: calculation code raises on unusable input.
CALLBACK_ERRORS = (PalcError, ArithmeticError, AttributeError, LookupError,
                   TypeError, ValueError)


def skipped_while_applying_setup(callback):
    """Let a widget callback do nothing while a setup file is applied.

    :func:`pypalc.gui.setup_io.apply_setup` sets the widgets of a session
    and then does what their callbacks would do itself, once and in the
    right order. While it runs (``state.applying_setup``), the decorated
    callbacks return at once.

    Parameters
    ----------
    callback : callable
        Method of :class:`~pypalc.gui.app.PalcApp` or of a page (an object
        with the attribute ``app``).

    Returns
    -------
    callable
        The decorated method.
    """
    @functools.wraps(callback)
    def wrapper(self, *args, **kwargs):
        app = getattr(self, 'app', self)
        if app.state.applying_setup:
            return None
        return callback(self, *args, **kwargs)
    return wrapper
