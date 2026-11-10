"""Serving the pyPALC app: the view of one session and the server start
used by ``pypalc gui`` (and by ``main.py`` for ``panel serve pyPALC``)."""
from __future__ import annotations

from typing import List, Optional

import panel as pn
from panel.viewable import Viewable

from pypalc.gui.app import PalcApp
from pypalc.gui.style import bokeh_figure_theme

#: URL path of the app.
ROUTE = "pyPALC"
#: Title of the browser tab.
TITLE = "pyPALC"

# tabulator: Panel loads the JavaScript of a widget only if the first page
# shows one; the tables of pages 3 and 4 are on later pages
pn.extension('tabulator')


def create_view() -> Viewable:
    """Build the app of one browser session and return its view."""
    app = PalcApp()
    app.doc.theme = bokeh_figure_theme()
    return app.view()


def serve(port: int = 5006, address: Optional[str] = None,
          allow_websocket_origin: Optional[List[str]] = None,
          prefix: Optional[str] = None, show: bool = True) -> None:
    """Serve the app at ``/pyPALC`` until the process is stopped.

    Parameters
    ----------
    port : int
        Port of the server.
    address : str, optional
        Address to listen on (Panel's default: localhost).
    allow_websocket_origin : list of str, optional
        Hosts that may open websocket connections to the app.
    prefix : str, optional
        URL prefix of the app.
    show : bool
        Open the app in a browser.
    """
    # options that are not given stay out, so Panel uses its defaults
    options = {"address": address, "websocket_origin": allow_websocket_origin,
               "prefix": prefix}
    options = {key: value for key, value in options.items()
               if value is not None}
    pn.serve({ROUTE: create_view}, port=port, show=show, title=TITLE,
             **options)
