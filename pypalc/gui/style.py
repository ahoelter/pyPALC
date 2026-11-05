"""Colors and themes of the GUI.

The look is that of the approved prototype (branch
``modern-design-prototype``, ``prototype/app.py``): Material UI in dark mode
with the colors of the former Bokeh GUI.

* :data:`MATERIAL_THEME` is the ``theme_config`` of the
  ``panel_material_ui.Page`` (:meth:`pypalc.gui.app.PalcApp.view`).
* :func:`bokeh_figure_theme` is the Bokeh theme of the figures; ``main.py``
  sets it on the document of every session.

This module holds constants only, no Panel or Bokeh objects (those belong
to a session).

author: Arne Hoelter
"""
from __future__ import annotations

from bokeh.themes import Theme

#: Background of the page.
BACKGROUND = '#323030'
#: Background of cards and figures.
SURFACE = '#3b3939'
#: Background of the header bar.
HEADER = '#272525'
#: Borders and dividers.
BORDER = '#4d4a4a'
#: Text.
TEXT = '#ffffff'
#: Secondary text (captions, axis tick labels).
TEXT_MUTED = '#b9b5b5'
#: Primary color (blue).
PRIMARY = '#428bca'
#: Success color (green buttons).
SUCCESS = '#5cb85c'
#: Danger color (red buttons).
DANGER = '#d9534f'
#: Warning color (orange buttons).
WARNING = '#f0ad4e'
#: Background of "PALC Run" while a calculation runs (darkred).
RUNNING = '#8b0000'
#: Links.
LINK = '#5cb85c'
#: Dash pattern of dashed lines: Bokeh 2's ``'dashed'``. Bokeh 3 draws the
#: name ``'dashed'`` as ``[3, 2]`` (its Python side still documents
#: ``[6]``, but sends the name), so glyphs use this pattern instead.
DASHED = [6]

#: Style sheet of the figures' toolbars: icons in the muted text color, and
#: every tool shown beside a short frame. Bokeh 3 makes a vertical toolbar
#: as high as the frame (inline style) and moves the tools that do not fit
#: into its height minus 15 px into an overflow menu (the short venue figure
#: of "Equal, variable height"); a toolbar as high as its tools plus that
#: reserve keeps them all, reaching below the frame as in Bokeh 2. It is at
#: least as high as the frame: Bokeh measures the toolbar again after each
#: layout of the figure, and a toolbar measured while its figure had no
#: height yet (hidden, or in a flex box) would otherwise keep the height of
#: the overflow button alone and never show its tools.
TOOLBAR_STYLESHEET = (
    f':host {{ --bokeh-icon-color: {TEXT_MUTED}; }}\n'
    ':host(.bk-left), :host(.bk-right) {\n'
    '  height: auto !important;\n'
    '  min-height: 100%;\n'
    '  padding-bottom: 16px;\n'
    '}')

#: Material UI theme of the page (``pmui.Page(theme_config=...)``).
MATERIAL_THEME = {
    'palette': {
        'mode': 'dark',
        'primary': {'main': PRIMARY},
        'success': {'main': SUCCESS, 'contrastText': TEXT},
        'error': {'main': DANGER, 'contrastText': TEXT},
        'warning': {'main': WARNING, 'contrastText': TEXT},
        'background': {'default': BACKGROUND, 'paper': BACKGROUND},
        'text': {'primary': TEXT, 'secondary': TEXT_MUTED},
        'divider': BORDER,
    },
    'shape': {'borderRadius': 8},
    'typography': {'fontSize': 13,
                   'button': {'textTransform': 'none', 'fontWeight': 500}},
    'components': {
        'MuiAppBar': {'styleOverrides': {'root': {
            'backgroundColor': HEADER, 'backgroundImage': 'none',
            'borderBottom': f'1px solid {BORDER}', 'boxShadow': 'none'}}},
        'MuiPaper': {'styleOverrides': {'root': {'backgroundImage': 'none'}}},
        'MuiButton': {'defaultProps': {'disableElevation': True}},
        'MuiLink': {'styleOverrides': {'root': {'color': LINK}}},
    },
}


def bokeh_figure_theme() -> Theme:
    """Bokeh theme of the figures: dark plot area, muted axes, faint
    dashed grid, toolbars of :data:`TOOLBAR_STYLESHEET`.

    Set on the document, it also keeps panel-material-ui from restyling
    these model types in the browser.

    Returns
    -------
    bokeh.themes.Theme
        A new theme.
    """
    plot = {'background_fill_color': SURFACE, 'border_fill_color': SURFACE,
            'outline_line_color': BORDER}
    return Theme(json={'attrs': {
        'Plot': plot, 'Figure': plot,
        'Axis': {'axis_line_color': TEXT_MUTED,
                 'major_tick_line_color': TEXT_MUTED,
                 'minor_tick_line_color': TEXT_MUTED,
                 'major_label_text_color': TEXT_MUTED,
                 'major_label_text_font_size': '11px',
                 'axis_label_text_color': TEXT,
                 'axis_label_text_font_size': '12px',
                 'axis_label_text_font_style': 'normal'},
        'Grid': {'grid_line_color': TEXT, 'grid_line_alpha': 0.12,
                 'grid_line_dash': [6, 4], 'minor_grid_line_color': TEXT,
                 'minor_grid_line_alpha': 0.04},
        'Title': {'text_color': TEXT, 'text_font_size': '13px',
                  'text_font_style': 'normal'},
        'Toolbar': {'stylesheets': [TOOLBAR_STYLESHEET]},
        'Legend': {'background_fill_color': HEADER,
                   'background_fill_alpha': 0.8, 'border_line_color': None,
                   'label_text_color': TEXT, 'label_text_font_size': '12px'},
    }})
