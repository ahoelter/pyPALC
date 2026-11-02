"""Theme of the Bokeh app: ``theme.yaml`` and ``templates/index.html``.

Bokeh serve reads both files from the app directory. :func:`write_theme`
writes them from the colors and fonts below (formerly the script
``create_layout.py``). Regenerate them after changing a color with::

    python -m pypalc.gui.theme

author: Arne Hoelter
"""
from __future__ import annotations

from pathlib import Path
from typing import List

#: Background color.
BACKGROUND_COLOR = '#323030'  # '#F5F5F5'  '#2F2F2F'
#: Text color.
TEXT_COLOR = 'white'
#: Selection color.
SELECTION_COLOR = 'darkred'
#: Link color: [color, hovered color].
LINK_COLORS = ('green', 'white')
#: If True, the figure font size is used for widgets and tables, too.
FIGURE_FONT_SIZE_FOR_ALL = False
#: Font size of figures.
FIGURE_FONT_SIZE = '10pt'
#: Font size of widgets and tables.
WIDGET_FONT_SIZE = '10pt'
#: Font (bokeh default is "Helvetica").
FONT = 'Helvetica'
#: Font style: normal, italic or bold.
FONT_STYLE = 'normal'


def _theme_yaml_lines() -> List[str]:
    """Lines of ``theme.yaml``."""
    bg_col, txt_col = BACKGROUND_COLOR, TEXT_COLOR
    ft_f, ft, ft_stl = FIGURE_FONT_SIZE, FONT, FONT_STYLE
    return [f'attrs:',
            f'    Figure:',
            f'        background_fill_color: "{bg_col}"',
            f'        border_fill_color: "{bg_col}"',
            f'        outline_line_color: "{txt_col}"',
            f'    Axis:',
            f'        axis_line_color: "{txt_col}"',
            f'        axis_label_text_color: "{txt_col}"',
            f'        major_label_text_color: "{txt_col}"',
            f'        major_label_text_font_size: "{ft_f}"',
            f'        major_label_text_font: "{ft}"',
            f'        major_tick_line_color: "{txt_col}"',
            f'        minor_tick_line_color: "{txt_col}"',
            f'        minor_tick_line_color: "{txt_col}"',
            f'        axis_label_text_font_size: "{ft_f}"',
            f'        axis_label_text_font: "{ft}"',
            f'        axis_label_text_font_style: "{ft_stl}"',
            f'    Grid:',
            f'        grid_line_dash: [6, 4]',
            f'        grid_line_alpha: .3',
            f'        grid_line_color: "{txt_col}"',
            f'        minor_grid_line_color: "{txt_col}"',
            f'        minor_grid_line_alpha: .1',
            f'    Title:',
            f'        background_fill_color: "{bg_col}"',
            f'        text_color: "{txt_col}"',
            f'        text_font_size: "{ft_f}"',
            f'        text_font: "{ft}"',
            f'    Widget:',
            f'        background: "{bg_col}"',
            f'    Label:',
            f'        background_fill_color: "{bg_col}"',
            f'        text_color: "{txt_col}"',
            f'        text_font: "{ft}"',
            f'    Legend:',
            f'        background_fill_color: "{bg_col}"',
            f'        border_line_color: "{bg_col}"',
            f'        title_text_color: "{txt_col}"',
            f'        title_text_font: "{ft}"',
            f'        label_text_color: "{txt_col}"',
            f'        label_text_font: "{ft}"',
            f'        label_text_font_size: "{ft_f}"',
            f'    ColorBar:',
            f'        background_fill_color: "{bg_col}"',
            f'        title_text_color: "{txt_col}"',
            f'        title_text_font_size: "{ft_f}"',
            f'        title_text_font: "{ft}"',
            f'        major_tick_line_color: "{txt_col}"',
            f'        minor_tick_line_color: "{txt_col}"',
            f'        major_label_text_color: "{txt_col}"',
            f'        major_label_text_font: "{ft}"',
            f'        label_standoff: 7',
            f'        major_label_text_font_size: "{ft_f}"']


def _index_html_lines() -> List[str]:
    """Lines of ``templates/index.html``."""
    bg_col, txt_col, sel_col = BACKGROUND_COLOR, TEXT_COLOR, SELECTION_COLOR
    link_col, ft_f, ft = LINK_COLORS, FIGURE_FONT_SIZE, FONT
    ft_w = FIGURE_FONT_SIZE if FIGURE_FONT_SIZE_FOR_ALL else WIDGET_FONT_SIZE
    return ['{% extends base %}',
            '{% block title %}PALC{% endblock %}',
            '{% block preamble %}',
            '<style>',
            '  a:link {',
            f'    color: {link_col[0]};',  # link color
            '    }',
            '  a:visited {',
            f'    color: {link_col[0]};',
            '    }',
            '  a:hover {',
            f'    color: {link_col[1]};',  # hovered link color
            '    }',
            '  a:active {',
            f'    color: {link_col[0]};',
            '    }',
            '  .slick-header-columns, .slick-headerrow-columns, .slick-footerrow-columns {',
            '    overflow: visible !important;',
            f'    border-left: 0px solid {bg_col} !important;',  # no borders in header
            f'    border-top: 2px solid {txt_col} !important;',
            f'    border-bottom: 0px solid {bg_col} !important;',
            f'    border-right: 0px solid {bg_col} !important;',
            '    }',
            '  .bk-root .slick-header-column.ui-state-default {',
            f'    border-left: 0px solid {bg_col} !important;',
            f'    border-top: 0px solid {txt_col} !important;',
            f'    border-bottom: 1px solid {txt_col} !important;',  # bottom rule in header
            f'    border-right: 0px solid {bg_col} !important;',
            '    }',
            '  .bk-root .bk-btn-success.bk-active {',
            '    background-color: darkred !important;',  # active success Toggle
            '    border-color: #403030;',
            '    }',
            '  .bk-root .bk-btn-success.bk-active:after {',
            '    content: "ning...";',  # "Run" -> "Running..." while the Toggle is active
            '    }',
            '  .bk {',
            f'    font-size: {ft_w} !important;',
            f'    font-family: {ft} !important;',
            '    }',
            '  .slick-cell, .slick-headerrow-column, .slick-footerrow-column {',
            f'    border-right: 0px solid {txt_col} !important;',  # no vertical borders
            f'    border-left: 0px solid {txt_col} !important;',
            '    }',
            '  .slick-header-column {',  # data table header
            f'    background-color: {bg_col} !important;',
            f'    background-image: none !important;',
            '    }',
            '  .slick-row {',  # data table rows
            f'    background-color: {bg_col} !important;',
            f'    background-image: none !important;',
            f'    color: {txt_col} !important;',
            f'    overflow: hidden;',
            '    }',
            '  .bk-cell-index {',  # data table index column
            f'    background-color: {bg_col} !important;',
            f'    background-image: none !important;',
            f'    color: {txt_col} !important;',
            '    }',
            '  .slick-cell.selected {',  # data table selection color
            f'    background-color: {sel_col} !important;',
            '    }',
            '  ::-webkit-scrollbar {',  # width of the browser scrollbar
            '    width: 8px;',
            '    }',
            '  body {',
            '    -webkit-font-smoothing: antialiased;',
            '    text-rendering: optimizeLegibility;',
            f'    color: {txt_col};',  # body text color
            f'    font-size: {ft_f};',  # body font size
            f'    background: {bg_col};',  # body background color
            '    }',
            '  ::-webkit-scrollbar-track {',  # scrollbar background
            f'    background: {bg_col}',
            '    }',
            '  ::-webkit-scrollbar-thumb {',  # scrollbar slider
            f'    background-color: {txt_col};',
            '    border-radius: 4px;',
            '    border: 2px;',
            '    }',
            '</style>',
            '{% endblock %}']


def write_theme(app_dir: Path) -> None:
    """Write ``theme.yaml`` and ``templates/index.html`` into the app directory.

    Parameters
    ----------
    app_dir : pathlib.Path
        Directory of the Bokeh app (the one with ``main.py``). The directory
        ``templates`` must exist in it.

    Returns
    -------
    None.
    """
    app_dir = Path(app_dir)
    for path, lines in ((app_dir / 'theme.yaml', _theme_yaml_lines()),
                        (app_dir / 'templates' / 'index.html',
                         _index_html_lines())):
        with open(path, 'w', encoding='utf-8', newline='\n') as file:
            file.write('\n'.join(lines))


if __name__ == '__main__':
    write_theme(Path(__file__).resolve().parents[2])
