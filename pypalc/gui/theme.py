"""Theme of the Bokeh app: ``theme.yaml`` and ``templates/index.html``.

Bokeh serve reads both files from the app directory. :func:`write_theme`
writes them from the colors and fonts below (formerly the script
``create_layout.py``). Regenerate them after changing a color with::

    python -m pypalc.gui.theme

Bokeh 3 renders every widget in its own shadow DOM, which the CSS of the
page does not reach. So ``templates/index.html`` styles the page
(:func:`_index_html_lines`), and ``theme.yaml`` holds, besides the
attributes of the figures (:func:`_theme_attrs`), the style sheets of the
widgets by model type (:func:`_stylesheets`).

author: Arne Hoelter
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

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
#: Color of the toolbar icons (the grey of the Bokeh 2 icons).
ICON_COLOR = '#a4a9ac'


def _theme_attrs() -> Dict[str, List[str]]:
    """Theme attributes of the figures: ``attribute: value`` lines by the
    name of the Bokeh model type."""
    bg_col, txt_col = BACKGROUND_COLOR, TEXT_COLOR
    ft_f, ft, ft_stl = FIGURE_FONT_SIZE, FONT, FONT_STYLE
    return {
        'Plot': [
            f'background_fill_color: "{bg_col}"',
            f'border_fill_color: "{bg_col}"',
            f'outline_line_color: "{txt_col}"'],
        'Axis': [
            f'axis_line_color: "{txt_col}"',
            f'axis_label_text_color: "{txt_col}"',
            f'major_label_text_color: "{txt_col}"',
            f'major_label_text_font_size: "{ft_f}"',
            f'major_label_text_font: "{ft}"',
            f'major_tick_line_color: "{txt_col}"',
            f'minor_tick_line_color: "{txt_col}"',
            f'minor_tick_line_color: "{txt_col}"',
            f'axis_label_text_font_size: "{ft_f}"',
            f'axis_label_text_font: "{ft}"',
            f'axis_label_text_font_style: "{ft_stl}"'],
        'Grid': [
            'grid_line_dash: [6, 4]',
            'grid_line_alpha: .3',
            f'grid_line_color: "{txt_col}"',
            f'minor_grid_line_color: "{txt_col}"',
            'minor_grid_line_alpha: .1'],
        'Title': [
            f'background_fill_color: "{bg_col}"',
            f'text_color: "{txt_col}"',
            f'text_font_size: "{ft_f}"',
            f'text_font: "{ft}"'],
        'Label': [
            f'background_fill_color: "{bg_col}"',
            f'text_color: "{txt_col}"',
            f'text_font: "{ft}"'],
        'Legend': [
            f'background_fill_color: "{bg_col}"',
            f'border_line_color: "{bg_col}"',
            f'title_text_color: "{txt_col}"',
            f'title_text_font: "{ft}"',
            f'label_text_color: "{txt_col}"',
            f'label_text_font: "{ft}"',
            f'label_text_font_size: "{ft_f}"'],
        'ColorBar': [
            f'background_fill_color: "{bg_col}"',
            f'title_text_color: "{txt_col}"',
            f'title_text_font_size: "{ft_f}"',
            f'title_text_font: "{ft}"',
            f'major_tick_line_color: "{txt_col}"',
            f'minor_tick_line_color: "{txt_col}"',
            f'major_label_text_color: "{txt_col}"',
            f'major_label_text_font: "{ft}"',
            'label_standoff: 7',
            f'major_label_text_font_size: "{ft_f}"'],
    }


def _theme_yaml_lines() -> List[str]:
    """Lines of ``theme.yaml``: the attributes of :func:`_theme_attrs` and
    the style sheets of :func:`_stylesheets`, one entry per model type."""
    attrs = {model: list(lines) for model, lines in _theme_attrs().items()}
    for model, css in _stylesheets().items():
        attrs.setdefault(model, []).extend(
            ['stylesheets:', '    - |'] + ['        ' + line for line in css])
    lines = ['attrs:']
    for model, model_lines in attrs.items():
        lines += [f'    {model}:'] + ['        ' + line for line in model_lines]
    return lines


def _rule(selector: str, *declarations: str) -> List[str]:
    """Lines of one CSS rule.

    Parameters
    ----------
    selector : str
        Selector of the rule.
    *declarations : str
        Declarations ``property: value``, without the semicolon.

    Returns
    -------
    list of str
        The rule; the declarations are indented by two spaces.
    """
    return [selector + ' {'] + ['  %s;' % decl for decl in declarations] \
        + ['  }']


def _link_rules() -> List[str]:
    """CSS of links, for the page and for the links in ``Div`` widgets."""
    link_col = LINK_COLORS
    return (_rule('a:link', f'color: {link_col[0]}')  # link color
            + _rule('a:visited', f'color: {link_col[0]}')
            + _rule('a:hover', f'color: {link_col[1]}')  # hovered link color
            + _rule('a:active', f'color: {link_col[0]}'))


def _stylesheets() -> Dict[str, List[str]]:
    """CSS of the widgets, by Bokeh model type.

    Bokeh 3 renders each widget in its own shadow DOM, which the CSS of the
    page does not reach. ``theme.yaml`` gives these lines to the property
    ``stylesheets`` of the model type, and Bokeh adds them to the shadow
    root of every widget of that type or of a subclass. The style sheets
    of a type replace those of its base classes; they are not merged.

    Besides the rules of the Bokeh 2 template, the style sheets restore
    Bokeh 2 defaults that Bokeh 3 changed (tabs, inputs, file input, table
    header and paragraphs in ``Div`` widgets). Bokeh 2 rounded the height of
    every widget up to whole pixels, Bokeh 3 does not: with the 10pt font a
    line of text is 19.05 px high, and a column of buttons drifts up by
    about a pixel per button. So buttons, tabs and paragraphs round their
    line height up, as Bokeh 2 did.

    Returns
    -------
    dict of str to list of str
        CSS lines by the name of the Bokeh model type.
    """
    bg_col, txt_col, sel_col = BACKGROUND_COLOR, TEXT_COLOR, SELECTION_COLOR
    # a line of text as high as in Bokeh 2: rounded up to whole pixels
    whole_pixel_line = 'line-height: round(up, var(--line-height-computed), 1px)'
    buttons = _rule('.bk-btn', whole_pixel_line)
    return {
        # links and paragraphs in the text of Div widgets (header, help
        # pages); Bokeh 3 sets the margin of paragraphs to 0
        'Div': _link_rules() + _rule('p', 'margin: 1em 0'),
        # dark data tables with a white rule above and below the header
        'DataTable': (
            _rule('.slick-header-columns, .slick-headerrow-columns, '
                  '.slick-footerrow-columns',
                  'overflow: visible !important',
                  f'border-left: 0px solid {bg_col} !important',  # no borders in header
                  f'border-top: 2px solid {txt_col} !important',
                  f'border-bottom: 0px solid {bg_col} !important',
                  f'border-right: 0px solid {bg_col} !important')
            + _rule('.slick-header-column.ui-state-default',
                    f'border-left: 0px solid {bg_col} !important',
                    f'border-top: 0px solid {txt_col} !important',
                    f'border-bottom: 1px solid {txt_col} !important',  # bottom rule in header
                    f'border-right: 0px solid {bg_col} !important')
            + _rule('.slick-cell, .slick-headerrow-column, '
                    '.slick-footerrow-column',
                    f'border-right: 0px solid {txt_col} !important',  # no vertical borders
                    f'border-left: 0px solid {txt_col} !important')
            + _rule('.slick-header, .slick-header-columns',  # white in Bokeh 3
                    'background-color: transparent !important')
            + _rule('.slick-header-column',  # data table header
                    f'background-color: {bg_col} !important',
                    'background-image: none !important')
            + _rule('.slick-row',  # data table rows
                    f'background-color: {bg_col} !important',
                    'background-image: none !important',
                    f'color: {txt_col} !important',
                    'overflow: hidden')
            + _rule('.bk-cell-index',  # data table index column
                    f'background-color: {bg_col} !important',
                    'background-image: none !important',
                    f'color: {txt_col} !important')
            + _rule('.slick-cell.selected',  # data table selection color
                    f'background-color: {sel_col} !important')),
        'Button': buttons,
        'RadioButtonGroup': buttons,
        # "Run" -> "Running..." on a dark red button while the Toggle is active
        'Toggle': buttons + (
            _rule('.bk-btn-success.bk-active',
                  'background-color: darkred !important',  # active success Toggle
                  'border-color: #403030')
            + _rule('.bk-btn-success.bk-active:after',
                    'content: "ning..."')),
        # tabs of Bokeh 2: grey text of the inactive tabs, light grey lines
        'Tabs': (
            _rule(':host', '--divider-color: #e6e6e6')
            + _rule('.bk-tab', 'color: #666666', whole_pixel_line)
            + _rule('.bk-tab:hover', 'background-color: #f2f2f2')
            + _rule('.bk-tab.bk-active',
                    'color: #4d4d4d',
                    'background-color: white',
                    'border-color: #e6e6e6')),
        'Paragraph': _rule(':host', whole_pixel_line),
        # figures keep their size, as in Bokeh 2: Bokeh 3 shrinks a figure
        # with a fixed frame (the venue figure) to the width of its column
        'Plot': _rule(':host', 'max-width: none'),
        # text inputs and selects of Bokeh 2
        'InputWidget': (
            _rule(':host',
                  '--input-min-height: 31px',
                  '--border-color: #ccc',
                  '--input-focus-border-color: #66afe9',
                  '--input-focus-halo-color: rgba(102, 175, 233, 0.6)',
                  '--disabled-background-color: #eee',
                  '--placeholder-color: #999')),
        # the browser's file input, as in Bokeh 2: no box around it, so
        # that "No file selected." shows next to the button
        'FileInput': (
            _rule('input[type=file].bk-input',
                  'background-color: transparent',
                  'border: none',
                  'border-radius: 0',
                  'min-height: 0',
                  'padding: 0')
            + _rule('input[type=file]::file-selector-button',
                    'all: revert')),
    }


def _index_html_lines() -> List[str]:
    """Lines of ``templates/index.html``.

    The template styles the page itself: body, scrollbars and links
    outside widgets. Its style comes after Bokeh's own (block
    ``postamble``), which sets the margin of the body to 0 and its height to
    the window's. It does not reach into the shadow DOM of the widgets, but
    custom properties are inherited into it, so Bokeh's variables
    ``--bokeh-font-size``, ``--bokeh-base-font`` and ``--bokeh-icon-color``
    set the font of every widget and the color of the toolbar icons.
    """
    bg_col, txt_col = BACKGROUND_COLOR, TEXT_COLOR
    ft_f, ft = FIGURE_FONT_SIZE, FONT
    ft_w = FIGURE_FONT_SIZE if FIGURE_FONT_SIZE_FOR_ALL else WIDGET_FONT_SIZE
    css = (_link_rules()
           + _rule('::-webkit-scrollbar',  # width of the browser scrollbar
                   'width: 8px')
           + _rule('html',  # as in Bokeh 2: as high as the body with its margin
                   'height: auto')
           + _rule('body',
                   '-webkit-font-smoothing: antialiased',
                   'text-rendering: optimizeLegibility',
                   f'color: {txt_col}',  # body text color
                   f'font-size: {ft_f}',  # body font size
                   f'background: {bg_col}',  # body background color
                   'height: auto',  # as in Bokeh 2: no scrolling of short pages
                   'margin: 8px',  # the browser's margin, as in Bokeh 2
                   f'--bokeh-font-size: {ft_w}',  # widget font size
                   f'--bokeh-base-font: {ft}',  # widget font
                   f'--bokeh-icon-color: {ICON_COLOR}')  # toolbar icons
           + _rule('::-webkit-scrollbar-track',  # scrollbar background
                   f'background: {bg_col}')
           + _rule('::-webkit-scrollbar-thumb',  # scrollbar slider
                   f'background-color: {txt_col}',
                   'border-radius: 4px',
                   'border: 2px'))
    return (['{% extends base %}',
             '{% block title %}PALC{% endblock %}',
             '{% block postamble %}',
             '<style>']
            + ['  ' + line for line in css]
            + ['</style>',
               '{% endblock %}'])


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
