"""Building blocks of the pages in the look of the prototype.

Cards with titles (:func:`card`), captions (:func:`caption`), outlined
inputs and selections (:func:`text_input`, :func:`select`), sliders
(:func:`slider`), full-width buttons (:func:`button`), download buttons
(:func:`download`), tables (:func:`table`) and the alert that shows an
error or note text (:func:`message`), plus the layout of a page with a
side column and a main area (:func:`page_row`). Every function returns a
new object: Panel objects belong to one session.

author: Arne Hoelter
"""
from __future__ import annotations

from typing import Any, Callable

import panel as pn
import panel_material_ui as pmui

from pypalc.gui.style import BORDER, SURFACE, TEXT, TEXT_MUTED

#: Gap between cards and columns in px.
GAP = 16
#: Preferred width of the side column of a page in px; it shrinks on
#: narrow windows.
SIDE_WIDTH = 380
#: Smallest width of the main area of a page before it moves below the
#: side column, in px.
MAIN_MIN_WIDTH = 560


def card(title: str, *objects: Any, **params: Any) -> pmui.Card:
    """A titled card holding a step group or a figure.

    Parameters
    ----------
    title : str
        Title of the card.
    *objects
        Content of the card.
    **params
        Further parameters of the card; by default it stretches to the
        width of its container.

    Returns
    -------
    panel_material_ui.Card
    """
    return pmui.Card(*objects, **{
        'title': title, 'collapsible': False, 'outlined': True,
        'elevation': 0, 'margin': (0, 0, GAP, 0),
        'title_variant': 'subtitle1', 'header_background': SURFACE,
        'sizing_mode': 'stretch_width',
        'sx': {'backgroundColor': SURFACE, 'border': f'1px solid {BORDER}',
               'borderRadius': '12px'},
        **params})


def caption(text: str, **params: Any) -> pn.pane.HTML:
    """A short explanation in muted small text.

    Parameters
    ----------
    text : str
        The text (HTML).
    **params
        Further parameters of the pane.

    Returns
    -------
    panel.pane.HTML
    """
    return pn.pane.HTML(
        f'<div style="color:{TEXT_MUTED};font-size:12px;'
        f'line-height:1.45">{text}</div>',
        **{'sizing_mode': 'stretch_width', 'margin': (2, 8), **params})


def text_input(label: str, value: str, **params: Any) -> pmui.TextInput:
    """An outlined text input that stretches to the width of its container.

    Parameters
    ----------
    label : str
        Label of the input (with its step number).
    value : str
        Initial text.
    **params
        Further parameters of the input.

    Returns
    -------
    panel_material_ui.TextInput
    """
    return pmui.TextInput(label=label, value=value, **{
        'sizing_mode': 'stretch_width', 'size': 'small', **params})


def select(label: str, options: Any, value: Any,
           **params: Any) -> pmui.Select:
    """An outlined selection that stretches to the width of its container.

    Parameters
    ----------
    label : str
        Label of the selection (with its step number).
    options : list or dict
        The options; a dict maps the shown label to the value.
    value : Any
        Selected value.
    **params
        Further parameters of the selection.

    Returns
    -------
    panel_material_ui.Select
    """
    return pmui.Select(label=label, options=options, value=value, **{
        'sizing_mode': 'stretch_width', 'size': 'small', **params})


def _table_stylesheet() -> str:
    """Style sheet of a :func:`table`: the colors of the cards."""
    return f"""
        .tabulator {{ background: transparent; border: none;
                      font-size: 13px; }}
        .tabulator .tabulator-header,
        .tabulator .tabulator-header .tabulator-col {{
            background: {SURFACE}; color: {TEXT_MUTED};
            border-color: {BORDER}; }}
        .tabulator .tabulator-tableholder {{ background: {SURFACE};
                                             overflow-x: hidden !important; }}
        .tabulator-row, .tabulator-row.tabulator-row-even,
        .tabulator-row.tabulator-row-odd {{
            background: {SURFACE} !important; color: {TEXT};
            border-bottom: 1px solid {BORDER}; }}
        .tabulator-row .tabulator-cell,
        .tabulator .tabulator-header .tabulator-col {{
            border-right: none !important;
            padding: 4px 10px !important; }}
        .tabulator .tabulator-header .tabulator-col {{
            font-size: 12px; font-weight: 500; }}
        .tabulator-col-content {{ padding: 6px 10px !important; }}
        .tabulator-col-title {{ padding-right: 0 !important; }}
        .tabulator-col-sorter {{ display: none !important; }}
        .tabulator .tabulator-header {{
            border-bottom: 1px solid {BORDER}; }}
        .tabulator-placeholder span,
        .tabulator-placeholder-contents {{
            color: {TEXT_MUTED} !important; font-size: 12px !important;
            font-weight: 400 !important; padding: 14px !important; }}
    """


def table(value: Any, titles: dict, widths: dict,
          **params: Any) -> pn.widgets.Tabulator:
    """A read-only table in the colors of the cards (no index, no sorting,
    no selection) that stretches to the width of its container.

    Parameters
    ----------
    value : pandas.DataFrame
        The table.
    titles : dict
        Column name -> shown title. The first column is centered, the
        others are right-aligned.
    widths : dict
        Column name -> width in px; columns without a width share the rest
        of the width.
    **params
        Further parameters of the table.

    Returns
    -------
    panel.widgets.Tabulator
    """
    first, *others = list(titles)
    align = dict({first: 'center'}, **{name: 'right' for name in others})
    return pn.widgets.Tabulator(value, **{
        'show_index': False, 'disabled': True, 'sortable': False,
        'selectable': False, 'layout': 'fit_columns', 'theme': 'midnight',
        'titles': titles, 'widths': widths, 'text_align': align,
        'header_align': align,
        'sizing_mode': 'stretch_width', 'margin': (4, 8),
        'stylesheets': [_table_stylesheet()], **params})


def button(label: str, color: str, **params: Any) -> pmui.Button:
    """A full-width button.

    Parameters
    ----------
    label : str
        Text of the button (with its step number).
    color : str
        ``'success'``, ``'danger'``, ``'warning'``, ``'primary'`` or
        ``'default'`` (the button types of the Bokeh GUI;
        panel-material-ui shows ``'danger'`` in the error color and keeps
        its deprecated ``button_type`` equal to ``color``).
    **params
        Further parameters of the button.

    Returns
    -------
    panel_material_ui.Button
    """
    return pmui.Button(label=label, color=color, **{
        'sizing_mode': 'stretch_width', 'margin': (4, 8), **params})


def slider(label: str, value: float, start: float, end: float,
           step: float, **params: Any) -> pmui.FloatSlider:
    """A slider that stretches to the width of its container.

    Parameters
    ----------
    label : str
        Label of the slider (with its step number).
    value, start, end, step : float
        Value, range and step.
    **params
        Further parameters of the slider.

    Returns
    -------
    panel_material_ui.FloatSlider
    """
    return pmui.FloatSlider(
        label=label, value=value, start=start, end=end, step=step, **{
            'sizing_mode': 'stretch_width', 'margin': (4, 16), **params})


def download(label: str, filename: str, callback: Callable[[], Any],
             later: Callable[[Callable[[], None]], None],
             **params: Any) -> pmui.FileDownload:
    """A full-width outlined button that downloads a file made on click.

    A click calls ``callback``, which returns the file as a file-like
    object (``io.StringIO`` or ``io.BytesIO``), or None to download
    nothing (e.g. when the file cannot be made; the callback then shows
    why). The browser downloads the file when the data of the button
    change, so the data are emptied again after each download (through
    ``later``, after the browser got them): a second click downloads the
    file again, also if it did not change.

    Parameters
    ----------
    label : str
        Text of the button.
    filename : str
        Name of the downloaded file.
    callback : callable
        Makes the file; without arguments.
    later : callable
        :meth:`~.app.PalcApp.later` of the session.
    **params
        Further parameters of the button.

    Returns
    -------
    panel_material_ui.FileDownload
    """
    widget = pmui.FileDownload(
        label=label, filename=filename, callback=callback, **{
            'color': 'success', 'variant': 'outlined', 'icon': 'download',
            'sizing_mode': 'stretch_width', 'margin': (4, 8), **params})
    send = widget._sync_data

    def sync_data(fileobj: Any) -> None:
        # Panel's FileDownload raises for None; nothing to send then
        if fileobj is None:
            return
        send(fileobj)
        later(lambda: widget.param.update(data=None))
    widget._sync_data = sync_data
    return widget


def _severity(text: str) -> str:
    """Severity of an alert that shows a message text."""
    start = text.strip().lower()
    if start.startswith(('error', 'couldnt')):
        return 'error'
    if start.startswith('warning'):
        return 'warning'
    return 'info'


def message(**params: Any) -> pmui.Alert:
    """An alert that shows a message text and hides while it is empty.

    Set its ``object`` to the text. Texts starting with "Error" or
    "Couldnt" are shown as errors, "Warning" as warnings, others as notes.

    Parameters
    ----------
    **params
        Further parameters of the alert.

    Returns
    -------
    panel_material_ui.Alert
    """
    alert = pmui.Alert(object='', severity='info', visible=False, **{
        'sizing_mode': 'stretch_width', 'margin': (4, 8), **params})

    def show(event) -> None:
        text = event.new or ''
        alert.param.update(severity=_severity(text),
                           visible=bool(text.strip()))
    alert.param.watch(show, 'object')
    return alert


def page_row(side: Any, main: Any, side_width: int = SIDE_WIDTH
             ) -> pn.FlexBox:
    """A page with a side column of cards and a main area.

    The side column is ``side_width`` wide and may shrink; the main area
    takes the rest of the width. If the window is too narrow for both
    (main area below :data:`MAIN_MIN_WIDTH`), the main area moves below the
    side column. The sizing mode and the styles of both are set here
    (styles they had before are replaced).

    Parameters
    ----------
    side : Viewable
        The side column (cards with the steps).
    main : Viewable
        The main area (figures).
    side_width : int, optional
        Width of the side column in px, :data:`SIDE_WIDTH` by default.

    Returns
    -------
    panel.FlexBox
    """
    side.param.update(sizing_mode='stretch_width', styles={
        'flex': f'1 1 {side_width}px', 'max-width': f'{side_width}px'})
    main.param.update(sizing_mode='stretch_width', styles={
        'flex': f'1000 1 {MAIN_MIN_WIDTH}px', 'min-width': '0'})
    return pn.FlexBox(side, main, flex_wrap='wrap', gap=f'{GAP}px',
                      align_items='flex-start', sizing_mode='stretch_width',
                      margin=(0, GAP))
