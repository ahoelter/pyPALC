"""Read, write and use GUI models by their Bokeh-era property names.

The GUI is ported from Bokeh widgets to Panel / panel-material-ui widgets
one page at a time, keeping the attribute names of the models. The
functions of this module let the steps of ``tests/gui_steps.py`` and the
recording of ``tools/record_gui_reference.py`` use one set of property
names (those of the Bokeh GUI) for both kinds of models:

* ``title`` -> ``label`` (the label of a Panel input is the old title);
* ``label`` exists for Panel models only on buttons, toggles and
  downloads;
* ``active`` of a toggle -> ``value``; ``value`` does not exist on Panel
  buttons, toggles, downloads and button groups (the Bokeh models have
  none; the ``value`` of a Panel toggle is the old ``active``);
* ``text`` of a text pane (Typography, HTML, Markdown, Alert) ->
  ``object``;
* ``value`` of a file input: the base64 text of the file (``''`` before
  an upload);
* ``options``: the list of option labels (also for Bokeh button and
  checkbox groups, whose labels are ``labels``, and for Panel menus such
  as the page tabs, whose labels are their ``items``);
* ``selected`` of a data source: its selected indices.

author: Arne Hoelter
"""
from __future__ import annotations

import base64
from typing import Any, Optional

import panel as pn
import panel_material_ui as pmui
import param
from bokeh.core.property.descriptors import UnsetValueError
from bokeh.events import ButtonClick
from bokeh.models import FileInput

#: Panel models that have a ``label`` in the Bokeh sense (button text).
_LABELED = (pn.widgets.Button, pn.widgets.Toggle, pn.widgets.FileDownload,
            pmui.Button, pmui.Toggle, pmui.FileDownload)
_TOGGLES = (pn.widgets.Toggle, pmui.Toggle)
_TEXT_PANES = (pmui.Typography, pmui.Alert, pn.pane.HTML, pn.pane.Markdown)


def _is_panel(model: Any) -> bool:
    """True for a Panel (param based) model, False for a Bokeh model."""
    return isinstance(model, param.Parameterized)


def _name(model: Any, prop: str) -> Optional[str]:
    """Name of the model's property behind a Bokeh-era name, or None."""
    if _is_panel(model):
        if prop == 'selected' or (prop == 'label'
                                  and not isinstance(model, _LABELED)):
            return None
        if prop == 'value' and (isinstance(model, _LABELED)
                                or 'active' in model.param):
            return None
        name = {'title': 'label'}.get(prop, prop)
        if prop == 'options' and 'options' not in model.param:
            name = 'items'
        if prop == 'active' and isinstance(model, _TOGGLES):
            name = 'value'
        elif prop == 'text' and isinstance(model, _TEXT_PANES):
            name = 'object'
        return name if name in model.param else None
    properties = model.properties()
    if prop == 'options' and 'options' not in properties:
        prop = 'labels'
    return prop if prop in properties else None


def _label(option: Any) -> Any:
    """Label of a Bokeh option: ``(value, label)`` or the value itself."""
    return option[1] if isinstance(option, (tuple, list)) else option


def _item_label(item: Any) -> Any:
    """Label of a Panel menu item: ``(label, value)``, a dict with a
    ``label`` or the label itself."""
    if isinstance(item, (tuple, list)):
        return item[0]
    if isinstance(item, dict):
        return item['label']
    return item


def has(model: Any, prop: str) -> bool:
    """Whether the model has a property of the Bokeh-era name.

    Parameters
    ----------
    model : bokeh.model.Model or param.Parameterized
        A Bokeh model or a Panel widget or pane.
    prop : str
        Bokeh-era property name, e.g. ``'value'``, ``'active'``,
        ``'text'``, ``'label'``, ``'options'``, ``'selected'``.

    Returns
    -------
    bool
        True if :func:`read` can read it.
    """
    return _name(model, prop) is not None


def read(model: Any, prop: str) -> Any:
    """Value of a property given by its Bokeh-era name.

    Parameters
    ----------
    model : bokeh.model.Model or param.Parameterized
        A Bokeh model or a Panel widget or pane.
    prop : str
        Bokeh-era property name (see the module docstring for the
        mapping).

    Returns
    -------
    Any
        The value, not rounded or normalized: options as a list of labels,
        the value of a file input as base64 text, the selection of a
        source as its list of indices.

    Raises
    ------
    AttributeError
        If the model has no such property.
    """
    name = _name(model, prop)
    if name is None:
        raise AttributeError('%s has no property %r'
                             % (type(model).__name__, prop))
    if prop == 'selected':
        return model.selected.indices
    if _is_panel(model):
        value = getattr(model, name)
        if prop == 'options' and isinstance(value, dict):
            return list(value)
        if prop == 'options' and name == 'items':
            return [_item_label(item) for item in value]
        if prop == 'value' and isinstance(model, pn.widgets.FileInput):
            return base64.b64encode(value).decode() if value else ''
        return value
    if prop == 'value' and isinstance(model, FileInput):
        try:
            return model.value
        except UnsetValueError:  # Bokeh 3: no value before an upload
            return ''
    value = getattr(model, name)
    if prop == 'options':
        return [_label(option) for option in value]
    return value


def write(model: Any, prop: str, value: Any) -> None:
    """Set a property given by its Bokeh-era name, as a user would.

    Watchers (Panel) and ``on_change`` callbacks (Bokeh) fire as when the
    browser changes the property.

    Parameters
    ----------
    model : bokeh.model.Model or param.Parameterized
        A Bokeh model or a Panel widget or pane.
    prop : str
        Bokeh-era property name; ``'selected'`` sets the selected indices
        of a source; ``'active'`` of a Panel button group sets the
        ``value`` of the option at that index (``active`` follows it).
    value : Any
        The new value.

    Raises
    ------
    AttributeError
        If the model has no such property.
    """
    name = _name(model, prop)
    if name is None:
        raise AttributeError('%s has no property %r'
                             % (type(model).__name__, prop))
    if prop == 'selected':
        model.selected.indices = value
    elif prop == 'active' and _is_panel(model) and 'options' in model.param:
        # a button group (pmui.RadioButtonGroup): the browser sets the
        # value, ``active`` follows it (not the other way round)
        options = model.options
        model.value = list(options.values() if isinstance(options, dict)
                           else options)[value]
    else:
        setattr(model, name, value)


def upload(model: Any, data: bytes, filename: str) -> None:
    """Upload a file into a file input, as the browser does.

    Parameters
    ----------
    model : bokeh.models.FileInput or panel.widgets.FileInput
        The file input.
    data : bytes
        Content of the file.
    filename : str
        Name of the file.
    """
    if _is_panel(model):
        model.filename = filename
        model.value = data
    else:
        # Bokeh 3: FileInput.value is read-only in Python; the browser's
        # upload arrives through set_from_json
        model.set_from_json('filename', filename)
        model.set_from_json('value', base64.b64encode(data).decode())


def click(model: Any) -> None:
    """Click a button, as the browser does.

    Parameters
    ----------
    model : bokeh.models.Button or panel.widgets.Button
        The button. For a Bokeh button its Python ``on_click`` handlers
        run; for a Panel button its ``clicks`` count goes up by one.
    """
    if _is_panel(model):
        model.clicks += 1
    else:
        model._trigger_event(ButtonClick(model))
