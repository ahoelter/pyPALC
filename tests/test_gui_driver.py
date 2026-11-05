"""The GUI test driver reads and writes Bokeh and Panel models alike."""
import base64

import panel as pn
import panel_material_ui as pmui
import pytest
from bokeh.models import (Button, ColumnDataSource, FileInput,
                          RadioButtonGroup, Select, TextInput, Toggle)

from tests.gui_driver import click, has, read, upload, write


def test_bokeh_models():
    select = Select(options=[('a', 'Label A'), 'b'], value='a')
    assert read(select, 'options') == ['Label A', 'b']
    group = RadioButtonGroup(labels=['x', 'y'], active=1)
    assert read(group, 'options') == ['x', 'y']
    assert read(group, 'active') == 1
    toggle = Toggle(label='Run')
    write(toggle, 'active', True)
    assert toggle.active and read(toggle, 'label') == 'Run'
    assert has(TextInput(), 'title') and not has(TextInput(), 'label')
    with pytest.raises(AttributeError):
        read(TextInput(), 'label')
    source = ColumnDataSource(data=dict(x=[1, 2]))
    write(source, 'selected', [1])
    assert read(source, 'selected') == [1]


def test_bokeh_upload_and_click():
    file_input = FileInput()
    assert read(file_input, 'value') == ''
    upload(file_input, b'1,2', 'a.csv')
    assert read(file_input, 'value') == base64.b64encode(b'1,2').decode()
    assert file_input.filename == 'a.csv'
    clicks = []
    button = Button()
    button.on_click(lambda: clicks.append(1))
    click(button)
    assert clicks == [1]


@pytest.mark.parametrize('pane', [pmui.Typography('t'), pmui.Alert(object='t'),
                                  pn.pane.HTML('t'), pn.pane.Markdown('t')])
def test_panel_text_panes(pane):
    assert read(pane, 'text') == 't'
    write(pane, 'text', 'u')
    assert pane.object == 'u'


def test_panel_widgets():
    text = pmui.TextInput(label='1. Start', value='0')
    assert read(text, 'title') == '1. Start'
    assert not has(text, 'label')
    write(text, 'value', '2')
    assert text.value == '2'
    select = pmui.Select(options={'Label A': 'a', 'Label B': 'b'}, value='a')
    assert read(select, 'options') == ['Label A', 'Label B']
    toggle = pmui.Toggle(label='Run', color='success')
    write(toggle, 'active', True)
    assert toggle.value is True and read(toggle, 'active') is True
    assert read(toggle, 'label') == 'Run'
    assert read(toggle, 'button_type') == 'success'
    assert not has(toggle, 'value')
    group = pmui.RadioButtonGroup(options={'x': 'a', 'y': 'b'}, value='a')
    write(group, 'active', 1)
    assert group.value == 'b'
    assert read(group, 'active') == 1 and read(group, 'options') == ['x', 'y']
    assert not has(group, 'value')
    # a multi-selection button group: ``active`` is the list of indices
    checks = pmui.CheckButtonGroup(options=['PALC', 'Reference'], value=[])
    write(checks, 'active', [0, 1])
    assert checks.value == ['PALC', 'Reference']
    assert read(checks, 'active') == [0, 1]
    write(checks, 'active', [1])
    assert checks.value == ['Reference'] and read(checks, 'active') == [1]
    assert read(checks, 'options') == ['PALC', 'Reference']
    assert not has(checks, 'value') and not has(checks, 'label')
    slider = pmui.RangeSlider(start=0, end=10, value=(1, 2), step=0.5)
    assert [read(slider, p) for p in ('start', 'end', 'step', 'value')] == \
        [0, 10, 0.5, (1, 2)]
    assert read(pmui.Tabs(('a', 'x'), ('b', 'y'), active=1), 'active') == 1
    menu = pmui.TabMenu(items=['Help', ('1 · Venue', 'venue'),
                               {'label': '2 · Array'}], active=0)
    assert read(menu, 'options') == ['Help', '1 · Venue', '2 · Array']
    write(menu, 'active', 2)
    assert menu.active == 2 and read(menu, 'active') == 2
    assert not has(menu, 'value') and not has(menu, 'label')
    assert read(pmui.FileDownload(label='Save'), 'label') == 'Save'
    with pytest.raises(AttributeError):
        read(text, 'selected')


def test_panel_upload_and_click():
    file_input = pmui.FileInput()
    assert read(file_input, 'value') == ''
    seen = []
    file_input.param.watch(lambda e: seen.append(file_input.filename),
                           'value')
    upload(file_input, b'1,2', 'a.csv')
    assert read(file_input, 'value') == base64.b64encode(b'1,2').decode()
    assert seen == ['a.csv']
    clicks = []
    button = pmui.Button(label='Go')
    button.on_click(lambda event: clicks.append(1))
    click(button)
    assert clicks == [1]
