"""The GUI reacts to the scripted user input as the Bokeh GUI did.

``tests/gui_reference.json`` holds the state of the Bokeh GUI (before the
port to Panel) after each of the 100 steps of ``tests/gui_steps.py``,
recorded by ``tools/record_gui_reference.py``. The test replays the steps
on a fresh app, records the same way and compares every step with the
reference.

Intended changes of the port are listed here, not recorded again:

* :data:`RENAMED_LABELS` maps an old text to its new text. It applies to
  the reference's ``label`` values and to its ``options`` entries and
  string ``value`` values that equal an old text exactly.
* :data:`INTENDED` lists keys (``fnmatch`` patterns) that are not
  compared, each with the reason.

author: Arne Hoelter
"""
import fnmatch
from typing import Any, Dict, List, Tuple

from tests.gui_steps import ROOT
from tools.record_gui_reference import read_states, replay

REFERENCE = ROOT / 'tests' / 'gui_reference.json'

#: Old text -> new text of renamed labels and options.
RENAMED_LABELS: Dict[str, str] = {
    # page tabs
    '1. PAL': '1 · Venue',
    '2. Loudspeaker Configuration': '2 · Loudspeaker',
    '3. Array Configuration': '3 · Array',
    '4. Algorithm': '4 · Algorithm',
    '5. Sound Field Prediction': '5 · Sound Field',
    # axis modes of the venue figure
    'Equal (variable figure height)': 'Equal, variable height',
    'Equal (const. figure height)': 'Equal, fixed height',
}

#: (key pattern, reason) of keys whose value changes on purpose.
INTENDED: List[Tuple[str, str]] = [
    ('app.page_select.button_type',
     'the page tabs are a Material tab menu, which has no button type'),
    ('app.venue_axes_select.button_type',
     'the axis modes are a Material button group, which has a button type '
     '(unset); the Bokeh radio group had none'),
    ('1.venue_error_info.disabled',
     'the error text of page 1 is a Material alert, which has no disabled '
     'state'),
    ('1.venue_error_info.visible',
     'the error alert of page 1 is hidden while its text is empty (the '
     'text itself is compared)'),
]


def _renamed(key: str, value: Any) -> Any:
    """The reference value with renamed labels replaced."""
    prop = key.rsplit('.', 1)[-1]
    if prop in ('label', 'value') and isinstance(value, str):
        return RENAMED_LABELS.get(value, value)
    if prop == 'options' and isinstance(value, list):
        return [RENAMED_LABELS.get(x, x) if isinstance(x, str) else x
                for x in value]
    return value


def _intended(key: str) -> bool:
    """Whether the key matches a pattern of :data:`INTENDED`."""
    return any(fnmatch.fnmatchcase(key, pattern) for pattern, _ in INTENDED)


def test_gui_matches_reference():
    reference = read_states(REFERENCE)
    replayed = list(replay())
    assert [name for name, _ in replayed] == \
        [name for name, _ in reference]
    differences = []
    for (name, state), (_, expected) in zip(replayed, reference):
        for key in sorted(set(state) | set(expected)):
            if _intended(key):
                continue
            want = _renamed(key, expected.get(key, '<missing>'))
            got = state.get(key, '<missing>')
            if got != want:
                differences.append('step %s, key %s: reference %r, now %r'
                                   % (name, key, want, got))
    assert not differences, '%d difference(s):\n%s' % (
        len(differences), '\n'.join(differences[:10]))
