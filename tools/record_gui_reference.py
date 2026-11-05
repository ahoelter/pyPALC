"""Record the state of the GUI after each of the scripted GUI steps.

Usage::

    python tools/record_gui_reference.py OUT.json

Replays the steps of ``tests/gui_steps.py`` on a fresh ``PalcApp()`` and
records after every step, for every model in ``MODELS``, the properties
:data:`PROPS` (read through ``tests/gui_driver.py`` by their Bokeh-era
names; a property the model lacks is skipped), the PALC configuration
values ``config.<key>`` for ``CONFIG_KEYS`` and the exception the step
raised (``error``). Floats are rounded to 9 decimals, values whose JSON
text is longer than :data:`MAX_INLINE` characters are stored as
``{"sha1", "size"}``. ``OUT.json`` holds ``[[step, changes], ...]``: the
first state completely, every later state as the values that changed.

``tests/gui_reference.json`` was written by this script from the Bokeh
GUI; ``tests/test_gui_reference.py`` compares the GUI with it.

Not recorded, since the Panel GUI changes them on purpose: input titles,
figure ranges, frame heights, borders and the number of models.

author: Arne Hoelter
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pypalc.gui.app import PalcApp  # noqa: E402
from tests.gui_driver import has, read  # noqa: E402
from tests.gui_steps import CONFIG_KEYS, MODELS, model, steps  # noqa: E402

#: Recorded properties (Bokeh-era names).
PROPS = ('visible', 'disabled', 'value', 'options', 'active', 'text',
         'label', 'button_type', 'start', 'end', 'step', 'data', 'selected')
#: Longer values (as JSON text) are stored as a hash.
MAX_INLINE = 1500


def norm(value: Any) -> Any:
    """JSON-able value with floats rounded to 9 decimals.

    Parameters
    ----------
    value : Any
        A property value: numbers, strings, lists, tuples, dicts and numpy
        arrays and scalars.

    Returns
    -------
    Any
        The value with tuples and arrays as lists.
    """
    if isinstance(value, dict):
        return {k: norm(x) for k, x in value.items()}
    if isinstance(value, np.ndarray):
        return norm(value.tolist())
    if isinstance(value, (list, tuple)):
        return [norm(x) for x in value]
    if isinstance(value, (float, np.floating)):
        return round(float(value), 9)
    if isinstance(value, np.integer):
        return int(value)
    return value


def compact(value: Any) -> Any:
    """The value, or ``{"sha1", "size"}`` of its JSON text if that is long.
    """
    text = json.dumps(value, sort_keys=True)
    if len(text) <= MAX_INLINE:
        return value
    return {"sha1": hashlib.sha1(text.encode()).hexdigest(),
            "size": len(text)}


def snapshot(app, error: Optional[str]) -> Dict[str, Any]:
    """State of the recorded models after a step.

    Parameters
    ----------
    app : PalcApp
        The app.
    error : str or None
        ``"<exception type>:<message>"`` if the step raised.

    Returns
    -------
    dict
        ``"<model path>.<prop>"``, ``"config.<key>"`` and ``"error"`` ->
        normalized (and possibly hashed) value.
    """
    state = {}
    for path in MODELS:
        obj = model(app, path)
        for prop in PROPS:
            if has(obj, prop):
                state[path + '.' + prop] = norm(read(obj, prop))
    config = app.state.config
    for key in CONFIG_KEYS:
        state['config.' + key] = norm(getattr(config, key))
    state['error'] = error
    state = json.loads(json.dumps(state, default=str))
    return {key: compact(value) for key, value in state.items()}


def replay() -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Replay the steps on a fresh app.

    Yields
    ------
    tuple of (str, dict)
        Step name and the state after the step (see :func:`snapshot`).
    """
    app = PalcApp()
    for name, step in steps(app):
        error = None
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                step()
        except Exception as exc:  # recorded, like a failing callback
            error = type(exc).__name__ + ':' + str(exc)[:100]
        yield name, snapshot(app, error)


def write_states(states: List[Tuple[str, Dict[str, Any]]],
                 path: Path) -> None:
    """Write states as the first state and then the changes of each step.

    Parameters
    ----------
    states : list of (str, dict)
        Step names and complete states, as from :func:`replay`.
    path : Path
        Output JSON file.
    """
    out, previous = [], {}
    for name, state in states:
        assert set(state) >= set(previous), name
        out.append([name, {k: v for k, v in state.items()
                           if k not in previous or previous[k] != v}])
        previous = state
    Path(path).write_text(json.dumps(out, indent=0, sort_keys=True) + '\n')


def read_states(path: Path) -> List[Tuple[str, Dict[str, Any]]]:
    """Read states written by :func:`write_states` as complete states.

    Parameters
    ----------
    path : Path
        JSON file.

    Returns
    -------
    list of (str, dict)
        Step names and complete states.
    """
    states, state = [], {}
    for name, changes in json.loads(Path(path).read_text()):
        state = dict(state, **changes)
        states.append((name, state))
    return states


def main(argv: Optional[List[str]] = None) -> int:
    """Command line entry point.

    Parameters
    ----------
    argv : list of str, optional
        Arguments without the program name; ``sys.argv[1:]`` if omitted.

    Returns
    -------
    int
        Exit code (0).
    """
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('out', type=Path, help='output JSON file')
    args = parser.parse_args(argv)
    write_states(list(replay()), args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
