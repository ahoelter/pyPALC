"""The GUI's widgets react to user input as in the legacy GUI.

A fixed sequence of 100 user interactions (typing into inputs, choosing
options, clicking buttons, tapping lines and points, dragging the target
slope, running PALC, switching pages) is replayed on a fresh
:class:`PalcApp`. After every step the state of about 70 widgets, sources
and figures (visible, disabled, value, options, label, text, active, data,
ranges, frame height, ...) and the number of models in the document are
compared with ``tests/gui_cascade_golden.json``.

How the golden file was made
----------------------------
The golden state is the state of the legacy GUI (the original ``main.py``
before the split into :mod:`pypalc.gui`), recorded by driving the legacy
module-level callbacks through the same steps. Only the accepted
differences listed below replace legacy values; every other value is the
legacy one. After the Bokeh 3 upgrade the golden file was recorded again
from the current GUI, once every difference to the Bokeh 2 recording was
explained (see "Accepted differences after the Bokeh 3 upgrade").

Accepted differences to the legacy GUI
--------------------------------------
* ``psi_bad`` (``2.aperture_input.value``): an invalid aperture angle now
  resets the input to the default ``'3.0'`` instead of ``'0.0524'``
  (bug fix 702f6a6).
* ``save_same`` (``1.venue_error_info.text``): saving a line that lies on
  an already drawn line now shows "Error: The audience line lies on an
  already drawn audience line"; the legacy GUI rejected the line without
  a text.
* ``run_ts`` and later steps (a target slope run with hard margin gap
  handling, whose optimization raises an IndexError): the legacy GUI
  crashed in the callback with ``UnboundLocalError`` after it had already
  written the new tilt angles, rays, cabinets and sound field into their
  sources. The GUI now shows "Error: PALC calculation failed", releases
  the start toggle and keeps the results of the previous run in all
  sources (``4.results_source``, ``5.sound_field_source``,
  ``app.cabinet_patches_source``, ``app.rays_source``,
  ``app.hit_points_source``, ``app.config_error_info.text`` and the
  recorded exception differ; see :func:`test_failed_run_keeps_results`).
* The rewording of the "more than 2000 points" text is not reached by
  these steps.
* ``document.model_count`` on page 4 (steps ``page4`` and ``page4b`` and
  the steps up to the next page change) was 8 higher than in the legacy
  GUI under Bokeh 2 (the counts are now those of Bokeh 3, see below): the
  "Save Setup"
  button, the "Load Setup" title and file input, the source of the setup
  file download and their JavaScript callbacks (Task 7). Nothing else of
  the golden file changed; the new widgets are not among :data:`MODELS`
  (``tests/test_apply_setup.py`` covers them). Refreshing the range of a
  fixed angle when the array position changes (Task 7) changes no
  recorded value.
* ``run`` (``app.rays_source.data``): the rays drawn in the venue figure
  now end where they cross a venue line (or, if they miss the venue, above
  or below its farthest point) instead of at the hit points PALC assumes;
  see ``tests/test_rays.py``. Only the ray ends of this step changed.

Renamed keys (Task 8)
---------------------
The descriptive names of Task 8 renamed some model attributes, the
configuration attributes (``config.<attr>``) and the columns of several
sources. The golden file got a mechanical key rename only: a script mapped
the old keys and column names to the new ones and left every value as it
was (hashed values were re-hashed from the recorded full values, which
hash to the old golden entries). The golden file was not recorded again
from the renamed app before the Bokeh 3 upgrade.

Accepted differences after the Bokeh 3 upgrade
----------------------------------------------
Before the golden file was recorded again under Bokeh 3.10,
``tools/cascade_full_states.py`` recorded the full, unhashed states of all
100 steps under Bokeh 2.2 (``master``, Python 3.8; these states hash to
the Bokeh 2 golden file) and under Bokeh 3.10 (``bokeh3``, Python 3.12),
and ``tools/compare_cascade_states.py`` compared them. Every difference
is caused by Bokeh:

* missing/renamed property (139 values, 2 keys):
  ``2.directivity_upload.title`` is new in all 100 steps (Bokeh 3's
  FileInput is an input widget with a title).
  ``2.directivity_upload.value`` is not recorded in the 39 steps before
  ``upload``: Bokeh 3's FileInput has no value before an upload (Bokeh 2:
  ``''``) and reading it raises ``UnsetValueError``. From ``upload`` on
  the values are equal. Since ``FileInput.value`` is read-only in Bokeh 3's
  Python, the ``upload`` step sets it through ``set_from_json``, the way
  the server receives a browser upload.
* model count (100 values, 1 key): ``document.model_count`` is higher on
  every page (page 0: 95 -> 108, page 1: 183 -> 222, page 2: 217 -> 266,
  page 3: 194 -> 234, page 4: 297 -> 353, page 5: 363 -> 440). The model
  types of the page differ only by these Bokeh 3 changes:

  - every glyph renderer has a muted glyph besides its glyph and
    nonselection glyph (the beam images each with their own color
    mapper);
  - every ``CDSView`` has an ``AllIndices`` filter, every figure two
    ``AllLabels`` label policies, every box zoom overlay
    ``BoxInteractionHandles`` with ``AreaVisuals``;
  - the color bars have a ``NoOverlap`` label policy and an ``'auto'``
    ticker and formatter instead of ``BasicTicker`` and
    ``BasicTickFormatter`` models;
  - the page is a ``GridPlot`` with its own (hidden) ``Toolbar`` instead of
    a ``GridBox``;
  - renamed types: ``Figure`` -> ``figure``, ``Panel`` -> ``TabPanel``,
    ``FuncTickFormatter`` -> ``CustomJSTickFormatter``, and the ``X`` and
    ``Circle`` markers are ``Scatter`` glyphs.
* model type name: none (no recorded value names a renamed type).
* new default (45 values, 9 keys): in the steps ``page1`` to ``ey`` the
  figures that start with a DataRange1d (``x_range`` and ``y_range`` of
  ``app.venue_figure``, ``5.beam_figure``, ``5.reference_beam_figure``
  and ``5.venue_points_figure``, ``4.spl_over_distance_figure.x_range``)
  have NaN as start and end, where Bokeh 2 had ``None``. They are
  recorded as ``'NaN'`` (:func:`_norm`), since NaN never equals the NaN
  read back from the golden file.
* numeric drift within rtol 1e-9: none; all numbers (rounded to 9
  decimals) are equal.
* size changes: none recorded. The help tabs are as wide as their widest
  panel (50030d4), but no width of them is recorded. Clearing the rays
  before the page is rebuilt (ea30381) and the Bokeh 2 dash patterns
  (4e4cfba) change no recorded value.

The values are rounded to 9 decimals. Besides the widgets, the values of
the PALC configuration that the configuration widgets store
(:data:`CONFIG_KEYS`) are compared. Values whose JSON text is longer
than :data:`MAX_INLINE` characters are stored as a hash of that text.
The golden file stores the first state completely and for every later step
only the values that changed.

Running ``python -m tests.test_gui_cascade OUT.json`` writes the states of
the current GUI in the same format, which helps to inspect a failure.
"""
import contextlib
import functools
import hashlib
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from bokeh.core.property.descriptors import UnsetValueError
from bokeh.document import Document
from bokeh.models import FileInput

from pypalc.gui.app import PalcApp
from tests.gui_steps import CONFIG_KEYS, MODELS, model, steps

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "tests" / "gui_cascade_golden.json"
#: Longer values (as JSON text) are compared by a hash.
MAX_INLINE = 1500
PROPS = ('visible', 'disabled', 'value', 'options', 'label', 'button_type',
         'text', 'active', 'start', 'end', 'step', 'title', 'data',
         'frame_height', 'x_range', 'y_range', 'min_border')


def _norm(v):
    """JSON-able value, floats rounded to 9 decimals, models by type."""
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    if isinstance(v, np.ndarray):
        return _norm(v.tolist())
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    if isinstance(v, float):
        return round(v, 9)
    if hasattr(v, 'start') and hasattr(v, 'end') and \
            type(v).__name__.endswith('Range1d'):
        # Bokeh 3 starts a DataRange1d at NaN (Bokeh 2: None); it is
        # recorded as 'NaN', since NaN never equals the NaN read back from
        # the golden file
        return [type(v).__name__] + [
            'NaN' if isinstance(x, float) and math.isnan(x) else _norm(x)
            for x in (v.start, v.end)]
    if hasattr(v, 'properties'):
        return type(v).__name__
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return round(float(v), 9)
    return v


def compact(value):
    """The value, or a hash of it if its JSON text is long."""
    text = json.dumps(value, sort_keys=True)
    if len(text) <= MAX_INLINE:
        return value
    return {"sha1": hashlib.sha1(text.encode()).hexdigest(),
            "size": len(text)}


def snapshot(app, error):
    """State of the recorded models after a step."""
    state = {}
    for path in MODELS:
        obj = model(app, path)
        for prop in PROPS:
            if prop in obj.properties():
                try:
                    value = getattr(obj, prop)
                except UnsetValueError:
                    # Bokeh 3: a FileInput without upload has no value
                    if isinstance(obj, FileInput) and prop == 'value':
                        continue
                    raise
                state[path + '.' + prop] = _norm(value)
        if 'selected' in obj.properties():
            state[path + '.selected'] = _norm(obj.selected.indices)
    roots = app.doc.roots
    state['document.model_count'] = \
        len(list(roots[0].references())) if roots else 0
    config = app.state.config
    for key in CONFIG_KEYS:
        state['config.' + key] = _norm(getattr(config, key))
    state['error'] = error
    state = json.loads(json.dumps(state, default=str))
    return {key: compact(value) for key, value in state.items()}


def replay():
    """Replay the steps on a new app; yield (step name, state)."""
    app = PalcApp(Document())
    app.show()
    for name, step in steps(app):
        error = None
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                step()
        except Exception as exc:  # recorded like in the golden file
            error = type(exc).__name__ + ':' + str(exc)[:100]
        yield name, snapshot(app, error)


@functools.lru_cache(maxsize=1)
def load_golden():
    """The golden states: list of (step name, state)."""
    steps = json.loads(GOLDEN.read_text())
    states, state = [], {}
    for name, changes in steps:
        state = dict(state, **changes)
        states.append((name, state))
    return states


def write_states(states, path):
    """Write states like the golden file (first state, then changes)."""
    out, previous = [], {}
    for name, state in states:
        assert set(state) >= set(previous)
        out.append([name, {k: v for k, v in state.items()
                           if k not in previous or previous[k] != v}])
        previous = state
    Path(path).write_text(json.dumps(out, indent=0, sort_keys=True) + '\n')


@pytest.fixture(scope="module")
def replayed():
    return list(replay())


def test_replays_all_steps(replayed):
    assert [name for name, _ in replayed] == \
        [name for name, _ in load_golden()]


@pytest.mark.parametrize("index", range(100))
def test_state_after_step(replayed, index):
    golden_name, golden = load_golden()[index]
    name, state = replayed[index]
    assert name == golden_name
    differences = sorted(key for key in set(state) | set(golden)
                         if state.get(key) != golden.get(key))
    assert not differences, (name, {key: (golden.get(key), state.get(key))
                                    for key in differences[:5]})


def test_failed_run_keeps_results(replayed):
    """A failing target slope run shows its error and changes nothing else.
    """
    states = dict(replayed)
    before, after = states['drag'], states['run_ts']
    assert after['app.config_error_info.text'] == \
        'Error: PALC calculation failed'
    assert after['4.start_toggle.active'] is False
    assert after['error'] is None
    changed = {key for key in after if after[key] != before[key]}
    assert changed == {'app.config_error_info.text'}


if __name__ == '__main__':
    write_states(list(replay()), sys.argv[1])
