"""Compare two full dumps of the GUI cascade replay.

Usage::

    python tools/compare_cascade_states.py OLD.json NEW.json \\
        [--size-keys PATTERN ...]

``OLD.json`` and ``NEW.json`` are written by
``tools/cascade_full_states.py``. For every step, every key whose value
differs is printed, grouped into these categories:

``missing/renamed property``
    The key exists in only one of the dumps.
``model_count``
    ``document.model_count``.
``model type name``
    The values differ only in names of Bokeh model types that Bokeh 3
    renamed or the port replaced (:data:`TYPE_RENAMES`).
``new default``
    The values differ only where the old value is ``None`` and the new one
    is NaN or ``'NaN'`` (Bokeh 3 starts a DataRange1d at NaN, Bokeh 2 at
    ``None``; ``tests/test_gui_cascade.py`` records it as ``'NaN'``).
``numeric within rtol 1e-9``
    The values differ only in numbers, each within a relative tolerance of
    1e-9 (or by at most one unit of the 9th decimal, to which the states
    are rounded).
``size-derived``
    The key matches one of the ``--size-keys`` patterns (``fnmatch``).
``OTHER``
    Anything else.

A key whose differences fall into several of these categories is listed
under each. The script exits with 1 if ``OTHER`` is not empty.

author: Arne Hoelter
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

MISSING = 'missing/renamed property'
MODEL_COUNT = 'model_count'
TYPE_NAME = 'model type name'
NEW_DEFAULT = 'new default'
NUMERIC = 'numeric within rtol 1e-9'
SIZE = 'size-derived'
OTHER = 'OTHER'
CATEGORIES = (MISSING, MODEL_COUNT, TYPE_NAME, NEW_DEFAULT, NUMERIC, SIZE,
              OTHER)
RTOL = 1e-9
#: One unit of the 9th decimal, to which the states are rounded.
ATOL = 1e-9
#: Bokeh 2 model type name -> Bokeh 3 model type name.
TYPE_RENAMES = {
    'Figure': 'figure',
    'Panel': 'TabPanel',
    'FuncTickFormatter': 'CustomJSTickFormatter',
    # the page grid (cc97d78)
    'GridBox': 'GridPlot',
    # markers are drawn with scatter (d2df727)
    'X': 'Scatter',
    'Circle': 'Scatter',
}


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_nan(value) -> bool:
    return isinstance(value, float) and math.isnan(value)


def leaf_differences(old, new, path: str = '') -> List[Tuple[str, object,
                                                              object]]:
    """The differing leaves of two JSON values.

    Parameters
    ----------
    old, new : object
        JSON values (dicts, lists, strings, numbers, booleans, None).
    path : str, optional
        Location of the values, used as prefix of the returned paths.

    Returns
    -------
    list of (str, object, object)
        ``(path, old leaf, new leaf)`` for every leaf that differs. A
        difference in structure (other dict keys, other list length, other
        type) is returned as one leaf with the whole values. NaN equals NaN.
    """
    if isinstance(old, dict) and isinstance(new, dict):
        if old.keys() != new.keys():
            return [(path, old, new)]
        out = []
        for key in sorted(old):
            out += leaf_differences(old[key], new[key], '%s/%s' % (path, key))
        return out
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            return [(path, old, new)]
        out = []
        for i, (a, b) in enumerate(zip(old, new)):
            out += leaf_differences(a, b, '%s[%d]' % (path, i))
        return out
    if _is_nan(old) and _is_nan(new):
        return []
    if type(old) is type(new) and old == new:
        return []
    if _is_number(old) and _is_number(new) and old == new:
        return []
    return [(path, old, new)]


def leaf_category(old, new) -> str:
    """The category of one differing leaf (not of a whole key).

    Parameters
    ----------
    old, new : object
        The differing leaf values.

    Returns
    -------
    str
        :data:`TYPE_NAME`, :data:`NEW_DEFAULT`, :data:`NUMERIC` or
        :data:`OTHER`.
    """
    if isinstance(old, str) and TYPE_RENAMES.get(old) == new:
        return TYPE_NAME
    if old is None and (_is_nan(new) or new == 'NaN'):
        return NEW_DEFAULT
    if _is_number(old) and _is_number(new) and not _is_nan(old) and \
            not _is_nan(new) and math.isclose(old, new, rel_tol=RTOL,
                                              abs_tol=ATOL):
        return NUMERIC
    return OTHER


def compare(old_states: list, new_states: list,
            size_keys: List[str]) -> Dict[str, list]:
    """Group the differences of two dumps by category.

    Parameters
    ----------
    old_states, new_states : list
        ``[[step name, state], ...]`` as written by
        ``tools/cascade_full_states.py``.
    size_keys : list of str
        ``fnmatch`` patterns of keys whose values follow a size change.

    Returns
    -------
    dict
        Category -> list of ``(step, key, old value, new value, leaves)``,
        where ``leaves`` are the differing leaves of that category.

    Raises
    ------
    ValueError
        If the dumps do not have the same steps.
    """
    old_names = [name for name, _ in old_states]
    new_names = [name for name, _ in new_states]
    if old_names != new_names:
        raise ValueError('the dumps have different steps: %s / %s'
                         % (old_names, new_names))
    groups = {category: [] for category in CATEGORIES}
    for (step, old), (_, new) in zip(old_states, new_states):
        for key in sorted(set(old) | set(new)):
            if key not in old or key not in new:
                groups[MISSING].append((step, key, old.get(key, '<absent>'),
                                        new.get(key, '<absent>'), []))
                continue
            leaves = leaf_differences(old[key], new[key])
            if not leaves:
                continue
            if key == 'document.model_count':
                groups[MODEL_COUNT].append((step, key, old[key], new[key],
                                            leaves))
                continue
            if any(fnmatch.fnmatchcase(key, p) for p in size_keys):
                groups[SIZE].append((step, key, old[key], new[key], leaves))
                continue
            by_category = {}
            for leaf in leaves:
                by_category.setdefault(leaf_category(leaf[1], leaf[2]),
                                       []).append(leaf)
            for category, items in by_category.items():
                groups[category].append((step, key, old[key], new[key],
                                         items))
    return groups


def _short(value, limit: int = 160) -> str:
    text = json.dumps(value, sort_keys=True)
    return text if len(text) <= limit else text[:limit] + '... (%d chars)' \
        % len(text)


def _drift(leaves) -> Tuple[float, float]:
    """Largest absolute and relative difference of numeric leaves."""
    absolute = max(abs(a - b) for _, a, b in leaves)
    relative = max(abs(a - b) / max(abs(a), abs(b)) for _, a, b in leaves)
    return absolute, relative


def report(groups: Dict[str, list]) -> str:
    """Text listing of :func:`compare`'s groups.

    Parameters
    ----------
    groups : dict
        Result of :func:`compare`.

    Returns
    -------
    str
    """
    lines = []
    largest = None
    for category in CATEGORIES:
        entries = groups[category]
        keys = sorted({key for _, key, _, _, _ in entries})
        lines.append('== %s: %d differing values (step, key), %d keys'
                     % (category, len(entries), len(keys)))
        for step, key, old, new, leaves in entries:
            if category == NUMERIC:
                absolute, relative = _drift(leaves)
                lines.append('  %-14s %s: %d numbers, max abs %.3g, max rel '
                             '%.3g' % (step, key, len(leaves), absolute,
                                       relative))
                if largest is None or relative > largest[0]:
                    worst = max(leaves, key=lambda leaf: abs(
                        leaf[1] - leaf[2]) / max(abs(leaf[1]), abs(leaf[2])))
                    largest = (relative, step, key + worst[0], worst[1],
                               worst[2])
            elif category in (TYPE_NAME, NEW_DEFAULT, OTHER) and leaves:
                lines.append('  %-14s %s: %d leaves' % (step, key,
                                                        len(leaves)))
                for path, a, b in leaves[:3]:
                    lines.append('      %s: %s -> %s'
                                 % (path or '.', _short(a), _short(b)))
            else:
                lines.append('  %-14s %s: %s -> %s'
                             % (step, key, _short(old), _short(new)))
    if largest is not None:
        lines.append('largest numeric drift: rel %.3g at step %s, %s: '
                     '%r -> %r' % largest)
    return '\n'.join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    """Command line entry point.

    Parameters
    ----------
    argv : list of str, optional
        Arguments without the program name; ``sys.argv[1:]`` if omitted.

    Returns
    -------
    int
        1 if there are differences of category ``OTHER``, else 0.
    """
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('old', type=Path, help='dump of the old checkout')
    parser.add_argument('new', type=Path, help='dump of the new checkout')
    parser.add_argument('--size-keys', nargs='*', default=[],
                        metavar='PATTERN',
                        help='fnmatch patterns of size-derived keys')
    args = parser.parse_args(argv)
    groups = compare(json.loads(args.old.read_text()),
                     json.loads(args.new.read_text()), args.size_keys)
    print(report(groups))
    return 1 if groups[OTHER] else 0


if __name__ == '__main__':
    sys.exit(main())
