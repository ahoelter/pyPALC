"""Dump the full states of the GUI cascade replay of a checkout.

Usage::

    python tools/cascade_full_states.py REPO_DIR OUT.json

Imports ``tests.test_gui_cascade`` (and with it ``pypalc``) from
``REPO_DIR``, replays its steps with ``compact`` replaced by the identity,
so that no value is stored as a hash, and writes ``[[step, state], ...]``
with every state complete (not only the changes) to ``OUT.json``.

The script runs under Python 3.8 with Bokeh 2.2 (against a worktree of
``master``) and under Python 3.12 with Bokeh 3 (against ``bokeh3``), so
that ``tools/compare_cascade_states.py`` can compare both replays without
hashes. Run it from a directory outside the checkouts, so that nothing is
imported from the current directory.

author: Arne Hoelter
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import List, Optional


def _identity(value):
    """The value itself: replaces ``compact`` so nothing is hashed."""
    return value


def full_states(repo_dir: Path) -> list:
    """Replay the GUI cascade of a checkout with full values.

    Parameters
    ----------
    repo_dir : Path
        Root of the checkout that holds ``pypalc`` and ``tests``.

    Returns
    -------
    list
        ``[[step name, state], ...]`` with the full state after each step.

    Raises
    ------
    RuntimeError
        If ``pypalc`` or ``tests`` were imported from somewhere else.
    """
    repo_dir = repo_dir.resolve()
    sys.path.insert(0, str(repo_dir))
    cascade = importlib.import_module('tests.test_gui_cascade')
    for name in ('pypalc', 'tests'):
        module_file = Path(sys.modules[name].__file__).resolve()
        if repo_dir not in module_file.parents:
            raise RuntimeError('%s was imported from %s, not from %s'
                               % (name, module_file, repo_dir))
    cascade.compact = _identity
    return [[name, state] for name, state in cascade.replay()]


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
    parser.add_argument('repo_dir', type=Path,
                        help='checkout with pypalc and tests')
    parser.add_argument('out', type=Path, help='JSON file to write')
    args = parser.parse_args(argv)
    states = full_states(args.repo_dir)
    args.out.write_text(json.dumps(states, sort_keys=True) + '\n')
    print('%d steps written to %s' % (len(states), args.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
