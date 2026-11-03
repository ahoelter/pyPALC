"""Command line interface: ``python -m pypalc run SETUP --out DIR``."""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import List, Optional

from .core.errors import (CalculationError, ConvergenceError, PalcError,
                          SetupError, VenueError)
from .core.export import write_results
from .core.pipeline import run
from .core.settings import load_setup

EXIT_SETUP = 2
EXIT_VENUE = 3
EXIT_CALCULATION = 4
EXIT_OUTPUT = 5

#: Prefix of the core's error texts, which the GUI shows as they are.
_GUI_PREFIX = "Error: "


def _parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="pypalc", description="pyPALC without the GUI.")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser(
        "run", help="run a setup file and write the results as CSV")
    run_parser.add_argument("setup", help="YAML setup file")
    run_parser.add_argument("--out", default="results",
                            help="output directory (default: ./results)")
    run_parser.add_argument("--traceback", action="store_true",
                            help="print the full traceback of an error")
    return parser


def _describe(exc: PalcError) -> str:
    """Text of an error: its message and, if it has one, its cause.

    The ``Error:`` prefix of the texts the GUI shows is removed. The cause
    (``raise ... from cause``) is appended unless the message already
    contains it.
    """
    message = exc.message if isinstance(exc, SetupError) else str(exc)
    if message.startswith(_GUI_PREFIX):
        message = message[len(_GUI_PREFIX):]
    cause = exc.__cause__
    if cause is not None:
        text = str(cause)
        if not text:
            message += " (caused by %s)" % type(cause).__name__
        elif text not in message:
            message += " (caused by %s: %s)" % (type(cause).__name__, text)
    return message


def _report(exc: BaseException, text: str, show_traceback: bool) -> None:
    """Print an error to stderr, with its traceback if asked for."""
    if show_traceback:
        traceback.print_exception(type(exc), exc, exc.__traceback__,
                                  file=sys.stderr)
    print(text, file=sys.stderr)


def main(argv: Optional[List[str]] = None) -> int:
    """Run the command line interface.

    Parameters
    ----------
    argv : list of str, optional
        Arguments; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code: 0 ok, 2 invalid setup, 3 venue error, 4 no convergence
        or calculation error, 5 output could not be written. An error is
        printed to stderr with its cause; with ``--traceback`` the full
        (chained) traceback is printed too.
    """
    args = _parser().parse_args(argv)
    try:
        try:
            setup = load_setup(args.setup)
        except (OSError, UnicodeDecodeError) as exc:
            raise SetupError("setup", "cannot read %s: %s"
                             % (args.setup, exc)) from exc
        result = run(setup)
    except SetupError as exc:
        _report(exc, "setup error at %s: %s" % (exc.key, _describe(exc)),
                args.traceback)
        return EXIT_SETUP
    except VenueError as exc:
        _report(exc, "venue error: %s" % _describe(exc), args.traceback)
        return EXIT_VENUE
    except (ConvergenceError, CalculationError) as exc:
        _report(exc, "calculation error: %s" % _describe(exc),
                args.traceback)
        return EXIT_CALCULATION
    try:
        paths = write_results(result, Path(args.out))
    except OSError as exc:
        _report(exc, "output error: cannot write the results to %s: %s"
                % (args.out, exc), args.traceback)
        return EXIT_OUTPUT
    for path in paths:
        print(path)
    return 0
