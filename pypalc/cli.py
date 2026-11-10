"""Command line interface: ``pypalc run SETUP``, ``pypalc examples DIR`` and
``pypalc gui``."""
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
from .examples import copy_examples

EXIT_SETUP = 2
EXIT_VENUE = 3
EXIT_CALCULATION = 4
EXIT_OUTPUT = 5
EXIT_GUI = 6

#: Prefix of the core's error texts, which the GUI shows as they are.
_GUI_PREFIX = "Error: "

#: Top-level packages of the ``gui`` extra; only their absence is reported
#: as "install pypalc[gui]".
_GUI_PACKAGES = {"panel", "bokeh", "panel_material_ui", "pandas",
                 "colorcet", "param"}


def _parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="pypalc", description="pyPALC: run setup files, copy the examples, "
        "serve the GUI.")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser(
        "run", help="run a setup file and write the results as CSV")
    run_parser.add_argument("setup", help="YAML setup file")
    run_parser.add_argument("--out", default="results",
                            help="output directory (default: ./results)")
    run_parser.add_argument("--traceback", action="store_true",
                            help="print the full traceback of an error")
    examples_parser = sub.add_parser(
        "examples", help="copy the example setup files into a directory")
    examples_parser.add_argument("directory", help="target directory")
    gui_parser = sub.add_parser("gui", help="serve the GUI in the browser")
    gui_parser.add_argument("--port", type=int, default=5006,
                            help="port of the server (default: 5006)")
    gui_parser.add_argument("--address", default=None,
                            help="address to listen on (default: all "
                            "interfaces; browsers of other hosts are refused "
                            "unless allowed with --allow-websocket-origin)")
    gui_parser.add_argument("--allow-websocket-origin", action="append",
                            metavar="HOST[:PORT]",
                            help="host allowed to open websocket "
                            "connections (repeatable; without it only "
                            "localhost:PORT is allowed)")
    gui_parser.add_argument("--prefix", default=None,
                            help="URL prefix of the app")
    gui_parser.add_argument("--show", action=argparse.BooleanOptionalAction,
                            default=True,
                            help="open the app in a browser (default: yes)")
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


def _copy_examples(directory: str) -> int:
    """Copy the examples to ``directory``; return the exit code."""
    try:
        paths = copy_examples(Path(directory))
    except OSError as exc:
        print("output error: cannot copy the examples to %s: %s"
              % (directory, exc), file=sys.stderr)
        return EXIT_OUTPUT
    for path in paths:
        print(path)
    return 0


def _gui(args: argparse.Namespace) -> int:
    """Serve the GUI; return the exit code."""
    try:
        from .gui.serve import serve
    except ImportError as exc:
        missing = (exc.name or "").split(".")[0]
        if isinstance(exc, ModuleNotFoundError) and missing in _GUI_PACKAGES:
            print('pypalc gui needs the GUI dependencies: '
                  'pip install "pypalc[gui]"', file=sys.stderr)
        else:
            print("pypalc gui could not start: %s" % exc, file=sys.stderr)
        return EXIT_GUI
    serve(port=args.port, address=args.address,
          allow_websocket_origin=args.allow_websocket_origin,
          prefix=args.prefix, show=args.show)
    return 0


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
        or calculation error, 5 output could not be written, 6 the GUI
        dependencies are missing or the GUI cannot start (``pypalc gui``). An error is
        printed to stderr with its cause; with ``--traceback`` the full
        (chained) traceback is printed too.
    """
    args = _parser().parse_args(argv)
    if args.command == "examples":
        return _copy_examples(args.directory)
    if args.command == "gui":
        return _gui(args)
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
