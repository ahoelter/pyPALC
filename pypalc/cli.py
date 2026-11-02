"""Command line interface: ``python -m pypalc run SETUP --out DIR``."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .core.errors import (CalculationError, ConvergenceError, SetupError,
                          VenueError)
from .core.export import write_results
from .core.pipeline import run
from .core.settings import load_setup

EXIT_SETUP = 2
EXIT_VENUE = 3
EXIT_CALCULATION = 4


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
    return parser


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
        or calculation error.
    """
    args = _parser().parse_args(argv)
    try:
        try:
            setup = load_setup(args.setup)
        except (OSError, UnicodeDecodeError) as exc:
            raise SetupError("setup", "cannot read %s: %s"
                             % (args.setup, exc))
        result = run(setup)
    except SetupError as exc:
        print("setup error at %s: %s" % (exc.key, exc.message),
              file=sys.stderr)
        return EXIT_SETUP
    except VenueError as exc:
        print("venue error: %s" % exc, file=sys.stderr)
        return EXIT_VENUE
    except (ConvergenceError, CalculationError) as exc:
        print("calculation error: %s" % exc, file=sys.stderr)
        return EXIT_CALCULATION
    for path in write_results(result, Path(args.out)):
        print(path)
    return 0
