"""Exception types raised by the pyPALC core."""
from __future__ import annotations


class PalcError(Exception):
    """Base class of all pyPALC errors."""


class SetupError(PalcError):
    """A setup file or setup object is invalid.

    Parameters
    ----------
    key : str
        Dotted path of the offending setting, e.g. ``"array.cabinet_count"``.
    message : str
        Description of the problem.
    """

    def __init__(self, key: str, message: str) -> None:
        super().__init__("%s: %s" % (key, message))
        self.key = key
        self.message = message


class VenueError(PalcError):
    """The venue (audience lines) is unusable for a calculation."""


class ConvergenceError(PalcError):
    """An iterative calculation did not converge."""


class CalculationError(PalcError):
    """A calculation failed."""
