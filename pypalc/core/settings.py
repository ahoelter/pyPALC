"""Setup settings of a pyPALC calculation and their YAML representation.

A :class:`Setup` bundles everything the user enters in the GUI. It can be
read from and written to YAML setup files.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, get_type_hints

import yaml

from .errors import SetupError

SETUP_VERSION = 1


@dataclass(frozen=True)
class AudienceLine:
    """A straight audience line.

    Parameters
    ----------
    start, stop : tuple of float
        End points ``(x, y)`` in m.
    points_per_meter : float
        Discretization of this line, as entered when it was saved.
    """
    start: Tuple[float, float]
    stop: Tuple[float, float]
    points_per_meter: float = 1.0


@dataclass(frozen=True)
class VenueSettings:
    """Venue description: the audience lines."""
    audience_lines: Tuple[AudienceLine, ...] = ()


@dataclass(frozen=True)
class MeasuredDirectivity:
    """Source of a measured directivity; exactly one of the fields is set."""
    path: Optional[str] = None
    csv: Optional[str] = None


@dataclass(frozen=True)
class LoudspeakerSettings:
    """Loudspeaker cabinet properties."""
    aperture_angle_deg: float = 3.0
    cabinet_height: float = 0.2
    directivity: str = "circular_piston"
    measured_directivity: Optional[MeasuredDirectivity] = None


@dataclass(frozen=True)
class ArraySettings:
    """Array geometry."""
    cabinet_count: int = 4
    top_position: Tuple[float, float] = (0.0, 2.0)
    cabinet_gap: float = 0.0
    discrete_tilt_angles_deg: Optional[Tuple[float, ...]] = None


@dataclass(frozen=True)
class ReferenceArraySettings:
    """Reference array against which the optimized array is compared."""
    type: str = "straight"
    top_angle_deg: float = 0.0
    angle_param_deg: float = 0.0
    use_discrete_angles: bool = False
    tilt_angles_deg: Tuple[float, ...] = ()


@dataclass(frozen=True)
class AlgorithmSettings:
    """Optimization algorithm settings."""
    variant: str = "palc2"
    gap_handling: str = "without"
    soft_margin_strength: float = 1.0
    weighting: str = "without"
    weighting_nu: float = 1.0
    tolerance: float = 1.0
    fixed_top_angle_deg: Optional[float] = None
    frequency_range_hz: Tuple[float, float] = (200.0, 8000.0)


@dataclass(frozen=True)
class TargetPoint:
    """A point of the target slope: distance in m, level in dB."""
    distance: float
    level_db: float


@dataclass(frozen=True)
class TargetSlopeSettings:
    """Target slope given by its start, hinges and end points."""
    points: Tuple[TargetPoint, ...]
    reference_distance: float


@dataclass(frozen=True)
class Setup:
    """Complete setup of a calculation."""
    venue: VenueSettings = VenueSettings()
    loudspeaker: LoudspeakerSettings = LoudspeakerSettings()
    array: ArraySettings = ArraySettings()
    reference_array: ReferenceArraySettings = ReferenceArraySettings()
    algorithm: AlgorithmSettings = AlgorithmSettings()
    target_slope: Optional[TargetSlopeSettings] = None


_DIRECTIVITIES = ("circular_piston", "line_piston", "combined_circular_line",
                  "measured")
_REFERENCE_TYPES = ("straight", "progressive", "arc", "user_defined")
_VARIANTS = ("palc1", "palc2", "palc3")
_GAP_HANDLING = ("without", "hard_margin", "soft_margin")
_WEIGHTINGS = ("without", "linear", "logarithmic", "target_slope")


# --------------------------------------------------------------------------
# Building dataclasses from parsed YAML
# --------------------------------------------------------------------------

def _type_name(tp: Any) -> str:
    return getattr(tp, "__name__", str(tp))


def _convert(tp: Any, value: Any, key: str) -> Any:
    """Convert ``value`` to type ``tp``, raising SetupError(key) on mismatch."""
    origin = getattr(tp, "__origin__", None)
    if origin is Union:  # Optional[X]
        if value is None:
            return None
        inner = [a for a in tp.__args__ if a is not type(None)][0]
        return _convert(inner, value, key)
    if origin is tuple:
        args = tp.__args__
        if not isinstance(value, (list, tuple)):
            raise SetupError(key, "expected a list")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_convert(args[0], v, "%s[%d]" % (key, i))
                         for i, v in enumerate(value))
        if len(value) != len(args):
            raise SetupError(key, "expected a list of %d entries" % len(args))
        return tuple(_convert(a, v, "%s[%d]" % (key, i))
                     for i, (a, v) in enumerate(zip(args, value)))
    if dataclasses.is_dataclass(tp):
        return _build(tp, value, key)
    if tp is bool:
        if not isinstance(value, bool):
            raise SetupError(key, "expected true or false")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise SetupError(key, "expected an integer")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SetupError(key, "expected a number")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise SetupError(key, "expected a string")
        return value
    raise TypeError("unsupported type %s" % _type_name(tp))  # pragma: no cover


def _build(cls: Any, data: Any, prefix: str) -> Any:
    """Build dataclass ``cls`` from mapping ``data``; ``prefix`` is the key path."""
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise SetupError(prefix or "setup", "expected a mapping")
    hints = get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    join = (prefix + ".") if prefix else ""
    for name in data:
        if name not in fields:
            raise SetupError(join + str(name), "unknown setting")
    kwargs = {}
    for name, f in fields.items():
        if name in data:
            kwargs[name] = _convert(hints[name], data[name], join + name)
        elif (f.default is dataclasses.MISSING
              and f.default_factory is dataclasses.MISSING):
            raise SetupError(join + name, "missing required setting")
    return cls(**kwargs)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def _check_range(key: str, value: float, low: float, high: float) -> None:
    if not low <= value <= high:
        raise SetupError(key, "must be between %g and %g, got %g"
                         % (low, high, value))


def _check_choice(key: str, value: str, options: Tuple[str, ...]) -> None:
    if value not in options:
        raise SetupError(key, "must be one of %s, got %r"
                         % (", ".join(options), value))


def validate(setup: Setup) -> None:
    """Check ranges, enumerations and cross-field constraints.

    Parameters
    ----------
    setup : Setup
        Setup to check.

    Raises
    ------
    SetupError
        With ``key`` naming the first offending setting.
    """
    for i, line in enumerate(setup.venue.audience_lines):
        if not line.points_per_meter > 0:
            raise SetupError("venue.audience_lines[%d].points_per_meter" % i,
                             "must be greater than 0")

    ls = setup.loudspeaker
    if not ls.cabinet_height > 0:
        raise SetupError("loudspeaker.cabinet_height", "must be greater than 0")
    _check_choice("loudspeaker.directivity", ls.directivity, _DIRECTIVITIES)
    if ls.directivity == "measured":
        md = ls.measured_directivity
        if md is None or (md.path is None) == (md.csv is None):
            raise SetupError("loudspeaker.measured_directivity",
                             "exactly one of path and csv is required")

    ar = setup.array
    _check_range("array.cabinet_count", ar.cabinet_count, 1, 30)
    if ar.cabinet_gap < 0:
        raise SetupError("array.cabinet_gap", "must not be negative")

    _check_choice("reference_array.type", setup.reference_array.type,
                  _REFERENCE_TYPES)

    alg = setup.algorithm
    _check_choice("algorithm.variant", alg.variant, _VARIANTS)
    _check_choice("algorithm.gap_handling", alg.gap_handling, _GAP_HANDLING)
    _check_choice("algorithm.weighting", alg.weighting, _WEIGHTINGS)
    _check_range("algorithm.soft_margin_strength", alg.soft_margin_strength,
                 0.1, 1.9)
    _check_range("algorithm.weighting_nu", alg.weighting_nu, 0.01, 10)
    _check_range("algorithm.tolerance", alg.tolerance, 0.05, 5)
    low, high = alg.frequency_range_hz
    if not low < high:
        raise SetupError("algorithm.frequency_range_hz",
                         "lower frequency must be below the upper one")

    if alg.weighting == "target_slope":
        if setup.target_slope is None:
            raise SetupError("target_slope",
                             "required when weighting is target_slope")
        if alg.gap_handling == "hard_margin":
            raise SetupError("algorithm.gap_handling",
                             "hard_margin cannot be combined with "
                             "weighting target_slope")
    if setup.target_slope is not None and len(setup.target_slope.points) < 2:
        raise SetupError("target_slope.points", "at least 2 points required")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_setup_text(text: str, base_dir: Optional[Path] = None) -> Setup:
    """Parse and validate a setup from YAML text.

    Parameters
    ----------
    text : str
        YAML document.
    base_dir : pathlib.Path, optional
        Directory against which a relative ``measured_directivity.path`` is
        resolved. If None, a ``path`` directivity is rejected.

    Returns
    -------
    Setup

    Raises
    ------
    SetupError
        If the text is not valid YAML or the setup is invalid.
    """
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SetupError("setup", "invalid YAML: %s" % exc)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise SetupError("setup", "expected a mapping at top level")
    data = dict(data)
    version = data.pop("version", SETUP_VERSION)
    if version != SETUP_VERSION or isinstance(version, bool):
        raise SetupError("version", "unsupported version %r, expected %d"
                         % (version, SETUP_VERSION))
    setup = _build(Setup, data, "")
    validate(setup)

    md = setup.loudspeaker.measured_directivity
    if md is not None and md.path is not None:
        if base_dir is None:
            raise SetupError("loudspeaker.measured_directivity.path",
                             "a file path needs a setup file to be relative "
                             "to; embed the data as csv instead")
        resolved = (Path(base_dir) / md.path).resolve()
        md = dataclasses.replace(md, path=str(resolved))
        setup = dataclasses.replace(
            setup, loudspeaker=dataclasses.replace(
                setup.loudspeaker, measured_directivity=md))
    return setup


def load_setup(path: Union[str, Path]) -> Setup:
    """Load and validate a setup file.

    A relative ``measured_directivity.path`` is resolved against the
    directory of the setup file and stored as an absolute path.

    Parameters
    ----------
    path : str or pathlib.Path
        YAML setup file.

    Returns
    -------
    Setup

    Raises
    ------
    SetupError
        If the file is invalid.
    """
    path = Path(path)
    return load_setup_text(path.read_text(encoding="utf-8"),
                           base_dir=path.resolve().parent)


# --------------------------------------------------------------------------
# Saving
# --------------------------------------------------------------------------

class _Literal(str):
    """String dumped as YAML literal block."""


class _Dumper(yaml.SafeDumper):
    pass


_Dumper.add_representer(
    _Literal, lambda d, v: d.represent_scalar("tag:yaml.org,2002:str", str(v),
                                              style="|"))


def _to_plain(value: Any, name: str = "") -> Any:
    if dataclasses.is_dataclass(value):
        return {f.name: _to_plain(getattr(value, f.name), f.name)
                for f in dataclasses.fields(value)
                if getattr(value, f.name) is not None}
    if isinstance(value, tuple):
        return [_to_plain(v) for v in value]
    if name == "csv" and isinstance(value, str):
        return _Literal(value)
    return value


def dump_setup(setup: Setup) -> str:
    """Serialize a setup to YAML text.

    Parameters
    ----------
    setup : Setup

    Returns
    -------
    str
        YAML document. Unset (None) settings are omitted.
    """
    data = {"version": SETUP_VERSION}
    data.update(_to_plain(setup))
    return yaml.dump(data, Dumper=_Dumper, sort_keys=False)


def save_setup(setup: Setup, path: Union[str, Path]) -> None:
    """Write a setup to a YAML file.

    Parameters
    ----------
    setup : Setup
    path : str or pathlib.Path
        Destination file.
    """
    Path(path).write_text(dump_setup(setup), encoding="utf-8")
