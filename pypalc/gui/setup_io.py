"""Exchange of the GUI's input with a :class:`~pypalc.core.settings.Setup`.

:func:`setup_from_app` collects what the user entered on the pages of a
:class:`~pypalc.gui.app.PalcApp` into a setup. The GUI computes this setup
with :func:`pypalc.core.pipeline.run`, the same input path as the headless
runner.

author: Arne Hoelter
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

from pypalc.core.errors import SetupError
from pypalc.core.settings import (AlgorithmSettings, ArraySettings,
                                  LoudspeakerSettings, MeasuredDirectivity,
                                  ReferenceArraySettings, Setup,
                                  TargetPoint, TargetSlopeSettings,
                                  VenueSettings)
from pypalc.gui.page_array import REFERENCE_TYPES

if TYPE_CHECKING:
    from .app import PalcApp

#: Option labels of the GUI -> setting values.
VARIANTS = {"PALC 1": "palc1", "PALC 2": "palc2", "PALC 3": "palc3"}
GAP_HANDLINGS = {"Without": "without", "Hard Margin": "hard_margin",
                 "Soft Margin": "soft_margin"}
WEIGHTINGS = {"Without": "without", "Linear Spacing": "linear",
              "Logarithmic Spacing": "logarithmic",
              "Target Slope": "target_slope"}
DIRECTIVITIES = {"Circular Piston": "circular_piston",
                 "Line Piston": "line_piston",
                 "Combined Circular/Line": "combined_circular_line",
                 "Measured Loudspeaker Data": "measured"}


def _number(text: str, key: str) -> float:
    """A number typed into a text input."""
    try:
        return float(text)
    except (TypeError, ValueError):
        raise SetupError(key, "expected a number, got %r" % (text,)) from None


def _numbers(text: str, key: str) -> Tuple[float, ...]:
    """Numbers typed into a text input, separated by ','."""
    try:
        return tuple(float(s) for s in text.split(','))
    except (TypeError, ValueError):
        raise SetupError(key, "expected numbers separated by ',', got %r"
                         % (text,)) from None


def _loudspeaker(app: PalcApp) -> LoudspeakerSettings:
    page = app.pages[2]
    directivity = DIRECTIVITIES[page.directivity_select.value]
    measured: Optional[MeasuredDirectivity] = None
    csv = app.state.measured_directivity_csv
    if directivity == "measured" and csv is not None:
        measured = MeasuredDirectivity(csv=csv)
    return LoudspeakerSettings(
        aperture_angle_deg=_number(page.aperture_input.value,
                                   "loudspeaker.aperture_angle_deg"),
        cabinet_height=_number(page.cabinet_height_input.value,
                               "loudspeaker.cabinet_height"),
        directivity=directivity, measured_directivity=measured)


def _array(app: PalcApp) -> ArraySettings:
    page = app.pages[3]
    try:
        cabinet_count = int(page.cabinet_count_input.value)
    except ValueError:
        raise SetupError("array.cabinet_count", "expected an integer, got %r"
                         % (page.cabinet_count_input.value,)) from None
    discrete = None
    if page.use_discrete_angles_select.value == 'Yes':
        discrete = _numbers(page.discrete_angles_input.value,
                            "array.discrete_tilt_angles_deg")
    return ArraySettings(
        cabinet_count=cabinet_count,
        top_position=(_number(page.top_x_input.value, "array.top_position"),
                      _number(page.top_y_input.value, "array.top_position")),
        cabinet_gap=_number(page.cabinet_gap_input.value, "array.cabinet_gap"),
        discrete_tilt_angles_deg=discrete)


def _reference_array(app: PalcApp) -> ReferenceArraySettings:
    page = app.pages[3]
    ref_type = REFERENCE_TYPES[page.reference_type_select.value]
    use_discrete = page.reference_discrete_angles_select.value == 'Yes'
    if ref_type == 'user_defined':
        return ReferenceArraySettings(
            type=ref_type, use_discrete_angles=use_discrete,
            tilt_angles_deg=_numbers(page.reference_user_angles_input.value,
                                     "reference_array.tilt_angles_deg"))
    angle_param = 0.0
    if ref_type != 'straight':
        angle_param = _number(page.reference_step_stop_input.value,
                              "reference_array.angle_param_deg")
    return ReferenceArraySettings(
        type=ref_type,
        top_angle_deg=_number(page.reference_start_input.value,
                              "reference_array.top_angle_deg"),
        angle_param_deg=angle_param, use_discrete_angles=use_discrete)


def _algorithm(app: PalcApp) -> AlgorithmSettings:
    page, config = app.pages[4], app.state.config
    # the configuration holds the fixed angle as clamped to its range and
    # ν as stored from the cubic slider
    fixed = float(config.fixed_angle) if config.use_fixed_angle else None
    return AlgorithmSettings(
        variant=VARIANTS[page.palc_variant_select.value],
        gap_handling=GAP_HANDLINGS[page.gap_handling_select.value],
        soft_margin_strength=float(page.soft_margin_slider.value),
        weighting=WEIGHTINGS[page.weighting_select.value],
        weighting_nu=float(config.weighting_nu),
        tolerance=float(page.tolerance_slider.value),
        fixed_top_angle_deg=fixed,
        frequency_range_hz=tuple(float(f) for f in
                                 page.frequency_range_slider.value))


def _target_slope(app: PalcApp) -> Optional[TargetSlopeSettings]:
    """The target slope as shown, once the initial run has shown one."""
    page = app.pages[4]
    if page.weighting_select.value != 'Target Slope' or \
            not app.state.target.init:
        return None
    target = page.target_source.data
    try:
        points = tuple(TargetPoint(float(x), float(spl))
                       for x, spl in zip(target['x'], target['SPL']))
        reference = float(page.target_reference_source.data['x_ref'][0])
    except (TypeError, ValueError, IndexError):
        raise SetupError("target_slope", "no target slope shown") from None
    return TargetSlopeSettings(points=points, reference_distance=reference)


def setup_from_app(app: PalcApp) -> Setup:
    """The setup the user entered in the GUI.

    Reads the widgets of the configuration pages, the drawn audience lines
    and the uploaded measured directivity of the session. Values that the
    GUI stores in a processed form are taken from the session's PALC
    configuration: the fixed angle of the uppermost cabinet (clamped to its
    range, None if not fixed) and ν (the cube of the slider value). The
    target slope is the one shown on page 4 (None before the initial run of
    "Target Slope").

    Parameters
    ----------
    app : PalcApp
        The GUI session.

    Returns
    -------
    Setup
        The setup; it is not validated.

    Raises
    ------
    SetupError
        If an input cannot be read (e.g. text in a number input).
    """
    return Setup(
        venue=VenueSettings(audience_lines=tuple(app.state.audience_lines)),
        loudspeaker=_loudspeaker(app), array=_array(app),
        reference_array=_reference_array(app), algorithm=_algorithm(app),
        target_slope=_target_slope(app))
