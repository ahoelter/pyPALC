"""Exchange of the GUI's input with a :class:`~pypalc.core.settings.Setup`.

:func:`setup_from_app` collects what the user entered on the pages of a
:class:`~pypalc.gui.app.PalcApp` into a setup. The GUI computes this setup
with :func:`pypalc.core.pipeline.run`, the same input path as the headless
runner. :func:`apply_setup` does the opposite for "Load Setup": it puts a
setup into a session and runs it.

author: Arne Hoelter
"""
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from pypalc.core.errors import SetupError, VenueError
from pypalc.core.geometry import reference_tilt_angles_deg
from pypalc.core.pipeline import build_venue
from pypalc.core.settings import (AlgorithmSettings, ArraySettings,
                                  LoudspeakerSettings, MeasuredDirectivity,
                                  ReferenceArraySettings, Setup,
                                  TargetPoint, TargetSlopeSettings,
                                  VenueSettings, validate)
from pypalc.core.sound_field import (SoundFieldConfig,
                                     parse_measured_directivity)
from pypalc.gui import CALLBACK_ERRORS
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
            not app.state.target.initialized:
        return None
    target = page.target_source.data
    try:
        points = tuple(TargetPoint(float(x), float(spl))
                       for x, spl in zip(target['distance_m'],
                                         target['level_db']))
        reference = float(
            page.target_reference_source.data['distance_m'][0])
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


def setup_to_save(app: PalcApp) -> Setup:
    """The setup "Save Setup" writes.

    This is :func:`setup_from_app`, except for a target slope that was not
    moved since the last calculation: after a calculation the GUI shows
    the target points snapped to the distance grid of the optimized array,
    and saving those would move them again when the file is loaded (it
    snaps them to the grid of the initial calculation). Then the target
    slope of that calculation is saved, so that loading the file gives the
    shown results. A target moved after the calculation is saved as shown.

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
        If an input cannot be read.
    """
    setup = setup_from_app(app)
    st = app.state
    if setup.target_slope is not None and st.result is not None and \
            st.result.setup.target_slope is not None and \
            st.target_of_result == app.pages[4].shown_target():
        setup = replace(setup, target_slope=st.result.setup.target_slope)
    return setup


# --------------------------------------------------------------------------
# Applying a setup ("Load Setup")
# --------------------------------------------------------------------------

def _labels(options):
    """Setting value -> option label."""
    return {value: label for label, value in options.items()}


#: Frequency range of the GUI's slider in Hz.
FREQUENCY_SLIDER_RANGE = (0.0, 20000.0)


def _typed(value: float) -> str:
    """Text a user types for a number: ``2.0`` -> ``'2'``."""
    text = repr(float(value))
    return text[:-2] if text.endswith('.0') else text


def _text(value: float, initial: str) -> str:
    """Text of a number input: its initial text if that is the value (a
    user leaves it), otherwise the typed number."""
    return initial if float(initial) == float(value) else _typed(value)


def _check_applicable(setup: Setup) -> None:
    """Raise a SetupError if the GUI cannot show or compute a setup.

    Checks everything :func:`apply_setup` could fail on before it changes
    the session: the settings (:func:`~pypalc.core.settings.validate`), the
    measured directivity, the venue, the reference array, the frequency
    range and the number of target points.
    """
    validate(setup)
    md = setup.loudspeaker.measured_directivity
    if md is not None and md.path is not None:
        raise SetupError("loudspeaker.measured_directivity.path",
                         "the GUI cannot read a file path")
    if md is not None and md.csv is not None:
        try:
            parse_measured_directivity(md.csv, SoundFieldConfig())
        except CALLBACK_ERRORS as exc:
            raise SetupError("loudspeaker.measured_directivity.csv",
                             "cannot be read as measured loudspeaker data"
                             ) from exc
    ar, ref, alg = setup.array, setup.reference_array, setup.algorithm
    if ar.discrete_tilt_angles_deg is not None and \
            len(ar.discrete_tilt_angles_deg) == 0:
        raise SetupError("array.discrete_tilt_angles_deg",
                         "at least one angle required")
    low, high = FREQUENCY_SLIDER_RANGE
    if not low <= alg.frequency_range_hz[0] < alg.frequency_range_hz[1] \
            <= high:
        raise SetupError("algorithm.frequency_range_hz",
                         "the GUI allows %g to %g Hz" % (low, high))
    try:
        build_venue(setup.venue, ar)
    except VenueError as exc:
        message = str(exc)
        if message.startswith('Error: '):
            message = message[len('Error: '):]
        raise SetupError("venue.audience_lines", message) from exc
    discrete_tilt_angles = [] if ar.discrete_tilt_angles_deg is None else \
        np.array(ar.discrete_tilt_angles_deg) * (np.pi/180)
    try:
        reference_tilt_angles_deg(
            ref.type, ar.cabinet_count, discrete_tilt_angles,
            ref.top_angle_deg, ref.angle_param_deg,
            ref.use_discrete_angles and ar.discrete_tilt_angles_deg
            is not None, ref.tilt_angles_deg)
    except CALLBACK_ERRORS as exc:
        raise SetupError("reference_array", str(exc)) from exc
    if alg.weighting == 'target_slope':
        lines = len(setup.venue.audience_lines)
        if len(setup.target_slope.points) - 2 > lines:
            raise SetupError(
                "target_slope.points",
                "at most %d points (%d hinges) for %d audience lines"
                % (lines + 2, lines, lines))


def _set_loudspeaker(app: PalcApp, ls: LoudspeakerSettings) -> None:
    page = app.pages[2]
    page.aperture_input.value = _text(ls.aperture_angle_deg, '3.0')
    page.cabinet_height_input.value = _text(ls.cabinet_height, '0.2')
    page.directivity_select.value = _labels(DIRECTIVITIES)[ls.directivity]


def _set_array(app: PalcApp, ar: ArraySettings,
               ref: ReferenceArraySettings) -> None:
    page = app.pages[3]
    count = ar.cabinet_count
    page.cabinet_count_input.value = '4' if count == 4 else str(int(count))
    page.top_x_input.value = _text(ar.top_position[0], '0')
    page.top_y_input.value = _text(ar.top_position[1], '2')
    page.cabinet_gap_input.value = _text(ar.cabinet_gap, '0')
    discrete = ar.discrete_tilt_angles_deg
    page.use_discrete_angles_select.value = 'No' if discrete is None \
        else 'Yes'
    page.discrete_angles_input.value = '' if discrete is None else \
        ','.join(_typed(a) for a in discrete)
    page.discrete_angles_input.disabled = discrete is None
    page.reference_discrete_angles_select.disabled = discrete is None
    # the widgets of a straight reference array are those of a new session
    page.reference_type_select.value = 'Straight'
    page.arrange_reference_widgets()
    page.reference_type_select.value = _labels(REFERENCE_TYPES)[ref.type]
    page.arrange_reference_widgets()
    page.reference_start_input.value = '0'
    page.reference_user_angles_input.value = ''
    if ref.type == 'user_defined':
        page.reference_user_angles_input.value = ','.join(
            _typed(a) for a in ref.tilt_angles_deg)
    else:
        page.reference_start_input.value = _text(ref.top_angle_deg, '0')
        if ref.type != 'straight':
            page.reference_step_stop_input.value = _text(
                ref.angle_param_deg, '0')
    page.reference_discrete_angles_select.value = 'Yes' \
        if ref.use_discrete_angles and discrete is not None else 'No'


def _set_algorithm(app: PalcApp, alg: AlgorithmSettings) -> None:
    """Set the algorithm widgets; weighting and fixed angle as in a new
    session (they are set after the venue check)."""
    page = app.pages[4]
    page.palc_variant_select.value = _labels(VARIANTS)[alg.variant]
    page.gap_handling_select.options = ['Without', 'Hard Margin',
                                        'Soft Margin']
    page.gap_handling_select.value = _labels(GAP_HANDLINGS)[alg.gap_handling]
    soft = alg.gap_handling == 'soft_margin'
    page.soft_margin_slider.disabled = not soft
    page.soft_margin_slider.value = float(alg.soft_margin_strength) \
        if soft else 1.0
    page.tolerance_slider.value = float(alg.tolerance)
    page.frequency_range_slider.value = tuple(
        float(f) for f in alg.frequency_range_hz)
    page.weighting_select.options = list(WEIGHTINGS)
    page.weighting_select.value = 'Without'
    page.weighting_select.width = 255
    page.hinges_select.visible = False
    page.weighting_nu_slider.value = 1.0
    for widget in (page.weighting_nu_slider, page.nu_plus_button,
                   page.nu_minus_button, page.nu_step_select):
        widget.disabled = True
    page.use_fixed_angle_select.value = 'No'
    page.use_fixed_angle_select.disabled = True
    page.fixed_angle_input.value = '0.0'
    page.fixed_angle_input.disabled = True
    page.show_initial_fixed_angle_range()
    page.start_toggle.active = False
    page.tabs.active = 0


def _set_fixed_angle(app: PalcApp, angle: Optional[float]) -> None:
    """Fix the angle of the uppermost cabinet (venue checked)."""
    if angle is None:
        return
    page, config = app.pages[4], app.state.config
    page.use_fixed_angle_select.value = 'Yes'
    page.update_fixed_angle_range()
    page.fixed_angle_input.disabled = False
    page.fixed_angle_input.value = _typed(angle)
    config.use_fixed_angle = True
    config.store_config('fixed_angle', page.fixed_angle_input.value)
    page.round_fixed_angle()


def _set_weighting(app: PalcApp, setup: Setup) -> bool:
    """Select the weighting; for target slope run the initial calculation
    and move the target points. False if that calculation failed."""
    page, alg = app.pages[4], setup.algorithm
    page.weighting_select.value = _labels(WEIGHTINGS)[alg.weighting]
    if not page.show_weighting():
        return False
    if alg.weighting in ('linear', 'logarithmic'):
        page.weighting_nu_slider.value = float(np.cbrt(alg.weighting_nu))
        app.state.config.weighting_nu = float(alg.weighting_nu)
    elif alg.weighting == 'target_slope':
        ts = setup.target_slope
        hinges = len(ts.points) - 2
        page.hinges_select.value = str(hinges)
        if hinges != 0:
            page.place_hinges(hinges)
        page.drag_target(ts.points, ts.reference_distance)
    return True


def apply_setup(app: PalcApp, setup: Setup) -> None:
    """Put a setup into a session and run it ("Load Setup").

    Leaves the session as a user who enters the setup by hand, page by
    page, and clicks "PALC Run": it replaces the whole session state,
    sets every widget, draws the audience lines, checks the venue
    (:meth:`~.app.PalcApp.check_venue`), fixes the angle of the uppermost
    cabinet, selects the weighting (for target slope: the initial
    calculation, the hinges, the target points and the reference point),
    runs the calculation (:meth:`~.app.PalcApp.run_calculation`) and shows
    page 5.

    While the widgets are set, ``state.applying_setup`` keeps their
    callbacks from running; this function does their work itself, once and
    in the right order.

    Everything that could make the setup unusable is checked before the
    session is changed; then a :class:`SetupError` is raised and nothing
    changes. Other exceptions (a bug) are not caught here: the session is
    then already replaced and may be partly applied. If the calculation
    itself fails (e.g. no convergence), the setup stays applied, the error
    text is shown like for a failed "PALC Run" and page 4 stays open.
    Unlike a failed "PALC Run", which keeps the previous results, the
    results of the previous setup are then cleared
    (:meth:`~.app.PalcApp.clear_results`), unless the initial calculation
    of a target slope setup succeeded (its results stay shown, as in
    ``state.result``).

    Parameters
    ----------
    app : PalcApp
        The session.
    setup : Setup
        The setup; a measured directivity must be embedded (``csv``).

    Returns
    -------
    None.

    Raises
    ------
    SetupError
        If the GUI cannot show or compute the setup (invalid settings, a
        measured directivity given as a path or unreadable, a venue that is
        not PALC compatible, ...).
    """
    from .app import SessionState
    _check_applicable(setup)
    venue, loudspeaker, array, algorithm, results = (
        app.pages[1], app.pages[2], app.pages[3], app.pages[4], app.pages[5])
    state = SessionState(applying_setup=True)
    app.state = state
    try:
        app.page_select.active = 4
        venue.lines_source.selected.indices = []
        results.venue_points_source.selected.indices = []
        loudspeaker.compute_modeled_directivity()  # as at start-up
        _set_loudspeaker(app, setup.loudspeaker)
        _set_array(app, setup.array, setup.reference_array)
        _set_algorithm(app, setup.algorithm)
        venue.draw_lines(setup.venue.audience_lines)
        md = setup.loudspeaker.measured_directivity
        if md is not None:
            loudspeaker.read_measured_directivity(md.csv)
        state.config.store_config(
            'use_discrete_angles', setup.array.discrete_tilt_angles_deg is not None)
        array.store_discrete_angles()
        app.store_config_widgets()
        array.update_reference_angles()
        app.check_venue()
        _set_fixed_angle(app, setup.algorithm.fixed_top_angle_deg)
        if _set_weighting(app, setup) and app.run_calculation():
            app.page_select.active = 5
        elif state.result is None:
            # nothing of this setup was calculated: show no results
            # instead of those of the previous setup
            app.clear_results()
    finally:
        state.applying_setup = False
