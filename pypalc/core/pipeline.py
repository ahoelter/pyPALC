"""One pipeline from a :class:`~pypalc.core.settings.Setup` to the results.

:func:`run` computes a setup the way the GUI does when a user enters it page
by page and clicks "PALC Run": it builds the venue (:func:`build_venue`),
runs the PALC algorithm, the sound field prediction of the optimized and the
reference array, and for target slope weighting the initial run and the
target slope optimization.

Every call builds fresh calculation objects, so calls are independent of
each other; neither the setup nor a venue passed in is modified.

GUI behavior reproduced here
----------------------------
The results equal the GUI's (``tests/baseline``). This needs some GUI state
that is not visible in a setup:

* The configuration starts as after the GUI's start-up: every run starts
  from the entered aperture angle; after every run the GUI resets the
  aperture angles to it.
* A mounting height that is too low for the array is raised as the GUI does
  (see :func:`top_position_y`).
* After the PALC run the GUI resets ``x_H``, ``y_H``, the aperture angles
  (and with them the start tilt angles) and ``last_angle_hm``, which the
  hard margin approach changes, before the results are visualized.
* Target slope weighting is two runs: an initial run without weighting
  (selecting "Target Slope" in the GUI), whose SPL over distance grid the
  target points and the reference point snap to, and the optimization run,
  which starts from the reset configuration. Afterwards the GUI snaps the
  shown target points to the grid of the optimization run.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .errors import CalculationError, ConvergenceError, SetupError, VenueError
from .geometry import (cabinet_outlines, inter_cabinet_angles,
                       reference_tilt_angles_deg)
from .palc import (Array_opt, PALC_configuration, PALC_plotting, calcPALC,
                   clamp_fixed_angle, fixed_angle_borders)
from .settings import (ArraySettings, AudienceLine, Setup, TargetPoint,
                       TargetSlopeSettings, VenueSettings)
from .sound_field import (SFP_configuration, Technical_meas_results,
                          calcHistogram, calcHomogeneity, calcSFP,
                          calcSPLoverX, get_freq_vec,
                          parse_measured_directivity)
from .target_slope import (Opt_weight, get_opt_region, get_weight_links,
                           optimize_PALC, shift2ref, shift_ref_on_zero,
                           snap_to_distance_grid)
from .venue import (AudienceLines, PalcVenue, check_draw_conditions,
                    fill_palc_venue, suggest_venue)

logger = logging.getLogger(__name__)

# Final names of the legacy classes (renamed in a later step).
ArrayEvaluation = Technical_meas_results
TargetSlopeState = Opt_weight
ArrayGeometry = Array_opt
RayPlotData = PALC_plotting

#: Maximum number of discrete venue points.
MAX_VENUE_POINTS = 2000
#: Maximum number of loops of the target slope optimization.
TARGET_SLOPE_MAX_LOOPS = 200
#: Number of cabinets the GUI's configuration starts with.
GUI_INITIAL_CABINET_COUNT = 4
#: Distance of the reference point in a new GUI session in m.
GUI_INITIAL_REFERENCE_DISTANCE = 10.0

# Exceptions the legacy calculation code raises on unusable input (the GUI
# catches all exceptions there and shows an error text).
_CALCULATION_ERRORS = (ArithmeticError, AttributeError, LookupError,
                       TypeError, ValueError)

# Setting values -> option labels used by the legacy configuration objects.
_VARIANTS = {"palc1": "PALC 1", "palc2": "PALC 2", "palc3": "PALC 3"}
_GAP_HANDLING = {"without": "Without", "hard_margin": "Hard Margin",
                 "soft_margin": "Soft Margin"}
_WEIGHTING = {"without": "Without", "linear": "Linear Spacing",
              "logarithmic": "Logarithmic Spacing",
              "target_slope": "Target Slope"}
_DIRECTIVITY = {"circular_piston": "Circular Piston",
                "line_piston": "Line Piston",
                "combined_circular_line": "Combined Circular/Line",
                "measured": "Measured Loudspeaker Data"}


@dataclass
class VenueResult:
    """The venue: drawn audience lines and the PALC compatible venue.

    Attributes
    ----------
    drawn : AudienceLines
        The audience lines as drawn, discretized (legacy ``created_pal``).
    palc_venue : PalcVenue
        Venue in the format of the PALC algorithm (legacy ``PALC_pal``).
    non_audience : AudienceLines
        Suggested non-audience lines in gaps (legacy ``suggested_nal``).
    sound_field_only : AudienceLines
        Audience lines only used in the sound field prediction (legacy
        ``suggested_sfp``).
    """
    drawn: AudienceLines
    palc_venue: PalcVenue
    non_audience: AudienceLines
    sound_field_only: AudienceLines


@dataclass
class ArrayResult:
    """Results of one array (the optimized or the reference array).

    Attributes
    ----------
    tilt_angles_deg : numpy.ndarray
        Tilt angles of the cabinets in degree; for the optimized array
        rounded to 0.01 deg as the GUI shows them.
    inter_cabinet_angles_deg : numpy.ndarray
        Angles between adjacent cabinets in degree, rounded to 0.01 deg.
    evaluation : ArrayEvaluation
        Sound field prediction, homogeneity and histogram.
    spl_over_distance : TargetSlopeState
        SPL over distance (attributes ``x`` and ``SPL``) as plotted.
    patches_x, patches_y : list
        Cabinet outlines for plotting.
    """
    tilt_angles_deg: np.ndarray
    inter_cabinet_angles_deg: np.ndarray
    evaluation: ArrayEvaluation
    spl_over_distance: TargetSlopeState
    patches_x: list
    patches_y: list


@dataclass
class RunResult:
    """Everything :func:`run` computed.

    Attributes
    ----------
    setup : Setup
        The setup that was run.
    venue : VenueResult
        The venue that was used.
    optimized : ArrayResult
        Results of the PALC optimized array.
    reference : ArrayResult
        Results of the reference array.
    geometry : ArrayGeometry
        Geometry of the optimized array (rays, hit points, ``num_iter``).
    ray_plot : RayPlotData
        Plot data of the optimized array.
    target : TargetSlopeState or None
        Final target slope state for target slope runs, otherwise None.
    weighting_weights : numpy.ndarray
        Weighting factors of the cabinets used in the final PALC run.
    """
    setup: Setup
    venue: VenueResult
    optimized: ArrayResult
    reference: ArrayResult
    geometry: ArrayGeometry
    ray_plot: RayPlotData
    target: Optional[TargetSlopeState]
    weighting_weights: np.ndarray


# --------------------------------------------------------------------------
# Venue
# --------------------------------------------------------------------------

def _new_audience_lines() -> AudienceLines:
    """Empty audience lines, with fresh lists for all attributes."""
    return AudienceLines(line_start_x=[], line_stop_x=[], line_start_y=[],
                         line_stop_y=[], slope=[], intercept=[], points_x=[],
                         points_y=[], line_end_indices=[], plot_x=[],
                         plot_y=[], cut_lines=[])


def _checked_line(lines: AudienceLines, line: AudienceLine
                  ) -> Tuple[float, float, float, float, float, float, float]:
    """Check a line as ``save_line`` does; return its values for appending."""
    x_start, y_start = float(line.start[0]), float(line.start[1])
    x_stop, y_stop = float(line.stop[0]), float(line.stop[1])
    condition1, condition2, condition3, slope, intercept, message = \
        check_draw_conditions(x_start, x_stop, y_start, y_stop, lines)
    if not (condition1 and condition2 and condition3):
        raise VenueError(message or 'Error: The audience line lies on an '
                         'already drawn audience line')
    diff = np.sqrt((x_stop - x_start)**2 + (y_stop - y_start)**2)
    eval_points = diff * float(line.points_per_meter)
    # AudienceLines.append_line discretizes with int(eval_points) points
    new_points = int(eval_points) if eval_points else 0
    if np.shape(lines.points_x)[0] + new_points > MAX_VENUE_POINTS:
        raise VenueError('Error: Maximum discretized points are %d. Please '
                         'reduce the discretization.' % MAX_VENUE_POINTS)
    return x_start, x_stop, y_start, y_stop, slope, intercept, eval_points


def check_line(lines: AudienceLines, line: AudienceLine) -> None:
    """Check whether an audience line may be appended to drawn lines.

    Parameters
    ----------
    lines : AudienceLines
        The lines drawn so far.
    line : AudienceLine
        The line to append.

    Raises
    ------
    VenueError
        If a draw condition fails (the message is the GUI's error text) or
        the venue would have more than :data:`MAX_VENUE_POINTS` points.
    """
    _checked_line(lines, line)


def build_venue(venue: VenueSettings, array: ArraySettings) -> VenueResult:
    """Draw the audience lines and create the PALC compatible venue.

    This is "Save Current Audience Line" for every line followed by "Get
    PALC Venue Proposal" in the GUI.

    Parameters
    ----------
    venue : VenueSettings
        The audience lines.
    array : ArraySettings
        The array settings. The venue proposal does not depend on them; the
        parameter keeps the interface stable.

    Returns
    -------
    VenueResult

    Raises
    ------
    VenueError
        If there are no lines, a line is rejected or no PALC compatible
        venue can be created.
    """
    if not venue.audience_lines:
        raise VenueError('Error: Please create audience lines first')
    drawn = _new_audience_lines()
    drawn.line_end_indices = np.array([0]).astype(int)
    for line in venue.audience_lines:
        values = _checked_line(drawn, line)
        drawn.append_line(*values)
    non_audience = _new_audience_lines()
    sound_field_only = _new_audience_lines()
    palc_venue = PalcVenue(segments=[], audience_segments=[])
    try:
        suggest_venue(drawn, non_audience, sound_field_only)
        fill_palc_venue(palc_venue, drawn, non_audience,
                                sound_field_only)
        palc_venue.collect_audience_segments(non_audience)
    except _CALCULATION_ERRORS as exc:
        raise VenueError('Error: Could not create PAL proposal') from exc
    if len(palc_venue.segments) == 0:
        raise VenueError('Error: Could not create PAL proposal')
    return VenueResult(drawn=drawn, palc_venue=palc_venue,
                       non_audience=non_audience,
                       sound_field_only=sound_field_only)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

def top_position_y(setup: Setup) -> float:
    """Height of the uppermost cabinet the GUI uses.

    The GUI raises the mounting height to the length of the array, rounded
    to 0.01 m, if the array does not fit below it.

    Parameters
    ----------
    setup : Setup

    Returns
    -------
    float
        Mounting height (``y_H``) in m.
    """
    ar = setup.array
    length = ar.cabinet_count * (setup.loudspeaker.cabinet_height
                                 + ar.cabinet_gap)
    y_H = float(ar.top_position[1])
    if length > y_H:
        y_H = np.round(length, decimals=2)
    return y_H


def _palc_config(setup: Setup, use_weighting: Optional[str] = None
                 ) -> PALC_configuration:
    """PALC configuration as the GUI holds it before the first run."""
    ls, ar, alg = setup.loudspeaker, setup.array, setup.algorithm
    if ar.discrete_tilt_angles_deg is not None:
        gamma_LSA = np.array([float(a) for a in ar.discrete_tilt_angles_deg]) \
            * (np.pi / 180)
    else:
        gamma_LSA = []
    if use_weighting is None:
        use_weighting = _WEIGHTING[alg.weighting]
    weighting_nu = alg.weighting_nu \
        if alg.weighting in ("linear", "logarithmic") else 1
    config = PALC_configuration(
        Lambda_gap=float(ar.cabinet_gap), Lambda_y=float(ls.cabinet_height),
        x_H=float(ar.top_position[0]), y_H=top_position_y(setup),
        constraint=_VARIANTS[alg.variant],
        use_gamma_LSA=ar.discrete_tilt_angles_deg is not None,
        gamma_LSA=gamma_LSA, use_weighting=use_weighting,
        weighting_nu=weighting_nu, weighting_factors=[],
        N=GUI_INITIAL_CABINET_COUNT,
        gap_handling=_GAP_HANDLING[alg.gap_handling],
        strength_sm=alg.soft_margin_strength
        if alg.gap_handling == "soft_margin" else 1,
        last_angle_hm=[], gap_weights=[], weighting_weights=[],
        use_fixed_angle=False, fixed_angle=0,
        directivity=_DIRECTIVITY[ls.directivity], tolerance=float(alg.tolerance),
        freq_range=tuple(alg.frequency_range_hz))
    # sets the start values of aperture and tilt angles, then N, as the GUI
    config.store_config('psi_n', float(ls.aperture_angle_deg))
    config.store_config('N', ar.cabinet_count)
    return config


def _measured_csv_text(setup: Setup) -> str:
    md = setup.loudspeaker.measured_directivity
    if md is None:
        raise SetupError("loudspeaker.measured_directivity",
                         "required for a measured directivity")
    if md.csv is not None:
        return md.csv
    try:
        return Path(md.path).read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise SetupError("loudspeaker.measured_directivity.path",
                         "cannot read %s: %s" % (md.path, exc)) from exc


def _sound_field_config(setup: Setup) -> SFP_configuration:
    """Sound field prediction configuration as the GUI holds it."""
    ls, ar, alg = setup.loudspeaker, setup.array, setup.algorithm
    config = SFP_configuration(
        dir_meas=np.zeros([2, 120]), dir_meas_deg=np.zeros([2, 120]),
        dir_meas_amp=[], gamma_n=[0], f=[], plot_dir_amp=[],
        plot_dir_amp_meas=[[0], [0], [0]], plot_beta_meas=[0])
    # SFP_configuration.__init__ calls PALC_configuration.__init__() with
    # its defaults, which also resets freq_range (set below).
    config.N = ar.cabinet_count
    config.Lambda_y = float(ls.cabinet_height)
    config.Lambda_gap = float(ar.cabinet_gap)
    config.x_H = float(ar.top_position[0])
    config.y_H = top_position_y(setup)
    config.directivity = _DIRECTIVITY[ls.directivity]
    config.freq_range = tuple(alg.frequency_range_hz)
    config.f = get_freq_vec(N_freq=120, step_freq=1/12,
                            freq_range=[20, 20000])
    if ls.directivity == "measured":
        try:
            parse_measured_directivity(_measured_csv_text(setup), config)
        except _CALCULATION_ERRORS as exc:
            raise SetupError("loudspeaker.measured_directivity",
                             "could not read the measured loudspeaker "
                             "data") from exc
    return config


def _reference_array(setup: Setup, config: PALC_configuration) -> Array_opt:
    """Reference array as the GUI computes it on page 3."""
    ref = setup.reference_array
    try:
        gamma_tilt_deg = reference_tilt_angles_deg(
            ref.type, config.N, config.gamma_LSA, ref.top_angle_deg,
            ref.angle_param_deg,
            ref.use_discrete_angles and bool(config.use_gamma_LSA),
            ref.tilt_angles_deg)
    except ValueError as exc:
        raise SetupError("reference_array", str(exc)) from exc
    reference = Array_opt(gamma_tilt_deg=gamma_tilt_deg)
    reference.gamma_n = reference.gamma_tilt_deg * (np.pi / 180)
    return reference


def fixed_angle_limits(venue: VenueResult, setup: Setup
                       ) -> Tuple[float, float]:
    """Allowed range of the fixed tilt angle of the uppermost cabinet.

    Parameters
    ----------
    venue : VenueResult
    setup : Setup

    Returns
    -------
    tuple of float
        ``(max_deg, min_deg)``, as the GUI computes them when "fix angle of
        uppermost cabinet" is enabled.
    """
    borders = fixed_angle_borders(venue.palc_venue.segments[-1],
                                  _palc_config(setup))
    return borders[0], borders[2]


# --------------------------------------------------------------------------
# Calculation
# --------------------------------------------------------------------------

class _Calculation:
    """Calculation objects of one GUI session and the run (``start_calc``).

    Parameters
    ----------
    setup : Setup
    venue : VenueResult
    use_weighting : str, optional
        Weighting label to start with; default from the setup.
    """

    def __init__(self, setup: Setup, venue: VenueResult,
                 use_weighting: Optional[str] = None) -> None:
        self.setup, self.venue = setup, venue
        self.config = _palc_config(setup, use_weighting)
        self.sf_config = _sound_field_config(setup)
        self.optimized = Array_opt(gamma_tilt_deg=[])
        self.reference = _reference_array(setup, self.config)
        self.plots, self.plots_ref = PALC_plotting(), PALC_plotting()
        self.evaluation = Technical_meas_results()
        self.evaluation_ref = Technical_meas_results()
        self.target = Opt_weight(SPL_interp=[0, 0])
        self.spl = Opt_weight(SPL_interp=[0])
        self.spl_ref = Opt_weight(SPL_interp=[0])
        # SPL over distance grid as shown (distance, venue x, venue y), and
        # the target points and reference point as shown in the GUI
        self.grid: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self.shown_target: Optional[Dict[str, list]] = None
        self.shown_reference: List[float] = [GUI_INITIAL_REFERENCE_DISTANCE]
        self.tilt_angles_deg: Optional[np.ndarray] = None
        self.patches: Tuple[list, list] = ([], [])
        self.patches_ref: Tuple[list, list] = ([], [])
        fixed = setup.algorithm.fixed_top_angle_deg
        if fixed is not None:
            borders = fixed_angle_borders(venue.palc_venue.segments[-1],
                                          self.config)
            self.optimized.fixed_angle = borders
            self.config.use_fixed_angle = True
            self.config.fixed_angle = clamp_fixed_angle(
                float(fixed), borders[0], borders[2])

    def show_target(self, x: list, SPL: list, x_v: list, y_v: list) -> None:
        """Set the shown target points and snap them to the grid."""
        self.shown_target = dict(x=list(x), SPL=list(SPL), x_v=list(x_v),
                                 y_v=list(y_v))
        self.snap_target()

    def show_reference(self, distance: float) -> None:
        """Set the shown reference point and snap it to the grid."""
        self.shown_reference = [distance]
        self.snap_target()

    def snap_target(self) -> None:
        """Snap the shown target to the grid (``update_draw``)."""
        x, x_v, y_v = self.grid
        snap_to_distance_grid(self.target, x, x_v, y_v, self.shown_target,
                              self.shown_reference)
        t = self.target
        self.shown_target = dict(x=t.x, SPL=t.SPL, x_v=t.x_v, y_v=t.y_v)
        self.shown_reference = [t.x_ref]

    def _reset_config(self) -> None:
        """Values the GUI resets after the PALC run (``start_calc``)."""
        ls, ar = self.setup.loudspeaker, self.setup.array
        for key, val in (("last_angle_hm", []),
                         ("x_H", float(ar.top_position[0])),
                         ("y_H", top_position_y(self.setup)),
                         ("psi_n", float(ls.aperture_angle_deg))):
            self.config.store_config(key, val)

    def calculate(self) -> None:
        """Run PALC and the sound field prediction (``start_calc``).

        Raises
        ------
        ConvergenceError
            If PALC did not converge.
        CalculationError
            If the calculation failed.
        """
        P, S = self.config, self.sf_config
        drawn, palc_venue = self.venue.drawn, self.venue.palc_venue
        P.get_initial_weights()
        target_run = P.use_weighting == 'Target Slope' and self.target.init
        opt_ind = None
        try:
            if target_run:
                self.target.calc_init()
                self.snap_target()
                opt_ind = optimize_PALC(P, S, palc_venue, self.plots,
                                        self.optimized, drawn,
                                        self.evaluation, self.spl,
                                        self.target, TARGET_SLOPE_MAX_LOOPS)
            else:
                calcPALC(self.plots, palc_venue, P, self.optimized)
        except _CALCULATION_ERRORS as exc:
            raise CalculationError('Error: PALC calculation failed') from exc
        self._reset_config()
        if self.optimized.num_iter == 500 or \
                np.shape(self.optimized.gamma_tilt_deg)[0] == 0:
            raise ConvergenceError('Error: No convergence reached')
        self.patches = cabinet_outlines(self.plots, P,
                                         self.optimized.gamma_n)
        self.plots.get_plot_array(self.optimized)
        gamma_tilt_deg_diff, self.tilt_angles_deg = \
            inter_cabinet_angles(self.optimized.gamma_tilt_deg)
        self.optimized.gamma_tilt_deg_diff = gamma_tilt_deg_diff
        try:
            x_S, y_S = calcSFP(self.optimized.gamma_tilt_deg, drawn, S,
                               self.evaluation)
        except _CALCULATION_ERRORS as exc:
            raise CalculationError('Error: SFP calculation failed') from exc
        calcSPLoverX(self.plots, drawn, self.evaluation.p_SPL, self.spl,
                     S.freq_range)
        calcHomogeneity(self.evaluation, x_S, y_S, drawn.points_x,
                        drawn.points_y)
        calcHistogram(self.evaluation)
        try:
            x_S, y_S = calcSFP(self.reference.gamma_tilt_deg, drawn, S,
                               self.evaluation_ref)
        except _CALCULATION_ERRORS as exc:
            raise CalculationError('Error: SFP calculation of reference '
                                   'array failed') from exc
        self.patches_ref = cabinet_outlines(self.plots_ref, P,
                                             self.reference.gamma_n)
        calcSPLoverX(self.plots_ref, drawn, self.evaluation_ref.p_SPL,
                     self.spl_ref, S.freq_range)
        calcHomogeneity(self.evaluation_ref, x_S, y_S, drawn.points_x,
                        drawn.points_y)
        calcHistogram(self.evaluation_ref)
        self.grid = (self.spl.x, drawn.points_x, drawn.points_y)
        if not self.target.init:
            shift_ref_on_zero(self.target, self.spl, self.spl_ref)
        elif target_run:
            shift2ref(self.target, self.spl, opt_ind, self.spl_ref)
            get_opt_region(self.spl, self.target)
            self.shown_target['SPL'] = self.target.SPL
            self.snap_target()

    def initial_target(self, hinges: int) -> None:
        """Target the GUI shows after the initial run and hinge selection.

        After the initial run the GUI shows a target from the first to the
        last grid point with the reference point in the middle of it;
        choosing the number of hinges places them on the grid
        (``chg_weight_links``).
        """
        x, x_v, y_v = self.grid
        SPL = self.spl.SPL
        self.show_target([x[0], x[-1]], [SPL[0], SPL[-1]],
                         [x_v[0], x_v[-1]], [y_v[0], y_v[-1]])
        x_interp = self.target.x_interp
        self.show_reference(x_interp[int(len(x_interp) / 2)])
        if hinges != 0:
            ind = get_weight_links(int(hinges), self.spl)
            self.show_target(list(self.spl.x[ind]),
                             list(np.array(self.spl.SPL)[ind]),
                             list(x_v[ind]), list(y_v[ind]))

    def result(self) -> RunResult:
        """Collect the results of the last :meth:`calculate`."""
        diff_ref, _ = inter_cabinet_angles(self.reference.gamma_tilt_deg)
        optimized = ArrayResult(
            tilt_angles_deg=self.tilt_angles_deg,
            inter_cabinet_angles_deg=self.optimized.gamma_tilt_deg_diff,
            evaluation=self.evaluation, spl_over_distance=self.spl,
            patches_x=self.patches[0], patches_y=self.patches[1])
        reference = ArrayResult(
            tilt_angles_deg=self.reference.gamma_tilt_deg,
            inter_cabinet_angles_deg=diff_ref,
            evaluation=self.evaluation_ref, spl_over_distance=self.spl_ref,
            patches_x=self.patches_ref[0], patches_y=self.patches_ref[1])
        target_slope = self.config.use_weighting == 'Target Slope' \
            and self.target.init
        return RunResult(
            setup=self.setup, venue=self.venue, optimized=optimized,
            reference=reference, geometry=self.optimized,
            ray_plot=self.plots,
            target=self.target if target_slope else None,
            weighting_weights=np.array(self.config.weighting_weights,
                                       dtype=float))


def _initial_run(setup: Setup, venue: VenueResult) -> _Calculation:
    """The run the GUI starts when "Target Slope" is selected."""
    calculation = _Calculation(setup, venue,
                               use_weighting=_WEIGHTING["without"])
    calculation.calculate()
    return calculation


def initial_target_slope(setup: Setup, venue: VenueResult, hinges: int
                         ) -> TargetSlopeSettings:
    """Target slope the GUI proposes after choosing Target Slope.

    Parameters
    ----------
    setup : Setup
        Settings of the initial run; the weighting is ignored.
    venue : VenueResult
    hinges : int
        Number of hinges between the first and the last target point.

    Returns
    -------
    TargetSlopeSettings
        Points on the SPL over distance grid of the initial run with the
        computed levels, and the reference point.

    Raises
    ------
    ValueError
        If ``hinges`` is negative.
    ConvergenceError, CalculationError
        If the initial run fails.
    """
    if hinges < 0:
        raise ValueError("hinges must not be negative")
    calculation = _initial_run(setup, venue)
    calculation.initial_target(hinges)
    target = calculation.target
    return TargetSlopeSettings(
        points=tuple(TargetPoint(float(x), float(spl))
                     for x, spl in zip(target.x, target.SPL)),
        reference_distance=float(target.x_ref))


def run(setup: Setup, venue: Optional[VenueResult] = None) -> RunResult:
    """Compute a setup as the GUI does.

    Parameters
    ----------
    setup : Setup
        The setup; it is not modified.
    venue : VenueResult, optional
        Venue built by :func:`build_venue` from ``setup.venue``; built if
        None. It is not modified.

    Returns
    -------
    RunResult

    Raises
    ------
    VenueError
        If the venue cannot be built.
    SetupError
        If the setup cannot be computed (e.g. unreadable directivity data).
    ConvergenceError
        If PALC does not converge.
    CalculationError
        If the calculation fails otherwise.
    """
    if venue is None:
        venue = build_venue(setup.venue, setup.array)
    if setup.algorithm.weighting != "target_slope":
        calculation = _Calculation(setup, venue)
        calculation.calculate()
        return calculation.result()
    if setup.target_slope is None:
        raise SetupError("target_slope",
                         "required when weighting is target_slope")
    calculation = _initial_run(setup, venue)
    calculation.config.store_config('use_weighting',
                                    _WEIGHTING["target_slope"])
    calculation.target.init = True
    ts = setup.target_slope
    n = len(ts.points)
    calculation.show_target([float(p.distance) for p in ts.points],
                            [float(p.level_db) for p in ts.points],
                            [0.0] * n, [0.0] * n)
    calculation.show_reference(float(ts.reference_distance))
    calculation.calculate()
    return calculation.result()


def result_arrays(result: RunResult) -> Dict[str, np.ndarray]:
    """Results as arrays, keyed like the recorded baselines.

    Parameters
    ----------
    result : RunResult

    Returns
    -------
    dict of str to numpy.ndarray
        Tilt angles, sound field, SPL over distance, homogeneity, histogram
        and venue points; for target slope runs also the target points and
        the weighting factors.
    """
    def _arr(value) -> np.ndarray:
        return np.array(value, dtype=float)

    opt, ref = result.optimized, result.reference
    ev, ev_ref = opt.evaluation, ref.evaluation
    out = {
        "tilt_angles_deg": _arr(opt.tilt_angles_deg),
        "inter_cabinet_angles_deg": _arr(opt.inter_cabinet_angles_deg),
        "reference_tilt_angles_deg": _arr(ref.tilt_angles_deg),
        "frequencies_hz": _arr(ev.f),
        "spl_db": _arr(ev.p_SPL),
        "reference_spl_db": _arr(ev_ref.p_SPL),
        "spl_over_distance_x": _arr(opt.spl_over_distance.x),
        "spl_over_distance_db": _arr(opt.spl_over_distance.SPL),
        "reference_spl_over_distance_db": _arr(ref.spl_over_distance.SPL),
        "homogeneity_db": _arr(ev.H),
        "homogeneity_low_db": _arr(ev.H_dist[0]),
        "homogeneity_high_db": _arr(ev.H_dist[1]),
        "reference_homogeneity_db": _arr(ev_ref.H),
        "histogram_steps": _arr(ev.Hist_steps),
        "histogram_tops": _arr(ev.Hist_tops),
        "reference_histogram_steps": _arr(ev_ref.Hist_steps),
        "reference_histogram_tops": _arr(ev_ref.Hist_tops),
        "venue_x": _arr(result.venue.drawn.points_x),
        "venue_y": _arr(result.venue.drawn.points_y),
    }
    if result.target is not None:
        out["target_x"] = _arr(result.target.x)
        out["target_db"] = _arr(result.target.SPL)
        out["weighting_weights"] = _arr(result.weighting_weights)
    return out
