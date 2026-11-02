"""Replay a setup through the legacy pyPALC GUI code and record the results.

:func:`run_legacy` reproduces what a user who enters a :class:`Setup` in the
Bokeh GUI (``main.py``) would get, using the legacy modules frozen in
``tests/legacy/fun`` (a copy of ``fun/`` before the computation moved to
:mod:`pypalc.core`; only bug fixes are applied to it). The callbacks of ``main.py`` are re-implemented here one to one,
minus the purely visual effects (plot ranges, layouts, texts, glyphs).
Widgets are replaced by :class:`_Widget` stand-ins that call their
``on_change`` callbacks only when the value really changes, like Bokeh does.
The ColumnDataSources the computation reads back (SPL over distance, target
slope) are real Bokeh ``ColumnDataSource`` objects wired to ``update_draw``
exactly as in ``main.py``, so the Bokeh change semantics (including the
recursive ``update_draw`` calls) are the original ones.

Assumed click order
-------------------
The user starts a fresh session (all widgets at their defaults) and then

1. Page 1 (venue): for every audience line enters start and stop point and
   the discretization, clicks "Save Current Audience Line"
   (:meth:`_LegacySession.save_line`); then clicks "Get PALC Venue Proposal"
   (:meth:`_LegacySession.check_pal`).
2. Page 2 (loudspeaker): aperture angle, cabinet height, directivity; for a
   measured directivity uploads the CSV file afterwards.
3. Page 3 (array): number of cabinets, x and y of the uppermost cabinet,
   gap, "use discrete tilt angles" and the angle list; then the reference
   array: type, top angle, angle parameter, "use angles of (17.)", user
   defined angles.
4. Page 4 (algorithm): variant, gap handling, soft margin strength,
   tolerance, frequency range, "fix angle of uppermost cabinet" and the
   angle, and last the weighting with its parameter. For target slope
   weighting, selecting it runs the initial calculation; the user then
   selects the number of hinges, drags the target points and the reference
   point to the setup's values, and finally clicks "PALC Run".

Only widgets whose value differs from the GUI default are touched; numbers
count as unchanged when they are numerically equal to the default text.
Choosing the weighting last means that the initial target-slope run already
uses all other settings, as a careful user would do it.

Deviations from the GUI
-----------------------
Where the GUI only shows an error text (and then possibly keeps stale
results), :func:`run_legacy` raises :class:`LegacyGuiError`, so that no
baseline is ever recorded from a failed calculation.

Usage::

    python -m tests.legacy.capture [CASE ...]

writes ``tests/baseline/<case>.npz`` for the given (default: all) cases in
``tests/cases``.
"""
from __future__ import annotations

import base64
import contextlib
import io
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CASES_DIR = REPO / "tests" / "cases"
BASELINE_DIR = REPO / "tests" / "baseline"
#: Frozen copy of the legacy ``fun/`` modules this replay runs on.
LEGACY_FUN = Path(__file__).resolve().parent / "fun"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from bokeh.models import ColumnDataSource  # noqa: E402

from pypalc.core.settings import Setup, load_setup  # noqa: E402

_LEGACY_MODULES = ("PALC_classes", "PALC_functions", "PALC_opt", "calcPALC",
                   "checkPAL", "gui_algorithm_sfp", "gui_help_text",
                   "gui_helper", "gui_ls_array", "gui_venue",
                   "sfp_functions")


def _import_legacy() -> Dict[str, Any]:
    """Import the legacy modules from :data:`LEGACY_FUN` in isolation.

    The legacy modules import each other by bare module names, which the
    application's ``fun/`` also uses. They are imported with
    :data:`LEGACY_FUN` first on ``sys.path`` and afterwards removed from
    ``sys.modules`` and ``sys.path`` again, so that code imported later in
    the same process (e.g. ``main.py``) gets its own modules.
    """
    import importlib
    saved = {name: sys.modules.pop(name) for name in _LEGACY_MODULES
             if name in sys.modules}
    sys.path.insert(0, str(LEGACY_FUN))
    try:
        return {name: importlib.import_module(name)
                for name in _LEGACY_MODULES}
    finally:
        sys.path.remove(str(LEGACY_FUN))
        for name in _LEGACY_MODULES:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


_LEGACY = _import_legacy()
calcPALC = _LEGACY["calcPALC"]
checkPAL = _LEGACY["checkPAL"]
PALC_classes = _LEGACY["PALC_classes"]
choose_gap_handling = _LEGACY["gui_algorithm_sfp"].choose_gap_handling
get_fixed_angle_borders = _LEGACY["gui_algorithm_sfp"].get_fixed_angle_borders
optimize_PALC = _LEGACY["gui_algorithm_sfp"].optimize_PALC
round_fixed_angle = _LEGACY["gui_algorithm_sfp"].round_fixed_angle
set_weighting_in = _LEGACY["gui_algorithm_sfp"].set_weighting_in
update_gh_w_select = _LEGACY["gui_algorithm_sfp"].update_gh_w_select
arrange_ref_arr_in = _LEGACY["gui_ls_array"].arrange_ref_arr_in
read_dir_data = _LEGACY["gui_ls_array"].read_dir_data
ref_array_angles = _LEGACY["gui_ls_array"].ref_array_angles
check_all_draw_conditions = _LEGACY["gui_venue"].check_all_draw_conditions
LSA_visualization = _LEGACY["PALC_functions"].LSA_visualization
calc_diff_tilt_angles = _LEGACY["PALC_functions"].calc_diff_tilt_angles
get_opt_region = _LEGACY["PALC_opt"].get_opt_region
get_weight_links = _LEGACY["PALC_opt"].get_weight_links
shift2ref = _LEGACY["PALC_opt"].shift2ref
shift_ref_on_zero = _LEGACY["PALC_opt"].shift_ref_on_zero
calc_directivity = _LEGACY["sfp_functions"].calc_directivity
calcHistogram = _LEGACY["sfp_functions"].calcHistogram
calcHomogeneity = _LEGACY["sfp_functions"].calcHomogeneity
calcSFP = _LEGACY["sfp_functions"].calcSFP
calcSPLoverX = _LEGACY["sfp_functions"].calcSPLoverX
init_dir_plot = _LEGACY["sfp_functions"].init_dir_plot

#: Cases recorded in ``tests/baseline``.
CASES = (
    "simple", "gap_without", "gap_hard_margin", "gap_soft_margin", "palc1",
    "palc3", "discrete_angles", "fixed_angle", "measured", "linear_weighting",
    "log_weighting", "target_slope", "gap_soft_margin_target_slope",
)

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
_REFERENCE = {"straight": "Straight", "progressive": "Progressive",
              "arc": "Arc", "user_defined": "User Defined"}


class LegacyGuiError(RuntimeError):
    """The GUI would have shown an error message instead of results."""


# --------------------------------------------------------------------------
# Widget stand-ins
# --------------------------------------------------------------------------

class _Widget:
    """Stand-in for a Bokeh widget.

    Attributes can be set freely. Setting ``value`` (or ``active``) to a
    value different from the current one calls the registered callbacks with
    ``(attr, old, new)``, like Bokeh's ``on_change``.
    """

    def __init__(self, **attrs: Any) -> None:
        object.__setattr__(self, "_callbacks", [])
        attrs.setdefault("disabled", False)
        attrs.setdefault("visible", True)
        self.__dict__.update(attrs)

    def on_change(self, callback: Callable[[str, Any, Any], None]) -> None:
        self._callbacks.append(callback)

    def __setattr__(self, name: str, new: Any) -> None:
        old = self.__dict__.get(name)
        self.__dict__[name] = new
        if name in ("value", "active") and not _matches(old, new):
            for callback in list(self._callbacks):
                callback(name, old, new)


def _matches(old: Any, new: Any) -> bool:
    try:
        return bool(old == new)
    except Exception:  # pragma: no cover - arrays etc.
        return False


def _fmt(value: float) -> str:
    """Text a user types for a number: ``2.0`` -> ``'2'``, ``0.5`` -> ``'0.5'``."""
    text = repr(float(value))
    return text[:-2] if text.endswith(".0") else text


def _new_pal(**kwargs: Any) -> Any:
    """``PALC_classes.PAL`` with fresh lists for all mutable defaults."""
    for name in ("xline_start", "xline_stop", "yline_start", "yline_stop",
                 "slope", "b", "xline", "yline", "pal_index", "plot_x",
                 "plot_y", "ind_iscut"):
        kwargs.setdefault(name, [])
    return PALC_classes.PAL(**kwargs)


# --------------------------------------------------------------------------
# The session: objects of main.py plus its callbacks
# --------------------------------------------------------------------------

class _LegacySession:
    """One GUI session: the objects of ``main.py`` and its callbacks."""

    def __init__(self) -> None:
        C = PALC_classes
        # objects (main.py, "Initialize all necessary objects")
        self.created_pal = _new_pal(pal_index=np.array([0]).astype(int))
        self.PALC_pal = C.PALC_compatible_PAL(pal=[], pal_no_nal=[])
        self.PALC_config = C.PALC_configuration(
            gamma_LSA=[], weighting_factors=[], last_angle_hm=[],
            gap_weights=[], weighting_weights=[], psi_n=[0.0524],
            gamma_n=[0])
        self.SFP_config = C.SFP_configuration(
            dir_meas=np.zeros([2, 120]), dir_meas_deg=np.zeros([2, 120]),
            dir_meas_amp=[], gamma_n=[0], f=[], plot_dir_amp=[],
            plot_dir_amp_meas=[[0], [0], [0]], plot_beta_meas=[0])
        # SFP_configuration calls PALC_configuration.__init__() without
        # arguments; replace the shared default lists it got from there.
        for name, val in (("gamma_LSA", []), ("weighting_factors", []),
                          ("last_angle_hm", []), ("gap_weights", []),
                          ("weighting_weights", []), ("psi_n", [0.0524]),
                          ("gamma_n", [0])):
            setattr(self.SFP_config, name, val)
        self.PALC_plots = C.PALC_plotting()
        self.PALC_plots_ref = C.PALC_plotting()
        self.Opt_arr = C.Array_opt(gamma_tilt_deg=[])
        self.Ref_arr = C.Array_opt(gamma_tilt_deg=np.zeros(4))
        self.Tech_res = C.Technical_meas_results()
        self.Tech_res_ref = C.Technical_meas_results()
        self.Opt_w = C.Opt_weight(SPL_interp=[0, 0])
        self.SPLoverX = C.Opt_weight(SPL_interp=[0])
        self.SPLoverX_ref = C.Opt_weight(SPL_interp=[0])

        # ColumnDataSources the computation reads back
        p_SPL = list(zip([0, 0]))
        x_vert = list(zip([0, 0]))
        self.SPLoverX_dict = ColumnDataSource(data=dict(
            SPL=p_SPL, x=x_vert, x_v=x_vert, y_v=x_vert))
        self.SPLoverX_ref_dict = ColumnDataSource(data=dict(
            SPL=p_SPL, x=x_vert, x_v=x_vert, y_v=x_vert))
        self.Opt_SPLoverX_dict = ColumnDataSource(data=dict(
            SPL=p_SPL, x=x_vert, x_v=[0, 0], y_v=[0, 0]))
        self.Opt_refpoint_dict = ColumnDataSource(data=dict(
            x_ref=[10], SPL=[0], x_v=[0], y_v=[0]))
        self.SPLoverX_optreg = ColumnDataSource(data=dict(
            x=[0, 0, 0, 0], y=[0, 0, 0, 0]))

        # initial directivity (main.py, after the ColumnDataSources)
        c0, alpha, beta_deg, beta, f, omega, dir_meas, dir_meas_deg, H_post \
            = init_dir_plot()
        for n in range(np.shape(f)[0]):
            H_post[:, n] = calc_directivity(
                self.PALC_config.directivity, alpha,
                self.PALC_config.Lambda_y, beta, omega[n], c0, f,
                dir_meas[:, n], dir_meas_deg[:, 1], n)
        setattr(self.SFP_config, "f", f)
        self.SFP_config.get_plot_dir_amp(H_post, [100, 1000, 10000])

        # widgets with their default values (main.py, "WIDGETS")
        W = _Widget
        self.pal_error_info = W(text="")
        self.config_error_info = W(text="")
        self.gui_psi_n = W(value="3.0")
        self.gui_Lambda_y = W(value="0.2")
        self.gui_directivity = W(value="Circular Piston")
        self.gui_N_LS = W(value="4")
        self.gui_x_H = W(value="0")
        self.gui_y_H = W(value="2")
        self.gui_Lambda_gap = W(value="0")
        self.gui_use_dangles = W(value="No")
        self.gui_discrete_angles = W(value="", disabled=True)
        self.gui_ref_array = W(value="Straight")
        self.gui_ref_discrete_angles = W(value="No", disabled=True,
                                         visible=False)
        self.gui_ref_start = W(value="0")
        self.gui_ref_step_stop = W(value="0", disabled=True)
        self.gui_ref_userdef = W(value="", visible=False)
        self.gui_PALC_alg = W(value="PALC 2")
        self.gui_gap_handling = W(
            value="Without", options=["Without", "Hard Margin", "Soft Margin"])
        self.gui_strength_smarg = W(value=1.0, disabled=True)
        self.gui_weighting = W(
            value="Without", options=["Without", "Linear Spacing",
                                      "Logarithmic Spacing", "Target Slope"])
        self.gui_opt_weight_link = W(value="0", options=["0", "1", "2"],
                                     visible=False)
        self.gui_weighting_nu = W(value=float(np.cbrt(1.00)), disabled=True)
        self.weighting_plus = W(disabled=True)
        self.weighting_minus = W(disabled=True)
        self.weighting_step_size = W(value="0.05", disabled=True)
        self.gui_tolerance = W(value=1.0)
        self.gui_use_fixed_angles = W(value="No", disabled=True)
        self.gui_fixed_first_angle = W(value="0.0", disabled=True)
        self.gui_freq_range = W(value=(200, 8000))

        # callbacks (main.py, "Callbacks of the Widgets")
        for widget in (self.gui_psi_n, self.gui_Lambda_y,
                       self.gui_directivity, self.gui_x_H, self.gui_y_H,
                       self.gui_N_LS, self.gui_Lambda_gap, self.gui_PALC_alg,
                       self.gui_gap_handling, self.gui_strength_smarg,
                       self.gui_tolerance, self.gui_fixed_first_angle,
                       self.gui_freq_range):
            widget.on_change(self.get_value)
        self.gui_use_dangles.on_change(self.choose_dangles)
        self.gui_discrete_angles.on_change(self.get_discrete_angles)
        self.gui_discrete_angles.on_change(
            lambda attr, old, new: self.get_ref_array_angles())
        self.gui_ref_array.on_change(self.reference_array_setting)
        for widget in (self.gui_ref_start, self.gui_ref_step_stop,
                       self.gui_ref_discrete_angles, self.gui_ref_userdef):
            widget.on_change(lambda attr, old, new: self.get_ref_array_angles())
        self.gui_weighting.on_change(self.weighting_select)
        self.gui_opt_weight_link.on_change(self.chg_weight_links)
        self.gui_weighting_nu.on_change(self.store_weighting_nu)
        self.gui_use_fixed_angles.on_change(self.use_fixed_angle)
        self.Opt_SPLoverX_dict.on_change("data", self.update_draw)
        self.Opt_refpoint_dict.on_change("data", self.update_draw)

        # main.py, "Set up values before showing on Data Table"
        self.PALC_config.store_config("N", self.gui_N_LS.value)

    # ---------------------------------------------------------------- page 1
    def save_line(self, start, stop, points_per_meter: float) -> None:
        """``update_point`` + ``save_line`` of main.py."""
        x_start, x_stop = float(start[0]), float(stop[0])
        y_start, y_stop = float(start[1]), float(stop[1])
        diff = np.sqrt((x_stop - x_start)**2 + (y_stop - y_start)**2)
        condition1, condition2, condition3, slope, b = \
            check_all_draw_conditions(x_start, x_stop, y_start, y_stop,
                                      self.created_pal, self.pal_error_info)
        if not (condition1 and condition2 and condition3):
            raise LegacyGuiError("audience line %s -> %s rejected: %s"
                                 % (start, stop, self.pal_error_info.text))
        l_d = float(points_per_meter)
        self.created_pal.append_line(x_start, x_stop, y_start, y_stop, slope,
                                     b, diff * l_d)
        if np.shape(self.created_pal.xline)[0] > 2000:
            raise LegacyGuiError("Maximum discretized points are 2000")
        self.change_start_pal_button(False)

    def check_pal(self) -> None:
        """``check_PAL`` of main.py (without its catch-all ``except``)."""
        suggested_nal = _new_pal(ind_iscut=[])
        suggested_sfp = _new_pal(ind_iscut=[])
        checkPAL.suggestPAL(self.created_pal, suggested_nal, suggested_sfp)
        self.PALC_pal.pal, self.PALC_pal.pal_no_nal = [], []
        checkPAL.get_PALC_compatible_PAL(self.PALC_pal, self.created_pal,
                                         suggested_nal, suggested_sfp)
        self.PALC_pal.create_pal_without_nal(suggested_nal)
        self.change_start_pal_button(True)
        if self.gui_weighting.value in ["Target Slope"]:
            self.gui_weighting.value = "Without"
            self.Opt_w.init = False
            self.weighting_select(0, 0, 0)
        weight_links = np.arange(np.shape(self.created_pal.xline_start)[0] + 1)
        self.gui_opt_weight_link.options = [str(n) for n in list(weight_links)]
        self.gui_opt_weight_link.value = "0"

    def change_start_pal_button(self, has_pal: bool) -> None:
        if has_pal and self.PALC_pal.pal != []:
            self.gui_use_fixed_angles.disabled = False
        else:
            self.gui_use_fixed_angles.disabled = True
            self.gui_use_fixed_angles.value = "No"
            self.gui_fixed_first_angle.disabled = True
            self.PALC_config.use_fixed_angle = False

    # ---------------------------------------------------------------- page 2
    def upload_directivity(self, b64: str) -> None:
        read_dir_data(self.SFP_config, b64)

    # ---------------------------------------------------------------- page 3
    def choose_dangles(self, attr, old, new) -> None:
        if self.gui_use_dangles.value == "Yes":
            self.gui_discrete_angles.value = "0,0.5,1,2,3.5,5,7,10"
            self.gui_discrete_angles.disabled = False
            self.PALC_config.store_config("use_gamma_LSA", True)
            self.gui_ref_discrete_angles.disabled = False
        elif self.gui_use_dangles.value == "No":
            self.gui_discrete_angles.value = ""
            self.gui_discrete_angles.disabled = True
            self.PALC_config.store_config("use_gamma_LSA", False)
            self.gui_ref_discrete_angles.disabled = True
            self.gui_ref_discrete_angles.value = "No"
        self.get_discrete_angles(attr, old, new)

    def get_discrete_angles(self, attr, old, new) -> None:
        if self.PALC_config.use_gamma_LSA == False:  # noqa: E712 (legacy)
            self.PALC_config.gamma_LSA = []
            return
        try:
            self.PALC_config.gamma_LSA = np.array(
                [float(s) for s in self.gui_discrete_angles.value.split(",")]
            ) * (np.pi / 180)
        except Exception:
            self.config_error_info.text = "Error: Could not process the input"

    def reference_array_setting(self, attr, old, new) -> None:
        arrange_ref_arr_in(self.gui_ref_array, self.gui_ref_start,
                           self.gui_ref_step_stop,
                           self.gui_ref_discrete_angles, self.gui_ref_userdef)
        self.get_ref_array_angles()

    def get_ref_array_angles(self, chg_n: bool = False) -> None:
        try:
            ref_array_angles(self.PALC_config, self.Ref_arr,
                             self.gui_ref_array, self.gui_ref_start,
                             self.gui_ref_step_stop,
                             self.gui_ref_discrete_angles,
                             self.gui_ref_userdef)
            self.Ref_arr.gamma_n = self.Ref_arr.gamma_tilt_deg * (np.pi / 180)
        except Exception:
            self.config_error_info.text = "Error: Could not process the input"
            self.gui_ref_start.value = "0"
            self.gui_ref_step_stop.value = "0"

    # ---------------------------------------------------------------- page 4
    def weighting_select(self, attr, old, new) -> None:
        set_weighting_in(self.gui_weighting, self.weighting_plus,
                         self.weighting_minus, self.weighting_step_size,
                         self.gui_weighting_nu)
        if self.gui_weighting.value in ["Target Slope"]:
            self.start_calc(True)
            self.Opt_w.init = True
            self.gui_opt_weight_link.visible = True
        else:
            self.gui_opt_weight_link.visible = False
        self.PALC_config.store_config("use_weighting", self.gui_weighting.value)
        update_gh_w_select(self.gui_weighting, self.gui_gap_handling)

    def store_weighting_nu(self, attr, old, new) -> None:
        self.PALC_config.weighting_nu = np.round(
            self.gui_weighting_nu.value**3, decimals=2)

    def use_fixed_angle(self, attr, old, new) -> None:
        if self.gui_use_fixed_angles.value == "Yes" and self.PALC_pal.pal != []:
            gamma_tilt_deg = get_fixed_angle_borders(self.PALC_pal.pal[-1],
                                                     self.PALC_config)
            self.Opt_arr.fixed_angle = [gamma_tilt_deg[0] - 1,
                                        gamma_tilt_deg[1],
                                        gamma_tilt_deg[1] - 10]
            self.gui_fixed_first_angle.disabled = False
            if self.Opt_arr.fixed_angle[1] < self.Opt_arr.fixed_angle[0]:
                self.gui_fixed_first_angle.value = str(self.Opt_arr.fixed_angle[1])
            else:
                self.gui_fixed_first_angle.value = str(self.Opt_arr.fixed_angle[0] - 1)
            self.PALC_config.use_fixed_angle = True
        elif self.gui_use_fixed_angles.value == "No":
            self.gui_fixed_first_angle.disabled = True
            self.Opt_arr.fixed_angle = [0, 0, 0]
            self.PALC_config.use_fixed_angle = False

    def start_calc(self, arg: bool) -> None:
        """``start_calc`` of main.py without plotting."""
        if not arg:
            return
        P, S = self.PALC_config, self.SFP_config
        if self.PALC_pal.pal == []:
            raise LegacyGuiError("Please CREATE and GET PALC compatible venue")
        P.get_initial_weights()
        opt_ind = None
        try:
            if P.use_weighting == "Target Slope" and self.Opt_w.init:
                self.Opt_w.calc_init()
                self.Opt_SPLoverX_dict.data.update(dict(SPL=list(self.Opt_w.SPL)))
                opt_ind = optimize_PALC(P, S, self.PALC_pal, self.PALC_plots,
                                        self.Opt_arr, self.created_pal,
                                        self.Tech_res, self.SPLoverX,
                                        self.Opt_w, 200)
            else:
                calcPALC.calcPALC(self.PALC_plots, self.PALC_pal, P,
                                  self.Opt_arr)
        except Exception as exc:
            raise LegacyGuiError("PALC calculation failed") from exc
        for key, val in [["last_angle_hm", []], ["x_H", self.gui_x_H.value],
                         ["y_H", self.gui_y_H.value],
                         ["psi_n", self.gui_psi_n.value]]:
            P.store_config(key, val)
        if self.Opt_arr.num_iter == 500 or \
                np.shape(self.Opt_arr.gamma_tilt_deg)[0] == 0:
            raise LegacyGuiError("No convergence reached")
        LSA_visualization(self.PALC_plots, P, self.Opt_arr.gamma_n)
        self.PALC_plots.get_plot_array(self.Opt_arr)
        gamma_tilt_deg_diff, gamma_tilt_deg_round = \
            calc_diff_tilt_angles(self.Opt_arr.gamma_tilt_deg)
        setattr(self.Opt_arr, "gamma_tilt_deg_diff", gamma_tilt_deg_diff)
        self.results = dict(b_gamma_tilt_deg=gamma_tilt_deg_round,
                            c_gamma_tilt_deg_diff=gamma_tilt_deg_diff)
        try:
            x_S, y_S = calcSFP(self.Opt_arr.gamma_tilt_deg, self.created_pal,
                               S, self.Tech_res)
        except Exception as exc:
            raise LegacyGuiError("SFP calculation failed") from exc
        calcSPLoverX(self.PALC_plots, self.created_pal, self.Tech_res.p_SPL,
                     self.SPLoverX, S.freq_range)
        calcHomogeneity(self.Tech_res, x_S, y_S, self.created_pal.xline,
                        self.created_pal.yline)
        calcHistogram(self.Tech_res)
        try:
            x_S, y_S = calcSFP(self.Ref_arr.gamma_tilt_deg, self.created_pal,
                               S, self.Tech_res_ref)
        except Exception as exc:
            raise LegacyGuiError(
                "SFP calculation of reference array failed") from exc
        LSA_visualization(self.PALC_plots_ref, P, self.Ref_arr.gamma_n)
        calcSPLoverX(self.PALC_plots_ref, self.created_pal,
                     self.Tech_res_ref.p_SPL, self.SPLoverX_ref, S.freq_range)
        calcHomogeneity(self.Tech_res_ref, x_S, y_S, self.created_pal.xline,
                        self.created_pal.yline)
        calcHistogram(self.Tech_res_ref)

        Opt_w, SPLoverX, SPLoverX_ref = self.Opt_w, self.SPLoverX, \
            self.SPLoverX_ref
        xline, yline = self.created_pal.xline, self.created_pal.yline
        if not Opt_w.init:
            shift_ref_on_zero(Opt_w, SPLoverX, SPLoverX_ref)
            self.SPLoverX_dict.data.update(dict(
                SPL=SPLoverX.SPL, x=SPLoverX.x, x_v=xline, y_v=yline))
            self.SPLoverX_ref_dict.data.update(dict(
                SPL=SPLoverX_ref.SPL, x=SPLoverX_ref.x, x_v=xline, y_v=yline))
            self.Opt_SPLoverX_dict.data.update(dict(
                SPL=[SPLoverX.SPL[0], SPLoverX.SPL[-1]],
                x=[SPLoverX.x[0], SPLoverX.x[-1]],
                x_v=[xline[0], xline[-1]], y_v=[yline[0], yline[-1]]))
            self.Opt_refpoint_dict.data.update(dict(
                x_ref=[Opt_w.x_interp[int(len(Opt_w.x_interp) / 2)]],
                SPL=[Opt_w.SPL_interp[int(len(Opt_w.SPL_interp) / 2)]],
                x_v=[xline[int(len(xline) / 2)]],
                y_v=[yline[int(len(yline) / 2)]]))
        elif P.use_weighting == "Target Slope" and Opt_w.init:
            shift2ref(Opt_w, SPLoverX, opt_ind, SPLoverX_ref)
            get_opt_region(SPLoverX, Opt_w)
            self.SPLoverX_dict.data.update(dict(
                SPL=SPLoverX.SPL, x=SPLoverX.x, x_v=xline, y_v=yline))
            self.Opt_SPLoverX_dict.data.update(dict(SPL=Opt_w.SPL))
            self.Opt_refpoint_dict.data.update(dict(
                SPL=[Opt_w.SPL_interp[Opt_w.ref_ind]]))
            self.SPLoverX_ref_dict.data.update(dict(SPL=SPLoverX_ref.SPL))
        else:
            self.SPLoverX_dict.data.update(dict(
                SPL=SPLoverX.SPL, x=SPLoverX.x, x_v=xline, y_v=yline))
            self.SPLoverX_ref_dict.data.update(dict(
                SPL=SPLoverX_ref.SPL, x=SPLoverX_ref.x, x_v=xline, y_v=yline))

    # ---------------------------------------------------- shared callbacks
    def get_value(self, attr, old, new) -> None:
        """``get_value`` of main.py without plotting."""
        P, S = self.PALC_config, self.SFP_config
        widget_dict = {"constraint": self.gui_PALC_alg.value,
                       "use_gamma_LSA": self.gui_use_dangles.value,
                       "gap_handling": self.gui_gap_handling.value,
                       "strength_sm": self.gui_strength_smarg.value,
                       "Lambda_y": self.gui_Lambda_y.value,
                       "Lambda_gap": self.gui_Lambda_gap.value,
                       "directivity": self.gui_directivity.value,
                       "fixed_angle": self.gui_fixed_first_angle.value,
                       "x_H": self.gui_x_H.value, "y_H": self.gui_y_H.value,
                       "N": self.gui_N_LS.value,
                       "psi_n": self.gui_psi_n.value,
                       "tolerance": self.gui_tolerance.value,
                       "freq_range": self.gui_freq_range.value}
        for key, val in widget_dict.items():
            if new == val:
                error_info, failed_key = P.store_config(key, val)
                if failed_key in ["Lambda_y"]:
                    setattr(self.gui_Lambda_y, attr, "0.2")
                if failed_key in ["N"]:
                    setattr(self.gui_N_LS, attr, "4")
                if failed_key in ["Lambda_gap"]:
                    setattr(self.gui_Lambda_gap, attr, "0")
                if failed_key in ["tolerance"]:
                    setattr(self.gui_tolerance, attr, "0.1")
                if failed_key in ["fixed_angle"]:
                    setattr(self.gui_fixed_first_angle, attr, "")
                if failed_key in ["x_H"]:
                    setattr(self.gui_x_H, attr, "0")
                if failed_key in ["y_H"]:
                    setattr(self.gui_y_H, attr, "2")
                if failed_key in ["too low"]:
                    setattr(self.gui_y_H, attr, str(P.y_H))
                if failed_key in ["psi_n"]:
                    setattr(self.gui_psi_n, attr, "0.0524")
                if hasattr(S, key) and failed_key in [None, "too low"]:
                    S.store_config(key, val)
                self.config_error_info.text = str(error_info)
                if key in ["N", "use_gamma_LSA"]:
                    self.get_ref_array_angles(chg_n=True)
        round_fixed_angle(P, self.Opt_arr, self.gui_fixed_first_angle)
        choose_gap_handling(self.gui_gap_handling, self.gui_strength_smarg)
        update_gh_w_select(self.gui_weighting, self.gui_gap_handling)
        if P.directivity in ["Circular Piston", "Line Piston",
                             "Combined Circular/Line"]:
            c0, alpha, beta_deg, beta, f, omega, dir_meas, dir_meas_deg, \
                H_post = init_dir_plot()
            for n in range(np.shape(f)[0]):
                H_post[:, n] = calc_directivity(
                    P.directivity, alpha, P.Lambda_y, beta, omega[n], c0, f,
                    dir_meas[:, n], dir_meas_deg[:, 1], n)
            setattr(S, "f", f)
            S.get_plot_dir_amp(H_post, [100, 1000, 10000])

    def chg_weight_links(self, attr, old, new) -> None:
        ind = get_weight_links(int(new), self.SPLoverX)
        self.Opt_SPLoverX_dict.data.update(dict(
            SPL=np.array(self.SPLoverX.SPL)[ind], x=self.SPLoverX.x[ind],
            x_v=self.created_pal.xline[ind], y_v=self.created_pal.yline[ind]))

    def update_draw(self, attr, old, new) -> None:
        """``update_draw`` of main.py (verbatim apart from ``self``)."""
        Opt_w = self.Opt_w
        x = self.SPLoverX_dict.data["x"]
        x_v, y_v = self.SPLoverX_dict.data["x_v"], self.SPLoverX_dict.data["y_v"]
        Opt_w.CDS2obj(self.Opt_SPLoverX_dict, ["x", "SPL", "x_v", "y_v"])
        Opt_w.CDS2obj(self.Opt_refpoint_dict, ["x_ref"])
        for ind, val in enumerate(Opt_w.x):
            arg = np.argmin(np.abs(np.array(x) - val))
            Opt_w.x[ind] = x[arg]
            Opt_w.x_v[ind], Opt_w.y_v[ind] = x_v[arg], y_v[arg]
        Opt_w.resort_opt_region()
        Opt_w.x_interp = list(x[np.argmin(np.abs(x - Opt_w.x[0])):
                                np.argmin(np.abs(x - Opt_w.x[-1])) + 1])
        Opt_w.interpolate_SPL()
        ref_ind = np.argmin(np.abs(Opt_w.x_ref[0] - np.array(Opt_w.x_interp)))
        Opt_w.x_ref, Opt_w.ref_ind = Opt_w.x_interp[ref_ind], ref_ind
        xv_ind = np.argmin(np.abs(np.array(x) - Opt_w.x_ref))
        if Opt_w.is2update < 4:
            Opt_w.is2update += 1
            self.Opt_SPLoverX_dict.data.update(dict(
                x=Opt_w.x, SPL=Opt_w.SPL, x_v=Opt_w.x_v, y_v=Opt_w.y_v))
            self.Opt_refpoint_dict.data.update(dict(
                x_ref=[Opt_w.x_ref], SPL=[Opt_w.SPL_interp[ref_ind]],
                x_v=[x_v[xv_ind]], y_v=[y_v[xv_ind]]))
        self.SPLoverX_optreg.data.update(dict(
            x=[Opt_w.x[0], Opt_w.x[0], Opt_w.x[-1], Opt_w.x[-1]],
            y=[Opt_w.SPL[0] + 100, Opt_w.SPL[-1] - 100,
               Opt_w.SPL[-1] - 100, Opt_w.SPL[0] + 100]))

    # ------------------------------------------------- user interaction
    def drag_target(self, distances, levels_db, reference_distance) -> None:
        """Drag the target slope points and the reference point in the plot.

        Tapping a point selects it (``optimal_SPLoverX`` resets
        ``is2update``); dragging changes the ``data`` of the source on the
        client, which triggers ``update_draw``.
        """
        data = self.Opt_SPLoverX_dict.data
        self.Opt_w.is2update = 0
        self.Opt_SPLoverX_dict.data = dict(
            x=[float(d) for d in distances],
            SPL=[float(v) for v in levels_db],
            x_v=list(data["x_v"]), y_v=list(data["y_v"]))
        ref = self.Opt_refpoint_dict.data
        self.Opt_w.is2update = 0
        self.Opt_refpoint_dict.data = dict(
            x_ref=[float(reference_distance)], SPL=list(ref["SPL"]),
            x_v=list(ref["x_v"]), y_v=list(ref["y_v"]))


def _enter_text(widget: _Widget, value: Any, as_int: bool = False) -> None:
    """Type a number into a text input unless it already shows that number."""
    try:
        if float(widget.value) == float(value):
            return
    except ValueError:
        pass
    widget.value = str(int(value)) if as_int else _fmt(value)


def _measured_csv_base64(setup: Setup) -> str:
    md = setup.loudspeaker.measured_directivity
    if md is None:
        raise ValueError("measured directivity without data")
    if md.path is not None:
        raw = Path(md.path).read_bytes()
    else:
        raw = md.csv.encode("utf-8")
    return base64.b64encode(raw).decode("ascii")


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def _replay(setup: Setup) -> _LegacySession:
    s = _LegacySession()
    ls, ar, ref, alg = (setup.loudspeaker, setup.array,
                        setup.reference_array, setup.algorithm)

    # page 1: venue
    for line in setup.venue.audience_lines:
        s.save_line(line.start, line.stop, line.points_per_meter)
    s.check_pal()

    # page 2: loudspeaker
    _enter_text(s.gui_psi_n, ls.aperture_angle_deg)
    _enter_text(s.gui_Lambda_y, ls.cabinet_height)
    s.gui_directivity.value = _DIRECTIVITY[ls.directivity]
    if ls.directivity == "measured":
        s.upload_directivity(_measured_csv_base64(setup))

    # page 3: array
    _enter_text(s.gui_N_LS, ar.cabinet_count, as_int=True)
    _enter_text(s.gui_x_H, ar.top_position[0])
    _enter_text(s.gui_y_H, ar.top_position[1])
    _enter_text(s.gui_Lambda_gap, ar.cabinet_gap)
    if ar.discrete_tilt_angles_deg is not None:
        s.gui_use_dangles.value = "Yes"
        s.gui_discrete_angles.value = ",".join(
            _fmt(a) for a in ar.discrete_tilt_angles_deg)
    s.gui_ref_array.value = _REFERENCE[ref.type]
    if ref.type != "user_defined":
        _enter_text(s.gui_ref_start, ref.top_angle_deg)
        if ref.type != "straight":
            _enter_text(s.gui_ref_step_stop, ref.angle_param_deg)
        if ref.use_discrete_angles and not s.gui_ref_discrete_angles.disabled:
            s.gui_ref_discrete_angles.value = "Yes"
    else:
        s.gui_ref_userdef.value = ",".join(_fmt(a) for a in ref.tilt_angles_deg)

    # page 4: algorithm
    s.gui_PALC_alg.value = _VARIANTS[alg.variant]
    s.gui_gap_handling.value = _GAP_HANDLING[alg.gap_handling]
    if not s.gui_strength_smarg.disabled:
        s.gui_strength_smarg.value = float(alg.soft_margin_strength)
    s.gui_tolerance.value = float(alg.tolerance)
    s.gui_freq_range.value = tuple(float(f) for f in alg.frequency_range_hz)
    if alg.fixed_top_angle_deg is not None:
        if s.gui_use_fixed_angles.disabled:
            raise LegacyGuiError("fixed angle needs a PALC compatible venue")
        s.gui_use_fixed_angles.value = "Yes"
        s.gui_fixed_first_angle.value = _fmt(alg.fixed_top_angle_deg)
    s.gui_weighting.value = _WEIGHTING[alg.weighting]
    if alg.weighting in ("linear", "logarithmic"):
        s.gui_weighting_nu.value = float(np.cbrt(alg.weighting_nu))
        # settings hold the actual nu, not the cubic slider position
        s.PALC_config.weighting_nu = float(alg.weighting_nu)
    if alg.weighting == "target_slope":
        ts = setup.target_slope
        s.gui_opt_weight_link.value = str(len(ts.points) - 2)
        s.drag_target([p.distance for p in ts.points],
                      [p.level_db for p in ts.points], ts.reference_distance)

    # run
    s.start_calc(True)
    return s


def _collect(s: _LegacySession, setup: Setup) -> Dict[str, np.ndarray]:
    def arr(value: Any) -> np.ndarray:
        return np.array(value, dtype=float)

    tr, trr = s.Tech_res, s.Tech_res_ref
    out = {
        "tilt_angles_deg": arr(s.results["b_gamma_tilt_deg"]),
        "inter_cabinet_angles_deg": arr(s.results["c_gamma_tilt_deg_diff"]),
        "reference_tilt_angles_deg": arr(s.Ref_arr.gamma_tilt_deg),
        "frequencies_hz": arr(tr.f),
        "spl_db": arr(tr.p_SPL),
        "reference_spl_db": arr(trr.p_SPL),
        "spl_over_distance_x": arr(s.SPLoverX_dict.data["x"]),
        "spl_over_distance_db": arr(s.SPLoverX_dict.data["SPL"]),
        "reference_spl_over_distance_db": arr(s.SPLoverX_ref_dict.data["SPL"]),
        "homogeneity_db": arr(tr.H),
        "homogeneity_low_db": arr(tr.H_dist[0]),
        "homogeneity_high_db": arr(tr.H_dist[1]),
        "reference_homogeneity_db": arr(trr.H),
        "histogram_steps": arr(tr.Hist_steps),
        "histogram_tops": arr(tr.Hist_tops),
        "reference_histogram_steps": arr(trr.Hist_steps),
        "reference_histogram_tops": arr(trr.Hist_tops),
        "venue_x": arr(s.created_pal.xline),
        "venue_y": arr(s.created_pal.yline),
    }
    if setup.algorithm.weighting == "target_slope":
        out["target_x"] = arr(s.Opt_SPLoverX_dict.data["x"])
        out["target_db"] = arr(s.Opt_SPLoverX_dict.data["SPL"])
        out["weighting_weights"] = arr(s.PALC_config.weighting_weights)
    return out


def run_legacy(setup: Setup) -> Dict[str, np.ndarray]:
    """Compute a setup with the legacy code, as the GUI would.

    Parameters
    ----------
    setup : Setup
        Setup to replay; a measured directivity ``path`` must be absolute
        (as :func:`pypalc.core.settings.load_setup` returns it).

    Returns
    -------
    dict of str to numpy.ndarray
        Results, keyed as documented in the task (result-key contract).

    Raises
    ------
    LegacyGuiError
        If the GUI would have shown an error instead of results.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        session = _replay(setup)
    return _collect(session, setup)


def case_path(case: str) -> Path:
    """Path of the setup file of a baseline case."""
    return CASES_DIR / ("%s.yaml" % case)


def baseline_path(case: str) -> Path:
    """Path of the recorded baseline of a case."""
    return BASELINE_DIR / ("%s.npz" % case)


def main(argv: Optional[List[str]] = None) -> int:
    """Record ``tests/baseline/<case>.npz`` for the given cases."""
    cases = list(argv if argv is not None else sys.argv[1:]) or list(CASES)
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    for case in cases:
        result = run_legacy(load_setup(case_path(case)))
        np.savez(baseline_path(case), **result)
        print("wrote %s" % baseline_path(case).relative_to(REPO))
    return 0


if __name__ == "__main__":
    sys.exit(main())
