"""Page 4 of the GUI: algorithm options, calculation and SPL over distance.

author: Arne Hoelter
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from bokeh.layouts import column, row
from bokeh.models import (ColumnDataSource, CustomJS, FuncTickFormatter,
                          LayoutDOM)
from bokeh.models.tools import PointDrawTool
from bokeh.models.widgets import (Button, DataTable, Panel, Paragraph,
                                  RangeSlider, Select, Slider, TableColumn,
                                  Tabs, TextInput, Toggle)
from bokeh.plotting import figure

from pypalc.core.palc import clamp_fixed_angle, fixed_angle_borders
from pypalc.core.target_slope import get_weight_links, snap_to_distance_grid
from pypalc.gui import DOWNLOAD_JS
from pypalc.gui.page_venue import RULE

if TYPE_CHECKING:
    from .app import PalcApp

logger = logging.getLogger(__name__)

#: Weightings available without target slope.
WEIGHTINGS_WITHOUT_TARGET_SLOPE = ['Without', 'Linear Spacing',
                                   'Logarithmic Spacing']


class AlgorithmPage:
    """Page 4: PALC variant, gap handling, weighting, tolerance, fixed
    angle, the start toggle, the results table and the SPL over distance.

    Parameters
    ----------
    app : PalcApp
        The session the page belongs to.
    """

    def __init__(self, app: PalcApp) -> None:
        self.app = app
        state = app.state
        self._build_widgets()
        # sources
        p_SPL = list(zip([0, 0]))
        x_vert = list(zip([0, 0]))
        self.results_source = ColumnDataSource(data=dict(
            a_num_LS=np.linspace(1, state.config.N, num=0),
            b_gamma_tilt_deg=np.array([]), c_gamma_tilt_deg_diff=np.array([])))
        """Tilt angles of the cabinets computed by PALC."""
        self.spl_over_distance_source = ColumnDataSource(data=dict(
            SPL=p_SPL, x=x_vert, x_v=x_vert, y_v=x_vert))
        """SPL over the distance between the array and the venue points,
        normalized to 0 dB (at the reference point for target slope)."""
        self.reference_spl_over_distance_source = ColumnDataSource(data=dict(
            SPL=p_SPL, x=x_vert, x_v=x_vert, y_v=x_vert))
        """SPL over distance of the reference array."""
        self.target_source = ColumnDataSource(data=dict(
            SPL=p_SPL, x=x_vert, x_v=[0, 0], y_v=[0, 0]))
        """Target slope points (draggable)."""
        self.target_reference_source = ColumnDataSource(data=dict(
            x_ref=[10], SPL=[0], x_v=[0], y_v=[0]))
        """Reference point of the target slope (draggable)."""
        self.target_region_source = ColumnDataSource(data=dict(
            x=[0, 0, 0, 0], y=[0, 0, 0, 0]))
        """Polygon of the optimization region."""
        self.results_table = DataTable(
            source=self.results_source,
            columns=[TableColumn(field="a_num_LS", title="LS", width=20),
                     TableColumn(field="b_gamma_tilt_deg",
                                 title="Total Angle", width=80),
                     TableColumn(field="c_gamma_tilt_deg_diff", title="ICA",
                                 width=80)],
            index_position=None, width=250, height=700, fit_columns=False)
        self._build_spl_over_distance_figure()
        self.tabs = Tabs(tabs=[
            Panel(child=app.venue_figure, title="Venue Slice"),
            Panel(child=column([self.frequency_range_slider,
                                self.spl_over_distance_figure]),
                  title="SPL over Distance")])
        self._wire_callbacks()

    def _build_widgets(self) -> None:
        """Create the widgets of the page."""
        fixed_angle = self.app.state.optimized_array.fixed_angle
        self.palc_variant_select = Select(
            title="18. PALC-Algorithm", value="PALC 2",
            options=["PALC 1", "PALC 2", "PALC 3"], width=255, height=50)
        self.gap_handling_select = Select(
            title="19. Gap Handling Approach", value="Without",
            options=["Without", "Hard Margin", "Soft Margin"], width=255,
            height=50)
        self.soft_margin_slider = Slider(
            title="20. Strength of Soft Margin", value=1.0, start=0.1,
            end=1.9, step=0.05, disabled=True, width=255)
        self.weighting_select = Select(
            title="21. Weighting", value="Without",
            options=["Without", "Linear Spacing", "Logarithmic Spacing",
                     "Target Slope"], width=255, height=50)
        self.hinges_select = Select(title="Hinges in Target", value="0",
                                    options=["0", "1", "2"], width=105,
                                    height=50, visible=False)
        """Number of hinges of the target slope; the choosable numbers
        depend on the number of audience lines."""
        # cubic spaced slider: the stored ν is the value to the power of 3
        self.weighting_nu_slider = Slider(
            title="Weighting Parameter ν", value=np.cbrt(1.00),
            start=np.cbrt(0.01), end=np.cbrt(10.00), step=0.01,
            disabled=True, width=255,
            format=FuncTickFormatter(
                code="return Math.pow(tick,3).toFixed(2)"))
        self.weighting_title = Paragraph(
            text="22. Adjustment of ν and Step Size", width=255, height=20)
        self.nu_plus_button = Button(label="+", button_type="success",
                                     disabled=True, width=70)
        self.nu_minus_button = Button(label="-", button_type="danger",
                                      disabled=True, width=70)
        self.nu_step_select = Select(
            value='0.05', options=['0.01', '0.05', '0.10', '0.50'],
            disabled=True, width=80, height=30)
        """Step size of the + and - buttons."""
        self.tolerance_slider = Slider(
            title="23. Tolerance (Abort Criterion) in m", value=1.0,
            start=0.05, end=5, step=0.05, width=255)
        self.use_fixed_angle_select = Select(
            title="24. Fix Angle of Uppermost Cabinet?", value="No",
            options=["No", "Yes"], disabled=True, width=255, height=50)
        self.fixed_angle_input = TextInput(
            title="25. Angle of Uppermost Cabinet in deg", value="0.0",
            disabled=True, width=255)
        self.fixed_angle_text = Paragraph(
            text='Max: ' + str(fixed_angle[0]) + ', Min: '
                 + str(fixed_angle[2]))
        """Allowed range of the fixed angle."""
        self.start_toggle = Toggle(label="26. PALC Run", button_type="success",
                                   disabled=True, width=255, active=False)
        """Starts the PALC computation; active while it runs."""
        self.clear_results_button = Button(
            label="27. Delete Result Visualization", button_type="danger",
            width=255)
        self.download_angles_button = Button(label="Download PALC Angles",
                                             button_type="success", width=255)
        self.frequency_range_slider = RangeSlider(
            title="Frequency Range", width=690, start=0, end=20000,
            value=(200, 8000), step=20)
        self.download_spl_button = Button(label="Download SPL",
                                          button_type="success", width=255)
        self.rule_gap = Paragraph(text=RULE)
        self.rule_tolerance = Paragraph(text=RULE)

    def _build_spl_over_distance_figure(self) -> None:
        """SPL over distance figure with the draggable target slope."""
        fig = figure(
            title="SPL over Distance (maximum SPL normalized to 0 dB)",
            x_axis_label="Distance LSA to Receiver in m",
            y_axis_label="SPL change in dB", toolbar_sticky=False,
            tools="box_zoom,pan,wheel_zoom,reset,hover,save, tap",
            tooltips=[("Title", "$name"), ("Index", "$index"),
                      ("SPL in dB", "$y"), ("Distance", "$x"),
                      ("(x,y) of Venue Slice in m", "@x_v, @y_v")],
            y_range=(-15, 0.3), height=550, width=690)
        fig.toolbar.logo = None
        fig.line('x', 'SPL', source=self.spl_over_distance_source,
                 color="lightskyblue", name="PALC",
                 legend_label='PALC (optimized)')
        fig.line('x', 'SPL', source=self.reference_spl_over_distance_source,
                 color="orange", name="Reference",
                 legend_label='Reference LSA')
        target_line = fig.line('x', 'SPL', source=self.target_source,
                               color="lime", name="Reference",
                               legend_label='Target Slope')
        target_points = fig.circle('x', 'SPL', source=self.target_source,
                                   line_color="yellow", fill_color="black",
                                   line_width=2, name="Reference",
                                   legend_label='Target Slope')
        reference_point = fig.circle('x_ref', 'SPL',
                                     source=self.target_reference_source,
                                     color="darkred", fill_color="red",
                                     size=10, name="Reference Point",
                                     legend_label="Reference Point")
        fig.patch('x', 'y', source=self.target_region_source, alpha=0.2,
                  line_width=0.1)
        fig.legend.location, fig.legend.orientation = 'top_right', \
            'horizontal'
        fig.legend.margin, fig.legend.glyph_width = 0, 30
        fig.legend.background_fill_alpha, fig.legend.spacing = 0.5, 15
        self.target_source.selected.on_change('indices',
                                              self.target_selected)
        self.target_reference_source.selected.on_change('indices',
                                                        self.target_selected)
        draw_tool = PointDrawTool(renderers=[target_points, target_line,
                                             reference_point], add=False)
        fig.add_tools(draw_tool)
        fig.toolbar.active_tap = draw_tool
        fig.background_fill_color, fig.border_fill_color = None, None
        fig.output_backend = "svg"
        self.spl_over_distance_figure = fig

    def _wire_callbacks(self) -> None:
        """Connect the widgets with their callbacks."""
        app = self.app
        self.palc_variant_select.on_change("value", app.store_value)
        self.gap_handling_select.on_change("value", app.store_value)
        self.soft_margin_slider.on_change("value", app.store_value)
        self.weighting_select.on_change("value", self.choose_weighting)
        self.hinges_select.on_change("value", self.choose_hinges)
        self.weighting_nu_slider.on_change("value", self.store_nu)
        self.nu_plus_button.on_click(self.increase_nu)
        self.nu_minus_button.on_click(self.decrease_nu)
        self.nu_step_select.on_change("value", self.set_nu_step)
        self.tolerance_slider.on_change("value", app.store_value)
        self.use_fixed_angle_select.on_change("value",
                                              self.choose_fixed_angle)
        self.fixed_angle_input.on_change("value", app.store_value)
        self.start_toggle.on_click(self.start_toggled)
        self.clear_results_button.on_click(app.clear_result_figure)
        self.target_source.on_change('data', self.snap_target)
        self.target_reference_source.on_change('data', self.snap_target)
        download_js = DOWNLOAD_JS.read_text()
        self.download_angles_button.js_on_click(CustomJS(
            args=dict(source=self.results_source), code=download_js))
        self.frequency_range_slider.on_change("value", app.store_value)
        for source in (self.spl_over_distance_source, self.target_source,
                       self.target_reference_source,
                       self.reference_spl_over_distance_source,
                       app.pages[5].homogeneity_source):
            self.download_spl_button.js_on_click(CustomJS(
                args=dict(source=source), code=download_js))

    def layout(self) -> LayoutDOM:
        """Body of the page below the header.

        Returns
        -------
        LayoutDOM
        """
        app = self.app
        col0 = column([self.palc_variant_select, self.gap_handling_select,
                       self.soft_margin_slider, self.rule_gap,
                       row([self.weighting_select, self.hinges_select]),
                       self.weighting_nu_slider, self.weighting_title,
                       row([self.nu_plus_button, self.nu_minus_button,
                            self.nu_step_select]),
                       self.rule_tolerance, self.tolerance_slider,
                       self.use_fixed_angle_select, self.fixed_angle_input,
                       self.fixed_angle_text], width=270)
        col1 = column([app.pages[1].check_venue_button, self.start_toggle,
                       self.clear_results_button, self.download_angles_button,
                       app.config_error_info, self.results_table,
                       self.download_spl_button], width=270)
        col2 = column([app.venue_axes_row(), self.tabs], width=690)
        return row([col0, col1, col2])

    # ------------------------------------------------------------------
    # Widget arrangement
    # ------------------------------------------------------------------

    def arrange_gap_handling_widgets(self) -> None:
        """Enable the soft margin strength only for soft margin."""
        if self.gap_handling_select.value == "Without":
            self.soft_margin_slider.disabled = True
            self.soft_margin_slider.value = 1.0
        elif self.gap_handling_select.value == "Hard Margin":
            self.soft_margin_slider.disabled = True
            self.soft_margin_slider.value = 1.0
        elif self.gap_handling_select.value == "Soft Margin":
            self.soft_margin_slider.disabled = False

    def exclude_hard_margin_target_slope(self) -> None:
        """Hard margin and target slope cannot be selected at once."""
        if self.weighting_select.value == 'Target Slope':
            self.gap_handling_select.options = ['Without', 'Soft Margin']
        elif self.gap_handling_select.value == 'Hard Margin':
            self.weighting_select.options = list(
                WEIGHTINGS_WITHOUT_TARGET_SLOPE)
        else:
            self.gap_handling_select.options = ['Without', 'Hard Margin',
                                                'Soft Margin']
            self.weighting_select.options = \
                WEIGHTINGS_WITHOUT_TARGET_SLOPE + ['Target Slope']

    def _arrange_weighting_widgets(self) -> None:
        """Enable the ν widgets for linear and logarithmic spacing."""
        if self.weighting_select.value in ['Without', 'Target Slope']:
            self.nu_plus_button.disabled = True
            self.nu_minus_button.disabled = True
            self.nu_step_select.disabled = True
            self.weighting_nu_slider.disabled = True
            self.weighting_nu_slider.value = 1.00
        elif self.weighting_select.value in ['Linear Spacing',
                                             'Logarithmic Spacing']:
            self.nu_plus_button.disabled = False
            self.nu_minus_button.disabled = False
            self.nu_step_select.disabled = False
            self.weighting_nu_slider.disabled = False

    def round_fixed_angle(self) -> None:
        """Clamp the fixed angle of the uppermost cabinet to its range."""
        config = self.app.state.config
        fixed_angle = self.app.state.optimized_array.fixed_angle
        max_deg, min_deg = fixed_angle[0], fixed_angle[2]
        if self.fixed_angle_input.disabled == False and \
                (config.fixed_angle > max_deg or
                 config.fixed_angle < min_deg):  # noqa: E712
            config.fixed_angle = clamp_fixed_angle(config.fixed_angle,
                                                   max_deg, min_deg)
            self.fixed_angle_input.value = str(config.fixed_angle)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def start_toggled(self, active: bool) -> None:
        """Run the calculation when the start toggle is pressed.

        Parameters
        ----------
        active : bool
            New state of the toggle.

        Returns
        -------
        None.
        """
        if active:
            self.app.run_calculation()

    def choose_weighting(self, attr, old, new) -> None:
        """Arrange the page for the selected weighting.

        Selecting "Target Slope" runs the initial calculation and shows the
        SPL over distance with the target slope.

        Parameters
        ----------
        attr : str
            Changed attribute (value).
        old, new : str
            Old and new weighting.

        Returns
        -------
        None.
        """
        self._arrange_weighting_widgets()
        if self.weighting_select.value in ['Target Slope']:
            self.app.run_calculation()
            self.app.state.target.init = True
            self.tabs.active = 1
            self.hinges_select.visible = True
            self.weighting_select.width = 135
        else:
            self.hinges_select.visible = False
            self.weighting_select.width = 255
        self.app.state.config.store_config('use_weighting',
                                           self.weighting_select.value)
        self.exclude_hard_margin_target_slope()

    def set_nu_step(self, attr, old, new) -> None:
        """Set the step size of the ν slider.

        Parameters
        ----------
        attr : str
            Changed attribute (value).
        old, new : str
            Old and new step size.

        Returns
        -------
        None.
        """
        self.weighting_nu_slider.step = float(self.nu_step_select.value)

    def increase_nu(self) -> None:
        """Increase ν by the step size."""
        self.weighting_nu_slider.value += float(self.nu_step_select.value)

    def decrease_nu(self) -> None:
        """Decrease ν by the step size."""
        self.weighting_nu_slider.value -= float(self.nu_step_select.value)

    def store_nu(self, attr, old, new) -> None:
        """Store ν (the slider value to the power of 3).

        Parameters
        ----------
        attr : str
            Changed attribute (value).
        old, new : float
            Old and new slider value.

        Returns
        -------
        None.
        """
        self.app.state.config.weighting_nu = np.round(
            self.weighting_nu_slider.value**3, decimals=2)

    def choose_fixed_angle(self, attr, old, new) -> None:
        """Enable or disable the fixed angle of the uppermost cabinet.

        Enabling computes the allowed range of the angle and proposes an
        angle in it.

        Parameters
        ----------
        attr : str
            Changed attribute (value).
        old, new : str
            'Yes' or 'No'.

        Returns
        -------
        None.
        """
        state = self.app.state
        if self.use_fixed_angle_select.value == 'Yes' and self.app.has_venue():
            fixed_angle = fixed_angle_borders(state.venue.palc_venue.pal[-1],
                                              state.config)
            state.optimized_array.fixed_angle = fixed_angle
            self.fixed_angle_input.disabled = False
            if fixed_angle[1] < fixed_angle[0]:
                self.fixed_angle_input.value = str(fixed_angle[1])
            else:
                self.fixed_angle_input.value = str(fixed_angle[0]-1)
            self.fixed_angle_text.text = \
                'Max: ' + str(np.round(fixed_angle[0], decimals=2)) + \
                ', Min: ' + str(np.round(fixed_angle[2], decimals=2))
            state.config.use_fixed_angle = True
        elif self.use_fixed_angle_select.value == 'No':
            self.fixed_angle_input.disabled = True
            state.optimized_array.fixed_angle = [0, 0, 0]
            state.config.use_fixed_angle = False

    def target_selected(self, attr, old, new) -> None:
        """A target point was selected for dragging; allow updates of the
        target sources again (see :meth:`snap_target`)."""
        self.app.state.target.is2update = 0

    def choose_hinges(self, attr, old, new) -> None:
        """Place the selected number of hinges on the target slope.

        Parameters
        ----------
        attr : str
            Changed attribute (value).
        old, new : str
            Old and new number of hinges.

        Returns
        -------
        None.
        """
        state = self.app.state
        SPLoverX, drawn = state.spl_over_distance, state.drawn
        ind = get_weight_links(int(new), SPLoverX)
        self.target_source.data.update(dict(
            SPL=np.array(SPLoverX.SPL)[ind], x=SPLoverX.x[ind],
            x_v=drawn.xline[ind], y_v=drawn.yline[ind]))

    def snap_target(self, attr, old, new) -> None:
        """Snap the dragged target slope to the SPL over distance grid.

        Moves the target points and the reference point to the nearest
        distances of the grid (:func:`snap_to_distance_grid`), writes them
        back into their sources and updates the optimization region.
        Triggered by data changes of the target sources.

        Parameters
        ----------
        attr : str
            Changed attribute (data).
        old, new : dict
            Old and new data.

        Returns
        -------
        None.
        """
        Opt_w = self.app.state.target
        grid = self.spl_over_distance_source.data
        x, x_v, y_v = grid['x'], grid['x_v'], grid['y_v']
        snap_to_distance_grid(Opt_w, x, x_v, y_v, self.target_source.data,
                              self.target_reference_source.data['x_ref'])
        xv_ind = np.argmin(np.abs(np.array(x)-Opt_w.x_ref))
        if Opt_w.is2update < 4:
            Opt_w.is2update += 1
            self.target_source.data.update(dict(x=Opt_w.x, SPL=Opt_w.SPL,
                                                x_v=Opt_w.x_v, y_v=Opt_w.y_v))
            self.target_reference_source.data.update(dict(
                x_ref=[Opt_w.x_ref], SPL=[Opt_w.SPL_interp[Opt_w.ref_ind]],
                x_v=[x_v[xv_ind]], y_v=[y_v[xv_ind]]))
        self.target_region_source.data.update(dict(
            x=[Opt_w.x[0], Opt_w.x[0], Opt_w.x[-1], Opt_w.x[-1]],
            y=[Opt_w.SPL[0]+100, Opt_w.SPL[-1]-100, Opt_w.SPL[-1]-100,
               Opt_w.SPL[0]+100]))
