"""Page 4 of the GUI: algorithm options, calculation and SPL over distance.

author: Arne Hoelter
"""
from __future__ import annotations

import base64
import binascii
import logging
from typing import TYPE_CHECKING, Sequence

import numpy as np
from bokeh.layouts import column, row
from bokeh.models import (Button, ColumnDataSource, CustomJS,
                          CustomJSTickFormatter, DataTable, Div, FileInput,
                          LayoutDOM, Paragraph, PointDrawTool, RangeSlider,
                          Select, Slider, TableColumn, TabPanel, Tabs,
                          TextInput, Toggle)
from bokeh.plotting import figure

from pypalc.core.errors import SetupError
from pypalc.core.palc import clamp_fixed_angle, fixed_angle_borders
from pypalc.core.settings import (TargetPoint, dump_setup, load_setup_text,
                                  validate)
from pypalc.core.target_slope import hinge_indices, snap_to_distance_grid
from pypalc.gui import (CALLBACK_ERRORS, DOWNLOAD_JS, DOWNLOAD_TEXT_JS,
                        skipped_while_applying_setup)
from pypalc.gui.page_venue import RULE

if TYPE_CHECKING:
    from .app import PalcApp

logger = logging.getLogger(__name__)

#: Weightings available without target slope.
WEIGHTINGS_WITHOUT_TARGET_SLOPE = ['Without', 'Linear Spacing',
                                   'Logarithmic Spacing']
#: Names of the CSV files "Download PALC Angles" (results) and "Download
#: SPL" (the others) download, by source.
DOWNLOAD_FILE_NAMES = {
    'results': 'tilt_angles.csv',
    'spl_over_distance': 'spl_over_distance.csv',
    'target': 'target_slope.csv',
    'target_reference': 'target_reference.csv',
    'reference_spl_over_distance': 'reference_spl_over_distance.csv',
    'homogeneity': 'homogeneity.csv',
}
#: Name of the file "Save Setup" downloads.
SETUP_FILE_NAME = 'pypalc_setup.yaml'
#: Key of the error of a setup file that gives the measured directivity as
#: a path, and the error text the GUI shows for it.
PATH_DIRECTIVITY_KEY = 'loudspeaker.measured_directivity.path'
PATH_DIRECTIVITY_MESSAGE = (
    "the GUI cannot read a measured directivity file given as a path. "
    "Upload the data on page 2 and use Save Setup (it embeds the data), or "
    "run the setup file with the headless runner (python -m pypalc run)")


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
        empty = list(zip([0, 0]))
        # The columns of the downloaded sources are those of the files of
        # the headless export (pypalc.core.export), in the same order.
        self.results_source = ColumnDataSource(data=dict(
            cabinet=np.linspace(1, state.config.cabinet_count, num=0),
            tilt_angle_deg=np.array([]),
            inter_cabinet_angle_deg=np.array([])))
        """Tilt angles of the cabinets computed by PALC."""
        self.spl_over_distance_source = ColumnDataSource(data=dict(
            distance_m=empty, spl_db=empty, x_m=empty, y_m=empty))
        """SPL over the distance between the array and the venue points,
        normalized to 0 dB (at the reference point for target slope)."""
        self.reference_spl_over_distance_source = ColumnDataSource(data=dict(
            distance_m=empty, reference_spl_db=empty, x_m=empty, y_m=empty))
        """SPL over distance of the reference array."""
        self.target_source = ColumnDataSource(data=dict(
            distance_m=empty, level_db=empty, x_m=[0, 0], y_m=[0, 0]))
        """Target slope points (draggable)."""
        self.target_reference_source = ColumnDataSource(data=dict(
            distance_m=[10], level_db=[0], x_m=[0], y_m=[0]))
        """Reference point of the target slope (draggable)."""
        self.target_region_source = ColumnDataSource(data=dict(
            x=[0, 0, 0, 0], y=[0, 0, 0, 0]))
        """Polygon of the optimization region."""
        self.setup_file_source = ColumnDataSource(data=dict(text=[]))
        """Text of the setup file "Save Setup" downloads."""
        self.results_table = DataTable(
            source=self.results_source,
            columns=[TableColumn(field="cabinet", title="LS", width=20),
                     TableColumn(field="tilt_angle_deg",
                                 title="Total Angle", width=80),
                     TableColumn(field="inter_cabinet_angle_deg",
                                 title="ICA", width=80)],
            index_position=None, width=250, height=700, fit_columns=False)
        self._build_spl_over_distance_figure()
        self.tabs = Tabs(tabs=[
            TabPanel(child=app.venue_figure, title="Venue Slice"),
            TabPanel(child=column([self.frequency_range_slider,
                                   self.spl_over_distance_figure]),
                     title="SPL over Distance")])
        self._wire_callbacks()

    def _build_widgets(self) -> None:
        """Create the widgets of the page."""
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
            format=CustomJSTickFormatter(
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
        self.fixed_angle_text = Paragraph(text='')
        """Allowed range of the fixed angle."""
        self.show_initial_fixed_angle_range()
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
        self.save_setup_button = Button(label="Save Setup",
                                        button_type="success", width=255)
        """Downloads the setup as a YAML file."""
        self.load_setup_title = Div(text="Load Setup")
        self.load_setup_input = FileInput(accept=".yaml,.yml", width=255)
        """Upload of a setup file; loading it restores the setup and runs
        the calculation."""
        self.rule_gap = Paragraph(text=RULE)
        self.rule_tolerance = Paragraph(text=RULE)

    def _build_spl_over_distance_figure(self) -> None:
        """SPL over distance figure with the draggable target slope."""
        spl_figure = figure(
            title="SPL over Distance (maximum SPL normalized to 0 dB)",
            x_axis_label="Distance LSA to Receiver in m",
            y_axis_label="SPL change in dB",
            tools="box_zoom,pan,wheel_zoom,reset,hover,save, tap",
            tooltips=[("Title", "$name"), ("Index", "$index"),
                      ("SPL in dB", "$y"), ("Distance", "$x"),
                      ("(x,y) of Venue Slice in m", "@x_m, @y_m")],
            y_range=(-15, 0.3), height=550, width=690)
        spl_figure.toolbar.logo = None
        spl_figure.line('distance_m', 'spl_db',
                        source=self.spl_over_distance_source,
                        color="lightskyblue", name="PALC",
                        legend_label='PALC (optimized)')
        spl_figure.line('distance_m', 'reference_spl_db',
                        source=self.reference_spl_over_distance_source,
                        color="orange", name="Reference",
                        legend_label='Reference LSA')
        target_line = spl_figure.line('distance_m', 'level_db',
                                      source=self.target_source,
                                      color="lime", name="Reference",
                                      legend_label='Target Slope')
        target_points = spl_figure.circle('distance_m', 'level_db',
                                          source=self.target_source,
                                          line_color="yellow",
                                          fill_color="black", line_width=2,
                                          name="Reference",
                                          legend_label='Target Slope')
        reference_point = spl_figure.circle(
            'distance_m', 'level_db', source=self.target_reference_source,
            color="darkred", fill_color="red", size=10,
            name="Reference Point", legend_label="Reference Point")
        spl_figure.patch('x', 'y', source=self.target_region_source,
                         alpha=0.2, line_width=0.1)
        legend = spl_figure.legend
        legend.location, legend.orientation = 'top_right', 'horizontal'
        legend.margin, legend.glyph_width = 0, 30
        legend.background_fill_alpha, legend.spacing = 0.5, 15
        self.target_source.selected.on_change('indices',
                                              self.target_selected)
        self.target_reference_source.selected.on_change('indices',
                                                        self.target_selected)
        draw_tool = PointDrawTool(renderers=[target_points, target_line,
                                             reference_point], add=False)
        spl_figure.add_tools(draw_tool)
        spl_figure.toolbar.active_tap = draw_tool
        spl_figure.background_fill_color = None
        spl_figure.border_fill_color = None
        spl_figure.output_backend = "svg"
        self.spl_over_distance_figure = spl_figure

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
            args=dict(source=self.results_source,
                      filename=DOWNLOAD_FILE_NAMES['results']),
            code=download_js))
        self.frequency_range_slider.on_change("value", app.store_value)
        for name, source in (
                ('spl_over_distance', self.spl_over_distance_source),
                ('target', self.target_source),
                ('target_reference', self.target_reference_source),
                ('reference_spl_over_distance',
                 self.reference_spl_over_distance_source),
                ('homogeneity', app.pages[5].homogeneity_source)):
            self.download_spl_button.js_on_click(CustomJS(
                args=dict(source=source, filename=DOWNLOAD_FILE_NAMES[name]),
                code=download_js))
        self.save_setup_button.on_click(self.save_setup)
        # the source must belong to the document so that its JavaScript
        # callback runs in the browser; the layout reaches it only through
        # the args of this (empty) callback of the button
        self.save_setup_button.js_on_click(CustomJS(
            args=dict(source=self.setup_file_source), code=''))
        self.setup_file_source.js_on_change('data', CustomJS(
            args=dict(source=self.setup_file_source,
                      filename=SETUP_FILE_NAME),
            code=DOWNLOAD_TEXT_JS.read_text()))
        self.load_setup_input.on_change('value', self.load_setup)

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
                       self.download_spl_button, self.save_setup_button,
                       self.load_setup_title, self.load_setup_input],
                      width=270)
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

    def update_fixed_angle_range(self) -> list:
        """Compute and show the allowed range of the fixed angle.

        The range depends on the venue (it must be checked) and on the
        position and height of the array in the configuration.

        Returns
        -------
        list of float
            Maximum, mid and minimum tilt angle in degree, also stored in
            ``state.optimized_array.fixed_angle``.
        """
        state = self.app.state
        fixed_angle = fixed_angle_borders(state.venue.palc_venue.segments[-1],
                                          state.config)
        state.optimized_array.fixed_angle = fixed_angle
        self.fixed_angle_text.text = \
            'Max: ' + str(np.round(fixed_angle[0], decimals=2)) + \
            ', Min: ' + str(np.round(fixed_angle[2], decimals=2))
        return fixed_angle

    def show_initial_fixed_angle_range(self) -> None:
        """Show the range of the fixed angle as in a new session."""
        fixed_angle = self.app.state.optimized_array.fixed_angle
        self.fixed_angle_text.text = 'Max: ' + str(fixed_angle[0]) + \
            ', Min: ' + str(fixed_angle[2])

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

    @skipped_while_applying_setup
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

    @skipped_while_applying_setup
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
        self.show_weighting()

    def show_weighting(self) -> bool:
        """Arrange the page for the selected weighting (body of
        :meth:`choose_weighting`).

        Returns
        -------
        bool
            False if the initial calculation of "Target Slope" failed,
            otherwise True.
        """
        done = True
        self._arrange_weighting_widgets()
        if self.weighting_select.value in ['Target Slope']:
            done = self.app.run_calculation()
            self.app.state.target.initialized = True
            self.tabs.active = 1
            self.hinges_select.visible = True
            self.weighting_select.width = 135
        else:
            self.hinges_select.visible = False
            self.weighting_select.width = 255
        self.app.state.config.store_config('use_weighting',
                                           self.weighting_select.value)
        self.exclude_hard_margin_target_slope()
        return done

    @skipped_while_applying_setup
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

    @skipped_while_applying_setup
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

    @skipped_while_applying_setup
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
            fixed_angle = self.update_fixed_angle_range()
            self.fixed_angle_input.disabled = False
            if fixed_angle[1] < fixed_angle[0]:
                self.fixed_angle_input.value = str(fixed_angle[1])
            else:
                self.fixed_angle_input.value = str(fixed_angle[0]-1)
            state.config.use_fixed_angle = True
        elif self.use_fixed_angle_select.value == 'No':
            self.fixed_angle_input.disabled = True
            state.optimized_array.fixed_angle = [0, 0, 0]
            state.config.use_fixed_angle = False

    def target_selected(self, attr, old, new) -> None:
        """A target point was selected for dragging; allow updates of the
        target sources again (see :meth:`snap_target`)."""
        self.app.state.target.update_count = 0

    @skipped_while_applying_setup
    def choose_hinges(self, attr, old, new) -> None:
        """Place the selected number of hinges on the target slope.

        The hinges are placed on the SPL over distance of the last
        calculation; without a calculation nothing happens.

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
        self.place_hinges(int(new))

    def place_hinges(self, hinges: int) -> None:
        """Place hinges on the target slope (body of :meth:`choose_hinges`).

        The hinges are placed on the SPL over distance of the last
        calculation; without a calculation nothing happens.

        Parameters
        ----------
        hinges : int
            Number of hinges.

        Returns
        -------
        None.
        """
        state = self.app.state
        if state.result is None:
            return
        spl_over_distance = state.result.optimized.spl_over_distance
        drawn_lines = state.drawn_lines
        ind = hinge_indices(hinges, spl_over_distance)
        self.target_source.data.update(dict(
            level_db=np.array(spl_over_distance.levels_db)[ind],
            distance_m=spl_over_distance.distances[ind],
            x_m=drawn_lines.points_x[ind], y_m=drawn_lines.points_y[ind]))

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
        target = self.app.state.target
        grid = self.spl_over_distance_source.data
        distances, points_x, points_y = \
            grid['distance_m'], grid['x_m'], grid['y_m']
        snap_to_distance_grid(target, distances, points_x, points_y,
                              self.target_source.data,
                              self.target_reference_source.data['distance_m'])
        reference_point = np.argmin(np.abs(np.array(distances)
                                           - target.reference_distance))
        if target.update_count < 4:
            target.update_count += 1
            self.target_source.data.update(dict(
                distance_m=target.distances, level_db=target.levels_db,
                x_m=target.points_x, y_m=target.points_y))
            self.target_reference_source.data.update(dict(
                distance_m=[target.reference_distance],
                level_db=[target.region_levels_db[target.reference_index]],
                x_m=[points_x[reference_point]],
                y_m=[points_y[reference_point]]))
        first, last = target.distances[0], target.distances[-1]
        self.target_region_source.data.update(dict(
            x=[first, first, last, last],
            y=[target.levels_db[0]+100, target.levels_db[-1]-100,
               target.levels_db[-1]-100, target.levels_db[0]+100]))

    def shown_target(self) -> tuple:
        """The shown target slope: ``(distances, levels, reference
        distance)`` as lists of float."""
        target = self.target_source.data
        reference = self.target_reference_source.data
        return ([float(x) for x in target['distance_m']],
                [float(level) for level in target['level_db']],
                [float(x) for x in reference['distance_m']])

    def drag_target(self, points: Sequence[TargetPoint],
                    reference_distance: float) -> None:
        """Move the target points and the reference point, as dragging them
        in the browser does; they snap to the grid (:meth:`snap_target`).

        Parameters
        ----------
        points : sequence of TargetPoint
            The target points; as many as the target slope shows.
        reference_distance : float
            Distance of the reference point in m.

        Returns
        -------
        None.
        """
        data = self.target_source.data
        self.target_selected('indices', [], [0])
        self.target_source.data = dict(
            distance_m=[float(p.distance) for p in points],
            level_db=[float(p.level_db) for p in points],
            x_m=list(data['x_m']), y_m=list(data['y_m']))
        data = self.target_reference_source.data
        self.target_selected('indices', [], [0])
        self.target_reference_source.data = dict(
            distance_m=[float(reference_distance)],
            level_db=list(data['level_db']),
            x_m=list(data['x_m']), y_m=list(data['y_m']))

    # ------------------------------------------------------------------
    # Setup files
    # ------------------------------------------------------------------

    def save_setup(self) -> None:
        """Download the entered setup as a YAML file ("Save Setup").

        Writes the text of :func:`~.setup_io.setup_to_save` into
        ``setup_file_source``; its JavaScript callback downloads it as
        :data:`SETUP_FILE_NAME`. An uploaded measured directivity is
        embedded. A setup that cannot be read or is invalid is not saved;
        the reason is shown in the error text.

        Returns
        -------
        None.
        """
        from .setup_io import setup_to_save
        app = self.app
        try:
            setup = setup_to_save(app)
            validate(setup)
        except SetupError as exc:
            app.config_error_info.text = \
                'Error: setup not saved, invalid input at %s: %s' \
                % (exc.key, exc.message)
            return
        # empty first, so that saving the same text again downloads again
        self.setup_file_source.data = dict(text=[])
        self.setup_file_source.data = dict(text=[dump_setup(setup)])

    def load_setup(self, attr, old, new) -> None:
        """Load an uploaded setup file ("Load Setup").

        Decodes the file (UTF-8), reads it with
        :func:`~pypalc.core.settings.load_setup_text` and applies it with
        :func:`~.setup_io.apply_setup`, which runs the calculation. An
        invalid file shows ``Error: setup file invalid at <key>: <msg>`` and
        changes nothing. A measured directivity given as a path is invalid
        here, because the GUI only receives the file's contents.

        Parameters
        ----------
        attr : str
            Changed attribute (value).
        old, new : str
            Old and new upload (base64 encoded).

        Returns
        -------
        None.
        """
        from .setup_io import apply_setup
        app = self.app
        try:
            try:
                text = base64.b64decode(new).decode('utf-8')
            except (binascii.Error, ValueError) as exc:
                raise SetupError('file', 'not a UTF-8 text file') from exc
            setup = load_setup_text(text)
            apply_setup(app, setup)
        except SetupError as exc:
            message = PATH_DIRECTIVITY_MESSAGE \
                if exc.key == PATH_DIRECTIVITY_KEY else exc.message
            logger.info('Setup file not loaded: %s', exc)
            logger.debug('Setup file not loaded', exc_info=True)
            app.config_error_info.text = \
                'Error: setup file invalid at %s: %s' % (exc.key, message)
        except CALLBACK_ERRORS:
            logger.exception('Could not load the setup file')
            app.config_error_info.text = 'Error: Could not load the setup file'
