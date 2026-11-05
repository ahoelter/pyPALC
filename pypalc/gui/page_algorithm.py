"""Page 4 of the GUI: algorithm options, calculation, files and the SPL
over distance.

The steps 18 to 25 are in the cards "PALC", "Weighting" and "Tolerance and
fixed angle"; the venue check, steps 26 and 27 and the error text of the
configuration pages in the card "Run"; the downloads and the setup file in
"Files" (:mod:`~.downloads`); the computed tilt angles in "Tilt angles".
The card "Results" shows the venue figure of the app
(:attr:`~.app.PalcApp.venue_figure`) and the SPL over distance with the
draggable target slope in tabs.

"26. PALC Run" (:attr:`AlgorithmPage.start_toggle`) and "Load Setup" run
the calculation in :meth:`~.app.PalcApp.later`, so that the browser first
shows the running look (:data:`RUNNING_LABEL`, dark red); while it runs,
another start is ignored.

author: Arne Hoelter
"""
from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING, Callable, Optional, Sequence

import numpy as np
import pandas as pd
import panel as pn
import panel_material_ui as pmui
from bokeh.models import ColumnDataSource, NumberFormatter, PointDrawTool
from bokeh.plotting import figure

from pypalc.core.errors import SetupError
from pypalc.core.palc import clamp_fixed_angle, fixed_angle_borders
from pypalc.core.settings import TargetPoint, load_setup_text
from pypalc.core.target_slope import hinge_indices, snap_to_distance_grid
from pypalc.gui import CALLBACK_ERRORS, skipped_while_applying_setup
from pypalc.gui.downloads import (DOWNLOAD_COLUMNS, DOWNLOAD_FILE_NAMES,
                                  RESULTS_FILE_NAME, SETUP_FILE_NAME,
                                  angles_csv, results_zip, setup_yaml)
from pypalc.gui.ranges import VENUE_MAX_HEIGHT
from pypalc.gui.style import RUNNING, SUCCESS, TEXT, TEXT_MUTED
from pypalc.gui.widgets import (GAP, button, caption, card, download,
                                page_row, select, slider, table, text_input)

if TYPE_CHECKING:
    from .app import PalcApp

logger = logging.getLogger(__name__)

#: Weightings available without target slope.
WEIGHTINGS_WITHOUT_TARGET_SLOPE = ['Without', 'Linear Spacing',
                                   'Logarithmic Spacing']
#: Key of the error of a setup file that gives the measured directivity as
#: a path, and the error text the GUI shows for it.
PATH_DIRECTIVITY_KEY = 'loudspeaker.measured_directivity.path'
PATH_DIRECTIVITY_MESSAGE = (
    "the GUI cannot read a measured directivity file given as a path. "
    "Upload the data on page 2 and use Save Setup (it embeds the data), or "
    "run the setup file with the headless runner (python -m pypalc run)")
#: Label and icon of "PALC Run", and while a calculation runs.
RUN_LABEL = '26. PALC Run'
RUN_ICON = 'play_arrow'
RUNNING_LABEL = 'Running...'
RUNNING_ICON = 'hourglass_top'
#: Label of the ν slider; ν is the slider value to the power of 3.
NU_LABEL = '22. Weighting parameter ν = %.2f'
#: Widths of the settings and the action column in px (they shrink on
#: narrow windows).
SETTINGS_WIDTH = 300
ACTIONS_WIDTH = 280
#: Hover color of the green "PALC Run" (Bootstrap 3's success hover).
SUCCESS_HOVER = '#449d44'


def run_toggle_sx(running: bool) -> dict:
    """Style of "PALC Run": a green button, dark red while running.

    While running, the toggle does not take clicks (a click would toggle
    its value off).

    Parameters
    ----------
    running : bool
        Whether a calculation runs.

    Returns
    -------
    dict
        The ``sx`` of the toggle; ``backgroundColor`` is its color.
    """
    color = RUNNING if running else SUCCESS
    hover = RUNNING if running else SUCCESS_HOVER
    sx = {'backgroundColor': color, 'color': TEXT, 'border': 'none',
          '&:hover, &.Mui-selected, &.Mui-selected:hover': {
              'backgroundColor': hover, 'color': TEXT}}
    if running:
        sx['pointerEvents'] = 'none'
        sx['&.Mui-disabled'] = {'backgroundColor': RUNNING, 'color': TEXT}
    else:
        sx['&.Mui-disabled'] = {
            'backgroundColor': 'action.disabledBackground',
            'color': 'action.disabled'}
    return sx


class AlgorithmPage:
    """Page 4: PALC variant, gap handling, weighting, tolerance, fixed
    angle, the start toggle, the downloads and setup files, the results
    table and the SPL over distance.

    Parameters
    ----------
    app : PalcApp
        The session the page belongs to.

    Attributes
    ----------
    running : bool
        True while a calculation started by "PALC Run" or "Load Setup"
        runs (from the click until it is done).
    """

    def __init__(self, app: PalcApp) -> None:
        self.app = app
        state = app.state
        self.running = False
        self._layout: Optional[pn.viewable.Viewable] = None
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
        self.results_table = table(
            self._angles_frame(),
            titles={'cabinet': 'LS', 'tilt_angle_deg': 'Total angle',
                    'inter_cabinet_angle_deg': 'ICA'},
            widths={'cabinet': 56, 'inter_cabinet_angle_deg': 64},
            formatters={'tilt_angle_deg': NumberFormatter(format='0.00'),
                        'inter_cabinet_angle_deg':
                            NumberFormatter(format='0.00')},
            configuration={'placeholder': 'Run PALC to see the angles'})
        """Table of the tilt angles (``results_source``)."""
        self._build_spl_over_distance_figure()
        self._venue_tab = pn.Column(sizing_mode='stretch_width',
                                    margin=(8, 0, 0, 0))
        self._spl_tab = pn.Column(sizing_mode='stretch_width',
                                  margin=(8, 0, 0, 0))
        self.tabs = pmui.Tabs(('Venue slice', self._venue_tab),
                              ('SPL over distance', self._spl_tab),
                              sizing_mode='stretch_width', margin=(0, 8))
        """Tabs of the venue figure and the SPL over distance (their
        figures are added with the layout, when the document has its
        theme)."""
        self._wire_callbacks()

    def _build_widgets(self) -> None:
        """Create the widgets of the page."""
        app = self.app
        self.palc_variant_select = select(
            '18. PALC algorithm', ['PALC 1', 'PALC 2', 'PALC 3'], 'PALC 2')
        self.gap_handling_select = select(
            '19. Gap handling', ['Without', 'Hard Margin', 'Soft Margin'],
            'Without')
        self.soft_margin_slider = slider(
            '20. Strength of soft margin', 1.0, 0.1, 1.9, 0.05,
            disabled=True)
        self.weighting_select = select(
            '21. Weighting', WEIGHTINGS_WITHOUT_TARGET_SLOPE
            + ['Target Slope'], 'Without')
        self.hinges_select = select('Hinges', ['0', '1', '2'], '0',
                                    visible=False, width=90,
                                    sizing_mode='fixed')
        """Number of hinges of the target slope; the choosable numbers
        depend on the number of audience lines."""
        # cubic spaced slider: the stored ν is the value to the power of 3
        self.weighting_nu_slider = slider(
            NU_LABEL % 1.0, np.cbrt(1.00), np.cbrt(0.01), np.cbrt(10.00),
            0.01, disabled=True, show_value=False)
        self.nu_minus_button = button('-', 'danger', disabled=True,
                                      variant='outlined', width=56,
                                      sizing_mode='fixed')
        self.nu_plus_button = button('+', 'success', disabled=True,
                                     variant='outlined', width=56,
                                     sizing_mode='fixed')
        self.nu_step_select = select(
            'Step', ['0.01', '0.05', '0.10', '0.50'], '0.05', disabled=True,
            width=100, sizing_mode='fixed')
        """Step size of the + and - buttons."""
        self.tolerance_slider = slider(
            '23. Tolerance (abort criterion) in m', 1.0, 0.05, 5, 0.05)
        self.use_fixed_angle_select = select(
            '24. Fix angle of uppermost cabinet?', ['No', 'Yes'], 'No',
            disabled=True)
        self.fixed_angle_input = text_input(
            '25. Angle of uppermost cabinet in deg', '0.0', disabled=True)
        self.fixed_angle_text = pn.pane.HTML(
            '', margin=(2, 8), styles={'color': TEXT_MUTED,
                                       'font-size': '12px'})
        """Allowed range of the fixed angle."""
        self.show_initial_fixed_angle_range()
        self.start_toggle = pmui.Toggle(
            label=RUN_LABEL, icon=RUN_ICON, color='success', disabled=True,
            value=False, sx=run_toggle_sx(False),
            sizing_mode='stretch_width', margin=(4, 8))
        """Starts the PALC computation (``value`` True); it reads
        :data:`RUNNING_LABEL` and is dark red while it runs."""
        self.clear_results_button = button(
            '27. Delete result visualization', 'danger', variant='outlined')
        self.frequency_range_slider = pmui.RangeSlider(
            label='Frequency range in Hz', start=0, end=20000,
            value=(200, 8000), step=20, sizing_mode='stretch_width',
            margin=(4, 16))
        self.download_angles_button = download(
            'Download PALC Angles', DOWNLOAD_FILE_NAMES['results'],
            lambda: io.StringIO(angles_csv(self)), app.later)
        """Downloads the tilt angles (:func:`~.downloads.angles_csv`)."""
        self.download_results_button = download(
            'Download Results', RESULTS_FILE_NAME,
            lambda: io.BytesIO(results_zip(self)), app.later)
        """Downloads the SPL over distance, target slope and homogeneity
        files in a zip file (:func:`~.downloads.results_zip`)."""
        self.save_setup_button = download(
            'Save Setup', SETUP_FILE_NAME, self._setup_file, app.later,
            icon='save')
        """Downloads the setup as a YAML file
        (:func:`~.downloads.setup_yaml`)."""
        self.load_setup_input = pmui.FileInput(
            label='Load Setup', uploaded_label='Load Setup',
            accept='.yaml,.yml', color='primary', variant='outlined',
            icon='upload_file', sizing_mode='stretch_width', margin=(4, 8))
        """Upload of a setup file; loading it restores the setup, runs the
        calculation and shows page 5."""

    def _setup_file(self) -> Optional[io.StringIO]:
        """The file of "Save Setup", None if the setup cannot be saved."""
        text = setup_yaml(self)
        return None if text is None else io.StringIO(text)

    def _angles_frame(self) -> pd.DataFrame:
        """The tilt angles as shown in the table."""
        data = self.results_source.data
        frame = pd.DataFrame({column: np.asarray(data[column], dtype=float)
                              for column in DOWNLOAD_COLUMNS['results']})
        frame['cabinet'] = frame['cabinet'].astype(int)
        return frame

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
            y_range=(-15, 0.3), sizing_mode='stretch_width',
            height=VENUE_MAX_HEIGHT)
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
        target_points = spl_figure.scatter('distance_m', 'level_db',
                                           source=self.target_source,
                                           line_color="yellow",
                                           fill_color="black", line_width=2,
                                           name="Reference",
                                           legend_label='Target Slope')
        reference_point = spl_figure.scatter(
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
        spl_figure.output_backend = "svg"
        self.spl_over_distance_figure = spl_figure

    def _wire_callbacks(self) -> None:
        """Connect the widgets with their callbacks."""
        app = self.app

        def watch(widget, callback: Callable) -> None:
            """Call ``callback(attr, old, new)`` on changes of the value."""
            widget.param.watch(lambda event: callback(
                'value', event.old, event.new), 'value')
        for widget in (self.palc_variant_select, self.gap_handling_select,
                       self.soft_margin_slider, self.tolerance_slider,
                       self.fixed_angle_input, self.frequency_range_slider):
            watch(widget, app.store_value)
        watch(self.weighting_select, self.choose_weighting)
        watch(self.hinges_select, self.choose_hinges)
        watch(self.weighting_nu_slider, self.store_nu)
        self.weighting_nu_slider.param.watch(
            lambda event: self.show_nu(), 'value')
        self.nu_plus_button.on_click(lambda event: self.increase_nu())
        self.nu_minus_button.on_click(lambda event: self.decrease_nu())
        watch(self.nu_step_select, self.set_nu_step)
        watch(self.use_fixed_angle_select, self.choose_fixed_angle)
        self.start_toggle.param.watch(
            lambda event: self.start_toggled(event.new), 'value')
        self.clear_results_button.on_click(
            lambda event: app.clear_result_figure())
        self.target_source.on_change('data', self.snap_target)
        self.target_reference_source.on_change('data', self.snap_target)
        self.results_source.on_change(
            'data', lambda attr, old, new: setattr(
                self.results_table, 'value', self._angles_frame()))
        self.load_setup_input.param.watch(
            lambda event: self.load_selected(event.new), 'value')

    def layout(self) -> pn.FlexBox:
        """Body of the page below the header.

        Built at the first call, then the same object: its widgets show
        every change in place.

        Returns
        -------
        panel.FlexBox
            The settings column (cards "PALC", "Weighting", "Tolerance and
            fixed angle") and the action column (cards "Run", "Files",
            "Tilt angles") beside (or, on narrow windows, above) the card
            "Results".
        """
        if self._layout is None:
            app = self.app
            self._venue_tab.objects = [
                app.venue_axes_row(),
                app.bokeh_pane(app.venue_figure,
                               sizing_mode='stretch_width')]
            self._spl_tab.objects = [
                self.frequency_range_slider,
                app.bokeh_pane(self.spl_over_distance_figure,
                               sizing_mode='stretch_width')]
            palc_card = card('PALC', self.palc_variant_select,
                             self.gap_handling_select,
                             self.soft_margin_slider)
            weighting_card = card(
                'Weighting',
                pn.Row(self.weighting_select, self.hinges_select,
                       sizing_mode='stretch_width', margin=0),
                self.weighting_nu_slider,
                pn.Row(self.nu_minus_button, self.nu_plus_button,
                       self.nu_step_select, align='center', margin=0))
            limits_card = card(
                'Tolerance and fixed angle', self.tolerance_slider,
                self.use_fixed_angle_select, self.fixed_angle_input,
                pn.Row(caption('Allowed range in deg:', margin=(2, 0, 2, 8),
                               sizing_mode=None),
                       self.fixed_angle_text, margin=0))
            run_card = card('Run', app.pages[1].check_venue_button,
                            self.start_toggle, self.clear_results_button,
                            app.config_error_info)
            files_card = card('Files', self.download_angles_button,
                              self.download_results_button,
                              self.save_setup_button, self.load_setup_input)
            table_card = card('Tilt angles', self.results_table)
            settings = pn.Column(palc_card, weighting_card, limits_card,
                                 margin=0, sizing_mode='stretch_width',
                                 styles={'flex': f'1 1 {SETTINGS_WIDTH}px',
                                         'min-width': '0'})
            actions = pn.Column(run_card, files_card, table_card, margin=0,
                                sizing_mode='stretch_width',
                                styles={'flex': f'1 1 {ACTIONS_WIDTH}px',
                                        'min-width': '0'})
            side = pn.FlexBox(settings, actions, flex_wrap='wrap',
                              gap=f'{GAP}px', align_items='flex-start',
                              margin=0)
            self._layout = page_row(
                side,
                # the tabs name the figures: no header, so that the figures
                # end inside the window like those of pages 1 and 3
                card('Results', self.tabs, hide_header=True),
                side_width=SETTINGS_WIDTH + GAP + ACTIONS_WIDTH)
        return self._layout

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------

    def set_running(self, running: bool) -> None:
        """Show whether a calculation runs.

        Running: "PALC Run" reads :data:`RUNNING_LABEL` and is dark red.
        Done: it reads :data:`RUN_LABEL`, is green and released
        (``value`` False).

        Parameters
        ----------
        running : bool
            Whether a calculation runs.

        Returns
        -------
        None.
        """
        self.running = running
        if running:
            self.start_toggle.param.update(label=RUNNING_LABEL,
                                           icon=RUNNING_ICON,
                                           sx=run_toggle_sx(True))
        else:
            self.start_toggle.param.update(value=False, label=RUN_LABEL,
                                           icon=RUN_ICON,
                                           sx=run_toggle_sx(False))

    def run_later(self, func: Callable[[], None]) -> bool:
        """Show the running look and run a calculation after the browser
        shows it (:meth:`~.app.PalcApp.later`); the look is reset when it
        is done, also after an error.

        Parameters
        ----------
        func : callable
            The calculation, without arguments.

        Returns
        -------
        bool
            False if a calculation runs already (``func`` is not run).
        """
        if self.running:
            return False
        self.set_running(True)

        def run() -> None:
            try:
                func()
            finally:
                self.set_running(False)
        self.app.later(run)
        return True

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
        self.fixed_angle_text.object = \
            'Max: ' + str(np.round(fixed_angle[0], decimals=2)) + \
            ', Min: ' + str(np.round(fixed_angle[2], decimals=2))
        return fixed_angle

    def show_initial_fixed_angle_range(self) -> None:
        """Show the range of the fixed angle as in a new session."""
        fixed_angle = self.app.state.optimized_array.fixed_angle
        self.fixed_angle_text.object = 'Max: ' + str(fixed_angle[0]) + \
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
        """Run the calculation when the start toggle is pressed
        (:meth:`run_later`); ignored while a calculation runs.

        Parameters
        ----------
        active : bool
            New state of the toggle (its ``value``).

        Returns
        -------
        None.
        """
        if active:
            self.run_later(self.app.run_calculation)

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
        else:
            self.hinges_select.visible = False
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

    def show_nu(self) -> None:
        """Show ν (the slider value to the power of 3) in the label of the
        slider."""
        self.weighting_nu_slider.label = NU_LABEL % (
            self.weighting_nu_slider.value**3)

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

    def load_selected(self, data: Optional[bytes]) -> None:
        """Load a setup file selected in "Load Setup" (:meth:`load_setup`)
        with the running look (:meth:`run_later`); ignored while a
        calculation runs.

        Afterwards the input is emptied, so that the same file selected
        again is loaded again (the browser sends it again).

        Parameters
        ----------
        data : bytes or None
            Content of the file.

        Returns
        -------
        None.
        """
        def load() -> None:
            try:
                self.load_setup('value', None, data)
            finally:
                self.load_setup_input.value = None
        if data and not self.run_later(load):
            self.load_setup_input.value = None

    def load_setup(self, attr, old, new) -> None:
        """Load a setup file ("Load Setup").

        Decodes the file (UTF-8), reads it with
        :func:`~pypalc.core.settings.load_setup_text` and applies it with
        :func:`~.setup_io.apply_setup`, which runs the calculation and
        shows page 5. An invalid file shows ``Error: setup file invalid at
        <key>: <msg>`` and changes nothing. A measured directivity given as
        a path is invalid here, because the GUI only receives the file's
        contents.

        Parameters
        ----------
        attr : str
            Changed attribute (value).
        old, new : bytes or None
            Old and new content of the uploaded file.

        Returns
        -------
        None.
        """
        from .setup_io import apply_setup
        app = self.app
        try:
            try:
                text = new.decode('utf-8')
            except ValueError as exc:  # UnicodeDecodeError
                raise SetupError('file', 'not a UTF-8 text file') from exc
            setup = load_setup_text(text)
            apply_setup(app, setup)
        except SetupError as exc:
            message = PATH_DIRECTIVITY_MESSAGE \
                if exc.key == PATH_DIRECTIVITY_KEY else exc.message
            logger.info('Setup file not loaded: %s', exc)
            logger.debug('Setup file not loaded', exc_info=True)
            app.config_error_info.object = \
                'Error: setup file invalid at %s: %s' % (exc.key, message)
        except CALLBACK_ERRORS:
            logger.exception('Could not load the setup file')
            app.config_error_info.object = \
                'Error: Could not load the setup file'
