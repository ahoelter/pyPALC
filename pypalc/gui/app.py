"""The pyPALC GUI of one browser session.

``main.py`` builds one :class:`PalcApp` per session::

    PalcApp(curdoc()).show()

The app holds everything that belongs to a session: the calculation objects
(:class:`SessionState`), the ColumnDataSources, widgets and figures, and
the callbacks. Every page of the GUI is a class in its own module that
creates its widgets, figures and sources and wires their callbacks:

* page 0, help: :class:`~.help_text.HelpPage`
* page 1, venue: :class:`~.page_venue.VenuePage`
* page 2, loudspeaker: :class:`~.page_loudspeaker.LoudspeakerPage`
* page 3, array: :class:`~.page_array.ArrayPage`
* page 4, algorithm: :class:`~.page_algorithm.AlgorithmPage`
* page 5, results: :class:`~.page_results.ResultsPage`

Widgets shown on several pages (header, page selection, venue axes
scaling, the configuration error text and the venue figure) and the
callbacks that read widgets of several pages live on :class:`PalcApp`.
No module of this package holds per-session state.

A calculation reads the entered setup (:func:`~.setup_io.setup_from_app`)
and computes it with :func:`pypalc.core.pipeline.run`, like the headless
runner.

author: Arne Hoelter
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Union

import numpy as np
from bokeh.document import Document
from bokeh.layouts import column, gridplot, row
from bokeh.models import (ColumnDataSource, Div, Model, Paragraph,
                          RadioButtonGroup, RadioGroup)
from bokeh.plotting import figure
from bokeh.transform import linear_cmap
from colorcet import rainbow

from pypalc.core import pipeline
from pypalc.core.errors import PalcError, SetupError
from pypalc.core.palc import ArrayGeometry, PalcConfig
from pypalc.core.pipeline import RunResult, VenueResult
from pypalc.core.settings import AudienceLine, Setup
from pypalc.core.sound_field import SoundFieldConfig
from pypalc.core.target_slope import TargetSlopeState
from pypalc.core.venue import (AudienceLines, PalcVenue,
                               fill_palc_venue, suggest_venue)
from pypalc.gui import (APP_NAME, CALLBACK_ERRORS,
                        skipped_while_applying_setup)
from pypalc.gui.help_text import HelpPage
from pypalc.gui.page_algorithm import AlgorithmPage
from pypalc.gui.page_array import ArrayPage
from pypalc.gui.page_loudspeaker import LoudspeakerPage
from pypalc.gui.page_results import ResultsPage
from pypalc.gui.page_venue import VenuePage
from pypalc.gui.setup_io import setup_from_app
from pypalc.gui.ranges import (PlotRanges, scale_venue_figure, set_figure_ranges,
                               show_rays)

logger = logging.getLogger(__name__)


Page = Union[HelpPage, VenuePage, LoudspeakerPage, ArrayPage, AlgorithmPage,
             ResultsPage]


def _initial_drawn_lines() -> AudienceLines:
    """Audience lines of a new session (no lines drawn)."""
    return AudienceLines(line_end_indices=np.array([0]).astype(int))


@dataclass
class SessionState:
    """Calculation objects of one GUI session.

    Attributes
    ----------
    drawn_lines : AudienceLines
        Audience lines drawn by the user.
    audience_lines : list of AudienceLine
        The drawn audience lines as settings, in the order of ``drawn``,
        with the discretization each line was saved with.
    venue : VenueResult or None
        Venue of the last successful "Get PALC Venue Proposal", None before.
        Like in the legacy GUI it is kept when lines change afterwards; the
        start toggle is disabled until the venue is checked again.
    measured_directivity_csv : str or None
        Text of the last readable upload of measured directivity data.
    config : PalcConfig
        PALC configuration, holding the values of the configuration
        widgets. Calculations do not change it.
    sound_field_config : SoundFieldConfig
        Sound field prediction configuration, used for the directivity
        and beam plots.
    ranges : PlotRanges
        Ranges of the figures.
    optimized_array : ArrayGeometry
        Holds the allowed range of the fixed angle of the uppermost cabinet
        (attribute ``fixed_angle``).
    reference_array : ArrayGeometry
        The reference array as computed on page 3.
    target : TargetSlopeState
        The target slope shown on page 4.
    result : RunResult or None
        Results of the last successful calculation, None before.
    target_of_result : tuple or None
        The target slope as shown right after the last successful
        calculation: ``(distances, levels, reference distance)`` of the
        target sources. Save compares it with the shown target to find out
        whether the target was moved since
        (:func:`~.setup_io.setup_to_save`).
    applying_setup : bool
        True while :func:`~.setup_io.apply_setup` sets the widgets; the
        widget callbacks then do nothing
        (:func:`~pypalc.gui.skipped_while_applying_setup`).
    """
    drawn_lines: AudienceLines = field(default_factory=_initial_drawn_lines)
    audience_lines: List[AudienceLine] = field(default_factory=list)
    venue: Optional[VenueResult] = None
    measured_directivity_csv: Optional[str] = None
    config: PalcConfig = field(default_factory=PalcConfig)
    sound_field_config: SoundFieldConfig = field(
        default_factory=SoundFieldConfig)
    ranges: PlotRanges = field(default_factory=PlotRanges)
    optimized_array: ArrayGeometry = field(default_factory=ArrayGeometry)
    reference_array: ArrayGeometry = field(
        default_factory=lambda: ArrayGeometry(tilt_angles_deg=np.zeros(4)))
    target: TargetSlopeState = field(default_factory=TargetSlopeState)
    result: Optional[RunResult] = None
    target_of_result: Optional[tuple] = None
    applying_setup: bool = False


def _copied(value):
    """A copy of a property value that may be changed in place (the dicts,
    lists and arrays of source data); other values (models, numbers) as
    they are. Bokeh's wrappers become plain dicts and lists."""
    if isinstance(value, dict):
        return {key: _copied(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copied(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copied(item) for item in value)
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def _error_text(exc: PalcError) -> str:
    """Error text shown for a failed calculation."""
    if isinstance(exc, SetupError):
        return 'Error: %s' % exc
    return str(exc)


def _logo_figure(width: int, image: str, x_offset: float):
    """Figure that shows a logo from the app's static directory."""
    x_range, y_range = (10, 30), (10, 30)
    logo = figure(x_range=x_range, y_range=y_range, width=width, height=80,
                  min_border=0)
    logo.toolbar.logo, logo.axis.visible = 'grey', False
    logo.outline_line_color, logo.grid.visible = None, False
    logo.toolbar_location, logo.toolbar.logo = None, 'grey'
    logo.image_url(url=[f"{APP_NAME}/static/{image}"],
                   x=x_range[0] + x_offset, y=y_range[1] - 0.3,
                   w=x_range[1] - (x_range[0] + 1),
                   h=y_range[1] - (y_range[0] + 0.6))
    return logo


class PalcApp:
    """The GUI of one browser session.

    Parameters
    ----------
    doc : Document, optional
        The Bokeh document of the session (``curdoc()`` in ``main.py``); a
        new document if None.

    Attributes
    ----------
    doc : Document
        The document the pages are shown in.
    state : SessionState
        Calculation objects of the session.
    pages : dict of int to page
        The pages, keyed by their index in the page selection.
    """

    def __init__(self, doc: Optional[Document] = None) -> None:
        self.doc = doc if doc is not None else Document()
        self.state = SessionState()
        self._build_shared_widgets()
        # sources of the venue figure that are filled by the calculation
        self.cabinet_patches_source = ColumnDataSource(data=dict(
            patches_x=[], patches_y=[], reference_patches_x=[],
            reference_patches_y=[]))
        """Outlines of the cabinets of the optimized and reference array."""
        self.hit_points_source = ColumnDataSource(data=dict(
            hit_points_x=[], hit_points_y=[], hit_points_start_x=[],
            hit_points_start_y=[], hit_points_stop_x=[], hit_points_stop_y=[]))
        """Points where the rays of the cabinets hit the venue."""
        self.rays_source = ColumnDataSource(data=dict(
            x_start_stop=[], y_start_stop=[], x_fin_unitn_start_stop=[],
            y_fin_unitn_start_stop=[], x_start_unitn_start=[],
            y_start_unitn_start=[], x_stop_unitn_stop=[],
            y_stop_unitn_stop=[], center_rays_x=[], center_rays_y=[],
            start_rays_x=[], start_rays_y=[], stop_rays_x=[],
            stop_rays_y=[]))
        """Rays from the cabinets to the venue."""
        self.venue_figure = figure(
            title="", x_axis_label="x in m", y_axis_label="y in m",
            toolbar_location="right",
            tools="box_zoom,pan,wheel_zoom,reset,tap,hover,save",
            frame_height=478, frame_width=637)
        """Venue slice figure (pages 1 to 4); its glyphs are added by the
        venue page."""
        self.venue_figure.toolbar.logo = None
        # Pages use widgets and sources of pages built before them; the
        # results page comes before the algorithm page, whose download
        # button needs the homogeneity source.
        self.pages: Dict[int, Page] = {}
        for index, page_class in ((0, HelpPage), (1, VenuePage),
                                  (2, LoudspeakerPage), (3, ArrayPage),
                                  (5, ResultsPage), (4, AlgorithmPage)):
            self.pages[index] = page_class(self)
        self.pages = {key: self.pages[key] for key in sorted(self.pages)}
        self.page_select.on_change('active',
                                   lambda attr, old, new: self.show())
        self.venue_axes_select.on_change(
            'active', lambda attr, old, new: self.scale_venue_axes())
        # values the legacy GUI stores at start-up
        self.state.config.store_config('aperture_angles',
                                       self.pages[2].aperture_input.value)
        self.state.config.store_config('cabinet_count',
                                       self.pages[3].cabinet_count_input.value)
        # what the session shows before the first result (clear_results)
        self._new_session_results = [
            (model, name, _copied(getattr(model, name)))
            for model, name in self._result_properties()]

    def _build_shared_widgets(self) -> None:
        """Header, page selection and the widgets shown on several pages."""
        self.ak_logo = _logo_figure(80, "logo_grau-schwarz.png", 0)
        self.tu_logo = _logo_figure(int(80*1.786), "TUB.png", 0.5)
        self.ak_text = Div(text='<b>A Software Tool by the <br><big><a href="https://www.ak.tu-berlin.de" target="_blank">Audio Communication Group</a></big></b>', width=270, height=60)
        self.page_select = RadioButtonGroup(
            labels=["Help", "1. PAL", "2. Loudspeaker Configuration",
                    "3. Array Configuration", "4. Algorithm",
                    "5. Sound Field Prediction"],
            active=1, button_type="primary", width=680)
        """Switches between the pages."""
        self.headline = Paragraph(text="_____________________________________________________________________________________________________________________________________________________________________________")
        self.venue_axes_select = RadioGroup(
            labels=['Equal (variable figure height)',
                    'Equal (const. figure height)', 'Stretched'],
            active=2, height=15, margin=5, inline=True)
        """Scaling of the venue figure."""
        self.venue_axes_label = Div(text='      <b> Venue Slice Axes: </b>',
                                    height=15, width=150)
        self.config_error_info = Paragraph(text="", width=250, height=50)
        """Error text of the configuration pages 2 to 4."""

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def venue_axes_row(self):
        """Row with the scaling selection of the venue figure."""
        return row([self.venue_axes_label, self.venue_axes_select])

    def show(self) -> None:
        """Show the page selected in the header (legacy ``steps``).

        Clears the document and adds the header and the layout of the
        selected page. Showing page 1 clears the result visualization in
        the venue figure.

        Returns
        -------
        None.
        """
        self.doc.clear()
        active = self.page_select.active
        if active == 1:
            self.clear_result_figure()
        header = row([column(self.ak_logo, width=100),
                      column(self.tu_logo, width=170),
                      column(self.ak_text, width=270),
                      column(self.page_select, width=690)])
        headline = row([self.headline], width=1230)
        body = self.pages[active].layout()
        self.doc.add_root(gridplot([[header], [headline], [body]],
                                   toolbar_location='right',
                                   merge_tools=False))

    # ------------------------------------------------------------------
    # Venue
    # ------------------------------------------------------------------

    def has_venue(self) -> bool:
        """True if a PALC compatible venue was created."""
        venue = self.state.venue
        return venue is not None and len(venue.palc_venue.segments) > 0

    def check_venue(self) -> None:
        """Create the PALC compatible venue (legacy ``check_PAL``).

        Callback of "Get PALC Venue Proposal". Shows the suggested
        non-audience lines and the lines only used in the sound field
        prediction, enables the start toggle, resets a selected target
        slope weighting and sets the choosable number of hinges.

        Returns
        -------
        None.
        """
        st = self.state
        venue_page, algorithm = self.pages[1], self.pages[4]
        try:
            non_audience_lines = AudienceLines(cut_lines=[])
            sound_field_only_lines = AudienceLines(cut_lines=[])
            suggest_venue(st.drawn_lines, non_audience_lines,
                          sound_field_only_lines)
            venue_page.non_audience_source.data = dict(
                lines_x=non_audience_lines.plot_x,
                lines_y=non_audience_lines.plot_y)
            venue_page.sound_field_only_source.data = dict(
                lines_x=sound_field_only_lines.plot_x,
                lines_y=sound_field_only_lines.plot_y)
            st.venue = None
            palc_venue = PalcVenue(segments=[], audience_segments=[])
            fill_palc_venue(palc_venue, st.drawn_lines, non_audience_lines,
                            sound_field_only_lines)
            palc_venue.collect_audience_segments(non_audience_lines)
            st.venue = VenueResult(drawn_lines=st.drawn_lines, palc_venue=palc_venue,
                                   non_audience_lines=non_audience_lines,
                                   sound_field_only_lines=sound_field_only_lines)
            self.set_venue_checked(True)
            self.config_error_info.text = ''
            self.show()
            if algorithm.weighting_select.value in ['Target Slope']:
                algorithm.weighting_select.value = 'Without'
                st.target.initialized = False
                algorithm.choose_weighting(0, 0, 0)
            weight_links = np.arange(np.shape(st.drawn_lines.line_start_x)[0]+1)
            algorithm.hinges_select.options = [str(n) for n in
                                               list(weight_links)]
            algorithm.hinges_select.value = '0'
        except CALLBACK_ERRORS:
            logger.exception('Could not create the PALC venue')
            self.config_error_info.text = 'Error: Could not create PAL proposal'

    def set_venue_checked(self, checked: bool) -> None:
        """Enable the start toggle if the venue is PALC compatible
        (legacy ``change_start_pal_button``).

        Parameters
        ----------
        checked : bool
            If True and a venue exists, PALC computation is enabled,
            otherwise disabled.

        Returns
        -------
        None.
        """
        venue_page, algorithm = self.pages[1], self.pages[4]
        if checked and self.has_venue():
            algorithm.start_toggle.disabled = False
            venue_page.check_venue_button.button_type = "success"
            venue_page.check_venue_button.label = "8. Venue is PALC compatible!"
            self.venue_figure.legend.visible = False
            algorithm.use_fixed_angle_select.disabled = False
        else:
            algorithm.start_toggle.disabled = True
            venue_page.check_venue_button.button_type = "warning"
            venue_page.check_venue_button.label = "8. Get PALC Venue Proposal"
            self.venue_figure.legend.visible = False
            algorithm.use_fixed_angle_select.disabled = True
            algorithm.use_fixed_angle_select.value = 'No'
            algorithm.fixed_angle_input.disabled = True
            self.state.config.use_fixed_angle = False

    # ------------------------------------------------------------------
    # Ranges
    # ------------------------------------------------------------------

    def update_venue_ranges(self) -> None:
        """Fit the ranges of the venue figures to the drawn lines.

        Updates the venue figure, the beam plots, the venue points figure
        and the x-range of the SPL over distance figure.

        Returns
        -------
        None.
        """
        st = self.state
        results = self.pages[5]
        x_range = st.ranges.update_range('venue_x', [-2, 2], st.drawn_lines.line_start_x,
                                         st.drawn_lines.line_stop_x, st.config.top_x)
        y_range = st.ranges.update_range('venue_y', [-2, 2], st.drawn_lines.line_start_y,
                                         st.drawn_lines.line_stop_y, st.config.top_y)
        self.scale_venue_axes(call=False)
        set_figure_ranges(st.ranges, [results.beam_figure,
                                      results.reference_beam_figure,
                                      results.venue_points_figure],
                          [True, True, False], x_range, y_range)
        self.pages[4].spl_over_distance_figure.x_range = x_range

    def scale_venue_axes(self, call: bool = True) -> None:
        """Scale the venue figure as selected (legacy ``scale_p_axis``).

        Parameters
        ----------
        call : bool, optional
            If True, shows the page again. The default is True.

        Returns
        -------
        None.
        """
        scale_venue_figure(self.venue_figure, self.state.ranges,
                       self.venue_axes_select)
        if call:
            self.show()

    def _result_properties(self) -> list:
        """(model, property) pairs that showing a result writes
        (:meth:`_show_result`)."""
        algorithm, results = self.pages[4], self.pages[5]
        data_sources = (
            self.cabinet_patches_source, self.hit_points_source,
            self.rays_source, algorithm.results_source,
            algorithm.spl_over_distance_source,
            algorithm.reference_spl_over_distance_source,
            algorithm.target_source, algorithm.target_reference_source,
            algorithm.target_region_source, results.venue_points_source,
            results.sound_field_source, results.homogeneity_source,
            results.histogram_source, results.reference_histogram_source,
            results.beam_source)
        ranges = (
            (algorithm.spl_over_distance_figure, 'y_range'),
            (results.histogram_figure, 'x_range'),
            (results.histogram_figure, 'y_range'),
            (results.reference_histogram_figure, 'x_range'),
            (results.reference_histogram_figure, 'y_range'),
            (results.homogeneity_figure, 'y_range'),
            (results.sound_field_figure, 'y_range'),
            (results.reference_sound_field_figure, 'y_range'))
        return ([(results.beam_checkbox, 'active'),
                 (results.beam_checkbox, 'disabled'),
                 (results.beam_figure, 'visible'),
                 (results.reference_beam_figure, 'visible')]
                + [(source, 'data') for source in data_sources]
                + list(ranges))

    def clear_results(self) -> None:
        """Show no results, as a new session does.

        Sets every source, figure range and widget that showing a result
        writes (:meth:`_show_result`) to its value in a new session; the
        target slope sources are reset without snapping them to the
        cleared distance grid. Used by :func:`~.setup_io.apply_setup` when
        a loaded setup cannot be calculated, so that no results of the
        previous setup stay visible or can be downloaded. The error text
        and ``state`` are not changed.

        Returns
        -------
        None.
        """
        algorithm = self.pages[4]
        targets = (algorithm.target_source,
                   algorithm.target_reference_source)
        for source in targets:
            source.remove_on_change('data', algorithm.snap_target)
        try:
            for model, name, value in self._new_session_results:
                setattr(model, name, _copied(value))
        finally:
            for source in targets:
                source.on_change('data', algorithm.snap_target)

    def clear_result_figure(self) -> None:
        """Remove the array and its rays from the venue figure (legacy
        ``clear_result_fig``)."""
        show_rays(self.rays_source, self.hit_points_source, [], [])

    # ------------------------------------------------------------------
    # Values of the configuration widgets
    # ------------------------------------------------------------------

    def _config_widgets(self) -> Dict[str, Model]:
        """Configuration key -> widget whose value is stored under it."""
        loudspeaker, array, algorithm = (self.pages[2], self.pages[3],
                                         self.pages[4])
        return {'palc_variant': algorithm.palc_variant_select,
                'use_discrete_angles': array.use_discrete_angles_select,
                'gap_handling': algorithm.gap_handling_select,
                'soft_margin_strength': algorithm.soft_margin_slider,
                'cabinet_height': loudspeaker.cabinet_height_input,
                'cabinet_gap': array.cabinet_gap_input,
                'directivity': loudspeaker.directivity_select,
                'fixed_angle': algorithm.fixed_angle_input,
                'top_x': array.top_x_input,
                'top_y': array.top_y_input,
                'cabinet_count': array.cabinet_count_input,
                'aperture_angles': loudspeaker.aperture_input,
                'tolerance': algorithm.tolerance_slider,
                'frequency_range_hz': algorithm.frequency_range_slider}

    def _store_config_value(self, key: str, val, attr: str = 'value'
                            ) -> None:
        """Store one configuration value as entered in its widget.

        Wrong input is replaced by the default value in the widget; a
        mounting height that is too low for the array is raised and shown.
        Values the sound field configuration also has are stored there too.
        """
        st = self.state
        loudspeaker, array, algorithm = (self.pages[2], self.pages[3],
                                         self.pages[4])
        defaults = {'cabinet_height': (loudspeaker.cabinet_height_input, '0.2'),
                    'cabinet_count': (array.cabinet_count_input, '4'),
                    'cabinet_gap': (array.cabinet_gap_input, '0'),
                    'tolerance': (algorithm.tolerance_slider, '0.1'),
                    'fixed_angle': (algorithm.fixed_angle_input, ''),
                    'top_x': (array.top_x_input, '0'),
                    'top_y': (array.top_y_input, '2'),
                    'aperture_angles': (loudspeaker.aperture_input, '3.0')}
        error_info, failed_key = st.config.store_config(key, val)
        # if wrong user input --> insert default value
        if failed_key in defaults:
            widget, default = defaults[failed_key]
            setattr(widget, attr, default)
        if failed_key in ['too low']:
            setattr(array.top_y_input, attr, str(st.config.top_y))
        # store the value in the sound field configuration too if it has it
        if hasattr(st.sound_field_config, key) and \
                failed_key in [None, 'too low']:
            st.sound_field_config.store_config(key, val)
        self.config_error_info.text = str(error_info)

    def _config_values_stored(self) -> None:
        """Arrange the pages after configuration values were stored.

        Refreshes the range of a fixed angle of the uppermost cabinet (it
        depends on the position and height of the array) and clamps the
        angle to it, arranges the gap handling and weighting widgets,
        updates the directivity plot and the ranges of the venue figures.
        """
        st = self.state
        loudspeaker, algorithm = self.pages[2], self.pages[4]
        if st.config.use_fixed_angle and self.has_venue():
            algorithm.update_fixed_angle_range()
        algorithm.round_fixed_angle()
        algorithm.arrange_gap_handling_widgets()
        algorithm.exclude_hard_margin_target_slope()
        loudspeaker.update_directivity_plot()
        # more than one venue point (as the legacy GUI tested it)
        if np.size(st.drawn_lines.points_x) > 1:
            self.update_venue_ranges()

    @skipped_while_applying_setup
    def store_value(self, attr, old, new) -> None:
        """Store the value of a configuration widget (legacy ``get_value``).

        Stores every configuration value whose widget shows ``new`` in the
        PALC configuration (and the sound field configuration). Wrong input
        is replaced by the default value. Afterwards refreshes the range of
        a fixed angle and clamps the angle to it, arranges the gap handling
        and weighting widgets, updates the directivity plot and the ranges
        of the venue figures.

        Triggered by the aperture angle, cabinet height, directivity,
        cabinet count, position, gap, PALC variant, gap handling, soft
        margin strength, tolerance, fixed angle and frequency range widgets.

        Parameters
        ----------
        attr : str
            Changed attribute (value).
        old, new : str, int, float or tuple
            Old and new value of the widget.

        Returns
        -------
        None.
        """
        values = {key: widget.value
                  for key, widget in self._config_widgets().items()}
        # iterate over the values to detect changed value and store it
        for key, val in values.items():
            if new == val:
                self._store_config_value(key, val, attr)
                if key in ['cabinet_count', 'use_discrete_angles']:
                    self.pages[3].update_reference_angles(
                        cabinet_count_changed=True)
        self._config_values_stored()

    def store_config_widgets(self) -> None:
        """Store the values of all configuration widgets.

        Used by :func:`~.setup_io.apply_setup` after it has set the
        widgets. Stores like :meth:`store_value`, but every value, the
        mounting height last (so that only the final values decide whether
        it is raised, as in the headless runner). The discrete tilt angles
        and the fixed angle are stored by their own steps.

        Returns
        -------
        None.
        """
        widgets = self._config_widgets()
        keys = [key for key in widgets
                if key not in ('use_discrete_angles', 'fixed_angle', 'top_y')]
        for key in keys + ['top_y']:
            self._store_config_value(key, widgets[key].value)
        self._config_values_stored()

    # ------------------------------------------------------------------
    # Calculation
    # ------------------------------------------------------------------

    def _calculation_setup(self) -> Setup:
        """The setup of the next calculation.

        While "Target Slope" is selected but no target slope is shown yet,
        this is the initial run without weighting, whose SPL over distance
        the target slope is placed on.
        """
        setup = setup_from_app(self)
        if setup.algorithm.weighting == 'target_slope' and \
                not self.state.target.initialized:
            setup = replace(setup, target_slope=None, algorithm=replace(
                setup.algorithm, weighting='without'))
        return setup

    def run_calculation(self) -> bool:
        """Run PALC and the sound field prediction (legacy ``start_calc``).

        Callback body of the start toggle; also runs when "Target Slope" is
        selected (the initial run). Computes :func:`setup_from_app` with
        :func:`pypalc.core.pipeline.run` on the checked venue, shows the
        result and then stores it in ``state.result``.

        If the calculation or showing its result fails, the error text is
        shown and ``state.result`` keeps the previous result (the sources
        keep it too if the calculation failed). The start toggle is
        released in every case; unexpected exceptions are logged and raised
        again.

        Returns
        -------
        bool
            True if the result was computed and shown.
        """
        st = self.state
        try:
            if not self.has_venue():
                self.config_error_info.text = 'Error: Please CREATE and GET PALC compatible venue proposal on page 1.'
                return False
            try:
                result = pipeline.run(self._calculation_setup(),
                                      venue=st.venue)
            except PalcError as exc:
                logger.warning('PALC calculation failed: %s', exc,
                               exc_info=True)
                self.config_error_info.text = _error_text(exc)
                return False
            except CALLBACK_ERRORS:
                logger.exception('PALC calculation failed')
                self.config_error_info.text = 'Error: PALC calculation failed'
                return False
            try:
                self._show_result(result)
            except CALLBACK_ERRORS:
                logger.exception('Could not store the output')
                self.config_error_info.text = 'Error: Could not store the output'
                return False
            st.result = result
            st.target_of_result = self.pages[4].shown_target()
            self.pages[4].start_toggle.active = False
            self.show()
            return True
        except Exception:
            logger.exception('PALC calculation failed unexpectedly')
            raise
        finally:
            self.pages[4].start_toggle.active = False

    def _show_result(self, result: RunResult) -> None:
        """Copy the results of a calculation into the sources and figures.

        Parameters
        ----------
        result : RunResult
            The results to show.

        Returns
        -------
        None.
        """
        st = self.state
        algorithm, results = self.pages[4], self.pages[5]
        cabinet_count = result.setup.array.cabinet_count
        optimized, reference = result.optimized, result.reference
        evaluation = optimized.evaluation
        evaluation_reference = reference.evaluation
        spl_over_distance = optimized.spl_over_distance
        reference_spl_over_distance = reference.spl_over_distance
        drawn_lines = result.venue.drawn_lines
        target = st.target
        self.config_error_info.text = ''
        # array and rays in the venue figure
        show_rays(self.rays_source, self.hit_points_source, result.ray_plot,
                  result.geometry)
        # tilt angles
        algorithm.results_source.data = dict(
            cabinet=np.linspace(1, cabinet_count, num=cabinet_count),
            tilt_angle_deg=optimized.tilt_angles_deg,
            inter_cabinet_angle_deg=optimized.inter_cabinet_angles_deg)
        # Sound Field Prediction
        results.venue_points_source.data = dict(
            x_m=drawn_lines.points_x, y_m=drawn_lines.points_y,
            point_index=list(np.arange(np.shape(drawn_lines.points_x)[0])))
        results.venue_points_figure.x(
            'x_m', 'y_m', source=results.venue_points_source,
            color=linear_cmap('point_index', rainbow[::-1], 0,
                              np.shape(evaluation.spl_db)[0]-1))
        results.sound_field_source.data = dict(
            spl_db=evaluation.plot_spl_db,
            frequencies_hz=evaluation.plot_frequencies,
            line_index=list(np.arange(np.shape(evaluation.plot_spl_db)[0])),
            reference_spl_db=evaluation_reference.plot_spl_db)
        results.sound_field_figure.multi_line(
            'frequencies_hz', 'spl_db', source=results.sound_field_source,
            color=linear_cmap(field_name='line_index', palette=rainbow[::-1],
                              low=0, high=np.shape(evaluation.spl_db)[0]-1),
            line_alpha=0.9)
        results.reference_sound_field_figure.multi_line(
            'frequencies_hz', 'reference_spl_db',
            source=results.sound_field_source,
            color=linear_cmap(
                field_name='line_index', palette=rainbow[::-1], low=0,
                high=np.shape(evaluation_reference.spl_db)[0]-1),
            line_alpha=0.9)
        # SPL over distance and target slope; writing the target sources
        # snaps the shown target to the new distance grid (snap_target)
        spl_source = algorithm.spl_over_distance_source
        reference_spl_source = algorithm.reference_spl_over_distance_source
        if result.target is not None:
            spl_source.data.update(dict(spl_db=spl_over_distance.levels_db,
                                        distance_m=spl_over_distance.distances,
                                        x_m=drawn_lines.points_x,
                                        y_m=drawn_lines.points_y))
            algorithm.target_source.data.update(dict(
                level_db=result.target.levels_db))
            algorithm.target_reference_source.data.update(dict(
                level_db=[target.region_levels_db[target.reference_index]]))
            reference_spl_source.data.update(dict(
                reference_spl_db=reference_spl_over_distance.levels_db))
        elif not target.initialized:
            spl_source.data.update(dict(spl_db=spl_over_distance.levels_db,
                                        distance_m=spl_over_distance.distances,
                                        x_m=drawn_lines.points_x,
                                        y_m=drawn_lines.points_y))
            reference_spl_source.data.update(dict(
                reference_spl_db=reference_spl_over_distance.levels_db,
                distance_m=reference_spl_over_distance.distances,
                x_m=drawn_lines.points_x, y_m=drawn_lines.points_y))
            algorithm.target_source.data.update(dict(
                level_db=[spl_over_distance.levels_db[0],
                          spl_over_distance.levels_db[-1]],
                distance_m=[spl_over_distance.distances[0],
                            spl_over_distance.distances[-1]],
                x_m=[drawn_lines.points_x[0], drawn_lines.points_x[-1]],
                y_m=[drawn_lines.points_y[0], drawn_lines.points_y[-1]]))
            middle = int(len(target.region_distances)/2)
            middle_point = int(len(drawn_lines.points_x)/2)
            algorithm.target_reference_source.data.update(dict(
                distance_m=[target.region_distances[middle]],
                level_db=[target.region_levels_db[
                    int(len(target.region_levels_db)/2)]],
                x_m=[drawn_lines.points_x[middle_point]],
                y_m=[drawn_lines.points_y[
                    int(len(drawn_lines.points_y)/2)]]))
        else:
            spl_source.data.update(dict(spl_db=spl_over_distance.levels_db,
                                        distance_m=spl_over_distance.distances,
                                        x_m=drawn_lines.points_x,
                                        y_m=drawn_lines.points_y))
            reference_spl_source.data.update(dict(
                reference_spl_db=reference_spl_over_distance.levels_db,
                distance_m=reference_spl_over_distance.distances,
                x_m=drawn_lines.points_x, y_m=drawn_lines.points_y))
        # Homogeneity
        results.homogeneity_source.data = dict(
            frequency_hz=evaluation.frequencies,
            homogeneity_db=evaluation.homogeneity_db,
            homogeneity_low_db=evaluation.homogeneity_low_db,
            homogeneity_high_db=evaluation.homogeneity_high_db,
            reference_homogeneity_db=evaluation_reference.homogeneity_db)
        # Histogram
        results.histogram_source.data = dict(count=evaluation.histogram_tops,
                                             spl_db=evaluation.histogram_steps)
        results.reference_histogram_source.data = dict(
            count=evaluation_reference.histogram_tops,
            spl_db=evaluation_reference.histogram_steps)
        # Ranges of bar charts, homogeneity, sound field and SPL over distance
        rng = st.ranges
        histogram_x_range = rng.update_range(
            'histogram_x', [-1, 1], evaluation.histogram_steps,
            evaluation_reference.histogram_steps)
        histogram_top_range = rng.update_range(
            'histogram_top', [-1, 2], evaluation.histogram_tops,
            evaluation_reference.histogram_tops)
        results.histogram_figure.x_range = histogram_x_range
        results.histogram_figure.y_range = histogram_top_range
        results.reference_histogram_figure.x_range = histogram_x_range
        results.reference_histogram_figure.y_range = histogram_top_range
        results.homogeneity_figure.y_range = rng.update_range(
            'homogeneity_y', [-2, 2], evaluation.homogeneity_db,
            evaluation_reference.homogeneity_db)
        sound_field_y_range = rng.update_range(
            'sound_field_y', [-5, 5], evaluation.spl_db,
            evaluation_reference.spl_db)
        results.sound_field_figure.y_range = sound_field_y_range
        results.reference_sound_field_figure.y_range = sound_field_y_range
        algorithm.spl_over_distance_figure.y_range = rng.update_range(
            'spl_over_distance_y', [-1, 1], target.region_levels_db,
            spl_over_distance.region_levels_db, spl_over_distance.levels_db,
            reference_spl_over_distance.levels_db)
        results.beam_checkbox.active, results.beam_checkbox.disabled = [], False
        # Array visualization
        self.cabinet_patches_source.data.update(dict(
            patches_x=optimized.patches_x, patches_y=optimized.patches_y,
            reference_patches_x=reference.patches_x,
            reference_patches_y=reference.patches_y))
