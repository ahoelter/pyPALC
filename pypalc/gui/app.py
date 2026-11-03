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
from bokeh.models import ColumnDataSource, Model
from bokeh.models.widgets import Div, Paragraph, RadioButtonGroup, RadioGroup
from bokeh.plotting import figure
from bokeh.transform import linear_cmap
from colorcet import rainbow

from pypalc.core import pipeline
from pypalc.core.errors import PalcError, SetupError
from pypalc.core.palc import Array_opt, PALC_configuration
from pypalc.core.pipeline import AudienceLines, RunResult, VenueResult
from pypalc.core.settings import AudienceLine, Setup
from pypalc.core.sound_field import SFP_configuration
from pypalc.core.target_slope import Opt_weight
from pypalc.core.venue import (PAL, PALC_compatible_PAL,
                               get_PALC_compatible_PAL, suggestPAL)
from pypalc.gui import (APP_NAME, CALLBACK_ERRORS,
                        skipped_while_applying_setup)
from pypalc.gui.help_text import HelpPage
from pypalc.gui.page_algorithm import AlgorithmPage
from pypalc.gui.page_array import ArrayPage
from pypalc.gui.page_loudspeaker import LoudspeakerPage
from pypalc.gui.page_results import ResultsPage
from pypalc.gui.page_venue import VenuePage
from pypalc.gui.setup_io import setup_from_app
from pypalc.gui.ranges import (Plot_ranges, get_p_axis_rng, set_figure_ranges,
                               venue_slice2CDS)

logger = logging.getLogger(__name__)


Page = Union[HelpPage, VenuePage, LoudspeakerPage, ArrayPage, AlgorithmPage,
             ResultsPage]


def _initial_drawn_lines() -> AudienceLines:
    """Audience lines of a new session (no lines drawn)."""
    return PAL(pal_index=np.array([0]).astype(int))


@dataclass
class SessionState:
    """Calculation objects of one GUI session.

    Attributes
    ----------
    drawn : AudienceLines
        Audience lines drawn by the user (legacy ``created_pal``).
    audience_lines : list of AudienceLine
        The drawn audience lines as settings, in the order of ``drawn``,
        with the discretization each line was saved with.
    venue : VenueResult or None
        Venue of the last successful "Get PALC Venue Proposal", None before.
        Like in the legacy GUI it is kept when lines change afterwards; the
        start toggle is disabled until the venue is checked again.
    measured_directivity_csv : str or None
        Text of the last readable upload of measured directivity data.
    config : PALC_configuration
        PALC configuration (legacy ``PALC_config``), holding the values of
        the configuration widgets. Calculations do not change it.
    sound_field_config : SFP_configuration
        Sound field prediction configuration (legacy ``SFP_config``), used
        for the directivity and beam plots.
    ranges : Plot_ranges
        Ranges of the figures.
    optimized_array : ArrayGeometry
        Holds the allowed range of the fixed angle of the uppermost cabinet
        (attribute ``fixed_angle``, legacy ``Opt_arr.fixed_angle``).
    reference_array : ArrayGeometry
        The reference array as computed on page 3 (legacy ``Ref_arr``).
    target : TargetSlopeState
        The target slope shown on page 4 (legacy ``Opt_w``).
    result : RunResult or None
        Results of the last successful calculation, None before.
    target_of_result : tuple or None
        The target slope as shown right after the last successful
        calculation: ``(x, SPL, x_ref)`` of the target sources. Save
        compares it with the shown target to find out whether the target
        was moved since (:func:`~.setup_io.setup_to_save`).
    applying_setup : bool
        True while :func:`~.setup_io.apply_setup` sets the widgets; the
        widget callbacks then do nothing
        (:func:`~pypalc.gui.skipped_while_applying_setup`).
    """
    drawn: AudienceLines = field(default_factory=_initial_drawn_lines)
    audience_lines: List[AudienceLine] = field(default_factory=list)
    venue: Optional[VenueResult] = None
    measured_directivity_csv: Optional[str] = None
    config: PALC_configuration = field(default_factory=PALC_configuration)
    sound_field_config: SFP_configuration = field(
        default_factory=SFP_configuration)
    ranges: Plot_ranges = field(default_factory=Plot_ranges)
    optimized_array: Array_opt = field(default_factory=Array_opt)
    reference_array: Array_opt = field(
        default_factory=lambda: Array_opt(gamma_tilt_deg=np.zeros(4)))
    target: Opt_weight = field(default_factory=Opt_weight)
    result: Optional[RunResult] = None
    target_of_result: Optional[tuple] = None
    applying_setup: bool = False


def _error_text(exc: PalcError) -> str:
    """Error text shown for a failed calculation."""
    if isinstance(exc, SetupError):
        return 'Error: %s' % exc
    return str(exc)


def _logo_figure(width: int, image: str, x_offset: float):
    """Figure that shows a logo from the app's static directory."""
    x_range, y_range = (10, 30), (10, 30)
    logo = figure(x_range=x_range, y_range=y_range, plot_width=width,
                  plot_height=80, min_border=0)
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
            x_list=[], y_list=[], x_list_ref=[], y_list_ref=[]))
        """Outlines of the cabinets of the optimized and reference array."""
        self.hit_points_source = ColumnDataSource(data=dict(
            seg_pos_x=[], seg_pos_y=[], seg_pos_start_x=[],
            seg_pos_start_y=[], seg_pos_stop_x=[], seg_pos_stop_y=[]))
        """Points where the rays of the cabinets hit the venue."""
        self.rays_source = ColumnDataSource(data=dict(
            x_start_stop=[], y_start_stop=[], x_fin_unitn_start_stop=[],
            y_fin_unitn_start_stop=[], x_start_unitn_start=[],
            y_start_unitn_start=[], x_stop_unitn_stop=[],
            y_stop_unitn_stop=[], x_c_n_unitn=[], y_c_n_unitn=[],
            x_c_n_unitn_psi1=[], y_c_n_unitn_psi1=[], x_c_n_unitn_psi2=[],
            y_c_n_unitn_psi2=[]))
        """Rays from the cabinets to the venue."""
        self.venue_figure = figure(
            title="", x_axis_label="x in m", y_axis_label="y in m",
            toolbar_sticky=False, toolbar_location="right",
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
        self.state.config.store_config('psi_n',
                                       self.pages[2].aperture_input.value)
        self.state.config.store_config('N',
                                       self.pages[3].cabinet_count_input.value)

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
            active=2, height=15, margin=5, orientation='horizontal',
            inline=True)
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
        return venue is not None and len(venue.palc_venue.pal) > 0

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
            suggested_nal = PAL(ind_iscut=[])
            suggested_sfp = PAL(ind_iscut=[])
            suggestPAL(st.drawn, suggested_nal, suggested_sfp)
            venue_page.non_audience_source.data = dict(
                xnalplt=suggested_nal.plot_x, ynalplt=suggested_nal.plot_y)
            venue_page.sound_field_only_source.data = dict(
                xsfpplt=suggested_sfp.plot_x, ysfpplt=suggested_sfp.plot_y)
            st.venue = None
            palc_venue = PALC_compatible_PAL(pal=[], pal_no_nal=[])
            get_PALC_compatible_PAL(palc_venue, st.drawn, suggested_nal,
                                    suggested_sfp)
            palc_venue.create_pal_without_nal(suggested_nal)
            st.venue = VenueResult(drawn=st.drawn, palc_venue=palc_venue,
                                   non_audience=suggested_nal,
                                   sound_field_only=suggested_sfp)
            self.set_venue_checked(True)
            self.config_error_info.text = ''
            self.show()
            if algorithm.weighting_select.value in ['Target Slope']:
                algorithm.weighting_select.value = 'Without'
                st.target.init = False
                algorithm.choose_weighting(0, 0, 0)
            weight_links = np.arange(np.shape(st.drawn.xline_start)[0]+1)
            algorithm.hinges_select.options = [str(n) for n in
                                               list(weight_links)]
            algorithm.hinges_select.value = '0'
        except CALLBACK_ERRORS:
            logger.exception('Could not create the PALC venue')
            self.config_error_info.text = 'Error: Could not create PAL proposal'

    def set_venue_checked(self, has_pal: bool) -> None:
        """Enable the start toggle if the venue is PALC compatible
        (legacy ``change_start_pal_button``).

        Parameters
        ----------
        has_pal : bool
            If True and a venue exists, PALC computation is enabled,
            otherwise disabled.

        Returns
        -------
        None.
        """
        venue_page, algorithm = self.pages[1], self.pages[4]
        if has_pal and self.has_venue():
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
        x_range = st.ranges.update_range('p_x', [-2, 2], st.drawn.xline_start,
                                         st.drawn.xline_stop, st.config.x_H)
        y_range = st.ranges.update_range('p_y', [-2, 2], st.drawn.yline_start,
                                         st.drawn.yline_stop, st.config.y_H)
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
        get_p_axis_rng(self.venue_figure, self.state.ranges,
                       self.venue_axes_select)
        if call:
            self.show()

    def clear_result_figure(self) -> None:
        """Remove the array and its rays from the venue figure (legacy
        ``clear_result_fig``)."""
        venue_slice2CDS(self.rays_source, self.hit_points_source, [], [])

    # ------------------------------------------------------------------
    # Values of the configuration widgets
    # ------------------------------------------------------------------

    def _config_widgets(self) -> Dict[str, Model]:
        """Configuration key -> widget whose value is stored under it."""
        loudspeaker, array, algorithm = (self.pages[2], self.pages[3],
                                         self.pages[4])
        return {'constraint': algorithm.palc_variant_select,
                'use_gamma_LSA': array.use_discrete_angles_select,
                'gap_handling': algorithm.gap_handling_select,
                'strength_sm': algorithm.soft_margin_slider,
                'Lambda_y': loudspeaker.cabinet_height_input,
                'Lambda_gap': array.cabinet_gap_input,
                'directivity': loudspeaker.directivity_select,
                'fixed_angle': algorithm.fixed_angle_input,
                'x_H': array.top_x_input,
                'y_H': array.top_y_input,
                'N': array.cabinet_count_input,
                'psi_n': loudspeaker.aperture_input,
                'tolerance': algorithm.tolerance_slider,
                'freq_range': algorithm.frequency_range_slider}

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
        defaults = {'Lambda_y': (loudspeaker.cabinet_height_input, '0.2'),
                    'N': (array.cabinet_count_input, '4'),
                    'Lambda_gap': (array.cabinet_gap_input, '0'),
                    'tolerance': (algorithm.tolerance_slider, '0.1'),
                    'fixed_angle': (algorithm.fixed_angle_input, ''),
                    'x_H': (array.top_x_input, '0'),
                    'y_H': (array.top_y_input, '2'),
                    'psi_n': (loudspeaker.aperture_input, '3.0')}
        error_info, failed_key = st.config.store_config(key, val)
        # if wrong user input --> insert default value
        if failed_key in defaults:
            widget, default = defaults[failed_key]
            setattr(widget, attr, default)
        if failed_key in ['too low']:
            setattr(array.top_y_input, attr, str(st.config.y_H))
        # Set attribute in SFP_config if possible
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
        # more than one venue point (the legacy test ``xline != []``)
        if np.size(st.drawn.xline) > 1:
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
                if key in ['N', 'use_gamma_LSA']:
                    self.pages[3].update_reference_angles(chg_n=True)
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
                if key not in ('use_gamma_LSA', 'fixed_angle', 'y_H')]
        for key in keys + ['y_H']:
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
                not self.state.target.init:
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
        N = result.setup.array.cabinet_count
        optimized, reference = result.optimized, result.reference
        Tech_res, Tech_res_ref = optimized.evaluation, reference.evaluation
        SPLoverX, SPLoverX_ref = optimized.spl_over_distance, \
            reference.spl_over_distance
        created_pal = result.venue.drawn
        Opt_w = st.target
        self.config_error_info.text = ''
        # array and rays in the venue figure
        venue_slice2CDS(self.rays_source, self.hit_points_source,
                        result.ray_plot, result.geometry)
        # tilt angles
        algorithm.results_source.data = dict(
            a_num_LS=np.linspace(1, N, num=N),
            b_gamma_tilt_deg=optimized.tilt_angles_deg,
            c_gamma_tilt_deg_diff=optimized.inter_cabinet_angles_deg)
        # Sound Field Prediction
        results.venue_points_source.data = dict(
            x_vert=created_pal.xline, y_vert=created_pal.yline,
            foo=list(np.arange(np.shape(created_pal.xline)[0])))
        results.venue_points_figure.x(
            'x_vert', 'y_vert', source=results.venue_points_source,
            color=linear_cmap('foo', rainbow[::-1], 0,
                              np.shape(Tech_res.p_SPL)[0]-1))
        results.sound_field_source.data = dict(
            p_SPL=Tech_res.plot_p_SPL, f=Tech_res.plot_f,
            foo=list(np.arange(np.shape(Tech_res.plot_p_SPL)[0])),
            p_SPL_ref=Tech_res_ref.plot_p_SPL)
        results.sound_field_figure.multi_line(
            'f', 'p_SPL', source=results.sound_field_source,
            color=linear_cmap(field_name='foo', palette=rainbow[::-1], low=0,
                              high=np.shape(Tech_res.p_SPL)[0]-1),
            line_alpha=0.9)
        results.reference_sound_field_figure.multi_line(
            'f', 'p_SPL_ref', source=results.sound_field_source,
            color=linear_cmap(field_name='foo', palette=rainbow[::-1], low=0,
                              high=np.shape(Tech_res_ref.p_SPL)[0]-1),
            line_alpha=0.9)
        # SPL over distance and target slope; writing the target sources
        # snaps the shown target to the new distance grid (snap_target)
        spl_source = algorithm.spl_over_distance_source
        spl_ref_source = algorithm.reference_spl_over_distance_source
        if result.target is not None:
            spl_source.data.update(dict(SPL=SPLoverX.SPL, x=SPLoverX.x,
                                        x_v=created_pal.xline,
                                        y_v=created_pal.yline))
            algorithm.target_source.data.update(dict(SPL=result.target.SPL))
            algorithm.target_reference_source.data.update(dict(
                SPL=[Opt_w.SPL_interp[Opt_w.ref_ind]]))
            spl_ref_source.data.update(dict(SPL=SPLoverX_ref.SPL))
        elif not Opt_w.init:
            spl_source.data.update(dict(SPL=SPLoverX.SPL, x=SPLoverX.x,
                                        x_v=created_pal.xline,
                                        y_v=created_pal.yline))
            spl_ref_source.data.update(dict(SPL=SPLoverX_ref.SPL,
                                            x=SPLoverX_ref.x,
                                            x_v=created_pal.xline,
                                            y_v=created_pal.yline))
            algorithm.target_source.data.update(dict(
                SPL=[SPLoverX.SPL[0], SPLoverX.SPL[-1]],
                x=[SPLoverX.x[0], SPLoverX.x[-1]],
                x_v=[created_pal.xline[0], created_pal.xline[-1]],
                y_v=[created_pal.yline[0], created_pal.yline[-1]]))
            algorithm.target_reference_source.data.update(dict(
                x_ref=[Opt_w.x_interp[int(len(Opt_w.x_interp)/2)]],
                SPL=[Opt_w.SPL_interp[int(len(Opt_w.SPL_interp)/2)]],
                x_v=[created_pal.xline[int(len(created_pal.xline)/2)]],
                y_v=[created_pal.yline[int(len(created_pal.yline)/2)]]))
        else:
            spl_source.data.update(dict(SPL=SPLoverX.SPL, x=SPLoverX.x,
                                        x_v=created_pal.xline,
                                        y_v=created_pal.yline))
            spl_ref_source.data.update(dict(SPL=SPLoverX_ref.SPL,
                                            x=SPLoverX_ref.x,
                                            x_v=created_pal.xline,
                                            y_v=created_pal.yline))
        # Homogeneity
        results.homogeneity_source.data = dict(
            H=Tech_res.H, f=Tech_res.f, H_dist_high=Tech_res.H_dist[1],
            H_dist_low=Tech_res.H_dist[0], H_str=Tech_res_ref.H)
        # Histogram
        results.histogram_source.data = dict(top=Tech_res.Hist_tops,
                                             x=Tech_res.Hist_steps)
        results.reference_histogram_source.data = dict(
            top_ref=Tech_res_ref.Hist_tops, x_ref=Tech_res_ref.Hist_steps)
        # Ranges of bar charts, homogeneity, sound field and SPL over distance
        rng = st.ranges
        bar_range = rng.update_range('bar_range', [-1, 1], Tech_res.Hist_steps,
                                     Tech_res_ref.Hist_steps)
        bar_top_range = rng.update_range('bartop', [-1, 2], Tech_res.Hist_tops,
                                         Tech_res_ref.Hist_tops)
        results.histogram_figure.x_range = bar_range
        results.histogram_figure.y_range = bar_top_range
        results.reference_histogram_figure.x_range = bar_range
        results.reference_histogram_figure.y_range = bar_top_range
        results.homogeneity_figure.y_range = rng.update_range(
            'h_y', [-2, 2], Tech_res.H, Tech_res_ref.H)
        psfp_y_range = rng.update_range('sfp_y', [-5, 5], Tech_res.p_SPL,
                                        Tech_res_ref.p_SPL)
        results.sound_field_figure.y_range = psfp_y_range
        results.reference_sound_field_figure.y_range = psfp_y_range
        algorithm.spl_over_distance_figure.y_range = rng.update_range(
            'SPLoverX_y', [-1, 1], Opt_w.SPL_interp, SPLoverX.SPL_interp,
            SPLoverX.SPL, SPLoverX_ref.SPL)
        results.beam_checkbox.active, results.beam_checkbox.disabled = [], False
        # Array visualization
        self.cabinet_patches_source.data.update(dict(
            x_list=optimized.patches_x, y_list=optimized.patches_y,
            x_list_ref=reference.patches_x, y_list_ref=reference.patches_y))
