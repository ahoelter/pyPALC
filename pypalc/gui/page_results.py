"""Page 5 of the GUI: visualization of the sound field prediction.

The figures of the steps 28 to 34 are in the tabs "Frequency responses",
"Homogeneity", "Bar charts" and "Beam plots" of one card. The beam plots
show the venue with equal scaling (:func:`~.ranges.set_figure_ranges`);
the other figures fill the width of the window.

author: Arne Hoelter
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
import panel as pn
import panel_material_ui as pmui
from bokeh.models import ColorBar, ColumnDataSource, LinearColorMapper
from bokeh.palettes import Cividis256
from bokeh.plotting import figure

from pypalc.core.sound_field import beam_plot_spl
from pypalc.gui import CALLBACK_ERRORS
from pypalc.gui.ranges import (VENUE_MAX_HEIGHT, add_venue_bounds,
                               scale_equally)
from pypalc.gui.style import DASHED
from pypalc.gui.widgets import GAP, caption, card, select

if TYPE_CHECKING:
    from .app import PalcApp

logger = logging.getLogger(__name__)

#: Ticks of the frequency axes and their labels in kHz.
F_TICKS = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
F_TICKS_OVERRIDE = {20: '0.02', 50: '0.05', 100: '0.1', 200: '0.2', 500: '0.5',
                    1000: '1', 2000: '2', 5000: '5', 10000: '10', 20000: '20'}
#: Frequencies of the beam plots in Hz.
BEAM_FREQUENCIES = ['63', '125', '250', '500', '1000', '2000', '4000', '8000',
                    '16000']
#: Height in px of the venue points figure and of the figures shown side
#: by side (frequency responses, bar charts).
POINTS_HEIGHT = 300
PAIR_HEIGHT = 400
#: Preferred width in px of a figure shown beside another one; below two
#: of them the second moves below the first.
PAIR_WIDTH = 420


class ResultsPage:
    """Page 5: frequency responses, homogeneity, bar charts and beam plots.

    Parameters
    ----------
    app : PalcApp
        The session the page belongs to.
    """

    def __init__(self, app: PalcApp) -> None:
        self.app = app
        lines_source = app.pages[1].lines_source
        patches_source = app.cabinet_patches_source
        self._layout: Optional[pn.viewable.Viewable] = None
        # widgets
        self.beam_frequency_select = select(
            'Frequency in Hz', list(BEAM_FREQUENCIES), '63', width=160,
            sizing_mode='fixed')
        """Frequency of the beam plots."""
        self.beam_checkbox = pmui.CheckButtonGroup(
            options=['PALC', 'Reference'], value=[], disabled=True,
            size='small', color='primary', align='center', margin=(4, 8))
        """Whether the PALC and / or the reference beam plot is shown;
        ``active`` holds the indices of the shown plots."""
        # sources
        empty = list(zip([0, 0]))
        self.sound_field_source = ColumnDataSource(data=dict(
            spl_db=empty, frequencies_hz=empty,
            line_index=list(range(np.shape(empty)[0])),
            reference_spl_db=empty))
        """SPL of the optimized and the reference array at the venue points."""
        self.venue_points_source = ColumnDataSource(data=dict(
            x_m=empty, y_m=empty, point_index=list(range(np.shape(empty)[0]))))
        """The discrete venue points."""
        self.homogeneity_source = ColumnDataSource(data=dict(
            frequency_hz=empty, homogeneity_db=empty,
            homogeneity_low_db=empty, homogeneity_high_db=empty,
            reference_homogeneity_db=empty))
        """Homogeneity of the optimized and the reference array; the
        columns are those of ``homogeneity.csv`` of the headless export."""
        self.histogram_source = ColumnDataSource(data=dict(count=empty,
                                                           spl_db=empty))
        self.reference_histogram_source = ColumnDataSource(data=dict(
            count=empty, spl_db=empty))
        self.beam_source = ColumnDataSource(data=dict(
            spl_db=[np.zeros([100, 100])],
            reference_spl_db=[np.zeros([100, 100])],
            x=[-2], y=[-2], dw=[4], dh=[4]))
        """SPL, position and size of the beam plots."""
        # venue points with the colors of the frequency responses
        self.venue_points_figure = figure(
            title="28. Venue slice positions", x_axis_label="x in m",
            y_axis_label="y in m",
            tools="box_zoom,pan,wheel_zoom,reset,tap,hover,save",
            tooltips=[("Index", "$index"), ("(x,y)", "(@x_m, @y_m)")],
            sizing_mode='stretch_width', height=POINTS_HEIGHT)
        self.venue_points_figure.toolbar.logo = None
        self.venue_points_figure.patches('patches_x', 'patches_y',
                                         source=patches_source, color="white")
        self.venue_points_source.selected.on_change('indices',
                                                    self.venue_point_tapped)
        # frequency responses of the optimized and the reference array
        self.sound_field_figure = self._response_figure(
            "29. SPL at all audience positions: PALC")
        self.reference_sound_field_figure = self._response_figure(
            "30. SPL at all audience positions: Reference array")
        # homogeneity
        homogeneity_figure = figure(
            title="31. Homogeneity", x_axis_type="log",
            x_axis_label="f in kHz", y_axis_label="H(f) in dB",
            tools="box_zoom,pan,wheel_zoom,reset,hover,save",
            tooltips=[("Frequency in Hz", "$x"), ("SPL in dB", "$y")],
            x_range=(15, 24000), sizing_mode='stretch_width',
            height=VENUE_MAX_HEIGHT)
        homogeneity_figure.toolbar.logo = None
        homogeneity_figure.xaxis.ticker = F_TICKS
        homogeneity_figure.xaxis.major_label_overrides = F_TICKS_OVERRIDE
        homogeneity_figure.line('frequency_hz', 'homogeneity_db',
                                source=self.homogeneity_source,
                                color="lightskyblue", legend_label="PALC")
        homogeneity_figure.line('frequency_hz', 'homogeneity_high_db',
                                source=self.homogeneity_source,
                                color="navajowhite", line_dash=DASHED)
        homogeneity_figure.line('frequency_hz', 'homogeneity_low_db',
                                source=self.homogeneity_source,
                                color="navajowhite", line_dash=DASHED)
        homogeneity_figure.line('frequency_hz', 'reference_homogeneity_db',
                                source=self.homogeneity_source,
                                color="orange", legend_label="Reference Array")
        homogeneity_figure.legend.location = "top_left"
        self.homogeneity_figure = homogeneity_figure
        # beam plots
        self.beam_mapper = LinearColorMapper(palette=Cividis256, low=50,
                                             high=100)
        self.beam_figure = self._beam_figure(
            "34. Beam plot: PALC (SPL in dB)", 'spl_db', self.beam_mapper,
            lines_source, ('patches_x', 'patches_y'))
        self.reference_beam_mapper = LinearColorMapper(palette=Cividis256,
                                                       low=50, high=100)
        self.reference_beam_figure = self._beam_figure(
            "34. Beam plot: Reference array (SPL in dB)", 'reference_spl_db',
            self.reference_beam_mapper, lines_source,
            ('reference_patches_x', 'reference_patches_y'))
        # bar charts
        self.histogram_figure = self._histogram_figure(
            "32. Bar chart: PALC", self.histogram_source)
        self.reference_histogram_figure = self._histogram_figure(
            "33. Bar chart: Reference array", self.reference_histogram_source)
        # tabs; their figures are added with the layout (when the document
        # has its theme)
        self._tab_columns = [pn.Column(sizing_mode='stretch_width',
                                       margin=(8, 0, 0, 0)) for _ in range(4)]
        self.tabs = pmui.Tabs(
            *zip(('Frequency responses', 'Homogeneity', 'Bar charts',
                  'Beam plots'), self._tab_columns),
            sizing_mode='stretch_width', margin=(0, 8))
        """Tabs of the figures."""
        self.scale_beam_figures()
        # callbacks
        self.beam_frequency_select.param.watch(lambda event: (
            self.update_beam_plot('value', event.old, event.new)), 'value')
        self.beam_checkbox.param.watch(lambda event: (
            self.update_beam_plot('active', event.old, event.new)), 'active')

    @staticmethod
    def _response_figure(title: str):
        """Figure of the frequency responses at the venue points."""
        response_figure = figure(
            title=title, x_axis_type="log", x_range=(15, 24000),
            y_range=(0, 100), x_axis_label="f in kHz",
            y_axis_label="SPL in dB",
            tools="box_zoom,pan,wheel_zoom,reset,tap,hover,save",
            tooltips=[("Index", "$index"), ("Frequency in Hz", "$x"),
                      ("SPL in dB", "$y")],
            sizing_mode='stretch_width', height=PAIR_HEIGHT)
        response_figure.toolbar.logo = None
        response_figure.xaxis.ticker = F_TICKS
        response_figure.xaxis.major_label_overrides = F_TICKS_OVERRIDE
        return response_figure

    @staticmethod
    def _histogram_figure(title: str, source: ColumnDataSource):
        """Bar chart of the SPL at the venue points."""
        histogram_figure = figure(
            title=title, x_axis_label="SPL in dB",
            y_axis_label="Number of positions", sizing_mode='stretch_width',
            height=PAIR_HEIGHT)
        histogram_figure.vbar(x='spl_db', top='count', source=source,
                              bottom=0, width=0.9, color="firebrick")
        histogram_figure.toolbar.logo = None
        return histogram_figure

    def _beam_figure(self, title: str, image: str,
                     mapper: LinearColorMapper,
                     lines_source: ColumnDataSource, patches: tuple):
        """Beam plot of one array: the SPL in the venue with the audience
        lines and the array, equally scaled (:meth:`scale_beam_figures`).

        Parameters
        ----------
        title : str
            Title of the figure.
        image : str
            Column of the SPL in :attr:`beam_source`.
        mapper : LinearColorMapper
            Color mapper of the SPL (its range is set with the SPL).
        lines_source : ColumnDataSource
            The audience lines.
        patches : tuple of str
            Columns of the outlines of the cabinets of the array in
            :attr:`~.app.PalcApp.cabinet_patches_source`.

        Returns
        -------
        figure
        """
        beam_figure = figure(
            title=title, x_axis_label="x in m", y_axis_label="y in m",
            visible=False, tools="box_zoom,pan,wheel_zoom,reset,hover,save",
            tooltips=[("SPL in dB", "@" + image),
                      ("(x in m, y in m)", "($x, $y)")])
        beam_figure.toolbar.logo = None
        beam_figure.add_layout(ColorBar(color_mapper=mapper, location=(0, 0),
                                        title="",
                                        title_text_font_style="normal"),
                               'right')
        beam_figure.image(image=image, x='x', y='y', dw='dw', dh='dh',
                          source=self.beam_source, palette=Cividis256)
        beam_figure.multi_line('lines_x', 'lines_y', source=lines_source,
                               line_dash='solid', line_color="black",
                               line_width=4, selection_color="darkred")
        beam_figure.patches(patches[0], patches[1],
                            source=self.app.cabinet_patches_source,
                            color="black")
        beam_figure.xaxis.axis_label_text_font_style = "normal"
        add_venue_bounds(beam_figure)
        return beam_figure

    def layout(self) -> pn.Column:
        """Body of the page below the header.

        Built at the first call, then the same object: its widgets and
        figures show every change in place.

        Returns
        -------
        panel.Column
            One card with the tabs of the figures.
        """
        if self._layout is None:
            app = self.app

            def pane(model, **params):
                return app.bokeh_pane(model, sizing_mode='stretch_width',
                                      **params)

            def pair(left, right):
                """Two figures side by side, one below the other on narrow
                windows (each in a column: the flex basis of a Bokeh pane
                itself does not keep them side by side)."""
                return pn.FlexBox(
                    *[pn.Column(pane(model), sizing_mode='stretch_width',
                                margin=0, styles={
                                    'flex': f'1 1 {PAIR_WIDTH}px',
                                    'min-width': '0'})
                      for model in (left, right)],
                    flex_wrap='wrap', gap=f'{GAP}px',
                    sizing_mode='stretch_width', margin=0)
            responses, homogeneity, bars, beams = self._tab_columns
            responses.objects = [
                pane(self.venue_points_figure),
                caption('Tap a position to see its frequency responses.'),
                pair(self.sound_field_figure,
                     self.reference_sound_field_figure)]
            homogeneity.objects = [pane(self.homogeneity_figure)]
            bars.objects = [pair(self.histogram_figure,
                                 self.reference_histogram_figure)]
            beams.objects = [
                pn.Row(self.beam_frequency_select,
                       caption('Beam plot of', align='center',
                               sizing_mode=None, margin=(2, 0, 2, 16)),
                       self.beam_checkbox, margin=0),
                pane(self.beam_figure), pane(self.reference_beam_figure)]
            self._layout = pn.Column(
                # the tabs name the figures: no header
                card('Sound field', self.tabs, hide_header=True),
                sizing_mode='stretch_width', margin=(0, GAP))
        return self._layout

    def scale_beam_figures(self) -> None:
        """Show the venue in the beam plots with equal scaling, the height
        following the width (:func:`~.ranges.scale_equally`).

        Returns
        -------
        None.
        """
        for beam_figure in (self.beam_figure, self.reference_beam_figure):
            scale_equally(beam_figure, self.app.state.ranges,
                          variable_height=True)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _beam_spl(self, mapper, tilt_angles, frequency, frequency_index):
        """Beam plot of one array at one frequency, clipped to 50..100 dB.

        Parameters
        ----------
        mapper : LinearColorMapper [out]
            Color mapper of the beam plot; set to the SPL range.
        tilt_angles : array
            Tilt angles of the array in radian.
        frequency : float
            Frequency in Hz.
        frequency_index : int
            Index of the frequency in the directivity data.

        Returns
        -------
        spl_db : ndarray
            Sound pressure levels on the mesh.
        """
        st = self.app.state
        sound_field_config = st.sound_field_config
        spl_db, mesh_x, mesh_y = beam_plot_spl(
            st.config, tilt_angles, st.ranges, frequency,
            np.array(sound_field_config.measured_directivity_data[
                :, frequency_index]),
            np.array(sound_field_config.measured_angles_deg[:, 1]))
        high_ind = np.argwhere(spl_db > 100)
        low_ind = np.argwhere(spl_db < 50)
        for n in range(np.shape(high_ind)[0]):
            spl_db[high_ind[n, 0], high_ind[n, 1]] = 100
        for n in range(np.shape(low_ind)[0]):
            spl_db[low_ind[n, 0], low_ind[n, 1]] = 50
        mapper.low = np.amin(spl_db)
        mapper.high = np.amax(spl_db)
        return spl_db

    def update_beam_plot(self, attr, old, new) -> None:
        """Show the selected beam plots and compute newly selected ones.

        The beam plot of the PALC optimized array uses the array of the last
        calculation (``state.result``). Triggered by the frequency selection
        and the beam plot checkboxes.

        Parameters
        ----------
        attr : str
            Changed attribute (value or active).
        old, new : str or list of int
            Old and new frequency or checked boxes.

        Returns
        -------
        None.
        """
        st = self.app.state
        rng = st.ranges
        try:
            # visibility of the plots
            active = np.array(self.beam_checkbox.active)
            self.beam_figure.visible = bool(np.any(active == 0))
            self.reference_beam_figure.visible = bool(np.any(active == 1))
            # if the frequency is changed, make a list of it
            if isinstance(old, str):
                old = [int(old)]
            if isinstance(new, str):
                new = [int(new)]
            frequency = float(self.beam_frequency_select.value)
            # index of the frequency (for measured loudspeaker data)
            frequency_index = (np.abs(
                frequency-st.sound_field_config.frequencies).argmin())
            frequency_int = int(self.beam_frequency_select.value)
            for n in self.beam_checkbox.active:
                if n == 0 and 0 not in old and \
                        (0 in new or frequency_int in new):
                    spl_db = self._beam_spl(self.beam_mapper,
                                            st.result.geometry.tilt_angles,
                                            frequency, frequency_index)
                    self.beam_source.data.update(dict(
                        spl_db=[spl_db], x=[rng.venue_x[0]],
                        y=[rng.venue_y[0]],
                        dw=[rng.venue_x[1]-rng.venue_x[0]],
                        dh=[rng.venue_y[1]-rng.venue_y[0]]))
                elif n == 1 and 1 not in old and \
                        (1 in new or frequency_int in new):
                    reference_spl_db = self._beam_spl(
                        self.reference_beam_mapper,
                        st.reference_array.tilt_angles, frequency,
                        frequency_index)
                    self.beam_source.data.update(dict(
                        reference_spl_db=[reference_spl_db],
                        x=[rng.venue_x[0]], y=[rng.venue_y[0]],
                        dw=[rng.venue_x[1]-rng.venue_x[0]],
                        dh=[rng.venue_y[1]-rng.venue_y[0]]))
        except CALLBACK_ERRORS:
            logger.exception('Could not compute the beam plot')

    def venue_point_tapped(self, attr, old, new) -> None:
        """Show only the frequency responses of the tapped venue points.

        Parameters
        ----------
        attr : str
            Changed attribute (indices).
        old, new : list of int
            Previously and now tapped venue points.

        Returns
        -------
        None.
        """
        result = self.app.state.result
        if result is None:
            return
        evaluation = result.optimized.evaluation
        evaluation_reference = result.reference.evaluation
        frequencies, spl_db, reference_spl_db = [], [], []
        if np.shape(new)[0] >= 1:
            for n in range(np.shape(new)[0]):
                frequencies.append(evaluation.plot_frequencies[new[n]])
                spl_db.append(evaluation.plot_spl_db[new[n]])
                reference_spl_db.append(
                    evaluation_reference.plot_spl_db[new[n]])
            self.sound_field_source.data = dict(
                frequencies_hz=frequencies, spl_db=spl_db,
                line_index=list(np.arange(np.shape(spl_db)[0])),
                reference_spl_db=reference_spl_db)
        else:
            self.sound_field_source.data = dict(
                frequencies_hz=evaluation.plot_frequencies,
                spl_db=evaluation.plot_spl_db,
                line_index=list(np.arange(
                    np.shape(evaluation.plot_spl_db)[0])),
                reference_spl_db=evaluation_reference.plot_spl_db)
