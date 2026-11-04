"""Page 5 of the GUI: visualization of the sound field prediction.

author: Arne Hoelter
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from bokeh.layouts import column, row
from bokeh.models import (CheckboxGroup, ColorBar, ColumnDataSource, Div,
                          LayoutDOM, LinearColorMapper, Select, TabPanel,
                          Tabs)
from bokeh.palettes import Cividis256
from bokeh.plotting import figure

from pypalc.core.sound_field import beam_plot_spl
from pypalc.gui import CALLBACK_ERRORS
from pypalc.gui.theme import DASHED

if TYPE_CHECKING:
    from .app import PalcApp

logger = logging.getLogger(__name__)

#: Ticks of the frequency axes and their labels in kHz.
F_TICKS = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
F_TICKS_OVERRIDE = {20: '0.02', 50: '0.05', 100: '0.1', 200: '0.2', 500: '0.5',
                    1000: '1', 2000: '2', 5000: '5', 10000: '10', 20000: '20'}


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
        # widgets
        self.beam_frequency_select = Select(
            title="Frequency in Hz", value="63", width=200,
            options=["63", "125", "250", "500", "1000", "2000", "4000",
                     "8000", "16000"])
        """Frequency of the beam plots."""
        self.beam_checkbox = CheckboxGroup(
            labels=["PALC", "Reference"], active=[], height=11, inline=True,
            disabled=True)
        """Whether the PALC and / or the reference beam plot is shown."""
        self.beam_label = Div(text='      <b> Beam Plot of: </b>', height=10,
                              width=120)
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
            title="28. Venue Slice Positions", x_axis_label="x in m",
            y_axis_label="y in m",
            tools="box_zoom,pan,wheel_zoom,reset,tap,hover,save",
            tooltips=[("Index", "$index"), ("(x,y)", "(@x_m, @y_m)")],
            width=1220, height=300)
        self.venue_points_figure.toolbar.logo = None
        self.venue_points_figure.patches('patches_x', 'patches_y',
                                         source=patches_source, color="white")
        self.venue_points_source.selected.on_change('indices',
                                                    self.venue_point_tapped)
        # frequency responses of the optimized and the reference array
        self.sound_field_figure = self._response_figure(
            "29. SPL at all Audience Positions: PALC")
        self.reference_sound_field_figure = self._response_figure(
            "30. SPL at all Audience Positions: Reference Array")
        # homogeneity
        homogeneity_figure = figure(
            title="31. Homogeneity", x_axis_type="log",
            x_axis_label="f in kHz", y_axis_label="H(f) in dB",
            tools="box_zoom,pan,wheel_zoom,reset,hover,save",
            tooltips=[("Frequency in Hz", "$x"), ("SPL in dB", "$y")],
            x_range=(15, 24000), width=1220)
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
        beam_figure = figure(
            title="34. Beam Plot PALC: SPL in dB", x_axis_label="x in m",
            y_axis_label="y in m", frame_width=1110, match_aspect=True,
            visible=False, tools="box_zoom,pan,wheel_zoom,reset,hover,save",
            tooltips=[("SPL in dB", "@spl_db"),
                      ("(x in m, y in m)", "($x, $y)")])
        beam_figure.toolbar.logo = None
        beam_figure.x_range.range_padding = \
            beam_figure.y_range.range_padding = 0
        self.beam_mapper = LinearColorMapper(palette=Cividis256, low=50,
                                             high=100)
        beam_figure.add_layout(ColorBar(color_mapper=self.beam_mapper,
                                        location=(0, 0), title="",
                                        title_text_font_style="normal"),
                               'right')
        beam_figure.image(image='spl_db', x='x', y='y', dw='dw', dh='dh',
                          source=self.beam_source, palette=Cividis256)
        beam_figure.multi_line('lines_x', 'lines_y', source=lines_source,
                               line_dash='solid', line_color="black",
                               line_width=4, selection_color="darkred")
        beam_figure.patches('patches_x', 'patches_y', source=patches_source,
                            color="black")
        self.beam_figure = beam_figure
        reference_beam_figure = figure(
            title="34. Beam Plot Reference: SPL in dB",
            x_axis_label="x in m", y_axis_label="y in m", frame_width=1110,
            match_aspect=True, visible=False,
            tools="box_zoom,pan,wheel_zoom,reset,hover,save",
            tooltips=[("SPL in dB", "@reference_spl_db"),
                      ("(x in m, y in m)", "($x, $y)")])
        reference_beam_figure.toolbar.logo = None
        # (as in the legacy code: the second assignment goes to beam_figure)
        reference_beam_figure.x_range.range_padding = \
            beam_figure.y_range.range_padding = 0
        self.reference_beam_mapper = LinearColorMapper(palette=Cividis256,
                                                       low=50, high=100)
        reference_beam_figure.add_layout(
            ColorBar(color_mapper=self.reference_beam_mapper,
                     location=(0, 0), title_standoff=10, title="",
                     title_text_font_style="normal"), 'right')
        reference_beam_figure.image(image='reference_spl_db', x='x', y='y',
                                    dw='dw', dh='dh', source=self.beam_source,
                                    palette=Cividis256)
        reference_beam_figure.multi_line('lines_x', 'lines_y',
                                         source=lines_source,
                                         line_dash='solid', line_color="black",
                                         line_width=4,
                                         selection_color="darkred")
        reference_beam_figure.patches('reference_patches_x',
                                      'reference_patches_y',
                                      source=patches_source, color="black")
        beam_figure.xaxis.axis_label_text_font_style = "normal"
        self.reference_beam_figure = reference_beam_figure
        # bar charts
        self.histogram_figure = figure(
            title="32. Bar Chart: PALC", x_axis_label="SPL in dB",
            y_axis_label="Number of Positions", width=610)
        self.histogram_figure.vbar(x='spl_db', top='count',
                                   source=self.histogram_source, bottom=0,
                                   width=0.9, color="firebrick")
        self.histogram_figure.toolbar.logo = None
        self.reference_histogram_figure = figure(
            title="33. Bar Chart: Reference Array", x_axis_label="SPL in dB",
            y_axis_label="Number of positions", width=610)
        self.reference_histogram_figure.vbar(
            x='spl_db', top='count', source=self.reference_histogram_source,
            bottom=0, width=0.9, color="firebrick")
        self.reference_histogram_figure.toolbar.logo = None
        # tabs
        self.tabs = Tabs(tabs=[
            TabPanel(child=column([self.venue_points_figure,
                                   row([self.sound_field_figure,
                                        self.reference_sound_field_figure])]),
                     title="Frequency Responses at Audience Positions"),
            TabPanel(child=homogeneity_figure, title="Homogeneity"),
            TabPanel(child=row([self.histogram_figure,
                                self.reference_histogram_figure]),
                     title="Bar Chart"),
            TabPanel(child=column([row([self.beam_frequency_select,
                                        column([self.beam_label,
                                                self.beam_checkbox])]),
                                   beam_figure, reference_beam_figure]),
                     title="Beam Plots")])
        # callbacks
        self.beam_frequency_select.on_change("value", self.update_beam_plot)
        self.beam_checkbox.on_change("active", self.update_beam_plot)

    @staticmethod
    def _response_figure(title: str):
        """Figure of the frequency responses at the venue points."""
        response_figure = figure(
            title=title, x_axis_type="log", x_range=(15, 24000),
            y_range=(0, 100), x_axis_label="f in kHz",
            y_axis_label="SPL in dB",
            tools="box_zoom,pan,wheel_zoom,reset,tap,hover,save",
            tooltips=[("Index", "$index"), ("Frequency in Hz", "$x"),
                      ("SPL in dB", "$y")], width=610)
        response_figure.toolbar.logo = None
        response_figure.xaxis.ticker = F_TICKS
        response_figure.xaxis.major_label_overrides = F_TICKS_OVERRIDE
        return response_figure

    def layout(self) -> LayoutDOM:
        """Body of the page below the header.

        Returns
        -------
        LayoutDOM
        """
        return row([self.tabs])

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
