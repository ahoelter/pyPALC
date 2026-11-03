"""Page 5 of the GUI: visualization of the sound field prediction.

author: Arne Hoelter
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from bokeh.layouts import column, row
from bokeh.models import (ColorBar, ColumnDataSource, LayoutDOM,
                          LinearColorMapper)
from bokeh.models.widgets import CheckboxGroup, Div, Panel, Select, Tabs
from bokeh.palettes import Cividis256
from bokeh.plotting import figure

from pypalc.core.sound_field import beam_plot_spl
from pypalc.gui import CALLBACK_ERRORS

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
            labels=["PALC", "Reference"], active=[], orientation='horizontal',
            height=11, inline=True, disabled=True)
        """Whether the PALC and / or the reference beam plot is shown."""
        self.beam_label = Div(text='      <b> Beam Plot of: </b>', height=10,
                              width=120)
        # sources
        p_SPL = list(zip([0, 0]))
        x_vert = list(zip([0, 0]))
        self.sound_field_source = ColumnDataSource(data=dict(
            p_SPL=p_SPL, f=p_SPL, foo=list(range(np.shape(p_SPL)[0])),
            p_SPL_ref=p_SPL))
        """SPL of the optimized and the reference array at the venue points."""
        self.venue_points_source = ColumnDataSource(data=dict(
            x_vert=x_vert, y_vert=x_vert,
            foo=list(range(np.shape(x_vert)[0]))))
        """The discrete venue points."""
        self.homogeneity_source = ColumnDataSource(data=dict(
            H=x_vert, f=x_vert, H_dist_high=x_vert, H_dist_low=x_vert,
            H_str=x_vert))
        self.histogram_source = ColumnDataSource(data=dict(top=x_vert,
                                                           x=x_vert))
        self.reference_histogram_source = ColumnDataSource(data=dict(
            top_ref=x_vert, x_ref=x_vert))
        self.beam_source = ColumnDataSource(data=dict(
            SPL=[np.zeros([100, 100])], SPL_ref=[np.zeros([100, 100])],
            x=[-2], y=[-2], dw=[4], dh=[4]))
        """SPL, position and size of the beam plots."""
        # venue points with the colors of the frequency responses
        self.venue_points_figure = figure(
            title="28. Venue Slice Positions", x_axis_label="x in m",
            y_axis_label="y in m",
            tools="box_zoom,pan,wheel_zoom,reset,tap,hover,save",
            tooltips=[("Index", "$index"), ("(x,y)", "(@x_vert, @y_vert)")],
            width=1220, height=300)
        self.venue_points_figure.toolbar.logo = None
        self.venue_points_figure.patches('x_list', 'y_list',
                                         source=patches_source, color="white")
        self.venue_points_source.selected.on_change('indices',
                                                    self.venue_point_tapped)
        # frequency responses of the optimized and the reference array
        self.sound_field_figure = self._response_figure(
            "29. SPL at all Audience Positions: PALC")
        self.reference_sound_field_figure = self._response_figure(
            "30. SPL at all Audience Positions: Reference Array")
        # homogeneity
        pHom = figure(title="31. Homogeneity", x_axis_type="log",
                      x_axis_label="f in kHz", y_axis_label="H(f) in dB",
                      tools="box_zoom,pan,wheel_zoom,reset,hover,save",
                      tooltips=[("Frequency in Hz", "$x"),
                                ("SPL in dB", "$y")],
                      x_range=(15, 24000), width=1220)
        pHom.toolbar.logo = None
        pHom.xaxis.ticker = F_TICKS
        pHom.xaxis.major_label_overrides = F_TICKS_OVERRIDE
        pHom.line('f', 'H', source=self.homogeneity_source,
                  color="lightskyblue", legend_label="PALC")
        pHom.line('f', 'H_dist_high', source=self.homogeneity_source,
                  color="navajowhite", line_dash="dashed")
        pHom.line('f', 'H_dist_low', source=self.homogeneity_source,
                  color="navajowhite", line_dash="dashed")
        pHom.line('f', 'H_str', source=self.homogeneity_source,
                  color="orange", legend_label="Reference Array")
        pHom.legend.location = "top_left"
        self.homogeneity_figure = pHom
        # beam plots
        pBeam = figure(title="34. Beam Plot PALC: SPL in dB",
                       x_axis_label="x in m", y_axis_label="y in m",
                       frame_width=1110, match_aspect=True, visible=False,
                       tools="box_zoom,pan,wheel_zoom,reset,hover,save",
                       tooltips=[("SPL in dB", "@SPL"),
                                 ("(x in m, y in m)", "($x, $y)")])
        pBeam.toolbar.logo = None
        pBeam.x_range.range_padding = pBeam.y_range.range_padding = 0
        self.beam_mapper = LinearColorMapper(palette=Cividis256, low=50,
                                             high=100)
        pBeam.add_layout(ColorBar(color_mapper=self.beam_mapper,
                                  location=(0, 0), title="",
                                  title_text_font_style="normal"), 'right')
        pBeam.image(image='SPL', x='x', y='y', dw='dw', dh='dh',
                    source=self.beam_source, palette=Cividis256)
        pBeam.multi_line('xlineplt', 'ylineplt', source=lines_source,
                         line_dash='solid', line_color="black", line_width=4,
                         selection_color="darkred")
        pBeam.patches('x_list', 'y_list', source=patches_source,
                      color="black")
        self.beam_figure = pBeam
        pBeam_ref = figure(title="34. Beam Plot Reference: SPL in dB",
                           x_axis_label="x in m", y_axis_label="y in m",
                           frame_width=1110, match_aspect=True, visible=False,
                           tools="box_zoom,pan,wheel_zoom,reset,hover,save",
                           tooltips=[("SPL in dB", "@SPL_ref"),
                                     ("(x in m, y in m)", "($x, $y)")])
        pBeam_ref.toolbar.logo = None
        # (as in the legacy code: the second assignment goes to pBeam)
        pBeam_ref.x_range.range_padding = pBeam.y_range.range_padding = 0
        self.reference_beam_mapper = LinearColorMapper(palette=Cividis256,
                                                       low=50, high=100)
        pBeam_ref.add_layout(ColorBar(color_mapper=self.reference_beam_mapper,
                                      location=(0, 0), title_standoff=10,
                                      title="",
                                      title_text_font_style="normal"),
                             'right')
        pBeam_ref.image(image='SPL_ref', x='x', y='y', dw='dw', dh='dh',
                        source=self.beam_source, palette=Cividis256)
        pBeam_ref.multi_line('xlineplt', 'ylineplt', source=lines_source,
                             line_dash='solid', line_color="black",
                             line_width=4, selection_color="darkred")
        pBeam_ref.patches('x_list_ref', 'y_list_ref', source=patches_source,
                          color="black")
        pBeam.xaxis.axis_label_text_font_style = "normal"
        self.reference_beam_figure = pBeam_ref
        # bar charts
        self.histogram_figure = figure(
            title="32. Bar Chart: PALC", x_axis_label="SPL in dB",
            y_axis_label="Number of Positions", width=610)
        self.histogram_figure.vbar(x='x', top='top',
                                   source=self.histogram_source, bottom=0,
                                   width=0.9, color="firebrick")
        self.histogram_figure.toolbar.logo = None
        self.reference_histogram_figure = figure(
            title="33. Bar Chart: Reference Array", x_axis_label="SPL in dB",
            y_axis_label="Number of positions", width=610)
        self.reference_histogram_figure.vbar(
            x='x_ref', top='top_ref', source=self.reference_histogram_source,
            bottom=0, width=0.9, color="firebrick")
        self.reference_histogram_figure.toolbar.logo = None
        # tabs
        self.tabs = Tabs(tabs=[
            Panel(child=column([self.venue_points_figure,
                                row([self.sound_field_figure,
                                     self.reference_sound_field_figure])]),
                  title="Frequency Responses at Audience Positions"),
            Panel(child=pHom, title="Homogeneity"),
            Panel(child=row([self.histogram_figure,
                             self.reference_histogram_figure]),
                  title="Bar Chart"),
            Panel(child=column([row([self.beam_frequency_select,
                                     column([self.beam_label,
                                             self.beam_checkbox])]),
                                pBeam, pBeam_ref]),
                  title="Beam Plots")])
        # callbacks
        self.beam_frequency_select.on_change("value", self.update_beam_plot)
        self.beam_checkbox.on_change("active", self.update_beam_plot)

    @staticmethod
    def _response_figure(title: str):
        """Figure of the frequency responses at the venue points."""
        fig = figure(title=title, x_axis_type="log", x_range=(15, 24000),
                     y_range=(0, 100), x_axis_label="f in kHz",
                     y_axis_label="SPL in dB",
                     tools="box_zoom,pan,wheel_zoom,reset,tap,hover,save",
                     toolbar_sticky=False,
                     tooltips=[("Index", "$index"),
                               ("Frequency in Hz", "$x"),
                               ("SPL in dB", "$y")], width=610)
        fig.toolbar.logo = None
        fig.xaxis.ticker = F_TICKS
        fig.xaxis.major_label_overrides = F_TICKS_OVERRIDE
        return fig

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

    def _beam_spl(self, mapper, tilt_angles, f, ind_f):
        """Beam plot of one array at one frequency, clipped to 50..100 dB.

        Parameters
        ----------
        mapper : LinearColorMapper [out]
            Color mapper of the beam plot; set to the SPL range.
        tilt_angles : array
            Tilt angles of the array in radian.
        f : float
            Frequency in Hz.
        ind_f : int
            Index of the frequency in the directivity data.

        Returns
        -------
        p_SPL : ndarray
            Sound pressure levels on the mesh.
        """
        st = self.app.state
        S = st.sound_field_config
        p_SPL, X, Y = beam_plot_spl(st.config, tilt_angles, st.ranges, f,
                                   np.array(S.measured_directivity_data[:, ind_f]),
                                   np.array(S.measured_angles_deg[:, 1]))
        high_ind = np.argwhere(p_SPL > 100)
        low_ind = np.argwhere(p_SPL < 50)
        for n in range(np.shape(high_ind)[0]):
            p_SPL[high_ind[n, 0], high_ind[n, 1]] = 100
        for n in range(np.shape(low_ind)[0]):
            p_SPL[low_ind[n, 0], low_ind[n, 1]] = 50
        mapper.low = np.amin(p_SPL)
        mapper.high = np.amax(p_SPL)
        return p_SPL

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
            f = float(self.beam_frequency_select.value)
            # index of the frequency (for measured loudspeaker data)
            ind_f = (np.abs(f-st.sound_field_config.frequencies).argmin())
            f_int = int(self.beam_frequency_select.value)
            for n in self.beam_checkbox.active:
                if n == 0 and 0 not in old and (0 in new or f_int in new):
                    p_SPL = self._beam_spl(self.beam_mapper,
                                           st.result.geometry.tilt_angles, f,
                                           ind_f)
                    self.beam_source.data.update(dict(
                        SPL=[p_SPL], x=[rng.p_x[0]], y=[rng.p_y[0]],
                        dw=[rng.p_x[1]-rng.p_x[0]],
                        dh=[rng.p_y[1]-rng.p_y[0]]))
                elif n == 1 and 1 not in old and (1 in new or f_int in new):
                    p_SPL_ref = self._beam_spl(self.reference_beam_mapper,
                                               st.reference_array.tilt_angles, f,
                                               ind_f)
                    self.beam_source.data.update(dict(
                        SPL_ref=[p_SPL_ref], x=[rng.p_x[0]], y=[rng.p_y[0]],
                        dw=[rng.p_x[1]-rng.p_x[0]],
                        dh=[rng.p_y[1]-rng.p_y[0]]))
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
        Tech_res = result.optimized.evaluation
        Tech_res_ref = result.reference.evaluation
        f, p_SPL, p_SPL_ref = [], [], []
        if np.shape(new)[0] >= 1:
            for n in range(np.shape(new)[0]):
                f.append(Tech_res.plot_frequencies[new[n]])
                p_SPL.append(Tech_res.plot_spl_db[new[n]])
                p_SPL_ref.append(Tech_res_ref.plot_spl_db[new[n]])
            self.sound_field_source.data = dict(
                f=f, p_SPL=p_SPL, foo=list(np.arange(np.shape(p_SPL)[0])),
                p_SPL_ref=p_SPL_ref)
        else:
            self.sound_field_source.data = dict(
                f=Tech_res.plot_frequencies, p_SPL=Tech_res.plot_spl_db,
                foo=list(np.arange(np.shape(Tech_res.plot_spl_db)[0])),
                p_SPL_ref=Tech_res_ref.plot_spl_db)
