"""Page 2 of the GUI: loudspeaker configuration.

author: Arne Hoelter
"""
from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING

import numpy as np
import panel as pn
from bokeh.layouts import column, row
from bokeh.models import (ColumnDataSource, FileInput, Range1d,
                          Select, TabPanel, Tabs, TextInput)
from bokeh.plotting import figure

from pypalc.core.sound_field import (cabinet_directivity,
                                     initial_directivity_plot,
                                     parse_measured_directivity)
from pypalc.gui import CALLBACK_ERRORS, skipped_while_applying_setup

if TYPE_CHECKING:
    from .app import PalcApp

logger = logging.getLogger(__name__)

#: Modeled directivities; the other option is measured loudspeaker data.
MODELED_DIRECTIVITIES = ('Circular Piston', 'Line Piston',
                         'Combined Circular/Line')


class LoudspeakerPage:
    """Page 2: aperture angle, cabinet height and directivity.

    Parameters
    ----------
    app : PalcApp
        The session the page belongs to.
    """

    def __init__(self, app: PalcApp) -> None:
        self.app = app
        self.aperture_input = TextInput(
            value="3.0", title="9. PALC Aperture Angle in deg", width=255,
            height=50)
        """Initial aperture angle of the cabinets in degree."""
        self.cabinet_height_input = TextInput(
            value="0.2", title="10. Loudspeaker Height in m", width=255)
        self.directivity_select = Select(
            title="11. Loudspeaker Directivity", value="Circular Piston",
            options=[("Circular Piston", "Modeled: Circular Piston"),
                     ("Line Piston", "Modeled: Line Piston"),
                     ("Combined Circular/Line",
                      "Modeled: Combined Circular/Line"),
                     "Measured Loudspeaker Data"], width=255, height=50)
        self.directivity_upload = FileInput(accept=".csv", visible=False)
        """Upload of measured (complex) directivity data."""
        # directivity of 100 Hz, 1 kHz and 10 kHz used in the sound field
        # prediction
        empty = list(zip([0, 0]))
        self.directivity_source = ColumnDataSource(data=dict(
            amplitude_100hz_db=empty, amplitude_1khz_db=empty,
            amplitude_10khz_db=empty, angle_deg=empty))
        self.directivity_figure = figure(
            title="Loudspeaker Directivity for Selected Frequencies",
            tools="box_zoom,pan,wheel_zoom,reset,hover,save",
            tooltips=[("Index", "$index"), ("Amplitude in dB", "$y"),
                      ("Angle in deg", "$x")],
            x_axis_label="Angle in deg", y_axis_label="Amplitude in dB",
            x_range=Range1d(-92, 92), height=550, width=690)
        directivity_figure = self.directivity_figure
        directivity_figure.toolbar.logo = None
        directivity_figure.line('angle_deg', 'amplitude_100hz_db',
                                source=self.directivity_source,
                                legend_label='100 Hz', color='deepskyblue')
        directivity_figure.line('angle_deg', 'amplitude_1khz_db',
                                source=self.directivity_source,
                                legend_label='1 kHz', color='lime')
        directivity_figure.line('angle_deg', 'amplitude_10khz_db',
                                source=self.directivity_source,
                                legend_label='10 kHz', color='orange')
        directivity_figure.legend.title = "Frequencies"
        directivity_figure.legend.location = "bottom_right"
        directivity_figure.legend.background_fill_alpha = 0.5
        directivity_figure.xaxis.ticker = [-90, -60, -30, 0, 30, 60, 90]
        self.tabs = Tabs(tabs=[TabPanel(child=app.venue_figure,
                                        title="Venue Slice"),
                               TabPanel(child=directivity_figure,
                                        title="Loudspeaker Directivity")],
                         sizing_mode='stretch_width')
        self.compute_modeled_directivity()
        # callbacks
        self.aperture_input.on_change("value", app.store_value)
        self.cabinet_height_input.on_change("value", app.store_value)
        self.directivity_select.on_change("value", app.store_value)
        self.directivity_upload.on_change("value", self.upload_directivity)

    def layout(self) -> pn.Row:
        """Body of the page below the header.

        The Bokeh widgets of the page are shown in Bokeh panes
        (:meth:`~.app.PalcApp.bokeh_pane`), beside the axis mode selection
        of the venue figure, which is a Panel widget.

        Returns
        -------
        panel.Row
        """
        col0 = column([self.aperture_input, self.cabinet_height_input],
                      width=270)
        col1 = column([self.directivity_select, self.directivity_upload,
                       self.app.config_error_info], width=270)
        col2 = pn.Column(self.app.venue_axes_row(),
                         self.app.bokeh_pane(self.tabs,
                                             sizing_mode='stretch_width'),
                         width=690)
        return pn.Row(self.app.bokeh_pane(row([col0, col1])), col2)

    def compute_modeled_directivity(self) -> None:
        """Compute and plot the modeled directivity of the configuration."""
        config = self.app.state.config
        sound_field_config = self.app.state.sound_field_config
        (c0, alpha, angles_deg, angles, frequencies, omega,
         measured_directivity_data, measured_angles_deg, directivity) = \
            initial_directivity_plot()
        for n in range(np.shape(frequencies)[0]):
            directivity[:, n] = cabinet_directivity(
                config.directivity, alpha, config.cabinet_height, angles,
                omega[n], c0, frequencies, measured_directivity_data[:, n],
                measured_angles_deg[:, 1], n)
        setattr(sound_field_config, 'frequencies', frequencies)
        sound_field_config.set_plot_directivity(directivity,
                                                [100, 1000, 10000])
        self.directivity_source.data = dict(
            amplitude_100hz_db=sound_field_config.plot_directivity_db[0],
            amplitude_1khz_db=sound_field_config.plot_directivity_db[1],
            amplitude_10khz_db=sound_field_config.plot_directivity_db[2],
            angle_deg=list(angles_deg))

    def _show_measured_directivity(self) -> None:
        """Plot the uploaded measured directivity."""
        sound_field_config = self.app.state.sound_field_config
        measured_db = sound_field_config.plot_measured_directivity_db
        self.directivity_source.data = dict(
            amplitude_100hz_db=measured_db[0],
            amplitude_1khz_db=measured_db[1],
            amplitude_10khz_db=measured_db[2],
            angle_deg=sound_field_config.plot_measured_angles_deg)

    def update_directivity_plot(self) -> None:
        """Show the upload for measured data or plot the modeled directivity.

        Returns
        -------
        None.
        """
        if self.app.state.config.directivity in MODELED_DIRECTIVITIES:
            self.directivity_upload.visible = False
            self.compute_modeled_directivity()
        else:
            self.directivity_upload.visible = True
            self._show_measured_directivity()

    def read_measured_directivity(self, text: str) -> None:
        """Read measured directivity data, keep its text and plot it.

        Parameters
        ----------
        text : str
            Text of the CSV file.

        Returns
        -------
        None.
        """
        parse_measured_directivity(text, self.app.state.sound_field_config)
        self.app.state.measured_directivity_csv = text
        self._show_measured_directivity()

    @skipped_while_applying_setup
    def upload_directivity(self, attr, old, new) -> None:
        """Read uploaded measured directivity data and plot it.

        The CSV file holds the frequency bins in the first line and the
        angles in degree in the first column; the imaginary unit is "j" or
        "i". The text of a readable file is kept in the session state
        (``measured_directivity_csv``) for the calculation. Triggered by the
        file upload.

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
        try:
            self.read_measured_directivity(
                base64.b64decode(new).decode('utf-8'))
        except CALLBACK_ERRORS:
            logger.exception('Could not read the measured loudspeaker data')
            self.app.config_error_info.text = 'Error: Could not read Measured Loudspeaker Data'
