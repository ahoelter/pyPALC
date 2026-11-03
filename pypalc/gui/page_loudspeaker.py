"""Page 2 of the GUI: loudspeaker configuration.

author: Arne Hoelter
"""
from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING

import numpy as np
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, LayoutDOM, Range1d
from bokeh.models.widgets import FileInput, Panel, Select, Tabs, TextInput
from bokeh.plotting import figure

from pypalc.core.sound_field import (calc_directivity, init_dir_plot,
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
        p_SPL = list(zip([0, 0]))
        self.directivity_source = ColumnDataSource(data=dict(
            amplitude_100=p_SPL, amplitude_1k=p_SPL, amplitude_10k=p_SPL,
            degree=p_SPL))
        self.directivity_figure = figure(
            title="Loudspeaker Directivity for Selected Frequencies",
            toolbar_sticky=False,
            tools="box_zoom,pan,wheel_zoom,reset,hover,save",
            tooltips=[("Index", "$index"), ("Amplitude in dB", "$y"),
                      ("Angle in deg", "$x")],
            x_axis_label="Angle in deg", y_axis_label="Amplitude in dB",
            x_range=Range1d(-92, 92), height=550, width=690)
        pdir = self.directivity_figure
        pdir.toolbar.logo = None
        pdir.line('degree', 'amplitude_100', source=self.directivity_source,
                  legend_label='100 Hz', color='deepskyblue')
        pdir.line('degree', 'amplitude_1k', source=self.directivity_source,
                  legend_label='1 kHz', color='lime')
        pdir.line('degree', 'amplitude_10k', source=self.directivity_source,
                  legend_label='10 kHz', color='orange')
        pdir.legend.title = "Frequencies"
        pdir.legend.location = "bottom_right"
        pdir.legend.background_fill_alpha = 0.5
        pdir.xaxis.ticker = [-90, -60, -30, 0, 30, 60, 90]
        self.tabs = Tabs(tabs=[Panel(child=app.venue_figure,
                                     title="Venue Slice"),
                               Panel(child=pdir,
                                     title="Loudspeaker Directivity")])
        self.compute_modeled_directivity()
        # callbacks
        self.aperture_input.on_change("value", app.store_value)
        self.cabinet_height_input.on_change("value", app.store_value)
        self.directivity_select.on_change("value", app.store_value)
        self.directivity_upload.on_change("value", self.upload_directivity)

    def layout(self) -> LayoutDOM:
        """Body of the page below the header.

        Returns
        -------
        LayoutDOM
        """
        col0 = column([self.aperture_input, self.cabinet_height_input],
                      width=270)
        col1 = column([self.directivity_select, self.directivity_upload,
                       self.app.config_error_info], width=270)
        col2 = column([self.app.venue_axes_row(), self.tabs], width=690)
        return row([col0, col1, col2])

    def compute_modeled_directivity(self) -> None:
        """Compute and plot the modeled directivity of the configuration."""
        config = self.app.state.config
        sf_config = self.app.state.sound_field_config
        c0, alpha, beta_deg, beta, f, omega, dir_meas, dir_meas_deg, H_post = \
            init_dir_plot()
        for n in range(np.shape(f)[0]):
            H_post[:, n] = calc_directivity(config.directivity, alpha,
                                            config.cabinet_height, beta, omega[n],
                                            c0, f, dir_meas[:, n],
                                            dir_meas_deg[:, 1], n)
        setattr(sf_config, 'f', f)
        sf_config.get_plot_dir_amp(H_post, [100, 1000, 10000])
        self.directivity_source.data = dict(
            amplitude_100=sf_config.plot_dir_amp[0],
            amplitude_1k=sf_config.plot_dir_amp[1],
            amplitude_10k=sf_config.plot_dir_amp[2], degree=list(beta_deg))

    def _show_measured_directivity(self) -> None:
        """Plot the uploaded measured directivity."""
        sf_config = self.app.state.sound_field_config
        self.directivity_source.data = dict(
            amplitude_100=sf_config.plot_dir_amp_meas[0],
            amplitude_1k=sf_config.plot_dir_amp_meas[1],
            amplitude_10k=sf_config.plot_dir_amp_meas[2],
            degree=sf_config.plot_beta_meas)

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
