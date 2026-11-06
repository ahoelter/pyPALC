"""Page 2 of the GUI: loudspeaker configuration.

The steps 9 to 11 are in the cards "Cabinet" and "Directivity"; the card
"Figures" shows the venue figure of the app
(:attr:`~.app.PalcApp.venue_figure`) and the directivity figure in tabs.

author: Arne Hoelter
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
import panel as pn
import panel_material_ui as pmui
from bokeh.models import ColumnDataSource, Range1d
from bokeh.plotting import figure

from pypalc.core.sound_field import (cabinet_directivity,
                                     initial_directivity_plot,
                                     parse_measured_directivity)
from pypalc.gui import CALLBACK_ERRORS, skipped_while_applying_setup
from pypalc.gui.ranges import VENUE_MAX_HEIGHT
from pypalc.gui.widgets import (caption, card, page_row, select,
                                text_input, watch)

if TYPE_CHECKING:
    from .app import PalcApp

logger = logging.getLogger(__name__)

#: Modeled directivities; the other option is measured loudspeaker data.
MODELED_DIRECTIVITIES = ('Circular Piston', 'Line Piston',
                         'Combined Circular/Line')
#: Shown label -> value of the directivity selection.
DIRECTIVITY_OPTIONS = {
    'Modeled: Circular Piston': 'Circular Piston',
    'Modeled: Line Piston': 'Line Piston',
    'Modeled: Combined Circular/Line': 'Combined Circular/Line',
    'Measured Loudspeaker Data': 'Measured Loudspeaker Data'}
#: Explains the file of measured directivity data.
_MEASURED_CAPTION = ('CSV file: the frequencies in Hz in the first row, the '
                     'angles in deg in the first column and the complex '
                     'directivity (imaginary unit j or i) in the other '
                     'cells.')


class LoudspeakerPage:
    """Page 2: aperture angle, cabinet height and directivity.

    Parameters
    ----------
    app : PalcApp
        The session the page belongs to.
    """

    def __init__(self, app: PalcApp) -> None:
        self.app = app
        self.aperture_input = text_input('9. Aperture angle in deg', '3.0')
        """Initial aperture angle of the cabinets in degree."""
        self.cabinet_height_input = text_input(
            '10. Loudspeaker height in m', '0.2')
        self.directivity_select = select(
            '11. Directivity', dict(DIRECTIVITY_OPTIONS), 'Circular Piston')
        self.directivity_upload = pmui.FileInput(
            label='Upload measured data', accept='.csv', visible=False,
            color='primary', variant='outlined', icon='upload_file',
            sizing_mode='stretch_width', margin=(4, 8))
        """Upload of measured (complex) directivity data; shown while
        measured data are selected."""
        self._layout: Optional[pn.viewable.Viewable] = None
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
            x_range=Range1d(-92, 92), sizing_mode='stretch_width',
            height=VENUE_MAX_HEIGHT)
        """Directivity figure; it follows the width of the window."""
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
        self.tabs: Optional[pmui.Tabs] = None
        """Tabs of the venue figure and the directivity figure (built with
        the layout, when the document has its theme)."""
        self.compute_modeled_directivity()
        # callbacks
        for widget in (self.aperture_input, self.cabinet_height_input,
                       self.directivity_select):
            watch(widget, lambda event: app.store_value(
                'value', event.old, event.new), 'value')
        watch(self.directivity_upload, lambda event: (
            self.upload_directivity('value', event.old, event.new)), 'value')

    def layout(self) -> pn.FlexBox:
        """Body of the page below the header.

        Built at the first call, then the same object: its widgets show
        every change in place.

        Returns
        -------
        panel.FlexBox
            The cards "Cabinet" and "Directivity" beside (or, on narrow
            windows, above) the card "Figures".
        """
        if self._layout is None:
            app = self.app
            self.tabs = pmui.Tabs(
                ('Venue slice', pn.Column(
                    app.venue_axes_row(),
                    app.bokeh_pane(app.venue_figure,
                                   sizing_mode='stretch_width'),
                    sizing_mode='stretch_width', margin=(8, 0, 0, 0))),
                ('Directivity', app.bokeh_pane(
                    self.directivity_figure, sizing_mode='stretch_width',
                    margin=(8, 0, 0, 0))),
                sizing_mode='stretch_width', margin=(0, 8))
            cabinet_card = card('Cabinet', self.aperture_input,
                                self.cabinet_height_input)
            directivity_card = card(
                'Directivity', self.directivity_select,
                caption(_MEASURED_CAPTION,
                        visible=self.directivity_upload.param.visible),
                self.directivity_upload, app.config_error_info)
            self._layout = page_row(
                pn.Column(cabinet_card, directivity_card, margin=0),
                # the tabs name the figures: no header, so that the figures
                # end inside the window like those of pages 1 and 3
                card('Figures', self.tabs, hide_header=True))
        return self._layout

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
        (``measured_directivity_csv``) for the calculation and the error
        text is cleared; for an unreadable file the error text is shown.
        Triggered by the file upload.

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
        if not new:
            return
        try:
            self.read_measured_directivity(new.decode('utf-8'))
        except CALLBACK_ERRORS:  # also UnicodeDecodeError (a ValueError)
            logger.exception('Could not read the measured loudspeaker data')
            self.app.config_error_info.object = 'Error: Could not read Measured Loudspeaker Data'
            return
        self.app.config_error_info.object = ''
