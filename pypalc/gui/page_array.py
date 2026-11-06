"""Page 3 of the GUI: array configuration and reference array.

The steps 12 to 17 are in the cards "Cabinets" and "Discrete tilt angles",
the reference array (with the table of its tilt angles) in the card
"Reference array"; the card "Venue slice" shows the venue figure of the app
(:attr:`~.app.PalcApp.venue_figure`) with its axis mode selection.

author: Arne Hoelter
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
import panel as pn
import panel_material_ui as pmui
from bokeh.models import ColumnDataSource

from pypalc.core.geometry import reference_tilt_angles_deg
from pypalc.gui import CALLBACK_ERRORS, skipped_while_applying_setup
from pypalc.gui.widgets import (caption, card, page_row, select, table,
                                text_input, watch)

if TYPE_CHECKING:
    from .app import PalcApp

logger = logging.getLogger(__name__)

#: Labels of the reference type selection -> reference array types of
#: :mod:`pypalc.core`.
REFERENCE_TYPES = {'Straight': 'straight', 'Progressive': 'progressive',
                   'Arc': 'arc', 'User Defined': 'user_defined'}
#: Labels of the reference array inputs.
ARRAY_TILT_LABEL = 'Array tilt angle in deg'
START_LABEL = 'Start angle in deg'
INTER_CABINET_LABEL = 'Inter cabinet angle in deg'
FINAL_LABEL = 'Final angle in deg'


class ArrayPage:
    """Page 3: cabinets, mounting position, gap, discrete angles and the
    reference array.

    Parameters
    ----------
    app : PalcApp
        The session the page belongs to.
    """

    def __init__(self, app: PalcApp) -> None:
        self.app = app
        config = app.state.config
        self.cabinet_count_input = text_input('12. Number of cabinets', '4')
        self.top_x_input = text_input('13. Top x in m', '0')
        """Front position of the highest cabinet in x-direction."""
        self.top_y_input = text_input('14. Top y in m', '2')
        """Mounting height (top front of the highest cabinet)."""
        self.cabinet_gap_input = text_input('15. Gap between cabinets in m',
                                            '0')
        self.use_discrete_angles_select = select(
            '16. Use discrete tilt angles?', ['No', 'Yes'], 'No')
        self.discrete_angles_input = text_input(
            '17. Discrete tilt angles in deg', '', disabled=True)
        self.reference_type_select = select(
            'Type', ['Straight', 'Progressive', 'Arc', 'User Defined'],
            'Straight')
        self.reference_discrete_angles_select = select(
            'Use angles of 17.?', ['Yes', 'No'], 'No', disabled=True,
            visible=False)
        """Whether the reference array uses the discrete angles of (17.)."""
        self.reference_start_input = text_input(ARRAY_TILT_LABEL, '0')
        """Angle of the highest cabinet of the reference array."""
        self.reference_step_stop_input = text_input(
            INTER_CABINET_LABEL, '0', disabled=True)
        """Inter cabinet angle or final angle of the reference array."""
        self.reference_user_angles_input = pmui.TextAreaInput(
            label='Tilt angles in deg', value='', visible=False, rows=2,
            sizing_mode='stretch_width', size='small')
        self.reference_angles_source = ColumnDataSource(data=dict(
            tilt_angle_deg=np.array([np.zeros(config.cabinet_count)]),
            cabinet=np.array([np.linspace(1, 4, num=config.cabinet_count)])))
        """Tilt angles of the reference array."""
        self._layout: Optional[pn.viewable.Viewable] = None
        # callbacks
        for widget in (self.top_x_input, self.top_y_input,
                       self.cabinet_count_input, self.cabinet_gap_input):
            watch(widget, lambda event: app.store_value(
                'value', event.old, event.new), 'value')
        watch(self.use_discrete_angles_select,
              lambda event: self.choose_discrete_angles(
                  'value', event.old, event.new), 'value')
        watch(self.discrete_angles_input,
              lambda event: self.read_discrete_angles(
                  'value', event.old, event.new), 'value')
        watch(self.discrete_angles_input,
              lambda event: self.reference_input_changed(
                  'value', event.old, event.new), 'value')
        watch(self.reference_type_select,
              lambda event: self.choose_reference_type(
                  'value', event.old, event.new), 'value')
        for widget in (self.reference_start_input,
                       self.reference_step_stop_input,
                       self.reference_discrete_angles_select,
                       self.reference_user_angles_input):
            watch(widget, lambda event: self.reference_input_changed(
                'value', event.old, event.new), 'value')

    def layout(self) -> pn.FlexBox:
        """Body of the page below the header.

        Built at the first call, then the same object: its widgets show
        every change in place.

        Returns
        -------
        panel.FlexBox
            The cards "Cabinets", "Discrete tilt angles" and "Reference
            array" beside (or, on narrow windows, above) the card "Venue
            slice".
        """
        if self._layout is None:
            cabinets_card = card(
                'Cabinets', self.cabinet_count_input,
                caption('Position of the top front edge of the uppermost '
                        'cabinet:', margin=(6, 8, 10, 8)),
                pn.GridBox(self.top_x_input, self.top_y_input, ncols=2,
                           sizing_mode='stretch_width'),
                self.cabinet_gap_input)
            discrete_card = card(
                'Discrete tilt angles', self.use_discrete_angles_select,
                self.discrete_angles_input,
                caption("Separate the angles by ','."),
                self.app.config_error_info)
            reference_card = card(
                'Reference array',
                caption('Array that the PALC result is compared with.',
                        margin=(0, 8, 6, 8)),
                self.reference_type_select,
                self.reference_discrete_angles_select,
                self.reference_start_input, self.reference_step_stop_input,
                self.reference_user_angles_input,
                caption("Separate the angles by ','.", visible=(
                    self.reference_user_angles_input.param.visible)),
                self._reference_table())
            figure_card = card(
                'Venue slice', self.app.venue_axes_row(),
                self.app.bokeh_pane(self.app.venue_figure,
                                    sizing_mode='stretch_width'))
            self._layout = page_row(
                pn.Column(cabinets_card, discrete_card, reference_card,
                          margin=0), figure_card)
        return self._layout

    def _reference_frame(self) -> pd.DataFrame:
        """The reference tilt angles as shown in the table (one row per
        cabinet; the start-up data hold all cabinets in one row)."""
        data = self.reference_angles_source.data
        return pd.DataFrame({
            'cabinet': np.ravel(data['cabinet']).astype(int),
            'tilt_angle_deg': np.ravel(data['tilt_angle_deg'])})

    def _reference_table(self) -> pn.widgets.Tabulator:
        """Table of the reference tilt angles that follows
        :attr:`reference_angles_source`."""
        reference_table = table(
            self._reference_frame(),
            titles={'cabinet': 'LS', 'tilt_angle_deg': 'Tilt angle in deg'},
            widths={'cabinet': 56})
        self.reference_angles_source.on_change(
            'data', lambda attr, old, new: setattr(
                reference_table, 'value', self._reference_frame()))
        return reference_table

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    @skipped_while_applying_setup
    def choose_discrete_angles(self, attr, old, new) -> None:
        """Enable the discrete tilt angles (with a default set) or disable
        them. Triggered by "Use Discrete Tilt Angles?".

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
        config = self.app.state.config
        if self.use_discrete_angles_select.value == 'Yes':
            self.discrete_angles_input.value = '0,0.5,1,2,3.5,5,7,10'
            self.discrete_angles_input.disabled = False
            config.store_config('use_discrete_angles', True)
            self.reference_discrete_angles_select.disabled = False
        elif self.use_discrete_angles_select.value == 'No':
            self.discrete_angles_input.value = ''
            self.discrete_angles_input.disabled = True
            config.store_config('use_discrete_angles', False)
            self.reference_discrete_angles_select.disabled = True
            self.reference_discrete_angles_select.value = "No"
        self.read_discrete_angles(attr, old, new)

    @skipped_while_applying_setup
    def read_discrete_angles(self, attr, old, new) -> None:
        """Read the set of discrete tilt angles (',' separates values).

        Parameters
        ----------
        attr : str
            Changed attribute.
        old, new : str
            Old and new text.

        Returns
        -------
        None.
        """
        self.store_discrete_angles()

    def store_discrete_angles(self) -> None:
        """Store the discrete tilt angles of the input in the configuration
        (body of :meth:`read_discrete_angles`)."""
        config = self.app.state.config
        error_info = self.app.config_error_info
        # discrete angles were disabled
        # the legacy value check
        if config.use_discrete_angles == False:  # noqa: E712
            error_info.object = ''
            config.discrete_tilt_angles = []
            return
        try:
            config.discrete_tilt_angles = np.array(
                [float(s) for s in self.discrete_angles_input.value.split(',')]
            ) * (np.pi/180)
            error_info.object = ''
        except ValueError:
            error_info.object = 'Error: Could not process the input'

    @skipped_while_applying_setup
    def reference_input_changed(self, attr, old, new) -> None:
        """Compute the reference angles after an input of them changed.

        Parameters
        ----------
        attr : str
            Changed attribute (value).
        old, new : str
            Old and new value.

        Returns
        -------
        None.
        """
        self.update_reference_angles()

    @skipped_while_applying_setup
    def choose_reference_type(self, attr, old, new) -> None:
        """Arrange the reference array widgets for the selected type and
        compute the reference angles. Triggered by the reference type.

        Parameters
        ----------
        attr : str
            Changed attribute (value).
        old, new : str
            Old and new reference array type.

        Returns
        -------
        None.
        """
        self.arrange_reference_widgets()
        self.update_reference_angles()

    def arrange_reference_widgets(self) -> None:
        """Show, hide, enable and disable the reference array widgets."""
        ref_type = self.reference_type_select.value
        start, step_stop = self.reference_start_input, \
            self.reference_step_stop_input
        discrete, user = self.reference_discrete_angles_select, \
            self.reference_user_angles_input
        if ref_type in ['Straight', 'Progressive', 'Arc']:
            step_stop.visible, start.visible = True, True
            user.visible = False
            if ref_type in ['Straight']:
                step_stop.value, step_stop.disabled = "0", True
                start.label = ARRAY_TILT_LABEL
                discrete.visible = False
                step_stop.label = INTER_CABINET_LABEL
            elif ref_type in ['Progressive']:
                step_stop.label = FINAL_LABEL
                start.label = START_LABEL
                step_stop.disabled, discrete.visible = False, True
            elif ref_type in ['Arc']:
                step_stop.label = INTER_CABINET_LABEL
                start.label = START_LABEL
                step_stop.disabled, discrete.visible = False, True
        elif ref_type in ['User Defined']:
            step_stop.visible, start.visible = False, False
            discrete.visible, user.visible = False, True

    def _compute_reference_angles(self) -> None:
        """Tilt angles of the reference array from the widgets."""
        config = self.app.state.config
        ref_type = REFERENCE_TYPES[self.reference_type_select.value]
        if ref_type == 'user_defined':
            user_angles = [float(s) for s in
                           self.reference_user_angles_input.value.split(',')]
            start, step_stop = 0, 0
        else:
            user_angles = ()
            start = float(self.reference_start_input.value)
            step_stop = float(self.reference_step_stop_input.value) \
                if ref_type != 'straight' else 0
        self.app.state.reference_array.tilt_angles_deg = \
            reference_tilt_angles_deg(
                ref_type, config.cabinet_count, config.discrete_tilt_angles,
                start, step_stop,
                str(self.reference_discrete_angles_select.value) == "Yes",
                user_angles)

    def update_reference_angles(self, cabinet_count_changed: bool = False
                                ) -> None:
        """Compute the reference array tilt angles and show them.

        Parameters
        ----------
        cabinet_count_changed : bool, optional
            True if the number of cabinets has changed; then the error
            text is kept. The default is False.

        Returns
        -------
        None.
        """
        config = self.app.state.config
        reference = self.app.state.reference_array
        try:
            self._compute_reference_angles()
            # in degree for the table
            self.reference_angles_source.data = dict(
                tilt_angle_deg=np.round(reference.tilt_angles_deg,
                                        decimals=2),
                cabinet=np.linspace(1, config.cabinet_count,
                                    config.cabinet_count))
            # in radian
            reference.tilt_angles = reference.tilt_angles_deg * (np.pi / 180)
            if cabinet_count_changed == False:  # noqa: E712
                self.app.config_error_info.object = ''
        except CALLBACK_ERRORS:
            logger.debug('Could not compute the reference array',
                         exc_info=True)
            self.app.config_error_info.object = 'Error: Could not process the input'
            self.reference_start_input.value = '0'
            self.reference_step_stop_input.value = '0'
