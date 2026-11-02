"""Page 3 of the GUI: array configuration and reference array.

author: Arne Hoelter
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, LayoutDOM
from bokeh.models.widgets import (DataTable, Paragraph, Select, TableColumn,
                                  TextAreaInput, TextInput)

from pypalc.core.geometry import reference_tilt_angles_deg
from pypalc.gui import CALLBACK_ERRORS
from pypalc.gui.page_venue import RULE, RULE_LONG

if TYPE_CHECKING:
    from .app import PalcApp

logger = logging.getLogger(__name__)

#: Labels of the reference type selection -> reference array types of
#: :mod:`pypalc.core`.
REFERENCE_TYPES = {'Straight': 'straight', 'Progressive': 'progressive',
                   'Arc': 'arc', 'User Defined': 'user_defined'}


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
        self.cabinet_count_input = TextInput(
            value="4", title="12. Number of Loudspeaker Cabinets", width=255,
            height=50)
        self.top_x_input = TextInput(
            value="0", title="13. Uppermost Cabinet: x in m", width=255,
            height=50)
        """Front position of the highest cabinet in x-direction."""
        self.top_y_input = TextInput(
            value="2", title="14. Uppermost Cabinet: y in m", width=255,
            height=50)
        """Mounting height (top front of the highest cabinet)."""
        self.cabinet_gap_input = TextInput(
            value="0", title="15. Gap between Cabinets in m", width=255,
            height=50)
        self.use_discrete_angles_select = Select(
            title="16. Use Discrete Tilt Angles?", value="No",
            options=["No", "Yes"], width=255, height=50)
        self.discrete_angles_input = TextInput(
            value='', disabled=True, width=255, height=50,
            title="17. Discrete Tilt Angles in deg (',': separator)")
        self.reference_type_select = Select(
            title="Reference Array: Type", value="Straight",
            options=["Straight", "Progressive", "Arc", "User Defined"],
            width=255, height=50)
        self.reference_discrete_angles_select = Select(
            title="Reference Array: Use Angles of (17.)?", value="No",
            options=["Yes", "No"], disabled=True, visible=False, width=255,
            height=50)
        """Whether the reference array uses the discrete angles of (17.)."""
        self.reference_start_input = TextInput(
            value="0", width=255, height=50,
            title="Reference Array: Array Tilt Angle in deg")
        """Angle of the highest cabinet of the reference array."""
        self.reference_step_stop_input = TextInput(
            value="0", disabled=True, width=255, height=50,
            title="Reference Array: Inter Cabinet Angle in deg")
        """Inter cabinet angle or final angle of the reference array."""
        self.reference_user_angles_input = TextAreaInput(
            value="", visible=False, cols=2, rows=3, width=255, height=70,
            title="User Defined Tilt Angles in deg (',': separator)")
        self.spacer = Paragraph(text="", height=50)
        self.rule_left = Paragraph(text=RULE_LONG)
        self.rule_right = Paragraph(text=RULE)
        self.reference_angles_source = ColumnDataSource(data=dict(
            gamma_ref=np.array([np.zeros(config.N)]),
            LS=np.array([np.linspace(1, 4, num=config.N)])))
        """Tilt angles of the reference array."""
        self.reference_table = DataTable(
            source=self.reference_angles_source,
            columns=[TableColumn(field="LS", title="LS", width=20),
                     TableColumn(field="gamma_ref", title="Tilt Angle in deg",
                                 width=140)],
            index_position=None, width=170, fit_columns=False)
        # callbacks
        self.top_x_input.on_change("value", app.store_value)
        self.top_y_input.on_change("value", app.store_value)
        self.cabinet_count_input.on_change("value", app.store_value)
        self.cabinet_gap_input.on_change("value", app.store_value)
        self.use_discrete_angles_select.on_change(
            "value", self.choose_discrete_angles)
        self.discrete_angles_input.on_change("value",
                                             self.read_discrete_angles)
        self.discrete_angles_input.on_change(
            "value", lambda attr, old, new: self.update_reference_angles())
        self.reference_type_select.on_change("value",
                                             self.choose_reference_type)
        for widget in (self.reference_start_input,
                       self.reference_step_stop_input,
                       self.reference_discrete_angles_select,
                       self.reference_user_angles_input):
            widget.on_change(
                "value", lambda attr, old, new: self.update_reference_angles())

    def layout(self) -> LayoutDOM:
        """Body of the page below the header.

        Returns
        -------
        LayoutDOM
        """
        col0 = column([self.cabinet_count_input, self.top_x_input,
                       self.top_y_input, self.spacer, self.rule_left,
                       self.reference_type_select,
                       self.reference_discrete_angles_select,
                       self.reference_user_angles_input], width=270)
        col1 = column([self.cabinet_gap_input,
                       self.use_discrete_angles_select,
                       self.discrete_angles_input, self.app.config_error_info,
                       self.rule_right, self.reference_start_input,
                       self.reference_step_stop_input, self.reference_table],
                      width=270)
        col2 = column([self.app.venue_axes_row(), self.app.venue_figure],
                      width=690)
        return row([col0, col1, col2])

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

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
            config.store_config('use_gamma_LSA', True)
            self.reference_discrete_angles_select.disabled = False
        elif self.use_discrete_angles_select.value == 'No':
            self.discrete_angles_input.value = ''
            self.discrete_angles_input.disabled = True
            config.store_config('use_gamma_LSA', False)
            self.reference_discrete_angles_select.disabled = True
            self.reference_discrete_angles_select.value = "No"
        self.read_discrete_angles(attr, old, new)

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
        config = self.app.state.config
        error_info = self.app.config_error_info
        # discrete angles were disabled
        if config.use_gamma_LSA == False:  # noqa: E712 (legacy value check)
            error_info.text = ''
            config.gamma_LSA = []
            return
        try:
            config.gamma_LSA = np.array(
                [float(s) for s in self.discrete_angles_input.value.split(',')]
            ) * (np.pi/180)
            error_info.text = ''
        except ValueError:
            error_info.text = 'Error: Could not process the input'

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
        self._arrange_reference_widgets()
        self.update_reference_angles()

    def _arrange_reference_widgets(self) -> None:
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
                start.title = "Reference Array: Array Tilt Angle in deg"
                discrete.visible = False
                step_stop.title = "Reference Array: Inter Cabinet Angle in deg"
            elif ref_type in ['Progressive']:
                step_stop.title = "Reference Array: Final Angle in deg"
                start.title = "Reference Array: Start Angle in deg"
                step_stop.disabled, discrete.visible = False, True
            elif ref_type in ['Arc']:
                step_stop.title = "Reference Array: Inter Cabinet Angle in deg"
                start.title = "Reference Array: Start Angle in deg"
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
        self.app.state.reference_array.gamma_tilt_deg = \
            reference_tilt_angles_deg(
                ref_type, config.N, config.gamma_LSA, start, step_stop,
                str(self.reference_discrete_angles_select.value) == "Yes",
                user_angles)

    def update_reference_angles(self, chg_n: bool = False) -> None:
        """Compute the reference array tilt angles and show them.

        Parameters
        ----------
        chg_n : bool, optional
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
                gamma_ref=np.round(reference.gamma_tilt_deg, decimals=2),
                LS=np.linspace(1, config.N, config.N))
            # in radian
            reference.gamma_n = reference.gamma_tilt_deg * (np.pi / 180)
            if chg_n == False:  # noqa: E712
                self.app.config_error_info.text = ''
        except CALLBACK_ERRORS:
            logger.debug('Could not compute the reference array',
                         exc_info=True)
            self.app.config_error_info.text = 'Error: Could not process the input'
            self.reference_start_input.value = '0'
            self.reference_step_stop_input.value = '0'
