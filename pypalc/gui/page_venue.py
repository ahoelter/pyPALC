"""Page 1 of the GUI: draw the polygonal audience line (the venue).

author: Arne Hoelter
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, LayoutDOM
from bokeh.models.widgets import Button, Div, Paragraph, TextInput

from pypalc.core.errors import VenueError
from pypalc.core.pipeline import check_line
from pypalc.core.settings import AudienceLine
from pypalc.core.venue import get_slope_b

if TYPE_CHECKING:
    from .app import PalcApp

logger = logging.getLogger(__name__)

#: Separator lines between widget groups.
RULE_LONG = "______________________________________"
RULE = "___________________________________"

#: Shown if the discretization cannot be read.
_DISCRETIZATION_ERROR = 'Couldnt process discretization input. Use default: 5 points per m'


class VenuePage:
    """Page 1: draw the audience lines and create the PALC venue.

    Parameters
    ----------
    app : PalcApp
        The session the page belongs to.
    """

    def __init__(self, app: PalcApp) -> None:
        self.app = app
        # widgets
        self.start_x_input = TextInput(
            value="0", width=255,
            title="1. Start Position (x) of Audience Line in m")
        """Start point of an audience line in x-direction."""
        self.start_y_input = TextInput(
            value="0", width=255,
            title="4. Start Position (y) of Audience Line in m")
        """Start point of an audience line in y-direction."""
        self.end_x_input = TextInput(
            value="0", width=255,
            title="2. End Position (x) of Audience Line in m")
        """End point of an audience line in x-direction."""
        self.end_y_input = TextInput(
            value="0", width=255,
            title="5. End Position (y) of Audience Line in m")
        """End point of an audience line in y-direction."""
        self.points_per_meter_input = TextInput(
            value="1", width=255,
            title="3. Discretization: Number of Points per m")
        """Discretization of audience lines in points per m."""
        self.save_line_button = Button(label="6. Save Current Audience Line",
                                       button_type="success", width=255)
        self.remove_line_button = Button(label="7. Remove Last Audience Line",
                                         button_type="danger", width=255)
        self.insert_line_button = Button(
            label="6. Add Line before Selected Line", button_type="success",
            visible=False, width=255)
        """Shown instead of the save button while a line is selected."""
        self.remove_selected_line_button = Button(
            label="7. Remove Selected Audience Line", button_type="danger",
            visible=False, width=255)
        """Shown instead of the remove button while a line is selected."""
        self.check_venue_button = Button(label="8. Get PALC Venue Proposal",
                                         button_type="warning", width=255)
        """Creates the PALC compatible venue (also shown on page 4)."""
        self.venue_error_info = Div(text="", width=250, height=300)
        self.x_axis_title = Div(text="<b>x-Axis</b>")
        self.y_axis_title = Div(text="<b>y-Axis</b>")
        self.rule_x = Paragraph(text=RULE_LONG)
        self.rule_y = Paragraph(text=RULE)
        # sources
        x_init = np.array([0, 0]).astype(float)
        self.point_source = ColumnDataSource(data=dict(x=x_init, y=x_init))
        """Start and end point entered by the user."""
        self.lines_source = ColumnDataSource(data=dict(xlineplt=[],
                                                       ylineplt=[]))
        """The drawn audience lines."""
        self.non_audience_source = ColumnDataSource(data=dict(xnalplt=[],
                                                              ynalplt=[]))
        """Non-audience lines, e.g., gaps between audience lines."""
        self.sound_field_only_source = ColumnDataSource(data=dict(
            xsfpplt=[], ysfpplt=[]))
        """Audience lines only used in the sound field prediction."""
        self.tapped_line_source = ColumnDataSource(data=dict(tapped=[]))
        """Index of the tapped audience line."""
        self._build_venue_figure()
        # callbacks
        for widget in (self.start_x_input, self.start_y_input,
                       self.end_x_input, self.end_y_input):
            widget.on_change("value", self.update_point)
        self.save_line_button.on_click(self.save_line)
        self.remove_line_button.on_click(self.remove_line)
        self.insert_line_button.on_click(self.insert_line)
        self.remove_selected_line_button.on_click(self.remove_selected_line)
        self.check_venue_button.on_click(app.check_venue)

    def _build_venue_figure(self) -> None:
        """Add the glyphs to the venue figure of the app."""
        app = self.app
        p = app.venue_figure
        p.xaxis.axis_label_text_font_style = "normal"
        p.yaxis.axis_label_text_font_style = "normal"
        p.x('x', 'y', source=self.point_source, line_color="lightskyblue")
        p.multi_line('xlineplt', 'ylineplt', source=self.lines_source,
                     legend_label='PALC and SFP', line_dash='solid',
                     line_color="lightskyblue", line_width=2,
                     selection_color="crimson", nonselection_line_alpha=0.5)
        p.multi_line('xnalplt', 'ynalplt', source=self.non_audience_source,
                     legend_label='PALC (non-audience line)',
                     line_dash='dashed', line_color="lime", line_width=2,
                     nonselection_line_alpha=0.5)
        p.multi_line('xsfpplt', 'ysfpplt', source=self.sound_field_only_source,
                     legend_label='only SFP', line_dash='solid',
                     line_color="orange", line_width=2,
                     nonselection_line_alpha=0.5)
        p.patches('x_list', 'y_list', source=app.cabinet_patches_source,
                  color="white")
        # array and its rays
        p.multi_line('x_c_n_unitn', 'y_c_n_unitn', source=app.rays_source,
                     color="white", line_dash="solid")
        p.multi_line('x_c_n_unitn_psi1', 'y_c_n_unitn_psi1',
                     source=app.rays_source, color="red", line_dash="dashed")
        p.multi_line('x_c_n_unitn_psi2', 'y_c_n_unitn_psi2',
                     source=app.rays_source, color="green", line_dash="dashed")
        p.x('seg_pos_x', 'seg_pos_y', source=app.hit_points_source,
            color="white")
        p.x('seg_pos_start_x', 'seg_pos_start_y', source=app.hit_points_source,
            color="red")
        p.x('seg_pos_stop_x', 'seg_pos_stop_y', source=app.hit_points_source,
            color="green")
        # tap on a line to insert a line before it or to delete it
        self.lines_source.selected.on_change('indices', self.line_tapped)
        p.legend.location, p.legend.orientation, p.legend.margin = \
            'top_center', 'horizontal', 0
        p.legend.background_fill_alpha, p.legend.spacing, p.legend.visible = \
            0.5, 15, False

    def layout(self) -> LayoutDOM:
        """Body of the page below the header.

        Returns
        -------
        LayoutDOM
        """
        col0 = column([self.x_axis_title, self.start_x_input, self.end_x_input,
                       self.rule_x, self.points_per_meter_input,
                       self.venue_error_info], width=270)
        col1 = column([self.y_axis_title, self.start_y_input, self.end_y_input,
                       self.rule_y, self.save_line_button,
                       self.remove_line_button, self.insert_line_button,
                       self.remove_selected_line_button,
                       self.check_venue_button], width=270)
        col2 = column([self.app.venue_axes_row(), self.app.venue_figure],
                      width=690)
        return row([col0, col1, col2])

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def update_point(self, attr, old, new) -> None:
        """Read the start and end point of the next audience line.

        Triggered by the start and end point inputs.

        Parameters
        ----------
        attr : str
            Changed attribute (value).
        old, new : str
            Old and new text of the input.

        Returns
        -------
        None.
        """
        try:
            x = np.array([float(self.start_x_input.value),
                          float(self.end_x_input.value)])
            y = np.array([float(self.start_y_input.value),
                          float(self.end_y_input.value)])
            self.venue_error_info.text = ''
        except ValueError:
            self.venue_error_info.text = 'Error: Wrong Input. Only Numbers are allowed'
            return
        self.point_source.data = dict(x=x, y=y)

    def _points_per_meter(self):
        """Discretization entered by the user; 1 if it cannot be read.

        Returns
        -------
        float
            Points per m.
        bool
            True if the input could not be read.
        """
        try:
            return float(self.points_per_meter_input.value), False
        except ValueError:
            return 1, True

    def _new_line(self, points_per_meter):
        """The entered line, checked against the drawn lines.

        Returns
        -------
        tuple or None
            ``(x_start, x_stop, y_start, y_stop, slope, b, eval_points)``
            for appending, or None if the line is rejected (the reason is
            shown in the error text).
        """
        [x_start, x_stop] = self.point_source.data['x']
        [y_start, y_stop] = self.point_source.data['y']
        try:
            check_line(self.app.state.drawn,
                       AudienceLine(start=(x_start, y_start),
                                    stop=(x_stop, y_stop),
                                    points_per_meter=points_per_meter))
        except VenueError as exc:
            self.venue_error_info.text = str(exc)
            return None
        slope, b = get_slope_b(x_start, x_stop, y_start, y_stop)
        diff = np.sqrt((x_stop - x_start)**2 + (y_stop - y_start)**2)
        return x_start, x_stop, y_start, y_stop, slope, b, \
            diff*points_per_meter

    def save_line(self) -> None:
        """Append the entered audience line ("Save Current Audience Line").

        The line is checked with :func:`pypalc.core.pipeline.check_line`;
        a rejected line shows the reason in the error text.

        Returns
        -------
        None.
        """
        points_per_meter, failed = self._points_per_meter()
        line = self._new_line(points_per_meter)
        if line is None:
            return
        self.venue_error_info.text = ''
        if failed:
            self.venue_error_info.text = _DISCRETIZATION_ERROR
        self.app.state.drawn.append_line(*line)
        self.app.update_venue_ranges()
        drawn = self.app.state.drawn
        self.lines_source.data = dict(xlineplt=drawn.plot_x,
                                      ylineplt=drawn.plot_y)
        self.app.set_venue_checked(False)
        self.app.show()

    def remove_line(self) -> None:
        """Remove the last audience line ("Remove Last Audience Line").

        Returns
        -------
        None.
        """
        drawn = self.app.state.drawn
        if np.shape(drawn.pal_index)[0] > 1:
            drawn.remove_last_line()
        if np.shape(drawn.pal_index)[0] > 1:
            self.app.update_venue_ranges()
        self.lines_source.data = dict(xlineplt=drawn.plot_x,
                                      ylineplt=drawn.plot_y)
        self.app.set_venue_checked(False)

    def insert_line(self) -> None:
        """Insert the entered line before the selected line ("Add Line
        before Selected Line").

        Returns
        -------
        None.
        """
        sel_ind = self.tapped_line_source.data['tapped'][0]
        points_per_meter, failed = self._points_per_meter()
        if failed:
            self.venue_error_info.text = _DISCRETIZATION_ERROR
        line = self._new_line(points_per_meter)
        if line is None:
            return
        self.venue_error_info.text = ''
        drawn = self.app.state.drawn
        drawn.insert_previous_line(sel_ind, *line)
        self.lines_source.data = dict(xlineplt=drawn.plot_x,
                                      ylineplt=drawn.plot_y)
        self.app.set_venue_checked(False)
        self.app.show()

    def remove_selected_line(self) -> None:
        """Remove the selected line ("Remove Selected Audience Line").

        Returns
        -------
        None.
        """
        sel_ind = self.tapped_line_source.data['tapped'][0]
        drawn = self.app.state.drawn
        drawn.remove_selected_line(sel_ind)
        self.lines_source.data = dict(xlineplt=drawn.plot_x,
                                      ylineplt=drawn.plot_y)
        self.app.set_venue_checked(False)
        self.app.show()

    def line_tapped(self, attr, old, new) -> None:
        """Switch the line buttons when a line is (de)selected.

        Triggered by a tap on a line in the venue figure.

        Parameters
        ----------
        attr : str
            Attribute name (indices).
        old, new : list of int
            Previously and now selected lines.

        Returns
        -------
        None.
        """
        selected = new != []
        self.insert_line_button.visible = selected
        self.remove_selected_line_button.visible = selected
        self.save_line_button.visible = not selected
        self.remove_line_button.visible = not selected
        self.tapped_line_source.data = dict(tapped=new)
