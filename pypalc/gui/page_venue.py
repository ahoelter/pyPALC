"""Page 1 of the GUI: draw the polygonal audience line (the venue).

author: Arne Hoelter
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

import numpy as np
from bokeh.layouts import column, row
from bokeh.models import (Button, ColumnDataSource, Div, LayoutDOM, Paragraph,
                          TextInput)

from pypalc.core.errors import VenueError
from pypalc.core.pipeline import check_line
from pypalc.core.settings import AudienceLine
from pypalc.core.venue import slope_and_intercept
from pypalc.gui import skipped_while_applying_setup

if TYPE_CHECKING:
    from .app import PalcApp

logger = logging.getLogger(__name__)

#: Separator lines between widget groups.
RULE_LONG = "______________________________________"
RULE = "___________________________________"

#: Shown if the discretization cannot be read.
_DISCRETIZATION_ERROR = 'Couldnt process discretization input. Use default: 5 points per m'


def _audience_line(line: tuple, points_per_meter: float) -> AudienceLine:
    """Setting of a line as returned by :meth:`VenuePage._new_line`."""
    x_start, x_stop, y_start, y_stop = (float(v) for v in line[:4])
    return AudienceLine(start=(x_start, y_start), stop=(x_stop, y_stop),
                        points_per_meter=float(points_per_meter))


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
        self.lines_source = ColumnDataSource(data=dict(lines_x=[],
                                                       lines_y=[]))
        """The drawn audience lines."""
        self.non_audience_source = ColumnDataSource(data=dict(lines_x=[],
                                                              lines_y=[]))
        """Non-audience lines, e.g., gaps between audience lines."""
        self.sound_field_only_source = ColumnDataSource(data=dict(
            lines_x=[], lines_y=[]))
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
        venue_figure = app.venue_figure
        venue_figure.xaxis.axis_label_text_font_style = "normal"
        venue_figure.yaxis.axis_label_text_font_style = "normal"
        venue_figure.scatter('x', 'y', source=self.point_source, marker='x',
                             line_color="lightskyblue")
        venue_figure.multi_line('lines_x', 'lines_y', source=self.lines_source,
                                legend_label='PALC and SFP', line_dash='solid',
                                line_color="lightskyblue", line_width=2,
                                selection_color="crimson",
                                nonselection_line_alpha=0.5)
        venue_figure.multi_line('lines_x', 'lines_y',
                                source=self.non_audience_source,
                                legend_label='PALC (non-audience line)',
                                line_dash='dashed', line_color="lime",
                                line_width=2, nonselection_line_alpha=0.5)
        venue_figure.multi_line('lines_x', 'lines_y',
                                source=self.sound_field_only_source,
                                legend_label='only SFP', line_dash='solid',
                                line_color="orange", line_width=2,
                                nonselection_line_alpha=0.5)
        venue_figure.patches('patches_x', 'patches_y',
                             source=app.cabinet_patches_source, color="white")
        # array and its rays
        venue_figure.multi_line('center_rays_x', 'center_rays_y',
                                source=app.rays_source, color="white",
                                line_dash="solid")
        venue_figure.multi_line('start_rays_x', 'start_rays_y',
                                source=app.rays_source, color="red",
                                line_dash="dashed")
        venue_figure.multi_line('stop_rays_x', 'stop_rays_y',
                                source=app.rays_source, color="green",
                                line_dash="dashed")
        venue_figure.scatter('hit_points_x', 'hit_points_y', marker='x',
                             source=app.hit_points_source, color="white")
        venue_figure.scatter('hit_points_start_x', 'hit_points_start_y',
                             marker='x', source=app.hit_points_source,
                             color="red")
        venue_figure.scatter('hit_points_stop_x', 'hit_points_stop_y',
                             marker='x', source=app.hit_points_source,
                             color="green")
        # tap on a line to insert a line before it or to delete it
        self.lines_source.selected.on_change('indices', self.line_tapped)
        legend = venue_figure.legend
        legend.location, legend.orientation, legend.margin = \
            'top_center', 'horizontal', 0
        legend.background_fill_alpha, legend.spacing, legend.visible = \
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

    @skipped_while_applying_setup
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
            ``(x_start, x_stop, y_start, y_stop, slope, intercept,
            eval_points)``
            for appending, or None if the line is rejected (the reason is
            shown in the error text).
        """
        [x_start, x_stop] = self.point_source.data['x']
        [y_start, y_stop] = self.point_source.data['y']
        try:
            check_line(self.app.state.drawn_lines,
                       AudienceLine(start=(x_start, y_start),
                                    stop=(x_stop, y_stop),
                                    points_per_meter=points_per_meter))
        except VenueError as exc:
            self.venue_error_info.text = str(exc)
            return None
        slope, intercept = slope_and_intercept(x_start, x_stop, y_start,
                                               y_stop)
        diff = np.sqrt((x_stop - x_start)**2 + (y_stop - y_start)**2)
        return x_start, x_stop, y_start, y_stop, slope, intercept, \
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
        self.app.state.drawn_lines.append_line(*line)
        self.app.state.audience_lines.append(
            _audience_line(line, points_per_meter))
        self.app.update_venue_ranges()
        drawn_lines = self.app.state.drawn_lines
        self.lines_source.data = dict(lines_x=drawn_lines.plot_x,
                                      lines_y=drawn_lines.plot_y)
        self.app.set_venue_checked(False)
        self.app.show()

    def draw_lines(self, lines: Sequence[AudienceLine]) -> None:
        """Draw audience lines into the empty venue of a new session state.

        Used by :func:`~.setup_io.apply_setup`. Every line is checked and
        appended as "Save Current Audience Line" does; the point inputs
        show the last line, as after entering the lines by hand.

        Parameters
        ----------
        lines : sequence of AudienceLine
            The lines, at least one; they must pass the draw conditions.

        Returns
        -------
        None.

        Raises
        ------
        VenueError
            If a line is rejected.
        """
        state = self.app.state
        drawn_lines = state.drawn_lines
        for line in lines:
            check_line(drawn_lines, line)
            (x_start, y_start), (x_stop, y_stop) = line.start, line.stop
            slope, intercept = slope_and_intercept(x_start, x_stop,
                                                   y_start, y_stop)
            diff = np.sqrt((x_stop - x_start)**2 + (y_stop - y_start)**2)
            drawn_lines.append_line(x_start, x_stop, y_start, y_stop, slope,
                              intercept,
                              diff*line.points_per_meter)
            state.audience_lines.append(line)
        last = lines[-1]
        self.start_x_input.value = repr(float(last.start[0]))
        self.end_x_input.value = repr(float(last.stop[0]))
        self.points_per_meter_input.value = repr(float(last.points_per_meter))
        self.start_y_input.value = repr(float(last.start[1]))
        self.end_y_input.value = repr(float(last.stop[1]))
        self.point_source.data = dict(
            x=np.array([float(last.start[0]), float(last.stop[0])]),
            y=np.array([float(last.start[1]), float(last.stop[1])]))
        self.venue_error_info.text = ''
        self.lines_source.data = dict(lines_x=drawn_lines.plot_x,
                                      lines_y=drawn_lines.plot_y)
        self.app.update_venue_ranges()

    def remove_line(self) -> None:
        """Remove the last audience line ("Remove Last Audience Line").

        Returns
        -------
        None.
        """
        drawn_lines = self.app.state.drawn_lines
        if np.shape(drawn_lines.line_end_indices)[0] > 1:
            drawn_lines.remove_last_line()
            del self.app.state.audience_lines[-1]
        if np.shape(drawn_lines.line_end_indices)[0] > 1:
            self.app.update_venue_ranges()
        self.lines_source.data = dict(lines_x=drawn_lines.plot_x,
                                      lines_y=drawn_lines.plot_y)
        self.app.set_venue_checked(False)

    def insert_line(self) -> None:
        """Insert the entered line before the selected line ("Add Line
        before Selected Line").

        Returns
        -------
        None.
        """
        selected_index = self.tapped_line_source.data['tapped'][0]
        points_per_meter, failed = self._points_per_meter()
        if failed:
            self.venue_error_info.text = _DISCRETIZATION_ERROR
        line = self._new_line(points_per_meter)
        if line is None:
            return
        self.venue_error_info.text = ''
        drawn_lines = self.app.state.drawn_lines
        drawn_lines.insert_previous_line(selected_index, *line)
        self.app.state.audience_lines.insert(
            selected_index, _audience_line(line, points_per_meter))
        self.lines_source.data = dict(lines_x=drawn_lines.plot_x,
                                      lines_y=drawn_lines.plot_y)
        self.app.set_venue_checked(False)
        self.app.show()

    def remove_selected_line(self) -> None:
        """Remove the selected line ("Remove Selected Audience Line").

        Returns
        -------
        None.
        """
        selected_index = self.tapped_line_source.data['tapped'][0]
        drawn_lines = self.app.state.drawn_lines
        drawn_lines.remove_selected_line(selected_index)
        del self.app.state.audience_lines[selected_index]
        self.lines_source.data = dict(lines_x=drawn_lines.plot_x,
                                      lines_y=drawn_lines.plot_y)
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
