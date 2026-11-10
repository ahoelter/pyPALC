"""Help page of the GUI (formerly ``fun/gui_help_text.py``).

The texts are in :data:`HELP`: one tab per page of the GUI, each with
columns of HTML paragraphs. A step is introduced with its label as the
GUI shows it (in bold).

author: Arne Hoelter
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import panel as pn
import panel_material_ui as pmui

from pypalc.gui.style import LINK, TEXT, TEXT_MUTED
from pypalc.gui.widgets import GAP, card

if TYPE_CHECKING:
    from .app import PalcApp

#: Preferred and largest width of a column of help text in px (lines stay
#: readable on wide windows); columns wrap below each other on narrow
#: windows.
COLUMN_WIDTH = 340
MAX_COLUMN_WIDTH = 720

_GENERAL = (
    ('<b>General information</b>',
     'This is a <i>Loudspeaker Array Curving Optimization Software</i>.',
     'Programmed and developed by: <br> <b><a href="https://www.ak.tu-berlin'
     '.de/menue/team/wissenschaftliche_mitarbeiter/arne_hoelter/" '
     'target="_blank">Arne Hölter</a></b> and <b>Florian Straube</b>',
     'Correspondence should be addressed to: <br> <a href="mailto:hoelter@'
     'campus.tu-berlin.de?subject=PALC-App">hoelter@campus.tu-berlin.de</a>',
     'The pages <b>1 · Venue</b>, <b>2 · Loudspeaker</b>, <b>3 · '
     'Array</b>, <b>4 · Algorithm</b> and <b>5 · Sound Field</b> in the '
     'header lead through an optimization; its steps are numbered 1 to '
     '34. '
     '<b>Load Setup</b> on page <b>1 · Venue</b> or <b>4 · Algorithm</b> '
     'restores a saved setup, runs it and shows <b>5 · Sound Field</b>; '
     '<b>Load Example</b> on page <b>1 · Venue</b> does the same with one '
     'of the example setups.'),
    ('Based on the following publications:',
     '<b>[1] Straube, F.; Schultz, F.; Bonillo, D.A.; Weinzierl, S. (2018): '
     '“An Analytical Approach for Optimizing the Curving of Line Source '
     'Arrays.” In: <i>J. Audio Eng. Soc.</i> 66(1/2), 4–20.</b>',
     '<b>[2] Hölter, A.; Straube, F.; Schultz, F.; Weinzierl, S. (2020): '
     '“Eine Web-Applikation zu Optimierung der Krümmung von Line Source '
     'Arrays.” In: <i>Fortschritte der Akustik: Tagungsband d. 46. DAGA</i>, '
     'Hannover.</b>',
     '<b>[3] Hölter, A.; Straube, F.; Schultz, F.; Weinzierl, S. (2021): '
     '“Enhanced Polygonal Audience Line Curving for Line Source Arrays.” '
     'In: <i>Proc. of the 149th Audio Eng. Soc. Conv.</i>, Online.</b>',
     '<b>[4] Hölter, A.; Schultz, F.; Straube F.; Weinzierl, S. (2021): '
     '"SPL-basierte Optimierung der Krümmung von Line-Source-Arrays mit '
     'PALC." In: <i>Fortschritte der Akustik: Tagungsband d. 47. DAGA</i>, '
     'Wien.</b>'),
)

_VENUE = (
    ('<b>Create the venue!</b>',
     'Only straight audience lines can be created; a venue is made of '
     'several lines.',
     '<b>1. Start x in m</b>: the start position of an audience line on '
     'the x-axis in meter.',
     '<b>2. End x in m</b>: the end position of an audience line on the '
     'x-axis in meter.',
     '<b>3. Discretization in points per m</b>: the number of points per '
     'meter of an audience line. Only integer values are allowed. Note: '
     'The maximum of discretized points is 1000, e.g., a discretization '
     'of 10 points per meter would allow around 100 meter of audience '
     'lines.',
     '<b>4. Start y in m</b>: the start position of an audience line on '
     'the y-axis in meter.',
     '<b>5. End y in m</b>: the end position of an audience line on the '
     'y-axis in meter.'),
    ('<b>6. Save Current Audience Line</b> saves the current audience '
     'line; it is shown in the figure. If a line is <b>tapped</b> in the '
     'figure, the button reads <b>6. Add Line before Selected Line</b> and '
     'creates the line in front of the tapped line.',
     '<b>7. Remove Last Audience Line</b> removes the last audience line '
     'that was created, also after a calculation. If a line is '
     '<b>tapped</b>, <b>7. Remove Selected Audience Line</b> removes the '
     'tapped line.',
     '<b>8. Get PALC Venue Proposal</b> creates a PALC compatible venue, '
     'i.e., a continuous polygonal audience line (PAL) based on the drawn '
     'audience lines. The gaps between the audience lines are processed. '
     'Afterwards the button reads <b>8. Venue is PALC compatible!</b>',
     '<b>Axes</b> of the venue figure (also on pages 2 to 4): <b>Equal, '
     'variable height</b>: one meter has the same length on both axes and '
     'both have the same tick spacing; the '
     'figure is as wide as in Stretched and its height follows (if it would '
     'run below the window, it is narrower); <b>Equal, fixed height</b>: '
     'the same scaling; the figure is as high as in Stretched and its width '
     'follows (if it would be too wide, it is lower); <b>Stretched</b>: the '
     'venue fills the figure. A narrower figure is centered.'),
)

_LOUDSPEAKER = (
    ('<b>Configure the loudspeakers!</b>',
     '<b>9. Aperture angle in deg</b>: the PALC aperture angle of the '
     'loudspeakers is the start value of the iterative PALC calculation. '
     'In some cases it can be useful to set the start value lower than the '
     'default value, e.g., if many loudspeakers shall reinforce a small '
     'venue. In most of the cases the default value can be maintained.',
     '<b>10. Loudspeaker height in m</b>: the height of the front grille '
     'of the deployed loudspeaker cabinets.'),
    ('<b>11. Directivity</b> of the loudspeakers, either modeled or '
     'measured data. Modeled: the directivities <i>circular piston</i>, '
     '<i>line piston</i> or a <i>combination of a circular and a line '
     'piston</i> are possible. For the latter the circular piston is used '
     'for the low frequencies and the line piston for the high '
     'frequencies. The crossover is modeled with a 4th-order '
     'Linkwitz-Riley filter at 1.5 kHz.',
     '<b>Upload measured data</b> (shown for measured loudspeaker data): '
     'only .csv files are accepted. The first entry must be a zero as a '
     'placeholder. The first line must contain the frequency bins in Hz. '
     'The first column must contain the angles in degree. The rest of the '
     'table must contain the complex directivity values in cartesian '
     'representation (amplitude and phase). The imaginary unit is \'i\' '
     'or \'j\'.'),
)

_ARRAY = (
    ('<b>Configure the loudspeaker array!</b>',
     '<b>12. Number of cabinets</b>: the number of loudspeaker cabinets '
     'the array is built of.',
     '<b>13. Top x in m</b>: the x-coordinate of the top front edge of the '
     'uppermost cabinet of the array in meter.',
     '<b>14. Top y in m</b>: the y-coordinate of the top front edge of the '
     'uppermost cabinet of the array in meter.',
     '<b>15. Gap between cabinets in m</b>: the gap between the '
     'loudspeaker cabinets within the array in meter.'),
    ('<b>16. Use discrete tilt angles?</b> Choose whether the calculation '
     'uses a set of discrete tilt angles.',
     '<b>17. Discrete tilt angles in deg</b>: enabled if (16.) is set to '
     '\'Yes\'. A set of discrete tilt angles that will be used for the '
     'line array curving. The value separator is \',\' and the decimal '
     'separator is \'.\', e.g., 0,1,2.5,3,5. This means the possible inter '
     'cabinet tilt angles are 0 deg, 1 deg, 2.5 deg, 3 deg and 5 deg. '
     '<br> <b>Note:</b> In most cases it is necessary to increase the '
     'tolerance in (23.)!',
     '<b>Reference array</b>: the array the PALC result is compared with, '
     'of the type Straight, Progressive, Arc or User Defined, and its '
     'angles.'),
)

_ALGORITHM = (
    ('<b>Algorithm options and calculation!</b>',
     '<b>18. PALC algorithm</b>: <i>PALC 1</i>: the PALC aperture angles '
     'of all loudspeaker cabinets are set to be constant. <i>PALC 2</i>: '
     'the product of the PALC aperture angle of a loudspeaker cabinet and '
     'the distance of the acoustic center of the respective loudspeaker '
     'cabinet to the audience line is set to be constant. <i>PALC 3</i> is '
     'similar to PALC 2 but it takes the tangent of the PALC aperture angle '
     'instead of the approximated angle itself.',
     '<b>19. Gap handling</b>: <i>Without</i>: no additional lines are '
     'inserted in the gaps between the audience lines. <i>Hard Margin</i>: '
     'the gaps are not handled as audience lines and the calculation runs '
     'separately for each continuous audience line. The combination with '
     'target slope weighting is not possible. <i>Soft Margin</i>: non '
     'audience lines are inserted in the gaps between the audience lines. '
     'They are treated with a lower weighting in the calculation.',
     '<b>20. Strength of soft margin</b>: enabled when choosing soft margin '
     'in (19.). Raise the slider to use lower weighting for gaps or reduce '
     'the slider to converge weighting of the gaps to the audience '
     'lines.'),
    ('<b>21. Weighting</b>: choose a weighting in order to affect the SPL '
     'loss over distance: linear spacing, logarithmic spacing or a target '
     'slope. <br> If linear or logarithmic spacing is chosen, choose the '
     'strength of the weighting factors in (22.). <br> If Target Slope is '
     'chosen, the SPL loss over distance will be optimized regarding the '
     'target slope, which is shown and can be dragged in the tab <b>SPL '
     'over distance</b> (<b>Frequency range in Hz</b>: the frequencies of '
     'the SPL over distance). Choose the number of <b>Hinges</b> (0, 1 or '
     '2). The combination with the hard margin gap handling is not '
     'possible.',
     '<b>22. Weighting parameter ν</b>: adjust the weighting parameter '
     'with the slider or by clicking on: <b>+</b> to increase the SPL loss '
     'over distance, <b>-</b> to decrease the SPL loss over distance, by '
     'the <b>Step</b>.',
     '<b>23. Tolerance (abort criterion) in m</b>: the criterion for the '
     'iterative algorithm to abort. If no convergence is reached in a '
     'calculation, it should always be the first step to increase the '
     'tolerance.',
     '<b>24. Fix angle of uppermost cabinet?</b> If the uppermost '
     'loudspeaker cabinet within the array is supposed to be mounted with '
     'a specific, <b>fixed</b> tilt angle, change to \'Yes\'.',
     '<b>25. Angle of uppermost cabinet in deg</b>: if (24.) is set to '
     '\'Yes\', the fixed tilt angle of the uppermost loudspeaker cabinet. '
     'It is only possible to change the angle in the allowed range shown '
     'below it, which depends on the array and the venue geometry.'),
    ('<b>26. PALC Run</b>: start the computation of the PALC algorithm; '
     'while it runs, the button reads <b>Running...</b> <br> <b>Note</b>, '
     'if no convergence is reached, decrease the initial aperture angle in '
     '(9.) or increase the tolerance (23.).',
     '<b>27. Delete result visualization</b>: removes the visualization of '
     'the results in the venue figure. This is also done when switching '
     'to <b>1 · Venue</b> in the header.',
     '<b>Download PALC Angles</b> downloads the tilt angles as '
     'tilt_angles.csv. <b>Download Results</b> downloads five CSV files '
     'in pypalc_results.zip: the SPL over distance of the optimized array '
     '(spl_over_distance.csv) and of the reference array '
     '(reference_spl_over_distance.csv), the target slope '
     '(target_slope.csv), its reference point (target_reference.csv) and '
     'the homogeneity (homogeneity.csv).',
     '<b>Save Setup</b> downloads all inputs, the drawn venue and the '
     'target slope as the file pypalc_setup.yaml (uploaded measured '
     'loudspeaker data is embedded); it also runs headless with \'python '
     '-m pypalc run\'. <b>Load Setup</b> reads such a file, restores '
     'everything and runs the calculation; it is also on page 1, where '
     '<b>Load Example</b> loads one of the example setups.'),
)

_SOUND_FIELD = (
    ('<b>Visualization of the sound field prediction!</b>',
     '<b>28. Venue slice positions</b>: the discrete audience positions in '
     'the colors of the corresponding SPL predictions in (29.) and (30.). '
     'Tap a position to see its SPL prediction in (29.) and (30.).',
     '<b>29. SPL at all audience positions: PALC</b>: frequency responses '
     'at all audience positions for the loudspeaker array using the '
     '<i>PALC</i> optimized tilt angles.',
     '<b>30. SPL at all audience positions: Reference array</b>: frequency '
     'responses at all audience positions for the <i>reference loudspeaker '
     'array</i>.'),
    ('<b>31. Homogeneity</b> over frequency at all audience positions for '
     'the loudspeaker array using the <i>PALC</i> optimized tilt angles in '
     'blue and for the <i>reference loudspeaker array</i> in orange.',
     '<b>32. Bar chart: PALC</b>: the SPL values at the audience positions '
     'for the loudspeaker array using the <i>PALC</i> optimized tilt '
     'angles.',
     '<b>33. Bar chart: Reference array</b>: the SPL values at the '
     'audience positions for the <i>reference loudspeaker array</i>.',
     '<b>34. Beam plot: PALC (SPL in dB)</b> and <b>34. Beam plot: '
     'Reference array (SPL in dB)</b> visualize the SPL in the vertical '
     'radiation plane of the arrays at the selected <b>Frequency in '
     'Hz</b>; choose the arrays at <b>Beam plot of</b>.'),
)

#: (tab title, columns of paragraphs) of the help page, one tab per page.
HELP: Tuple[Tuple[str, Tuple[Tuple[str, ...], ...]], ...] = (
    ('General information', _GENERAL),
    ('1 · Venue', _VENUE),
    ('2 · Loudspeaker', _LOUDSPEAKER),
    ('3 · Array', _ARRAY),
    ('4 · Algorithm', _ALGORITHM),
    ('5 · Sound Field', _SOUND_FIELD),
)


def _stylesheet() -> str:
    """Style sheet of a column of help text: paragraphs with space between
    them, green links."""
    return f"""
        .help {{ color: {TEXT}; font-size: 14px; line-height: 1.5; }}
        .help p {{ margin: 0 0 12px 0; }}
        .help p:first-child {{ color: {TEXT}; }}
        .help a {{ color: {LINK}; }}
        .help i {{ color: {TEXT_MUTED}; }}
    """


def help_column(paragraphs: Tuple[str, ...]) -> pn.pane.HTML:
    """A column of help text.

    Parameters
    ----------
    paragraphs : tuple of str
        The paragraphs (HTML).

    Returns
    -------
    panel.pane.HTML
    """
    text = ''.join(f'<p>{paragraph}</p>' for paragraph in paragraphs)
    return pn.pane.HTML(
        f'<div class="help">{text}</div>', stylesheets=[_stylesheet()],
        sizing_mode='stretch_width', margin=0,
        styles={'flex': f'1 1 {COLUMN_WIDTH}px', 'min-width': '0',
                'max-width': f'{MAX_COLUMN_WIDTH}px'})


class HelpPage:
    """Page 0: general information and help.

    Parameters
    ----------
    app : PalcApp
        The session the page belongs to.
    """

    def __init__(self, app: PalcApp) -> None:
        self.app = app
        self._layout: Optional[pn.viewable.Viewable] = None
        self.tabs = pmui.Tabs(
            *[(title, pn.FlexBox(*[help_column(column) for column in columns],
                                 flex_wrap='wrap', gap=f'{2 * GAP}px',
                                 sizing_mode='stretch_width',
                                 margin=(GAP, 8, 0, 8)))
              for title, columns in HELP],
            sizing_mode='stretch_width', margin=(0, 8))
        """Tabs with the help of each page."""

    def layout(self) -> pn.Column:
        """Body of the page below the header.

        Returns
        -------
        panel.Column
            One card with the help tabs.
        """
        if self._layout is None:
            self._layout = pn.Column(
                card('Help', self.tabs, hide_header=True),
                sizing_mode='stretch_width', margin=(0, GAP))
        return self._layout
