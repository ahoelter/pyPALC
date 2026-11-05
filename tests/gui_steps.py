"""The scripted GUI interactions and the models whose state is recorded.

:func:`steps` lists 100 user interactions (typing into inputs, choosing
options, uploading a file, tapping lines and points, dragging the target
slope, running PALC, switching pages). They use the property names of the
Bokeh GUI through :mod:`tests.gui_driver` and the methods of the pages, so
they run on the Bokeh GUI and on the Panel GUI alike.
``tools/record_gui_reference.py`` records the state of :data:`MODELS` and
:data:`CONFIG_KEYS` after every step.

author: Arne Hoelter
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List, Tuple

from tests.gui_driver import upload, write

ROOT = Path(__file__).resolve().parent.parent

#: The recorded models: "<page index>.<attribute>" or "app.<attribute>".
MODELS = (
    "1.start_x_input", "1.start_y_input", "1.end_x_input", "1.end_y_input",
    "1.points_per_meter_input", "1.save_line_button",
    "1.remove_line_button", "1.insert_line_button",
    "1.remove_selected_line_button", "1.check_venue_button",
    "1.venue_error_info", "1.lines_source", "1.non_audience_source",
    "1.sound_field_only_source", "1.tapped_line_source",
    "2.aperture_input", "2.cabinet_height_input", "2.directivity_select",
    "2.directivity_upload", "2.directivity_source",
    "3.cabinet_count_input", "3.top_x_input", "3.top_y_input",
    "3.cabinet_gap_input", "3.use_discrete_angles_select",
    "3.discrete_angles_input", "3.reference_type_select",
    "3.reference_discrete_angles_select", "3.reference_start_input",
    "3.reference_step_stop_input", "3.reference_user_angles_input",
    "3.reference_angles_source",
    "4.palc_variant_select", "4.gap_handling_select",
    "4.soft_margin_slider", "4.tolerance_slider",
    "4.frequency_range_slider", "4.weighting_select", "4.hinges_select",
    "4.weighting_nu_slider", "4.nu_plus_button", "4.nu_minus_button",
    "4.nu_step_select", "4.use_fixed_angle_select", "4.fixed_angle_input",
    "4.fixed_angle_text", "4.start_toggle", "4.clear_results_button",
    "4.results_source", "4.spl_over_distance_source",
    "4.reference_spl_over_distance_source", "4.target_source",
    "4.target_reference_source", "4.target_region_source",
    "4.spl_over_distance_figure",
    "5.sound_field_source", "5.venue_points_source",
    "5.homogeneity_source", "5.histogram_source", "5.beam_source",
    "5.beam_figure", "5.reference_beam_figure", "5.venue_points_figure",
    "5.beam_frequency_select", "5.beam_checkbox",
    "app.config_error_info", "app.page_select", "app.venue_axes_select",
    "app.venue_figure", "app.cabinet_patches_source", "app.rays_source",
    "app.hit_points_source",
)

#: Values of the PALC configuration the configuration widgets store.
CONFIG_KEYS = ('aperture_angles', 'cabinet_count', 'cabinet_gap',
               'cabinet_height', 'directivity', 'discrete_tilt_angles',
               'fixed_angle', 'frequency_range_hz', 'gap_handling',
               'palc_variant', 'soft_margin_strength', 'tolerance', 'top_x',
               'top_y', 'use_discrete_angles', 'use_fixed_angle',
               'use_weighting', 'weighting_nu')


def model(app, path: str) -> Any:
    """The model of a :data:`MODELS` path in the app."""
    owner, name = path.split('.')
    return getattr(app if owner == 'app' else app.pages[int(owner)], name)


def steps(app) -> List[Tuple[str, Callable[[], None]]]:
    """The interactions on the app.

    Parameters
    ----------
    app : PalcApp
        A fresh app.

    Returns
    -------
    list of (str, callable)
        Name and function of each of the 100 steps, in order.
    """
    venue, algorithm, results = app.pages[1], app.pages[4], app.pages[5]
    csv = (ROOT / 'tests' / 'cases' / 'measured.csv').read_bytes()

    def setv(path, value, prop='value'):
        return lambda: write(model(app, path), prop, value)

    def select(source, indices):
        return lambda: write(source, 'selected', indices)

    def drag():
        data = algorithm.target_source.data
        write(algorithm.target_source, 'data', dict(
            distance_m=[5.0, 12.0, 18.0, 25.0],
            level_db=[0.0, -2.0, -4.0, -8.0],
            x_m=list(data['x_m']), y_m=list(data['y_m'])))

    return [
        ('page1', setv('app.page_select', 1, 'active')),
        ('sx', setv('1.start_x_input', '2')),
        ('sy', setv('1.start_y_input', '0')),
        ('ex', setv('1.end_x_input', '10')),
        ('ey', setv('1.end_y_input', '1')),
        ('save', venue.save_line),
        ('save_same', venue.save_line),
        ('sx2', setv('1.start_x_input', '10')),
        ('sy2', setv('1.start_y_input', '1.5')),
        ('ex2', setv('1.end_x_input', '20')),
        ('ey2', setv('1.end_y_input', '4')),
        ('ppm_bad', setv('1.points_per_meter_input', 'abc')),
        ('save2', venue.save_line),
        ('ppm', setv('1.points_per_meter_input', '5')),
        ('sx3', setv('1.start_x_input', '20')),
        ('sy3', setv('1.start_y_input', '5')),
        ('ex3', setv('1.end_x_input', '30')),
        ('ey3', setv('1.end_y_input', '8')),
        ('save3', venue.save_line),
        ('axes1', setv('app.venue_axes_select', 1, 'active')),
        ('axes2', setv('app.venue_axes_select', 2, 'active')),
        ('axes0', setv('app.venue_axes_select', 0, 'active')),
        ('tap', select(venue.lines_source, [1])),
        ('sxi', setv('1.start_x_input', '5')),
        ('sxi2', setv('1.end_x_input', '9')),
        ('syi', setv('1.start_y_input', '-1')),
        ('eyi', setv('1.end_y_input', '-0.5')),
        ('insert', venue.insert_line),
        ('untap', select(venue.lines_source, [])),
        ('tap0', select(venue.lines_source, [1])),
        ('remove_sel', venue.remove_selected_line),
        ('untap2', select(venue.lines_source, [])),
        ('start_nocheck', setv('4.start_toggle', True, 'active')),
        ('check', app.check_venue),
        ('page2', setv('app.page_select', 2, 'active')),
        ('psi_bad', setv('2.aperture_input', 'x')),
        ('psi', setv('2.aperture_input', '4')),
        ('dir_line', setv('2.directivity_select', 'Line Piston')),
        ('dir_meas', setv('2.directivity_select',
                          'Measured Loudspeaker Data')),
        ('upload', lambda: upload(model(app, '2.directivity_upload'), csv,
                                  'measured.csv')),
        ('dir_circ', setv('2.directivity_select', 'Circular Piston')),
        ('page3', setv('app.page_select', 3, 'active')),
        ('N', setv('3.cabinet_count_input', '8')),
        ('N_bad', setv('3.cabinet_count_input', '-3')),
        ('N2', setv('3.cabinet_count_input', '10')),
        ('yh_low', setv('3.top_y_input', '-50')),
        ('yh', setv('3.top_y_input', '6')),
        ('dang', setv('3.use_discrete_angles_select', 'Yes')),
        ('dang_vals', setv('3.discrete_angles_input',
                           '0,1,2,3,4,5,6,7,8,9,10')),
        ('dang_bad', setv('3.discrete_angles_input', 'a,b')),
        ('dang_vals2', setv('3.discrete_angles_input', '0,1,2,4,6,8,10')),
        ('ref_prog', setv('3.reference_type_select', 'Progressive')),
        ('ref_arc', setv('3.reference_type_select', 'Arc')),
        ('ref_step', setv('3.reference_step_stop_input', '3')),
        ('ref_dang', setv('3.reference_discrete_angles_select', 'Yes')),
        ('ref_user', setv('3.reference_type_select', 'User Defined')),
        ('ref_userv', setv('3.reference_user_angles_input', '1,2,3')),
        ('ref_userv2', setv('3.reference_user_angles_input',
                            '1,2,3,4,5,6,7,8,9,10')),
        ('ref_str', setv('3.reference_type_select', 'Straight')),
        ('dang_no', setv('3.use_discrete_angles_select', 'No')),
        ('page4', setv('app.page_select', 4, 'active')),
        ('gh_hm', setv('4.gap_handling_select', 'Hard Margin')),
        ('gh_sm', setv('4.gap_handling_select', 'Soft Margin')),
        ('sm', setv('4.soft_margin_slider', 1.5)),
        ('w_lin', setv('4.weighting_select', 'Linear Spacing')),
        ('stepsz', setv('4.nu_step_select', '0.1')),
        ('plus', algorithm.increase_nu),
        ('plus2', algorithm.increase_nu),
        ('minus', algorithm.decrease_nu),
        ('nu', setv('4.weighting_nu_slider', 1.3)),
        ('w_log', setv('4.weighting_select', 'Logarithmic Spacing')),
        ('fix', setv('4.use_fixed_angle_select', 'Yes')),
        ('fixv', setv('4.fixed_angle_input', '-100')),
        ('fixv2', setv('4.fixed_angle_input', 'zz')),
        ('fixv3', setv('4.fixed_angle_input', '2')),
        ('fix_no', setv('4.use_fixed_angle_select', 'No')),
        ('tol', setv('4.tolerance_slider', 0.5)),
        ('fr', setv('4.frequency_range_slider', (200.0, 8000.0))),
        ('w_none', setv('4.weighting_select', 'Without')),
        ('gh_none', setv('4.gap_handling_select', 'Without')),
        ('run', setv('4.start_toggle', True, 'active')),
        ('page5', setv('app.page_select', 5, 'active')),
        ('beam0', setv('5.beam_checkbox', [0], 'active')),
        ('beam01', setv('5.beam_checkbox', [0, 1], 'active')),
        ('fbeam', setv('5.beam_frequency_select', '2000')),
        ('beam1', setv('5.beam_checkbox', [1], 'active')),
        ('sfptap', select(results.venue_points_source, [2, 5])),
        ('sfpuntap', select(results.venue_points_source, [])),
        ('page4b', setv('app.page_select', 4, 'active')),
        ('w_ts', setv('4.weighting_select', 'Target Slope')),
        ('gh_hm2', setv('4.gap_handling_select', 'Hard Margin')),
        ('hinges', setv('4.hinges_select', '2')),
        ('tsel', lambda: algorithm.target_selected('indices', [], [0])),
        ('drag', drag),
        ('run_ts', setv('4.start_toggle', True, 'active')),
        ('clear', app.clear_result_figure),
        ('page1b', setv('app.page_select', 1, 'active')),
        ('remove_last', venue.remove_line),
        ('check2', app.check_venue),
        ('page0', setv('app.page_select', 0, 'active')),
    ]
