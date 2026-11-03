# pyPALC
A [bokeh](https://bokeh.org) application for the geometrical optimization of Line Source Arrays.
The application is based on different scientific publication, which can be found in the help page when running the application.

### Try it!

Install dependencies in a conda environment:
`conda create -n pyPALC --file requirements.txt`

Or install depencencies with pip:
`pip install -r requirements_pip.txt`

Run the app locally:
`bokeh serve --show pyPALC`
"pyPALC" is the folder that contains the app.

The colors and fonts of the app are set in `pypalc/gui/theme.py`. After
changing them, regenerate `theme.yaml` and `templates/index.html` with
`python -m pypalc.gui.theme`.

More info soon!

### Structure

| path | content |
|---|---|
| `main.py` | entry point of `bokeh serve`: builds one `PalcApp` per browser session |
| `pypalc/core/` | the calculation without Bokeh: settings and setup files, venue, PALC, sound field prediction, target slope, pipeline, CSV export |
| `pypalc/gui/` | the Bokeh GUI: `PalcApp`, one module per page, plot ranges, setup save/load, theme |
| `pypalc/cli.py` | the headless runner (`python -m pypalc run ...`) |
| `examples/` | example setup files |
| `tests/` | the tests; `tests/cases/*.yaml` are setup files whose results recorded from the original code are in `tests/baseline/*.npz`, and `tests/gui_cascade_golden.json` holds the recorded widget states of the GUI |
| `templates/`, `static/`, `theme.yaml` | page template, logos and Bokeh theme of the GUI |
| `ext/` | JavaScript of the CSV downloads |

### Running the tests

In the conda environment from above, run from this folder:

`python -m pytest`

The full suite takes several minutes. `tests/test_pipeline_regression.py`
and `tests/test_gui_equivalence.py` compare the results of every case in
`tests/cases` with `tests/baseline`; a change of these numbers is a bug fix
and comes with new baseline files. Set `PYTHONNOUSERSITE=1` if pip packages
in `~/.local` would shadow those of the environment.

### Run without the GUI

Run a setup file headless and write the results as CSV:

`python -m pypalc run examples/simple.yaml --out results`

`--out` defaults to `./results`; it is created if missing and existing files
are overwritten. A relative measured-directivity `path` in the setup file is
resolved against the directory of the setup file. [`examples/simple.yaml`](examples/simple.yaml)
is a small example setup. Exit codes: 0 ok, 2 invalid setup (or unreadable
setup file), 3 venue error, 4 no convergence or calculation error; messages go
to stderr.

Output files and their columns:

| file | columns |
|---|---|
| `tilt_angles.csv` | `cabinet` (1-based), `tilt_angle_deg`, `inter_cabinet_angle_deg` |
| `reference_tilt_angles.csv` | `cabinet`, `tilt_angle_deg` |
| `spl_over_distance.csv` | `distance_m`, `spl_db`, `reference_spl_db`, `x_m`, `y_m` |
| `homogeneity.csv` | `frequency_hz`, `homogeneity_db`, `homogeneity_low_db`, `homogeneity_high_db`, `reference_homogeneity_db` |
| `sound_field.csv`, `reference_sound_field.csv` | `x_m`, `y_m`, then one column `spl_db_<f>hz` per frequency (f rounded to an integer) |
| `target_slope.csv` (target slope only) | `distance_m`, `level_db`, `x_m`, `y_m` |
| `target_reference.csv` (target slope only) | `distance_m`, `level_db`, `x_m`, `y_m` of the reference point |

### CSV header changes

The GUI downloads ("Download PALC Angles" and "Download SPL" on page 4) now
use the file names and column names of the headless export. Before, every
file was saved as `PALC_result.csv`.

| file (was `PALC_result.csv`) | old columns | new columns |
|---|---|---|
| `tilt_angles.csv` | `a_num_LS`, `b_gamma_tilt_deg`, `c_gamma_tilt_deg_diff` | `cabinet`, `tilt_angle_deg`, `inter_cabinet_angle_deg` |
| `spl_over_distance.csv` | `SPL`, `x`, `x_v`, `y_v` | `distance_m`, `spl_db`, `x_m`, `y_m` |
| `reference_spl_over_distance.csv` | `SPL`, `x`, `x_v`, `y_v` | `distance_m`, `reference_spl_db`, `x_m`, `y_m` |
| `target_slope.csv` | `SPL`, `x`, `x_v`, `y_v` | `distance_m`, `level_db`, `x_m`, `y_m` |
| `target_reference.csv` | `x_ref`, `SPL`, `x_v`, `y_v` | `distance_m`, `level_db`, `x_m`, `y_m` |
| `homogeneity.csv` | `H`, `f`, `H_dist_high`, `H_dist_low`, `H_str` | `frequency_hz`, `homogeneity_db`, `homogeneity_low_db`, `homogeneity_high_db`, `reference_homogeneity_db` |

The GUI saves the SPL over distance of the optimized and of the reference
array as two files; the headless `spl_over_distance.csv` has both
(`spl_db` and `reference_spl_db`).

### Names in the code

The code uses descriptive names instead of the paper notation; the table
with the paper symbols is in the docstring of `pypalc/core/__init__.py`.
Former names:

| former | now |
|---|---|
| `PAL`, `PALC_compatible_PAL` | `AudienceLines`, `PalcVenue` |
| `PALC_configuration`, `SFP_configuration` | `PalcConfig`, `SoundFieldConfig` |
| `Array_opt`, `Technical_meas_results` | `ArrayGeometry`, `ArrayEvaluation` |
| `Opt_weight`, `PALC_plotting`, `Plot_ranges` | `TargetSlopeState`, `RayPlotData`, `PlotRanges` |
| `N`, `Lambda_y`, `Lambda_gap` | `cabinet_count`, `cabinet_height`, `cabinet_gap` |
| `x_H`, `y_H` | `top_x`, `top_y` |
| `psi_n`, `gamma_n` | `aperture_angles`, `tilt_angles` (rad) |
| `gamma_tilt_deg`, `gamma_tilt_deg_diff` | `tilt_angles_deg`, `inter_cabinet_angles_deg` |
| `gamma_LSA`, `use_gamma_LSA` | `discrete_tilt_angles` (rad), `use_discrete_angles` |
| `constraint`, `strength_sm`, `last_angle_hm`, `freq_range` | `palc_variant`, `soft_margin_strength`, `hard_margin_last_angle`, `frequency_range_hz` |
| `pal`, `pal_no_nal`, `nal` | `segments`, `audience_segments`, `non_audience_lines` |
| `xline`, `yline`, `pal_index` | `points_x`, `points_y`, `line_end_indices` |
| `xline_start`, `xline_stop`, `yline_start`, `yline_stop`, `b` | `line_start_x`, `line_stop_x`, `line_start_y`, `line_stop_y`, `intercept` |
| `x_c_n`, `y_c_n` | `cabinet_centers_x`, `cabinet_centers_y` |
| `seg_pos`, `seg_pos_start`, `seg_pos_stop` | `hit_points`, `hit_points_start`, `hit_points_stop` |
| `p_SPL`, `f`, `H`, `H_dist` | `spl_db`, `frequencies`, `homogeneity_db`, `homogeneity_low_db` / `homogeneity_high_db` |
| `SPLoverX`, `Tech_res(_ref)` | `spl_over_distance`, `evaluation(_reference)` |
| `Opt_arr`, `Ref_arr` | `optimized_array`, `reference_array` |
| `created_pal`, `PALC_pal` | `drawn_lines`, `palc_venue` |
| target slope `x`, `SPL`, `x_v`, `y_v`, `x_ref`, `x_interp`, `SPL_interp` | `distances`, `levels_db`, `points_x`, `points_y`, `reference_distance`, `region_distances`, `region_levels_db` |
| `calcPALC`, `calc_angles`, `calcSFP`, `calcSPLoverX`, `optimize_PALC` | `run_palc`, `compute_tilt_angles`, `predict_sound_field`, `compute_spl_over_distance`, `optimize_target_slope` |
