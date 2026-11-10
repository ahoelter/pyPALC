# pyPALC
A [Panel](https://panel.holoviz.org) application (with [Bokeh](https://bokeh.org) figures) for the geometrical optimization of Line Source Arrays.
The application is based on different scientific publication, which can be found in the help page when running the application.

### Try it!

pyPALC runs on Python 3.12 with Panel 1.10, panel-material-ui 0.16 and
Bokeh 3.10. Install the dependencies in a conda environment from the
conda-forge channel:
`conda create -n pyPALC3 -c conda-forge --file requirements.txt`
(an environment made for an earlier version of pyPALC has no Panel; create
it anew).

Or install the dependencies with pip, in a Python 3.12 environment (the
pinned versions are the same as in `requirements.txt`):
`pip install -r requirements_pip.txt`

If you have pip packages in `~/.local`, they shadow those of the
environment; set `PYTHONNOUSERSITE=1` for every command, e.g.
`PYTHONNOUSERSITE=1 conda run --no-capture-output -n pyPALC3 panel serve pyPALC --show`
(without `--no-capture-output`, `conda run` holds back the server log until
the server stops), or activate the environment first:
`conda activate pyPALC3` and `PYTHONNOUSERSITE=1 panel serve pyPALC --show`.

Run the app locally, from the folder that contains this folder:
`panel serve pyPALC --show`
"pyPALC" is the folder that contains the app.

The colors and themes of the app are set in `pypalc/gui/style.py`.

More info soon!

### Structure

| path | content |
|---|---|
| `main.py` | entry point of `panel serve`: builds one `PalcApp` per browser session and serves its view |
| `pypalc/core/` | the calculation without Bokeh: settings and setup files, venue, PALC, sound field prediction, target slope, pipeline, CSV export |
| `pypalc/gui/` | the GUI in Panel / panel-material-ui with Bokeh figures: `PalcApp` (page frame and switching, `app.py`), one module per page (the help texts in `help_text.py`, the sound field in `page_results.py`), plot ranges, the axis modes of the venue figure and the equal scaling of the beam plots (`ranges.py`), setup save/load, the files of the downloads (`downloads.py`), style, card and input building blocks (`widgets.py`) |
| `pypalc/cli.py` | the headless runner (`python -m pypalc run ...`) |
| `examples/` | example setup files |
| `tests/` | the tests; `tests/cases/*.yaml` are setup files whose results recorded from the original code are in `tests/baseline/*.npz`, and `tests/gui_reference.json` holds the application data of the former Bokeh GUI after each scripted GUI step (`tests/gui_steps.py`) |
| `tools/` | development tools: browser check with screenshots, comparison of screenshots and of GUI states (see "Browser tools") |
| `pypalc/gui/static/` | logos |

### Running the tests

In the conda environment from above, run from this folder:

`PYTHONNOUSERSITE=1 conda run --no-capture-output -n pyPALC3 python -m pytest`

The full suite takes several minutes. `tests/test_pipeline_regression.py`
and `tests/test_gui_equivalence.py` compare the results of every case in
`tests/cases` with `tests/baseline`; a change of these numbers is a bug fix
and comes with new baseline files. `PYTHONNOUSERSITE=1` keeps pip packages
in `~/.local` from shadowing those of the environment.

### Browser tools

The tests drive the GUI on the server. What only a browser does (the page
layout, the file input, the downloads, dragging, several tabs) is checked
by the tools in `tools/`. They are optional and not in the requirements:
install selenium into the environment
(`PYTHONNOUSERSITE=1 conda run --no-capture-output -n pyPALC3 pip install selenium`), and
Firefox and geckodriver (the default geckodriver path is
`/snap/bin/geckodriver`, which matches snap Firefox; pass `--geckodriver`
for another one). Run the tools from this folder:

- `python tools/browser_check.py --url http://localhost:5006/pyPALC --out shots [--width 1280] [--height H] [--setup FILE] [--two-tabs] [--download-dir DIR] [--clean] [--server-log LOG]`:
  checks a running app (`panel serve pyPALC` from the parent folder) in a
  headless Firefox window of the given size (default 1280 x 800, from a
  width of 1920 px on 1080 px high): screenshots of every page before and
  after loading a setup, loading through the "Load Setup" file input of page 4 and of page 1 (also an invalid file), the
  downloads ("Download PALC Angles", the zip of "Download Results" with the
  headers of its five CSV files, "Save Setup"), the running look of "PALC
  Run" during a run and during a load, dragging a target point, the rays on
  page 1, the equal scaling of the venue figure and of the beam plots
  (pixels per meter on x and y, ranges that are the data extent or the
  extent of the beam image, the size of the figure), the axis labels of the venue figure, the
  linked zoom of the beam plots, the toolbars, no horizontal scrolling of
  any page at the window width, that a value typed right before a click
  is not lost (with `--server-log`, the log file of the server, also that
  no exception is logged meanwhile) and, with `--two-tabs`, that two
  browser windows do not share a session. Writes `shots/*.png` (the whole page),
  `shots/report.json` and the downloaded files to `shots/downloads`;
  exits with 1 if a check failed. `shots` and a given download folder
  (`--download-dir`; by default the browser downloads into a new temporary
  folder in the home folder, which is removed afterwards; snap Firefox
  cannot write to `/tmp`) must be empty or new; the tool refuses to run
  otherwise. `--clean` first removes the files of an earlier run of the
  tool (the screenshots, `report.json` and the download files with their
  numbered copies such as `pypalc_setup(1).yaml`) from them and leaves
  every other file alone. The checks are listed in
  `docs/superpowers/gui-checklist.md`.
- `python tools/compare_screenshots.py OLD_DIR NEW_DIR compare.html [--title TITLE] [--jpeg QUALITY] [--max-width PX]`:
  puts two folders of screenshots side by side in one self-contained HTML
  page, one row per file name (`--jpeg` keeps the page small).
- `python tools/record_gui_reference.py OUT.json`: records the application
  data after each scripted GUI step (as `tests/gui_reference.json`).

After an upgrade of panel, panel-material-ui or bokeh, run
`tools/browser_check.py` (at 1280 and 1920 px): the GUI relies on some
private or internal APIs that the tests on the server cannot check
(`FileDownload._sync_data` and its `_transfer` watcher, the document
lock read by `pypalc.gui.widgets.locked`, the views of BokehJS that
`pypalc.gui.ranges.ScaledFigure` measures and lays out in the browser
(`view.get_child_view` of Panel's `JSComponent`, `bbox`, `frame.bbox` and
`compute_layout` of a Bokeh plot view), the frame size that a Bokeh plot
view writes to its model (`inner_width`, `inner_height`, read by
`pypalc.gui.ranges.equal_ticker` in the browser; the check reads the
drawn ticks through `tick_coords` and `dimension` of the axis views), CSS
on MUI, Bokeh and Tabulator class names).

### Run without the GUI

Run a setup file headless and write the results as CSV:

`python -m pypalc run examples/simple.yaml --out results`

`--out` defaults to `./results`; it is created if missing and existing files
are overwritten. A relative measured-directivity `path` in the setup file is
resolved against the directory of the setup file. [`examples/simple.yaml`](examples/simple.yaml)
is a small example setup. Exit codes: 0 ok, 2 invalid setup (or unreadable
setup file), 3 venue error, 4 no convergence or calculation error, 5 output
could not be written; messages go to stderr, with the cause of the error.
`--traceback` also prints the full traceback.

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

The GUI downloads ("Download PALC Angles" and "Download Results" on page 4)
now use the file names and column names of the headless export, and write
the numbers as it does. Before, every file was saved as `PALC_result.csv`.
"Download Results" (formerly "Download SPL") saves the five files other than
`tilt_angles.csv` in one zip file, `pypalc_results.zip`. Before the first
calculation the files hold the header only.

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
