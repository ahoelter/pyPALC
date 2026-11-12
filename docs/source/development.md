# Development

This page is for working on pyPALC itself, in a clone of the repository
with the pinned dependencies (see {doc}`installation`, "From source"). The
commands run from the repository folder, in the conda environment
`pyPALC3`; `PYTHONNOUSERSITE=1` keeps pip packages in `~/.local` from
shadowing those of the environment.

## Structure

| path | content |
|---|---|
| `main.py` | entry point of `panel serve` from a checkout: serves `pypalc.gui.serve.create_view()` (one `PalcApp` per browser session) |
| `pypalc/core/` | the calculation without Bokeh: settings and setup files, venue, PALC, sound field prediction, target slope, pipeline, CSV export |
| `pypalc/gui/` | the GUI in Panel / panel-material-ui with Bokeh figures: `PalcApp` (page frame and switching, `app.py`), one module per page (the help texts in `help_text.py`, the sound field in `page_results.py`), plot ranges, the axis modes of the venue figure and the equal scaling of the beam plots (`ranges.py`), setup save/load, the files of the downloads (`downloads.py`), style, card and input building blocks (`widgets.py`); see {doc}`gui` |
| `pypalc/gui/static/` | logos |
| `pypalc/cli.py` | the command line interface (`pypalc run`, `pypalc examples`, `pypalc gui`; also `python -m pypalc ...`) |
| `pypalc/examples/` | example setup files (`pypalc examples DIR` copies them) |
| `tests/` | the tests; `tests/cases/*.yaml` are setup files whose results recorded from the original code are in `tests/baseline/*.npz`, and `tests/gui_reference.json` holds the application data of the former Bokeh GUI after each scripted GUI step (`tests/gui_steps.py`) |
| `tools/` | development tools: browser check with screenshots, comparison of screenshots and of GUI states (see "Browser tools") |
| `docs/source/` | this documentation (Sphinx, Markdown pages) |

## Running the tests

```console
PYTHONNOUSERSITE=1 conda run --no-capture-output -n pyPALC3 python -m pytest
```

The full suite takes several minutes. `tests/test_pipeline_regression.py`
and `tests/test_gui_equivalence.py` compare the results of every case in
`tests/cases` with `tests/baseline`; a change of these numbers is a bug fix
and comes with new baseline files.

The tests also keep this documentation honest:
`tests/test_docs_content.py` checks that {doc}`setup-file` names every
setting of {mod}`pypalc.core.settings` and no other, runs the scripts of
the {doc}`tutorial`, and checks that {doc}`command-line` names every exit
code and every result file; `tests/test_docs_build.py` builds the
documentation with warnings as errors (it needs the `docs` extra, see
below, and is skipped without it).

## Browser tools

The tests drive the GUI on the server. What only a browser does (the page
layout, the file input, the downloads, dragging, several tabs) is checked
by the tools in `tools/`. They are optional and not in the requirements:
install selenium into the environment
(`PYTHONNOUSERSITE=1 conda run --no-capture-output -n pyPALC3 pip install selenium`),
and Firefox and geckodriver (the default geckodriver path is
`/snap/bin/geckodriver`, which matches snap Firefox; pass `--geckodriver`
for another one). Run the tools from the repository folder:

`python tools/browser_check.py --url http://localhost:5006/pyPALC --out shots [--width 1280] [--height H] [--setup FILE] [--two-tabs] [--download-dir DIR] [--clean] [--server-log LOG]`
: Checks a running app (`panel serve pyPALC` from the parent folder) in a
  headless Firefox window of the given size (default 1280 x 800, from a
  width of 1920 px on 1080 px high): screenshots of every page before and
  after loading a setup, loading through the "Load Setup" file input of
  page 4 and of page 1 (also an invalid file), "Load Example" on page 1,
  the downloads ("Download PALC Angles", the zip of "Download Results" with
  the headers of its five CSV files, "Save Setup"), the running look of
  "PALC Run" during a run and during a load, dragging a target point, the
  rays on page 1, the equal scaling of the venue figure and of the beam
  plots (pixels per meter on x and y, ranges that are the data extent or
  the extent of the beam image, the size of the figure), the axis labels
  of the venue figure, the linked zoom of the beam plots, the toolbars, no
  horizontal scrolling of any page at the window width, that a value typed
  right before a click is not lost (with `--server-log`, the log file of
  the server, also that no exception is logged meanwhile) and, with
  `--two-tabs`, that two browser windows do not share a session.

  Writes `shots/*.png` (the whole page), `shots/report.json` and the
  downloaded files to `shots/downloads`; exits with 1 if a check failed.
  `shots` and a given download folder (`--download-dir`; by default the
  browser downloads into a new temporary folder in the home folder, which
  is removed afterwards; snap Firefox cannot write to `/tmp`) must be empty
  or new; the tool refuses to run otherwise. `--clean` first removes the
  files of an earlier run of the tool (the screenshots, `report.json` and
  the download files with their numbered copies such as
  `pypalc_setup(1).yaml`) from them and leaves every other file alone. The
  checks are listed in `docs/superpowers/gui-checklist.md`.

`python tools/compare_screenshots.py OLD_DIR NEW_DIR compare.html [--title TITLE] [--jpeg QUALITY] [--max-width PX]`
: Puts two folders of screenshots side by side in one self-contained HTML
  page, one row per file name (`--jpeg` keeps the page small).

`python tools/record_gui_reference.py OUT.json`
: Records the application data after each scripted GUI step (as
  `tests/gui_reference.json`).

### After upgrading Panel, panel-material-ui or Bokeh

Run `tools/browser_check.py` (at 1280 and 1920 px): the GUI relies on some
private or internal APIs that the tests on the server cannot check:

* `FileDownload._sync_data` and its `_transfer` watcher,
* the document lock read by `pypalc.gui.widgets.locked`,
* the views of BokehJS that `pypalc.gui.ranges.ScaledFigure` measures and
  lays out in the browser (`view.get_child_view` of Panel's `JSComponent`,
  `bbox`, `frame.bbox` and `compute_layout` of a Bokeh plot view),
* the frame size that a Bokeh plot view writes to its model
  (`inner_width`, `inner_height`, read by `pypalc.gui.ranges.equal_ticker`
  in the browser; the check reads the drawn ticks through `tick_coords` and
  `dimension` of the axis views),
* CSS on MUI, Bokeh and Tabulator class names.

## Building the documentation

Install the `docs` extra (Sphinx, Furo, MyST-Parser) into the environment,
for example with `pip install -e ".[docs]"`, and build the HTML pages:

```console
python -m sphinx -W -b html docs/source docs/_build/html
```

`-W` turns warnings into errors, as on Read the Docs
(`.readthedocs.yaml`, `fail_on_warning: true`). The pages are Markdown
files (MyST) in `docs/source/`; the API reference is generated from the
docstrings (NumPy style) into `docs/source/api/generated/`, which is not
in git. `docs/_build/` is not in git either. Open
`docs/_build/html/index.html` to read the result.

## Names in the code

The code uses descriptive names instead of the paper notation; the table
with the paper symbols is in the docstring of {mod}`pypalc.core`. Former
names:

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

## Releasing

Releases are published to PyPI by the GitHub workflow
`.github/workflows/publish.yml`, using Trusted Publishing: no password or
token is stored anywhere. The workflow builds the package, runs
`twine check` and, depending on how it was started, uploads to TestPyPI or
PyPI.

### One-time setup

1. On [pypi.org](https://pypi.org/manage/account/publishing/), open
   "Publishing" in your account settings and add a *pending publisher*
   with these values: PyPI project name `pypalc`, owner `ahoelter`,
   repository `pyPALC`, workflow name `publish.yml`, environment name
   `pypi`.
2. Do the same on [test.pypi.org](https://test.pypi.org/manage/account/publishing/)
   (a separate account is needed there), with environment name `testpypi`.
3. In the GitHub repository, open Settings, Environments, and create the
   environment `testpypi`.
4. Create the environment `pypi` as well. Optionally add yourself as a
   required reviewer, so that every upload to PyPI has to be approved by
   hand.

The first successful upload turns each pending publisher into a normal
trusted publisher and creates the project.

### Each release

1. Update `__version__` in `pypalc/__init__.py` and add the entry to
   {doc}`changelog`.
2. Run the tests and build the documentation.
3. Merge the changes into `master`.
4. In GitHub, open Actions, "Publish", "Run workflow" on `master`. This
   uploads to TestPyPI (approve the `testpypi` environment if asked).
5. Check the upload in a fresh environment:

   ```bash
   pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ pypalc
   ```

   Then start `pypalc gui` (and try `pypalc_gui --help`), load an example on page 1, and run
   `pypalc examples my_examples` followed by
   `pypalc run my_examples/simple.yaml`.

6. Tag the release and push the tag (here for version 0.1.0):

   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

   The workflow first checks that the tag equals `v` plus `__version__`
   (`tools/check_version_tag.py`), then uploads to PyPI (approve the `pypi`
   environment if you added a reviewer).

A version can be uploaded only once, so a faulty release is fixed by a new
version number.
