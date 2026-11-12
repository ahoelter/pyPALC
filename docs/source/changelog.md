# Changelog

## 0.1.0

The first release on PyPI (`pip install pypalc`).

* **GUI:** redesigned with Panel and panel-material-ui (formerly a Bokeh
  app), with Bokeh figures. Every browser window gets its own session.
* **Headless runner:** `pypalc run SETUP --out DIR` computes a setup file
  and writes the results as CSV files, with exit codes for the kinds of
  errors (see {doc}`command-line`).
* **Setup files:** everything entered in the GUI can be saved ("Save
  Setup") and loaded ("Load Setup", on pages 1 and 4) as a YAML setup file,
  which `pypalc run` and {func}`pypalc.core.settings.load_setup` read as
  well (see {doc}`setup-file`).
* **`pypalc gui`:** serves the GUI from the installed package, with the
  options `--port`, `--address`, `--allow-websocket-origin`, `--prefix` and
  `--show`/`--no-show` (see {doc}`gui` and {doc}`server`).
* **Examples:** example setups ship with the package. `pypalc examples DIR`
  copies them; page 1 of the GUI loads them ("Load Example").
* **Package:** the calculation (`pypalc.core`) imports only numpy, scipy and
  PyYAML; the GUI is installed with it, and `pypalc_gui` is a one-word
  command for `pypalc gui`. The code uses descriptive
  names instead of the paper notation (see {doc}`development`).

### Changed: the files of the GUI downloads

The GUI downloads ("Download PALC Angles" and "Download Results" on page 4)
use the file names and column names of the headless export and write the
numbers as it does. Before, every file was saved as `PALC_result.csv`.
"Download Results" (formerly "Download SPL") saves the five files other
than `tilt_angles.csv` in one zip file, `pypalc_results.zip`. Before the
first calculation the files hold the header only.

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
