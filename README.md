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
