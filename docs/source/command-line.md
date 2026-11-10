# Command line

Installing pyPALC adds the command `pypalc` with three subcommands:

| command | does |
|---|---|
| `pypalc run SETUP --out DIR` | computes a setup file and writes the results as CSV files |
| `pypalc examples DIR` | copies the example setup files into a folder |
| `pypalc gui` | serves the GUI and opens it in the browser (see {doc}`gui`) |

`python -m pypalc ...` is the same as `pypalc ...`, for example
`python -m pypalc run setup.yaml`. `pypalc --help` and
`pypalc run --help` list the options.

## Get the examples

```console
pypalc examples my_examples
```

copies the example setups that ship with pyPALC (with any data files they
need) into the folder `my_examples`, creates the folder if needed and
prints the path of every copied file. Nothing is overwritten: if one of the
files exists in the folder already, no file is copied and the command ends
with exit code 5.

The examples can also be downloaded here:
{download}`simple.yaml <../../pypalc/examples/simple.yaml>`.

## Run a setup file

```console
pypalc run my_examples/simple.yaml --out results
```

reads and checks the setup file, computes it as the GUI does after
"PALC Run" and writes the results into the folder `results`:

```text
results/tilt_angles.csv
results/reference_tilt_angles.csv
results/spl_over_distance.csv
results/homogeneity.csv
results/sound_field.csv
results/reference_sound_field.csv
```

Options:

`SETUP`
: The YAML setup file (see {doc}`setup-file`). The GUI writes such a file
  with "Save Setup" on page 4.

`--out DIR`
: Output folder, default `results` in the current folder. It is created if
  missing; existing files in it are overwritten.

`--traceback`
: On an error, print the full traceback (with its causes) in addition to
  the error message.

The paths of the written files go to standard output; error messages go
to standard error.

## The example `simple.yaml`

```{literalinclude} ../../pypalc/examples/simple.yaml
:language: yaml
```

* `version: 1` is the version of the setup file format.
* `venue.audience_lines` is the venue: two straight audience lines, the
  first from x = 2 m, y = 0 m to x = 10 m, y = 1 m, the second from there to
  x = 20 m, y = 3 m. x is the horizontal coordinate and y the height, in
  m. Each line is discretized with 5 points per meter; these points
  are the audience positions of the results.
* Everything else has its default value (see {doc}`setup-file`): an array
  of 4 cabinets of 0.2 m height without gaps, whose uppermost cabinet has
  its top front edge at x = 0 m, y = 2 m; the directivity of a circular
  piston; the PALC 2 variant without gap handling and without weighting,
  with a tolerance of 1 m; a straight reference array with a tilt angle of
  0°; and the frequency range 200 Hz to 8000 Hz for the level over
  distance.

`results/tilt_angles.csv` of this example:

```text
cabinet,tilt_angle_deg,inter_cabinet_angle_deg
1,-1.4,-1.4
2,2.05,3.45
3,7.54,5.49
4,17.21,9.67
```

## Result files

All files are CSV files with a header row, separated by commas. Numbers
are written so that they read back exactly (Python's `repr`). Units are in
the column names: `_m` meter, `_deg` degree, `_db` decibel, `_hz` hertz.
The GUI's downloads on page 4 ("Download PALC Angles", "Download Results")
use the same file and column names, except that the GUI saves the SPL over
distance of the reference array as a file of its own,
`reference_spl_over_distance.csv`.

`tilt_angles.csv`
: One row per cabinet of the optimized array, from the uppermost
  (cabinet 1) downwards.

  - `cabinet`: number of the cabinet, starting at 1.
  - `tilt_angle_deg`: tilt angle of the cabinet, rounded to 0.01°.
  - `inter_cabinet_angle_deg`: angle between the cabinet and the one above
    it, rounded to 0.01°; for cabinet 1 its tilt angle.

`reference_tilt_angles.csv`
: One row per cabinet of the reference array.

  - `cabinet`: number of the cabinet, starting at 1.
  - `tilt_angle_deg`: tilt angle of the cabinet.

`spl_over_distance.csv`
: The level over distance, one row per audience position.

  - `distance_m`: distance from the center of the optimized array to the
    position.
  - `spl_db`: level of the optimized array at the position over the
    frequency range `algorithm.frequency_range_hz`; a relative level, as in
    the figure "SPL over distance" on page 4 of the GUI.
  - `reference_spl_db`: the same for the reference array.
  - `x_m`, `y_m`: the position.

`homogeneity.csv`
: The homogeneity, one row per frequency: 120 frequencies from 20 Hz in
  steps of 1/12 octave (up to about 19.3 kHz).

  - `frequency_hz`: the frequency.
  - `homogeneity_db`: homogeneity of the optimized array, the difference
    between the 90 % and the 10 % quantile of the levels at all audience
    positions.
  - `homogeneity_low_db`, `homogeneity_high_db`: the reference lines of the
    homogeneity figure. `homogeneity_high_db` is 20·log10 of the ratio of
    the largest to the smallest distance between the array center and an
    audience position, `homogeneity_low_db` is half of it; both are the
    same for every frequency.
  - `reference_homogeneity_db`: homogeneity of the reference array.

`sound_field.csv`
: The predicted sound pressure level of the optimized array, one row per
  audience position.

  - `x_m`, `y_m`: the position.
  - `spl_db_<f>hz`: the level in dB at frequency f, one column per
    frequency of `homogeneity.csv`, with f rounded to an integer in the
    name (`spl_db_20hz`, `spl_db_21hz`, ...).

`reference_sound_field.csv`
: The same for the reference array.

`target_slope.csv`
: Only for target slope weighting (`algorithm.weighting: target_slope`).
  One row per point of the target slope (start, hinges, end), moved to the
  nearest audience position as in the GUI.

  - `distance_m`: distance from the array center.
  - `level_db`: level of the target slope.
  - `x_m`, `y_m`: the audience position of the point.

`target_reference.csv`
: Only for target slope weighting. One row: the reference point of the
  target slope.

  - `distance_m`: its distance from the array center.
  - `level_db`: the level of the target slope there.
  - `x_m`, `y_m`: the audience position nearest to that distance.

## Errors and exit codes

On an error `pypalc` prints one line to standard error that names the
kind of error and, for an invalid setup, the setting, for example

```text
setup error at array.cabinet_count: must be between 1 and 30, got 40
venue error: Please create audience lines first
```

If the error has a cause (for example the operating system's message when
a file cannot be read), it is appended as `(caused by ...)`. With
`--traceback` the full traceback is printed before the message.

| exit code | meaning |
|---|---|
| 0 | success |
| 2 | invalid setup: the file cannot be read, is not valid YAML or a setting is invalid (message `setup error at KEY: ...`); also wrong command line arguments |
| 3 | the venue cannot be built from the audience lines (message `venue error: ...`) |
| 4 | PALC did not converge or the calculation failed (message `calculation error: ...`) |
| 5 | the results (or, for `pypalc examples`, the examples) cannot be written (message `output error: ...`) |
| 6 | `pypalc gui` without the GUI dependencies; install them with `pip install "pypalc[gui]"` |

If PALC does not converge, a smaller `loudspeaker.aperture_angle_deg` or a
larger `algorithm.tolerance` usually helps (see {doc}`setup-file`).
