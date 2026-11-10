# Python tutorial

`pypalc.core` is the calculation without the GUI; it needs only numpy,
scipy and PyYAML (`pip install pypalc`). This page runs a setup file from
Python, looks at the results and builds a setup in code. The code is in two
scripts, which the tests of pyPALC run:
{download}`run_setup.py <tutorial/run_setup.py>` and
{download}`build_setup.py <tutorial/build_setup.py>`. Run them in an empty
folder; they write files into the current folder.

The modules used here:

| module | contents |
|---|---|
| {mod}`pypalc.core.settings` | {class}`~pypalc.core.settings.Setup` and its parts, {func}`~pypalc.core.settings.load_setup`, {func}`~pypalc.core.settings.validate`, {func}`~pypalc.core.settings.save_setup`, {func}`~pypalc.core.settings.dump_setup` |
| {mod}`pypalc.core.pipeline` | {func}`~pypalc.core.pipeline.run`, {class}`~pypalc.core.pipeline.RunResult`, {func}`~pypalc.core.pipeline.result_arrays` |
| {mod}`pypalc.core.export` | {func}`~pypalc.core.export.write_results`, the CSV files of `pypalc run` |
| {mod}`pypalc.examples` | {func}`~pypalc.examples.load_example`, {func}`~pypalc.examples.copy_examples` |
| {mod}`pypalc.core.errors` | the errors: {class}`~pypalc.core.errors.SetupError`, {class}`~pypalc.core.errors.VenueError`, {class}`~pypalc.core.errors.ConvergenceError`, {class}`~pypalc.core.errors.CalculationError`, all subclasses of {class}`~pypalc.core.errors.PalcError` |

## Run a setup file

### Load the setup

{func}`~pypalc.examples.load_example` loads an example that ships with
pyPALC by its name; {func}`~pypalc.core.settings.load_setup` loads any
setup file (see {doc}`setup-file`). Both check the setup and raise a
{class}`~pypalc.core.errors.SetupError` (with the attributes `key` and
`message`) if it is invalid.

```{literalinclude} tutorial/run_setup.py
:language: python
:start-after: "# [load]"
:end-before: "# [run]"
```

### Compute it

{func}`pipeline.run <pypalc.core.pipeline.run>` computes the setup as the
GUI does after "PALC Run" and returns a
{class}`~pypalc.core.pipeline.RunResult`. Every call is independent of
earlier ones, and the setup is not changed. It raises a
{class}`~pypalc.core.errors.VenueError` if the venue cannot be built, a
{class}`~pypalc.core.errors.ConvergenceError` if PALC does not converge and
a {class}`~pypalc.core.errors.CalculationError` if the calculation fails
otherwise.

```{literalinclude} tutorial/run_setup.py
:language: python
:start-after: "# [run]"
:end-before: "# [arrays]"
```

The fields of {class}`~pypalc.core.pipeline.RunResult`:

`setup`
: The setup that was run.

`venue`
: A {class}`~pypalc.core.pipeline.VenueResult`: the drawn audience lines,
  discretized (`drawn_lines`, with the audience positions `points_x` and
  `points_y`), the PALC compatible venue (`palc_venue`), the non-audience
  lines suggested for the gaps and the lines used only by the sound field
  prediction.

`optimized`, `reference`
: An {class}`~pypalc.core.pipeline.ArrayResult` each, for the PALC
  optimized and the reference array: `tilt_angles_deg` and
  `inter_cabinet_angles_deg` (degree; for the optimized array rounded to
  0.01° as the GUI shows them), `evaluation` (the sound field prediction:
  `frequencies`, `spl_db` with one row per audience position and one column
  per frequency, `homogeneity_db` and the histogram of the bar charts),
  `spl_over_distance` (attributes `distances` and `levels_db`) and the
  outlines of the cabinets for plotting (`patches_x`, `patches_y`).

`geometry`
: The geometry of the optimized array (rays, hit points and
  `iteration_count`, the number of PALC iterations).

`ray_plot`
: The rays and cabinets of the optimized array for plotting.

`target`
: For target slope weighting the final target slope (`distances`,
  `levels_db`, the reference point), otherwise `None`.

`weighting_weights`
: The weighting factors of the cabinets in the final PALC run.

### The results as arrays

{func}`~pypalc.core.pipeline.result_arrays` collects the numbers of a
result as numpy arrays of floats in a dict:

```{literalinclude} tutorial/run_setup.py
:language: python
:start-after: "# [arrays]"
:end-before: "# [write]"
```

For `simple.yaml` (4 cabinets, 90 audience positions, 120 frequencies) it
holds:

| key | shape | contents |
|---|---|---|
| `tilt_angles_deg`, `inter_cabinet_angles_deg` | (4,) | angles of the optimized array in degree |
| `reference_tilt_angles_deg` | (4,) | tilt angles of the reference array in degree |
| `frequencies_hz` | (120,) | frequencies of the sound field prediction |
| `spl_db`, `reference_spl_db` | (90, 120) | level in dB at each audience position (rows) and frequency (columns) |
| `spl_over_distance_x` | (90,) | distance of each audience position from the array center in m |
| `spl_over_distance_db`, `reference_spl_over_distance_db` | (90,) | the level over distance in dB |
| `homogeneity_db`, `reference_homogeneity_db` | (120,) | homogeneity over frequency in dB |
| `homogeneity_low_db`, `homogeneity_high_db` | (120,) | reference lines of the homogeneity figure in dB |
| `histogram_steps`, `histogram_tops`, `reference_histogram_steps`, `reference_histogram_tops` | (n,) | the histogram of the bar charts |
| `venue_x`, `venue_y` | (90,) | the audience positions in m |

For target slope weighting it also holds `target_x` and `target_db` (the
target points) and `weighting_weights`. The columns of the CSV files are
explained in {doc}`command-line`.

### Write the CSV files

{func}`~pypalc.core.export.write_results` writes the files of
`pypalc run` into a folder (created if missing) and returns their paths:

```{literalinclude} tutorial/run_setup.py
:language: python
:start-after: "# [write]"
```

## Build a setup in code

A {class}`~pypalc.core.settings.Setup` is made of frozen dataclasses, one
per section of the setup file:
{class}`~pypalc.core.settings.VenueSettings` (with
{class}`~pypalc.core.settings.AudienceLine`),
{class}`~pypalc.core.settings.LoudspeakerSettings` (with
{class}`~pypalc.core.settings.MeasuredDirectivity`),
{class}`~pypalc.core.settings.ArraySettings`,
{class}`~pypalc.core.settings.ReferenceArraySettings`,
{class}`~pypalc.core.settings.AlgorithmSettings` and
{class}`~pypalc.core.settings.TargetSlopeSettings` (with
{class}`~pypalc.core.settings.TargetPoint`). Their fields are the keys of
the {doc}`setup-file`, with the same defaults; lists are tuples.

```{literalinclude} tutorial/build_setup.py
:language: python
:start-after: "# [build]"
:end-before: "# [validate]"
```

{func}`~pypalc.core.settings.validate` applies the checks of a setup file;
building the dataclasses does not check anything:

```{literalinclude} tutorial/build_setup.py
:language: python
:start-after: "# [validate]"
:end-before: "# [save]"
```

{func}`~pypalc.core.settings.dump_setup` returns the setup as the text of
a setup file and {func}`~pypalc.core.settings.save_setup` writes it to a
file; settings that are `None` are left out. The file can be run with
`pypalc run` and loaded in the GUI.

```{literalinclude} tutorial/build_setup.py
:language: python
:start-after: "# [save]"
:end-before: "# [run]"
```

Compute the setup and a variant of it:

```{literalinclude} tutorial/build_setup.py
:language: python
:start-after: "# [run]"
```
