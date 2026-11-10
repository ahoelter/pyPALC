# Setup file reference

A setup file holds everything that is entered on the pages of the GUI: the
venue, the loudspeaker, the array, the reference array, the algorithm
options and the target slope. It is a YAML file. The GUI writes it with
"Save Setup" and reads it with "Load Setup"; `pypalc run` computes it (see
{doc}`command-line`); in Python, {func}`pypalc.core.settings.load_setup`
reads it into a {class}`~pypalc.core.settings.Setup` (see {doc}`tutorial`).

## General rules

* The file is a YAML mapping with the sections `venue`, `loudspeaker`,
  `array`, `reference_array`, `algorithm` and `target_slope`, and the key
  `version`. Every section and every key may be left out; it then has its
  default value. Only the end points of an audience line and the fields of
  a target slope have no default.
* A key that is not listed here is an error ("unknown setting").
* Numbers may be written as integers or decimals (`2` or `2.0`); they must
  be finite. Where an integer is required, a decimal is an error. `true`
  and `false` are not numbers.
* Lists are YAML lists, written as `[1.0, 2.0]` or one entry per line
  starting with `-`. A point is a list of two numbers `[x, y]` in m: x is
  the horizontal coordinate, y the height.
* A key whose default is `null` may also be written as `null`, which is the
  same as leaving it out.
* Angles are in degree, lengths and coordinates in meter, frequencies in
  hertz and levels in decibel.
* The settings are checked when the file is loaded, by the GUI and by
  `pypalc run` alike. The first invalid setting is reported with its key,
  for example `setup error at algorithm.tolerance: must be between 0.05
  and 5, got 10`. A venue that cannot be built is reported when the setup
  is computed.

The file with every default value, as
{func}`~pypalc.core.settings.dump_setup` writes it:

```yaml
version: 1
venue:
  audience_lines: []
loudspeaker:
  aperture_angle_deg: 3.0
  cabinet_height: 0.2
  directivity: circular_piston
array:
  cabinet_count: 4
  top_position:
  - 0.0
  - 2.0
  cabinet_gap: 0.0
reference_array:
  type: straight
  top_angle_deg: 0.0
  angle_param_deg: 0.0
  use_discrete_angles: false
  tilt_angles_deg: []
algorithm:
  variant: palc2
  gap_handling: without
  soft_margin_strength: 1.0
  weighting: without
  weighting_nu: 1.0
  tolerance: 1.0
  frequency_range_hz:
  - 200.0
  - 8000.0
```

It cannot be computed as it is: a venue needs at least one audience line.
The example `simple.yaml` (see {doc}`command-line`) adds two.

## `version`

`version`
: Type: integer. Default: `1`. Allowed: `1`.

  Version of the setup file format.

## `venue`

The venue is a vertical slice through the audience area, given as straight
audience lines (GUI page 1).

`venue.audience_lines`
: Type: list of audience lines. Default: `[]`. Allowed: at least one line to compute.

  The audience lines, in the order in which they are drawn in the GUI.

`venue.audience_lines[].start`
: Type: point `[x, y]`, m. Default: required. Allowed: x ≥ 0, y ≥ 0.

  Start point of the line.

`venue.audience_lines[].stop`
: Type: point `[x, y]`, m. Default: required. Allowed: x ≥ start x, y ≥ start y.

  End point of the line; it differs from the start point.

`venue.audience_lines[].points_per_meter`
: Type: number, points per m. Default: `1.0`. Allowed: > 0.

  Discretization of the line: the line gets int(length × points_per_meter) audience positions.

An audience line must not lie on a line drawn before it, and the venue has
at most 2000 audience positions in all. These rules, and whether a PALC
compatible venue can be made from the lines, are checked when the setup is
computed (`pypalc run` exit code 3).

An audience line in YAML:

```yaml
venue:
  audience_lines:
  - {start: [2.0, 0.0], stop: [10.0, 1.0], points_per_meter: 5.0}
```

## `loudspeaker`

The loudspeaker cabinets (GUI page 2).

`loudspeaker.aperture_angle_deg`
: Type: number, degree. Default: `3.0`. Allowed: any.

  PALC aperture angle of the cabinets, the start value of the iterative calculation. A smaller value can help if PALC does not converge, for example when many cabinets cover a small venue.

`loudspeaker.cabinet_height`
: Type: number, m. Default: `0.2`. Allowed: > 0.

  Height of the front grille of a cabinet.

`loudspeaker.directivity`
: Type: string. Default: `circular_piston`. Allowed: `circular_piston`, `line_piston`, `combined_circular_line`, `measured`.

  Directivity of a cabinet: a modeled circular piston, line piston, or their combination (circular piston for low and line piston for high frequencies, crossover a 4th-order Linkwitz-Riley filter at 1.5 kHz), or measured data.

`loudspeaker.measured_directivity`
: Type: mapping. Default: `null`. Allowed: required for `measured`, with exactly one of `path` and `csv`.

  The measured directivity data; used only with `directivity: measured`.

`loudspeaker.measured_directivity.path`
: Type: string. Default: `null`. Allowed: a readable CSV file.

  The data as a file: a relative path is relative to the folder of the setup file.

`loudspeaker.measured_directivity.csv`
: Type: string. Default: `null`. Allowed: the CSV text.

  The data embedded in the setup file.

### Measured directivity

The measured data is a CSV file of complex directivity values:

* The first row holds the frequencies in Hz; its first entry is a `0` as a
  placeholder.
* The first column holds the radiation angles in degree.
* The other entries are the complex directivity values in cartesian form,
  for example `-0.0026485-0.027895i`; the imaginary unit is `i` or `j`.
* The data must not end with a line break: an empty last line makes it
  unreadable (`setup error at loudspeaker.measured_directivity: could not
  read the measured loudspeaker data`).

```text
0,20.508,23.438,26.367,...
-180,-0.0026485-0.027895i,-0.019059-0.010067i,-0.028558-0.013908i,...
-178,0.00091594-0.023526i,-0.021165-0.015261i,-0.024264-0.0062392i,...
```

The data is interpolated to the frequencies of the sound field prediction.
The repository has an example file,
`MeasuredLoudspeakerData_VerticalDirectivity_angle_frequency.csv`.

The data is given in one of two ways:

* `path`: a file next to the setup file (or anywhere, with an absolute
  path). Only `pypalc run` and {func}`~pypalc.core.settings.load_setup`
  read it; the GUI's "Load Setup" rejects a `path`, because the server
  cannot read files of the user's computer. An example setup with a `path`
  works in the GUI too: loading an example embeds the file.

  ```yaml
  loudspeaker:
    directivity: measured
    measured_directivity:
      path: measured.csv
  ```

* `csv`: the file's text embedded as a YAML block. "Save Setup" in the GUI
  writes uploaded data this way, and both the GUI and `pypalc run` read it.
  Start the block with `|-`, which drops the line break after the last
  line:

  ```yaml
  loudspeaker:
    directivity: measured
    measured_directivity:
      csv: |-
        0,100,1000
        -90,1+0j,0.5+0.1j
        0,1+0j,1+0j
        90,1+0j,0.5+0.1j
  ```

## `array`

The array of cabinets (GUI page 3).

`array.cabinet_count`
: Type: integer. Default: `4`. Allowed: 1 to 30.

  Number of cabinets of the array.

`array.top_position`
: Type: point `[x, y]`, m. Default: `[0.0, 2.0]`. Allowed: any.

  Top front edge of the uppermost cabinet. If the array is longer than y (cabinet_count × (cabinet_height + cabinet_gap)), y is raised to that length, rounded to 0.01 m, as the GUI does.

`array.cabinet_gap`
: Type: number, m. Default: `0.0`. Allowed: ≥ 0.

  Gap between two cabinets.

`array.discrete_tilt_angles_deg`
: Type: list of numbers, degree, or `null`. Default: `null`. Allowed: at least one angle.

  `null`: any angle between the cabinets. A list: the allowed angles between adjacent cabinets, e.g. `[0, 1, 2.5, 3, 5]`. With discrete angles a larger `algorithm.tolerance` is usually needed.

## `reference_array`

The array the PALC result is compared with (GUI page 3, card "Reference
array").

`reference_array.type`
: Type: string. Default: `straight`. Allowed: `straight`, `progressive`, `arc`, `user_defined`.

  `straight`: every cabinet has the tilt angle `top_angle_deg`. `progressive`: the angles between the cabinets grow linearly, from `top_angle_deg` at the uppermost to `angle_param_deg` at the lowest cabinet. `arc`: the same angle `angle_param_deg` between all cabinets, starting at `top_angle_deg`. `user_defined`: the angles `tilt_angles_deg`.

`reference_array.top_angle_deg`
: Type: number, degree. Default: `0.0`. Allowed: any.

  Tilt angle of the uppermost cabinet (`straight`, `progressive`, `arc`).

`reference_array.angle_param_deg`
: Type: number, degree. Default: `0.0`. Allowed: any.

  `progressive`: tilt angle of the lowest cabinet; `arc`: angle between adjacent cabinets. Not used by `straight` and `user_defined`.

`reference_array.use_discrete_angles`
: Type: boolean. Default: `false`. Allowed: `true`, `false`.

  `progressive` and `arc`: round each angle between cabinets to the nearest of `array.discrete_tilt_angles_deg` (if that list has at least two angles).

`reference_array.tilt_angles_deg`
: Type: list of numbers, degree. Default: `[]`. Allowed: at least one angle for `user_defined`.

  `user_defined`: the tilt angles of the cabinets from the uppermost downwards. Missing angles repeat the last one; surplus angles are dropped.

## `algorithm`

The PALC algorithm and its options (GUI page 4).

`algorithm.variant`
: Type: string. Default: `palc2`. Allowed: `palc1`, `palc2`, `palc3`.

  `palc1`: the PALC aperture angles of all cabinets are the same. `palc2`: the product of the aperture angle of a cabinet and the distance from its acoustic center to the audience line is the same for all cabinets. `palc3`: as `palc2`, with the tangent of the aperture angle instead of the angle.

`algorithm.gap_handling`
: Type: string. Default: `without`. Allowed: `without`, `hard_margin`, `soft_margin`.

  Gaps between audience lines. `without`: no lines in the gaps. `hard_margin`: the gaps are no audience lines and each continuous audience line is computed separately; not with `weighting: target_slope`. `soft_margin`: non-audience lines in the gaps, with a lower weighting.

`algorithm.soft_margin_strength`
: Type: number. Default: `1.0`. Allowed: 0.1 to 1.9.

  Strength of the soft margin; a higher value weights the gaps lower. Used only with `gap_handling: soft_margin`.

`algorithm.weighting`
: Type: string. Default: `without`. Allowed: `without`, `linear`, `logarithmic`, `target_slope`.

  Weighting that shapes the level over distance: none, linear or logarithmic spacing (strength `weighting_nu`), or a target slope (section `target_slope`).

`algorithm.weighting_nu`
: Type: number. Default: `1.0`. Allowed: 0.01 to 10.

  Weighting parameter ν of the linear and logarithmic weighting. Used only with `weighting: linear` or `logarithmic`.

`algorithm.tolerance`
: Type: number, m. Default: `1.0`. Allowed: 0.05 to 5.

  Abort criterion of the iterative calculation. If PALC does not converge, increase it first.

`algorithm.fixed_top_angle_deg`
: Type: number, degree, or `null`. Default: `null`. Allowed: any; limited to the possible range.

  `null`: the tilt angle of the uppermost cabinet is computed. A number: the uppermost cabinet gets this tilt angle. An angle outside the range that the venue and the array allow is set to the nearest limit of that range (see {func}`pypalc.core.pipeline.fixed_angle_limits`).

`algorithm.frequency_range_hz`
: Type: list of two numbers, Hz. Default: `[200.0, 8000.0]`. Allowed: 0 to 20000, lower below upper.

  Frequency range of the level over distance (`spl_over_distance.csv`) and thus of the target slope.

## `target_slope`

The target slope of the level over distance (GUI page 4, figure "SPL over
distance"). It is required with `algorithm.weighting: target_slope` and
not used otherwise.

`target_slope`
: Type: mapping or `null`. Default: `null`. Allowed: required for `weighting: target_slope`.

  The target slope.

`target_slope.points`
: Type: list of target points. Default: required. Allowed: at least 2 points; the GUI accepts at most the number of audience lines + 2.

  The points of the target slope: start, hinges and end, in order of distance.

`target_slope.points[].distance`
: Type: number, m. Default: required. Allowed: any.

  Distance of the point from the center of the array.

`target_slope.points[].level_db`
: Type: number, dB. Default: required. Allowed: any.

  Level of the target slope at that distance, relative as in the figure "SPL over distance".

`target_slope.reference_distance`
: Type: number, m. Default: required. Allowed: any.

  Distance of the reference point.

The points and the reference point are moved to the nearest audience
position of the level over distance before the optimization, as when they
are dragged in the GUI. {func}`pypalc.core.pipeline.initial_target_slope`
computes the target slope that the GUI proposes when "Target Slope" is
selected; it is a good start for your own points.

```yaml
algorithm:
  weighting: target_slope
target_slope:
  points:
  - {distance: 2.6, level_db: 0.0}
  - {distance: 11.1, level_db: -3.0}
  - {distance: 20.1, level_db: -6.0}
  reference_distance: 11.1
```
