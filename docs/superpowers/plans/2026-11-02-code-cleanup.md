# pyPALC Code Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure pyPALC into a Bokeh-free computation core plus a per-session GUI, with a shared pipeline, a YAML setup file, a headless runner, GUI save/load and descriptive names. The app's layout and behavior stay the same.

**Architecture:** Approach A from the spec, an incremental refactor:
1. Record the current results as tests.
2. Move the computation into `pypalc/core` behind one `run()` pipeline.
3. Point the GUI at that pipeline.
4. Add the CLI.
5. Split the GUI into page modules owned by a per-session `PalcApp`.
6. Add save/load.
7. Rename.

The app runs and `pytest` passes after every task.

**Tech Stack:** Python 3.8 (conda env `pyPALC`), Bokeh 2.2.x, numpy 1.20, scipy 1.6, colorcet, PyYAML 6, pytest 8.

**Spec:** `docs/superpowers/specs/2026-10-30-code-cleanup-design.md`

## Global Constraints

- Run everything in the conda env `pyPALC`: `conda run -n pyPALC <cmd>`, or activate it first. Python 3.8, so no `match`, no `X | Y` types, and no `list[int]` at runtime. Use `from __future__ import annotations` in new modules.
- No layout changes. The only new widgets are the save and load buttons on page 4.
- No Bokeh upgrade. Avoid *new* uses of APIs that Bokeh 3 removed: `plot_width`/`plot_height`, `Panel`, `FuncTickFormatter`, `toolbar_sticky`. Existing uses may stay until project 2.
- `pypalc/core/**` must not import `bokeh`. A test enforces this from Task 3 on.
- No per-session state at module level in any imported module. That covers configuration objects, `ColumnDataSource`s, widgets and figures. Everything per session lives on a `PalcApp` instance built by `main.py`.
- `bokeh serve --show pyPALC`, run from the parent directory, must keep working.
- Measured-directivity CSV format unchanged. The setup file holds it either as `path` or as embedded `csv` text.
- Leave `docs/source`, `docs/build` and `docs/Makefile` untouched.
- Bug-fix protocol: one bug per commit, message `fix: <bug> (changes results: <which cases>)`. The same commit regenerates the affected `tests/baseline/*.npz`. No other commit may change baseline files.
- No `Co-Authored-By` or other attribution lines in commit messages.
- New code and new public names use the descriptive names from the naming table in Task 8 right away. Legacy internals are renamed in Task 8.
- Docstrings: numpy style (napoleon), like the existing code. Every public function and class in `pypalc` has one.

## Review Focus

1. **A venue with a gap, combined with target slope and soft margin.** The target-slope loop must converge or stop at `max_loops=200` and still produce results. It must not raise. Pinned in Task 2 (case `gap_soft_margin_target_slope`).
2. **A setup file with a measured directivity given as a relative `path`, run from a different working directory.** The path resolves relative to the setup file. Pinned in Task 4 (`test_cli_resolves_relative_directivity_path`).
3. **Loading a setup file in the GUI that uses `path` for measured directivity.** The GUI shows a clear error and changes no state. Pinned in Task 7 (`test_apply_setup_rejects_path_directivity`).
4. **Two sessions in one server process.** Drawing a line in one must not appear in the other. This covers the mutable default arguments in the legacy classes (`PAL(..., xline=[])`, `PALC_compatible_PAL(pal=[])`). Pinned in Task 5 (`test_sessions_do_not_share_state`).
5. **Fixed top angle outside the allowed range in a setup file.** The headless runner clamps it the same way the GUI does (`round_fixed_angle`), so GUI and CLI agree. Pinned in Task 3 (`test_fixed_angle_is_clamped`).

---

## File map (end state)

```
main.py                         ~20 lines: sys.path, theme check, PalcApp(curdoc()).show()
pypalc/__init__.py              version string
pypalc/__main__.py              from pypalc.cli import main; main()
pypalc/cli.py                   argparse: `run SETUP --out DIR`
pypalc/core/__init__.py         package docstring incl. name ↔ paper-symbol table
pypalc/core/errors.py           PalcError and subclasses
pypalc/core/settings.py         dataclasses + YAML load/dump + validation
pypalc/core/venue.py            AudienceLines, PalcVenue, draw conditions, venue proposal (was PALC_classes.PAL*, checkPAL, gui_venue logic)
pypalc/core/geometry.py         repmat, source positions, intersections, LSA patches, reference-array curvings (was PALC_functions minus calc_angles)
pypalc/core/palc.py             PalcConfig, ArrayGeometry, RayPlotData, calc_angles, run_palc (was calcPALC + calc_angles + config classes)
pypalc/core/sound_field.py      SoundFieldConfig, ArrayEvaluation, CDPS model, directivity, measured-directivity parser (was sfp_functions + read_dir_data)
pypalc/core/target_slope.py     TargetSlopeState, optimize loop (was PALC_opt + optimize_PALC + Opt_weight)
pypalc/core/pipeline.py         build_venue, run, initial_target_slope, fixed_angle_limits
pypalc/core/export.py           write_results(result, out_dir)
pypalc/gui/__init__.py
pypalc/gui/app.py               PalcApp: session state, shared widgets, page switching (was steps()), callbacks that span pages
pypalc/gui/theme.py             was create_layout.py
pypalc/gui/help_text.py         was fun/gui_help_text.py
pypalc/gui/ranges.py            PlotRanges (was Plot_ranges), set_ranges, axis scaling helpers
pypalc/gui/page_venue.py        page 1
pypalc/gui/page_loudspeaker.py  page 2
pypalc/gui/page_array.py        page 3
pypalc/gui/page_algorithm.py    page 4, incl. save/load setup
pypalc/gui/page_results.py      page 5
pypalc/gui/setup_io.py          setup_from_app, apply_setup
ext/download.js                 unchanged mechanism; also download_text.js for YAML
tests/conftest.py
tests/cases/*.yaml              baseline setups
tests/baseline/*.npz            recorded results
tests/legacy/capture.py         drives the legacy fun/ code (deleted in Task 9)
tests/test_*.py
```

`fun/` and `create_layout.py` are gone at the end. Keep `create_layout.py` as a one-line shim that calls `pypalc.gui.theme.write_theme()` only if something still references it. Otherwise delete it.

---

### Task 1: Tooling and settings with YAML setup files

**Files:**
- Create: `pypalc/__init__.py`, `pypalc/core/__init__.py`, `pypalc/core/errors.py`, `pypalc/core/settings.py`, `tests/conftest.py`, `tests/test_settings.py`, `pytest.ini`
- Modify: `requirements.txt` (add `pyyaml=6.0.1`, `pytest`), `requirements_pip.txt` (add `pyyaml==6.0.1`, `pytest`), `.gitignore` (add `.pytest_cache/`, `__pycache__/`)

**Interfaces:**
- Produces:
  - `errors.py`:
    - `class PalcError(Exception)`
    - `class SetupError(PalcError)`, with attribute `key: str` holding a dotted path such as `"array.cabinet_count"`
    - `class VenueError(PalcError)`
    - `class ConvergenceError(PalcError)`
    - `class CalculationError(PalcError)`
  - `settings.py`: frozen dataclasses with exactly these fields and defaults. The defaults equal today's GUI defaults.

    ```python
    @dataclass(frozen=True)
    class AudienceLine:
        start: Tuple[float, float]          # (x, y) in m
        stop: Tuple[float, float]
        points_per_meter: float = 1.0       # discretization of this line, as entered when it was saved

    @dataclass(frozen=True)
    class VenueSettings:
        audience_lines: Tuple[AudienceLine, ...] = ()

    @dataclass(frozen=True)
    class MeasuredDirectivity:
        path: Optional[str] = None          # exactly one of path / csv is set
        csv: Optional[str] = None           # raw CSV text in the upload format

    @dataclass(frozen=True)
    class LoudspeakerSettings:
        aperture_angle_deg: float = 3.0     # was psi_n (GUI field 9)
        cabinet_height: float = 0.2         # was Lambda_y
        directivity: str = "circular_piston"  # circular_piston | line_piston | combined_circular_line | measured
        measured_directivity: Optional[MeasuredDirectivity] = None  # required iff directivity == "measured"

    @dataclass(frozen=True)
    class ArraySettings:
        cabinet_count: int = 4              # 1..30
        top_position: Tuple[float, float] = (0.0, 2.0)   # was (x_H, y_H)
        cabinet_gap: float = 0.0            # was Lambda_gap
        discrete_tilt_angles_deg: Optional[Tuple[float, ...]] = None  # None = continuous

    @dataclass(frozen=True)
    class ReferenceArraySettings:
        type: str = "straight"              # straight | progressive | arc | user_defined
        top_angle_deg: float = 0.0          # GUI "Array Tilt Angle" / "Start Angle"
        angle_param_deg: float = 0.0        # progressive: final angle; arc: inter-cabinet angle; unused otherwise
        use_discrete_angles: bool = False   # progressive/arc only
        tilt_angles_deg: Tuple[float, ...] = ()  # user_defined only

    @dataclass(frozen=True)
    class AlgorithmSettings:
        variant: str = "palc2"              # palc1 | palc2 | palc3
        gap_handling: str = "without"       # without | hard_margin | soft_margin
        soft_margin_strength: float = 1.0   # 0.1..1.9
        weighting: str = "without"          # without | linear | logarithmic | target_slope
        weighting_nu: float = 1.0           # the actual ν (GUI slider value cubed, rounded to 2 decimals), 0.01..10
        tolerance: float = 1.0              # m, 0.05..5
        fixed_top_angle_deg: Optional[float] = None  # None = not fixed
        frequency_range_hz: Tuple[float, float] = (200.0, 8000.0)

    @dataclass(frozen=True)
    class TargetPoint:
        distance: float                     # m, x-axis of SPL-over-distance plot
        level_db: float                     # dB, same normalization as that plot

    @dataclass(frozen=True)
    class TargetSlopeSettings:
        points: Tuple[TargetPoint, ...]     # ≥ 2, start, hinges, end
        reference_distance: float           # m

    @dataclass(frozen=True)
    class Setup:
        venue: VenueSettings = VenueSettings()
        loudspeaker: LoudspeakerSettings = LoudspeakerSettings()
        array: ArraySettings = ArraySettings()
        reference_array: ReferenceArraySettings = ReferenceArraySettings()
        algorithm: AlgorithmSettings = AlgorithmSettings()
        target_slope: Optional[TargetSlopeSettings] = None   # required iff weighting == "target_slope"
    ```

  - Functions:
    - `SETUP_VERSION = 1`
    - `load_setup(path: Union[str, Path]) -> Setup`. Resolves a relative `measured_directivity.path` against `path.parent` and stores it as an absolute path.
    - `load_setup_text(text: str, base_dir: Optional[Path] = None) -> Setup`. With `base_dir=None`, a `path` directivity raises `SetupError(key="loudspeaker.measured_directivity.path")`.
    - `dump_setup(setup: Setup) -> str`
    - `save_setup(setup: Setup, path) -> None`

  - YAML shape: top-level keys `version, venue, loudspeaker, array, reference_array, algorithm, target_slope`. Field names exactly as above. Tuples are written as lists. `AudienceLine` is written as `{start: [x, y], stop: [x, y], points_per_meter: n}`. `TargetPoint` is written as `{distance, level_db}`. `csv` is written as a YAML literal block (`|`).

- [ ] **Step 1: Write failing tests in `tests/test_settings.py`**
  - `test_defaults_round_trip`: `load_setup_text(dump_setup(Setup())) == Setup()`.
  - `test_full_round_trip`: a Setup with two lines, measured `csv`, discrete angles, arc reference, soft margin, target slope with 3 points and fixed angle 2.5 survives `dump` → `load_setup_text` unchanged.
  - `test_unknown_key_names_key`: YAML with `array: {cabinets: 4}` raises `SetupError` whose `.key == "array.cabinets"`.
  - `test_wrong_type_names_key`: `array: {cabinet_count: "four"}` → `.key == "array.cabinet_count"`.
  - `test_out_of_range`, parametrized:

    | value | expected `.key` |
    |---|---|
    | `cabinet_count: 0` | `array.cabinet_count` |
    | `cabinet_count: 31` | `array.cabinet_count` |
    | `tolerance: 0` | `algorithm.tolerance` |
    | `variant: palc4` | `algorithm.variant` |

  - `test_target_slope_required`: weighting `target_slope` without a `target_slope` section → `.key == "target_slope"`.
  - `test_target_slope_with_hard_margin_rejected`: `.key == "algorithm.gap_handling"`. The GUI forbids this combination.
  - `test_measured_requires_exactly_one_source`: both `path` and `csv`, or neither → `.key == "loudspeaker.measured_directivity"`.
  - `test_relative_path_resolved_against_file(tmp_path)`: `load_setup(tmp_path/"sub"/"s.yaml")` with `path: d.csv` → `tmp_path/"sub"/"d.csv"`, as an absolute path.
  - `test_text_load_rejects_path_without_base_dir`.
  - `test_version_mismatch`: `version: 2` → `.key == "version"`.

- [ ] **Step 2: Run** `pytest tests/test_settings.py -v`. Expected: all FAIL with `ModuleNotFoundError: pypalc`.

- [ ] **Step 3: Implement `errors.py` and `settings.py`.** Use `yaml.safe_load` and `yaml.safe_dump(sort_keys=False)`. Write a small recursive `_build(cls, data, key_prefix)` over `dataclasses.fields` that checks unknown keys and types. Range and cross-field checks go in a `validate(setup)` called at the end of loading. `pytest.ini`: `testpaths = tests`.

- [ ] **Step 4: Run** `pytest -v`. Expected: all PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add pypalc tests pytest.ini requirements.txt requirements_pip.txt .gitignore
  git commit -m "Add setup settings with YAML load/save and validation"
  ```

---

### Task 2: Record baseline results from the legacy code

**Files:**
- Create:
  - `tests/cases/*.yaml`, one per case below
  - `tests/legacy/__init__.py`
  - `tests/legacy/capture.py`
  - `tests/baseline/<case>.npz`
  - `tests/test_baseline_capture.py`
  - `tests/cases/measured.csv` (copy of `MeasuredLoudspeakerData_VerticalDirectivity_angle_frequency.csv`)

**Interfaces:**
- Consumes: `load_setup` from Task 1.
- Produces:
  - `tests/legacy/capture.py: run_legacy(setup: Setup) -> Dict[str, np.ndarray]`
  - CLI: `python -m tests.legacy.capture [CASE ...]` writes `tests/baseline/<case>.npz`
  - Result-key contract, used by every later regression test (each value is an ndarray):

    | key(s) | content |
    |---|---|
    | `tilt_angles_deg`, `inter_cabinet_angles_deg` | from `results_dict` (rounded as the GUI rounds) |
    | `reference_tilt_angles_deg` | |
    | `frequencies_hz` | `Tech_res.f` |
    | `spl_db` | `Tech_res.p_SPL` (positions × frequencies) |
    | `reference_spl_db` | |
    | `spl_over_distance_x`, `spl_over_distance_db` | `SPLoverX_dict` x/SPL after the GUI post-processing in `start_calc` |
    | `reference_spl_over_distance_db` | |
    | `homogeneity_db`, `homogeneity_low_db`, `homogeneity_high_db`, `reference_homogeneity_db` | |
    | `histogram_steps`, `histogram_tops`, `reference_histogram_steps`, `reference_histogram_tops` | |
    | `venue_x`, `venue_y` | `created_pal.xline`/`yline` |
    | `target_x`, `target_db` | target-slope cases only: `Opt_SPLoverX_dict` after the run |
    | `weighting_weights` | target-slope cases only |

- Cases. Every case starts from GUI defaults plus what's listed:

  | case | venue | settings |
  |---|---|---|
  | `simple` | lines (2,0)→(10,1), (10,1)→(20,3); 5 pts/m | defaults |
  | `gap_without` | (2,0)→(10,1), (12,2)→(20,4); 5 pts/m; 6 cabinets | |
  | `gap_hard_margin` | as `gap_without` | hard margin |
  | `gap_soft_margin` | as `gap_without` | soft margin, strength 0.5 |
  | `palc1`, `palc3` | as `simple` | variant palc1 / palc3 |
  | `discrete_angles` | as `simple` | `[0,0.5,1,2,3.5,5,7,10]`; reference progressive 0→10 with discrete angles |
  | `fixed_angle` | as `simple` | `fixed_top_angle_deg` = midpoint of `fixed_angle_limits` |
  | `measured` | as `simple` | measured directivity `path: measured.csv`; reference arc 0/2 |
  | `linear_weighting` | as `simple` | linear, ν = 2.0 |
  | `log_weighting` | as `simple` | logarithmic, ν = 0.5 |
  | `target_slope` | as `simple` | 1 hinge; points (first x, 0 dB), (middle x, −3 dB), (last x, −6 dB); reference at middle x |
  | `gap_soft_margin_target_slope` | as `gap_soft_margin` | target slope as in `target_slope` |

  - **`fixed_angle` midpoint:** the half-way point between the max and min that `use_fixed_angle` computes. Record the computed value in the YAML.
  - **Target slope x values:** they are snapped to the grid. Record the snapped values in the YAML.
  - **Reference array:** a straight array at 0° unless the table says otherwise.

- [ ] **Step 1: Write `run_legacy`.** It imports the `fun/` modules (`sys.path.insert(0, <repo>/fun)`) and replays exactly what the GUI does, in this order:
  1. Initial directivity setup (`main.py:180-194`).
  2. `store_config` for every changed field, in GUI order: N, psi_n, Lambda_y, gap, x_H, y_H, directivity, algorithm, gap handling, strength, tolerance, freq_range. Then `weighting_nu`, `use_gamma_LSA`/`gamma_LSA`, and the reference array via `ref_array_angles`. Pass simple stand-in objects with a `.value` attribute in place of widgets.
  3. For each line: `check_all_draw_conditions`, then `created_pal.append_line(..., diff*points_per_meter)`. Use a stand-in text widget.
  4. The `check_PAL` body without widgets.
  5. Fixed angle: `get_fixed_angle_borders` → `Opt_arr.fixed_angle`, then the clamping from `round_fixed_angle`.
  6. Non-target cases: the `start_calc` body.
  7. Target slope cases:
     - Run `start_calc` once with `Opt_w.init=False`, which gives `SPLoverX` and the defaults.
     - Then set `Opt_w.init=True`.
     - Then emulate `chg_weight_links(n_hinges)` and `update_draw` with the case's points and reference distance.
     - Then run `start_calc` again.
  8. Collect the result keys.

  Fresh objects per call. Pass explicit fresh lists where the legacy constructors have mutable defaults.

- [ ] **Step 2: Write `tests/test_baseline_capture.py::test_capture_is_deterministic[case]`.** It runs `run_legacy` twice per case and asserts every key is equal (`np.testing.assert_array_equal`). Run `pytest tests/test_baseline_capture.py -v`. Expected: PASS for all cases.

  If a case raises inside legacy code, that is a pre-existing bug. Note it in `tests/legacy/KNOWN_ISSUES.md` with the traceback. Drop the case from the baseline, but keep it in the task list for the matching bug-fix commit.

- [ ] **Step 3: Generate the baseline.** Run `python -m tests.legacy.capture`. Expected: one `.npz` per case in `tests/baseline/`.

- [ ] **Step 4: Commit**

  ```bash
  git add tests
  git commit -m "Record baseline results of the legacy code"
  ```

---

### Task 3: Move the computation into `pypalc/core` behind a pipeline

**Files:**
- Create: `pypalc/core/{venue,geometry,palc,sound_field,target_slope,pipeline}.py`, `pypalc/gui/__init__.py`, `pypalc/gui/ranges.py`, `tests/test_pipeline_regression.py`, `tests/test_core_has_no_bokeh.py`, `tests/test_pipeline.py`
- Delete: `fun/PALC_classes.py`, `fun/PALC_functions.py`, `fun/calcPALC.py`, `fun/checkPAL.py`, `fun/sfp_functions.py`, `fun/PALC_opt.py`, `fun/gui_venue.py`. Their contents move to core.
- Modify:
  - `main.py`: imports switch to `pypalc.core.*`, keeping the legacy class and function names for now. Add `sys.path.insert(0, abs_path)`.
  - `fun/gui_*.py`: update imports.
  - `tests/legacy/capture.py`: keep importing the *old* code from git, since the legacy files are deleted. In Step 1, copy the current `fun/` into `tests/legacy/fun/` and point `capture.py` there. That keeps it runnable for bug-fix commits until Task 9.

**Interfaces:**
- Consumes: `Setup` and friends, and the errors, from Task 1. The baseline from Task 2.
- Produces in `pipeline.py`. Class names in these annotations are the final names. Until Task 8 they refer to the legacy classes:

  | final name | legacy class |
  |---|---|
  | `AudienceLines` | `PAL` |
  | `PalcVenue` | `PALC_compatible_PAL` |
  | `ArrayEvaluation` | `Technical_meas_results` |
  | `TargetSlopeState` | `Opt_weight` |
  | `ArrayGeometry` | `Array_opt` |
  | `RayPlotData` | `PALC_plotting` |

  The pipeline functions and dataclasses below are new code and use their final names from the start.

  ```python
  @dataclass
  class VenueResult:
      drawn: AudienceLines          # legacy PAL: the lines as drawn, discretized
      palc_venue: PalcVenue         # legacy PALC_compatible_PAL
      non_audience: AudienceLines   # suggested gap lines (legacy suggested_nal)
      sound_field_only: AudienceLines  # legacy suggested_sfp

  @dataclass
  class ArrayResult:                # one per array (optimized, reference)
      tilt_angles_deg: np.ndarray
      inter_cabinet_angles_deg: np.ndarray
      evaluation: ArrayEvaluation   # legacy Technical_meas_results
      spl_over_distance: TargetSlopeState   # legacy SPLoverX object
      patches_x: list; patches_y: list      # cabinet outlines for plotting

  @dataclass
  class RunResult:
      setup: Setup
      venue: VenueResult
      optimized: ArrayResult
      reference: ArrayResult
      geometry: ArrayGeometry       # legacy Opt_arr (rays, seg_pos, num_iter)
      ray_plot: RayPlotData         # legacy PALC_plots
      target: Optional[TargetSlopeState]   # final Opt_w for target-slope runs
      weighting_weights: np.ndarray

  def build_venue(venue: VenueSettings, array: ArraySettings) -> VenueResult        # raises VenueError
  def check_line(lines: AudienceLines, line: AudienceLine) -> None                  # raises VenueError(reason) on draw-condition failure or >2000 points
  def fixed_angle_limits(venue: VenueResult, setup: Setup) -> Tuple[float, float]   # (max_deg, min_deg) as in use_fixed_angle
  def initial_target_slope(setup: Setup, venue: VenueResult, hinges: int) -> TargetSlopeSettings  # GUI defaults after choosing Target Slope
  def run(setup: Setup, venue: Optional[VenueResult] = None) -> RunResult           # raises ConvergenceError / CalculationError
  def result_arrays(result: RunResult) -> Dict[str, np.ndarray]                    # same key contract as run_legacy
  ```

  `run` builds fresh legacy config objects from `setup` (`PalcConfig`, `SoundFieldConfig`) every call. It never mutates `setup`. When `fixed_top_angle_deg` is set, it clamps it into `fixed_angle_limits`, rounded to 2 decimals. For target slope:
  1. It runs once without weighting to get the distance grid, as the GUI does.
  2. It snaps the target points and reference distance to that grid, using `update_draw`'s nearest-point logic.
  3. It runs `optimize` with `max_loops=200`.

- [ ] **Step 1: Copy the legacy code.** Copy `fun/` to `tests/legacy/fun/` and repoint `capture.py`. Run `pytest`. Expected: PASS.
- [ ] **Step 2: Write the failing tests.**
  - `test_pipeline_regression.py::test_matches_baseline[case]`: `result_arrays(run(load_setup(case)))` equals the `.npz`. Use `np.testing.assert_allclose(rtol=1e-9, atol=1e-9)` for every key present in the baseline.
  - `test_core_has_no_bokeh`: walk `pypalc/core/*.py` with `ast` and assert no `import bokeh` / `from bokeh`.
  - `test_pipeline.py`:
    - `test_fixed_angle_is_clamped`: `fixed_top_angle_deg=90` → the result's top cabinet angle equals `round(max_deg, 2)` from `fixed_angle_limits`.
    - `test_check_line_rejects_negative_slope`: raises `VenueError`.
    - `test_build_venue_without_lines_raises`: raises `VenueError`.
    - `test_run_does_not_mutate_setup`.
- [ ] **Step 3: Run** `pytest -v`. Expected: the new tests FAIL with import errors.
- [ ] **Step 4: Move the modules into core, per the file map.**
  - Keep the legacy function and class names. Renaming is Task 8.
  - Use relative package imports instead of `sys.path` hacks.
  - Replace `from PALC_opt import *` (a self-import in `PALC_opt.py`) with nothing.
  - `Plot_ranges` (it uses `Range1d`) moves to `pypalc/gui/ranges.py`. `venue_slice2CDS` and `get_p_axis_rng` from `gui_helper.py` move there too.
  - `read_dir_data`: split into `parse_measured_directivity(csv_text: str, sound_field_config)` in `sound_field.py`, and a GUI wrapper that base64-decodes.
  - Replace `print` with `logging.getLogger(__name__)`.
  - Replace bare `except:` in core with specific exceptions, or remove them.
- [ ] **Step 5: Implement `pipeline.py`.** Extract the bodies of `check_PAL`, `save_line`'s checks, `use_fixed_angle`, `start_calc`, `chg_weight_links` and `update_draw` into the functions above. The no-convergence condition (`num_iter == 500` or empty tilt angles) raises `ConvergenceError`.
- [ ] **Step 6: Point `main.py` at the moved code.** It now imports from `pypalc.core` and `pypalc.gui.ranges`, with no other behavior change. Start the app with `cd .. && bokeh serve pyPALC --port 5007` (background). Check `curl -s localhost:5007/pyPALC | grep -c bokeh`; expected ≥ 1. Then stop the server.
- [ ] **Step 7: Run** `pytest -v`. Expected: all PASS.

  A regression mismatch is a refactoring error: fix the refactor. The exception is when the mismatch comes from removing a GUI-state dependency, for example hard margin mutating `PALC_config.x_H`, which the GUI resets afterwards. In that case:
  - Decide whether it is a real bug.
  - If it is, follow the bug-fix protocol in a separate commit: fix the legacy copy in `tests/legacy/fun` the same way, re-capture, and commit.
- [ ] **Step 8: Commit**

  ```bash
  git add -A pypalc fun main.py tests
  git commit -m "Move computation into pypalc.core behind a shared pipeline"
  ```

---

### Task 4: Headless runner and CSV export

**Files:**
- Create: `pypalc/__main__.py`, `pypalc/cli.py`, `pypalc/core/export.py`, `tests/test_cli.py`
- Modify: `README.md` (section "Run without the GUI")

**Interfaces:**
- Consumes: `load_setup`, `run`, `result_arrays`, the errors.
- Produces:
  - `export.write_results(result: RunResult, out_dir: Path) -> List[Path]`
  - `cli.main(argv: Optional[List[str]] = None) -> int`
  - CSV files and headers. These are the *new* descriptive headers. Task 8 makes the GUI downloads use the same ones.

    | file | headers |
    |---|---|
    | `tilt_angles.csv` | `cabinet,tilt_angle_deg,inter_cabinet_angle_deg` |
    | `reference_tilt_angles.csv` | `cabinet,tilt_angle_deg` |
    | `spl_over_distance.csv` | `distance_m,spl_db,reference_spl_db,x_m,y_m` |
    | `target_slope.csv` (target slope only) | `distance_m,level_db,x_m,y_m`, plus one row per point; reference point in `target_reference.csv`: `distance_m,level_db,x_m,y_m` |
    | `homogeneity.csv` | `frequency_hz,homogeneity_db,homogeneity_low_db,homogeneity_high_db,reference_homogeneity_db` |
    | `sound_field.csv` | `x_m,y_m,` then one column per frequency named `spl_db_<f>hz`, with f rounded to an integer |
    | `reference_sound_field.csv` | same as `sound_field.csv` |

  - Exit codes: `0` ok, `2` invalid setup (message: `setup error at <key>: <msg>`), `3` venue error, `4` no convergence / calculation error. Messages go to stderr.

- [ ] **Step 1: Write the failing tests** in `tests/test_cli.py`:
  - `test_cli_writes_csvs(tmp_path)`: `main(["run", "tests/cases/simple.yaml", "--out", str(tmp_path)])` returns 0. The expected files exist with exactly the headers above. The `tilt_angle_deg` column equals the baseline `tilt_angles_deg`.
  - `test_cli_invalid_setup_exit_code`: returns 2, and stderr contains `array.cabinet_count`.
  - `test_cli_resolves_relative_directivity_path(tmp_path, monkeypatch)`: `monkeypatch.chdir(tmp_path)` and run `tests/cases/measured.yaml` by absolute path. Expected: returns 0.
  - `test_cli_target_slope_writes_target_files`.
  - `test_python_m_entry_point`: `subprocess.run([sys.executable, "-m", "pypalc", "run", ...])` returns 0.
- [ ] **Step 2: Run** `pytest tests/test_cli.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement `export.py`, `cli.py` and `__main__.py`.** Use `argparse` with a `run` subcommand, `--out` defaulting to `./results`, and the directory created if missing. Use the `csv` module with float formatting `repr`.
- [ ] **Step 4: Run** `pytest -v`. Expected: PASS.
- [ ] **Step 5: Update the README** with the headless usage and an example setup file. Copy `tests/cases/simple.yaml` to `examples/simple.yaml` and reference it from the README.
- [ ] **Step 6: Commit**

  ```bash
  git add pypalc tests README.md examples
  git commit -m "Add headless runner with CSV export"
  ```

---

### Task 5: Per-session `PalcApp` and GUI split into page modules

**Files:**
- Create: `pypalc/gui/{__init__,app,theme,help_text,page_venue,page_loudspeaker,page_array,page_algorithm,page_results}.py`, `tests/test_gui_sessions.py`, `docs/superpowers/gui-checklist.md`
- Modify: `main.py` (shrinks to the entry point), `pypalc/gui/ranges.py`
- Delete: `fun/` (the rest), `create_layout.py` (moved to `gui/theme.py`)

**Interfaces:**
- Consumes: the pipeline from Task 3 and `pypalc.gui.ranges`.
- Produces:

  ```python
  class PalcApp:
      def __init__(self, doc: Optional[Document] = None): ...   # builds all per-session state, sources, widgets, figures, wires callbacks; no curdoc() calls
      state: SessionState            # drawn lines (AudienceLines), venue (Optional[VenueResult]), last result (Optional[RunResult]), target-slope state, plot ranges
      pages: Dict[int, Page]
      def show(self) -> None         # adds the current page layout to self.doc (replaces steps())
      def run_calculation(self) -> None   # callback body of the start toggle: setup_from_app → pipeline.run → update sources
      def check_venue(self) -> None       # callback body of the check-PAL button
  ```

  - Each `page_*.py` defines a `class <Name>Page` with `__init__(self, app: PalcApp)` (it creates its widgets, figures and sources on `self`, and registers callbacks) and `layout(self) -> LayoutDOM` (it returns the exact rows and columns from today's `steps()` for that page).
  - Shared widgets (header, logos, `gui_step`, `p_equal_axis`, `config_error_info`, venue figure `p`) live on `PalcApp`.
  - `main.py`: `from pypalc.gui.app import PalcApp; PalcApp(curdoc()).show()`, plus the `sys.path` line.
  - `theme.write_theme(app_dir: Path) -> None` keeps today's output of `create_layout.py` byte-identical.

- [ ] **Step 1: Write `docs/superpowers/gui-checklist.md`.** Make it one checkbox per behavior from the spec's anti-goals, plus these:
  - page switching
  - equal-axis radio
  - the directivity plot updates on directivity change
  - reference array widgets show and hide per type
  - Target Slope ⇄ Hard Margin mutual exclusion
  - ν slider with +/− and step size
  - fixed angle enable after the venue check, and its Max/Min text
  - running indicator on the start toggle
  - every download button
  - beam plot checkboxes and frequency select
  - SFP venue tap highlighting responses
  - legend visibility

  Walk the checklist on the current app first and note anything already broken (that is pre-existing) at the top.
- [ ] **Step 2: Write the failing tests** in `tests/test_gui_sessions.py`:
  - `test_sessions_do_not_share_state`: build `a = PalcApp(Document())` and `b = PalcApp(Document())`. Add a line through `a`'s page-1 widgets and save callback. Assert that `b.state.drawn` has no lines, and that `a.pages[1].lines_source is not b.pages[1].lines_source`.
  - `test_no_module_level_bokeh_models`: import every `pypalc.gui.*` module and assert no module attribute is an instance of `bokeh.model.Model`.
  - `test_run_calculation_matches_pipeline`:
    1. Drive app `a` through the `simple` case via its widgets and callbacks.
    2. Call `a.check_venue()`, then `a.run_calculation()`.
    3. Compare the results table source columns with the baseline `tilt_angles_deg` and `inter_cabinet_angles_deg`.
- [ ] **Step 3: Run** `pytest tests/test_gui_sessions.py -v`. Expected: FAIL.
- [ ] **Step 4: Implement the split.** Move each callback into the page that owns its widgets. GUI callbacks call the pipeline and copy results into sources.
  - **Errors:** catch `PalcError` at the callback boundary and show today's message text in the same widget. Log the traceback with `logger.exception`. Do not use bare `except:`.
  - **Venue checks:** `save_line` and `create_prev_line` use `pipeline.check_line`, and the `VenueError` message goes into `pal_error_info`.
  - **Target Slope and fixed angle:** selecting Target Slope calls `pipeline.initial_target_slope` and fills the target sources. The fixed-angle path uses `pipeline.fixed_angle_limits`.
- [ ] **Step 5: Run** `pytest -v`. Expected: PASS.
- [ ] **Step 6: Manual check.** Run `cd .. && bokeh serve --show pyPALC`, walk the whole checklist, then open a second browser tab and confirm it starts clean. Record the outcome in the checklist file.
- [ ] **Step 7: Commit**

  ```bash
  git add -A
  git commit -m "Split GUI into per-session app and page modules"
  ```

---

### Task 6: Pull the GUI's setup into a `Setup`

**Files:**
- Create: `pypalc/gui/setup_io.py`, `tests/test_setup_io.py`

**Interfaces:**
- Consumes: `PalcApp` and `Setup`.
- Produces: `setup_from_app(app: PalcApp) -> Setup`. It reads the session state and widgets, and puts the uploaded directivity CSV text into `MeasuredDirectivity(csv=...)`. `PalcApp.run_calculation` now uses it, so the GUI and the CLI share one input path.

  For this, `PalcApp.state` stores the raw uploaded CSV text (`state.measured_directivity_csv: Optional[str]`), decoded once at upload.

- [ ] **Step 1: Write the failing tests.**
  - `test_setup_from_app_defaults`: a fresh app gives `Setup()`.
  - `test_setup_from_app_case[case]`: for each non-measured baseline case, drive the widgets to that case, call `setup_from_app`, and compare with `load_setup(case)`. Target slope cases drag points by writing to the target source, as `PointDrawTool` does.
  - `test_setup_from_app_measured`: after an upload of `tests/cases/measured.csv` (base64 into the FileInput value), `setup.loudspeaker.measured_directivity.csv` equals the file text.
- [ ] **Step 2: Run** the tests. Expected: FAIL.
- [ ] **Step 3: Implement** `setup_from_app` and switch `run_calculation` to `pipeline.run(setup_from_app(self), venue=self.state.venue)`.
- [ ] **Step 4: Run** `pytest -v`. Expected: PASS.
- [ ] **Step 5: Commit**

  ```bash
  git commit -am "Build the GUI's run from a Setup"
  ```

---

### Task 7: Save and load setup buttons

**Files:**
- Create: `ext/download_text.js`, `tests/test_apply_setup.py`
- Modify: `pypalc/gui/setup_io.py`, `pypalc/gui/page_algorithm.py`, `pypalc/gui/help_text.py`

**Interfaces:**
- Consumes: `setup_from_app`, `dump_setup`, `load_setup_text`, and the pipeline.
- Produces:
  - `apply_setup(app: PalcApp, setup: Setup) -> None`. It replaces all session state, sets every widget without letting widget callbacks re-trigger calculations, rebuilds the drawn lines, then runs `check_venue()` and `run_calculation()`, and switches to page 5. Callbacks are suppressed with a `state.applying_setup` flag that the callbacks check.
  - Page 4 col1, under the "Download SPL" button:
    - `Button(label="Save Setup", button_type="success", width=255)`: Python writes `dump_setup(setup_from_app(app))` into `ColumnDataSource(data={"text": [...]})`. A `js_on_change('data')` callback runs `download_text.js`, which downloads `pypalc_setup.yaml`.
    - `FileInput(accept=".yaml,.yml", width=255)` titled by a `Div` "Load Setup". On change, Python base64-decodes the value, calls `load_setup_text(text)` (no `base_dir`), then `apply_setup`.
  - Errors: a `SetupError` shows `Error: setup file invalid at <key>: <msg>` in `config_error_info`, and nothing changes. That includes a `path` directivity, whose message says to save the setup from the GUI or use the headless runner.

- [ ] **Step 1: Write the failing tests** in `tests/test_apply_setup.py`:
  - `test_apply_then_extract_round_trip[case]`: for each case, `setup_from_app(app)` after `apply_setup(app, load_setup_text(text_of_case_with_csv_embedded))` equals the loaded setup.
  - `test_apply_setup_results_equal_headless[case]`: after `apply_setup`, `result_arrays(app.state.result)` equals `result_arrays(run(setup))`.
  - `test_apply_setup_rejects_path_directivity`: the error text names `loudspeaker.measured_directivity.path`, and `app.state` is unchanged (compare `setup_from_app` before and after).
  - `test_save_writes_yaml_to_download_source`: clicking save puts YAML into the source, and `load_setup_text` of that text equals `setup_from_app(app)`.
  - `test_target_slope_restored`: after applying the `target_slope` case, the target source x and SPL match the setup points, and the hinge select equals the hinge count.
- [ ] **Step 2: Run** the tests. Expected: FAIL.
- [ ] **Step 3: Implement.** Add one line to page 4 of the help text describing save and load.
- [ ] **Step 4: Run** `pytest -v`. Expected: PASS.
- [ ] **Step 5: Manual check.** In the browser: set up the `target_slope` case by hand, save, reload the page, load the file. Expected: you land on the results with the same tilt angles. Load an invalid file and check the error text. Add both checks to the checklist.
- [ ] **Step 6: Commit**

  ```bash
  git add -A
  git commit -m "Add GUI save and load of setup files"
  ```

---

### Task 8: Descriptive names throughout

**Files:**
- Modify: all of `pypalc/**`, `ext/download.js` (no change to the mechanism), `README.md`, `pypalc/core/__init__.py` (the name table), and the tests that reference legacy names.

**Interfaces:**
- Produces the final names. Apply this table everywhere: attributes, parameters, locals, functions, classes, source column keys, and widget variable names. Names not in the table follow the same style.

  | legacy | new | paper symbol |
  |---|---|---|
  | `PAL` | `AudienceLines` | |
  | `PALC_compatible_PAL` | `PalcVenue` | |
  | `PALC_configuration` | `PalcConfig` | |
  | `SFP_configuration` | `SoundFieldConfig` | |
  | `Array_opt` | `ArrayGeometry` | |
  | `Technical_meas_results` | `ArrayEvaluation` | |
  | `Opt_weight` | `TargetSlopeState` | |
  | `PALC_plotting` | `RayPlotData` | |
  | `Plot_ranges` | `PlotRanges` | |
  | `N` | `cabinet_count` | N |
  | `Lambda_y` | `cabinet_height` | Λ_y |
  | `Lambda_gap` | `cabinet_gap` | Λ_gap |
  | `x_H`, `y_H` | `top_x`, `top_y` | x_H, y_H |
  | `psi_n` | `aperture_angles` (rad) | ψ_n |
  | `gamma_n` | `tilt_angles` (rad) | γ_n |
  | `gamma_tilt_deg` | `tilt_angles_deg` | |
  | `gamma_tilt_deg_diff` | `inter_cabinet_angles_deg` | |
  | `gamma_LSA` | `discrete_tilt_angles` (rad) | |
  | `use_gamma_LSA` | `use_discrete_angles` | |
  | `constraint` | `palc_variant` | |
  | `pal` / `pal_no_nal` | `segments` / `audience_segments` | PAL |
  | `nal` | `non_audience_lines` | |
  | `xline`, `yline` | `points_x`, `points_y` | |
  | `pal_index` | `line_end_indices` | |
  | `x_c_n`, `y_c_n` | `cabinet_centers_x`, `cabinet_centers_y` | |
  | `seg_pos*` | `hit_points*` | |
  | `p_SPL` | `spl_db` | |
  | `f` (frequency arrays) | `frequencies` | |
  | `H` | `homogeneity_db` | H(f) |
  | `H_dist` | `homogeneity_low_db` / `homogeneity_high_db` | |
  | `SPLoverX` | `spl_over_distance` | |
  | `Tech_res(_ref)` | `evaluation(_reference)` | |
  | `Opt_arr` / `Ref_arr` | `optimized_array` / `reference_array` | |
  | `created_pal` / `PALC_pal` | `drawn_lines` / `palc_venue` | |
  | `weighting_nu` | `weighting_nu` | ν (kept: the GUI label says ν) |
  | `gui_*` widgets | descriptive without prefix, e.g. `cabinet_height_input`, `palc_variant_select`, `start_toggle` | |
  | sources `*_dict` | `*_source`, e.g. `results_source`, `spl_over_distance_source` | |
  | figures `p`, `pSFP`, `pHom`, … | `venue_figure`, `sound_field_figure`, `homogeneity_figure`, … | |

  - GUI download CSV headers equal the export headers from Task 4. The results table and download use `cabinet,tilt_angle_deg,inter_cabinet_angle_deg`.
  - The "Download SPL" button currently fires 5 downloads that all have the same file name `PALC_result.csv`. Change `download.js` to take a `filename` argument and use `tilt_angles.csv`, `spl_over_distance.csv`, `target_slope.csv`, `target_reference.csv`, `reference_spl_over_distance.csv` and `homogeneity.csv`. This is a usability bug, so it gets its own commit.
  - The `result_arrays` key contract stays as Task 2 defined it, because the baseline files use those keys.

- [ ] **Step 1: Rename module by module** (core first, then gui) and run `pytest -v` after each module. Expected: PASS. Commit after each module: `git commit -am "Rename <module> to descriptive names"`.
- [ ] **Step 2: Write `test_download_headers`.** It asserts the GUI results and SPL source column names equal the export headers, in the same order. Run it. Expected: PASS.
- [ ] **Step 3: Write the name table** with paper symbols into the `pypalc/core/__init__.py` docstring. Add a "CSV header changes" old → new section to the README.
- [ ] **Step 4: Check for leftover legacy names.** Run `grep -rnE "Lambda_y|gamma_tilt_deg|PALC_config|SPLoverX|Tech_res|created_pal" pypalc main.py`. Expected: no output.
- [ ] **Step 5: Manual check.** Walk the GUI checklist again.
- [ ] **Step 6: Commit** the remaining changes: `git commit -am "Document names and CSV header changes"`.

---

### Task 9: Final cleanup and verification

**Files:**
- Delete: `tests/legacy/` (the baseline `.npz` files stay), `create_layout.py` if it's still there, and `ipython.html` only if it is unreferenced. Check first, and ask the user if unsure.
- Modify: `README.md` (structure overview, running tests), `requirements*.txt`

- [ ] **Step 1:** Remove `tests/legacy` and `test_baseline_capture.py`. The regression test keeps the `.npz` files. Run `pytest -v`. Expected: PASS.
- [ ] **Step 2:** Check for bare excepts. Run `grep -rn "except:" pypalc main.py`. Expected: no output.
- [ ] **Step 3:** Check for leftover module state. Run `grep -rn "^[a-z_]* *= *ColumnDataSource\|^[a-z_]* *= *figure(" pypalc main.py`. Expected: no output.
- [ ] **Step 4:** Do a fresh-env check. Run `conda create -n pyPALC_cleanup_check --file requirements.txt -y && conda run -n pyPALC_cleanup_check pytest`. Expected: PASS. Then remove the env.
- [ ] **Step 5:** Walk the full GUI checklist one last time, including the two-tab check.
- [ ] **Step 6:** Commit with `git commit -am "Remove legacy capture code and finish cleanup"`.

---

## Bug fixes found along the way

Known candidates to confirm while refactoring. Each is fixed only if confirmed, and follows the bug-fix protocol in Global Constraints:
- **Mutable default arguments** in `PAL`, `PALC_compatible_PAL` and `PALC_configuration`. This is real shared state across sessions.
- **Hard margin mutates `PALC_config`** (`x_H`, `y_H`, `gamma_n`, `psi_n`). `start_calc` restores only some of these.
- **Hard margin starts with a fixed `psi_n = 0.00524` rad (0.3°)** instead of the configured aperture, 10× smaller than the default. Ask the user before changing it; this may be intended.
- **`Opt_weight.calc_init`** drops the `'same'` key from `diffps`.
- **"Download SPL" writes five files** that all have the same name (Task 8).
- **`update_draw`'s `is2update < 4` re-entry guard**, which depends on how often Bokeh fires callbacks.
