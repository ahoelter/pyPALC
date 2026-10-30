# pyPALC code cleanup: design

Date: 2026-10-30
Status: draft, awaiting approval

This is the first of four projects, in this order:

1. **Code cleanup** (this document)
2. Migration to the current stable Bokeh 3
3. Sphinx documentation published on Read the Docs
4. Better approaches for target slope optimization

Each of the other three gets its own design later.

## Intent

pyPALC began as a student project. It is a Bokeh server app that optimizes the curving (tilt angles) of line source arrays with the PALC algorithm, and it evaluates the result with a sound field prediction (CDPS model). The app works, but the code is hard to read:

- `main.py` is about 2,000 lines. It mixes the definition of state, widgets, figures and callbacks with the calculation sequence.
- Names follow the paper notation (`psi_n`, `Lambda_y`, `gamma_tilt_deg`, `x_H`).
- Errors are swallowed by bare `except:` blocks.
- There are no tests.

The goal is code that is easy to read and extend, so the following projects build on solid ground. That means the Bokeh 3 migration, the documentation and especially new target slope methods. A second goal is **reproducible runs without the GUI**, driven by a setup file that the GUI can also save and load.

## Goals

- Readable code, with descriptive names in place of paper notation.
- Split `main.py` into a computation core with no Bokeh dependency and a GUI with one module per page.
- One shared calculation pipeline that both the GUI and a headless runner call.
- A setup file (YAML) that describes a complete run. The headless runner reads it, and the GUI gets buttons to **save** the current setup and **load** one.
- A headless runner that reads a setup file and writes the results as CSV.
- Errors that keep their real cause. The GUI still shows its short user-facing messages.
- Automated tests, based on recorded results from the current code.
- Fix real bugs found along the way (see "Bug fixes").

## Non-goals

- **No layout changes.** The pages, widgets, figures and their arrangement stay as they are. The only addition is the save/load buttons.
- **No Bokeh upgrade.** The cleanup stays on the current environment (conda env `pyPALC`: Python 3.8, Bokeh 2.2.x, numpy 1.20, scipy 1.6). Bokeh 3, and the newer Python it needs, is project 2.
- **No documentation rework.** The cleaned code keeps numpy-style docstrings that work with the existing Sphinx/napoleon setup. The `.rst` files in `docs/source` and the committed `docs/build` stay untouched even when they go stale. Project 3 redoes them.
- **No new or changed optimization methods.** Target slope stays algorithmically the same, apart from bug fixes. Project 4 covers new methods.

## Anti-goals: these count as failure even if the code runs

- The GUI quietly loses behavior. This includes:
  - tap a line to insert a line in front of it, or to delete it
  - widgets that enable, disable, show or hide depending on other choices (weighting vs. gap handling, fixed first angle, discrete angles, reference array type, hinge count)
  - the running indicator on the start toggle
  - dragging the target slope points
  - the CSV download buttons
- The GUI and the headless runner give different numbers for the same setup.
- Two browser sessions on the same server share or overwrite each other's state.
- Results change without an explanation in the commit history.

## Constraints

- **Bokeh session model.** `bokeh serve` runs `main.py` once for every browser session, but it imports other modules only once per server process. So no per-session state (configuration objects, `ColumnDataSource`s, widgets, figures) may sit at the top level of an imported module. The GUI is built by a function or class that `main.py` calls once per session.
- **How the app is started.** `bokeh serve --show pyPALC`, run from the parent directory, has to keep working. The repository root is the app directory.
- **Compatibility.** Code must run on the current env (Python 3.8). Code that is reasonable to keep under Bokeh 3 is preferred, for example avoiding new uses of APIs that Bokeh 3 removed. The migration itself is still project 2.
- **Measured directivity.** The CSV format that the upload accepts stays the same. The setup file points to such a CSV by path.

## Approaches considered

- **A. Incremental refactor that keeps the app working (chosen).**
  1. Record the current results as tests.
  2. Extract the venue check and the calculation into plain functions.
  3. Add settings classes and the setup file.
  4. Add the headless runner.
  5. Split the GUI into page modules.
  6. Rename.

  The app runs and the tests pass after every step, so a regression points to the step that caused it. Trade-off: slower, and the code is half old and half new for a while.
- **B. Rewrite from scratch.** This gives the cleanest result, but the GUI has a lot of small behavior that no test covers, and a rewrite would likely lose some of it. Rejected.
- **C. Clean the computation now and the GUI together with the Bokeh 3 migration.** The GUI would be touched only once. But when the GUI breaks, you couldn't tell whether the restructuring or the upgrade caused it, with no GUI tests to help. Rejected.

**Why A:** once the GUI is split into small page modules, the Bokeh 3 migration becomes mostly mechanical changes inside each module.

## Decisions made with the user

- **Restructuring is allowed.** `main.py` may be split, and Bokeh callbacks do not have to live in `main.py`.
- **Descriptive names** replace paper notation everywhere in code. A mapping table from new names to paper symbols goes into the docs. A short version goes in a module docstring now, and project 3 can expand it.
- **Headless use is for reproducible runs from a setup file**, not primarily a Python API for scripting. (A clean Python function will exist anyway, as a side effect of the shared pipeline.)
- **GUI gets both buttons:** save the current setup to a file, and load a setup file.
- **GUI load restores everything**, including running the venue check and the calculation, so the user lands on the results.
- **CSV headers are renamed** to descriptive names, and the README documents the change.
- **Bugs get fixed during the cleanup, not just listed.** The old behavior stays in the git history.
- **Approach A.**

## Design

### Package structure

The structure below is a proposal. Exact module boundaries are left to the builder, as long as the core/GUI split and the session rule hold.

```
pyPALC/                     (repo root = Bokeh app directory)
  main.py                   short Bokeh entry point: build the app for this session, add to curdoc
  pypalc/
    __init__.py
    __main__.py             enables `python -m pypalc run ...`
    cli.py                  headless runner
    core/                   no Bokeh imports anywhere in here
      settings.py           settings dataclasses + YAML load/save + validation
      venue.py              audience lines, draw conditions, PALC-compatible venue (was checkPAL, gui_venue logic, PAL classes)
      palc.py               PALC algorithm (was calcPALC, PALC_functions)
      sound_field.py        CDPS sound field prediction, directivity (was sfp_functions)
      target_slope.py       target slope optimization loop and helpers (was PALC_opt, optimize_PALC, Opt_weight)
      results.py            homogeneity, histogram, SPL over distance; result containers
      pipeline.py           build venue → run PALC (or target slope loop) → SFP → measures, for optimized and reference array
      export.py             write results to CSV
    gui/
      app.py                builds one session's state, widgets, figures; wires callbacks
      theme.py              (was create_layout.py; still writes theme.yaml and templates/index.html)
      help_text.py
      page_venue.py
      page_loudspeaker.py
      page_array.py
      page_algorithm.py     includes save/load setup buttons
      page_results.py
  tests/
    fixtures/               setup files + recorded results
    ...
```

### Shared pipeline

The calculation steps now in `check_PAL` and `start_calc` become plain functions in `core/pipeline.py`. They take settings objects and return result objects. They never touch widgets or `ColumnDataSource`s. The GUI callbacks:

1. build the settings from the widgets,
2. call the pipeline,
3. copy the results into the `ColumnDataSource`s and set plot ranges and widget states.

The headless runner:

1. builds the settings from the YAML file,
2. calls the same pipeline,
3. writes the results.

The reference array (straight, progressive, arc, user-defined) is part of the pipeline output, as today.

### Settings and the setup file

Settings are a set of small dataclasses, grouped like the GUI pages: venue, loudspeaker, array, reference array, algorithm and target slope. They replace configuration that is currently spread over `PALC_configuration`, `SFP_configuration` and widget callbacks.

The YAML setup file serializes exactly these dataclasses. It contains:
- **Venue:** the audience lines as the user drew them (start and stop x and y in m), plus the discretization. The PALC-compatible venue is *derived* by the pipeline, not stored.
- **Loudspeaker:** PALC aperture start angle, cabinet height, and directivity. The directivity is one of the modeled options or a path to a measured-directivity CSV. A relative path is resolved relative to the setup file.
- **Array:** position, number of cabinets, gap, and discrete tilt angles, with on/off and the list.
- **Reference array:** type and its parameters.
- **Algorithm:** PALC variant, gap handling, soft margin strength, weighting type, weighting parameter, tolerance, fixed first angle, frequency range.
- **Target slope** (only when weighting is Target Slope): number of hinges, the target points (distance in m, level in dB, in the same normalization the SPL-over-distance plot uses) and the reference point.

Plus a `version` field so the format can evolve.

Loading validates the file. Unknown keys, missing required keys, wrong types and out-of-range values give an error that names the key. The GUI shows that error in its existing error text field and keeps its current state.

**How the GUI buttons reach the file (builder's call, recorded here).**
- **Save** is a browser download, the same mechanism as the existing CSV downloads in `ext/download.js`.
- **Load** is a browser upload through a `FileInput`, so the server receives the file contents but no path.

Because of that, a measured directivity uploaded in the GUI has no path to save. The setup file therefore supports two forms of measured directivity:
- a `path` to a CSV, resolved relative to the setup file when run headless
- the directivity data embedded in the YAML, which is what GUI save writes after an upload

The headless runner accepts both forms. A GUI load of a file that uses `path` reports a clear error, because the GUI cannot see the server's file system relative to an uploaded file. The embedded form must be loadable both headless and in the GUI.

**What GUI load restores (decided with the user): everything, in one go.** Load does four things:
1. Validates the file.
2. Sets all widgets.
3. Rebuilds the drawn audience lines and runs the venue check.
4. Runs the calculation, so the user lands on the results as if they had just clicked "Run".

For target slope setups, load also restores the hinge count, the target points and the reference point. A plain replay of widget callbacks does not achieve this: the venue check resets Target Slope weighting to "Without", and the target point store only exists after a first calculation. So load drives the session state directly, in the right order, and then updates the widgets to match.

The acceptance check for load: load a saved setup, and the results must equal the headless runner's results for the same file.

### Headless runner

```
python -m pypalc run setup.yaml --out results/
```

It writes CSVs for:
- tilt angles (absolute and between neighboring cabinets)
- SPL over distance, for the optimized and the reference array
- homogeneity, for the optimized and the reference array
- the sound field prediction (SPL per receiver position and frequency)

Column layout (which columns, in which order) should match the GUI's CSV downloads wherever there is one. GUI downloads and headless output use the same headers.

**CSV headers get descriptive names (decided with the user).** The GUI downloads currently use the `ColumnDataSource` keys as headers, for example `a_num_LS`, `b_gamma_tilt_deg`, `c_gamma_tilt_deg_diff`, `H_dist_high`, `x_v`, `y_v`. These get renamed along with the code. The README notes the change and lists old → new headers, so existing scripts that read old downloads can be updated. It exits non-zero with a clear message on invalid setup or failed calculation. Beam plots and figures are not part of the headless output.

### Errors

- Remove the bare `except:` blocks.
- Core functions raise specific exceptions: invalid venue line, no convergence, invalid setup, and so on.
- The GUI catches them at the callback boundary, shows the same short message as today in the error text field, and prints the traceback to the server log.
- Draw-condition checks (`gui_venue.py`) raise or return a reason instead of writing into a widget.

### Naming

Descriptive names throughout core and GUI: modules, functions, classes, variables and widget variables, for example `cabinet_height` (was `Lambda_y`) and `tilt_angles_deg` (was `gamma_tilt_deg`). Classes get PEP 8 names. User-facing labels in the GUI stay as they are, since the layout and text don't change. The mapping table from new names to paper symbols lives in the `core` package docstring.

### Testing

- **Before any refactor**, record results from the current code for a small set of setups. Store them as fixtures, as setup files plus result arrays. The setups cover:
  - a simple two-line venue
  - a venue with a gap, under each gap-handling mode
  - each PALC variant
  - discrete tilt angles
  - fixed first angle
  - measured directivity
  - linear and logarithmic weighting
  - one target slope case with hinges
- pytest tests compare the pipeline output against these fixtures, within a small floating-point tolerance.
- Further tests cover:
  - the setup file round trip (save → load gives identical settings)
  - validation errors
  - a headless run end to end
  - a check that two app instances built in one process don't share state
- GUI behavior has no automated tests. After the GUI split and the rename steps, the builder runs the app and walks through every behavior listed under the anti-goals.

### Bug fixes

When a real bug turns up, fix it in its own commit. The commit message explains the bug and its effect, and the same commit updates the affected recorded results. Nothing else goes into that commit. A bug-fix commit is the only kind allowed to change recorded results.

### Dependencies

- Add `pyyaml` for the setup file.
- Add `pytest` for tests.
- Update `requirements.txt` and `requirements_pip.txt`.
- Update the README with how to run the tests and the headless runner.

## Left to the builder

- Exact module boundaries and names inside `core/` and `gui/`, as long as the constraints hold.
- The exact YAML schema and key names.
- Exception class design.
- How the GUI app object holds its session state: a class with attributes, or a small state container.
- The order of small steps inside approach A, and how many commits.
- Floating-point tolerances in the tests.
- Whether the old `fun/` modules are removed at the end or moved piece by piece. They must be gone at the end of the project, though.
