# pyPALC Bokeh 3 Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** pyPALC runs on Bokeh 3.10.x and Python 3.12 with current packages, looks and works as it does today, and its results stay the same.

**Architecture:** Four steps on branch `bokeh3`:
1. Build a new environment and settle the numerics with the calculation tests alone.
2. Port the GUI to the Bokeh 3 API.
3. Build a headless-Firefox screenshot tool, then move the widget CSS into Bokeh 3 style sheets until the screenshots match Bokeh 2.
4. Re-record the GUI cascade golden after proving that every difference comes from Bokeh, from rounding drift, or from a listed size change.

**Tech Stack:** Python 3.12, Bokeh 3.10.x, numpy/scipy (current), pyyaml, pytest. The browser tooling uses selenium, Firefox (`/usr/bin/firefox`) and geckodriver (`/snap/bin/geckodriver`).

**Spec:** `docs/superpowers/specs/2026-11-04-bokeh3-upgrade-design.md`

## Global Constraints

- Python 3.12, Bokeh 3.10.x (the newest stable release when the work starts), current numpy, scipy, colorcet, pyyaml and pytest.
- `requirements.txt` (conda format) and `requirements_pip.txt` pin exact versions. The pins that only existed for Bokeh 2.2 (`jinja2`, `markupsafe`, `pillow`) are dropped unless the new stack still needs them.
- New conda env `pyPALC3`; the old env `pyPALC` stays untouched (it is needed for the Bokeh 2 screenshots). Run `conda run -n pyPALC3 ...`, or the env's python, with `PYTHONNOUSERSITE=1`, because the user has packages in `~/.local`.
- Same layout and dark style:
  - dark grey `#323030` background, white text, Helvetica 10pt
  - green links, dark tables
- The colors of buttons and widgets stay as they are, including the "PALC Run" toggle, which turns dark red and reads "Running..." while a calculation runs.
- Results stay the same: the calculation tests compare against `tests/baseline/*.npz` at rtol 1e-9.
- No redesign, no new features, no change to the setup file or CSV formats, no target slope work.
- `docs/source`, `docs/build` and `docs/Makefile` stay untouched.
- `pypalc/core` must not import Bokeh. No per-session state at module level. numpy-style docstrings.
- Bug-fix protocol: one bug per commit, message `fix: <bug> (changes results: <which cases>)`.
- No `Co-Authored-By` or other attribution lines in commits.
- Work on branch `bokeh3`. No push and no merge without the user's word.
- The app is still started with `bokeh serve --show pyPALC` from the repo's parent directory (`/home/arne_hoelter/university/04_pyPALC/git/pyPALC`).
- **Numerics rule:**
  - Baseline failures are root-caused first.
  - Pure floating-point drift (all differences explained, no behavior change) means the baselines are re-recorded in one commit that names the largest difference and its cause.
  - Anything else means: **stop and ask the user**.
- **Golden rule** (`tests/gui_cascade_golden.json`): it is re-recorded only after showing that every difference falls into one of these categories:
  - (a) Bokeh-caused: renamed properties, model counts, new defaults, model type names
  - (b) numeric drift within rtol 1e-9, shown by comparing *full unhashed* states from both environments
  - (c) an intentional size change, each one listed with its reason

  Each category is listed in the test's docstring. The test's own comparison is not loosened.

## Review Focus

1. **Load Setup through the browser's file input.** In Bokeh 3, the FileInput `value`/`filename` events must still reach `load_setup`, and the page must show the loaded values (gap handling, weighting, hinges) without a reload. Pinned in Task 3: the browser check loads a target-slope setup.
2. **Downloads.** The `CustomJS` callbacks (`ext/download.js`, `ext/download_text.js`) must still save files with the right name and the export header in Bokeh 3. The way `args` reach the callback and the BokehJS internals both changed. Pinned in Task 3: the browser check clicks the angle download and the Save Setup download, and checks the files in the profile's download folder.
3. **Two browser tabs on one server.** A line drawn in one tab must not appear in the other. Pinned in Task 6 (browser check).
4. **Ragged arrays and other numpy 2 errors.** Today's suite prints `VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences`; numpy ≥1.24 raises `ValueError` for that instead. Other numpy 2 removals could also turn into silent dtype or promotion changes. Pinned in Task 1: the calculation tests and the 13 baseline cases run in the new env.
5. **Dragging target points.** The PointDrawTool edits must still update the server-side target and trigger the drag callback in Bokeh 3. Pinned in Task 3: the browser check changes the target point source from JS and asserts that the server applied it, read back through the displayed "SPL over Distance" table or source.

---

## File Structure

| File | Change |
|---|---|
| `requirements.txt`, `requirements_pip.txt` | new pins (Task 1) |
| `pypalc/core/*.py` | only numpy-2 fixes if needed (Task 1) |
| `pypalc/gui/*.py` | Bokeh 3 API (Task 2), sizes if needed (Task 4) |
| `pypalc/gui/theme.py` | stays the single source of `theme.yaml` and `templates/index.html`; gains the Bokeh 3 widget style sheets (Task 4) |
| `theme.yaml`, `templates/index.html` | regenerated by `python -m pypalc.gui.theme` (Task 4) |
| `ext/download.js`, `ext/download_text.js` | Bokeh 3 fixes if needed (Task 2/3) |
| `tools/browser_check.py` (new) | headless-Firefox driver: screenshots, load, downloads, drag, two tabs, running indicator (Task 3) |
| `tools/compare_screenshots.py` (new) | writes a side-by-side HTML page of two screenshot folders (Task 3) |
| `tools/cascade_full_states.py` (new) | dumps the unhashed cascade states of a checkout (Task 5) |
| `tools/compare_cascade_states.py` (new) | compares two full dumps, groups the differences (Task 5) |
| `tests/test_gui_cascade.py`, `tests/gui_cascade_golden.json` | re-recorded (Task 5) |
| `README.md`, `docs/superpowers/gui-checklist.md` | env, start command, browser tools (Task 6) |

The `tools/` scripts are developer tools. They are not collected by pytest and not needed to run the app; selenium is documented as an optional dev install.

---

### Task 1: New environment and numerics

**Files:**
- Modify: `requirements.txt`, `requirements_pip.txt`
- Modify, only if needed: `pypalc/core/*.py` (numpy 2 fixes); `tests/baseline/*.npz` (drift re-record, only under the numerics rule)

**Interfaces:**
- Produces: conda env `pyPALC3` (Python 3.12, Bokeh 3.10.x), and requirement files that build it.

- [ ] **Step 1: Create the env from conda-forge with the current releases:**

  ```bash
  conda create -n pyPALC3 -c conda-forge python=3.12 "bokeh=3.10" numpy scipy colorcet pyyaml pytest
  ```

  Record the resolved versions with `conda list -n pyPALC3`.
- [ ] **Step 2: Rewrite the requirement files.**
  - Write `requirements.txt` with exact `name=version` pins for python, bokeh, numpy, scipy, colorcet, pyyaml and pytest, and a first-line comment naming the channel (`conda-forge`).
  - Write `requirements_pip.txt` with the same versions as `==` pins, without python.
  - Drop jinja2, markupsafe and pillow unless step 4 shows they are needed.
- [ ] **Step 3: Run the calculation tests in the new env.**

  ```bash
  PYTHONNOUSERSITE=1 conda run -n pyPALC3 python -m pytest -q -p no:cacheprovider tests/test_settings.py tests/test_pipeline.py tests/test_pipeline_regression.py tests/test_cli.py tests/test_rays.py tests/test_target_slope.py tests/test_core_has_no_bokeh.py
  ```

  Expected: PASS, or failures to analyse in step 4.
- [ ] **Step 4: Root-cause every failure and every new warning category before changing anything.**
  - numpy 2 errors (ragged arrays, removed aliases, scalar conversion) get one `fix:` commit each, with `changes results: none` unless the baselines show otherwise.
  - Baseline mismatches beyond rtol 1e-9:
    - pure floating-point drift: re-record the affected `.npz` in one commit, with message `Re-record baselines for numpy X/scipy Y (max relative difference <value> in <key>, cause: <cause>)`
    - anything else: STOP and report NEEDS_CONTEXT
- [ ] **Step 5: Verify the requirements.** Build a fresh env from the new `requirements.txt` (`conda create -n pyPALC3_check -c conda-forge --file requirements.txt -y`), run the step-3 tests there (expected PASS), then remove the env.
- [ ] **Step 6: Commit** the requirements (`Update to Python 3.12, Bokeh 3.10 and current packages`), separately from any fix or re-record commits.

The GUI tests are expected to fail in `pyPALC3` until Task 2. This is the only task that leaves them failing.

---

### Task 2: Bokeh 3 API

**Files:**
- Modify: `pypalc/gui/app.py`, `page_venue.py`, `page_loudspeaker.py`, `page_array.py`, `page_algorithm.py`, `page_results.py`, `help_text.py`, `ranges.py`, `setup_io.py` as needed; `ext/download.js`, `ext/download_text.js` if needed; `main.py` if needed
- Test: the existing GUI tests

**Interfaces:**
- Consumes: env `pyPALC3` (Task 1).
- Produces: the app starts under Bokeh 3, and all tests except `tests/test_gui_cascade.py` pass.

- [ ] **Step 1: Run the full suite in `pyPALC3`** and record which tests fail and why. Expected: import or attribute errors in the GUI tests.
- [ ] **Step 2: Replace the removed APIs:**

  | Bokeh 2 | Bokeh 3 | Where |
  |---|---|---|
  | `Panel` | `TabPanel` | 14 uses |
  | `FuncTickFormatter` | `CustomJSTickFormatter` | ν slider, `page_algorithm.py` |
  | `plot_width` / `plot_height` | `width` / `height` | `app.py` logo figures |
  | `toolbar_sticky` | removed | 4 figures |
  | `bokeh.models.widgets` | `bokeh.models` | 7 imports |

  Also check every `legend` use: `legend=` on glyphs becomes `legend_label=`. The remaining uses must be checked against the Bokeh 3 migration notes:
  - `gridplot(..., merge_tools=False)`
  - page switching with `doc.clear()` / `add_root`
  - `FileInput`
  - `CustomJS` args
  - `PointDrawTool`
  - `DataTable` / `TableColumn`
  - `ColorBar` / `LinearColorMapper`
  - `RadioButtonGroup`, `Toggle`, `Div`, `Paragraph`
  - `figure.image`
  - the empty `js_on_click` workaround

  No behavior change. Anything Bokeh 3 does differently in a way that changes what the user sees or gets is reported, not silently adapted.
- [ ] **Step 3: Run the suite again.** Everything except `tests/test_gui_cascade.py` must pass, and `tests/test_gui_equivalence.py` stays exact. Run with `PYTHONNOUSERSITE=1 conda run -n pyPALC3 python -m pytest -q -p no:cacheprovider --deselect tests/test_gui_cascade.py`, or use `--ignore` (both work; the cascade is fixed in Task 5). Expected: PASS.
- [ ] **Step 4: Smoke-start the server.**
  - Start it from the parent dir: `PYTHONNOUSERSITE=1 conda run -n pyPALC3 bokeh serve pyPALC --port 5077`.
  - Fetch `http://localhost:5077/pyPALC`. Expected: HTTP 200 and no errors in the server log.
  - Stop the server.
- [ ] **Step 5: Commit** in logical pieces, for example one commit per page module or per API kind.

---

### Task 3: Browser check and screenshot tools

**Files:**
- Create: `tools/browser_check.py`, `tools/compare_screenshots.py`

**Interfaces:**
- Consumes: a running server URL. A Bokeh 2 server is started by the controller or the implementer from a `git worktree` of `master` in env `pyPALC`.
- Produces:
  - `python tools/browser_check.py --url URL --out DIR [--setup FILE]` writes `DIR/<page>_<state>.png` and `DIR/report.json`, with one entry per check: `{"name": str, "ok": bool, "detail": str}`. It exits 1 if any check failed.
  - `python tools/compare_screenshots.py OLD_DIR NEW_DIR OUT.html` writes a page with the image pairs side by side, by file name.

- [ ] **Step 1: Write `tools/browser_check.py` with selenium.**
  - Firefox headless, window 1400×1000, geckodriver at `/snap/bin/geckodriver`.
  - Pages are switched by clicking the `.bk-btn` with the page label.
  - A setup is loaded by setting the FileInput model's `filename`, `mime_type` and `value` (base64) through `Bokeh.documents[0]`. In Bokeh 3, widgets live in shadow DOM, so DOM reads go through the models or `shadowRoot`; the script must work on Bokeh 2.2 and 3.10.
  - It runs these checks:
    1. **screenshots:** every page (Help, 1–5), in state `fresh`
    2. **load:** loading `tests/cases/gap_soft_margin_target_slope.yaml` switches to page 5. Page 4 then shows "Soft Margin", "Target Slope" and hinges "1". Screenshots of every page in state `loaded`.
    3. **running indicator:** a screenshot of page 4 while a run is in progress; the check reads the toggle's label or computed style during the run.
    4. **downloads:** clicking "Download PALC Angles" and "Save Setup" creates the expected files in the Firefox download folder, which is set through profile preferences to a temp dir. The angle CSV header equals `pypalc.core.export.TILT_ANGLES_HEADER`.
    5. **drag:** setting a new target point position in the target source from JS is applied on the server. The check verifies this by reading back the model after a round trip.
- [ ] **Step 2: Write `tools/compare_screenshots.py`.** It produces a plain HTML page: one row per file name, old on the left and new on the right, with captions. No external assets; images are referenced relatively, or embedded as base64.
- [ ] **Step 3: Record the Bokeh 2 reference.**
  - Create a worktree of `master` in the scratch dir.
  - Serve it with env `pyPALC` on port 5078.
  - Run `browser_check.py --out <scratch>/shots_bokeh2`. Expected: report all ok.
  - Remove the worktree afterwards.
- [ ] **Step 4: Run against the `bokeh3` app** (env `pyPALC3`, port 5077, `--out <scratch>/shots_bokeh3`). Failures here are the input for Task 4; functional failures (load, downloads, drag) are fixed in this task, each as a `fix:` commit if it is a bug.
- [ ] **Step 5: Commit the tools** (`Add browser check and screenshot comparison tools`).

---

### Task 4: Style and layout

**Files:**
- Modify: `pypalc/gui/theme.py`, `theme.yaml`, `templates/index.html` (regenerated), `tests/test_theme.py` if the file list changes
- Modify, only if needed: sizes in `pypalc/gui/*.py`; each size change is listed in the commit message with its reason

**Interfaces:**
- Consumes: `tools/browser_check.py` and `tools/compare_screenshots.py` (Task 3), and the Bokeh 2 screenshots.
- Produces: `python -m pypalc.gui.theme` still writes all theme files, and `tests/test_theme.py` checks them byte for byte.

- [ ] **Step 1: Map each CSS rule of today's `templates/index.html`** (generated by `theme.py` `_index_html_lines`) to its Bokeh 3 target:
  - page-level rules stay in the template: body, scrollbars, links outside widgets
  - widget rules become Bokeh 3 style sheets: the `.slick-*` table rules, `.bk` fonts, `.bk-btn-success.bk-active` with its `:after` "ning...", and links inside `Div`s
- [ ] **Step 2: Implement the style sheets in `theme.py`.**
  - Preferred: through the theme (`theme.yaml`), for each model type, using `stylesheets` as raw CSS strings.
  - Otherwise: one shared `InlineStyleSheet` per widget kind, attached where the widget is built.
  - Each rule is written once.
  - Widget defaults that differ in Bokeh 3 are restored to today's colors: button colors per `button_type`, inputs, selects, sliders and tabs.
- [ ] **Step 3: Compare and iterate.**
  - Run `browser_check.py` on the `bokeh3` app.
  - Run `compare_screenshots.py <scratch>/shots_bokeh2 <scratch>/shots_bokeh3 <scratch>/compare.html`.
  - Fix deviations in CSS first. Change sizes in code only where CSS cannot match the arrangement.
  - Repeat until each page pair matches in arrangement and colors.
  - Remaining deviations are listed in the report with a reason, for example "Bokeh 3 slider handle shape".
- [ ] **Step 4: Run the suite** without the cascade test, as in Task 2 step 3. Expected: PASS, including `tests/test_theme.py`.
- [ ] **Step 5: Commit** (`Port the dark style to Bokeh 3 style sheets`, plus separate commits for any size changes).

---

### Task 5: Cascade golden under Bokeh 3

**Files:**
- Create: `tools/cascade_full_states.py`, `tools/compare_cascade_states.py`
- Modify: `tests/test_gui_cascade.py` (docstring: accepted differences; code only where a Bokeh 3 property read needs it), `tests/gui_cascade_golden.json`

**Interfaces:**
- Produces:
  - `python tools/cascade_full_states.py REPO_DIR OUT.json` imports `tests.test_gui_cascade` from `REPO_DIR`, replays it with `compact` replaced by the identity, and writes `[[step, state], ...]` with full values. It must run under py3.8/Bokeh 2.2 against a `master` worktree and under py3.12/Bokeh 3 against `bokeh3`.
  - `python tools/compare_cascade_states.py OLD.json NEW.json` prints every differing key per step, grouped into: `missing/renamed property`, `model_count`, `model type name`, `numeric within rtol 1e-9`, `size-derived` (keys listed with `--size-keys`), and `OTHER`. It exits 1 if `OTHER` is non-empty.

- [ ] **Step 1: Write both tools.**
- [ ] **Step 2: Dump the full states on both sides.**
  - Old: a `master` worktree in env `pyPALC`.
  - New: `bokeh3` in env `pyPALC3`.
- [ ] **Step 3: Compare the dumps.** Expected: `OTHER` is empty. If not, each `OTHER` key is either a bug, fixed with a `fix:` commit and a re-dump, or a case for the user: STOP and report NEEDS_CONTEXT.
- [ ] **Step 4: Re-record the golden.** Run `python -m tests.test_gui_cascade tests/gui_cascade_golden.json` in `pyPALC3`. Then update the test docstring: replace the Bokeh 2 model-count note and add an "Accepted differences after the Bokeh 3 upgrade" section with the categories and their counts from step 3.
- [ ] **Step 5: Run the full suite** in `pyPALC3`. Expected: PASS, all tests.
- [ ] **Step 6: Commit** the tools first (`Add tools to compare full cascade states`), then the golden with its docstring (`Re-record GUI cascade golden for Bokeh 3`).

---

### Task 6: Final verification and docs

**Files:**
- Modify: `README.md`, `docs/superpowers/gui-checklist.md`

- [ ] **Step 1: Update the README.**
  - Environment creation (`conda create -n pyPALC3 -c conda-forge --file requirements.txt`, or pip with Python 3.12).
  - The start command (unchanged).
  - `PYTHONNOUSERSITE=1`.
  - The optional browser tools: install selenium, Firefox and geckodriver, and the `tools/` usage.
  - Nothing else in the README changes.
- [ ] **Step 2: Update the GUI checklist.** Mark which items `tools/browser_check.py` now covers.
- [ ] **Step 3: Do a fresh-env check.** Create `pyPALC3_check` from `requirements.txt`, run the full suite with `PYTHONNOUSERSITE=1` (expected: all pass), then remove the env.
- [ ] **Step 4: Check the headless runner.** Run `python -m pypalc run examples/simple.yaml --out <scratch>/out` in both envs (master worktree in `pyPALC`, `bokeh3` in `pyPALC3`). The CSVs must match within rtol 1e-9 (compare the numbers, not the bytes).
- [ ] **Step 5: Do the two-tab check** with selenium: two windows on one server, a line drawn in tab 1 (through the venue inputs and "Save Current Audience Line"), and tab 2's line source is still empty. Add this as a check in `tools/browser_check.py` (`--two-tabs`). Expected: ok.
- [ ] **Step 6: Run the final browser check** on `bokeh3` and the Bokeh 2 reference. Write `<scratch>/compare.html` for the user, and note the remaining deviations in the report.
- [ ] **Step 7: Commit** (`Document the Bokeh 3 environment and browser tools`).
