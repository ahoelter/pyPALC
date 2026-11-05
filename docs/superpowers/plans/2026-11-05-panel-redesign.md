# Panel Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Bokeh widget GUI of pyPALC with a Panel / panel-material-ui GUI in the prototype's modern dark Material look, on all six pages, with the same features and results.

**Architecture:** `PalcApp` and one class per page stay (same module names, same attribute names for widgets, sources, figures and callback methods), but the widgets become panel-material-ui widgets and the callbacks `param.watch` / `on_click` handlers. The figures stay Bokeh figures embedded with `pn.pane.Bokeh`, now responsive. A Panel shell (`pmui.Page`) replaces `show()`; during the port, pages not yet ported are embedded as Bokeh layouts, so the app and the tests stay green after every task. A GUI-agnostic test driver lets one scripted 100-step sequence compare the new GUI's application data with data recorded from the Bokeh GUI before it changes.

**Tech Stack:** Python 3.12, panel 1.10.0, panel-material-ui 0.16.6, bokeh 3.10.0 (figures), numpy 2, pytest, selenium + headless Firefox for browser checks.

**Spec:** `docs/superpowers/specs/2026-11-05-panel-redesign-design.md`

## Global Constraints

- All six pages (Help, 1–5) in the prototype's look: dark mode; background `#323030`, white text, success `#5cb85c`, danger `#d9534f`, warning `#f0ad4e`, primary `#428bca`, green links; header bar with app name, page tabs, ACG/TU logos and link; steps in cards with titles; outlined inputs; rounded corners. Reference: `prototype/app.py` on branch `modern-design-prototype` (theme config, colors, header, page 1 and 4 layouts, logo assets).
- Same pages, steps 1–27 and features; page flow unchanged (Load Setup runs the calculation and switches to page 5).
- Shorter labels as in the prototype; step numbers 1–27 stay; labels the prototype did not cover are shortened in the same spirit; help text and README follow.
- "PALC Run" turns dark red (`darkred` / `#8b0000`) and reads "Running..." while a calculation runs, also during Load Setup.
- "Download PALC Angles": one CSV (columns of `DOWNLOAD_COLUMNS['results']`). "Download SPL" is renamed **"Download Results"**: one zip with the five CSVs of today (names `DOWNLOAD_FILE_NAMES`, columns `DOWNLOAD_COLUMNS`). "Save Setup" downloads the YAML file.
- No horizontal scrolling at window width 1280 px and up; figures resize with the window. Venue figure axis modes: "Equal, variable height" (equal scaling, height follows width up to a maximum so the figure never runs below the window, then behaves like fixed height), "Equal, fixed height" (equal scaling at a fixed height; the whole venue stays visible and one range widens to fill the figure), "Stretched" (as today). Page 5 figures keep equal scaling the same way. No layout decision may block a later phone/tablet layout (no fixed pixel positions, no widths that cannot shrink).
- Results equal the headless runner (as tested today). `pypalc/core` imports neither Bokeh nor Panel. No per-session state at module level.
- numpy-style docstrings. One bug per commit: `fix: <bug> (changes results: <which cases>)`. No `Co-Authored-By` or other attribution lines in commits.
- Exact pins in `requirements.txt` (conda-forge) and `requirements_pip.txt`; `panel==1.10.0`, `panel-material-ui==0.16.6`.
- Env `pyPALC3-panel`; every command with `PYTHONNOUSERSITE=1`, e.g. `PYTHONNOUSERSITE=1 conda run -n pyPALC3-panel python -m pytest -q`. Servers: `conda run --no-capture-output`; stop them by PID, never `pkill -f` with a pattern that can match your own shell.
- Branch `panel-redesign`. Never push or merge. `docs/source`, `docs/build`, `docs/Makefile` untouched.
- Start command: `panel serve pyPALC --show` from the folder that contains the repo folder (verified: `panel serve` serves a directory app with `main.py`).

## Review Focus

1. **Load Setup while watchers fire.** Setting Panel widget values from Python fires `param.watch` callbacks synchronously, like Bokeh `on_change`; `apply_setup` must still leave the session exactly as the hand path. Pinned by `tests/test_apply_setup.py` (ported in every page task).
2. **Two browser sessions.** Panel objects or caches created at import time (`pn.state.cache`, module-level widgets, a module-level `pmui.Page`, theme objects holding widgets) leak state between sessions. Pinned by `test_no_module_level_panel_objects` (Task 2) and the two-tab browser check (Task 7).
3. **Run clicked twice / during Load.** A second click on "PALC Run" while a calculation runs, or during Load, must not start a second calculation; the running look always resets, also after an error. Pinned by `test_run_ignored_while_running` and `test_running_look_resets_after_error` (Task 5).
4. **Invalid files.** A broken YAML in Load Setup or a broken CSV in the directivity upload shows the error text, does not raise, and leaves the session usable. Pinned by `test_load_invalid_yaml_shows_error` (Task 5) and `test_upload_invalid_csv_shows_error` (Task 4).
5. **Downloads before any run.** "Download PALC Angles" and "Download Results" before a calculation give header-only CSVs (as today), not an exception. Pinned by `test_downloads_before_run_are_header_only` (Task 5).

---

### Task 1: Environment, test driver and the reference recording

Record what the Bokeh GUI does, in a form the new GUI can be compared with, before anything changes.

**Files:**
- Modify: `requirements.txt`, `requirements_pip.txt`
- Create: `tests/gui_driver.py`, `tests/gui_steps.py`, `tools/record_gui_reference.py`, `tests/gui_reference.json` (generated), `tests/test_gui_reference.py`
- Modify: `tests/test_gui_cascade.py` (import its steps from `tests/gui_steps.py`; otherwise unchanged)

**Interfaces:**
- Produces `tests/gui_driver.py` (works for Bokeh and Panel models; legacy property names):
  - `read(model, prop: str) -> Any` — `prop` is a Bokeh-era name (`value`, `active`, `text`, `label`, `title`, `options`, `disabled`, `visible`, `button_type`, `start`, `end`, `step`, `data`, `selected`). For Panel models it maps: Toggle `active` → `value`; text panes (`pmui.Typography`, `pn.pane.HTML`/`Markdown`, `pmui.Alert`) `text` → `object`; `title` → `label`; FileInput `value` → base64 text of the bytes (`''` when empty). Options are returned as a list of labels. Raises `AttributeError` if the model has no such property. Rounds nothing (normalizing is the caller's job).
  - `has(model, prop: str) -> bool` — for Panel models `label` counts only on buttons, toggles and downloads (the label of a Panel input is the old `title`, which is not recorded).
  - `write(model, prop: str, value) -> None` — same mapping; for `active` of a Bokeh Toggle sets `active`.
  - `upload(model, data: bytes, filename: str) -> None` — Bokeh FileInput: `set_from_json('value', base64)` (and filename); Panel FileInput: sets `filename` then `value`.
  - `click(model) -> None` — Bokeh Button: fires its Python `on_click` handlers; Panel Button: `model.clicks += 1`.
- Produces `tests/gui_steps.py`: `steps(app) -> list[tuple[str, callable]]` — the 100 steps of today's `test_gui_cascade._steps`, unchanged in order and names, written with the driver (`write`, `upload`) and the page methods, so they run on both GUIs. `MODELS` and `CONFIG_KEYS` move here too.
- Produces `tools/record_gui_reference.py OUT.json`: replays `steps` on a fresh `PalcApp()` and records after every step, for every model in `MODELS`, the properties `visible, disabled, value, options, active, text, label, button_type, start, end, step, data, selected` (through `read`; skip a property the model lacks), `config.<key>` for `CONFIG_KEYS`, and `error`. Not recorded: `title`, ranges, `frame_height`, `min_border`, model counts (intended to change). Floats rounded to 9 decimals; values whose JSON text exceeds 1500 chars stored as `{"sha1", "size"}` as in today's golden; first state complete, later steps only changes (same format as `gui_cascade_golden.json`).
- Produces `tests/test_gui_reference.py`:
  - `RENAMED_LABELS: dict[str, str]` (old → new text; empty now; every later task adds its renamed labels). Applied to recorded `label` values and to the entries of `options` and string `value`s that equal a key exactly (for example the venue axis options), nothing else.
  - `INTENDED: list[tuple[str, str]]` (key glob → reason; empty now).
  - `test_gui_matches_reference` — replays the steps on a fresh `PalcApp()`, records as the tool does, maps recorded `label` values through `RENAMED_LABELS`, skips keys matching `INTENDED`, and asserts equality with `tests/gui_reference.json` per step (message names step and key). A key present in one recording but not the other is a difference too, unless `INTENDED` covers it.

- [ ] **Step 1: Pin the new packages.** Add `panel==1.10.0`, `panel-material-ui==0.16.6` and exact pins (from `conda list -n pyPALC3-panel`) for every package they pull in that is not pinned yet, to both requirement files. Check every pip pin exists on PyPI. Build a throwaway env from `requirements.txt` (`conda create -n pyPALC3-panel-check -c conda-forge --override-channels --file requirements.txt`), run the suite in it, remove it.
- [ ] **Step 2: Write `tests/gui_driver.py` and `tests/gui_steps.py`;** switch `tests/test_gui_cascade.py` to import the steps. Run `python -m pytest tests/test_gui_cascade.py -q` → PASS (proves the steps are unchanged).
- [ ] **Step 3: Write `tools/record_gui_reference.py`, run it** → `tests/gui_reference.json`.
- [ ] **Step 4: Write `tests/test_gui_reference.py`;** run it → PASS on the Bokeh GUI. Also run it once with one recorded value edited by hand → FAIL naming that step and key; revert.
- [ ] **Step 5: Full suite** → all pass. Commit (requirements; driver + steps; reference tool + file + test).

### Task 2: Panel shell, style and page switching

**Files:**
- Create: `pypalc/gui/style.py` (colors, `MATERIAL_THEME` dict for `pmui.Page(theme_config=...)`, `bokeh_figure_theme() -> bokeh.themes.Theme` for the figures; values from the prototype), `static/ak_logo_light.png`, `static/tu_logo.png` (copied from `prototype/assets/` on `modern-design-prototype`)
- Modify: `pypalc/gui/app.py`, `main.py`, `README.md` (start command)
- Create: `tools/page_screenshots.py`
- Create: `tests/test_shell.py`
- Delete: `pypalc/gui/theme.py`, `theme.yaml`, `templates/index.html`, `tests/test_theme.py`, `tests/test_show_detached.py`, `tests/test_gui_cascade.py`, `tests/gui_cascade_golden.json`, `tools/cascade_full_states.py`, `tools/compare_cascade_states.py`, `tools/compare_screenshots.py` only if Task 7 does not need it (keep otherwise)
- Modify: `tests/test_gui_sessions.py`

**Interfaces:**
- `PalcApp(doc: Optional[Document] = None)` — `doc` defaults to `pn.state.curdoc` when served, else a new `Document()`. `show()` is removed.
- `PalcApp.view() -> pmui.Page` — header: "pyPALC" + subtitle, `app.page_select` (page tabs Help, 1 Venue, 2 Loudspeaker, 3 Array, 4 Algorithm, 5 Sound Field; `active` = page index 0–5), logos as data URIs from `static/` with the ACG link; main: one content area holding the current page only.
- `PalcApp.show_page(index: int) -> None` — swaps the content area to `pages[index]`'s view; called by `page_select`'s `active` watcher (keep today's page change side effects, e.g. clearing the result figure on page 1, from `show()`).
- `PalcApp.later(func) -> None` — served: `doc.add_timeout_callback(func, 120)` so the browser shows changes made so far first; not served (tests): calls `func()` now.
- Every page keeps `layout()`; a page not yet ported returns a Bokeh layout, which `show_page` wraps in `pn.pane.Bokeh(...)`. Until their task, unported pages may look plain (the Bokeh widget style sheets go with `theme.py`); they must work. `templates/index.html` and `theme.yaml` must go in this task, because the directory handler of `panel serve` would apply them to the Panel page too. `config_error_info` stays shared (Bokeh `Paragraph` until Task 5).
- `main.py`: `PalcApp().view().servable(title="pyPALC")`, and sets the figure theme on the document.

- [ ] **Step 1: Failing tests in `tests/test_shell.py`:** `test_page_select_switches_content` (writing `active` 0..5 shows exactly that page's view in the content area); `test_view_is_material_page` (`isinstance(app.view(), pmui.Page)`, dark theme, `MATERIAL_THEME` palette colors as in Global Constraints); in `tests/test_gui_sessions.py`: `test_no_module_level_panel_objects` (no `panel.viewable.Viewable` or Bokeh `Model` instance at module level in any `pypalc.gui` module, and `pn.state.cache` empty after building two apps).
- [ ] **Step 2:** Run them → FAIL.
- [ ] **Step 3:** Implement shell, style, `later`, `show_page`; delete the files listed above; update `README.md` start command to `panel serve pyPALC --show`.
- [ ] **Step 4:** Full suite → PASS (reference test unchanged).
- [ ] **Step 5:** Write `tools/page_screenshots.py --url URL --out DIR --width W [--height H]` (headless Firefox, `/usr/bin/firefox`, `/snap/bin/geckodriver`; clicks each page tab, waits until the page is stable, saves `<index>_<name>.png`; reports `document.documentElement.scrollWidth > innerWidth` per page as horizontal overflow). Start the app with `panel serve pyPALC --port 5012` from the parent folder, run at width 1280 → all six pages render (unported pages in the old widgets), no server errors in the log.
- [ ] **Step 6:** Commit.

### Task 3: Page 1 (venue) and the shared venue figure

**Files:**
- Modify: `pypalc/gui/page_venue.py`, `pypalc/gui/app.py` (shared widgets `venue_axes_select`, venue figure), `pypalc/gui/ranges.py`
- Modify: `tests/test_gui_reference.py` (`RENAMED_LABELS`, `INTENDED`), `tests/gui_driver.py` if a widget type needs mapping
- Test: `tests/test_venue_axes.py` (new)

**Interfaces:**
- Page 1 widgets become pmui widgets with the same attribute names (`start_x_input`, `start_y_input`, `end_x_input`, `end_y_input`, `points_per_meter_input`, `save_line_button`, `remove_line_button`, `insert_line_button`, `remove_selected_line_button`, `check_venue_button`, `venue_error_info`); sources and methods unchanged. Layout and labels as the prototype's page 1 (cards "Audience line", "Lines and venue", "Venue slice").
- `app.venue_axes_select`: `pmui.RadioButtonGroup`, `active` 0/1/2 = "Equal, variable height" / "Equal, fixed height" / "Stretched".
- The venue figure is responsive (`sizing_mode` that follows the width). `ranges.scale_venue_figure(venue_figure, ranges, axes_select)` keeps its name; how equal scaling is achieved (for example `match_aspect`, `aspect_ratio` with `max_height`, or a size-dependent range) is the implementer's choice, meeting the axis-mode rules of the Global Constraints. The maximum height of "variable height" is a constant in `ranges.py`.

- [ ] **Step 1: Failing tests** in `tests/test_venue_axes.py`: for each axis mode, after drawing the lines of `tests/cases/` case `simple` (use `apply_setup` or the venue page methods), the figure's server-side settings imply equal scaling for modes 0 and 1 (`match_aspect` or equal data-per-pixel in the configured sizes, whichever mechanism is chosen) and not for mode 2; mode 0 sets a maximum height equal to the constant.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Port page 1 and the shared widgets; add renamed labels to `RENAMED_LABELS` and the venue figure keys that changed to `INTENDED` (reason: "responsive venue figure" — only figure-derived keys, never source data).
- [ ] **Step 4:** Full suite → PASS (`test_gui_reference`, `test_gui_equivalence`, `test_apply_setup`, `test_gui_sessions`).
- [ ] **Step 5:** Browser: `tools/page_screenshots.py` at 1280 and 1920; in addition measure, for each axis mode with the `simple` case drawn, the venue figure's pixels per meter on x and y in the browser (via `Bokeh.documents[0]`: frame size and range) → modes 0 and 1 differ by < 1 %, and in mode 0 the figure's bottom is inside the window at 1280×800. Look at the screenshots: page 1 matches the prototype's page 1 look. No horizontal overflow.
- [ ] **Step 6:** Commit.

### Task 4: Pages 2 (loudspeaker) and 3 (array)

**Files:**
- Modify: `pypalc/gui/page_loudspeaker.py`, `pypalc/gui/page_array.py`, `pypalc/gui/setup_io.py` (only where it reads/writes widget properties that changed names)
- Modify: `tests/test_gui_reference.py`, `tests/gui_driver.py` as needed
- Test: `tests/test_page_loudspeaker.py` (new)

**Interfaces:**
- Same attribute names as today (see `MODELS` in `tests/gui_steps.py`: `aperture_input`, `cabinet_height_input`, `directivity_select`, `directivity_upload`, `directivity_source`, `cabinet_count_input`, `top_x_input`, `top_y_input`, `cabinet_gap_input`, `use_discrete_angles_select`, `discrete_angles_input`, `reference_type_select`, `reference_discrete_angles_select`, `reference_start_input`, `reference_step_stop_input`, `reference_user_angles_input`, `reference_angles_source`). `directivity_upload` is a `pmui.FileInput` (accept `.csv`); its watcher calls `read_measured_directivity(text)` with the decoded file.
- Cards in the prototype's style; figures (directivity, array, venue) responsive.

- [ ] **Step 1: Failing test** `test_upload_invalid_csv_shows_error`: uploading `b"not,a\ncsv"` through the driver shows an error text (non-empty, starts with "Error"), does not raise, and a following valid upload of `tests/cases/measured.csv` works.
- [ ] **Step 2:** Run → FAIL (the widget is still Bokeh or the behavior differs).
- [ ] **Step 3:** Port both pages; update `RENAMED_LABELS` / `INTENDED` (figure-derived keys only, with reason).
- [ ] **Step 4:** Full suite → PASS.
- [ ] **Step 5:** Browser screenshots at 1280/1920 of pages 2 and 3, also after a measured-directivity upload; no horizontal overflow.
- [ ] **Step 6:** Commit.

### Task 5: Page 4 (algorithm): run, downloads, Save and Load

**Files:**
- Modify: `pypalc/gui/page_algorithm.py`, `pypalc/gui/app.py` (`config_error_info` as a pmui text/alert, running state), `pypalc/gui/setup_io.py` as needed
- Create: `pypalc/gui/downloads.py`
- Delete: `ext/download.js`, `ext/download_text.js` (and `ext/` if empty)
- Rewrite: `tests/test_download.py`
- Modify: `tests/test_apply_setup.py`, `tests/test_gui_equivalence.py`, `tests/test_gui_reference.py` as needed
- Test: `tests/test_run_indicator.py` (new)

**Interfaces:**
- Same attribute names as today for the page 4 widgets (see `MODELS`), plus `download_angles_button`, `download_results_button` (renamed from `download_spl_button`; label "Download Results"), `save_setup_button`, `load_setup_input`, `results_table`. Downloads are `pmui.FileDownload` with callbacks; `load_setup_input` is a `pmui.FileInput` (accept `.yaml,.yml`).
- `start_toggle`: a `pmui.Toggle`; `value=True` starts a calculation through `app.later` (synchronously in tests). While running: label "Running..." (prototype: "26. PALC Run" otherwise) and dark red background; reset in a `finally`.
- `pypalc/gui/downloads.py`:
  - `angles_csv(page) -> str` — header + rows of `results_source` in `DOWNLOAD_COLUMNS['results']` order.
  - `results_zip(page) -> bytes` — zip with one CSV per entry of `DOWNLOAD_FILE_NAMES` (except the angles), columns of `DOWNLOAD_COLUMNS`.
  - `setup_yaml(page) -> Optional[str]` — what "Save Setup" writes today (`setup_to_save` + `validate` + `dump_setup`); on `SetupError` sets the error text as today and returns None.
  - Numbers formatted with `pypalc.core.export._fmt` (exact round trip; replaces the JavaScript number format).
- Layout as the prototype's page 4 (cards PALC, Weighting, Tolerance and fixed angle, Run, Files, Tilt angles, Results with tabs "Venue slice" / "SPL over distance"); button order: check venue, run, delete result, error message, download angles, download results, save setup, load setup, then the table.

- [ ] **Step 1: Failing tests:**
  - `tests/test_download.py`: `test_angles_csv_matches_results_source`, `test_results_zip_has_five_csvs_with_headers` (names = `DOWNLOAD_FILE_NAMES` without angles, headers = `DOWNLOAD_COLUMNS`), `test_results_zip_values_equal_sources` (every value parses back to the source value exactly), `test_downloads_before_run_are_header_only`, `test_save_setup_round_trips` (`setup_yaml` → `load_setup_text` → `apply_setup` gives the same `setup_to_save`).
  - `tests/test_run_indicator.py`: `test_running_look_during_run` (inside the calculation, observed via a patched `pipeline.run`, the toggle label is "Running..." and its style is dark red; after it, label and style are back), `test_run_ignored_while_running`, `test_running_look_resets_after_error` (patched `pipeline.run` raising `PalcError`), `test_running_look_during_load`.
  - `tests/test_apply_setup.py`: `test_load_invalid_yaml_shows_error` (uploading `b"::: not yaml"` sets an error text, toggle not running, no exception).
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Port page 4 and the shared error text; implement `downloads.py`; delete `ext/*.js`; update `RENAMED_LABELS` (including "Download SPL" → "Download Results") and `INTENDED` (figure-derived keys of `spl_over_distance_figure`, reason).
- [ ] **Step 4:** Full suite → PASS.
- [ ] **Step 5:** Browser: page 4 at 1280/1920 fresh, after Load of `tests/cases/gap_soft_margin_target_slope.yaml` (switches to page 5), while running; drag a target point in the SPL tab (the server snaps it); download angles, results zip and setup and check their contents. No horizontal overflow.
- [ ] **Step 6:** Commit.

### Task 6: Page 5 (sound field) and Help; remove the Bokeh widget leftovers

**Files:**
- Modify: `pypalc/gui/page_results.py`, `pypalc/gui/help_text.py`, `pypalc/gui/ranges.py` (`set_figure_ranges`), `pypalc/gui/app.py`, `README.md`
- Modify: `tests/test_gui_reference.py`, tests as needed

**Interfaces:**
- Page 5 widgets with today's names (`beam_frequency_select`, `beam_checkbox`, sources and figures as in `MODELS`); figures responsive with equal scaling as in the Global Constraints.
- Help: `pmui.Tabs` with the existing help texts as HTML/Markdown panes, labels in the texts updated to the new short labels.
- After this task no page returns a Bokeh widget layout; `show_page`'s `pn.pane.Bokeh` wrapping of whole pages is removed (figures are still `pn.pane.Bokeh`).

- [ ] **Step 1: Failing tests:** `tests/test_shell.py::test_no_bokeh_widgets_left` (no `bokeh.models.widgets` instance among the attributes of the app and its pages); `test_help_mentions_new_labels` (every label in `RENAMED_LABELS`' values that names a step appears in the help text, none of its keys does).
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Port page 5 and Help; update `RENAMED_LABELS` / `INTENDED`; README: new labels where quoted, structure table, start command.
- [ ] **Step 4:** Full suite → PASS.
- [ ] **Step 5:** Browser screenshots of pages Help and 5 at 1280/1920, fresh and after Load; no horizontal overflow; tap a venue point on page 5 (beam plot updates).
- [ ] **Step 6:** Commit.

### Task 7: Browser check, comparison page, final verification

**Files:**
- Rewrite: `tools/browser_check.py` (for the Panel GUI; same checks as today plus the new ones below)
- Modify/Delete: `tools/page_screenshots.py` (fold into `browser_check.py` if natural), `tools/compare_screenshots.py`
- Modify: `README.md`

**Interfaces:**
- `tools/browser_check.py --url URL --out DIR [--width W] [--two-tabs] [--clean] ...` — checks (each prints `ok`/`FAIL` with detail): screenshots fresh/loaded of all pages; `load` (switches to page 5, page 4 shows the loaded settings); `download_angles`, `download_results` (zip with the five CSVs and their headers), `download_setup`; `running_indicator` (dark red + "Running..." during a run and during Load); `drag`; `rays_cleared`; `no_horizontal_overflow` (every page at the given width); `equal_axes` (pixels per meter x vs y < 1 % for modes 0 and 1); `two_tabs`.

- [ ] **Step 1:** Port the browser check; run it at `--width 1280 --two-tabs` and `--width 1920` → all ok.
- [ ] **Step 2:** Screenshots of the Bokeh GUI from `master` (env `pyPALC3`, `bokeh serve pyPALC` in a `git worktree` of `master`) and of the new GUI, fresh and after Load, all pages, at 1280 and 1920; one self-contained HTML page (images base64, old left, new right, dark background) in the session scratchpad for the user.
- [ ] **Step 3:** Fresh env from `requirements.txt` (throwaway name), full suite with `PYTHONNOUSERSITE=1` → all pass; remove the env.
- [ ] **Step 4:** Headless runner: `python -m pypalc run examples/simple.yaml --out <tmp>` works and matches the baseline CSV as today.
- [ ] **Step 5:** README final check: environment, start command `panel serve pyPALC --show`, structure, tools. Commit.
