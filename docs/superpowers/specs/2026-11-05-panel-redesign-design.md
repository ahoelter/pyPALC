# pyPALC Panel redesign: design

Date: 2026-11-05
Status: approved (2026-11-05)

This is the third of five projects:

1. Code cleanup (done, merged into `master`)
2. Migration to Bokeh 3 (done, merged into `master` on 2026-11-04)
3. **Redesign of the GUI with Panel** (this document)
4. Sphinx documentation on Read the Docs
5. Better approaches for target slope optimization

## Intent

The user saw a throwaway prototype of pages 1 and 4 in a modern Material
design (branch `modern-design-prototype`, folder `prototype/`) and liked it.
The whole GUI now gets that design for real. It replaces the Bokeh GUI on
`master`; the old GUI stays in the git history.

The redesign comes before the Sphinx docs, so that the docs describe and
show the final GUI.

## Goals

- **The modern Material look of the prototype on all six pages** (Help,
  1 venue, 2 loudspeaker, 3 array, 4 algorithm, 5 sound field):
  - dark mode, with today's colors: background `#323030`, white text, green
    success buttons (`#5cb85c`), red danger buttons (`#d9534f`), orange
    warning buttons (`#f0ad4e`), blue primary (`#428bca`), green links
  - a header bar with the app name, the pages as tabs and the
    Audio Communication Group / TU Berlin logos with the link
  - the numbered steps grouped in cards with titles, outlined inputs,
    rounded corners
  - the prototype screenshots of pages 1 and 4 are the reference for the
    look; pages 2, 3, 5 and Help follow the same style
- **Same pages, same steps (1–27), same features** as the current GUI.
  The page flow stays (for example: Load Setup runs the calculation and
  switches to page 5).
- **Shorter labels, as in the prototype** (user decision). The step numbers
  1–27 stay; the wording is shortened so labels fit the outlined Material
  inputs, for example "1. Start Position (x) of Audience Line in m" →
  "1. Start x in m", "19. Gap Handling Approach" → "19. Gap handling",
  "Hinges in Target" → "Hinges". Labels the prototype did not cover are
  shortened in the same spirit. The help page text and the README are
  updated to the new labels.
- **The "PALC Run" toggle** turns dark red and reads "Running..." while a
  calculation runs, also during Load Setup.
- **Downloads:**
  - "Download PALC Angles" stays one CSV file (same columns as today).
  - "Download SPL" is renamed **"Download Results"** and downloads one zip
    file with all result CSV files (today five separate downloads: SPL over
    distance, reference SPL over distance, target slope, target reference,
    homogeneity). The CSV files inside keep today's names and columns.
  - "Save Setup" downloads the YAML setup file as today.
- **Works on laptops and large monitors.** No horizontal scrolling at a
  window width of 1280 px (a common laptop) and up. On wide screens the
  cards and figures use the space; on narrower ones they shrink or move
  below each other. The figures resize with the window, while the venue
  figure keeps its three axis modes (user decision on how they behave):
  - **Equal, variable height:** equal scaling (1 m on x looks as long as
    1 m on y); the figure height follows the width, but only up to a
    maximum height so the figure never runs below the window. Above that,
    it behaves like "Equal, fixed height".
  - **Equal, fixed height:** equal scaling at a fixed height; the visible
    x range follows the width.
  - **Stretched:** the data fills the figure, as today.

  The page 5 figures keep equal scaling in the same way. The axis ranges
  and figure sizes may differ from the current GUI; in the comparison with
  the recorded data they are listed as intended changes.
- **Everything keeps working:** drawing, tapping and removing audience
  lines; PALC venue proposal; loudspeaker and array settings including the
  measured directivity upload; the PALC run; dragging the target slope;
  the sound field prediction; downloads; Save and Load Setup; two browser
  tabs on one server must not share state; the headless runner.
- **Results stay the same:** the GUI computes exactly what the headless
  runner computes (as tested today).

## Non-goals

- No change to the calculation (`pypalc/core`), the setup file format or the
  CSV columns.
- No new features beyond "Download Results" as a zip.
- No reorganization of pages or steps.
- No target slope work (project 5) and no Sphinx docs (project 4).
- No phone or tablet layout (window widths below 1280 px may scroll).
  But a phone/tablet layout must stay possible later (user wish, after the
  other projects): no layout decision may block it, for example cards at
  fixed pixel positions or widths that cannot shrink.

## Anti-goals

This would be a failure even if the app technically runs:

- Results differ from the headless runner, or Load/Save stop round-tripping.
- Widgets exist twice (visible Panel widgets mirrored onto hidden Bokeh
  widgets, as in the prototype). That approach was rejected (see below).
- The GUI tests are dropped or weakened instead of rewritten: every check
  the current GUI tests make must have a counterpart for the new GUI.
- The app needs horizontal scrolling on a 1280 px wide laptop window.
- A feature of the current GUI silently disappears.

## Constraints

- Everything from projects 1 and 2 still holds:
  - `pypalc/core` does not import Bokeh or Panel.
  - No per-session state at module level.
  - numpy-style docstrings.
  - One bug per commit, with the message `fix: <bug> (changes results: <which cases>)`.
  - No `Co-Authored-By` or other attribution lines in commits.
- Exact version pins in `requirements.txt` (conda-forge) and
  `requirements_pip.txt`. New: `panel` 1.10.0 and `panel-material-ui`
  0.16.6 (pre-1.0, so pinned exactly) plus whatever they need. The pins
  that only the Bokeh GUI needed may go.
- Run from the env `pyPALC3-panel` during the work (Python 3.12,
  bokeh 3.10.0, panel 1.10.0, panel-material-ui 0.16.6), always with
  `PYTHONNOUSERSITE=1`. The builder may rebuild the env from the new
  requirements; the final check is a fresh env from `requirements.txt`.
- Work on branch `panel-redesign` (from `master`). Push and merge only when
  the user says so; the user uses SSH for GitHub.
- `docs/source`, `docs/build` and `docs/Makefile` stay untouched (project 4).

## Approaches

- **A. Rebuild the pages directly in Panel** (chosen). Each page becomes a
  Panel component with panel-material-ui widgets; callbacks are written
  against those widgets. Kept: `pypalc/core`, the session state, Save/Load
  logic (`setup_io`), the plot ranges and the Bokeh figures (embedded in
  Panel, made responsive). One widget layer; the tests check what the user
  sees. Most work: every page's callbacks and the GUI tests are rewritten.
- **B. Keep the prototype's approach** (Panel widgets mirrored onto hidden
  Bokeh widgets). Fastest and keeps most tests, but every widget exists
  twice, sync bugs are likely (for example when Load changes widgets), the
  tests check hidden widgets, and responsive layout fights the hidden one.
  Rejected.
- **C. Restyle the Bokeh widgets with CSS only.** No new dependency, tests
  barely change, but Bokeh widgets cannot reach the Material look the user
  liked (outlined inputs, card titles, header tab bar). Rejected.

The user chose A.

## Decided by the user

- Same pages, steps and features; just the new design.
- "Download SPL" becomes "Download Results", one zip file.
- Laptops and large monitors, both if possible.
- Shorter labels as in the prototype (step numbers stay).
- Equal axis modes keep equal scaling; "variable height" gets a maximum
  height; ranges and sizes may differ from the current GUI.
- The new GUI replaces the Bokeh GUI when merged; the old one stays in the
  git history.

## Proposals the user has not contradicted (designer's calls)

- Pages 2, 3, 5 and Help get the same card style as pages 1 and 4 of the
  prototype.
- The GUI tests are rewritten with the same checks as today:
  - GUI equals headless (today 13 cases in `tests/test_gui_equivalence.py`)
  - two sessions do not share state (`tests/test_gui_sessions.py`)
  - Save/Load round trip and applying a setup (`tests/test_apply_setup.py`,
    `tests/test_setup_io.py`)
  - downloads (`tests/test_download.py`)
  - the browser check with screenshots (`tools/browser_check.py`), ported
    to the new GUI
- The 100-step cascade golden (`tests/gui_cascade_golden.json`,
  `tests/test_gui_cascade.py`) records Bokeh widget states that will no
  longer exist. It is replaced by an equivalent check for the new GUI: a
  long scripted sequence of GUI steps whose application data (source data,
  texts, values, options, the stored configuration, the results) is
  compared with the data the current Bokeh GUI produced for the same steps.
  The reference data is recorded from `master` before the Bokeh GUI is
  removed. Differences are allowed only where they come from intended
  changes (the renamed download, the shorter labels, figure ranges and
  sizes) and are listed. Calculation results, source data and the stored
  configuration must match.
- The start command becomes `panel serve` (exact form left to the builder;
  the README says how). `main.py` stays the entry point if practical.
- `theme.py`, `theme.yaml`, `templates/index.html` and `ext/*.js` go if the
  new GUI no longer needs them.

## How the current GUI works, and what changes

- `main.py` builds `PalcApp(curdoc()).show()` per session.
  `pypalc/gui/app.py` holds the session (`SessionState`, shared widgets,
  venue figure, cross-page callbacks); one module per page
  (`page_venue.py`, `page_loudspeaker.py`, `page_array.py`,
  `page_algorithm.py`, `page_results.py`, `help_text.py`).
  `show()` switches pages with `doc.clear()` / `add_root`.
- The venue figure is shared by pages 1 to 4. In Panel only one page is
  shown at a time, so it can stay shared if pages are swapped, not all
  rendered at once.
- Callbacks are Bokeh `on_change` / `on_click`. Setting a widget from Python
  fires them too, which is why `apply_setup` sets `state.applying_setup` to
  suppress them. In Panel, `param.watch` behaves the same way; the builder
  keeps an equivalent guard.
- Downloads use `CustomJS` (`ext/download.js`, `ext/download_text.js`). In
  Panel: `FileDownload` with Python callbacks (as in the prototype).
- The prototype (`prototype/app.py` on `modern-design-prototype`) holds the
  Material theme config, the logo assets (a light version of the ACG logo
  for the dark header) and the page 1 and 4 layouts. Its theme, assets and
  layouts can be reused; its widget mirroring cannot.

## Verification

1. All tests pass in a fresh environment built from the new
   `requirements.txt` (with `PYTHONNOUSERSITE=1`).
2. Screenshots of every page (Help, 1–5), fresh and after loading a target
   slope setup, at window widths 1280 px and 1920 px. Put side by side with
   the current Bokeh GUI (from `master`) for the user, as an HTML page.
3. The "Running..." indicator during a run and during Load.
4. Two-tab check: drawing in one tab does not show up in the other.
5. Download Results zip: contains the expected files with today's headers.
6. The headless runner works unchanged.
7. README updated: environment, start command, structure.

## Left to the builder

- Module structure of the new GUI (for example one module per page as now).
- How the responsive layout is done (Panel sizing modes, CSS grid, breakpoints).
- The widget for the tilt angle table (Tabulator or Bokeh DataTable).
- The form of the rewritten tests and of the browser check.
- The order of commits.
