# pyPALC Bokeh 3 upgrade: design

Date: 2026-11-04
Status: draft, awaiting approval

This is the second of four projects:

1. Code cleanup (done, merged into `master` on 2026-11-04)
2. **Migration to the current stable Bokeh 3** (this document)
3. Sphinx documentation published on Read the Docs
4. Better approaches for target slope optimization

## Intent

pyPALC runs on Bokeh 2.2 and Python 3.8. Both are old, and their dependencies no longer install cleanly. Project 1 had to pin `jinja2`, `markupsafe` and `pillow` to old versions just to get a working environment. The current stable Bokeh is 3.10 (3.10.1, released 2026-10-23), and it needs Python 3.12 or newer.

The app should run on the current Bokeh and an up-to-date Python stack, so that the documentation (project 3) and new target slope methods (project 4) build on a maintained base. To the user, the app should look and work as it does today.

The user also wants to see what a **more modern design** could look like before deciding on it. That is a separate, throwaway prototype (see the last section) and not part of the upgrade itself.

## Goals

- The app runs on **Bokeh 3.10.x** (the newest stable release when the work starts) on **Python 3.12**.
- **All Python packages are updated** to current versions: numpy, scipy, colorcet, pyyaml, pytest and Bokeh's own dependencies.
  - `requirements.txt` (conda format) and `requirements_pip.txt` are updated to match, with exact pins as before.
  - The pins that only existed for Bokeh 2.2 (`jinja2`, `markupsafe`, `pillow`) are dropped, unless the new stack still needs them.
- **Same layout and the same dark style:**
  - the pages, widgets, figures and their arrangement
  - the dark grey background (`#323030`), white text and Helvetica 10pt
  - the green links and the dark tables
- **The colors of buttons and other widgets stay as they are now.** That includes:
  - the success, danger, warning and primary buttons
  - the page buttons
  - the "PALC Run" toggle, which turns dark red and reads "Running..." while a calculation runs
- **Everything keeps working:**
  - drawing and selecting lines
  - dragging target points
  - all downloads
  - Save and Load Setup
  - the measured directivity upload
  - two browser tabs on one server, which must not share state
  - the headless runner (`python -m pypalc run ...`)
- **Results stay the same.** The calculation tests compare against the recorded baselines at rtol 1e-9.
- The work happens on the branch **`bokeh3`**, which the user reviews before it is merged.

## Non-goals

- No redesign and no new features. The modern-design prototype is separate.
- No changes to the calculation, except bug fixes that the upgrade forces.
- No target slope work (project 4) and no Sphinx docs (project 3).
- No change to the setup file format or the CSV export format.

## Anti-goals

This would be a failure even if the app technically runs:

- Results change silently because of new numpy or scipy versions.
- The app looks noticeably different: colors change, widgets move, or the "Running..." indicator is gone.
- The tests are weakened to make the upgrade pass. Re-recording a golden file without explaining every difference counts as weakening.
- The headless runner or the setup files stop working.

## Constraints

- Everything from project 1 still holds:
  - `pypalc/core` does not import Bokeh.
  - No per-session state at module level.
  - One bug per commit, with the message `fix: <bug> (changes results: <which cases>)`.
  - numpy-style docstrings.
  - No `Co-Authored-By` or other attribution lines in commits.
- The app is still started with `bokeh serve --show pyPALC`, run from the parent directory of the repo.
- Push and merge only when the user says so. The user uses SSH for GitHub.
- `docs/source`, `docs/build` and `docs/Makefile` stay untouched. They belong to project 3.

## Approaches

Three ways were considered:

- **A. One branch, four steps in order** (chosen):
  1. New environment, then run the calculation tests alone (they don't use Bokeh).
  2. Move the code to the Bokeh 3 API until the app starts.
  3. Move the styling into Bokeh 3 style sheets and compare screenshots page by page.
  4. Bring the GUI tests up to date.

  This separates changes in the numbers from changes in the look, so each problem is found where it starts.
- **B. Two hops** (Bokeh 2.4 on the new Python first, then Bokeh 3). Smaller steps, but Bokeh 2.4 is a throwaway version and much of the work would be done twice. Step 1 of A already separates numerics from the GUI. Rejected.
- **C. Rebuild the GUI on Panel**, a higher-level library built on Bokeh with modern widgets and themes. Rebuilding every page conflicts with "same layout". Rejected for the upgrade; it is used for the throwaway design prototype instead.

The user chose A.

## How it works today, and what changes

### Styling

- **`theme.yaml`** themes the plots: figure, axis, grid and title colors and fonts. Bokeh 3 still supports themes. Model names may need adjusting; for example, `Figure` in Bokeh 2 corresponds to `Plot` or `figure` in Bokeh 3.
- **`templates/index.html`** holds CSS for:
  - the page body
  - the scrollbars
  - links
  - the data tables (`.slick-*`)
  - fonts (`.bk`)
  - the running indicator: `.bk-root .bk-btn-success.bk-active` turns dark red, and an `:after` adds "ning..." to the label "Run"

  In Bokeh 3, every widget renders inside its own shadow DOM, so page-level CSS no longer reaches the widgets. Rules for the page itself (body, scrollbars) can stay in the template. **Rules for widgets must move into Bokeh 3 style sheets**, either on the widgets (`stylesheets=`) or, preferably, once for each model type through the theme. The builder decides how, but each rule should be written once, not repeated on every widget. The class names inside widgets changed in Bokeh 3 (`bk-root` is gone, for example), so each rule has to be re-targeted, not just copied.
- **Widget colors.** Bokeh 3's default widget style differs from Bokeh 2's (for example, button colors, borders and input fields). Where the defaults differ, style sheets restore the current colors.

### API changes known so far

| Bokeh 2 | Bokeh 3 | Where |
|---|---|---|
| `Panel` | `TabPanel` | 14 uses (tabs on several pages) |
| `FuncTickFormatter` | `CustomJSTickFormatter` | ν slider on page 4 |
| `plot_width` / `plot_height` | `width` / `height` | logo figures in `gui/app.py` |
| `toolbar_sticky` | removed | 4 figures |
| `legend=` on glyphs | `legend_label=` | check the 26 legend uses |
| `bokeh.models.widgets` imports | `bokeh.models` | 7 imports |

The builder checks every other use against the Bokeh 3 migration notes:
- `gridplot(..., merge_tools=False)`
- `doc.clear()` / `add_root` page switching
- `FileInput`
- `CustomJS` downloads (`ext/download.js`, `ext/download_text.js`)
- `PointDrawTool`
- `DataTable`
- `ColorBar` / `LinearColorMapper`
- `RadioButtonGroup`, `Toggle`, `Div` and `Paragraph`

### Layout

Bokeh 3 places and sizes elements with CSS grid and flexbox instead of Bokeh 2's own layout engine. Widths, heights, margins and gaps can change. The arrangement of every page must match today's, compared by screenshot (see Verification). Fixed sizes in the code are adjusted where needed.

### Numerics

New numpy and scipy versions can change results in the last digits.

- The calculation tests compare with the baselines at rtol 1e-9.
- If they fail, the cause is found before anything else changes.
  - **Pure floating-point drift** (all differences explained, no behavior change): the baselines are re-recorded in one commit that names the largest difference and its cause. *(Builder's call how to show "explained"; this rule is the designer's call.)*
  - **Anything that looks like a behavior change**, or a difference that can't be explained: stop and ask the user.
- numpy 2 removed or renamed some functions and changed some type rules, such as scalar promotion and ragged arrays. The core code is checked for these, and the warnings the test suite prints today (about 860, mostly from legacy core code) are looked at for things that numpy 2 turns into errors.

### Tests

- **Stay unchanged:**
  - the calculation and headless tests
  - the baseline comparisons
  - "GUI equals headless" (13 cases)
  - the setup save/load tests
- **The cascade golden (`tests/gui_cascade_golden.json`)** records Bokeh model state and model counts after 100 GUI steps. Under Bokeh 3, property names and model counts change. Before re-recording it:
  1. Record the new states.
  2. Show that every difference to the old golden comes from Bokeh: renamed properties, different model counts, or new default values.
  3. Show that all application data is identical: source data, texts, values, options and the stored configuration. There are two exceptions, both *designer's calls*:
     - **Numeric drift.** The golden rounds floats to 9 decimals and stores long values (results, sound field, rays) as sha1 hashes. So a drift that passes the rtol 1e-9 baseline test can still change a rounded value or a hash. If the baselines show only explained floating-point drift (see Numerics), golden values may differ by the same drift. This must be shown by recording the full, unhashed states in both environments (old `pyPALC` and new) and comparing them at rtol 1e-9. A difference beyond that means stop and ask the user. The golden test's own comparison is not loosened.
     - **Intentional size changes.** Layout fixes are done in CSS where possible. Where a size in the code has to change to match the Bokeh 2 screenshot (for example `frame_width`, from which `frame_height` and the venue ranges are computed), the recorded values that follow from it may change. Each such change is listed with its reason.
  4. Re-record the golden in one commit that lists those difference categories in the test's docstring, as project 1 did for its accepted differences.
- **Browser check.** In project 1, the browser checks were done by hand. Now a headless Firefox (selenium + geckodriver) can drive the real app.
  - The builder adds a small browser smoke script. It opens every page, loads a setup, checks the displayed values, and takes screenshots.
  - The script is **not** part of the default `pytest` run, because it needs Firefox. Either a separate script or a pytest marker that is skipped by default works; the builder decides.
  - It is used for the screenshot comparison and for the user's review.

## Verification

1. All tests pass in a fresh environment built from the new `requirements.txt` (with `PYTHONNOUSERSITE=1`).
2. Screenshots of every page (1–5 and Help) are taken at the same window size:
   - Bokeh 2: from `master`, in the old `pyPALC` environment
   - Bokeh 3: from `bokeh3`

   Two page states are included: a fresh session, and after loading a target slope setup and running it. The pairs are put side by side for the user (as an HTML page or images). Deviations are fixed, or listed with a reason.
3. The "Running..." indicator is checked in the browser during a run.
4. The two-tab check: drawing in one tab doesn't show up in the other.
5. Headless runner: `python -m pypalc run examples/simple.yaml --out ...` works, and its CSV matches.
6. The README is updated for the new Python version, the environment setup and the start command.

## Left to the builder

- How style sheets are organized (theme versus per widget), as long as each rule is written once.
- The exact package versions (current releases at the time of the work, pinned exactly).
- Whether the environment is created as a new conda env, for example `pyPALC3`, keeping the old `pyPALC` env for the Bokeh 2 screenshots.
- The form of the browser smoke script.
- The order of commits within the four steps.

## Afterwards: modern-design prototype (separate, throwaway)

**Not part of the plan for this upgrade.** The prototype starts only after the user has reviewed `bokeh3`. As a throwaway spike, it gets no plan and no review process of its own. It can use the new environment: Panel 1.10.0 needs Bokeh >=3.10.0,<3.11.

Once `bokeh3` is ready for review, a **throwaway prototype** shows the user what a more modern design could look like. The user then decides whether a redesign becomes a project of its own.

- **Branch:** its own branch, `modern-design-prototype`, from `bokeh3`. It is never merged as is.
- **Scope:** two pages, **1 (venue)** and **4 (algorithm)**. Together they include buttons, selects, sliders, text inputs, the table, file load and the figures.
- **Library:** Panel, with a modern theme. The colors stay close to the current ones: dark grey background, white text, green and red buttons, green links.
- **Logic:** the prototype reuses `pypalc.core` and as much of `pypalc/gui` as is practical.
- **Quality bar:** no tests and no polish. Just enough for the two pages to work and be shown.
- **Result:** screenshots of the current pages and the prototype pages side by side, for the user to judge.
