# GUI checklist

Walk this list in the browser after every change to the GUI
(`cd .. && panel serve --show pyPALC`). One box per behavior; the first
group comes from the anti-goals of the code cleanup design
(`docs/superpowers/specs/2026-10-30-code-cleanup-design.md`).

Status legend: `[x]` checked in the browser, `[ ]` not yet checked.
"needs browser" marks items that were only verified programmatically.
"browser_check: `name`" marks items that `tools/browser_check.py` checks
automatically in a headless Firefox (check `name` of its `report.json`;
see "Automated browser check" below); "partly" says what is left for a
walk by hand.

## Known before the split (pre-existing, from reading the code)

These were not walked in a browser; they are noted from the legacy
`main.py` and kept unchanged on purpose:

- The legend of the venue figure is never shown: it is created hidden and
  every venue check hides it again.
- An unreadable discretization shows "Use default: 5 points per m" but uses
  1 point per m.
- Every "PALC Run" adds another glyph renderer to the venue points figure
  and to both frequency response figures (the old ones stay).
- After drawing or removing a line following a venue check, the start
  toggle is disabled, but selecting "Target Slope" still calculates with the
  old venue and the new lines.
- "Download SPL" starts five downloads with the same file name
  `PALC_result.csv` (fixed in Task 8).

## Behavior changes of the split (Task 5)

- Saving a line that lies on an already drawn line now shows "Error: The
  audience line lies on an already drawn audience line" (before: no
  message, nothing happened).
- A line that would exceed 2000 venue points shows "Error: Maximum
  discretized points are 2000. Please reduce the discretization." (before:
  "Couldnt append line. Maximum discretized points are 2000. ...").
- An invalid aperture angle resets the input to 3.0 (before: 0.0524, read
  as degree) — bug fix commit.

## Anti-goals

- [ ] Tap a line, then "Add Line before Selected Line" inserts a line in
  front of it — needs browser
- [ ] Tap a line, then "Remove Selected Audience Line" deletes it — needs
  browser
- [ ] Weighting vs. gap handling: hard margin removes "Target Slope" from
  the weightings, target slope removes "Hard Margin" from the gap handling
  options; the soft margin strength is enabled only for soft margin — needs
  browser
- [ ] Fixed first angle: "24. Fix angle of uppermost cabinet?" enabled only
  after the venue check; the angle input enabled only for "Yes" — needs
  browser
- [ ] Discrete angles: "16. Use discrete tilt angles?" = Yes fills and
  enables the angle input and enables "Use angles of 17.?" (reference
  array) — needs browser
- [ ] Reference array type shows and hides its widgets — needs browser
- [ ] Hinge count select appears only for target slope weighting — needs
  browser
- [ ] Running indicator on the start toggle (dark red, "Running..." while
  the calculation runs and while a setup loads) — browser_check:
  `running_indicator`
- [ ] Dragging the target slope points and the reference point snaps them
  to the distance grid and moves the optimization region — partly
  browser_check: `drag` (a target point; the reference point needs
  browser)
- [ ] Download buttons work — browser_check: `download_angles`,
  `download_results`, `download_setup`
- [ ] The numbers equal the headless runner (`python -m pypalc run`) for
  the same setup — verified by `tests/test_gui_equivalence.py` (13 cases)
- [ ] Two browser tabs do not share state: a line drawn in one tab does not
  appear in the other, a new tab starts clean — browser_check:
  `two_tabs` (run with `--two-tabs`; also pinned by
  `tests/test_gui_sessions.py`)

## GUI behaviors

- [ ] Page switching (Help, 1 to 5) shows the same layouts as before —
  partly browser_check: `screenshots_fresh`, `screenshots_loaded` take
  the screenshots; comparing them with those of the former Bokeh GUI
  (`tools/compare_screenshots.py`) is by eye
- [ ] Axis modes of the venue figure ("Equal, variable height", "Equal,
  fixed height", "Stretched") — browser_check: `equal_axes` (equal
  pixels per meter in the equal modes on pages 1 to 4, also after an
  input changed while the other tab of pages 2 and 4 was shown, the
  figure inside the window, a screenshot in each mode), `axis_labels` (labels kept
  after a mode change and after inputs on pages 2 and 3)
- [ ] The directivity plot updates on directivity change; "Measured
  Loudspeaker Data" shows the upload — needs browser
- [ ] Reference array widgets show and hide per type (Straight,
  Progressive, Arc, User Defined) — needs browser
- [ ] Target Slope ⇄ Hard Margin mutual exclusion — needs browser
- [ ] ν slider with + / − and step size — needs browser
- [ ] Fixed angle enabled after the venue check, and its Max/Min text —
  needs browser
- [ ] Running indicator on the start toggle — browser_check:
  `running_indicator`
- [ ] Every download button ("Download PALC Angles", "Download Results",
  "Save Setup") — browser_check: `download_angles`, `download_results`,
  `download_setup`
- [ ] Beam plot buttons and frequency select — partly browser_check:
  `equal_axes` (both beam plots shown, equal scaling), `beam_link` (a
  zoom and a reset move both plots); the frequency select needs browser
- [ ] SFP venue tap highlights the responses of the tapped points — needs
  browser
- [ ] Legend visibility (venue legend hidden; legends of directivity, SPL
  over distance and homogeneity shown) — needs browser

## Save and load setup (Task 7)

- [ ] Set up the `target_slope` case by hand (two lines (2,0)-(10,1) and
  (10,1)-(20,3) at 5 points per m, check venue, weighting "Target
  Slope", 1 hinge, drag the points), run, click "Save Setup": the browser
  downloads `pypalc_setup.yaml`. Reload the page and load the file with
  "Load Setup": the session lands on page 5 with the same tilt angles as
  before — partly browser_check: `load` (loads
  `tests/cases/gap_soft_margin_target_slope.yaml` through the file input:
  lands on page 5, page 4 shows its gap handling, weighting and hinges)
  and `download_setup` ("Save Setup" downloads `pypalc_setup.yaml`, a
  valid setup with the loaded weighting, gap handling, lines and target
  points); setting up by hand and comparing the tilt angles need browser
- [ ] "Save Setup" clicked twice without a change downloads twice — needs
  browser (pinned on the server by `test_download_twice_sends_twice`)
- [ ] Load an invalid file (e.g. `algorithm: {variant: palc9}`): page 4
  shows "Error: setup file invalid at algorithm.variant: ..." and nothing
  else changes; load `tests/cases/measured.yaml` (directivity as a path):
  the error names `loudspeaker.measured_directivity.path` and says to use
  Save Setup or the headless runner — needs browser
- [ ] After loading a setup with measured directivity, the directivity
  upload on page 2 shows no file name (a browser file input cannot be
  filled by the server); the directivity plot shows the loaded data —
  needs browser
- [ ] Choosing the same file again in "Load Setup" loads it again — needs
  browser (pinned on the server by
  `test_load_same_file_again_loads_again`)

## Re-check after the rename (Task 8)

Task 8 renamed the code and the columns of the ColumnDataSources that the
JavaScript and the glyphs use. The tests drive the GUI on the server; the
browser side of these columns is only seen in a browser.

- [ ] Walk the whole checklist above again after the rename — needs browser
- [ ] Every plot still draws: venue lines, non-audience and SFP-only lines,
  cabinets, rays and hit points (pages 1 to 4), directivity (page 2), SPL
  over distance with target slope and reference point (page 4), frequency
  responses, venue points, homogeneity, bar charts and beam plots
  (page 5) — partly browser_check: `screenshots_fresh`,
  `screenshots_loaded` (looked at by eye)
- [ ] Hover tooltips show values (SPL over distance "(x,y) of Venue Slice",
  venue points "(x,y)", beam plots "SPL in dB") — needs browser
- [ ] Dragging the target points and the reference point still snaps them
  to the SPL over distance — partly browser_check: `drag` (a target
  point; the reference point needs browser)
- [ ] "Download PALC Angles" saves `tilt_angles.csv` with the header
  `cabinet,tilt_angle_deg,inter_cabinet_angle_deg` — browser_check:
  `download_angles` (fails on Bokeh 2: Bokeh 2.2 sorts the columns of the
  download alphabetically; the Bokeh 3 upgrade fixes that)
- [ ] "Download Results" (formerly "Download SPL") saves
  `pypalc_results.zip` with five files: `spl_over_distance.csv`,
  `target_slope.csv`, `target_reference.csv`,
  `reference_spl_over_distance.csv` and `homogeneity.csv`, each with the
  headers listed in `README.md` ("CSV header changes") — browser_check:
  `download_results` (the files are written on the server since the Panel
  redesign, so the column order no longer depends on the browser)

## Programmatic evidence (Task 5)

Besides the tests, the split was compared with the legacy `main.py`
(`git show 702f6a6:main.py`) by driving both through the same 106
interaction steps (every page, invalid inputs, line tap / insert / delete,
venue check, directivity upload, discrete and reference angles, gap
handling, weighting, ν buttons, fixed angle, runs with every weighting,
beam plots, venue point tap, target drag, clearing, re-check) and comparing
after every step all widget properties, source data, figure ranges and
renderer counts, the calculation objects and the layout tree of the
document. The only differences were the two error texts listed above.

## Automated browser check (Panel GUI)

`tools/browser_check.py` (usage in `README.md`, "Browser tools") checks a
running server in a headless Firefox window (`--width`, default 1280 x
800). Its checks, in this order:

| check | what it does |
|---|---|
| `screenshots_fresh` | screenshot of every page of a new session (the whole page) |
| `load` | loads a setup through the "Load Setup" file input (default `tests/cases/gap_soft_margin_target_slope.yaml`); the app must switch to page 5, and page 4 must show the setup's gap handling, weighting and hinges and "26. PALC Run" ready again |
| `screenshots_loaded` | screenshot of every page after loading |
| `download_angles` | "Download PALC Angles" saves `tilt_angles.csv` once, with the header of the headless export and a row per cabinet |
| `download_results` | "Download Results" saves `pypalc_results.zip` with the five files of `README.md` ("CSV header changes"), each with its header and at least one row |
| `download_setup` | "Save Setup" saves `pypalc_setup.yaml` once (not a second copy `pypalc_setup(1).yaml`), a valid setup with the loaded weighting, gap handling, number of lines and of target points |
| `running_indicator` | "PALC Run" is dark red (`rgb(139, 0, 0)`) and reads "Running..." while the calculation runs and while the setup of `load` loaded, and is green and ready afterwards |
| `drag` | moves a target point as the PointDrawTool does; the server snaps it to the distance grid and moves the optimization region |
| `rays_cleared` | page 4 shows the rays of the result, page 1 afterwards shows none |
| `equal_axes` | the venue figure in "Equal, variable height" and "Equal, fixed height" on pages 1 to 4 and page 1 again, on pages 2 and 4 also after a round trip to the other tab during which an input there ("9. Aperture angle", "23. Tolerance") gave the figure new ranges, and both beam plots on page 5 (again after page 1): pixels per meter on x and y differ by less than 1 %; in "Equal, variable height" the figure ends inside the window; screenshots of page 1 in all three axis modes |
| `beam_link` | a box zoom (mouse drag) in the PALC beam plot zooms the reference beam plot too, the reset tool of the reference plot resets both |
| `toolbars` | no toolbar hides tools in an overflow menu, in every screenshot and in every tab of pages 4 and 5 |
| `axis_labels` | the venue figure keeps its tick labels and axis titles (space left of and below its frame) after axis mode changes and, in "Equal, fixed height", after an input on pages 2 and 3 |
| `no_horizontal_overflow` | no page scrolls horizontally at the window width (document and the Material main area, which scrolls by itself) |
| `two_tabs` | with `--two-tabs`: a line drawn in one window (venue inputs and "Save Current Audience Line") does not appear in a second window, and a new window starts without lines |

## Walkthrough log

| date | by | result |
|---|---|---|
| 2026-11-02 | Task 5 implementer | no browser available; headless server smoke check only (page served to two sessions, no errors in the server log) |
| 2026-11-03 | Task 7 implementer | no browser available; server smoke check (session pulled with bokeh.client, 183 models, no errors in the log); save/load checks above need a browser |
| 2026-11-03 | Task 8 implementer | no browser available; server smoke check only (see the Task 8 report); the re-check after the rename needs a browser |
| 2026-11-04 | Bokeh 3 upgrade, Task 6 | `tools/browser_check.py --two-tabs` in headless Firefox (automated, not a walk by hand): `bokeh3` (Bokeh 3.10) 10 of 10 checks ok; `master` (Bokeh 2.2) 9 of 10, `download_angles` fails there because Bokeh 2.2 sorts the columns of the download |
| 2026-11-05 | Panel redesign, Task 7 | `tools/browser_check.py` for the Panel GUI in headless Firefox (automated): `--width 1280 --two-tabs` 15 of 15 checks ok, `--width 1920` 14 of 14 ok |
