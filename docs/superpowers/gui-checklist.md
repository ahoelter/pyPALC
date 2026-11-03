# GUI checklist

Walk this list in the browser after every change to the GUI
(`cd .. && bokeh serve --show pyPALC`). One box per behavior; the first
group comes from the anti-goals of the code cleanup design
(`docs/superpowers/specs/2026-10-30-code-cleanup-design.md`).

Status legend: `[x]` checked in the browser, `[ ]` not yet checked.
"needs browser" marks items that were only verified programmatically.

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
- [ ] Fixed first angle: "Fix Angle of Uppermost Cabinet?" enabled only
  after the venue check; the angle input enabled only for "Yes" — needs
  browser
- [ ] Discrete angles: "Use Discrete Tilt Angles?" = Yes fills and enables
  the angle input and enables "Reference Array: Use Angles of (17.)?" —
  needs browser
- [ ] Reference array type shows and hides its widgets — needs browser
- [ ] Hinge count select appears only for target slope weighting — needs
  browser
- [ ] Running indicator on the start toggle ("Running..." while the
  calculation runs) — needs browser
- [ ] Dragging the target slope points and the reference point snaps them
  to the distance grid and moves the optimization region — needs browser
- [ ] CSV download buttons work — needs browser
- [ ] The numbers equal the headless runner (`python -m pypalc run`) for
  the same setup — verified by `tests/test_gui_equivalence.py` (13 cases)
- [ ] Two browser tabs do not share state: a line drawn in one tab does not
  appear in the other, a new tab starts clean — needs browser (pinned by
  `tests/test_gui_sessions.py`)

## GUI behaviors

- [ ] Page switching (Help, 1 to 5) shows the same layouts as before —
  needs browser
- [ ] Equal-axis radio of the venue figure (variable height, constant
  height, stretched) — needs browser
- [ ] The directivity plot updates on directivity change; "Measured
  Loudspeaker Data" shows the upload — needs browser
- [ ] Reference array widgets show and hide per type (Straight,
  Progressive, Arc, User Defined) — needs browser
- [ ] Target Slope ⇄ Hard Margin mutual exclusion — needs browser
- [ ] ν slider with + / − and step size — needs browser
- [ ] Fixed angle enabled after the venue check, and its Max/Min text —
  needs browser
- [ ] Running indicator on the start toggle — needs browser
- [ ] Every download button ("Download PALC Angles", "Download SPL") —
  needs browser
- [ ] Beam plot checkboxes and frequency select — needs browser
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
  before — needs browser
- [ ] "Save Setup" clicked twice without a change downloads twice — needs
  browser
- [ ] Load an invalid file (e.g. `algorithm: {variant: palc9}`): page 4
  shows "Error: setup file invalid at algorithm.variant: ..." and nothing
  else changes; load `tests/cases/measured.yaml` (directivity as a path):
  the error names `loudspeaker.measured_directivity.path` and says to use
  Save Setup or the headless runner — needs browser
- [ ] After loading a setup with measured directivity, the directivity
  upload on page 2 shows no file name (a browser file input cannot be
  filled by the server); the directivity plot shows the loaded data —
  needs browser
- [ ] Known limitation: choosing the same file again in "Load Setup" may
  not load it again (browsers send no change for the same file) — needs
  browser

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

## Walkthrough log

| date | by | result |
|---|---|---|
| 2026-11-02 | Task 5 implementer | no browser available; headless server smoke check only (page served to two sessions, no errors in the server log) |
| 2026-11-03 | Task 7 implementer | no browser available; server smoke check (session pulled with bokeh.client, 183 models, no errors in the log); save/load checks above need a browser |
