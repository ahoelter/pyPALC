# pyPALC on PyPI and Read the Docs — Design

Project 4 of the pyPALC modernization (after cleanup, Bokeh 3 and the Panel
redesign; before the target slope work). Branch `docs-and-pypi`.

## Intent

Two audiences use pyPALC:

* **Browser users** use the app on a server the author deploys. Their guide
  stays the in-app Help page; this project does not change it.
* **Python users** work with the code. They either write a setup file and
  run it with the command line runner, or import `pypalc.core` in their own
  scripts; some want to adapt the GUI for their own needs. For them pyPALC
  becomes a package on PyPI with documentation on Read the Docs.

Why now: the GUI is final (Panel redesign merged), so the docs describe the
app as it stays; the target slope work comes after and will change the
interface, hence a 0.x version.

## Goals

1. `pip install pypalc` installs the calculation code (`pypalc.core`) and
   the `pypalc` command (`pypalc run ...`), with only numpy, scipy and
   pyyaml as dependencies.
2. `pip install "pypalc[gui]"` adds Panel, panel-material-ui, Bokeh and the
   GUI's other dependencies; `pypalc gui` then starts the app (Panel server)
   and opens the browser. The app runs from the installed package, without
   the repository (the author's server deployment uses this).
3. In a cloned repository `panel serve pyPALC --show` (from the parent
   folder) keeps working as now.
4. Version **0.1.0**, published on PyPI by a GitHub Action when a tag
   `v0.1.0` is pushed, using PyPI Trusted Publishing (no stored token). A
   test upload to TestPyPI comes before the first real release.
5. Sphinx documentation on Read the Docs (project already created by the
   author, https://pypalc.readthedocs.io), built from the repository by a
   `.readthedocs.yaml`, covering:
   * **Introduction:** what PALC does, the publications (the same as in the
     Help page), link to the browser app (URL left as a placeholder until the
     author's server is live), license.
   * **Installation:** pip (core and `[gui]`), from source (clone, conda or
     pip requirements, `PYTHONNOUSERSITE=1` note), Python version.
   * **Command line:** `pypalc run setup.yaml --out DIR` (and
     `python -m pypalc run`), walk-through of `examples/simple.yaml`, the
     result files the runner writes (names, columns, units), error messages
     and exit codes.
   * **Setup file reference:** every key of the YAML setup file with
     meaning, unit, type, default and allowed values/range, and the
     measured-directivity options.
   * **Python tutorial:** `load_setup` / building a `Setup` in code,
     `pipeline.run`, what `RunResult` contains, `write_results`,
     `result_arrays`, `save_setup`/`dump_setup`; a short plot of a result
     with plain matplotlib or numpy is optional (builder's call, no new
     dependency of the package).
   * **The GUI:** `pypalc gui` and its options, how the GUI code is organised
     (one module per page, `PalcApp`, `style.py` for colors/theme,
     `help_text.py`, `widgets.py` with the `watch`/`on_click`/`locked`
     rule) for people adapting the app, and running the app on a server
     (`pypalc gui` / `panel serve` options for address, port and allowed
     websocket origins; a reverse proxy such as Apache in front).
   * **API reference:** automatic pages for `pypalc.core`, `pypalc.cli` and
     `pypalc.gui`, generated from the existing NumPy-style docstrings.
   * **Changelog** with the 0.1.0 entry.

6. **Examples** (author's decision 2026-11-10): example setups ship inside
   the package (e.g. `pypalc/examples/`), so pip users and browser users on
   the author's server have them.
   * `pypalc examples DIR` copies them (with any data files they need) to
     a folder for command line and Python users; the docs offer them as
     downloads and link them.
   * Page 1's card "Setup file" gets an "Example" selection next to "Load
     Setup": choosing an example loads it exactly like a setup file (same
     code path, same running look, same error display, shows page 5). The
     list is built from the files in the package's examples folder, so a
     new YAML file there appears without code changes. Its caption no
     longer mentions an "examples folder" (wording: builder's call, e.g.
     "Load a saved setup file or one of the examples.").
   * The author builds the actual examples later. This project ships the
     mechanism with `simple.yaml` as the only example (moved from
     `examples/`); adding examples must need nothing but dropping a YAML
     file (plus its data file, if any) into the folder. How an example
     gets its display name (file name or a `title`/comment in the YAML) is
     the builder's call, documented for the author.

## Non-goals

* No change to the in-app Help page or to the GUI's behaviour, except the
  example selection of goal 6 (and the Help text sentence about it).
* No change to calculation results (`tests/baseline` and
  `tests/gui_reference.json` stay as they are; never re-recorded).
* No conda-forge package, no CI test workflow beyond what publishing needs
  (may be proposed later).
* No translation; English only.
* No rewrite of docstrings beyond fixing what breaks or misleads the
  generated pages.

## Anti-goals

* Docs that drift from the code: the setup file reference and the tutorial
  code must be checked by tests (see Constraints).
* A package that installs but cannot start the GUI because files (logos,
  entry script) are missing from the wheel.
* Credentials stored in the repository or in GitHub secrets for PyPI.

## Constraints

* Python: tested on 3.12; `requires-python = ">=3.12"`.
* Dependencies in `pyproject.toml` use lower bounds, not pins: core
  `numpy`, `scipy`, `pyyaml`; extra `gui`: `panel>=1.10`,
  `panel-material-ui>=0.16.6`, `bokeh>=3.10` and whatever else the GUI
  imports directly (e.g. `colorcet`); extra `docs`: Sphinx, Furo,
  MyST-Parser; extra `test`: pytest. `requirements.txt` and
  `requirements_pip.txt` stay as the exact pins for development.
  Upper bounds are left to the builder where a known incompatibility exists
  (the GUI uses private Panel/Bokeh APIs, see README "upgrade note").
* Single source of the version: `pypalc.__version__ = "0.1.0"`, read by
  the build backend. The publish workflow fails if the tag differs from it.
* The wheel contains the logos and the app entry; `APP_DIR / 'static'` is
  replaced by a location inside the package (e.g. `pypalc/gui/static/`,
  read via `importlib.resources` or a path relative to the package). The
  repository's `main.py` stays as a thin shim so that `panel serve pyPALC`
  works from a checkout.
* `pypalc gui` serves one `PalcApp` per browser session exactly like
  `main.py` (same `pn.extension('tabulator')`, Bokeh theme and title) and
  passes through at least `--port`, `--address`,
  `--allow-websocket-origin` and `--show/--no-show` (default: show).
* Docs: Sphinx with `autodoc`, `autosummary`, `napoleon` (NumPy style),
  `viewcode`, `intersphinx` (Python, numpy); prose pages in Markdown via
  MyST-Parser; theme Furo. Read the Docs installs `.[gui,docs]` so the
  GUI modules import during the build. The build has no warnings
  (`-W` locally and `fail_on_warning: true` on Read the Docs).
* The old Sphinx sources (`docs/source/*.rst` of 2021, describing modules
  that no longer exist) and the committed build output (`docs/build/`) are
  removed; `docs/superpowers/` stays and is excluded from the Sphinx build.
* Tests keep the docs honest:
  * every field of the setup dataclasses in `pypalc.core.settings` appears
    in the setup file reference page, and every key documented there exists;
  * the tutorial's code runs (e.g. the code blocks are executed by a test or
    are doctests run in the suite);
  * every example in the package runs through `pypalc run` and loads
    through page 1's "Example" selection (so an example the author adds
    later is checked automatically);
  * the built wheel contains the static files and `pypalc gui` can build the
    app (smoke test without a browser);
  * the Sphinx build with `-W` succeeds (may be a slow/optional test).
* Publishing: GitHub Actions workflow on tag push `v*`: build sdist and
  wheel (`python -m build`), check (`twine check`), compare tag and
  `__version__`, publish with `pypa/gh-action-pypi-publish` via Trusted
  Publishing in a GitHub environment `pypi`; a manual trigger
  (`workflow_dispatch`) publishes to TestPyPI (environment `testpypi`).
* Commits never carry Co-Authored-By or "Generated with" lines. Push,
  tagging, merging and anything on pypi.org / test.pypi.org / GitHub
  settings happen only on the author's word; the author does the one-time
  Trusted Publishing setup on PyPI/TestPyPI and the GitHub environments
  following a step list in the docs or README.

## Approaches

**Docs** (author chose A):
* **A. Sphinx autodoc/autosummary importing the package**, prose in
  Markdown (MyST), Furo theme. Precise API pages (signatures, defaults,
  inherited members); needs the package and its GUI extra installed on Read
  the Docs. Chosen for precision.
* B. sphinx-autoapi (parses source without importing): more robust build,
  less precise pages.
* MkDocs dropped: the roadmap names Sphinx.

**Publishing** (author chose A):
* **A. GitHub Action on version tag with PyPI Trusted Publishing**, TestPyPI
  by manual trigger first. No stored credentials, releases by tagging.
* B. Manual `python -m build` + `twine upload` with an API token each
  release.

**Install shape** (author chose): PyPI package `pypalc`, core by default,
GUI as extra `[gui]`, launched by `pypalc gui`; the GUI's code is
documented too, for people adapting the app.

**Version** (author chose): 0.1.0 — works, interface may still change;
1.0.0 once the target slope work has settled the interface.

## Decided by the author

* Docs for Python users only; the browser guide stays the in-app Help page;
  the server deployment is for browser-only users.
* Both kinds of Python users (command line/setup file and importers), and the
  GUI code for people adapting the app.
* pip-installable from PyPI; `pypalc gui`; version 0.1.0; Docs approach A;
  publishing approach A.
* Read the Docs project is set up by the author.
* Examples in the package, `pypalc examples DIR`, and an "Example"
  selection on page 1; the author writes the examples later, the project
  delivers the mechanism with `simple.yaml`.

## Controller's calls (author may change them)

* Build backend setuptools (`pyproject.toml` only, no `setup.py`).
* Furo theme; Markdown prose via MyST.
* Sphinx sources in `docs/` (conf.py in `docs/source/` or `docs/`, builder's
  choice), `docs/superpowers/` excluded.
* A "Running the app on a server" page in the GUI section, because the
  author deploys the app and `pypalc gui` options matter there.
* Changelog page starting at 0.1.0.
* `pypalc gui` serves the app at the same path as the checkout,
  `/pyPALC` (so `http://localhost:5006/pyPALC`, as in README and
  `tools/browser_check.py`), and also offers `--prefix` for a reverse proxy
  that forwards a sub-path; the server page shows an Apache example for
  both a whole host and a sub-path, including the websocket forwarding.
* Development environment: the docs and build tools (sphinx, furo,
  myst-parser, build, twine) are installed into the existing development
  environment `pyPALC3-panel` (the one the redesign used; the author's
  `pyPALC3` is not touched). They are declared as the `docs` extra and a
  `build`/`dev` extra in `pyproject.toml`; `requirements.txt` and
  `requirements_pip.txt` keep only the runtime + pytest pins as now.
* README shortened to an overview with install commands and links to the
  docs; development details (tests, browser tools) move to a "Development"
  docs page or stay in README (builder's call, nothing lost).

## Left to the builder

* Exact page titles and order, wording, code examples.
* How the tutorial code is tested (doctest, a test that extracts the
  Markdown code blocks, or example scripts included into the page).
* Where the GUI entry lives inside the package and how `pypalc gui` calls
  Panel (`pn.serve` with a per-session function, or Panel's serve command).
* Upper bounds on dependencies where justified.
