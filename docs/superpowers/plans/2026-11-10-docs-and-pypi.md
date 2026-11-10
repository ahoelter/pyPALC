# pyPALC on PyPI and Read the Docs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make pyPALC a pip-installable package (`pypalc` 0.1.0 on PyPI, GUI as extra, `pypalc gui`, packaged examples with a page-1 "Example" selection) with Sphinx documentation on Read the Docs for Python users.

**Architecture:** `pyproject.toml` (setuptools) packages `pypalc` with its logos and examples as package data; `pypalc.cli` gains the subcommands `gui` and `examples`; a new `pypalc.examples` module owns the example files; `pypalc.gui.serve` builds and serves the app (the repository's `main.py` becomes a shim). Docs are Sphinx (autodoc/autosummary/napoleon, MyST Markdown, Furo) in `docs/source`, built by Read the Docs from `.readthedocs.yaml`; a GitHub Action publishes on tag via Trusted Publishing.

**Tech Stack:** Python 3.12, setuptools ≥ 77, Panel 1.10 / panel-material-ui 0.16.6 / Bokeh 3.10, Sphinx + Furo + MyST-Parser, GitHub Actions (`pypa/gh-action-pypi-publish`).

**Spec:** `docs/superpowers/specs/2026-11-10-docs-and-pypi-design.md`

## Global Constraints

- Repository root = `/home/arne_hoelter/university/04_pyPALC/git/pyPALC/pyPALC` (the folder is named `pyPALC`; `panel serve pyPALC --show` runs from its parent and must keep working).
- Branch `docs-and-pypi`. Plain commits, never Co-Authored-By or "Generated with" lines. No push, tag, merge, or anything on pypi.org / test.pypi.org / GitHub settings.
- Environment: conda env `pyPALC3-panel`; every command prefixed `PYTHONNOUSERSITE=1 conda run --no-capture-output -n pyPALC3-panel ...`. The env `pyPALC3` is never touched. Docs/build tools (`sphinx`, `furo`, `myst-parser`, `build`, `twine`) may be pip-installed into `pyPALC3-panel`.
- Version **0.1.0**, single source `pypalc.__version__ = "0.1.0"`.
- `requires-python = ">=3.12"`. Dependencies as lower bounds: core `numpy>=2.0`, `scipy>=1.13`, `pyyaml>=6.0`; extra `gui`: `panel>=1.10`, `panel-material-ui>=0.16.6`, `bokeh>=3.10`, plus every other third-party package `pypalc/gui` imports directly; extra `docs`: `sphinx>=8`, `furo`, `myst-parser>=4`; extra `test`: `pytest>=8`; extra `dev`: `build`, `twine`. `requirements.txt` and `requirements_pip.txt` unchanged.
- `pypalc.core`, `pypalc.cli` (at import), `pypalc.examples` never import Panel or Bokeh (`tests/test_core_has_no_bokeh.py` pattern).
- GUI callbacks are registered only via `watch()`/`on_click()`/`watch_value()` of `pypalc/gui/widgets.py` (`tests/test_locked_callbacks.py`).
- `tests/baseline/*` and `tests/gui_reference.json` are never re-recorded; calculation results do not change. An unavoidable widget difference on page 1 gets a narrowly scoped `INTENDED` entry in `tests/test_gui_reference.py` with a comment.
- `pypalc gui` serves the app at route `/pyPALC`; options `--port` (default 5006), `--address`, `--allow-websocket-origin` (repeatable), `--prefix`, `--show/--no-show` (default show).
- Docs: Sphinx build with `-W` has no warnings; theme Furo; prose in Markdown (MyST); `docs/superpowers/` is not part of the Sphinx sources.
- Full suite: `PYTHONNOUSERSITE=1 conda run --no-capture-output -n pyPALC3-panel python -m pytest -q` (≈6 min; currently 353 passed, 14 known RuntimeWarnings).

## Review Focus

1. A pip-installed GUI without the repository: missing logos or examples in the wheel crash `pypalc gui` at the first session → Task 1/2 wheel-content test, Task 3 view built from package paths only.
2. `pypalc run` / `pypalc examples` on a core-only install (no Panel): must work, and `pypalc gui` must print "install pypalc[gui]" and exit non-zero instead of a traceback → Task 3 test with Panel import blocked.
3. An example that uses a measured directivity data file next to it: the GUI rejects path directivities, so loading it through page 1 must embed the file → Task 2 `load_example` test with a temporary examples folder, Task 4 GUI test.
4. Behind Apache with a sub-path: `--prefix` and websocket origins reach Panel's server → Task 3 test of the arguments handed to `pn.serve`.
5. Docs drift: a setup key added later without docs, or a tutorial that stops running → Task 6 reference-coverage test and tutorial-script test.

---

### Task 1: Installable package with version, logos inside the package and the `pypalc` command

**Files:**
- Create: `pyproject.toml`, `tests/test_packaging.py`
- Modify: `pypalc/__init__.py` (add `__version__`), `pypalc/gui/__init__.py` (replace `APP_DIR` with `STATIC_DIR`), `pypalc/gui/app.py:59,188-189`, `.gitignore` (`build/`, `dist/`, `*.egg-info/`, `docs/_build/`), `README.md` structure table (`static/` row → `pypalc/gui/static/`)
- Move: `static/*` → `pypalc/gui/static/` (`git mv`)

**Interfaces:**
- Produces: `pypalc.__version__: str == "0.1.0"`; `pypalc.gui.STATIC_DIR: pathlib.Path` (the package's `static` folder); console script `pypalc = "pypalc.cli:main"`; extras `gui`, `docs`, `test`, `dev`.

- [ ] **Step 1: Write failing tests** in `tests/test_packaging.py`:
  - `test_version`: `pypalc.__version__ == "0.1.0"` and `tomllib`-read `pyproject.toml` has `dynamic = ["version"]` with `[tool.setuptools.dynamic] version = {attr = "pypalc.__version__"}`.
  - `test_console_script`: `[project.scripts]["pypalc"] == "pypalc.cli:main"`.
  - `test_core_dependencies`: `[project].dependencies` names (lower-cased, before any version specifier) == `{"numpy", "scipy", "pyyaml"}`; `optional-dependencies["gui"]` contains `panel`, `panel-material-ui`, `bokeh`.
  - `test_static_inside_package`: `STATIC_DIR == Path(pypalc.gui.__file__).parent / "static"` and every PNG the GUI loads (grep the names used with the data-URI helper in `app.py`) exists there.
  - `test_wheel_contents` (skips with a reason if `pip wheel` or setuptools ≥ 77 is unavailable): run `python -m pip wheel . --no-deps --no-build-isolation -w <tmp>`; the wheel's namelist contains `pypalc/gui/static/<each png>`, `pypalc/cli.py`, and no `tests/`, `tools/`, `docs/` or `main.py` entries.
- [ ] **Step 2: Run** `... python -m pytest tests/test_packaging.py -q` → FAIL.
- [ ] **Step 3: Implement.** `pyproject.toml`: `[build-system] requires = ["setuptools>=77"]`, `build-backend = "setuptools.build_meta"`; `[project]` name `pypalc`, description, `readme = "README.md"`, `license = "MIT"`, `license-files = ["LICENSE"]`, authors (Arne Hölter, Florian Straube), `requires-python`, classifiers, `[project.urls]` (Homepage = GitHub, Documentation = `https://pypalc.readthedocs.io`); `[tool.setuptools.packages.find] include = ["pypalc*"]`; `[tool.setuptools.package-data]` `"pypalc.gui" = ["static/*.png"]` (plus other static file types present). `git mv static pypalc/gui/static`; `STATIC_DIR` replaces `APP_DIR` everywhere.
- [ ] **Step 4: Run** the new tests and `tests/test_shell.py` → PASS; start `panel serve pyPALC` from the parent folder briefly (or a session test) to confirm the logos still load.
- [ ] **Step 5: Commit** "Make pyPALC an installable package (0.1.0) with the logos inside the package".

### Task 2: Examples inside the package and `pypalc examples DIR`

**Files:**
- Create: `pypalc/examples/__init__.py`, `tests/test_examples.py`
- Move: `examples/simple.yaml` → `pypalc/examples/simple.yaml` (`git mv`; remove the empty `examples/`)
- Modify: `pyproject.toml` package-data (`"pypalc.examples" = ["*.yaml", "*.yml", "*.csv"]`), `pypalc/cli.py` (subcommand `examples`), `README.md` (paths `examples/simple.yaml` → `pypalc/examples/simple.yaml` / `pypalc examples DIR`), `tests/test_packaging.py::test_wheel_contents` (also `pypalc/examples/simple.yaml`), `tests/test_cli.py`

**Interfaces:**
- Produces in `pypalc.examples`:
  - `EXAMPLES_DIR: pathlib.Path`
  - `@dataclass(frozen=True) class Example: name: str; title: str; path: pathlib.Path`
  - `examples(directory: Path = EXAMPLES_DIR) -> list[Example]` — every `*.yaml`/`*.yml` file, sorted by title; `name` = file stem; `title` = stem with `_` → space, first letter upper-case (`gap_soft_margin` → `Gap soft margin`).
  - `load_example(name: str, directory: Path = EXAMPLES_DIR) -> Setup` — `load_setup` of the file; a relative `measured_directivity.path` is replaced by `MeasuredDirectivity(csv=<file contents>)` so the GUI can apply it. Unknown name → `SetupError("example", "unknown example '<name>'")`.
  - `copy_examples(destination: Path, directory: Path = EXAMPLES_DIR) -> list[Path]` — copies every file of the folder (YAML and data files, not `__init__.py`/`__pycache__`), creating `destination`; refuses to overwrite an existing file (`FileExistsError`), returns the written paths.
- CLI: `pypalc examples DIR` prints the written paths, exit 0; existing files → message on stderr, exit 5 (`EXIT_OUTPUT`).

- [ ] **Step 1: Write failing tests** in `tests/test_examples.py`:
  - `test_simple_is_listed`: `[e.name for e in examples()] == ["simple"]`, title `"Simple"`.
  - `test_every_example_runs` parametrized over `examples()`: `pipeline.run(load_example(e.name))` returns a result with `len(tilt angles) == cabinet_count` (use the `RunResult` field names).
  - `test_measured_example_embeds_csv`: copy `tests/cases/measured.yaml` and `tests/cases/measured.csv` into `tmp_path`; `load_example("measured", tmp_path).loudspeaker.measured_directivity` has `path is None` and `csv == (tmp_path/"measured.csv").read_text()`.
  - `test_unknown_example`: raises `SetupError` with key `"example"`.
  - `test_copy_examples` (new dir gets `simple.yaml`; second call raises `FileExistsError`).
  - in `tests/test_cli.py`: `main(["examples", str(tmp_path/"ex")])` → 0 and file exists; again → 5.
  - `tests/test_core_has_no_bokeh.py`: extend to `pypalc.examples`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** module, package data and CLI subcommand.
- [ ] **Step 4: Run** `tests/test_examples.py tests/test_cli.py tests/test_packaging.py tests/test_core_has_no_bokeh.py` → PASS.
- [ ] **Step 5: Commit** "Ship the example setups inside the package and copy them with 'pypalc examples'".

### Task 3: `pypalc gui` and the app entry inside the package

**Files:**
- Create: `pypalc/gui/serve.py`, `tests/test_gui_command.py`
- Modify: `main.py` (thin shim), `pypalc/cli.py` (subcommand `gui`, lazy import), `README.md` (start command: `pypalc gui`; checkout: `panel serve pyPALC --show`)

**Interfaces:**
- Produces in `pypalc.gui.serve`:
  - `ROUTE = "pyPALC"`, `TITLE = "pyPALC"`
  - `create_view() -> panel.viewable.Viewable` — exactly what `main.py` does today per session: `PalcApp()`, `app.doc.theme = bokeh_figure_theme()`, returns `app.view()`. `pn.extension('tabulator')` is called once at module import (same reason as the comment in today's `main.py`).
  - `serve(port: int = 5006, address: str | None = None, allow_websocket_origin: list[str] | None = None, prefix: str | None = None, show: bool = True) -> None` — `pn.serve({ROUTE: create_view}, port=..., address=..., websocket_origin=..., prefix=..., show=..., title=TITLE)`, passing `None` options as Panel's defaults.
- `main.py`: keeps the `sys.path` insert, then `from pypalc.gui.serve import create_view` and `create_view().servable(title="pyPALC")`.
- CLI `pypalc gui [--port N] [--address A] [--allow-websocket-origin H]... [--prefix P] [--show | --no-show]`: imports `pypalc.gui.serve` only inside the command; `ImportError` of panel/bokeh/panel_material_ui → stderr `pypalc gui needs the GUI dependencies: pip install "pypalc[gui]"`, exit 6 (new `EXIT_GUI = 6`, documented in `main`'s docstring).

- [ ] **Step 1: Write failing tests** in `tests/test_gui_command.py`:
  - `test_create_view_builds_session`: inside a `bokeh.document.Document` context (pattern of `tests/test_shell.py`), `create_view()` returns a viewable whose rendered model tree contains the page-1 widgets.
  - `test_serve_arguments`: monkeypatch `panel.serve` with a recorder; `cli.main(["gui", "--port", "5010", "--address", "0.0.0.0", "--allow-websocket-origin", "a.org", "--allow-websocket-origin", "b.org", "--prefix", "pypalc", "--no-show"])` → recorder got `{"pyPALC": create_view}`, `port=5010`, `address="0.0.0.0"`, `websocket_origin=["a.org", "b.org"]`, `prefix="pypalc"`, `show=False`; default call → `port=5006`, `show=True`.
  - `test_gui_without_panel`: with `sys.modules` entries for `panel`, `pypalc.gui.serve` set to `None` (monkeypatch), `main(["gui"])` returns 6 and stderr contains `pip install "pypalc[gui]"`.
  - `test_cli_import_has_no_panel`: in a subprocess, `import pypalc.cli` leaves `panel` and `bokeh` out of `sys.modules`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** `serve.py`, shim, CLI.
- [ ] **Step 4: Run** the new tests, `tests/test_cli.py`, `tests/test_shell.py` → PASS. Start `pypalc gui --no-show --port 5011` via `python -m pypalc gui ...` in the background, open `http://localhost:5011/pyPALC` with `tools/browser_check.py --url ... --out <scratch>` (checks all ok), stop it; same for `panel serve pyPALC` from the parent folder.
- [ ] **Step 5: Commit** "Add 'pypalc gui', serving the app from the package".

### Task 4: "Example" selection on page 1

**Files:**
- Modify: `pypalc/gui/page_venue.py` (card "Setup file": select + button, caption), `pypalc/gui/page_algorithm.py` (`load_example`, watcher registration next to the two `load_setup_input` watchers ~l.352-360), `pypalc/gui/help_text.py` (sentence at ~l.40 and ~l.210), `docs/superpowers/gui-checklist.md`, `tools/browser_check.py` (check `load_example`), `README.md` browser-tools list
- Create: `tests/test_load_example.py`

**Interfaces:**
- Consumes: `pypalc.examples.examples()`, `load_example(name)`; `AlgorithmPage.run_later`, the error handling of `AlgorithmPage.load_setup`.
- Produces: `VenuePage.example_select: pmui.Select` (label `Example`, options `{example.title: example.name}`, value = first example), `VenuePage.load_example_button` (`button('Load Example', ...)`, same look as other page-1 buttons that are not step buttons — builder's call within `widgets.py` helpers); `AlgorithmPage.load_example(name: str) -> None` — with the running look via `run_later`, ignored while a calculation runs, applies `load_example(name)` with `apply_setup` (shows page 5), errors shown in `app.config_error_info` with the same texts as `load_setup` (`Error: setup file invalid at <key>: <msg>` / `Error: Could not load the setup file`). Refactor `load_setup` so both share one apply-and-report helper; behaviour of file loading unchanged.
- Caption of the card: `Load a saved setup file or one of the examples. It restores all pages, runs the calculation and shows 5 · Sound Field.`
- The select and button are disabled when there is no example (empty folder) — not expected in practice, but no crash.

- [ ] **Step 1: Write failing tests** in `tests/test_load_example.py` (helpers as in `tests/test_load_setup_page1.py`):
  - `test_card_has_example_select_and_button` (options == titles of `examples()`).
  - `test_every_example_loads_on_page_1` parametrized over `examples()`: click → page 5 shown, `angles_csv(app.pages[4])` equals the angles after uploading the same file through page 4's input.
  - `test_example_with_data_file_loads`: monkeypatch `pypalc.examples.EXAMPLES_DIR` (and whatever the page reads) to a `tmp_path` with `measured.yaml` + `measured.csv` from `tests/cases`; select "Measured", click → no error, directivity is measured.
  - `test_example_ignored_while_running` (as the file-input test).
  - `test_example_error_shown_on_page_1`: monkeypatch `load_example` to raise `SetupError("x", "bad")` → `config_error_info.object == "Error: setup file invalid at x: bad"`, page 1 still shown.
  - Existing `tests/test_load_setup_page1.py`, `tests/test_venue_page.py`, `tests/test_gui_reference.py`, `tests/test_locked_callbacks.py` still pass (update card-content assertions in `test_venue_page.py` to include the new widgets).
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement**; update help text, checklist, `browser_check.py` (`load_example`: select the first example on page 1, click "Load Example", page 5 shows its figures, no alert).
- [ ] **Step 4: Run** full suite → all pass; `browser_check.py` at 1280 (`--two-tabs --server-log`), 1920 and 1366×768 against `pypalc gui --no-show` → all checks ok, 0 server errors; screenshots of page 1 to the scratchpad.
- [ ] **Step 5: Commit** "Add an example selection to page 1".

### Task 5: Sphinx scaffold, API reference and Read the Docs config

**Files:**
- Delete: `docs/Makefile`, `docs/make.bat`, `docs/html_make`, `docs/build/`, all of `docs/source/` (old 2021 sources)
- Create: `docs/source/conf.py`, `docs/source/index.md`, `docs/source/api/index.md` (+ autosummary stubs or `autosummary_generate = True` with a recursive template), `docs/source/_templates/` (if needed), `.readthedocs.yaml`, `tests/test_docs_build.py`
- Modify: docstrings in `pypalc/` only where the `-W` build fails (no other docstring rewrites)

**Interfaces:**
- `conf.py`: project `pyPALC`, `release`/`version` from `pypalc.__version__` (import with `sys.path` to repo root), author "Arne Hölter, Florian Straube", `html_theme = "furo"`, extensions `sphinx.ext.autodoc`, `sphinx.ext.autosummary`, `sphinx.ext.napoleon` (NumPy style), `sphinx.ext.viewcode`, `sphinx.ext.intersphinx` (python, numpy), `myst_parser`; `autosummary_generate = True`; `exclude_patterns` keeps `docs/superpowers` out (it is outside `docs/source`, confirm nothing includes it).
- API pages: `pypalc.core` (all submodules), `pypalc.cli`, `pypalc.examples`, `pypalc.gui` (all submodules), recursive.
- `.readthedocs.yaml`: `version: 2`; `build.os: ubuntu-24.04`; `build.tools.python: "3.12"`; `python.install: [{method: pip, path: ., extra_requirements: [gui, docs]}]`; `sphinx.configuration: docs/source/conf.py`; `sphinx.fail_on_warning: true`.
- `index.md`: title, one-paragraph intro, toctree with `api/index` (Task 6 adds the other pages).

- [ ] **Step 1: Write failing test** `tests/test_docs_build.py::test_sphinx_build_without_warnings`: `pytest.importorskip("sphinx")`, `importorskip("furo")`, `importorskip("myst_parser")`; run `python -m sphinx -W --keep-going -b html docs/source <tmp>`; return code 0; `<tmp>/api/` contains a page for `pypalc.core.pipeline` and `pypalc.gui.app`. Also `test_readthedocs_config`: YAML-load `.readthedocs.yaml`, assert the values above.
- [ ] **Step 2:** `pip install sphinx furo myst-parser build twine` into `pyPALC3-panel`; run → FAIL.
- [ ] **Step 3: Implement**; fix docstring markup that makes `-W` fail (record each fix in the report).
- [ ] **Step 4: Run** the docs tests → PASS; open the built `index.html` and an API page (screenshot via headless Firefox to the scratchpad) to confirm Furo renders.
- [ ] **Step 5: Commit** "Rebuild the Sphinx docs with an API reference and a Read the Docs config".

### Task 6: Documentation pages

**Files:**
- Create in `docs/source/`: `introduction.md`, `installation.md`, `command-line.md`, `setup-file.md`, `tutorial.md`, `tutorial/run_setup.py`, `tutorial/build_setup.py` (scripts included with `literalinclude`), `gui.md`, `server.md`, `development.md`, `changelog.md`; `tests/test_docs_content.py`
- Modify: `docs/source/index.md` (toctree order: introduction, installation, command-line, setup-file, tutorial, gui, server, api/index, development, changelog), `README.md` (short overview: what pyPALC is, `pip install pypalc` / `"pypalc[gui]"`, `pypalc gui`, `pypalc run`, link to `https://pypalc.readthedocs.io`, license; the development details — tests, browser tools, structure — move to `development.md`, nothing dropped)

**Content requirements (wording is the builder's):**
- `introduction.md`: what PALC does; the publications as listed in `pypalc/gui/help_text.py`; "Browser app: <URL follows>" placeholder; MIT license; authors.
- `installation.md`: `pip install pypalc`, `pip install "pypalc[gui]"`, from source (clone, `requirements.txt` conda / `requirements_pip.txt` pip, `pip install -e .`), Python ≥ 3.12, the `PYTHONNOUSERSITE=1` note.
- `command-line.md`: `pypalc run SETUP --out DIR [--traceback]` and `python -m pypalc run`; `pypalc examples DIR`; `pypalc gui` (pointer to gui.md); walk-through of `simple.yaml` (included via `literalinclude` from `pypalc/examples/simple.yaml`); every file `write_results` writes with its columns and units (from `pypalc/core/export.py`); exit codes 0/2/3/4/5/6 with meaning.
- `setup-file.md`: one entry per key, written as the dotted key in backticks (e.g. `` `loudspeaker.aperture_angle_deg` ``; list items as `` `venue.audience_lines[].start` ``), with type, unit, default, allowed values/range (from `pypalc/core/settings.py`: `validate`, `_DIRECTIVITIES`, `_REFERENCE_TYPES`, `_VARIANTS`, `_GAP_HANDLING`, `_WEIGHTINGS`, `FREQUENCY_RANGE_HZ`), `version`, measured directivity (`path` relative to the setup file vs embedded `csv`, the CSV format, GUI only accepts `csv`).
- `tutorial.md`: run `simple.yaml` from Python (`load_example` / `load_setup`, `pipeline.run`, the `RunResult` fields, `result_arrays`, `write_results`); build a `Setup` in code with the dataclasses and `save_setup`/`dump_setup`. Code lives in the two scripts and is included.
- `gui.md`: `pypalc gui` and its options, `panel serve pyPALC --show` in a checkout, how the GUI code is organised (one module per page, `PalcApp`, `style.py`, `help_text.py`, `widgets.py` with the `watch`/`on_click`/`locked` rule and why), how to add an example (drop a YAML file, plus its data file, into `pypalc/examples/`; title from the file name).
- `server.md`: `pypalc gui --address 0.0.0.0 --port 5006 --allow-websocket-origin host --no-show`; Apache reverse proxy (`mod_proxy`, `mod_proxy_http`, `mod_proxy_wstunnel`) for a whole host and for a sub-path with `--prefix`, including the websocket rule; a systemd unit example; the note that Panel's `--allow-websocket-origin` must name the public host.
- `development.md`: tests, browser tools, structure table (from README), building the docs (`python -m sphinx -W -b html docs/source docs/_build/html`), release steps (Task 7 fills in).
- `changelog.md`: `0.1.0` — first release on PyPI: Panel redesign of the GUI, headless runner, setup files, `pypalc gui`, examples.

- [ ] **Step 1: Write failing tests** `tests/test_docs_content.py`:
  - `test_every_setup_key_documented`: walk the `Setup` dataclass tree (`dataclasses.fields`, recursing into dataclass types, `Optional[...]` and `Tuple[Dataclass, ...]` → `[]`) to the set of dotted keys (plus `version`); each appears as `` `key` `` in `setup-file.md`.
  - `test_no_unknown_setup_key_documented`: every backticked token in `setup-file.md` that starts with one of the top-level keys (`venue.`, `loudspeaker.`, `array.`, `reference_array.`, `algorithm.`, `target_slope.`) is in that set.
  - `test_tutorial_scripts_run` parametrized over `docs/source/tutorial/*.py`: run with `runpy.run_path` in `tmp_path` as cwd; no exception.
  - `test_exit_codes_documented`: each `EXIT_*` value of `pypalc.cli` appears in `command-line.md`.
  - `test_result_files_documented`: run `simple` through `write_results` into `tmp_path`; every written file name appears in `command-line.md`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Write the pages**, README.
- [ ] **Step 4: Run** `tests/test_docs_content.py tests/test_docs_build.py` → PASS; screenshot two pages.
- [ ] **Step 5: Commit** "Write the documentation pages and shorten the README".

### Task 7: Publishing workflow and release steps

**Files:**
- Create: `.github/workflows/publish.yml`, `tools/check_version_tag.py`, `tests/test_release.py`
- Modify: `docs/source/development.md` (section "Releasing"), `docs/source/changelog.md` if needed

**Interfaces:**
- `tools/check_version_tag.py TAG`: exit 0 if `TAG == "v" + pypalc.__version__`, else prints both and exits 1 (reads the version without importing numpy-dependent modules — `pypalc/__init__.py` must stay import-light).
- Workflow: triggers `push: tags: ["v*"]` and `workflow_dispatch`. Job `build` (ubuntu-latest, Python 3.12): checkout, `pip install build twine`, on tag run `python tools/check_version_tag.py "$GITHUB_REF_NAME"`, `python -m build`, `twine check dist/*`, upload `dist/` artifact. Job `publish-testpypi` (`if: github.event_name == 'workflow_dispatch'`, `environment: testpypi`, `permissions: id-token: write`): download artifact, `pypa/gh-action-pypi-publish@release/v1` with `repository-url: https://test.pypi.org/legacy/`. Job `publish-pypi` (`if: startsWith(github.ref, 'refs/tags/v')`, `environment: pypi`, `permissions: id-token: write`): same action without repository-url. No secrets referenced.
- "Releasing" section: one-time steps for the author — on pypi.org and test.pypi.org add a *pending publisher* (owner `ahoelter`, repository `pyPALC`, workflow `publish.yml`, environment `pypi` / `testpypi`); in GitHub repository settings create environments `pypi` and `testpypi`. Per release: update `__version__` and changelog, run tests, merge, run the workflow manually for TestPyPI, check `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ pypalc`, then `git tag v0.1.0 && git push origin v0.1.0`.

- [ ] **Step 1: Write failing tests** `tests/test_release.py`: `check_version_tag.py v0.1.0` → 0; `v0.1.1` → 1; YAML-load the workflow: triggers as above, both publish jobs have `permissions: {id-token: write}` and the right `environment`, uses `pypa/gh-action-pypi-publish`, the file contains no `secrets.`; `python -m build` + `twine check` succeed into `tmp_path` (skip with reason if `build`/`twine` missing; `--no-isolation` allowed when offline) and the sdist contains `pypalc/examples/simple.yaml` and `pypalc/gui/static/`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** full suite → all pass (report count); Sphinx `-W` build passes.
- [ ] **Step 5: Commit** "Add the PyPI publishing workflow and the release steps".
