# The GUI

The GUI is a [Panel](https://panel.holoviz.org) app with
[panel-material-ui](https://panel-material-ui.holoviz.org) widgets and
[Bokeh](https://bokeh.org) figures. It runs on a Panel server, and every
browser window (session) gets its own copy of the app. How to use it is
explained on its Help page; this page is about starting it and about its
code, for people who adapt the app.

## Start it

With pyPALC installed (`pip install pypalc`, see {doc}`installation`):

```console
pypalc gui
```

starts the server and opens `http://localhost:5006/pyPALC` in the browser.
The server runs until it is stopped with Ctrl+C. The one-word command
`pypalc_gui` does exactly the same and takes the same options. If a
package of the GUI is missing, the command prints that the installation
is incomplete and how to reinstall pyPALC, and ends with exit code 6.

In a cloned repository, `pip install -e .` once in the conda or pip
environment makes `pypalc`, `pypalc gui` and `pypalc_gui` available from
any folder; `panel serve pyPALC --show` (see below) keeps working without
it.

Options (for `pypalc gui` and `pypalc_gui`):

`--port PORT`
: Port of the server, default 5006.

`--address ADDRESS`
: Address the server listens on. Without it, the server listens on all
  network interfaces of the computer, but accepts the app's websocket
  connection only from `localhost:PORT` (see `--allow-websocket-origin`).

`--allow-websocket-origin HOST[:PORT]`
: A host whose pages may open the app's websocket connection; give the
  option once per host. It must name the host (and port, unless it is 80
  or 443) that the browser shows in its address bar. The default is
  `localhost:PORT`.

`--prefix PREFIX`
: A URL prefix: the app is then served at `PREFIX/pyPALC`, for a reverse
  proxy that forwards a sub-path (see {doc}`server`).

`--show`, `--no-show`
: Open the app in the browser (default) or not.

The app is always served at the path `/pyPALC`.

### From a checkout

In a clone of the repository the app also runs without installing pyPALC,
with Panel's own command, from the folder that *contains* the repository
folder `pyPALC`:

```console
panel serve pyPALC --show
```

Panel runs `pyPALC/main.py` for every session, which serves the same app
as `pypalc gui` ({func}`pypalc.gui.serve.create_view`). `panel serve`
takes the options `--port`, `--address`, `--allow-websocket-origin` and
`--prefix` too.

## How the code is organised

The GUI is the package `pypalc.gui`. It does not compute anything itself:
it collects what is entered on its pages into a
{class}`~pypalc.core.settings.Setup` and computes it with
{func}`pypalc.core.pipeline.run`, the same way as `pypalc run`.

| module | contents |
|---|---|
| `serve.py` | {func}`~pypalc.gui.serve.create_view` builds the app of one session; {func}`~pypalc.gui.serve.serve` starts the server for `pypalc gui` |
| `app.py` | {class}`~pypalc.gui.app.PalcApp`, the app of one session: the page frame (header with the page tabs, content area), switching pages, the widgets and figures shared by several pages (the venue figure, the axis mode selection, the error text of pages 2 to 4) and the calculation; {class}`~pypalc.gui.app.SessionState`, the calculation objects of a session |
| `help_text.py` | page 0, Help (`HelpPage`), and its texts ({data}`~pypalc.gui.help_text.HELP`) |
| `page_venue.py` | page 1, Venue (`VenuePage`): the audience lines, "Load Setup" and the example selection |
| `page_loudspeaker.py` | page 2, Loudspeaker (`LoudspeakerPage`) |
| `page_array.py` | page 3, Array (`ArrayPage`), with the reference array |
| `page_algorithm.py` | page 4, Algorithm (`AlgorithmPage`): the algorithm options, "PALC Run", the downloads, "Save Setup" / "Load Setup" and the SPL over distance with the target slope |
| `page_results.py` | page 5, Sound Field (`ResultsPage`): frequency responses, homogeneity, bar charts and beam plots |
| `setup_io.py` | {func}`~pypalc.gui.setup_io.setup_from_app` (entered values to a `Setup`) and {func}`~pypalc.gui.setup_io.apply_setup` (a loaded `Setup` to the widgets, then run it) |
| `downloads.py` | the files of the downloads on page 4 |
| `ranges.py` | the ranges of the figures, the axis modes of the venue figure and the equal scaling of the venue figure and the beam plots |
| `style.py` | the colors and the themes |
| `widgets.py` | the building blocks of the pages (cards, captions, inputs, selections, sliders, buttons, downloads, tables, the error alert, the page layout) and the callback rule below |
| `static/` | the logos |

Each page is a class in its own module. It gets the
{class}`~pypalc.gui.app.PalcApp` of its session (as `app`), creates its
widgets, figures and data sources, connects their callbacks and builds its
layout (`layout()`). The app holds the pages in `app.pages`, keyed by their
number. No module keeps state of a session at module level: Panel objects
belong to one session, so every function that builds a widget returns a
new one.

### Colors and theme

`style.py` holds the colors as constants (`BACKGROUND`, `SURFACE`,
`TEXT`, `PRIMARY`, ...), the Material UI theme of the page
(`MATERIAL_THEME`, used by {meth}`PalcApp.view
<pypalc.gui.app.PalcApp.view>`) and the Bokeh theme of the figures
({func}`~pypalc.gui.style.bokeh_figure_theme`). To change the look of the
app, change them there.

### Help texts

`help_text.py` holds the texts of the Help page in
{data}`~pypalc.gui.help_text.HELP`: one tab per page, each with columns of
HTML paragraphs. A step is introduced with its label as the GUI shows it,
in bold.

### The callback rule

Every callback of the app that Panel calls, a widget watcher or a button
click, is registered through {func}`~pypalc.gui.widgets.watch` or
{func}`~pypalc.gui.widgets.on_click` of `widgets.py`, never with
`widget.param.watch(...)` or `button.on_click(...)` directly:

```python
# page_venue.py
watch(widget, lambda event: self.update_point(
    'value', event.old, event.new), 'value')
on_click(self.save_line_button, lambda event: self.save_line())
```

Both wrap the callback with {func}`~pypalc.gui.widgets.locked`. The reason:
a session may change its Bokeh document only while it holds the lock of
the document. Panel calls widget callbacks with the lock held, except when
a browser event (a click, Enter) arrives while a typed value is still
waiting for its debounce: then it runs both without the lock, the callback
fails when it changes a figure, and the typed value or the click is lost.
A `locked` callback runs at once when the lock is held and otherwise
schedules itself as a next tick callback of the document, which Bokeh runs
with the lock and in order, so a value typed right before a click is
stored before the click is handled. Python `on_change` callbacks of Bokeh
models are called by Bokeh with the lock held and need no wrapper.

`tests/test_locked_callbacks.py` checks the rule: every callback of a
session is a `locked` one, and no module of `pypalc/gui` calls
`.param.watch(` or `.on_click(` itself.

## Add an example

The examples of the "Example" selection on page 1 and of `pypalc examples`
are the setup files in the folder `pypalc/examples/` of the package. To
add one, put a setup file there, with the extension `.yaml` (or `.yml`).
If it reads its measured directivity from a file (`path`), put that file
next to it as a `.csv` file: the package includes `.yaml`, `.yml` and
`.csv` files of this folder, and `pypalc examples` copies every file of
it except Python files.

* Add a download link for the new file to the list of example downloads
  in {doc}`command-line` (one bullet per file, under "Get the examples").
* The title in the selection comes from the file name: underscores become
  spaces and the first letter is upper case, so `gap_soft_margin.yaml` is
  shown as "Gap soft margin". The examples are sorted by title.
* The selection is built when a session starts, so a new example appears
  in the next browser window or after reloading the page.
* Loading an example in the GUI embeds a `path` data file in the setup,
  so examples with a data file work in the GUI as well.
* The tests compute every example (`tests/test_examples.py`) and load it
  through page 1 (`tests/test_load_example.py`).
