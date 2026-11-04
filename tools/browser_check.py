"""Check the running pyPALC app in a real browser and take screenshots.

Drives a headless Firefox (selenium, geckodriver) against a running pyPALC
server, takes a full-page screenshot of every page in several states and
checks the functions that only a browser exercises: loading a setup through
the file input, the running indicator of "PALC Run", the downloads,
dragging the target slope and that the toolbars show all their tools.

Works against a Bokeh 2.2 and a Bokeh 3.10 server. In Bokeh 3 the widgets
live in shadow DOM, so every DOM query descends into the shadow roots, and
models are read through ``Bokeh.documents[0]``.

Usage (from the repository root, with selenium installed)::

    python tools/browser_check.py --url http://localhost:5006/pyPALC \\
        --out shots [--setup tests/cases/gap_soft_margin_target_slope.yaml]

Writes ``OUT/<page>_<state>.png`` and ``OUT/report.json`` (a list of
``{"name": str, "ok": bool, "detail": str}``, one entry per check) and the
downloaded files to ``OUT/downloads``; screenshots of an earlier run in OUT
are removed first. Exits with 1 if any check failed.

author: Arne Hoelter
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, List, Optional

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pypalc.core.export import TILT_ANGLES_HEADER  # noqa: E402
from pypalc.core.settings import load_setup, load_setup_text  # noqa: E402
from pypalc.gui.setup_io import GAP_HANDLINGS, WEIGHTINGS  # noqa: E402

#: Default setup file loaded by the load check.
DEFAULT_SETUP = REPO / 'tests' / 'cases' / 'gap_soft_margin_target_slope.yaml'
#: Default path of geckodriver (the snap's, which matches snap Firefox).
GECKODRIVER = '/snap/bin/geckodriver'
#: Browser window size in pixels.
WINDOW = (1400, 1000)
#: Pages: index of the page selection, label of its button and file name
#: part of its screenshots.
PAGES = [(0, 'Help', 'help'), (1, '1. PAL', '1_pal'),
         (2, '2. Loudspeaker Configuration', '2_loudspeaker'),
         (3, '3. Array Configuration', '3_array'),
         (4, '4. Algorithm', '4_algorithm'),
         (5, '5. Sound Field Prediction', '5_sound_field')]
#: Order of the screenshots after loading: page 4 comes before page 1,
#: because showing page 1 removes the rays from the venue figure.
LOADED_ORDER = [5, 4, 3, 2, 0, 1]
#: Labels of the start toggle and the buttons the checks click.
RUN_LABEL = '26. PALC Run'
DOWNLOAD_ANGLES_LABEL = 'Download PALC Angles'
SAVE_SETUP_LABEL = 'Save Setup'
#: Files the downloads create (pypalc.gui.page_algorithm).
ANGLES_FILE = 'tilt_angles.csv'
SETUP_FILE = 'pypalc_setup.yaml'
#: Look of the running toggle in Bokeh 2 (templates/index.html): dark red
#: background and the label extended to "Running...".
RUNNING_BACKGROUND = 'rgb(139, 0, 0)'
RUNNING_AFTER = '"ning..."'
#: GUI labels of the setup values, for the load check.
GAP_HANDLING_LABELS = {value: label for label, value in GAP_HANDLINGS.items()}
WEIGHTING_LABELS = {value: label for label, value in WEIGHTINGS.items()}

#: JavaScript helpers prepended to every script. ``deep`` queries the
#: document and all shadow roots (Bokeh 3); ``models`` lists the models of
#: the document (a Map in Bokeh 2.2 and 3.10); ``button`` finds a
#: ``.bk-btn`` by its label.
JS_HELPERS = r"""
const doc = Bokeh.documents[0];
function deep(sel, root, out) {
    root = root || document; out = out || [];
    for (const e of root.querySelectorAll(sel)) out.push(e);
    for (const e of root.querySelectorAll('*'))
        if (e.shadowRoot) deep(sel, e.shadowRoot, out);
    return out;
}
function typeName(m) { return m.type || m.constructor.__name__; }
function models(type) {
    const all = doc._all_models;
    const list = all.values ? Array.from(all.values()) : Object.values(all);
    return type ? list.filter(m => typeName(m) === type) : list;
}
function pageSelect() {
    return models('RadioButtonGroup').find(
        m => m.labels.indexOf('4. Algorithm') >= 0);
}
function rootId() {
    const roots = typeof doc.roots === 'function' ? doc.roots() : doc.roots;
    return roots.map(r => r.id).join(',');
}
// doc.is_idle of Bokeh 3 only covers the first roots, so ask the root
// views (Bokeh.index maps root ids to views in Bokeh 2 and 3)
function isIdle() {
    const roots = typeof doc.roots === 'function' ? doc.roots() : doc.roots;
    return roots.every(r => {
        const view = Bokeh.index[r.id];
        return view != null && (typeof view.has_finished !== 'function' ||
                                view.has_finished());
    });
}
function button(label) {
    return deep('.bk-btn').find(b => b.textContent.trim() === label);
}
function visible(e) { return e.getClientRects().length > 0; }
"""


class BrowserCheck:
    """The checks of one browser session.

    Parameters
    ----------
    driver : selenium.webdriver.Firefox
        The browser.
    url : str
        URL of the pyPALC app.
    out : Path
        Folder of the screenshots and the report.
    download_dir : Path
        Download folder of the browser.
    setup : Path
        Setup file the load check loads.
    settle : float
        Seconds the document must stay unchanged before a page counts as
        shown.
    timeout : float
        Seconds to wait for a page, a calculation or a download.
    """

    def __init__(self, driver, url: str, out: Path, download_dir: Path,
                 setup: Path, settle: float = 1.5,
                 timeout: float = 180.0) -> None:
        self.driver = driver
        self.url = url
        self.out = out
        self.download_dir = download_dir
        self.setup = setup
        self.settle = settle
        self.timeout = timeout
        self.results: List[dict] = []
        #: Screenshots in which a toolbar had moved tools into its overflow
        #: menu (Bokeh 3), with the number of such toolbars.
        self.toolbar_overflows: List[str] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def js(self, code: str, *args):
        """Run JavaScript with the helpers of :data:`JS_HELPERS`.

        Parameters
        ----------
        code : str
            Body of the script; ``arguments`` are the further parameters.
        *args
            Arguments of the script.

        Returns
        -------
        object
            The value the script returns.
        """
        return self.driver.execute_script(JS_HELPERS + code, *args)

    def add(self, name: str, ok: bool, detail: str) -> None:
        """Record the result of a check and print it.

        Parameters
        ----------
        name : str
            Name of the check.
        ok : bool
            True if the check passed.
        detail : str
            What was found.

        Returns
        -------
        None.
        """
        self.results.append(dict(name=name, ok=bool(ok), detail=detail))
        print('%-4s %s: %s' % ('ok' if ok else 'FAIL', name, detail),
              flush=True)

    def run_check(self, name: str, check: Callable[[], tuple]) -> bool:
        """Run one check; an exception counts as a failure.

        Parameters
        ----------
        name : str
            Name of the check.
        check : callable
            Returns ``(ok, detail)``.

        Returns
        -------
        bool
            True if the check passed.
        """
        try:
            ok, detail = check()
        except Exception as exc:  # the report must cover every check
            ok, detail = False, '%s: %s' % (type(exc).__name__, exc)
        self.add(name, ok, detail)
        return ok

    def wait_for(self, condition: Callable[[], object], what: str,
                 timeout: Optional[float] = None, poll: float = 0.25):
        """Wait until ``condition()`` is true.

        Parameters
        ----------
        condition : callable
            Returns a true value when the wait is over.
        what : str
            What is waited for (for the error).
        timeout : float, optional
            Seconds; default the timeout of the session.
        poll : float
            Seconds between two calls.

        Returns
        -------
        object
            The true value of ``condition()``.

        Raises
        ------
        TimeoutError
            If the condition stays false.
        """
        end = time.time() + (self.timeout if timeout is None else timeout)
        while True:
            value = condition()
            if value:
                return value
            if time.time() > end:
                raise TimeoutError('timed out waiting for ' + what)
            time.sleep(poll)

    def wait_settled(self) -> None:
        """Wait until the roots of the document stay the same and BokehJS
        is idle for :attr:`settle` seconds."""
        last, since = None, time.time()
        end = time.time() + self.timeout
        while time.time() < end:
            state = self.js('return [rootId(), isIdle()];')
            if state != last or not state[1]:
                last, since = state, time.time()
            elif time.time() - since >= self.settle:
                return
            time.sleep(0.25)
        raise TimeoutError('the page did not settle')

    def active_page(self) -> int:
        """Index of the page the page selection shows."""
        return self.js('return pageSelect().active;')

    def show_page(self, index: int) -> None:
        """Show a page by clicking its button and wait until it is drawn.

        Parameters
        ----------
        index : int
            Index of the page (0: Help).

        Returns
        -------
        None.
        """
        label = PAGES[index][1]
        if self.active_page() != index:
            old_root = self.js('return rootId();')
            clicked = self.js("""
                const b = button(arguments[0]);
                if (!b) return false;
                b.click(); return true;""", label)
            if not clicked:
                raise RuntimeError('no button %r' % label)
            self.wait_for(lambda: self.js('return rootId();') != old_root,
                          'page %r' % label, timeout=60)
        self.wait_settled()

    def screenshot(self, index: int, state: str) -> Path:
        """Save a full-page screenshot as ``OUT/<page>_<state>.png``.

        Parameters
        ----------
        index : int
            Index of the page shown.
        state : str
            State of the app (fresh, loaded, running).

        Returns
        -------
        Path
            The PNG file.
        """
        path = self.out / ('%s_%s.png' % (PAGES[index][2], state))
        overflows = self.js("""
            return deep('.bk-tool-overflow').filter(
                e => e.getBoundingClientRect().height > 0).length;""")
        if overflows:
            self.toolbar_overflows.append('%s: %d' % (path.name, overflows))
        self.driver.save_full_page_screenshot(str(path))
        return path

    def selects(self) -> dict:
        """Shown selects: title -> [value, visible]."""
        return self.js("""
            const out = {};
            for (const s of deep('select')) {
                let label = s.parentElement.querySelector('label');
                if (!label && s.getRootNode().querySelector)
                    label = s.getRootNode().querySelector('label');
                const title = label ? label.textContent.trim() : '';
                out[title.replace(/:$/, '')] = [s.value, visible(s)];
            }
            return out;""")

    def wait_download(self, name: str) -> Path:
        """Wait until the browser has saved a download.

        Parameters
        ----------
        name : str
            File name of the download.

        Returns
        -------
        Path
            The downloaded file.
        """
        path = self.download_dir / name
        part = self.download_dir / (name + '.part')
        self.wait_for(lambda: path.exists() and not part.exists()
                      and path.stat().st_size > 0,
                      'download ' + name, timeout=30)
        time.sleep(0.5)
        return path

    # ------------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------------

    def check_fresh_screenshots(self) -> tuple:
        """Screenshot of every page of a new session (state ``fresh``)."""
        files = []
        for index, _, _ in PAGES:
            self.show_page(index)
            files.append(self.screenshot(index, 'fresh').name)
        return True, ', '.join(files)

    def check_load(self) -> tuple:
        """Load the setup through the file input of page 4.

        Sets ``filename``, ``mime_type`` and ``value`` (base64) of the
        FileInput model, as the browser does after a file was chosen. The
        app must switch to page 5; page 4 must then show the gap handling,
        the weighting and the number of hinges of the setup.
        """
        setup = load_setup(self.setup)
        expected = {
            'Gap Handling': GAP_HANDLING_LABELS[setup.algorithm.gap_handling],
            'Weighting': WEIGHTING_LABELS[setup.algorithm.weighting]}
        if setup.target_slope is not None:
            expected['Hinges'] = str(len(setup.target_slope.points) - 2)
        self.show_page(4)
        data = base64.b64encode(self.setup.read_bytes()).decode('ascii')
        found = self.js("""
            const inputs = models('FileInput').filter(
                m => m.accept.indexOf('.yaml') >= 0);
            if (inputs.length !== 1) return inputs.length;
            inputs[0].setv({filename: arguments[1],
                            mime_type: 'application/x-yaml',
                            value: arguments[0]});
            return 1;""", data, self.setup.name)
        if found != 1:
            return False, '%s setup file inputs found' % found
        start = time.time()
        try:
            self.wait_for(lambda: self.active_page() == 5,
                          'page 5 after loading')
        except TimeoutError:
            error = self.js("""
                const p = models('Paragraph').find(
                    m => m.text.indexOf('Error') === 0);
                return p ? p.text : '';""")
            return False, 'app did not switch to page 5 within %ds; page ' \
                '%s; error text %r' % (self.timeout, self.active_page(),
                                      error)
        self.wait_settled()
        seconds = time.time() - start
        self.screenshot(5, 'loaded')
        self.show_page(4)
        shown = self.selects()
        problems = []
        for key, value in expected.items():
            matches = [v for title, v in shown.items() if key in title]
            if len(matches) != 1:
                problems.append('%s: %d selects' % (key, len(matches)))
            elif matches[0] != [value, True]:
                problems.append('%s: shows %r (visible %s), expected %r'
                                % (key, matches[0][0], matches[0][1], value))
        self.screenshot(4, 'loaded')
        detail = 'switched to page 5 after %.1fs; page 4 shows %s' % (
            seconds, ', '.join('%s=%r' % (k, shown[t][0])
                               for k in expected for t in shown if k in t))
        if problems:
            detail += '; ' + '; '.join(problems)
        return not problems, detail

    def check_loaded_screenshots(self) -> tuple:
        """Screenshots of the other pages after loading (state
        ``loaded``); pages 5 and 4 were taken by the load check."""
        files = []
        for index in LOADED_ORDER[2:]:
            self.show_page(index)
            files.append(self.screenshot(index, 'loaded').name)
        return True, ', '.join(files)

    def check_download_angles(self) -> tuple:
        """"Download PALC Angles" saves ``tilt_angles.csv`` with the header
        of the headless export."""
        self.show_page(4)
        if not self.js("""const b = button(arguments[0]);
                          if (b) b.click(); return !!b;""",
                       DOWNLOAD_ANGLES_LABEL):
            return False, 'no button %r' % DOWNLOAD_ANGLES_LABEL
        path = self.wait_download(ANGLES_FILE)
        lines = path.read_text(encoding='utf-8').splitlines()
        header = ','.join(TILT_ANGLES_HEADER)
        detail = '%s: header %r, %d rows' % (path.name, lines[0],
                                             len(lines) - 1)
        return lines[0] == header and len(lines) > 1, detail

    def check_download_setup(self) -> tuple:
        """"Save Setup" saves ``pypalc_setup.yaml``, a valid setup."""
        self.show_page(4)
        if not self.js("""const b = button(arguments[0]);
                          if (b) b.click(); return !!b;""",
                       SAVE_SETUP_LABEL):
            return False, 'no button %r' % SAVE_SETUP_LABEL
        path = self.wait_download(SETUP_FILE)
        saved = load_setup_text(path.read_text(encoding='utf-8'))
        loaded = load_setup(self.setup)
        same = (saved.algorithm.weighting == loaded.algorithm.weighting and
                saved.algorithm.gap_handling ==
                loaded.algorithm.gap_handling)
        return same, '%s: valid setup, weighting %s, gap handling %s' % (
            path.name, saved.algorithm.weighting,
            saved.algorithm.gap_handling)

    def check_running_indicator(self) -> tuple:
        """Click "PALC Run" and read the look of the toggle while the
        calculation runs; screenshot of page 4 in state ``running``.

        In Bokeh 2 the template's CSS turns the active toggle dark red and
        extends its label to "Running..." with an ``:after`` content.
        """
        self.show_page(4)
        read = """
            const b = deep('.bk-btn').find(
                e => e.textContent.trim().indexOf(arguments[0]) === 0);
            if (!b) return null;
            const toggle = models('Toggle').find(
                m => m.label === arguments[0]);
            const style = getComputedStyle(b);
            return {active: toggle.active, disabled: toggle.disabled,
                    background: style.backgroundColor,
                    after: getComputedStyle(b, '::after').content,
                    label: b.textContent.trim()};"""
        before = self.js(read, RUN_LABEL)
        if before is None:
            return False, 'no toggle %r' % RUN_LABEL
        if before['disabled']:
            return False, 'toggle disabled'
        self.js("button(arguments[0]).click();", RUN_LABEL)
        during = self.js(read, RUN_LABEL)
        self.screenshot(4, 'running')
        still = self.js(read, RUN_LABEL)['active']
        start = time.time()
        self.wait_for(lambda: not self.js(read, RUN_LABEL)['active'],
                      'the end of the calculation')
        self.wait_settled()
        label = during['label'] + (during['after'].strip('"')
                                   if during['after'] not in
                                   ('none', 'normal', '') else '')
        detail = 'during the run: active %s, background %s, label %r ' \
            '(after %s); before: background %s; screenshot taken while ' \
            'running: %s; run ended %.1fs later' % (
                during['active'], during['background'], label,
                during['after'], before['background'], still,
                time.time() - start)
        ok = (during['active'] and still and
              during['background'] == RUNNING_BACKGROUND and
              during['after'] == RUNNING_AFTER)
        return ok, detail

    def check_drag(self) -> tuple:
        """Move a target point in the browser, as the PointDrawTool does,
        and read the server's answer back.

        Selects the middle target point, writes a new distance (a little
        off a grid point) and level into the target source and clears the
        selection. The server must snap the point to the grid of the SPL
        over distance (distance and venue position of the grid point) and
        update the optimization region; both are read back from the models
        in the browser after the round trip.
        """
        self.show_page(4)
        state = self.js("""
            const tool = models('PointDrawTool')[0];
            const target = tool.renderers[0].data_source;
            const grid = models('ColumnDataSource').find(m =>
                Object.keys(m.data).sort().join() ===
                'distance_m,spl_db,x_m,y_m');
            return {distance_m: Array.from(target.data.distance_m),
                    level_db: Array.from(target.data.level_db),
                    grid_d: Array.from(grid.data.distance_m),
                    grid_x: Array.from(grid.data.x_m)};""")
        distances, grid_d, grid_x = (state['distance_m'], state['grid_d'],
                                     state['grid_x'])
        if len(distances) < 3:
            return False, 'target has %d points' % len(distances)
        point = len(distances) // 2
        index = [abs(d - distances[point]) for d in grid_d]
        index = index.index(min(index))
        last = [abs(d - distances[-1]) for d in grid_d]
        last = last.index(min(last))
        new_index = (index + last) // 2
        if new_index == index:
            return False, 'no grid point between the middle and last point'
        spacing = min(abs(d - grid_d[new_index])
                      for k, d in enumerate(grid_d) if k != new_index)
        new_distance = grid_d[new_index] + 0.25 * spacing
        new_level = state['level_db'][point] - 1.5
        self.js("""
            const target = models('PointDrawTool')[0]
                .renderers[0].data_source;
            const i = arguments[0];
            // tap: select the point
            target.selected.indices = [i];
            // drag: move it in place, then end the drag as the tool's
            // _emit_cds_changes(cds, false, true, true) does
            target.get_array('distance_m')[i] = arguments[1];
            target.get_array('level_db')[i] = arguments[2];
            target.selection_manager.clear();
            if (parseInt(Bokeh.version) >= 3) {
                target.setv({data: target.data}, {check_eq: false});
            } else {
                target.data = target.data;
                target.properties.data.change.emit();
            }""", point, new_distance, new_level)

        def answered():
            data = self.js("""
                const target = models('PointDrawTool')[0]
                    .renderers[0].data_source;
                const region = models('GlyphRenderer').find(
                    r => typeName(r.glyph) === 'Patch').data_source;
                return {distance_m: Array.from(target.data.distance_m),
                        level_db: Array.from(target.data.level_db),
                        x_m: Array.from(target.data.x_m),
                        region_x: Array.from(region.data.x)};""")
            return data if data['distance_m'][point] != new_distance \
                else None
        data = self.wait_for(answered, 'the server to snap the point',
                             timeout=30)
        self.wait_settled()
        data = answered() or data
        problems = []
        if not math.isclose(data['distance_m'][point], grid_d[new_index],
                            rel_tol=1e-12):
            problems.append('distance %r, expected grid point %r'
                            % (data['distance_m'][point], grid_d[new_index]))
        if not math.isclose(data['x_m'][point], grid_x[new_index],
                            rel_tol=1e-12):
            problems.append('x %r, expected %r'
                            % (data['x_m'][point], grid_x[new_index]))
        if not math.isclose(data['level_db'][point], new_level,
                            rel_tol=1e-12):
            problems.append('level %r, expected %r'
                            % (data['level_db'][point], new_level))
        if data['region_x'] != [data['distance_m'][0]] * 2 + \
                [data['distance_m'][-1]] * 2:
            problems.append('region x %r does not span the target %r'
                            % (data['region_x'], data['distance_m']))
        self.screenshot(4, 'dragged')
        detail = 'moved point %d to %.4f m / %.2f dB; server snapped it ' \
            'to %.6f m (grid point %d), level %.2f dB, region %.3f..%.3f m' \
            % (point, new_distance, new_level, data['distance_m'][point],
               new_index, data['level_db'][point], data['region_x'][0],
               data['region_x'][2])
        if problems:
            detail += '; ' + '; '.join(problems)
        return not problems, detail

    def check_toolbars(self) -> tuple:
        """Show page 1 with the venue axes "Equal (variable figure height)",
        which makes the venue figure short, take its screenshot (state
        ``equal_axes``) and check that no toolbar of any screenshot so far
        moved tools into an overflow menu (Bokeh 3 does that when the tools
        do not fit beside the frame; Bokeh 2 showed all tools). The venue
        axes are set back to "Stretched" afterwards.

        Returns
        -------
        tuple
            ``(ok, detail)``.
        """
        self.show_page(1)
        set_axes = """
            const rg = models('RadioGroup').find(
                m => m.labels[0].startsWith('Equal (variable'));
            const old = rg.active; rg.active = arguments[0]; return old;"""
        old = self.js(set_axes, 0)
        time.sleep(self.settle)
        self.wait_settled()
        self.screenshot(1, 'equal_axes')
        self.js(set_axes, old)
        self.wait_settled()
        if self.toolbar_overflows:
            return False, ('toolbars with an overflow menu in ' +
                           ', '.join(self.toolbar_overflows))
        return True, 'every tool shown in all screenshots'

    def run(self) -> List[dict]:
        """Open the app and run all checks.

        Returns
        -------
        list of dict
            One entry ``{"name", "ok", "detail"}`` per check.
        """
        self.driver.get(self.url)
        self.wait_for(lambda: self.driver.execute_script(
            "return !!(window.Bokeh && Bokeh.documents && "
            "Bokeh.documents.length);"), 'BokehJS', timeout=60)
        self.wait_settled()
        version = self.driver.execute_script('return Bokeh.version;')
        print('BokehJS %s at %s' % (version, self.url), flush=True)
        self.run_check('screenshots_fresh', self.check_fresh_screenshots)
        if self.run_check('load', self.check_load):
            self.run_check('screenshots_loaded',
                           self.check_loaded_screenshots)
            self.run_check('download_angles', self.check_download_angles)
            self.run_check('download_setup', self.check_download_setup)
            self.run_check('running_indicator', self.check_running_indicator)
            self.run_check('drag', self.check_drag)
            self.run_check('toolbars', self.check_toolbars)
        else:
            for name in ('screenshots_loaded', 'download_angles',
                         'download_setup', 'running_indicator', 'drag',
                         'toolbars'):
                self.add(name, False, 'skipped: the setup did not load')
        return self.results


def make_driver(download_dir: Path, geckodriver: str):
    """Start headless Firefox with downloads saved to ``download_dir``.

    Parameters
    ----------
    download_dir : Path
        Download folder; snap Firefox can only write below the home folder.
    geckodriver : str
        Path of geckodriver.

    Returns
    -------
    selenium.webdriver.Firefox
    """
    from selenium import webdriver
    from selenium.webdriver.firefox.options import Options
    from selenium.webdriver.firefox.service import Service
    options = Options()
    options.add_argument('-headless')
    for key, value in (
            ('browser.download.folderList', 2),
            ('browser.download.dir', str(download_dir)),
            ('browser.download.useDownloadDir', True),
            ('browser.download.manager.showWhenStarting', False),
            ('browser.download.always_ask_before_handling_new_types', False),
            ('browser.download.start_downloads_in_tmp_dir', False),
            ('browser.helperApps.neverAsk.saveToDisk',
             'text/csv,text/yaml,application/x-yaml,application/octet-stream'),
            ('browser.download.panel.shown', False)):
        options.set_preference(key, value)
    driver = webdriver.Firefox(options=options,
                               service=Service(geckodriver))
    driver.set_window_size(*WINDOW)
    driver.set_script_timeout(30)
    return driver


def main(argv: Optional[List[str]] = None) -> int:
    """Command line entry point.

    Parameters
    ----------
    argv : list of str, optional
        Arguments; default ``sys.argv[1:]``.

    Returns
    -------
    int
        0 if all checks passed, otherwise 1.
    """
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--url', required=True,
                        help='URL of the app, '
                        'e.g. http://localhost:5006/pyPALC')
    parser.add_argument('--out', required=True, type=Path,
                        help='folder of the screenshots and report.json')
    parser.add_argument('--setup', type=Path, default=DEFAULT_SETUP,
                        help='setup file of the load check (default: %s)'
                        % DEFAULT_SETUP.relative_to(REPO))
    parser.add_argument('--download-dir', type=Path, default=None,
                        help='download folder of the browser (default: a '
                        'new folder in the home folder, removed afterwards; '
                        'snap Firefox cannot write to /tmp)')
    parser.add_argument('--geckodriver', default=GECKODRIVER,
                        help='path of geckodriver (default: %(default)s)')
    parser.add_argument('--settle', type=float, default=1.5,
                        help='seconds a page must stay unchanged '
                        '(default: %(default)s)')
    parser.add_argument('--timeout', type=float, default=180.0,
                        help='seconds to wait for a calculation '
                        '(default: %(default)s)')
    args = parser.parse_args(argv)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob('*.png'):
        old.unlink()
    remove = args.download_dir is None
    download_dir = Path(tempfile.mkdtemp(prefix='pypalc_browser_check_',
                                         dir=Path.home())) \
        if remove else args.download_dir.resolve()
    download_dir.mkdir(parents=True, exist_ok=True)
    for name in (ANGLES_FILE, SETUP_FILE):
        (download_dir / name).unlink(missing_ok=True)
    driver = make_driver(download_dir, args.geckodriver)
    check = BrowserCheck(driver, args.url, out, download_dir,
                         args.setup.resolve(), settle=args.settle,
                         timeout=args.timeout)
    try:
        check.run()
    except Exception as exc:  # report what ran so far
        check.add('session', False, '%s: %s' % (type(exc).__name__, exc))
    finally:
        driver.quit()
        downloads = out / 'downloads'
        shutil.rmtree(downloads, ignore_errors=True)
        shutil.copytree(download_dir, downloads)
        if remove:
            shutil.rmtree(download_dir, ignore_errors=True)
    (out / 'report.json').write_text(json.dumps(check.results, indent=2)
                                     + '\n', encoding='utf-8')
    failed = [r['name'] for r in check.results if not r['ok']]
    print('%d of %d checks ok%s' % (len(check.results) - len(failed),
                                     len(check.results),
                                     '; failed: ' + ', '.join(failed)
                                     if failed else ''))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
