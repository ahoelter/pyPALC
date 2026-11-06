"""Check the running pyPALC app in a real browser and take screenshots.

Drives a headless Firefox (selenium, geckodriver) against a running pyPALC
server (``panel serve pyPALC`` from the parent folder), takes a screenshot
of every page in several states and checks what only a browser exercises:
loading a setup through the file input, the running look of "PALC Run",
the downloads (file names and CSV headers), dragging the target slope, that
page 1 removes the rays of a result, the equal scaling and the axis labels
of the figures, the linked beam plots, that every toolbar shows all its
tools, that no page scrolls horizontally at the window width and that a
value typed right before a click is not lost. With ``--two-tabs`` it also
checks that two browser windows on one server do not share a session.

The page is a Material page of Panel: its content scrolls inside the
``.main-content`` element, never the document, so horizontal overflow is
measured there. Widgets are read and used through the DOM (their labels),
Bokeh figures, sources and tools through the views of BokehJS (the models
the page shows; ``Bokeh.documents[0]`` may still hold models of pages shown
before).

Usage (from the repository root, with selenium installed)::

    python tools/browser_check.py --url http://localhost:5006/pyPALC \\
        --out shots [--width 1280] [--height H] [--two-tabs] \\
        [--setup tests/cases/gap_soft_margin_target_slope.yaml] \\
        [--download-dir DIR] [--clean] [--server-log LOG]

Writes ``OUT/<page>_<state>.png`` (page e.g. ``1_venue``), ``OUT/report.json``
(a list of ``{"name": str, "ok": bool, "detail": str}``, one entry per
check) and the downloaded files to ``OUT/downloads``. Exits with 1 if any
check failed. A screenshot shows the whole page: while it is taken, the
window is made as high as the page (the window width stays); the
screenshot of a running calculation shows the window only.

OUT and a given ``--download-dir`` must be empty or new, so that no file of
an earlier run is taken for a result of this one. With ``--clean`` the tool
first removes the files it writes itself (:func:`tool_files`: the
screenshots, ``report.json`` and the download files with their numbered
copies) from them, and never touches any other file.

author: Arne Hoelter
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Callable, List, Optional

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pypalc.core.settings import load_setup, load_setup_text  # noqa: E402
from pypalc.gui.app import (AXIS_MODES, PAGE_LABELS,  # noqa: E402
                            VENUE_FIGURE_NAME)
from pypalc.gui.downloads import (DOWNLOAD_COLUMNS,  # noqa: E402
                                  DOWNLOAD_FILE_NAMES, RESULTS_FILE_NAME,
                                  SETUP_FILE_NAME)
from pypalc.gui.page_algorithm import RUN_LABEL, RUNNING_LABEL  # noqa: E402
from pypalc.gui.ranges import (VENUE_MAX_HEIGHT,  # noqa: E402
                               window_limit)
from pypalc.gui.setup_io import GAP_HANDLINGS, WEIGHTINGS  # noqa: E402

#: Default setup file loaded by the load check.
DEFAULT_SETUP = REPO / 'tests' / 'cases' / 'gap_soft_margin_target_slope.yaml'
#: Default path of geckodriver (the snap's, which matches snap Firefox).
GECKODRIVER = '/snap/bin/geckodriver'
#: Default window width; the default height is :data:`HEIGHT` below
#: :data:`WIDE` px width and :data:`WIDE_HEIGHT` from there on.
WIDTH = 1280
HEIGHT = 800
WIDE = 1920
WIDE_HEIGHT = 1080
#: File name part of the screenshots of each page (index = page index).
PAGE_NAMES = ['%d_%s' % (index, '_'.join(re.findall(r'[a-z]+',
                                                     label.lower())))
              for index, label in enumerate(PAGE_LABELS)]
#: Order of the screenshots after loading: page 5 and 4 come first (load
#: check), page 1 last, because showing page 1 removes the rays.
LOADED_ORDER = [5, 4, 3, 2, 0, 1]
#: Labels of the buttons the checks click.
DOWNLOAD_ANGLES_LABEL = 'Download PALC Angles'
DOWNLOAD_RESULTS_LABEL = 'Download Results'
SAVE_SETUP_LABEL = 'Save Setup'
SAVE_LINE_LABEL = '6. Save Current Audience Line'
REMOVE_LINE_LABEL = '7. Remove Last Audience Line'
#: Tabs of page 4 and 5 and the buttons of the beam plots.
PAGE_TABS = {4: ['Venue slice', 'SPL over distance'],
             5: ['Frequency responses', 'Homogeneity', 'Bar charts',
                 'Beam plots']}
VENUE_TAB = 'Venue slice'
BEAM_TAB = 'Beam plots'
BEAM_BUTTONS = ['PALC', 'Reference']
#: Titles of the two beam plots.
BEAM_TITLES = ['34. Beam plot: PALC (SPL in dB)',
               '34. Beam plot: Reference array (SPL in dB)']
#: Venue inputs of page 1 (label -> value) and the audience line they draw
#: in the two-tab check.
TWO_TABS_LINE = {'1. Start x in m': '2', '4. Start y in m': '0',
                 '2. End x in m': '10', '5. End y in m': '1',
                 '3. Discretization in points per m': '5'}
#: The input the fast input check types into right before it clicks
#: "Save Current Audience Line", and the values it types; the other inputs
#: of the line are those of :data:`TWO_TABS_LINE`.
FAST_INPUT_LABEL = '5. End y in m'
FAST_INPUT_VALUES = ('1.5', '2.5', '3.5', '4.5', '5.5')
#: Lines of a server log that show an exception in a callback.
SERVER_LOG_ERRORS = ('Traceback', 'Error thrown from callback')
#: Inputs of pages 2 and 3 the axis label check changes: page -> (label,
#: value). The array height changes the venue ranges, so the venue figure
#: gets new ranges.
CONFIG_INPUTS = {2: ('9. Aperture angle in deg', '3.5'),
                 3: ('14. Top y in m', '2.5')}
#: Pages whose venue figure is in a tab, the other tab there and an input
#: of the page that gives the venue figure new ranges: page -> (tab, label,
#: value). The equal axes check changes the input (and sets it back) while
#: the other tab is shown; a value of None is a slider moved one step up
#: (and down again) with the arrow keys.
HIDDEN_VENUE_INPUTS = {
    2: ('Directivity', '9. Aperture angle in deg', '3.5'),
    4: ('SPL over distance', '23. Tolerance (abort criterion) in m', None)}
#: The CSV files of "Download Results" (all but the tilt angles) and their
#: headers.
RESULT_FILES = {DOWNLOAD_FILE_NAMES[key]: list(DOWNLOAD_COLUMNS[key])
                for key in DOWNLOAD_FILE_NAMES if key != 'results'}
ANGLES_FILE = DOWNLOAD_FILE_NAMES['results']
#: All files the downloads create.
DOWNLOAD_FILES = (ANGLES_FILE, RESULTS_FILE_NAME, SETUP_FILE_NAME)
#: States of the screenshots (``OUT/<page>_<state>.png``); one per axis
#: mode of the venue figure (``axes_0`` ...).
SCREENSHOT_STATES = ('fresh', 'loaded', 'running', 'dragged', 'beam_plots',
                     'axes_0', 'axes_1', 'axes_2', 'tabs_1', 'tabs_2',
                     'tabs_3')
#: Name of the report in OUT and of the copy of the downloads in OUT.
REPORT_FILE = 'report.json'
DOWNLOADS_COPY = 'downloads'
#: Look of "PALC Run" while a calculation runs: dark red background.
RUNNING_BACKGROUND = 'rgb(139, 0, 0)'
#: Largest allowed difference of the pixels per meter on the x and the y
#: axis of an equally scaled figure, relative.
EQUAL_TOLERANCE = 0.01
#: Largest allowed difference in px of a size of an equally scaled figure
#: from the size it must have (the width of "Stretched", the height of
#: "Stretched", the largest height); the browser rounds sizes to whole px.
SIZE_TOLERANCE = 1.5
#: Largest allowed gap in px between the image of a beam plot and its
#: frame.
IMAGE_GAP = 1
#: Smallest space in px left of and below the frame of the venue figure:
#: its tick labels and axis titles are there (about 45 px), without them
#: the frame fills the figure (a few px).
AXIS_SPACE = 25
#: GUI labels of the setup values, for the load check.
GAP_HANDLING_LABELS = {value: label for label, value in GAP_HANDLINGS.items()}
WEIGHTING_LABELS = {value: label for label, value in WEIGHTINGS.items()}

#: JavaScript helpers prepended to every script. ``deep`` queries the
#: document and all shadow roots; ``views`` lists the views of BokehJS (the
#: models shown); ``text`` is the text of an element without its Material
#: icon; ``fields`` maps the label of each shown input or select to its
#: value; ``plotState`` measures a figure: its frame (``width``,
#: ``height``), ranges, outer size, the width of its container
#: (``available``) and, for a beam plot, the extent of its image and the
#: largest gap in px between the image and the frame.
JS_HELPERS = r"""
function deep(sel, root, out) {
    root = root || document; out = out || [];
    for (const e of root.querySelectorAll(sel)) out.push(e);
    for (const e of root.querySelectorAll('*'))
        if (e.shadowRoot) deep(sel, e.shadowRoot, out);
    return out;
}
function typeName(m) { return m.type || m.constructor.__name__; }
function views() {
    const out = [];
    function walk(v) {
        out.push(v);
        for (const c of (v.children ? v.children() : [])) walk(c);
    }
    for (const v of Bokeh.index) walk(v);
    return out;
}
function live(type) {
    return views().map(v => v.model).filter(m => typeName(m) === type);
}
function liveSources() {
    const out = [];
    for (const r of live('GlyphRenderer'))
        if (out.indexOf(r.data_source) < 0) out.push(r.data_source);
    return out;
}
function figureViews() {
    return views().filter(v => typeName(v.model) === 'Figure');
}
function figureView(title) {
    return figureViews().find(
        v => v.model.title && v.model.title.text === title);
}
function venueView() {
    return figureViews().find(v => v.model.name === VENUE_FIGURE_NAME);
}
function visible(e) { return e.getClientRects().length > 0; }
function text(e) {
    const c = e.cloneNode(true);
    for (const i of c.querySelectorAll('.material-icons')) i.remove();
    return c.textContent.trim();
}
function button(label) {
    return deep('button').find(b => visible(b) && text(b) === label);
}
function tab(label) {
    return deep('.MuiTab-root').find(e => text(e) === label);
}
function fields() {
    const out = {};
    for (const e of deep('.MuiFormControl-root')) {
        const label = e.querySelector('label');
        if (!label || !visible(e)) continue;
        const select = e.querySelector('.MuiSelect-select');
        const input = e.querySelector('input');
        out[label.textContent.trim()] =
            select ? select.textContent.trim() : (input ? input.value : null);
    }
    return out;
}
function mainArea() { return deep('.main-content')[0]; }
function isIdle() {
    for (const v of Bokeh.index)
        if (typeof v.has_finished === 'function' && !v.has_finished())
            return false;
    return true;
}
function imageExtent(m) {
    const r = m.renderers.find(r => typeName(r.glyph) === 'Image');
    if (!r) return null;
    const d = r.data_source.data;
    return [d.x[0], d.x[0] + d.dw[0], d.y[0], d.y[0] + d.dh[0]];
}
function plotState(v) {
    if (!v) return null;
    const m = v.model, f = v.frame.bbox, r = v.el.getBoundingClientRect();
    const num = x => (x == null || isNaN(x)) ? null : x;
    const parent = v.el.parentElement;
    const p = parent ? parent.getBoundingClientRect() : null;
    const image = imageExtent(m);
    let image_gap = null;
    if (image) {
        const xs = v.frame.x_scale, ys = v.frame.y_scale;
        image_gap = Math.max(
            Math.abs(xs.compute(image[0]) - f.left),
            Math.abs(xs.compute(image[1]) - f.right),
            Math.abs(ys.compute(image[2]) - f.bottom),
            Math.abs(ys.compute(image[3]) - f.top));
    }
    return {visible: m.visible && visible(v.el), width: f.width,
            height: f.height,
            x: [num(m.x_range.start), num(m.x_range.end)],
            y: [num(m.y_range.start), num(m.y_range.end)],
            left: f.left, below: v.bbox.height - f.bottom,
            bottom: r.bottom, window_height: window.innerHeight,
            outer_width: r.width, outer_height: r.height,
            available: parent ? parent.clientWidth : null,
            gaps: p ? [r.left - p.left, p.right - r.right] : null,
            image: image, image_gap: image_gap,
            range_ids: [m.x_range.id, m.y_range.id]};
}
"""


JS_HELPERS = JS_HELPERS.replace('VENUE_FIGURE_NAME',
                                json.dumps(VENUE_FIGURE_NAME))


def pixels_per_meter(state: dict) -> tuple:
    """Pixels per meter of a figure on the x and the y axis.

    Parameters
    ----------
    state : dict
        Result of the JavaScript ``plotState``.

    Returns
    -------
    tuple of float
        ``(x, y)``; NaN for an axis without a range.
    """
    def per_meter(size, rng):
        if None in rng or rng[1] == rng[0]:
            return math.nan
        return size / (rng[1] - rng[0])
    return (per_meter(state['width'], state['x']),
            per_meter(state['height'], state['y']))


def equal_difference(state: dict) -> float:
    """Relative difference of the pixels per meter on x and y.

    Parameters
    ----------
    state : dict
        Result of the JavaScript ``plotState``.

    Returns
    -------
    float
        ``|x - y| / y``; NaN if a range is missing.
    """
    x, y = pixels_per_meter(state)
    return abs(x - y) / y if y else math.nan


class BrowserCheck:
    """The checks of one browser session.

    Parameters
    ----------
    driver : selenium.webdriver.Firefox or None
        The browser; may be set after construction (``None`` until then).
    url : str
        URL of the pyPALC app.
    out : Path
        Folder of the screenshots and the report.
    download_dir : Path
        Download folder of the browser.
    setup : Path
        Setup file the load check loads.
    window : tuple of int
        Window width and height in px.
    settle : float
        Seconds the page must stay unchanged before it counts as shown.
    timeout : float
        Seconds to wait for a page, a calculation or a download.
    server_log : Path, optional
        Log file of the server; the fast input check then also checks that
        no exception was logged while it ran.
    """

    def __init__(self, driver, url: str, out: Path, download_dir: Path,
                 setup: Path, window: tuple = (WIDTH, HEIGHT),
                 settle: float = 1.5, timeout: float = 180.0,
                 server_log: Optional[Path] = None) -> None:
        self.driver = driver
        self.server_log = server_log
        self.url = url
        self.out = out
        self.download_dir = download_dir
        self.setup = setup
        self.window = tuple(window)
        self.settle = settle
        self.timeout = timeout
        self.results: List[dict] = []
        #: Screenshots in which a toolbar had moved tools into its overflow
        #: menu, with the number of such toolbars.
        self.toolbar_overflows: List[str] = []
        #: Pages that scrolled horizontally, with their widths.
        self.horizontal_overflows: List[str] = []
        #: Pages whose horizontal overflow was measured.
        self.overflow_pages: set = set()
        #: Looks of "PALC Run" seen while a setup was loaded.
        self.load_looks: List[tuple] = []

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
        """Wait until the page (its elements, also in shadow roots, and
        its size) stays the same and BokehJS is idle for :attr:`settle`
        seconds.

        Returns
        -------
        None.

        Raises
        ------
        TimeoutError
            If the page does not settle within the timeout.
        """
        last, since = None, time.time()
        end = time.time() + self.timeout
        while time.time() < end:
            state = self.js("""
                const m = mainArea();
                return [deep('*').length, m ? m.scrollHeight : 0,
                        m ? m.scrollWidth : 0, isIdle()];""")
            if state != last or not state[-1]:
                last, since = state, time.time()
            elif time.time() - since >= self.settle:
                return
            time.sleep(0.25)
        raise TimeoutError('the page did not settle')

    def active_page(self) -> int:
        """Index of the page whose tab is selected.

        Returns
        -------
        int
            Index of the selected page tab (0: Help), -1 if none.
        """
        return self.js("""
            return arguments[0].findIndex(label => {
                const t = tab(label);
                return t && t.getAttribute('aria-selected') === 'true';
            });""", list(PAGE_LABELS))

    def show_page(self, index: int) -> None:
        """Show a page by clicking its tab and wait until it is drawn.

        Parameters
        ----------
        index : int
            Index of the page (0: Help).

        Returns
        -------
        None.
        """
        if self.active_page() != index:
            if not self.js("""const t = tab(arguments[0]);
                              if (t) t.click(); return !!t;""",
                           PAGE_LABELS[index]):
                raise RuntimeError('no page tab %r' % PAGE_LABELS[index])
            self.wait_for(lambda: self.active_page() == index,
                          'page %r' % PAGE_LABELS[index], timeout=60)
        self.js("const m = mainArea(); if (m) m.scrollTop = 0;")
        self.wait_settled()

    def show_tab(self, label: str) -> None:
        """Select a tab inside a page (e.g. "Beam plots") and wait.

        Parameters
        ----------
        label : str
            Label of the tab.

        Returns
        -------
        None.
        """
        if not self.js("""const t = tab(arguments[0]);
                          if (t && t.getAttribute('aria-selected') !== 'true')
                              t.click();
                          return !!t;""", label):
            raise RuntimeError('no tab %r' % label)
        self.wait_settled()

    def click(self, label: str) -> None:
        """Click the shown button with this label.

        Parameters
        ----------
        label : str
            Label of the button (without its icon).

        Returns
        -------
        None.

        Raises
        ------
        RuntimeError
            If no such button is shown.
        """
        if not self.js("""const b = button(arguments[0]);
                          if (b) b.click(); return !!b;""", label):
            raise RuntimeError('no button %r' % label)

    def enter(self, label: str, value: str) -> None:
        """Type a value into the shown input with this label and press
        Enter, as a user does.

        Parameters
        ----------
        label : str
            Label of the input.
        value : str
            New text.

        Returns
        -------
        None.
        """
        from selenium.webdriver.common.keys import Keys
        element = self.input_element(label)
        element.click()
        element.send_keys(Keys.CONTROL, 'a')
        element.send_keys(value, Keys.ENTER)

    def input_element(self, label: str):
        """The shown input with this label, scrolled into view.

        Parameters
        ----------
        label : str
            Label of the input.

        Returns
        -------
        selenium.webdriver.remote.webelement.WebElement

        Raises
        ------
        RuntimeError
            If no such input is shown.
        """
        element = self.js("""
            for (const e of deep('.MuiFormControl-root')) {
                const l = e.querySelector('label');
                if (l && visible(e) && l.textContent.trim() === arguments[0])
                    return e.querySelector('input');
            }
            return null;""", label)
        if element is None:
            raise RuntimeError('no input %r' % label)
        self.js("arguments[0].scrollIntoView({block: 'center'});", element)
        return element

    def type_and_click(self, label: str, value: str, button: str) -> None:
        """Type a value into an input and at once click a button with the
        mouse, without Enter, as a fast user does.

        The mouse button is pressed on the button and released at once:
        the press leaves the input, which sends the value to the server
        right before the click; the server still waits for its debounce
        (50 ms) when the click arrives. (Selenium's ``WebElement.click``
        is not used: in Firefox the browser then sends the click before
        the value, which no real click does.)

        Parameters
        ----------
        label : str
            Label of the input.
        value : str
            New text.
        button : str
            Label of the button.

        Returns
        -------
        None.
        """
        from selenium.webdriver.common.action_chains import ActionChains
        from selenium.webdriver.common.keys import Keys
        target = self.js('const b = button(arguments[0]); if (b) '
                         'b.scrollIntoView({block: "center"}); return b;',
                         button)
        if target is None:
            raise RuntimeError('no button %r' % button)
        element = self.input_element(label)
        element.click()
        element.send_keys(Keys.CONTROL, 'a')
        element.send_keys(value)
        # no wait: within the debounce
        ActionChains(self.driver, duration=0).move_to_element(target) \
            .click_and_hold().release().perform()

    def step_slider(self, label: str, key: str) -> str:
        """Move the shown slider with this label one step with an arrow
        key, as a user does with the keyboard.

        Parameters
        ----------
        label : str
            Label of the slider (its ``aria-label``).
        key : str
            ``'up'`` or ``'down'``.

        Returns
        -------
        tuple of str
            The value of the slider before and after (``aria-valuenow``).
        """
        from selenium.webdriver.common.action_chains import ActionChains
        from selenium.webdriver.common.keys import Keys
        find = """return deep('input[type=range]').find(
                        e => e.getAttribute('aria-label') === arguments[0]
                             && visible(e.closest('.MuiSlider-root'))
                             && !e.disabled) || null;"""
        element = self.js(find, label)
        if element is None:
            raise RuntimeError('no enabled slider %r' % label)
        before = element.get_attribute('aria-valuenow')
        self.js("arguments[0].scrollIntoView({block: 'center'}); "
                "arguments[0].focus();", element)
        ActionChains(self.driver).send_keys(
            Keys.ARROW_RIGHT if key == 'up' else Keys.ARROW_LEFT).perform()
        time.sleep(0.5)
        return before, self.js(find, label).get_attribute('aria-valuenow')

    def set_axis_mode(self, mode: int) -> None:
        """Select an axis mode of the venue figure on the shown page.

        Parameters
        ----------
        mode : int
            Index in :data:`~pypalc.gui.app.AXIS_MODES`.

        Returns
        -------
        None.
        """
        self.click(AXIS_MODES[mode])
        time.sleep(0.5)
        self.wait_settled()

    def page_widths(self) -> list:
        """Scroll width and visible width of the document and of the main
        area.

        Returns
        -------
        list
            ``[[scrollWidth, innerWidth], [scrollWidth, clientWidth]]``;
            the second entry is missing without a main area.
        """
        return self.js("""
            const m = mainArea();
            const out = [[document.documentElement.scrollWidth,
                          window.innerWidth]];
            if (m) out.push([m.scrollWidth, m.clientWidth]);
            return out;""")

    def record_page(self, name: str) -> None:
        """Record the toolbars with an overflow menu and a horizontal
        overflow of the page shown.

        Parameters
        ----------
        name : str
            Name of the view (screenshot file or page and tab).

        Returns
        -------
        None.
        """
        overflows = self.js("""
            return deep('.bk-tool-overflow').filter(
                e => e.getBoundingClientRect().height > 0).length;""")
        if overflows:
            self.toolbar_overflows.append('%s: %d' % (name, overflows))
        widths = self.page_widths()
        self.overflow_pages.add(PAGE_NAMES[self.active_page()])
        if any(scroll > shown for scroll, shown in widths):
            self.horizontal_overflows.append('%s: %s' % (name, ', '.join(
                '%d/%d px' % tuple(w) for w in widths)))

    def screenshot(self, index: int, state: str, whole: bool = True) -> Path:
        """Save a screenshot as ``OUT/<page>_<state>.png``.

        Records the toolbars and the horizontal overflow of the page first
        (:meth:`record_page`). For the whole page the window is made as
        high as the page while the screenshot is taken.

        Parameters
        ----------
        index : int
            Index of the page shown.
        state : str
            State of the app (:data:`SCREENSHOT_STATES`).
        whole : bool, optional
            Show the whole page (default); False: the window only.

        Returns
        -------
        Path
            The PNG file.
        """
        if state not in SCREENSHOT_STATES:  # --clean must know the file
            raise ValueError('unknown screenshot state %r' % state)
        path = self.out / ('%s_%s.png' % (PAGE_NAMES[index], state))
        self.record_page(path.name)
        hidden = self.js("""const m = mainArea();
                            return m ? m.scrollHeight - m.clientHeight : 0;""")
        if not whole or hidden <= 0:
            self.driver.save_screenshot(str(path))
            return path
        width, height = self.window
        try:
            for _ in range(3):  # the page may grow a little with the window
                height += hidden
                self.driver.set_window_size(width, height)
                time.sleep(0.5)
                self.wait_settled()
                hidden = self.js("""const m = mainArea();
                    return m ? m.scrollHeight - m.clientHeight : 0;""")
                if hidden <= 0:
                    break
            self.driver.save_screenshot(str(path))
        finally:
            self.driver.set_window_size(*self.window)
            time.sleep(0.5)
            self.wait_settled()
        return path

    def wait_download(self, name: str) -> Path:
        """Wait until the browser has saved a download.

        Parameters
        ----------
        name : str
            File name of the download.

        Returns
        -------
        Path
            The downloaded file, once it exists without a ``.part`` file
            and its size stayed the same over two polls.
        """
        path = self.download_dir / name
        part = self.download_dir / (name + '.part')
        sizes = []

        def complete():
            if not path.exists() or part.exists():
                sizes.clear()
                return False
            sizes.append(path.stat().st_size)
            return len(sizes) >= 2 and sizes[-1] == sizes[-2] > 0
        self.wait_for(complete, 'download ' + name, timeout=30)
        return path

    def run_look(self) -> Optional[dict]:
        """Label, background and state of "PALC Run" in the browser.

        Returns
        -------
        dict or None
            ``{"label", "background", "disabled"}``; None if the toggle is
            not shown.
        """
        return self.js("""
            const b = deep('button').find(e => visible(e) &&
                [arguments[0], arguments[1]].indexOf(text(e)) >= 0);
            if (!b) return null;
            return {label: text(b),
                    background: getComputedStyle(b).backgroundColor,
                    disabled: b.disabled ||
                              b.classList.contains('Mui-disabled')};""",
                       RUN_LABEL, RUNNING_LABEL)

    def is_running_look(self, look: Optional[dict]) -> bool:
        """True if "PALC Run" shows the running look.

        Parameters
        ----------
        look : dict or None
            Result of :meth:`run_look`.

        Returns
        -------
        bool
        """
        return bool(look) and look['label'] == RUNNING_LABEL and \
            look['background'] == RUNNING_BACKGROUND

    def venue_state(self) -> Optional[dict]:
        """Measure the venue figure shown (JavaScript ``plotState``).

        Returns
        -------
        dict or None
        """
        return self.js("return plotState(venueView());")

    def beam_states(self) -> list:
        """Measure the two beam plots (JavaScript ``plotState``).

        Returns
        -------
        list of dict or None
        """
        return self.js(
            "return arguments[0].map(t => plotState(figureView(t)));",
            BEAM_TITLES)

    def show_beam_plots(self) -> None:
        """Show page 5 with both beam plots and wait until both are drawn.

        Returns
        -------
        None.
        """
        self.show_page(5)
        self.show_tab(BEAM_TAB)
        for label in BEAM_BUTTONS:
            pressed = self.js("""
                const b = button(arguments[0]);
                return b ? b.getAttribute('aria-pressed') : null;""", label)
            if pressed is None:
                raise RuntimeError('no button %r' % label)
            if pressed != 'true':
                self.click(label)
        self.wait_for(lambda: all(s and s['visible'] and s['width'] > 0
                                  for s in self.beam_states()),
                      'both beam plots', timeout=60)
        self.wait_settled()

    # ------------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------------

    def check_fresh_screenshots(self) -> tuple:
        """Screenshot of every page of a new session (state ``fresh``).

        Returns
        -------
        tuple
            ``(ok, detail)``.
        """
        files = []
        for index in range(len(PAGE_LABELS)):
            self.show_page(index)
            files.append(self.screenshot(index, 'fresh').name)
        return True, ', '.join(files)

    def check_load(self) -> tuple:
        """Load the setup through the "Load Setup" file input of page 4.

        Gives the file input the setup file, as a user does by choosing it,
        and records the look of "PALC Run" while the setup loads. The app
        must switch to page 5; page 4 must then show the gap handling, the
        weighting and the number of hinges of the setup and "PALC Run"
        ready again.

        Returns
        -------
        tuple
            ``(ok, detail)``.
        """
        setup = load_setup(self.setup)
        expected = {
            '19. Gap handling':
                GAP_HANDLING_LABELS[setup.algorithm.gap_handling],
            '21. Weighting': WEIGHTING_LABELS[setup.algorithm.weighting]}
        if setup.target_slope is not None:
            expected['Hinges'] = str(len(setup.target_slope.points) - 2)
        self.show_page(4)
        inputs = self.js("""return deep('input[type=file]').filter(
            e => e.accept.indexOf('.yaml') >= 0);""")
        if len(inputs) != 1:
            return False, '%d setup file inputs found' % len(inputs)
        start = time.time()
        inputs[0].send_keys(str(self.setup))
        end = start + self.timeout
        while self.active_page() != 5:
            look = self.run_look()
            if look is not None:
                self.load_looks.append((look['label'], look['background']))
            if time.time() > end:
                alerts = self.js("""return deep('.MuiAlert-message').map(
                    e => e.textContent.trim()).filter(t => t);""")
                return False, 'app did not switch to page 5 within %ds; ' \
                    'page %s; alerts %r' % (self.timeout, self.active_page(),
                                            alerts)
            time.sleep(0.05)
        self.wait_settled()
        seconds = time.time() - start
        self.screenshot(5, 'loaded')
        self.show_page(4)
        shown = self.js('return fields();')
        look = self.run_look()
        problems = []
        for key, value in expected.items():
            if shown.get(key) != value:
                problems.append('%s: shows %r, expected %r'
                                % (key, shown.get(key), value))
        if not look or look['label'] != RUN_LABEL or look['disabled']:
            problems.append('"PALC Run" not ready: %r' % look)
        self.screenshot(4, 'loaded')
        detail = 'switched to page 5 after %.1fs; page 4 shows %s; ' \
            '"PALC Run" %r' % (seconds, ', '.join(
                '%s=%r' % (k, shown.get(k)) for k in expected), look)
        if problems:
            detail += '; ' + '; '.join(problems)
        return not problems, detail

    def check_loaded_screenshots(self) -> tuple:
        """Screenshots of the other pages after loading (state
        ``loaded``); pages 5 and 4 were taken by the load check.

        Returns
        -------
        tuple
            ``(ok, detail)``.
        """
        files = []
        for index in LOADED_ORDER[2:]:
            self.show_page(index)
            files.append(self.screenshot(index, 'loaded').name)
        return True, ', '.join(files)

    def check_download_angles(self) -> tuple:
        """"Download PALC Angles" saves the tilt angles once, with the
        header of the headless export and a row per cabinet.

        Returns
        -------
        tuple
            ``(ok, detail)``.
        """
        self.show_page(4)
        self.click(DOWNLOAD_ANGLES_LABEL)
        path = self.wait_download(ANGLES_FILE)
        lines = path.read_text(encoding='utf-8').splitlines()
        header = ','.join(DOWNLOAD_COLUMNS['results'])
        files = sorted(p.name for p in downloaded_files(self.download_dir,
                                                        ANGLES_FILE))
        detail = '%s: header %r, %d rows; downloaded files: %s' % (
            path.name, lines[0], len(lines) - 1, ', '.join(files))
        return lines[0] == header and len(lines) > 1 and \
            files == [ANGLES_FILE], detail

    def check_download_results(self) -> tuple:
        """"Download Results" saves one zip file with the five CSV files of
        :data:`RESULT_FILES`, each with its header in its order and at
        least one row.

        Returns
        -------
        tuple
            ``(ok, detail)``.
        """
        self.show_page(4)
        self.click(DOWNLOAD_RESULTS_LABEL)
        path = self.wait_download(RESULTS_FILE_NAME)
        problems, found = [], []
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if sorted(names) != sorted(RESULT_FILES):
                problems.append('files %s, expected %s'
                                % (names, sorted(RESULT_FILES)))
            for name in names:
                rows = list(csv.reader(io.StringIO(
                    archive.read(name).decode('utf-8'))))
                found.append('%s %r (%d rows)' % (name, ','.join(rows[0]),
                                                 len(rows) - 1))
                if name in RESULT_FILES and rows[0] != RESULT_FILES[name]:
                    problems.append('%s: expected header %r' % (
                        name, ','.join(RESULT_FILES[name])))
                if len(rows) < 2:
                    problems.append('%s: no rows' % name)
        copies = sorted(p.name for p in downloaded_files(
            self.download_dir, RESULTS_FILE_NAME) if p.name != path.name)
        if copies:
            problems.append('downloaded more than once: %s'
                            % ', '.join(copies))
        return not problems, '%s: %s' % (path.name,
                                         '; '.join(found + problems))

    def check_download_setup(self) -> tuple:
        """"Save Setup" saves the setup file once (Firefox would save a
        second download as ``pypalc_setup(1).yaml``), a valid setup with
        the weighting, the gap handling, the number of audience lines and
        the number of target slope points of the loaded setup.

        Returns
        -------
        tuple
            ``(ok, detail)``.
        """
        self.show_page(4)
        self.click(SAVE_SETUP_LABEL)
        path = self.wait_download(SETUP_FILE_NAME)
        saved = load_setup_text(path.read_text(encoding='utf-8'))
        loaded = load_setup(self.setup)

        def summary(setup):
            return (setup.algorithm.weighting, setup.algorithm.gap_handling,
                    len(setup.venue.audience_lines),
                    None if setup.target_slope is None
                    else len(setup.target_slope.points))
        got, expected = summary(saved), summary(loaded)
        files = sorted(p.name for p in downloaded_files(self.download_dir,
                                                        SETUP_FILE_NAME))
        detail = '%s: valid setup, weighting %s, gap handling %s, %d ' \
            'audience lines, target slope points %s' % ((path.name,) + got)
        detail += '; downloaded files: %s' % ', '.join(files)
        if got != expected:
            detail += '; expected %s, %s, %d, %s' % expected
        return got == expected and files == [SETUP_FILE_NAME], detail

    def check_running_indicator(self) -> tuple:
        """"PALC Run" is dark red and reads "Running..." while a
        calculation runs and while a setup loads (seen by the load check),
        and is ready again afterwards; screenshot of page 4 in state
        ``running``.

        Returns
        -------
        tuple
            ``(ok, detail)``.
        """
        self.show_page(4)
        before = self.run_look()
        if before is None:
            return False, 'no toggle %r' % RUN_LABEL
        if before['disabled']:
            return False, 'toggle disabled'
        self.click(RUN_LABEL)
        start = time.time()
        try:
            during = self.wait_for(
                lambda: (lambda look: look if self.is_running_look(look)
                         else None)(self.run_look()),
                'the running look', timeout=10, poll=0.05)
        except TimeoutError:
            during = self.run_look()
        self.screenshot(4, 'running', whole=False)
        still = self.is_running_look(self.run_look())
        self.wait_for(lambda: (self.run_look() or {}).get('label')
                      == RUN_LABEL, 'the end of the calculation')
        self.wait_settled()
        after = self.run_look()
        loading = sorted(set(self.load_looks))
        detail = 'during the run: %r (still running at the screenshot: ' \
            '%s); after %.1fs: %r; before: %r; while loading: %s' % (
                during, still, time.time() - start, after, before, loading)
        ok = (self.is_running_look(during) and
              after['label'] == RUN_LABEL and not after['disabled'] and
              after['background'] != RUNNING_BACKGROUND and
              (RUNNING_LABEL, RUNNING_BACKGROUND) in loading)
        return ok, detail

    def check_drag(self) -> tuple:
        """Move a target point in the browser, as the PointDrawTool does,
        and read the server's answer back.

        Selects the middle target point, writes a new distance (a little
        off a grid point) and level into the target source and clears the
        selection. The server must snap the point to the grid of the SPL
        over distance (distance and venue position of the grid point) and
        update the optimization region; both are read back from the shown
        models in the browser after the round trip.

        Returns
        -------
        tuple
            ``(ok, detail)``.
        """
        self.show_page(4)
        self.show_tab('SPL over distance')
        sources = """
            const tool = live('PointDrawTool')[0];
            const target = tool.renderers[0].data_source;
            const grid = liveSources().find(m =>
                Object.keys(m.data).sort().join() ===
                'distance_m,spl_db,x_m,y_m');
            const region = live('GlyphRenderer').find(
                r => typeName(r.glyph) === 'Patch' &&
                     tool.renderers.indexOf(r) < 0 &&
                     'x' in r.data_source.data).data_source;
        """
        state = self.js(sources + """
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
        self.js(sources + """
            const i = arguments[0];
            // tap: select the point
            target.selected.indices = [i];
            // drag: move it in place, then end the drag as the tool's
            // _emit_cds_changes(cds, false, true, true) does
            target.get_array('distance_m')[i] = arguments[1];
            target.get_array('level_db')[i] = arguments[2];
            target.selection_manager.clear();
            target.setv({data: target.data}, {check_eq: false});""",
                point, new_distance, new_level)

        def answered():
            data = self.js(sources + """
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

    def rays(self) -> int:
        """Number of rays in the rays source of the page shown.

        Returns
        -------
        int
        """
        return self.js("""
            const s = liveSources().find(m => 'start_rays_x' in m.data);
            return s ? Array.from(s.data.start_rays_x).length : -1;""")

    def check_rays_cleared(self) -> tuple:
        """Show page 4 with the rays of a result, then page 1, which must
        show no rays (the server clears them when page 1 is shown). Reads
        the rays source in the browser.

        Returns
        -------
        tuple
            ``(ok, detail)``.
        """
        self.show_page(4)
        self.show_tab(VENUE_TAB)
        before = self.rays()
        self.show_page(1)
        after = self.rays()
        detail = ('rays in the browser: %d on page 4, %d on page 1'
                  % (before, after))
        if before <= 0:
            detail += '; no rays on page 4, so nothing was checked'
        return before > 0 and after == 0, detail

    def check_equal_axes(self) -> tuple:
        """Equal scaling: one meter has the same length in px on both
        axes (within :data:`EQUAL_TOLERANCE`) in the venue figure in the
        axis modes "Equal, variable height" and "Equal, fixed height" (pages
        1 to 4 and page 1 again) and in both beam plots (page 5, again
        after page 1 was shown). On pages 2 and 4, whose venue figure is in
        a tab, also after a round trip to the other tab during which an
        input gave the figure new ranges (:data:`HIDDEN_VENUE_INPUTS`; the
        input is set back with a second round trip, and the check fails if
        the ranges were not renewed).

        The figure is sized, its ranges are not widened: in the equal modes
        the ranges are those of "Stretched" on the same page (the data
        extent; measured first on pages 1 to 4, where the figure must be as
        wide as its container). In "Equal, variable height" the figure is
        as wide as its container (as in "Stretched") unless it has the
        largest height (:func:`~pypalc.gui.ranges.window_limit` of the
        window; it may be higher than in "Stretched"); in "Equal, fixed
        height" it is as high as in "Stretched" (at most that window limit)
        unless it is as wide as its container. In both it ends inside the
        window, and a figure narrower than its container is centered in it.
        The beam plots are sized as "Equal, variable height" with a
        largest height of at most
        :data:`~pypalc.gui.ranges.VENUE_MAX_HEIGHT`, their ranges are the
        extent of their image and the image fills the frame (no gap of more
        than :data:`IMAGE_GAP` px). Screenshots of page 1 in every
        axis mode (states ``axes_0`` to ``axes_2``); the mode is
        "Stretched" again afterwards.

        Returns
        -------
        tuple
            ``(ok, detail)``.
        """
        found, problems = [], []
        stretched = {}

        def close(a, b):
            return all(p is not None and q is not None and
                       math.isclose(p, q, rel_tol=1e-9, abs_tol=1e-9)
                       for p, q in zip(a, b))

        def sized(name, state, mode, page, extent, largest=None):
            """The size, the position and the ranges of an equally scaled
            figure."""
            near = lambda a, b: abs(a - b) <= SIZE_TOLERANCE  # noqa: E731
            width, height = state['outer_width'], state['outer_height']
            available = state['available']
            left, right = state['gaps']
            if width < available - SIZE_TOLERANCE and \
                    abs(left - right) > SIZE_TOLERANCE:
                problems.append('%s: not centered (%d px left, %d px '
                                'right)' % (name, left, right))
            if extent:
                if not close(state['x'] + state['y'],
                             stretched[page]['x'] + stretched[page]['y']):
                    problems.append('%s: ranges %s are not the data extent '
                                    '%s' % (name, _fmt(state['x'] + state['y']),
                                            _fmt(stretched[page]['x'] +
                                                 stretched[page]['y'])))
            if mode == 0:
                limit = window_limit(state['window_height'])
                if largest is not None:
                    limit = min(limit, largest)
                capped = near(height, limit)
                if not (near(width, available) or capped) or \
                        width > available + SIZE_TOLERANCE or \
                        height > limit + SIZE_TOLERANCE:
                    problems.append(
                        '%s: %dx%d px, not as wide as its container (%d px) '
                        'and not the largest height (%d px)'
                        % (name, width, height, available, limit))
                return 'width %s' % ('capped by the height %d px' % limit
                                     if capped and not near(width, available)
                                     else 'of its container %d px'
                                     % available)
            high = min(stretched[page]['outer_height'],
                       window_limit(state['window_height']))
            capped = near(width, available)
            if not (near(height, high) or capped) or \
                    height > high + SIZE_TOLERANCE or \
                    width > available + SIZE_TOLERANCE:
                problems.append(
                    '%s: %dx%d px, not as high as in Stretched (%d px) and '
                    'not as wide as its container (%d px)'
                    % (name, width, height, high, available))
            return 'height %s' % ('capped by the width %d px' % available
                                  if capped and not near(height, high)
                                  else 'of Stretched %d px' % high)

        def measure(name, state, mode=None, page=None, bottom=False,
                    extent=True, largest=None):
            if not state or not state['visible']:
                problems.append('%s: figure not shown' % name)
                return
            x, y = pixels_per_meter(state)
            diff = equal_difference(state)
            size = sized(name, state, mode, page, extent, largest) \
                if mode is not None else ''
            found.append('%s %dx%d px (frame %dx%d), %.2f/%.2f px/m '
                         '(%.2f %%), %s%s' % (
                             name, state['outer_width'],
                             state['outer_height'], state['width'],
                             state['height'], x, y, 100 * diff, size,
                             ', bottom %d of %d px' % (
                                 state['bottom'], state['window_height'])
                             if bottom else ''))
            if not diff < EQUAL_TOLERANCE:
                problems.append('%s: px/m differ by %.2f %%'
                                % (name, 100 * diff))
            if bottom and state['bottom'] > state['window_height']:
                problems.append('%s: figure ends below the window' % name)

        def measure_beam(name, state):
            measure(name, state, mode=0, extent=False,
                    largest=VENUE_MAX_HEIGHT)
            if not state or not state['visible']:
                return
            image = state['image']
            if image is None or not close(state['x'] + state['y'], image):
                problems.append('%s: ranges %s are not the image extent %s'
                                % (name, _fmt(state['x'] + state['y']),
                                   image and _fmt(image)))
            elif state['image_gap'] > IMAGE_GAP:
                problems.append('%s: %.1f px between image and frame'
                                % (name, state['image_gap']))
            else:
                found.append('%s: ranges = image extent %s, gap %.2g px'
                             % (name, _fmt(image), state['image_gap']))

        def measure_stretched(page):
            state = self.venue_state()
            if not state or not state['visible']:
                problems.append('page %d Stretched: figure not shown' % page)
                return
            stretched[page] = state
            found.append('page %d Stretched %dx%d px in %d px, %s' % (
                page, state['outer_width'], state['outer_height'],
                state['available'], _fmt(state['x'] + state['y'])))
            if abs(state['outer_width'] - state['available']) > \
                    SIZE_TOLERANCE:
                problems.append('page %d Stretched: %d px wide in %d px'
                                % (page, state['outer_width'],
                                   state['available']))

        def change_hidden(page, mode, step):
            # change the input while the other tab hides the venue figure;
            # step 0 changes it, step 1 sets it back
            tab, label, value = HIDDEN_VENUE_INPUTS[page]
            before = self.venue_state()
            self.show_tab(tab)
            if value is None:
                old, shown = self.step_slider(label, ('up', 'down')[step])
                if step == 0:
                    old_inputs[page] = old
                elif shown != old_inputs[page]:
                    problems.append('page %d: %s not set back to %s (%s)' % (
                        page, label, old_inputs[page], shown))
            else:
                if step == 0:
                    old_inputs[page] = self.js(
                        'return fields()[arguments[0]];', label)
                shown = (value, old_inputs[page])[step]
                self.enter(label, shown)
                time.sleep(0.5)
            self.wait_settled()
            self.show_tab(VENUE_TAB)
            self.js('const m = mainArea(); if (m) m.scrollTop = 0;')
            state = self.venue_state()
            name = 'page %d mode %d %s=%s on tab %r' % (
                page, mode, label.split('.')[0], shown, tab)
            # set back, the ranges are the data extent of Stretched again
            measure(name, state, mode=mode, page=page, bottom=True,
                    extent=step == 1)
            if before and state and \
                    before['range_ids'] == state['range_ids']:
                problems.append('%s: the venue figure got no new ranges, '
                                'so the round trip checked nothing' % name)

        def show_venue(page):
            self.show_page(page)
            if page in HIDDEN_VENUE_INPUTS:
                self.show_tab(VENUE_TAB)

        old_inputs = {}
        self.show_page(1)
        self.set_axis_mode(2)
        for page in (1, 2, 3, 4):
            show_venue(page)
            measure_stretched(page)
        self.show_page(1)
        for mode in (0, 1):
            self.set_axis_mode(mode)
            measure('page 1 mode %d' % mode, self.venue_state(), mode=mode,
                    page=1, bottom=True)
            self.screenshot(1, 'axes_%d' % mode)
            for page in (2, 3, 4):
                show_venue(page)
                measure('page %d mode %d' % (page, mode), self.venue_state(),
                        mode=mode, page=page, bottom=True)
                if page in HIDDEN_VENUE_INPUTS:
                    for step in (0, 1):
                        change_hidden(page, mode, step)
            self.show_page(1)
            measure('page 1 again mode %d' % mode, self.venue_state(),
                    mode=mode, page=1, bottom=True)
        self.set_axis_mode(2)
        self.screenshot(1, 'axes_2')
        self.show_beam_plots()
        for title, state in zip(('PALC', 'reference'), self.beam_states()):
            measure_beam('beam plot %s' % title, state)
        self.screenshot(5, 'beam_plots')
        self.show_page(1)
        self.show_page(5)
        self.wait_for(lambda: all(s and s['visible'] and s['width'] > 0
                                  for s in self.beam_states()),
                      'both beam plots', timeout=60)
        for title, state in zip(('PALC', 'reference'), self.beam_states()):
            measure_beam('beam plot %s after page 1' % title, state)
        return not problems, '; '.join(found + problems)

    def tool_button(self, title: str, tool: str) -> tuple:
        """Scroll a toolbar button of a figure into view.

        Parameters
        ----------
        title : str
            Title of the figure.
        tool : str
            Start of the button's tooltip, e.g. ``'Reset'``.

        Returns
        -------
        tuple of float
            Center of the button in the window.
        """
        return tuple(self.js("""
            const v = views().find(
                v => v.model === figureView(arguments[0]).model.toolbar);
            const root = v.shadow_el || v.el.shadowRoot;
            const b = Array.from(root.querySelectorAll('[title]')).find(
                e => e.title.startsWith(arguments[1]));
            b.scrollIntoView({block: 'center'});
            const r = b.getBoundingClientRect();
            return [r.left + r.width / 2, r.top + r.height / 2];""",
                             title, tool))

    def mouse(self, *points: tuple) -> None:
        """Press the mouse at the first point, move it through the others
        and release it at the last one (a click for one point).

        Parameters
        ----------
        *points : tuple of float
            Points in the window.

        Returns
        -------
        None.
        """
        from selenium.webdriver.common.actions.action_builder import \
            ActionBuilder
        actions = ActionBuilder(self.driver)
        actions.pointer_action.move_to_location(
            int(points[0][0]), int(points[0][1])).pointer_down()
        for x, y in points[1:]:
            actions.pointer_action.move_to_location(int(x), int(y))
        actions.pointer_action.pointer_up()
        actions.perform()
        time.sleep(1)

    def check_beam_link(self) -> tuple:
        """A zoom in one beam plot zooms the other one too, and a reset in
        the other one resets both to the extent of the image: the box zoom
        tool of the PALC plot (mouse drag over a part of its frame), then
        the reset tool of the reference plot, both clicked in their
        toolbars.

        Returns
        -------
        tuple
            ``(ok, detail)``.
        """
        self.show_beam_plots()

        def ranges():
            return [s['x'] + s['y'] for s in self.beam_states()]

        def same(a, b):
            return all(p is not None and q is not None and
                       math.isclose(p, q, rel_tol=1e-9, abs_tol=1e-9)
                       for p, q in zip(a, b))
        start = ranges()
        self.mouse(self.tool_button(BEAM_TITLES[0], 'Box Zoom'))
        x, y, w, h = self.js("""
            const v = figureView(arguments[0]);
            v.el.scrollIntoView({block: 'center'});
            const r = v.el.getBoundingClientRect(), b = v.frame.bbox;
            return [r.left + b.left, r.top + b.top, b.width, b.height];""",
                             BEAM_TITLES[0])
        self.mouse(*[(x + (0.3 + 0.06 * k) * w, y + (0.3 + 0.04 * k) * h)
                     for k in range(6)])
        zoomed = ranges()
        self.mouse(self.tool_button(BEAM_TITLES[1], 'Reset'))
        after = ranges()
        detail = 'x/y ranges PALC %s, reference %s; after a box zoom in ' \
            'PALC: %s, %s; after reset in reference: %s, %s' % (
                _fmt(start[0]), _fmt(start[1]), _fmt(zoomed[0]),
                _fmt(zoomed[1]), _fmt(after[0]), _fmt(after[1]))
        width = [r[1] - r[0] for r in (start[0], zoomed[0])]
        image = [s['image'] for s in self.beam_states()]
        detail += '; image extent %s' % ', '.join(
            'none' if i is None else _fmt(i) for i in image)
        ok = (width[1] < 0.9 * width[0] and same(zoomed[0], zoomed[1])
              and same(after[0], start[0]) and same(after[1], start[1])
              and all(i is not None and same(a, i)
                      for a, i in zip(after, image)))
        return ok, detail

    def check_toolbars(self) -> tuple:
        """No toolbar hides tools in an overflow menu: checks the
        screenshots so far and every tab of pages 4 and 5 (screenshots of
        the further tabs of page 5 in states ``tabs_1`` to ``tabs_3``).

        Returns
        -------
        tuple
            ``(ok, detail)``.
        """
        for index, labels in PAGE_TABS.items():
            self.show_page(index)
            for k, label in enumerate(labels):
                self.show_tab(label)
                if index == 5 and k:
                    self.screenshot(5, 'tabs_%d' % k)
                else:
                    self.record_page('%s, tab %s' % (PAGE_NAMES[index],
                                                     label))
        if self.toolbar_overflows:
            return False, ('toolbars with an overflow menu in ' +
                           ', '.join(self.toolbar_overflows))
        return True, 'every tool shown in every screenshot and every tab ' \
            'of pages 4 and 5'

    def check_axis_labels(self) -> tuple:
        """The venue figure keeps its tick labels and axis titles (space
        of at least :data:`AXIS_SPACE` px left of and below its frame)
        after a change of the axis mode and, in an equal mode, after an
        input on pages 2 and 3 that gives the figure new ranges. Sets the
        inputs back and the mode to "Stretched" afterwards.

        Returns
        -------
        tuple
            ``(ok, detail)``.
        """
        found, problems = [], []

        def measure(name):
            state = self.venue_state()
            if not state or not state['visible']:
                problems.append('%s: figure not shown' % name)
                return
            found.append('%s: %d px left, %d px below' % (
                name, state['left'], state['below']))
            if state['left'] < AXIS_SPACE or state['below'] < AXIS_SPACE:
                problems.append('%s: axis labels missing' % name)

        self.show_page(1)
        measure('page 1 Stretched')
        for mode in (1, 0, 1):
            self.set_axis_mode(mode)
            measure('page 1 mode %d' % mode)
        for index, (label, value) in CONFIG_INPUTS.items():
            self.show_page(index)
            old = self.js('return fields()[arguments[0]];', label)
            for text in (value, old):
                self.enter(label, text)
                time.sleep(0.5)
                self.wait_settled()
                measure('page %d %s=%s' % (index, label.split('.')[0], text))
        self.set_axis_mode(2)
        measure('page %d Stretched' % index)
        return not problems, '; '.join(found + problems)

    def check_no_horizontal_overflow(self) -> tuple:
        """No page scrolled horizontally (document or main area) in any
        screenshot or tab so far.

        Returns
        -------
        tuple
            ``(ok, detail)``.
        """
        missing = sorted(set(PAGE_NAMES) - self.overflow_pages)
        if self.horizontal_overflows:
            return False, 'horizontal overflow: ' + \
                '; '.join(self.horizontal_overflows)
        if missing:
            return False, 'pages not measured: ' + ', '.join(missing)
        return True, 'no horizontal overflow at %d px on %s' % (
            self.window[0], ', '.join(sorted(self.overflow_pages)))

    def line_counts(self) -> List[int]:
        """Number of lines in each venue line source of the page shown.

        Returns
        -------
        list of int
            One count per source with the columns ``lines_x`` and
            ``lines_y`` (drawn audience lines, non-audience lines, lines of
            the sound field only), sorted.
        """
        return sorted(self.js("""
            return liveSources().filter(m =>
                Object.keys(m.data).sort().join() === 'lines_x,lines_y')
                .map(m => Array.from(m.data.lines_x).length);"""))

    def open_window(self) -> str:
        """Open the app in a new browser window (a new session) and show
        page 1.

        Returns
        -------
        str
            Handle of the window, which stays the current one.
        """
        self.driver.switch_to.new_window('window')
        self.driver.set_window_size(*self.window)
        self.open_app()
        self.show_page(1)
        return self.driver.current_window_handle

    def check_two_tabs(self) -> tuple:
        """Two browser windows on one server have their own sessions.

        Opens the app in two new windows, enters an audience line on page 1
        of the first one (the venue inputs and "Save Current Audience
        Line") and waits until its line source shows the line. The second
        window must still show no line and the default inputs, and a third
        window opened afterwards must start without a line. The windows are
        closed afterwards.

        Returns
        -------
        tuple
            ``(ok, detail)``.
        """
        main_window = self.driver.current_window_handle
        windows = []
        try:
            windows.append(self.open_window())
            windows.append(self.open_window())
            self.driver.switch_to.window(windows[0])
            for label, value in TWO_TABS_LINE.items():
                self.enter(label, value)
            self.wait_settled()
            entered = self.js('return fields();')
            self.click(SAVE_LINE_LABEL)
            self.wait_for(lambda: sum(self.line_counts()) > 0,
                          'the line in window 1', timeout=30)
            self.wait_settled()
            first = self.line_counts()
            line_x = self.js("""
                const s = liveSources().find(m =>
                    Object.keys(m.data).sort().join() === 'lines_x,lines_y'
                    && Array.from(m.data.lines_x).length > 0);
                const x = Array.from(s.data.lines_x[0]);
                return [x[0], x[x.length - 1]];""")
            self.driver.switch_to.window(windows[1])
            self.wait_settled()
            second = self.line_counts()
            start_x = self.js('return fields()[arguments[0]];',
                              '1. Start x in m')
            windows.append(self.open_window())
            third = self.line_counts()
        finally:
            # also a window whose app did not load (not yet in windows)
            for handle in self.driver.window_handles:
                if handle != main_window:
                    self.driver.switch_to.window(handle)
                    self.driver.close()
            self.driver.switch_to.window(main_window)
        detail = 'inputs of window 1 %s; lines per venue source: window 1 ' \
            '%s (line x %s..%s), window 2 %s (start x input %r), new ' \
            'window 3 %s' % (', '.join('%s=%s' % (k, entered.get(k))
                                       for k in TWO_TABS_LINE),
                             first, line_x[0], line_x[1], second, start_x,
                             third)
        ok = (first == [0, 0, 1] and line_x == [2, 10] and
              second == [0, 0, 0] and start_x == '0' and third == [0, 0, 0])
        return ok, detail

    def server_log_size(self) -> Optional[int]:
        """Size of the server log in bytes, None without a log."""
        if self.server_log is None:
            return None
        return self.server_log.stat().st_size

    def server_log_errors(self, start: Optional[int]) -> List[str]:
        """Lines of the server log written since ``start`` (a size of
        :meth:`server_log_size`) that show an exception."""
        if self.server_log is None or start is None:
            return []
        with open(self.server_log, 'rb') as log:
            log.seek(start)
            text = log.read().decode('utf-8', 'replace')
        return [line.strip() for line in text.splitlines()
                if any(error in line for error in SERVER_LOG_ERRORS)]

    def check_fast_input(self) -> tuple:
        """A value typed right before a click is not lost.

        Opens the app in a new window, enters the line of
        :data:`TWO_TABS_LINE` but its end y, then for each value of
        :data:`FAST_INPUT_VALUES` types the value into "5. End y in m" and
        at once clicks "Save Current Audience Line" with the mouse
        (:meth:`type_and_click`), waits for the saved line and removes it
        again. Each saved line must end at the typed y. With a server log
        (``--server-log``), no exception may be logged meanwhile. The
        window is closed afterwards.

        Returns
        -------
        tuple
            ``(ok, detail)``.
        """
        main_window = self.driver.current_window_handle
        start = self.server_log_size()
        saved = []
        try:
            self.open_window()
            for label, value in TWO_TABS_LINE.items():
                if label != FAST_INPUT_LABEL:
                    self.enter(label, value)
            self.wait_settled()
            for value in FAST_INPUT_VALUES:
                self.type_and_click(FAST_INPUT_LABEL, value, SAVE_LINE_LABEL)
                try:
                    self.wait_for(lambda: sum(self.line_counts()) > 0,
                                  'the saved line', timeout=10)
                except TimeoutError:
                    saved.append(None)
                    continue
                saved.append(self.js("""
                    const s = liveSources().find(m =>
                        Object.keys(m.data).sort().join() === 'lines_x,lines_y'
                        && Array.from(m.data.lines_y).length > 0);
                    const y = Array.from(s.data.lines_y[0]);
                    return y[y.length - 1];"""))
                self.click(REMOVE_LINE_LABEL)
                self.wait_for(lambda: sum(self.line_counts()) == 0,
                              'the removed line', timeout=10)
                self.wait_settled()
        finally:
            for handle in self.driver.window_handles:
                if handle != main_window:
                    self.driver.switch_to.window(handle)
                    self.driver.close()
            self.driver.switch_to.window(main_window)
        errors = self.server_log_errors(start)
        typed = [float(value) for value in FAST_INPUT_VALUES]
        ok = saved == typed and not errors
        detail = 'typed end y %s, saved line ends at y %s; server log: %s' % (
            ', '.join(FAST_INPUT_VALUES),
            ', '.join('none' if y is None else '%g' % y for y in saved),
            'not given' if self.server_log is None else
            '%d error lines%s' % (len(errors), (' (%s)' % errors[0])
                                  if errors else ''))
        return ok, detail

    def open_app(self) -> str:
        """Load the app in the current window and wait until it is drawn.

        Returns
        -------
        str
            The version of BokehJS.
        """
        self.driver.get(self.url)
        self.wait_for(lambda: self.driver.execute_script(
            "return !!(window.Bokeh && Bokeh.documents && "
            "Bokeh.documents.length);"), 'BokehJS', timeout=60)
        self.wait_for(lambda: self.active_page() >= 0, 'the page tabs',
                      timeout=60)
        self.wait_settled()
        return self.driver.execute_script('return Bokeh.version;')

    def run(self, two_tabs: bool = False) -> List[dict]:
        """Open the app and run all checks.

        Parameters
        ----------
        two_tabs : bool, optional
            Also run :meth:`check_two_tabs` (check ``two_tabs``, last).

        Returns
        -------
        list of dict
            One entry ``{"name", "ok", "detail"}`` per check.
        """
        version = self.open_app()
        print('BokehJS %s at %s, window %dx%d' % ((version, self.url)
                                                  + self.window), flush=True)
        self.run_check('screenshots_fresh', self.check_fresh_screenshots)
        after_load = [
            ('screenshots_loaded', self.check_loaded_screenshots),
            ('download_angles', self.check_download_angles),
            ('download_results', self.check_download_results),
            ('download_setup', self.check_download_setup),
            ('running_indicator', self.check_running_indicator),
            ('drag', self.check_drag),
            ('rays_cleared', self.check_rays_cleared),
            ('equal_axes', self.check_equal_axes),
            ('beam_link', self.check_beam_link),
            ('toolbars', self.check_toolbars),
            ('axis_labels', self.check_axis_labels)]
        if self.run_check('load', self.check_load):
            for name, check in after_load:
                self.run_check(name, check)
        else:
            for name, _ in after_load:
                self.add(name, False, 'skipped: the setup did not load')
        self.run_check('no_horizontal_overflow',
                       self.check_no_horizontal_overflow)
        self.run_check('fast_input', self.check_fast_input)
        if two_tabs:
            self.run_check('two_tabs', self.check_two_tabs)
        return self.results


def _fmt(values: list) -> str:
    """Ranges as text, e.g. ``x -2.9..22.9 y -2.0..6.0``."""
    def num(v):
        return 'NaN' if v is None else '%.3g' % v
    return 'x %s..%s y %s..%s' % tuple(num(v) for v in values)


def downloaded_files(folder: Path, name: str) -> List[Path]:
    """A download file and the numbered copies Firefox saves when the file
    exists already (``name(1).ext``, ...) in a folder.

    Parameters
    ----------
    folder : Path
        Download folder.
    name : str
        File name of the download.

    Returns
    -------
    list of Path
        The files that exist.
    """
    stem, suffix = name.rsplit('.', 1)
    copy = re.compile(r'%s\(\d+\)\.%s\Z' % (re.escape(stem),
                                              re.escape(suffix)))
    return [path for path in sorted(folder.glob('%s*' % stem))
            if path.is_file() and (path.name == name
                                   or copy.match(path.name))]


def tool_files(out: Path, download_dir: Optional[Path]) -> List[Path]:
    """The files of an earlier run that exist in OUT and the download
    folder: the files this tool writes, nothing else.

    Parameters
    ----------
    out : Path
        Folder of the screenshots and the report.
    download_dir : Path or None
        Download folder given by the user (None: a new temporary one).

    Returns
    -------
    list of Path
        Screenshots ``OUT/<page>_<state>.png``, ``OUT/report.json``, and
        the download files (with their numbered copies and ``.part``
        files) in ``OUT/downloads`` and in the download folder.
    """
    files = [out / ('%s_%s.png' % (page, state))
             for page in PAGE_NAMES for state in SCREENSHOT_STATES]
    files.append(out / REPORT_FILE)
    files = [path for path in files if path.is_file()]
    for folder in (out / DOWNLOADS_COPY, download_dir):
        if folder is not None and folder.is_dir():
            for name in DOWNLOAD_FILES:
                files.extend(downloaded_files(folder, name))
                part = folder / (name + '.part')  # an unfinished download
                if part.is_file():
                    files.append(part)
    return files


def make_driver(download_dir: Path, geckodriver: str,
                window: tuple = (WIDTH, HEIGHT)):
    """Start headless Firefox with downloads saved to ``download_dir``.

    Parameters
    ----------
    download_dir : Path
        Download folder; snap Firefox can only write below the home folder.
    geckodriver : str
        Path of geckodriver.
    window : tuple of int, optional
        Window width and height in px.

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
             'text/csv,text/yaml,application/x-yaml,application/zip,'
             'application/octet-stream'),
            ('browser.download.panel.shown', False)):
        options.set_preference(key, value)
    driver = webdriver.Firefox(options=options,
                               service=Service(geckodriver))
    driver.set_window_size(*window)
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
                        help='folder of the screenshots and report.json, '
                        'empty or new (see --clean)')
    parser.add_argument('--width', type=int, default=WIDTH,
                        help='window width in px (default: %(default)s)')
    parser.add_argument('--height', type=int, default=None,
                        help='window height in px (default: %d, or %d from '
                        'a width of %d px on)' % (HEIGHT, WIDE_HEIGHT, WIDE))
    parser.add_argument('--setup', type=Path, default=DEFAULT_SETUP,
                        help='setup file of the load check (default: %s)'
                        % DEFAULT_SETUP.relative_to(REPO))
    parser.add_argument('--download-dir', type=Path, default=None,
                        help='download folder of the browser, empty or new '
                        '(default: a new folder in the home folder, removed '
                        'afterwards; snap Firefox cannot write to /tmp)')
    parser.add_argument('--clean', action='store_true',
                        help='first remove the files of an earlier run of '
                        'this tool (screenshots, report.json, download files) '
                        'from OUT and the download folder; other files stay, '
                        'and without --clean both must be empty')
    parser.add_argument('--geckodriver', default=GECKODRIVER,
                        help='path of geckodriver (default: %(default)s)')
    parser.add_argument('--settle', type=float, default=1.5,
                        help='seconds a page must stay unchanged '
                        '(default: %(default)s)')
    parser.add_argument('--timeout', type=float, default=180.0,
                        help='seconds to wait for a calculation '
                        '(default: %(default)s)')
    parser.add_argument('--two-tabs', action='store_true',
                        help='also check that two browser windows on one '
                        'server do not share a session (check two_tabs)')
    parser.add_argument('--server-log', type=Path, default=None,
                        help='log file of the server; the check fast_input '
                        'then also fails on an exception logged while it '
                        'runs')
    args = parser.parse_args(argv)
    height = args.height if args.height is not None else \
        WIDE_HEIGHT if args.width >= WIDE else HEIGHT
    window = (args.width, height)
    out = args.out.resolve()
    given_dir = None if args.download_dir is None \
        else args.download_dir.resolve()
    if args.clean:
        # only the files this tool writes, never anything else
        for old in tool_files(out, given_dir):
            print('removing %s' % old, flush=True)
            old.unlink()
    else:
        for folder, option in ((out, '--out'), (given_dir, '--download-dir')):
            if folder is not None and folder.is_dir() and \
                    any(folder.iterdir()):
                parser.error('%s %s is not empty; use a new folder, or pass '
                             '--clean to remove the files of an earlier run '
                             'of this tool from it' % (option, folder))
    out.mkdir(parents=True, exist_ok=True)
    remove = given_dir is None
    download_dir = Path(tempfile.mkdtemp(prefix='pypalc_browser_check_',
                                         dir=Path.home())) \
        if remove else given_dir
    download_dir.mkdir(parents=True, exist_ok=True)
    check = BrowserCheck(None, args.url, out, download_dir,
                         args.setup.resolve(), window=window,
                         settle=args.settle, timeout=args.timeout,
                         server_log=args.server_log)
    try:
        # inside the try: the download folder is removed and a report is
        # written also if Firefox or geckodriver do not start
        check.driver = make_driver(download_dir, args.geckodriver, window)
        check.run(two_tabs=args.two_tabs)
    except Exception as exc:  # report what ran so far
        check.add('session', False, '%s: %s' % (type(exc).__name__, exc))
    finally:
        if check.driver is not None:
            try:
                check.driver.quit()
            except Exception as exc:
                print('could not quit the browser: %s' % exc, flush=True)
        try:
            # only the download files: a given download folder may hold
            # other files
            downloads = out / DOWNLOADS_COPY
            downloads.mkdir(exist_ok=True)
            for name in DOWNLOAD_FILES:
                for path in downloaded_files(download_dir, name):
                    shutil.copy2(path, downloads / path.name)
        except Exception as exc:  # report.json is still written
            check.add('copy_downloads', False,
                      '%s: %s' % (type(exc).__name__, exc))
        if remove:  # the temporary folder this run created
            shutil.rmtree(download_dir, ignore_errors=True)
    (out / REPORT_FILE).write_text(json.dumps(check.results, indent=2)
                                   + '\n', encoding='utf-8')
    failed = [r['name'] for r in check.results if not r['ok']]
    print('%d of %d checks ok%s' % (len(check.results) - len(failed),
                                     len(check.results),
                                     '; failed: ' + ', '.join(failed)
                                     if failed else ''))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
