"""Take a screenshot of every page of the running pyPALC app.

Opens a running pyPALC server (``panel serve pyPALC`` from the parent
folder) in a headless Firefox (selenium, geckodriver), clicks each page tab
in the header, waits until the page is stable and saves a full-page
screenshot ``OUT/<index>_<name>.png``. For every page it reports whether
the page scrolls horizontally at the window width: the document
(``document.documentElement.scrollWidth > innerWidth``) or the main area of
the Material page (``.main-content``, which scrolls by itself, so that the
document never does).

Usage (from the repository root, with selenium installed)::

    python tools/page_screenshots.py --url http://localhost:5006/pyPALC \\
        --out shots --width 1280 [--height 900]

Prints one line per page and exits with 0, or with 1 if a page could not be
shown.

author: Arne Hoelter
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import List, Optional

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pypalc.gui.app import PAGE_LABELS  # noqa: E402

#: Default path of geckodriver (the snap's, which finds snap Firefox,
#: ``/usr/bin/firefox``, itself).
GECKODRIVER = '/snap/bin/geckodriver'
#: Seconds to wait for the app and for each page.
TIMEOUT = 60
#: A page is stable when its DOM signature did not change for this many
#: polls in a row, :data:`POLL` seconds apart.
STABLE_POLLS = 4
POLL = 0.4

#: ``deep(sel)`` queries the document and all shadow roots (Bokeh and
#: Panel render their models in shadow DOM).
JS_DEEP = r"""
function deep(sel, root, out) {
    root = root || document; out = out || [];
    for (const e of root.querySelectorAll(sel)) out.push(e);
    for (const e of root.querySelectorAll('*'))
        if (e.shadowRoot) deep(sel, e.shadowRoot, out);
    return out;
}
"""
#: Clicks the page tab with the label ``arguments[0]``; true if found.
JS_CLICK_TAB = JS_DEEP + r"""
const tab = deep('.MuiTab-root').find(
    e => e.textContent.trim() === arguments[0]);
if (!tab) return false;
tab.click();
return true;
"""
#: Signature of the page: whether the tab ``arguments[0]`` is selected, the
#: number of elements (also in shadow roots) and the page size.
JS_SIGNATURE = JS_DEEP + r"""
const tab = deep('.MuiTab-root').find(
    e => e.textContent.trim() === arguments[0]);
const html = document.documentElement;
return [tab ? tab.getAttribute('aria-selected') : null,
        deep('*').length, html.scrollWidth, html.scrollHeight];
"""
#: ``[[scrollWidth, innerWidth]`` of the document, then ``[scrollWidth,
#: clientWidth]`` of each main area]``.
JS_WIDTHS = JS_DEEP + r"""
return [[document.documentElement.scrollWidth, window.innerWidth]].concat(
    deep('.main-content').map(e => [e.scrollWidth, e.clientWidth]));
"""


def page_name(index: int, label: str) -> str:
    """File name (without ``.png``) of a page's screenshot.

    Parameters
    ----------
    index : int
        Page index.
    label : str
        Label of the page tab, e.g. ``'1 · Venue'``.

    Returns
    -------
    str
        For example ``'1_venue'``, ``'0_help'``.
    """
    words = re.findall(r'[a-z]+', label.lower())
    return '%d_%s' % (index, '_'.join(words))


def make_driver(width: int, height: int, geckodriver: str,
                firefox: Optional[str] = None):
    """Start headless Firefox with the given window size.

    Parameters
    ----------
    width, height : int
        Window size in pixels.
    geckodriver : str
        Path of geckodriver.
    firefox : str, optional
        Path of the Firefox binary; if None, geckodriver finds Firefox (the
        snap wrapper ``/usr/bin/firefox`` is no binary geckodriver accepts).

    Returns
    -------
    selenium.webdriver.Firefox
    """
    from selenium import webdriver
    from selenium.webdriver.firefox.options import Options
    from selenium.webdriver.firefox.service import Service
    options = Options()
    options.add_argument('-headless')
    if firefox is not None:
        options.binary_location = firefox
    driver = webdriver.Firefox(options=options,
                               service=Service(geckodriver))
    driver.set_window_size(width, height)
    driver.set_script_timeout(30)
    return driver


def wait_stable(driver, label: str, timeout: float = TIMEOUT) -> bool:
    """Wait until the tab ``label`` is selected and the page stops
    changing.

    Parameters
    ----------
    driver : selenium.webdriver.Firefox
        The browser.
    label : str
        Label of the page tab that must be selected.
    timeout : float, optional
        Seconds to wait at most.

    Returns
    -------
    bool
        True if the page became stable in time.
    """
    end = time.monotonic() + timeout
    last, same = None, 0
    while time.monotonic() < end:
        signature = driver.execute_script(JS_SIGNATURE, label)
        if signature[0] == 'true' and signature == last:
            same += 1
            if same >= STABLE_POLLS:
                return True
        else:
            same = 0
        last = signature
        time.sleep(POLL)
    return False


def take_screenshots(driver, url: str, out: Path) -> List[str]:
    """Show every page, save its screenshot and report its overflow.

    Parameters
    ----------
    driver : selenium.webdriver.Firefox
        The browser.
    url : str
        URL of the app.
    out : Path
        Folder of the screenshots.

    Returns
    -------
    list of str
        The problems (pages that could not be shown); empty if none.
    """
    problems = []
    driver.get(url)
    for index, label in enumerate(PAGE_LABELS):
        end = time.monotonic() + TIMEOUT
        while not driver.execute_script(JS_CLICK_TAB, label):
            if time.monotonic() > end:
                break
            time.sleep(POLL)
        if not wait_stable(driver, label):
            problems.append('%s: page did not become stable' % label)
            print('%s: FAIL, page did not become stable' % label)
            continue
        name = page_name(index, label)
        driver.save_full_page_screenshot(str(out / (name + '.png')))
        widths = driver.execute_script(JS_WIDTHS)
        overflow = any(scroll > visible for scroll, visible in widths)
        print('%s: horizontal overflow %s (document %d/%d, main area %s)'
              % (name, 'YES' if overflow else 'no', widths[0][0],
                 widths[0][1], ', '.join('%d/%d' % tuple(w)
                                         for w in widths[1:]) or 'none'))
    return problems


def main(argv: Optional[List[str]] = None) -> int:
    """Command line entry point.

    Parameters
    ----------
    argv : list of str, optional
        Arguments without the program name; ``sys.argv[1:]`` if omitted.

    Returns
    -------
    int
        Exit code: 0, or 1 if a page could not be shown.
    """
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--url', required=True, help='URL of the app')
    parser.add_argument('--out', required=True, type=Path,
                        help='folder of the screenshots')
    parser.add_argument('--width', required=True, type=int,
                        help='window width in pixels')
    parser.add_argument('--height', type=int, default=900,
                        help='window height in pixels (default 900)')
    parser.add_argument('--firefox', default=None,
                        help='path of the Firefox binary (default: the one '
                             'geckodriver finds)')
    parser.add_argument('--geckodriver', default=GECKODRIVER,
                        help='path of geckodriver (default %s)' % GECKODRIVER)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    driver = make_driver(args.width, args.height, args.geckodriver,
                         args.firefox)
    try:
        problems = take_screenshots(driver, args.url, args.out)
    finally:
        driver.quit()
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
