"""Put two folders of screenshots side by side in one HTML page.

Usage::

    python tools/compare_screenshots.py OLD_DIR NEW_DIR OUT.html

Writes a plain HTML page with one row per file name (``*.png`` in either
folder): the old screenshot on the left, the new one on the right, each with
a caption. The images are embedded as base64, so the page needs no other
files. A file that exists in only one folder is shown as missing on the
other side.

author: Arne Hoelter
"""
from __future__ import annotations

import argparse
import base64
import html
import sys
from pathlib import Path
from typing import List, Optional

STYLE = """
body { font-family: sans-serif; margin: 16px; background: #fff; color: #222; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #ccc; padding: 6px; vertical-align: top;
         width: 50%; }
th { background: #eee; text-align: left; }
tr.name td { background: #f6f6f6; font-weight: bold; }
img { max-width: 100%; height: auto; display: block; }
.missing { color: #b00; font-style: italic; }
figcaption { font-size: 12px; color: #555; margin-top: 4px; }
"""


def _cell(folder: Path, name: str) -> str:
    """HTML of one table cell: the embedded image and its caption.

    Parameters
    ----------
    folder : Path
        Folder of the screenshot.
    name : str
        File name of the screenshot.

    Returns
    -------
    str
    """
    path = folder / name
    caption = html.escape(str(path))
    if not path.is_file():
        return '<td><p class="missing">missing: %s</p></td>' % caption
    data = base64.b64encode(path.read_bytes()).decode('ascii')
    return ('<td><figure><img alt="%s" src="data:image/png;base64,%s">'
            '<figcaption>%s</figcaption></figure></td>'
            % (html.escape(name), data, caption))


def compare_page(old: Path, new: Path) -> str:
    """HTML page with the screenshots of both folders side by side.

    Parameters
    ----------
    old, new : Path
        Folders of the old (left) and new (right) screenshots.

    Returns
    -------
    str
        The HTML page.
    """
    names = sorted({p.name for p in old.glob('*.png')} |
                   {p.name for p in new.glob('*.png')})
    rows = []
    for name in names:
        rows.append('<tr class="name"><td colspan="2">%s</td></tr>'
                    % html.escape(name))
        rows.append('<tr>%s%s</tr>' % (_cell(old, name), _cell(new, name)))
    return ('<!doctype html>\n<html><head><meta charset="utf-8">'
            '<title>Screenshot comparison</title><style>%s</style></head>'
            '<body><h1>Screenshot comparison</h1>'
            '<table><tr><th>old: %s</th><th>new: %s</th></tr>\n%s\n</table>'
            '</body></html>\n'
            % (STYLE, html.escape(str(old)), html.escape(str(new)),
               '\n'.join(rows)))


def main(argv: Optional[List[str]] = None) -> int:
    """Command line entry point.

    Parameters
    ----------
    argv : list of str, optional
        Arguments; default ``sys.argv[1:]``.

    Returns
    -------
    int
        0, or 1 if a folder does not exist.
    """
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('old', type=Path, help='folder of the old screenshots')
    parser.add_argument('new', type=Path, help='folder of the new screenshots')
    parser.add_argument('out', type=Path, help='HTML page to write')
    args = parser.parse_args(argv)
    for folder in (args.old, args.new):
        if not folder.is_dir():
            print('not a folder: %s' % folder, file=sys.stderr)
            return 1
    args.out.write_text(compare_page(args.old, args.new), encoding='utf-8')
    print('wrote %s' % args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
