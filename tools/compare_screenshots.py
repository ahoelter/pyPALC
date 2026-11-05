"""Put two folders of screenshots side by side in one HTML page.

Usage::

    python tools/compare_screenshots.py OLD_DIR NEW_DIR OUT.html \\
        [--title TITLE] [--jpeg QUALITY] [--max-width PX]

Writes a self-contained HTML page (dark background) with one row per file
name (``*.png`` in either folder), sorted by name: a heading, the old
screenshot on the left and the new one on the right. The heading is the
file name without ``.png`` and without a leading sort key ``<digits>-``,
with ``_`` shown as a space (``03-1280 px_1 Venue_loaded.png`` gives "1280
px 1 Venue loaded"). The images are embedded as base64, so the page needs
no other files; with ``--jpeg`` they are embedded as JPEG of that quality
(and with ``--max-width`` scaled down to at most that width), which keeps
the page small. A file that exists in only one folder is shown as missing
on the other side.

author: Arne Hoelter
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import re
import sys
from pathlib import Path
from typing import List, Optional

STYLE = """
body { font-family: sans-serif; margin: 16px; background: #1e1e1e;
       color: #eee; }
h1 { font-size: 22px; font-weight: normal; }
h2 { font-size: 17px; font-weight: normal; margin: 28px 0 8px;
     padding-bottom: 4px; border-bottom: 1px solid #555; }
.row { display: flex; gap: 12px; align-items: flex-start; }
.row > figure { flex: 1 1 0; min-width: 0; margin: 0; }
.head { display: flex; gap: 12px; position: sticky; top: 0;
        background: #1e1e1e; padding: 6px 0; z-index: 1; }
.head > div { flex: 1 1 0; color: #aaa; }
img { max-width: 100%; height: auto; display: block;
      border: 1px solid #444; }
.missing { color: #f08080; font-style: italic; }
figcaption { font-size: 12px; color: #999; margin-top: 4px; }
"""


def heading(name: str) -> str:
    """Heading of the row of a file.

    Parameters
    ----------
    name : str
        File name, e.g. ``03-1280 px_1 Venue_loaded.png``.

    Returns
    -------
    str
        For example ``'1280 px 1 Venue loaded'``.
    """
    stem = re.sub(r'^\d+-', '', name[:-len('.png')]
                  if name.endswith('.png') else name)
    return stem.replace('_', ' ')


def image_data(path: Path, jpeg: Optional[int] = None,
               max_width: Optional[int] = None) -> tuple:
    """The image of a file as a data URI payload.

    Parameters
    ----------
    path : Path
        PNG file.
    jpeg : int, optional
        JPEG quality; None keeps the PNG.
    max_width : int, optional
        Scale a JPEG down to at most this width in px.

    Returns
    -------
    tuple
        ``(mime type, base64 text, (width, height))``; the size is None
        for a PNG that was kept.
    """
    if jpeg is None:
        return ('image/png', base64.b64encode(path.read_bytes())
                .decode('ascii'), None)
    from PIL import Image  # Pillow comes with Bokeh
    with Image.open(path) as image:
        image = image.convert('RGB')
        if max_width and image.width > max_width:
            image = image.resize(
                (max_width, round(image.height * max_width / image.width)),
                Image.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, 'JPEG', quality=jpeg, optimize=True)
        return ('image/jpeg', base64.b64encode(buffer.getvalue())
                .decode('ascii'), image.size)


def _figure(folder: Path, name: str, jpeg: Optional[int],
            max_width: Optional[int]) -> str:
    """HTML of one screenshot: the embedded image and its caption.

    Parameters
    ----------
    folder : Path
        Folder of the screenshot.
    name : str
        File name of the screenshot.
    jpeg, max_width : int or None
        See :func:`image_data`.

    Returns
    -------
    str
    """
    path = folder / name
    caption = html.escape(str(path))
    if not path.is_file():
        return '<figure><p class="missing">missing: %s</p></figure>' % caption
    mime, data, _ = image_data(path, jpeg, max_width)
    return ('<figure><img alt="%s" src="data:%s;base64,%s">'
            '<figcaption>%s</figcaption></figure>'
            % (html.escape(name), mime, data, caption))


def compare_page(old: Path, new: Path, title: str = 'Screenshot comparison',
                 jpeg: Optional[int] = None,
                 max_width: Optional[int] = None) -> str:
    """HTML page with the screenshots of both folders side by side.

    Parameters
    ----------
    old, new : Path
        Folders of the old (left) and new (right) screenshots.
    title : str, optional
        Title and heading of the page.
    jpeg, max_width : int or None, optional
        See :func:`image_data`.

    Returns
    -------
    str
        The HTML page.
    """
    names = sorted({p.name for p in old.glob('*.png')} |
                   {p.name for p in new.glob('*.png')})
    rows = []
    for name in names:
        rows.append('<h2>%s</h2>\n<div class="row">%s%s</div>' % (
            html.escape(heading(name)),
            _figure(old, name, jpeg, max_width),
            _figure(new, name, jpeg, max_width)))
    return ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, '
            'initial-scale=1"><title>%s</title><style>%s</style></head>'
            '<body><h1>%s</h1><div class="head"><div>old: %s</div>'
            '<div>new: %s</div></div>\n%s\n</body></html>\n'
            % (html.escape(title), STYLE, html.escape(title),
               html.escape(str(old)), html.escape(str(new)),
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
    parser.add_argument('--title', default='Screenshot comparison',
                        help='title of the page (default: %(default)s)')
    parser.add_argument('--jpeg', type=int, default=None, metavar='QUALITY',
                        help='embed the images as JPEG of this quality '
                        '(1-95; default: PNG as they are)')
    parser.add_argument('--max-width', type=int, default=None, metavar='PX',
                        help='with --jpeg: scale the images down to at most '
                        'this width')
    args = parser.parse_args(argv)
    for folder in (args.old, args.new):
        if not folder.is_dir():
            print('not a folder: %s' % folder, file=sys.stderr)
            return 1
    args.out.write_text(compare_page(args.old, args.new, args.title,
                                     args.jpeg, args.max_width),
                        encoding='utf-8')
    print('wrote %s (%.1f MB)' % (args.out,
                                  args.out.stat().st_size / 1e6))
    return 0


if __name__ == '__main__':
    sys.exit(main())
