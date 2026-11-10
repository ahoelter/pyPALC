"""Check that a git tag matches ``pypalc.__version__`` (tag ``vX.Y.Z``).

Usage: python tools/check_version_tag.py TAG

The version is read from ``pypalc/__init__.py`` without importing the package.
Exit code 0 if ``TAG == "v" + __version__``, else 1.
"""

import re
import sys
from pathlib import Path

INIT = Path(__file__).resolve().parent.parent / "pypalc" / "__init__.py"


def read_version() -> str:
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', INIT.read_text(), re.M)
    if match is None:
        raise SystemExit(f"no __version__ found in {INIT}")
    return match.group(1)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    tag, expected = argv[1], "v" + read_version()
    if tag != expected:
        print(f"tag {tag!r} does not match the package version (expected {expected!r})")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
