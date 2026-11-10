"""``pypalc.core`` stays free of Bokeh."""
import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "pypalc"
CORE = PACKAGE / "core"
EXAMPLES = PACKAGE / "examples"


def _imported_modules(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def test_core_has_no_bokeh():
    files = sorted(CORE.rglob("*.py")) + sorted(EXAMPLES.rglob("*.py"))
    assert files
    offending = [
        "%s imports %s" % (path.name, module)
        for path in files for module in _imported_modules(path)
        if module == "bokeh" or module.startswith("bokeh.")
    ]
    assert offending == []
