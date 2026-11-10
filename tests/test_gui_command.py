"""``pypalc gui``: the app entry inside the package."""
import subprocess
import sys

import pytest
from bokeh.document import Document
from panel.io.state import set_curdoc

from pypalc import cli


def test_create_view_builds_session():
    from pypalc.gui.serve import create_view
    doc = Document()
    with set_curdoc(doc):
        view = create_view()
        view.server_doc(doc)
    names = {type(obj).__name__ for obj in view.select()}
    assert "Page" in names
    labels = [getattr(obj, "label", None) for obj in view.select()]
    assert "Save line" in labels or any(
        isinstance(label, str) and "Save" in label for label in labels)


def test_serve_arguments(monkeypatch):
    import panel
    from pypalc.gui.serve import create_view
    calls = []
    monkeypatch.setattr(
        panel, "serve", lambda *args, **kwargs: calls.append((args, kwargs)))
    assert cli.main(["gui", "--port", "5010", "--address", "0.0.0.0",
                     "--allow-websocket-origin", "a.org",
                     "--allow-websocket-origin", "b.org",
                     "--prefix", "pypalc", "--no-show"]) == 0
    args, kwargs = calls[-1]
    assert args == ({"pyPALC": create_view},)
    assert kwargs["port"] == 5010
    assert kwargs["address"] == "0.0.0.0"
    assert kwargs["websocket_origin"] == ["a.org", "b.org"]
    assert kwargs["prefix"] == "pypalc"
    assert kwargs["show"] is False
    assert kwargs["title"] == "pyPALC"
    assert cli.main(["gui"]) == 0
    args, kwargs = calls[-1]
    assert kwargs["port"] == 5006
    assert kwargs["show"] is True
    # not given: left out, so Panel uses its defaults
    assert not {"address", "websocket_origin", "prefix"} & set(kwargs)


def test_gui_without_panel(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "panel", None)
    monkeypatch.delitem(sys.modules, "pypalc.gui.serve", raising=False)
    assert cli.main(["gui"]) == 6
    assert 'pip install "pypalc[gui]"' in capsys.readouterr().err


def test_gui_other_import_error_is_shown(monkeypatch, capsys):
    class Broken:
        """Stands in for pypalc.gui.serve: importing it fails."""
        def __getattr__(self, name):
            raise ImportError("cannot import name 'x' from 'pypalc.gui'")

    monkeypatch.setitem(sys.modules, "pypalc.gui.serve", Broken())
    assert cli.main(["gui"]) == 6
    err = capsys.readouterr().err
    assert "cannot import name 'x'" in err
    assert "pypalc gui could not start" in err
    assert "pip install" not in err


def test_cli_import_has_no_panel():
    code = ("import sys, pypalc.cli; "
            "assert 'panel' not in sys.modules and 'bokeh' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)
