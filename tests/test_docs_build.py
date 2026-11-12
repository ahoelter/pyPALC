"""The Sphinx documentation builds without warnings; Read the Docs config."""
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


def test_sphinx_build_without_warnings(tmp_path):
    pytest.importorskip("sphinx")
    pytest.importorskip("furo")
    pytest.importorskip("myst_parser")
    result = subprocess.run(
        [sys.executable, "-m", "sphinx", "-W", "--keep-going", "-b", "html",
         str(ROOT / "docs" / "source"), str(tmp_path)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "index.html").is_file()
    assert (tmp_path / "api" / "generated" / "pypalc.core.pipeline.html").is_file()
    assert (tmp_path / "api" / "generated" / "pypalc.gui.app.html").is_file()
    # the API pages document the members of the modules, not only their
    # docstrings
    generated = tmp_path / "api" / "generated"
    for page, name in [
            ("pypalc.core.pipeline", "pypalc.core.pipeline.run"),
            ("pypalc.core.pipeline", "pypalc.core.pipeline.RunResult"),
            ("pypalc.core.settings", "pypalc.core.settings.Setup"),
            ("pypalc.examples", "pypalc.examples.load_example"),
            ("pypalc.gui.app", "pypalc.gui.app.PalcApp")]:
        html = (generated / (page + ".html")).read_text(encoding="utf-8")
        assert 'id="%s"' % name in html, name


def test_readthedocs_config():
    config = yaml.safe_load((ROOT / ".readthedocs.yaml").read_text())
    assert config["version"] == 2
    assert config["build"]["os"] == "ubuntu-24.04"
    assert config["build"]["tools"]["python"] == "3.12"
    assert config["python"]["install"] == [
        {"method": "pip", "path": ".", "extra_requirements": ["docs"]}]
    assert config["sphinx"]["configuration"] == "docs/source/conf.py"
    assert config["sphinx"]["fail_on_warning"] is True
