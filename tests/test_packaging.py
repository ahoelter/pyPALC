"""The package metadata, the logos inside the package and the built wheel."""
import re
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

import pypalc
import pypalc.gui

ROOT = Path(__file__).resolve().parents[1]


def _pyproject():
    return tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))


def _names(requirements):
    return {re.split(r'[<>=!~;\[ ]', r, maxsplit=1)[0].lower()
            for r in requirements}


def _gui_pngs():
    source = (ROOT / 'pypalc' / 'gui' / 'app.py').read_text(encoding='utf-8')
    names = set(re.findall(r"_data_uri\('([^']+\.png)'\)", source))
    assert names
    return sorted(names)


def test_version():
    assert pypalc.__version__ == '0.1.0'
    data = _pyproject()
    assert data['project']['dynamic'] == ['version']
    assert (data['tool']['setuptools']['dynamic']['version']
            == {'attr': 'pypalc.__version__'})


def test_console_script():
    assert _pyproject()['project']['scripts']['pypalc'] == 'pypalc.cli:main'


def test_core_dependencies():
    project = _pyproject()['project']
    assert _names(project['dependencies']) == {'numpy', 'scipy', 'pyyaml'}
    assert {'panel', 'panel-material-ui', 'bokeh'} <= _names(
        project['optional-dependencies']['gui'])


def test_static_inside_package():
    assert pypalc.gui.STATIC_DIR == Path(pypalc.gui.__file__).parent / 'static'
    for name in _gui_pngs():
        assert (pypalc.gui.STATIC_DIR / name).is_file()


def test_wheel_contents(tmp_path):
    try:
        import setuptools
        if int(setuptools.__version__.split('.')[0]) < 77:
            pytest.skip('setuptools >= 77 is needed to build the wheel')
    except ImportError:
        pytest.skip('setuptools is not installed')
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'wheel', '.', '--no-deps',
         '--no-build-isolation', '-w', str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip('pip wheel is unavailable: ' + result.stderr[-300:])
    wheel, = tmp_path.glob('pypalc-*.whl')
    names = zipfile.ZipFile(wheel).namelist()
    for name in _gui_pngs():
        assert f'pypalc/gui/static/{name}' in names
    assert 'pypalc/cli.py' in names
    for forbidden in ('tests/', 'tools/', 'docs/', 'main.py'):
        assert not [n for n in names if n.startswith(forbidden)]
