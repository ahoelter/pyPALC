"""The documentation pages match the code.

The setup file reference names every setting of :mod:`pypalc.core.settings`
and nothing else, the tutorial scripts run, and the command line page names
every exit code and every file the runner writes.
"""
import dataclasses
import re
import runpy
import typing
from pathlib import Path

import pytest

from pypalc import cli
from pypalc.core import pipeline
from pypalc.core.export import write_results
from pypalc.core.settings import Setup, load_setup
from pypalc.examples import load_example

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "docs" / "source"
CASES = ROOT / "tests" / "cases"
TOP_LEVEL = tuple(f.name for f in dataclasses.fields(Setup))


def _dataclass_in(tp):
    """The dataclass inside ``tp`` and whether it is a list of them.

    ``Optional[X]`` is unwrapped and ``Tuple[X, ...]`` counts as a list.
    Returns ``(None, False)`` if ``tp`` holds no dataclass.
    """
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin is typing.Union:
        inner = [a for a in args if a is not type(None)]
        return _dataclass_in(inner[0]) if len(inner) == 1 else (None, False)
    if origin is tuple and len(args) == 2 and args[1] is Ellipsis:
        cls, _ = _dataclass_in(args[0])
        return cls, cls is not None
    if dataclasses.is_dataclass(tp):
        return tp, False
    return None, False


def _keys(cls, prefix=""):
    """Dotted keys of every setting of dataclass ``cls`` and below."""
    hints = typing.get_type_hints(cls)
    keys = set()
    for field in dataclasses.fields(cls):
        key = prefix + field.name
        keys.add(key)
        inner, is_list = _dataclass_in(hints[field.name])
        if inner is not None:
            keys |= _keys(inner, key + ("[]." if is_list else "."))
    return keys


SETUP_KEYS = _keys(Setup) | {"version"}


def _text(name):
    return (SOURCE / name).read_text(encoding="utf-8")


def test_setup_keys_are_collected():
    assert "venue.audience_lines[].start" in SETUP_KEYS
    assert "loudspeaker.measured_directivity.csv" in SETUP_KEYS
    assert "target_slope.points[].level_db" in SETUP_KEYS


def test_every_setup_key_documented():
    text = _text("setup-file.md")
    missing = sorted(k for k in SETUP_KEYS if "`%s`" % k not in text)
    assert missing == []


def test_no_unknown_setup_key_documented():
    tokens = re.findall(r"`([^`\s]+)`", _text("setup-file.md"))
    prefixes = tuple(name + "." for name in TOP_LEVEL)
    unknown = sorted({t for t in tokens
                      if t.startswith(prefixes) and t not in SETUP_KEYS})
    assert unknown == []


@pytest.mark.parametrize(
    "script", sorted((SOURCE / "tutorial").glob("*.py")),
    ids=lambda p: p.name)
def test_tutorial_scripts_run(script, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(script), run_name="__main__")


def test_tutorial_scripts_exist():
    names = {p.name for p in (SOURCE / "tutorial").glob("*.py")}
    assert {"run_setup.py", "build_setup.py"} <= names


def test_exit_codes_documented():
    text = _text("command-line.md")
    codes = [value for name, value in vars(cli).items()
             if name.startswith("EXIT_")]
    assert codes
    missing = [code for code in [0] + codes
               if not re.search(r"\|\s*`?%d`?\s*\|" % code, text)]
    assert missing == []


@pytest.mark.parametrize("setup", [
    pytest.param(lambda: load_example("simple"), id="simple"),
    pytest.param(lambda: load_setup(CASES / "target_slope.yaml"),
                 id="target_slope"),
])
def test_result_files_documented(setup, tmp_path):
    files = write_results(pipeline.run(setup()), tmp_path)
    text = _text("command-line.md")
    missing = [p.name for p in files if "`%s`" % p.name not in text]
    assert missing == []
