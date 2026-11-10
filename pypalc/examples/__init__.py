"""Example setup files shipped with pyPALC.

The files live next to this module. Every function takes an optional
``directory``; by default the module attribute :data:`EXAMPLES_DIR` is used,
looked up at call time.
"""
from __future__ import annotations

import dataclasses
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ..core.errors import SetupError
from ..core.settings import MeasuredDirectivity, Setup, load_setup

#: Folder of the example files.
EXAMPLES_DIR = Path(__file__).resolve().parent

_SUFFIXES = (".yaml", ".yml")


@dataclass(frozen=True)
class Example:
    """An example setup: file stem, display title and file path."""
    name: str
    title: str
    path: Path


def _title(name: str) -> str:
    text = name.replace("_", " ")
    return text[:1].upper() + text[1:]


def _directory(directory: Optional[Path]) -> Path:
    return Path(EXAMPLES_DIR if directory is None else directory)


def examples(directory: Optional[Path] = None) -> List[Example]:
    """All example setups of ``directory``, sorted by title."""
    found = [Example(p.stem, _title(p.stem), p)
             for p in _directory(directory).iterdir()
             if p.is_file() and p.suffix in _SUFFIXES]
    return sorted(found, key=lambda e: (e.title, e.name))


def load_example(name: str, directory: Optional[Path] = None) -> Setup:
    """Load the example ``name``.

    A relative measured-directivity ``path`` is replaced by the contents of
    the file (``MeasuredDirectivity(csv=...)``), so that the GUI can apply
    the setup without access to the file.

    Raises
    ------
    SetupError
        With key ``"example"`` if there is no such example.
    """
    for example in examples(directory):
        if example.name == name:
            break
    else:
        raise SetupError("example", "unknown example '%s'" % name)
    setup = load_setup(example.path)
    measured = setup.loudspeaker.measured_directivity
    if measured is not None and measured.path is not None:
        try:
            text = Path(measured.path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise SetupError("loudspeaker.measured_directivity.path",
                             "cannot read %s: %s" % (measured.path, exc)
                             ) from exc
        loudspeaker = dataclasses.replace(
            setup.loudspeaker, measured_directivity=MeasuredDirectivity(
                csv=text))
        setup = dataclasses.replace(setup, loudspeaker=loudspeaker)
    return setup


def copy_examples(destination: Path,
                  directory: Optional[Path] = None) -> List[Path]:
    """Copy the example files into ``destination`` and return the new paths.

    ``destination`` is created. Nothing is overwritten: if any target file
    exists, ``FileExistsError`` is raised before anything is written.
    """
    source = _directory(directory)
    destination = Path(destination)
    files = sorted(p for p in source.iterdir()
                   if p.is_file() and p.suffix != ".py")
    for path in files:
        if (destination / path.name).exists():
            raise FileExistsError(
                "%s already exists" % (destination / path.name))
    destination.mkdir(parents=True, exist_ok=True)
    written = []
    for path in files:
        target = destination / path.name
        shutil.copyfile(path, target)
        written.append(target)
    return written
