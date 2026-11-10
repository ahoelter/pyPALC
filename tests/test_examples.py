"""The example setups shipped inside the package."""
import shutil
from pathlib import Path

import pytest

from pypalc.core import pipeline
from pypalc.core.errors import SetupError
from pypalc.examples import copy_examples, examples, load_example

CASES = Path(__file__).resolve().parent / "cases"


def test_simple_is_listed():
    assert [e.name for e in examples()] == ["simple"]
    assert examples()[0].title == "Simple"


@pytest.mark.parametrize("example", examples(), ids=lambda e: e.name)
def test_every_example_runs(example):
    setup = load_example(example.name)
    result = pipeline.run(setup)
    assert len(result.optimized.tilt_angles_deg) == setup.array.cabinet_count


def test_titles_and_order(tmp_path):
    for name in ("gap_soft_margin", "alpha"):
        (tmp_path / (name + ".yaml")).write_text("version: 1\n")
    (tmp_path / "notes.txt").write_text("x")
    found = examples(tmp_path)
    assert [e.name for e in found] == ["alpha", "gap_soft_margin"]
    assert found[1].title == "Gap soft margin"


def test_measured_example_embeds_csv(tmp_path):
    for name in ("measured.yaml", "measured.csv"):
        shutil.copy(CASES / name, tmp_path / name)
    measured = load_example("measured", tmp_path).loudspeaker.measured_directivity
    assert measured.path is None
    assert measured.csv == (tmp_path / "measured.csv").read_text()


def test_unknown_example():
    with pytest.raises(SetupError) as info:
        load_example("nope")
    assert info.value.key == "example"


def test_copy_examples(tmp_path):
    destination = tmp_path / "new" / "ex"
    written = copy_examples(destination)
    assert (destination / "simple.yaml").is_file()
    assert destination / "simple.yaml" in written
    assert not (destination / "__init__.py").exists()
    with pytest.raises(FileExistsError):
        copy_examples(destination)
