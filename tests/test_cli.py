"""Tests of the headless runner."""
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pypalc.cli import main

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "tests" / "cases"
BASELINE = ROOT / "tests" / "baseline"
SOUND_HEADER_START = ["x_m", "y_m"]


def _read(path):
    with open(str(path), newline="") as handle:
        rows = list(csv.reader(handle))
    return rows[0], rows[1:]


def _column(path, name):
    header, rows = _read(path)
    i = header.index(name)
    return np.array([float(r[i]) for r in rows])


def test_cli_writes_csvs(tmp_path):
    assert main(["run", str(CASES / "simple.yaml"),
                 "--out", str(tmp_path / "new" / "out")]) == 0
    out = tmp_path / "new" / "out"
    base = np.load(str(BASELINE / "simple.npz"))
    headers = {
        "tilt_angles.csv":
            ["cabinet", "tilt_angle_deg", "inter_cabinet_angle_deg"],
        "reference_tilt_angles.csv": ["cabinet", "tilt_angle_deg"],
        "spl_over_distance.csv":
            ["distance_m", "spl_db", "reference_spl_db", "x_m", "y_m"],
        "homogeneity.csv":
            ["frequency_hz", "homogeneity_db", "homogeneity_low_db",
             "homogeneity_high_db", "reference_homogeneity_db"],
    }
    for name, header in headers.items():
        assert _read(out / name)[0] == header
    for name in ("sound_field.csv", "reference_sound_field.csv"):
        header, rows = _read(out / name)
        assert header[:2] == SOUND_HEADER_START
        assert header[2:] == ["spl_db_%dhz" % round(f)
                              for f in base["frequencies_hz"]]
        assert len(rows) == len(base["venue_x"])
    assert not (out / "target_slope.csv").exists()
    assert list(_column(out / "tilt_angles.csv", "tilt_angle_deg")) \
        == list(base["tilt_angles_deg"])
    assert _read(out / "tilt_angles.csv")[1][0][0] == "1"
    assert list(_column(out / "spl_over_distance.csv", "spl_db")) \
        == list(base["spl_over_distance_db"])
    sf = _read(out / "sound_field.csv")[1]
    assert np.array([[float(v) for v in r[2:]] for r in sf]).tolist() \
        == base["spl_db"].tolist()


def test_cli_invalid_setup_exit_code(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("version: 1\narray:\n  cabinet_count: 0\n")
    assert main(["run", str(bad), "--out", str(tmp_path / "o")]) == 2
    assert "array.cabinet_count" in capsys.readouterr().err


def test_cli_missing_setup_file(tmp_path, capsys):
    assert main(["run", str(tmp_path / "nope.yaml"),
                 "--out", str(tmp_path / "o")]) == 2
    assert "nope.yaml" in capsys.readouterr().err


def test_cli_resolves_relative_directivity_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["run", str(CASES / "measured.yaml"),
                 "--out", str(tmp_path / "o")]) == 0


def test_cli_target_slope_writes_target_files(tmp_path):
    assert main(["run", str(CASES / "target_slope.yaml"),
                 "--out", str(tmp_path)]) == 0
    base = np.load(str(BASELINE / "target_slope.npz"))
    header, rows = _read(tmp_path / "target_slope.csv")
    assert header == ["distance_m", "level_db", "x_m", "y_m"]
    assert len(rows) == len(base["target_x"])
    assert list(_column(tmp_path / "target_slope.csv", "distance_m")) \
        == list(base["target_x"])
    assert list(_column(tmp_path / "target_slope.csv", "level_db")) \
        == list(base["target_db"])
    header, rows = _read(tmp_path / "target_reference.csv")
    assert header == ["distance_m", "level_db", "x_m", "y_m"]
    assert len(rows) == 1


def test_python_m_entry_point(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "pypalc", "run",
         str(CASES / "simple.yaml"), "--out", str(tmp_path)],
        cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "tilt_angles.csv").exists()
