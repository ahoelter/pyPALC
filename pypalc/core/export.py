"""CSV export of run results."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Sequence

import numpy as np

from .pipeline import RunResult, result_arrays


def _fmt(value) -> str:
    """Format a number so that it round-trips exactly."""
    return repr(float(value))


def _write(path: Path, header: Sequence[str], rows) -> Path:
    """Write ``header`` and ``rows`` (iterables of numbers or strings)."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow([c if isinstance(c, str) else _fmt(c)
                             for c in row])
    return path


def _sound_field_header(frequencies: np.ndarray) -> List[str]:
    """Header of a sound field file, one column per frequency."""
    return ["x_m", "y_m"] + ["spl_db_%dhz" % int(round(float(f)))
                             for f in frequencies]


def write_results(result: RunResult, out_dir: Path) -> List[Path]:
    """Write the results of a run as CSV files.

    Existing files are overwritten. Numbers are written with ``repr`` so
    that they round-trip exactly.

    Parameters
    ----------
    result : RunResult
        Result of :func:`pypalc.core.pipeline.run`.
    out_dir : pathlib.Path
        Output directory, created if missing.

    Returns
    -------
    list of pathlib.Path
        The written files: ``tilt_angles.csv``, ``reference_tilt_angles.csv``,
        ``spl_over_distance.csv``, ``homogeneity.csv``, ``sound_field.csv``,
        ``reference_sound_field.csv`` and, for target slope runs,
        ``target_slope.csv`` and ``target_reference.csv``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    a = result_arrays(result)
    files = []

    def cabinets(n):
        return [str(i + 1) for i in range(n)]

    n = len(a["tilt_angles_deg"])
    files.append(_write(
        out_dir / "tilt_angles.csv",
        ["cabinet", "tilt_angle_deg", "inter_cabinet_angle_deg"],
        zip(cabinets(n), a["tilt_angles_deg"], a["inter_cabinet_angles_deg"])))
    n_ref = len(a["reference_tilt_angles_deg"])
    files.append(_write(
        out_dir / "reference_tilt_angles.csv",
        ["cabinet", "tilt_angle_deg"],
        zip(cabinets(n_ref), a["reference_tilt_angles_deg"])))
    files.append(_write(
        out_dir / "spl_over_distance.csv",
        ["distance_m", "spl_db", "reference_spl_db", "x_m", "y_m"],
        zip(a["spl_over_distance_x"], a["spl_over_distance_db"],
            a["reference_spl_over_distance_db"], a["venue_x"],
            a["venue_y"])))
    if result.target is not None:
        t = result.target
        files.append(_write(
            out_dir / "target_slope.csv",
            ["distance_m", "level_db", "x_m", "y_m"],
            zip(t.distances, t.levels_db, t.points_x, t.points_y)))
        ref_distance = float(np.ravel(t.reference_distance)[0])
        ref_level = float(np.asarray(t.region_levels_db, dtype=float)
                          [int(np.ravel(t.reference_index)[0])])
        grid = np.asarray(a["spl_over_distance_x"], dtype=float)
        i = int(np.argmin(np.abs(grid - ref_distance)))
        files.append(_write(
            out_dir / "target_reference.csv",
            ["distance_m", "level_db", "x_m", "y_m"],
            [[ref_distance, ref_level, a["venue_x"][i], a["venue_y"][i]]]))
    files.append(_write(
        out_dir / "homogeneity.csv",
        ["frequency_hz", "homogeneity_db", "homogeneity_low_db",
         "homogeneity_high_db", "reference_homogeneity_db"],
        zip(a["frequencies_hz"], a["homogeneity_db"],
            a["homogeneity_low_db"], a["homogeneity_high_db"],
            a["reference_homogeneity_db"])))
    header = _sound_field_header(a["frequencies_hz"])
    for name, key in (("sound_field.csv", "spl_db"),
                      ("reference_sound_field.csv", "reference_spl_db")):
        files.append(_write(
            out_dir / name, header,
            ([x, y] + list(row) for x, y, row in
             zip(a["venue_x"], a["venue_y"], a[key]))))
    return files
