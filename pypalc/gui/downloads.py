"""The files page 4 downloads, written on the server.

* "Download PALC Angles": :func:`angles_csv`, the tilt angles as one CSV
  file (:data:`DOWNLOAD_FILE_NAMES` ``['results']``).
* "Download Results": :func:`results_zip`, a zip file
  (:data:`RESULTS_FILE_NAME`) with one CSV file per further entry of
  :data:`DOWNLOAD_FILE_NAMES`: the SPL over distance of the optimized and
  of the reference array, the target slope, its reference point and the
  homogeneity.
* "Save Setup": :func:`setup_yaml`, the setup file
  (:data:`SETUP_FILE_NAME`).

The CSV files have the columns of :data:`DOWNLOAD_COLUMNS`, the column
names of the headless export (:mod:`pypalc.core.export`), and write the
numbers as it does (they round-trip exactly). Before the first successful
calculation of a session (``state.result`` is None) the CSV files hold
the header only: the sources show placeholders then, not results.

author: Arne Hoelter
"""
from __future__ import annotations

import csv
import io
import zipfile
from typing import TYPE_CHECKING, Dict, Optional, Sequence

from bokeh.models import ColumnDataSource

from pypalc.core import export
from pypalc.core.errors import SetupError
from pypalc.core.settings import dump_setup, validate
from pypalc.gui.setup_io import setup_to_save

if TYPE_CHECKING:
    from .page_algorithm import AlgorithmPage

#: Names of the CSV files: "Download PALC Angles" writes ``results``,
#: "Download Results" the others, by source.
DOWNLOAD_FILE_NAMES = {
    'results': 'tilt_angles.csv',
    'spl_over_distance': 'spl_over_distance.csv',
    'target': 'target_slope.csv',
    'target_reference': 'target_reference.csv',
    'reference_spl_over_distance': 'reference_spl_over_distance.csv',
    'homogeneity': 'homogeneity.csv',
}
#: Columns of the CSV files, in the order of the headless export (the two
#: SPL over distance files leave out the other array's column).
DOWNLOAD_COLUMNS = {
    'results': export.TILT_ANGLES_HEADER,
    'spl_over_distance': [c for c in export.SPL_OVER_DISTANCE_HEADER
                          if c != 'reference_spl_db'],
    'target': export.TARGET_SLOPE_HEADER,
    'target_reference': export.TARGET_REFERENCE_HEADER,
    'reference_spl_over_distance': [c for c in export.SPL_OVER_DISTANCE_HEADER
                                    if c != 'spl_db'],
    'homogeneity': export.HOMOGENEITY_HEADER,
}
#: Name of the zip file "Download Results" downloads.
RESULTS_FILE_NAME = 'pypalc_results.zip'
#: Name of the file "Save Setup" downloads.
SETUP_FILE_NAME = 'pypalc_setup.yaml'


def _sources(page: AlgorithmPage) -> Dict[str, ColumnDataSource]:
    """Key of :data:`DOWNLOAD_FILE_NAMES` -> the source written there."""
    return {'results': page.results_source,
            'spl_over_distance': page.spl_over_distance_source,
            'target': page.target_source,
            'target_reference': page.target_reference_source,
            'reference_spl_over_distance':
                page.reference_spl_over_distance_source,
            'homogeneity': page.app.pages[5].homogeneity_source}


def _cell(column: str, value) -> str:
    """Text of a value: the cabinet number as an integer (as the headless
    export writes it), other numbers exactly (``export._fmt``)."""
    if column == 'cabinet':
        return str(int(value))
    return export._fmt(value)


def _csv(source: ColumnDataSource, columns: Sequence[str],
         rows: bool) -> str:
    """CSV text of the columns of a source, with or without its rows."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator='\n')
    writer.writerow(columns)
    if rows:
        data = source.data
        for i in range(len(data[columns[0]])):
            writer.writerow([_cell(column, data[column][i])
                             for column in columns])
    return buffer.getvalue()


def _file(page: AlgorithmPage, key: str) -> str:
    """CSV text of the file of a key of :data:`DOWNLOAD_FILE_NAMES`."""
    return _csv(_sources(page)[key], DOWNLOAD_COLUMNS[key],
                page.app.state.result is not None)


def angles_csv(page: AlgorithmPage) -> str:
    """The file of "Download PALC Angles".

    Parameters
    ----------
    page : AlgorithmPage
        Page 4 of the session.

    Returns
    -------
    str
        CSV text: the header :data:`DOWNLOAD_COLUMNS` ``['results']`` and
        one row per cabinet of ``results_source`` (header only before the
        first calculation).
    """
    return _file(page, 'results')


def results_zip(page: AlgorithmPage) -> bytes:
    """The file of "Download Results".

    Parameters
    ----------
    page : AlgorithmPage
        Page 4 of the session.

    Returns
    -------
    bytes
        A zip file with one CSV file per entry of
        :data:`DOWNLOAD_FILE_NAMES` except the tilt angles, in that order,
        with the columns of :data:`DOWNLOAD_COLUMNS` (header only before
        the first calculation).
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for key, name in DOWNLOAD_FILE_NAMES.items():
            if key != 'results':
                archive.writestr(name, _file(page, key))
    return buffer.getvalue()


def setup_yaml(page: AlgorithmPage) -> Optional[str]:
    """The file of "Save Setup".

    The setup of :func:`~.setup_io.setup_to_save`, validated; an uploaded
    measured directivity is embedded. A setup that cannot be read or is
    invalid is not saved: the reason is shown in the error text
    (``config_error_info``).

    Parameters
    ----------
    page : AlgorithmPage
        Page 4 of the session.

    Returns
    -------
    str or None
        YAML text of the setup, or None if it cannot be saved.
    """
    app = page.app
    try:
        setup = setup_to_save(app)
        validate(setup)
    except SetupError as exc:
        app.config_error_info.object = \
            'Error: setup not saved, invalid input at %s: %s' \
            % (exc.key, exc.message)
        return None
    return dump_setup(setup)
