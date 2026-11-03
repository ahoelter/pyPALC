"""The CSV downloads of page 4 ("Download PALC Angles", "Download SPL")."""
import contextlib
import io

import pytest
from bokeh.document import Document
from bokeh.models import CustomJS

from pypalc.core import export
from pypalc.core.settings import load_setup
from pypalc.gui.app import PalcApp
from pypalc.gui.page_algorithm import DOWNLOAD_FILE_NAMES
from tests.baseline_cases import case_path
from tests.test_gui_equivalence import drive


def _downloads(button):
    """(source, filename) of the CustomJS callbacks of a button."""
    return [(cb.args['source'], cb.args.get('filename'))
            for cb in button.js_event_callbacks['button_click']
            if isinstance(cb, CustomJS)]


def test_every_download_has_its_own_file_name():
    app = PalcApp(Document())
    algorithm, results = app.pages[4], app.pages[5]
    assert _downloads(algorithm.download_angles_button) == [
        (algorithm.results_source, 'tilt_angles.csv')]
    assert _downloads(algorithm.download_spl_button) == [
        (algorithm.spl_over_distance_source, 'spl_over_distance.csv'),
        (algorithm.target_source, 'target_slope.csv'),
        (algorithm.target_reference_source, 'target_reference.csv'),
        (algorithm.reference_spl_over_distance_source,
         'reference_spl_over_distance.csv'),
        (results.homogeneity_source, 'homogeneity.csv')]
    assert len(set(DOWNLOAD_FILE_NAMES.values())) == len(DOWNLOAD_FILE_NAMES)


def _columns(source):
    return list(source.data.keys())


def _without(header, column):
    return [c for c in header if c != column]


def _assert_download_headers(app):
    algorithm, results = app.pages[4], app.pages[5]
    assert _columns(algorithm.results_source) == export.TILT_ANGLES_HEADER
    assert _columns(algorithm.spl_over_distance_source) == \
        _without(export.SPL_OVER_DISTANCE_HEADER, 'reference_spl_db')
    assert _columns(algorithm.reference_spl_over_distance_source) == \
        _without(export.SPL_OVER_DISTANCE_HEADER, 'spl_db')
    assert _columns(algorithm.target_source) == export.TARGET_SLOPE_HEADER
    assert _columns(algorithm.target_reference_source) == \
        export.TARGET_REFERENCE_HEADER
    assert _columns(results.homogeneity_source) == export.HOMOGENEITY_HEADER


@pytest.mark.parametrize("case", ["simple", "target_slope"])
def test_download_headers(case):
    """The downloaded sources have the columns of the export files, in the
    same order, in a new session and after a calculation."""
    app = PalcApp(Document())
    _assert_download_headers(app)
    with contextlib.redirect_stdout(io.StringIO()):
        drive(app, load_setup(case_path(case)))
    _assert_download_headers(app)
