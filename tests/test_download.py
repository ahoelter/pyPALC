"""The downloads of page 4: "Download PALC Angles" (one CSV), "Download
Results" (a zip of five CSVs) and "Save Setup" (the YAML setup file),
written on the server by :mod:`pypalc.gui.downloads`."""
import base64
import contextlib
import csv
import io
import zipfile

import panel_material_ui as pmui
import pytest
from bokeh.document import Document

from pypalc.core import export
from pypalc.core.settings import load_setup, load_setup_text
from pypalc.gui.app import PalcApp
from pypalc.gui.downloads import (RESULTS_FILE_NAME, angles_csv,
                                  results_zip, setup_yaml)
from pypalc.gui.page_algorithm import (DOWNLOAD_COLUMNS, DOWNLOAD_FILE_NAMES,
                                       SETUP_FILE_NAME)
from pypalc.gui.setup_io import apply_setup, setup_to_save
from tests.baseline_cases import case_path
from tests.gui_driver import read
from tests.test_gui_equivalence import drive, enter

#: Files of the zip of "Download Results", in this order.
RESULT_FILES = [name for key, name in DOWNLOAD_FILE_NAMES.items()
                if key != 'results']


def quiet(function, *args):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args)


@pytest.fixture(scope='module')
def ran():
    """case -> a session in which the case was entered and run by hand."""
    apps = {}

    def get(case):
        if case not in apps:
            apps[case] = PalcApp(Document())
            quiet(drive, apps[case], load_setup(case_path(case)))
        return apps[case]
    return get


def _sources(app):
    """File name -> (source, columns) of every download."""
    algorithm, results = app.pages[4], app.pages[5]
    sources = {'results': algorithm.results_source,
               'spl_over_distance': algorithm.spl_over_distance_source,
               'target': algorithm.target_source,
               'target_reference': algorithm.target_reference_source,
               'reference_spl_over_distance':
                   algorithm.reference_spl_over_distance_source,
               'homogeneity': results.homogeneity_source}
    return {DOWNLOAD_FILE_NAMES[key]: (source, DOWNLOAD_COLUMNS[key])
            for key, source in sources.items()}


def _rows(text):
    return list(csv.reader(io.StringIO(text)))


def _zip_texts(data):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        return {info.filename: archive.read(info).decode('utf-8')
                for info in archive.infolist()}


def _assert_equals_source(text, source, columns):
    """The CSV text has the columns as header and every value of the
    source in its rows; each number parses back to the source value."""
    header, *rows = _rows(text)
    assert header == list(columns)
    data = source.data
    assert len(rows) == len(data[columns[0]]) > 0
    for i, row in enumerate(rows):
        for column, cell in zip(columns, row):
            assert float(cell) == float(data[column][i]), (column, i)


def test_every_download_has_its_own_file_name():
    assert len(set(DOWNLOAD_FILE_NAMES.values())) == len(DOWNLOAD_FILE_NAMES)
    assert RESULTS_FILE_NAME not in DOWNLOAD_FILE_NAMES.values()


@pytest.mark.parametrize("case", ["simple", "target_slope"])
def test_angles_csv_matches_results_source(case, ran):
    app = ran(case)
    source, columns = _sources(app)['tilt_angles.csv']
    _assert_equals_source(angles_csv(app.pages[4]), source, columns)


def test_angles_csv_equals_headless_export(ran, tmp_path):
    """The angles file of the GUI is the file the headless runner writes
    for the same result."""
    app = ran('simple')
    export.write_results(app.state.result, tmp_path)
    assert angles_csv(app.pages[4]) == \
        (tmp_path / 'tilt_angles.csv').read_text(encoding='utf-8')


def test_results_zip_has_five_csvs_with_headers(ran):
    texts = _zip_texts(results_zip(ran('target_slope').pages[4]))
    assert list(texts) == RESULT_FILES
    for key, name in DOWNLOAD_FILE_NAMES.items():
        if key != 'results':
            assert _rows(texts[name])[0] == list(DOWNLOAD_COLUMNS[key])


@pytest.mark.parametrize("case", ["simple", "target_slope"])
def test_results_zip_values_equal_sources(case, ran):
    app = ran(case)
    texts = _zip_texts(results_zip(app.pages[4]))
    for name, (source, columns) in _sources(app).items():
        if name in texts:
            _assert_equals_source(texts[name], source, columns)


def test_downloads_before_run_are_header_only():
    algorithm = PalcApp(Document()).pages[4]
    assert angles_csv(algorithm) == ','.join(DOWNLOAD_COLUMNS['results']) \
        + '\n'
    texts = _zip_texts(results_zip(algorithm))
    assert list(texts) == RESULT_FILES
    for key, name in DOWNLOAD_FILE_NAMES.items():
        if key != 'results':
            assert texts[name] == ','.join(DOWNLOAD_COLUMNS[key]) + '\n'


@pytest.mark.parametrize("case", ["simple", "target_slope", "measured"])
def test_save_setup_round_trips(case, ran):
    app = ran(case)
    text = setup_yaml(app.pages[4])
    loaded = PalcApp(Document())
    quiet(apply_setup, loaded, load_setup_text(text))
    assert loaded.state.result is not None
    assert setup_to_save(loaded) == setup_to_save(app)


def test_setup_yaml_shows_error_for_invalid_input():
    app = PalcApp(Document())
    app.pages[2].directivity_select.value = "Measured Loudspeaker Data"
    assert setup_yaml(app.pages[4]) is None
    assert read(app.config_error_info, 'text').startswith(
        "Error: setup not saved, invalid input at "
        "loudspeaker.measured_directivity: ")


def _fetch(button):
    """Click a download button as the browser does; the bytes sent to the
    browser (None if nothing was sent)."""
    sent = []
    watcher = button.param.watch(lambda event: sent.append(event.new),
                                 'data')
    try:
        button._clicks += 1
    finally:
        button.param.unwatch(watcher)
    sent = [data for data in sent if data is not None]
    assert len(sent) <= 1
    return base64.b64decode(sent[0].split(',', 1)[1]) if sent else None


def test_download_buttons(ran):
    app = ran('target_slope')
    algorithm = app.pages[4]
    buttons = {'download_angles_button': ('Download PALC Angles',
                                          'tilt_angles.csv'),
               'download_results_button': ('Download Results',
                                           RESULTS_FILE_NAME),
               'save_setup_button': ('Save Setup', SETUP_FILE_NAME)}
    for name, (label, filename) in buttons.items():
        button = getattr(algorithm, name)
        assert isinstance(button, pmui.FileDownload), name
        assert (button.label, button.filename) == (label, filename), name
    assert _fetch(algorithm.download_angles_button).decode() == \
        angles_csv(algorithm)
    assert _zip_texts(_fetch(algorithm.download_results_button)) == \
        _zip_texts(results_zip(algorithm))
    assert load_setup_text(_fetch(algorithm.save_setup_button).decode()) \
        == load_setup_text(setup_yaml(algorithm))
    load = algorithm.load_setup_input
    assert isinstance(load, pmui.FileInput)
    assert load.accept == '.yaml,.yml'


def test_download_twice_sends_twice(ran):
    """Each click sends the file again, also when it did not change (the
    browser downloads when the data changes)."""
    button = ran('simple').pages[4].download_angles_button
    first, second = _fetch(button), _fetch(button)
    assert first is not None and first == second


def test_invalid_setup_downloads_nothing():
    app = PalcApp(Document())
    app.pages[2].directivity_select.value = "Measured Loudspeaker Data"
    assert _fetch(app.pages[4].save_setup_button) is None
    assert read(app.config_error_info, 'text').startswith("Error: ")


@pytest.mark.parametrize("case", ["simple", "target_slope"])
def test_download_headers(case):
    """The downloaded sources have the columns of the export files, in the
    same order, in a new session and after a calculation."""
    app = PalcApp(Document())
    for name, (source, columns) in _sources(app).items():
        assert list(source.data) == list(columns), name
    quiet(drive, app, load_setup(case_path(case)))
    for name, (source, columns) in _sources(app).items():
        assert list(source.data) == list(columns), name


def test_enter_without_run_downloads_no_results():
    """Entered but not run: no results to download yet."""
    app = PalcApp(Document())
    quiet(enter, app, load_setup(case_path('simple')))
    assert angles_csv(app.pages[4]) == \
        ','.join(DOWNLOAD_COLUMNS['results']) + '\n'
