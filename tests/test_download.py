"""The CSV downloads of page 4 ("Download PALC Angles", "Download SPL")."""
from bokeh.document import Document
from bokeh.models import CustomJS

from pypalc.gui.app import PalcApp
from pypalc.gui.page_algorithm import DOWNLOAD_FILE_NAMES


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
