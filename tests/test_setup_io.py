"""The GUI builds a :class:`Setup` from its widgets and session state."""
import base64
import contextlib
import io
from pathlib import Path

import pytest
from bokeh.document import Document

from pypalc.core.settings import Setup, load_setup
from pypalc.gui.app import PalcApp
from pypalc.gui.setup_io import setup_from_app
from tests.baseline_cases import CASES
from tests.test_gui_equivalence import enter

ROOT = Path(__file__).resolve().parent.parent
NOT_MEASURED = tuple(case for case in CASES if case != "measured")


def test_setup_from_app_defaults():
    assert setup_from_app(PalcApp(Document())) == Setup()


@pytest.mark.parametrize("case", NOT_MEASURED)
def test_setup_from_app_case(case):
    setup = load_setup(ROOT / "tests" / "cases" / ("%s.yaml" % case))
    app = PalcApp(Document())
    app.show()
    with contextlib.redirect_stdout(io.StringIO()):
        enter(app, setup)
    assert setup_from_app(app) == setup


def test_setup_from_app_measured():
    csv_path = ROOT / "tests" / "cases" / "measured.csv"
    app = PalcApp(Document())
    loudspeaker = app.pages[2]
    loudspeaker.directivity_select.value = "Measured Loudspeaker Data"
    loudspeaker.directivity_upload.value = base64.b64encode(
        csv_path.read_bytes()).decode("ascii")
    setup = setup_from_app(app)
    assert setup.loudspeaker.directivity == "measured"
    assert setup.loudspeaker.measured_directivity.csv == \
        csv_path.read_text(encoding="utf-8")
    assert setup.loudspeaker.measured_directivity.path is None


def test_setup_lines_follow_drawn_lines():
    app = PalcApp(Document())
    venue = app.pages[1]

    def enter_line(start, stop, points_per_meter):
        venue.start_x_input.value = str(start[0])
        venue.end_x_input.value = str(stop[0])
        venue.points_per_meter_input.value = str(points_per_meter)
        venue.start_y_input.value = str(start[1])
        venue.end_y_input.value = str(stop[1])

    for start, stop, ppm in (((2, 0), (10, 1), 5), ((10, 1.5), (20, 4), 2),
                             ((20, 5), (30, 8), 3)):
        enter_line(start, stop, ppm)
        venue.save_line()
    venue.lines_source.selected.indices = [0]
    enter_line((0.5, 0), (1.5, 0), 4)
    venue.insert_line()
    assert len(setup_from_app(app).venue.audience_lines) == 4
    venue.lines_source.selected.indices = [2]
    venue.remove_selected_line()
    venue.lines_source.selected.indices = []
    venue.remove_line()
    drawn_lines = app.state.drawn_lines
    lines = setup_from_app(app).venue.audience_lines
    assert [(line.start, line.stop) for line in lines] == [
        ((x0, y0), (x1, y1)) for x0, y0, x1, y1 in zip(
            drawn_lines.line_start_x, drawn_lines.line_start_y, drawn_lines.line_stop_x,
            drawn_lines.line_stop_y)]
    assert [line.points_per_meter for line in lines] == [4.0, 5.0]
