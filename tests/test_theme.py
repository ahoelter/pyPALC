"""``write_theme`` writes the committed theme files byte for byte."""
from pathlib import Path

from pypalc.gui.theme import write_theme

ROOT = Path(__file__).resolve().parent.parent


def test_write_theme_is_byte_identical(tmp_path):
    (tmp_path / 'templates').mkdir()
    write_theme(tmp_path)
    for name in ('theme.yaml', 'templates/index.html'):
        assert (tmp_path / name).read_bytes() == (ROOT / name).read_bytes(), name
