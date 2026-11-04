"""``write_theme`` writes the committed theme files byte for byte, and
Bokeh can load the generated ``theme.yaml``."""
from pathlib import Path

import yaml
from bokeh.models import LinearAxis
from bokeh.plotting import figure
from bokeh.themes import Theme

from pypalc.gui.theme import BACKGROUND_COLOR, TEXT_COLOR, write_theme

ROOT = Path(__file__).resolve().parent.parent


def test_write_theme_is_byte_identical(tmp_path):
    (tmp_path / 'templates').mkdir()
    write_theme(tmp_path)
    for name in ('theme.yaml', 'templates/index.html'):
        assert (tmp_path / name).read_bytes() == (ROOT / name).read_bytes(), name


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects a key given twice in one mapping (PyYAML
    keeps the last value silently)."""

    def construct_mapping(self, node, deep=False):
        keys = [self.construct_object(k, deep=deep) for k, _ in node.value]
        duplicates = sorted({k for k in keys if keys.count(k) > 1})
        assert not duplicates, 'duplicate keys %s at %s' % (duplicates,
                                                             node.start_mark)
        return super().construct_mapping(node, deep=deep)


def test_theme_yaml_loads_in_bokeh(tmp_path):
    (tmp_path / 'templates').mkdir()
    write_theme(tmp_path)
    path = tmp_path / 'theme.yaml'
    yaml.load(path.read_text(encoding='utf-8'), Loader=_UniqueKeyLoader)
    theme = Theme(filename=str(path))
    plot, axis = figure(), LinearAxis()
    theme.apply_to_model(plot)
    theme.apply_to_model(axis)
    assert plot.background_fill_color == BACKGROUND_COLOR
    assert axis.minor_tick_line_color == TEXT_COLOR
