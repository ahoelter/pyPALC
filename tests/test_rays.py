"""The rays drawn in the venue figure end on the venue lines.

PALC assumes that the start ray of a cabinet hits the venue where the stop
ray of the cabinet above hit it. With discrete tilt angles or a strong soft
margin the cabinet does not point there, so the plot must end each ray
where the ray itself crosses a venue line, not at the assumed point.
"""
import contextlib
import io

import numpy as np
import pytest

from pypalc.core.pipeline import run
from pypalc.core.settings import load_setup_text

SETUP = """version: 1
venue:
  audience_lines:
  - {start: [2.0, 0.0], stop: [10.0, 1.0], points_per_meter: 5.0}
  - {start: [12.0, 2.0], stop: [20.0, 4.0], points_per_meter: 5.0}
array:
  cabinet_count: 6
%s
algorithm:
  gap_handling: %s
  soft_margin_strength: %s
"""
DISCRETE = "  discrete_tilt_angles_deg: [0.0, 0.5, 1.0, 2.0, 3.5, 5.0, 7.0, 10.0]"


def _distance_to_venue(x, y, segments):
    """Distance of a point to the nearest venue line."""
    best = np.inf
    for segment in segments:
        a, b = segment[0, :2], segment[-1, :2]
        d = b - a
        t = np.clip(np.dot([x, y] - a, d) / np.dot(d, d), 0.0, 1.0)
        best = min(best, float(np.hypot(*([x, y] - a - t * d))))
    return best


@pytest.mark.parametrize("gap_handling, strength, discrete, passes_over", [
    # the start ray of cabinet 3 rises at 7.4 deg and passes over the
    # far end of the venue (20, 4)
    ("soft_margin", 1.9, True, {("start", 2)}),
    ("soft_margin", 1.9, False, set()),
    ("soft_margin", 0.5, False, set()),
    ("without", 1.0, True, set()),
    ("hard_margin", 1.0, False, set()),
])
def test_rays_end_on_venue_lines(gap_handling, strength, discrete,
                                 passes_over):
    setup = load_setup_text(SETUP % (DISCRETE if discrete else "",
                                     gap_handling, strength))
    with contextlib.redirect_stdout(io.StringIO()):
        result = run(setup)
    segments = result.venue.palc_venue.segments
    rays = result.ray_plot
    tilt = np.ravel(result.geometry.tilt_angles)
    aperture = np.ravel(result.geometry.aperture_angles)
    if aperture.shape != tilt.shape:
        # hard margin keeps the aperture angles of the last group only;
        # the direction is then checked against the uncorrected end
        aperture = None
    sign = {"start": 1, "center": 0, "stop": -1}
    for name in sign:
        xs = getattr(rays, "%s_rays_x" % name)
        ys = getattr(rays, "%s_rays_y" % name)
        assert len(xs) == len(tilt) == setup.array.cabinet_count
        for n, ((x0, x1), (y0, y1)) in enumerate(zip(xs, ys)):
            if (name, n) in passes_over:
                # misses the venue: ends above its farthest point
                assert x1 == pytest.approx(20.0) and y1 > 4.0
            else:
                assert _distance_to_venue(x1, y1, segments) < 1e-9, (name, n)
            if aperture is None:
                continue
            angle = -tilt[n] + sign[name] * aperture[n]
            # the end lies on the ray, in front of the cabinet
            assert np.cos(angle) * (x1 - x0) + np.sin(angle) * (y1 - y0) > 0
            assert abs(-np.sin(angle) * (x1 - x0)
                       + np.cos(angle) * (y1 - y0)) < 1e-9, (name, n)


def test_ray_end_on_venue_takes_the_first_crossing():
    from pypalc.core.palc import ray_end_on_venue
    near = np.array([[5.0, -5.0], [5.0, 5.0]])
    far = np.array([[10.0, -5.0], [10.0, 5.0]])
    # the ray along +x from the origin, given by any point on it
    assert ray_end_on_venue(0, 0, 20, 0, [far, near]) == (5.0, 0.0)
    assert ray_end_on_venue(0, 0, 1, 0, [far, near]) == (5.0, 0.0)
    # a line behind the start and a parallel line are not crossed
    behind = np.array([[-3.0, -5.0], [-3.0, 5.0]])
    parallel = np.array([[0.0, 1.0], [30.0, 1.0]])
    assert ray_end_on_venue(0, 0, 1, 0, [behind, parallel, far]) == (10.0, 0.0)


def test_ray_end_on_venue_when_the_ray_misses_the_venue():
    from pypalc.core.palc import ray_end_on_venue
    line = np.array([[5.0, 0.0], [10.0, 1.0]])
    # passes over the venue: ends above its farthest point
    x, y = ray_end_on_venue(0, 2, 1, 2.5, [line])
    assert (x, y) == pytest.approx((10.0, 7.0))
    # points away from the venue: keeps the given end
    assert ray_end_on_venue(0, 2, -1, 2, [line]) == (-1, 2)
