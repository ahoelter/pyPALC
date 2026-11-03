"""Stop conditions of the target slope optimization."""
import types

import numpy as np

from pypalc.core import target_slope
from pypalc.core.pipeline import run
from pypalc.core.settings import load_setup
from pypalc.core.target_slope import TargetSlopeState
from tests.baseline_cases import case_path


def _started():
    """A target slope state as at the start of an optimization."""
    state = TargetSlopeState()
    state.reset_optimization()
    return state


def _config(weights):
    return types.SimpleNamespace(weighting_weights=np.array(weights))


def test_equal_differences_before_a_worse_loop_stop_with_current_weights():
    state, config = _started(), _config([1.0, 0.9, 0.8])
    stops = [state.store_previous(config, 2.5, 20) for _ in range(6)]
    assert stops == [False] * 5 + [True]
    np.testing.assert_array_equal(config.weighting_weights, [1.0, 0.9, 0.8])


def test_no_convergence_in_the_first_loop_stops_with_current_weights():
    state, config = _started(), _config([1.0, 0.7])
    assert state.store_previous(config, 2.5, 500) is True
    np.testing.assert_array_equal(config.weighting_weights, [1.0, 0.7])


def test_stop_restores_the_weights_of_the_lowest_difference():
    state = _started()
    config = _config([1.0, 0.5])
    assert state.store_previous(config, 2.0, 20) is False
    config.weighting_weights = np.array([1.0, 0.6])
    assert state.store_previous(config, 3.0, 20) is False
    config.weighting_weights = np.array([1.0, 0.7])
    assert state.store_previous(config, 3.0, 500) is True
    np.testing.assert_array_equal(config.weighting_weights, [1.0, 0.5])


def test_run_stops_cleanly_when_the_difference_does_not_change(monkeypatch):
    stops = []
    store_previous = TargetSlopeState.store_previous

    def recorded(self, *args):
        stops.append(store_previous(self, *args))
        return stops[-1]

    monkeypatch.setattr(target_slope, "single_value_difference",
                        lambda *args, **kwargs: (1.0, 1.0))
    monkeypatch.setattr(TargetSlopeState, "store_previous", recorded)
    result = run(load_setup(case_path("target_slope")))
    assert stops == [False] * 5 + [True]
    assert result.target is not None
    assert len(result.optimized.tilt_angles_deg) > 0
