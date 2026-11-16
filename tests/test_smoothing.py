"""Fractional octave smoothing of the frequency responses
(:func:`pypalc.core.sound_field.smooth_fractional_octave`)."""
import numpy as np
import pytest

from pypalc.core.sound_field import frequency_vector, smooth_fractional_octave

#: The frequencies of the sound field prediction: 1/12 octave steps.
FREQUENCIES = frequency_vector(frequency_count=120, octave_step=1/12,
                               frequency_range_hz=[20, 20000])


def _peak(index, power=4.0):
    """SPL in dB with the power ``power`` at ``index`` and 1 elsewhere."""
    spl = np.zeros(len(FREQUENCIES))
    spl[index] = 10 * np.log10(power)
    return spl


@pytest.mark.parametrize('fraction', [None, 3, 1])
def test_a_constant_response_stays_constant(fraction):
    spl = np.full(len(FREQUENCIES), 87.5)
    np.testing.assert_allclose(
        smooth_fractional_octave(FREQUENCIES, spl, fraction), spl)


def test_no_smoothing_returns_the_response_unchanged():
    spl = np.random.default_rng(1).uniform(60, 100, len(FREQUENCIES))
    smoothed = smooth_fractional_octave(FREQUENCIES, spl, None)
    assert np.array_equal(smoothed, spl)
    assert smoothed is not spl


@pytest.mark.parametrize('fraction, half_bins', [(3, 2), (1, 6)])
def test_the_window_spans_the_fraction_of_an_octave(fraction, half_bins):
    """A 1/b octave window on the 1/12 octave grid covers 12/b bins: the
    bins within +-(6/b - 1) of the center fully, the two at +-6/b half;
    the power (not the level) is averaged."""
    k = 60
    smoothed = smooth_fractional_octave(FREQUENCIES, _peak(k), fraction)
    bins = 12 / fraction
    power = 10 ** (smoothed / 10)
    # the peak of power 4 is fully in the windows of k-(half_bins-1)..
    # k+(half_bins-1)
    for offset in range(-(half_bins - 1), half_bins):
        assert power[k + offset] == pytest.approx((bins + 3) / bins)
    # half in the windows of k +- half_bins
    for offset in (-half_bins, half_bins):
        assert power[k + offset] == pytest.approx((bins + 1.5) / bins)
    # not in the windows further away
    for offset in (-half_bins - 1, half_bins + 1):
        assert power[k + offset] == pytest.approx(1)


def test_the_window_is_cut_at_the_ends():
    """At the lowest frequency the window holds only the bins above it."""
    smoothed = smooth_fractional_octave(FREQUENCIES, _peak(0), 3)
    # the window reaches 1/6 octave below 20 Hz, but there is no bin: it
    # holds bin 0 and bin 1 fully and bin 2 half (2.5 bins)
    power = 10 ** (smoothed / 10)
    assert power[0] == pytest.approx((2.5 + 3) / 2.5)


def test_rows_are_smoothed_independently():
    responses = np.vstack([_peak(30), _peak(90), np.full(len(FREQUENCIES),
                                                         70.0)])
    smoothed = smooth_fractional_octave(FREQUENCIES, responses, 3)
    assert smoothed.shape == responses.shape
    for row, response in zip(smoothed, responses):
        np.testing.assert_allclose(
            row, smooth_fractional_octave(FREQUENCIES, response, 3))
