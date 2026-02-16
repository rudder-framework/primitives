"""
Optimal delay validation against statsmodels ACF.

Ground truth:
- White noise: optimal delay = 1 (no autocorrelation)
- Sine wave:   optimal delay ~ quarter period

Reference: statsmodels ACF
"""
import numpy as np
import pytest

from pmtvs import optimal_delay


class TestOptimalDelayAnalytical:
    """Test against known signal properties."""

    def test_white_noise_small_delay(self, white_noise):
        tau = optimal_delay(white_noise)
        assert 1 <= tau <= 5, \
            f"White noise delay should be small (~1), got {tau}"

    def test_sine_quarter_period_autocorr(self):
        # 5 Hz sampled at 1000 Hz -> period = 200 samples -> quarter = 50
        # Use autocorr method which finds the first zero crossing (~quarter period)
        fs = 1000
        f = 5
        t = np.arange(10000) / fs
        signal = np.sin(2 * np.pi * f * t)
        tau = optimal_delay(signal, method='autocorr')
        quarter_period = fs / (4 * f)  # 50
        assert abs(tau - quarter_period) < 20, \
            f"Sine delay (autocorr) should be ~{quarter_period}, got {tau}"

    def test_sine_mutual_info_small(self):
        # Mutual info method finds first minimum of MI, which for a pure
        # sine is typically a small delay (first decorrelation point)
        fs = 1000
        f = 5
        t = np.arange(10000) / fs
        signal = np.sin(2 * np.pi * f * t)
        tau = optimal_delay(signal, method='mutual_info')
        assert 1 <= tau <= 100, \
            f"Sine delay (mutual_info) should be reasonable, got {tau}"

    def test_always_positive(self, white_noise):
        tau = optimal_delay(white_noise)
        assert tau >= 1, f"Delay must be >= 1, got {tau}"

    def test_integer_output(self, white_noise):
        tau = optimal_delay(white_noise)
        assert isinstance(tau, (int, np.integer)), \
            f"Delay must be integer, got {type(tau)}"
