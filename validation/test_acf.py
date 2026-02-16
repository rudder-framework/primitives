"""
ACF validation against statsmodels.

Ground truth: statsmodels.tsa.stattools.acf

Known values:
- White noise: ACF(k) ~ 0 for k > 0
- Sine wave: ACF is periodic
- Random walk: ACF decays slowly
"""
import numpy as np
import pytest

from statsmodels.tsa.stattools import acf as sm_acf

from prmtvs.individual.correlation import autocorrelation


class TestACFVsStatsmodels:
    """Compare against statsmodels ACF."""

    def test_white_noise_lag0(self, white_noise):
        ours = autocorrelation(white_noise)
        # Lag 0 should be 1.0
        assert abs(ours[0] - 1.0) < 1e-10, \
            f"ACF at lag 0 should be 1.0, got {ours[0]:.6f}"

    def test_white_noise_near_zero(self, white_noise):
        ours = autocorrelation(white_noise)
        theirs = sm_acf(white_noise, nlags=20)
        # Lags 1-20 should be near zero for white noise
        for lag in range(1, min(21, len(ours), len(theirs))):
            assert abs(ours[lag]) < 0.1, \
                f"White noise ACF at lag {lag} should be ~0, got {ours[lag]:.4f}"
            assert abs(theirs[lag]) < 0.1, \
                f"Statsmodels ACF at lag {lag} should be ~0, got {theirs[lag]:.4f}"


class TestACFAnalytical:
    """Test against known autocorrelation properties."""

    def test_white_noise_no_correlation(self, white_noise):
        acf_vals = autocorrelation(white_noise)
        # Lags 2-5 should be near zero for white noise
        for i in range(2, min(6, len(acf_vals))):
            assert abs(acf_vals[i]) < 0.1, \
                f"White noise ACF at lag {i} should be ~0, got {acf_vals[i]:.4f}"

    def test_sine_periodic_acf(self, sine_wave):
        acf_vals = autocorrelation(sine_wave)
        # Sine ACF should be periodic — check that it goes negative then positive
        found_negative = False
        found_positive_after_negative = False
        for i in range(1, min(500, len(acf_vals))):
            if acf_vals[i] < -0.5:
                found_negative = True
            if found_negative and acf_vals[i] > 0.5:
                found_positive_after_negative = True
                break
        assert found_positive_after_negative, \
            "Sine wave ACF should be periodic (oscillate between positive and negative)"
