"""
ADF test validation against statsmodels.

Ground truth: statsmodels.tsa.stattools.adfuller

Known values:
- White noise (stationary):   reject null (p < 0.05)
- Random walk (non-stationary): fail to reject (p > 0.05)
"""
import numpy as np
import pytest

from statsmodels.tsa.stattools import adfuller as sm_adfuller

from pmtvs.stat_tests.stationarity_tests import adf_test


class TestADFVsStatsmodels:
    """Compare against statsmodels ADF."""

    def test_stationary_detection(self, white_noise):
        # adf_test returns (statistic, pvalue, lags, critical_values)
        ours = adf_test(white_noise)
        theirs = sm_adfuller(white_noise)
        # Both should reject null (stationary signal)
        assert ours[1] < 0.05, f"Our ADF p={ours[1]:.4f}"
        assert theirs[1] < 0.05, f"Statsmodels ADF p={theirs[1]:.4f}"

    def test_nonstationary_detection(self, random_walk):
        ours = adf_test(random_walk)
        theirs = sm_adfuller(random_walk)
        # Both should fail to reject null (non-stationary signal)
        assert ours[1] > 0.05, f"Our ADF p={ours[1]:.4f}"
        assert theirs[1] > 0.05, f"Statsmodels ADF p={theirs[1]:.4f}"


class TestADFAnalytical:
    """Test against known stationarity properties."""

    def test_white_noise_stationary(self, white_noise):
        result = adf_test(white_noise)
        assert result[1] < 0.05, \
            f"White noise should be stationary, got p={result[1]:.4f}"

    def test_random_walk_nonstationary(self, random_walk):
        result = adf_test(random_walk)
        assert result[1] > 0.05, \
            f"Random walk should be non-stationary, got p={result[1]:.4f}"

    def test_trend_nonstationary(self, linear_trend):
        result = adf_test(linear_trend)
        assert result[1] > 0.05, \
            f"Linear trend should be non-stationary, got p={result[1]:.4f}"
