"""
Granger causality validation against statsmodels.

Ground truth: statsmodels.tsa.stattools.grangercausalitytests

Known values:
- X causes Y (lagged): Granger detects (p < 0.05)
- Independent signals:  Granger fails to detect (p > 0.05)
"""
import numpy as np
import pytest

from pmtvs.pairwise.causality import granger_causality


class TestGrangerAnalytical:
    """Test against known causal relationships."""

    def test_causal_detected(self, causal_pair):
        x, y = causal_pair
        # granger_causality returns (F_stat, pvalue, lag)
        result = granger_causality(x, y)
        f_stat, pvalue, lag = result[0], result[1], result[2]
        assert pvalue < 0.05, \
            f"Should detect causality: F={f_stat:.4f}, p={pvalue:.4f}"

    def test_independent_not_detected(self, uncorrelated_pair):
        x, y = uncorrelated_pair
        result = granger_causality(x, y)
        f_stat, pvalue, lag = result[0], result[1], result[2]
        assert pvalue > 0.05, \
            f"Should NOT detect causality: F={f_stat:.4f}, p={pvalue:.4f}"

    def test_returns_three_values(self, causal_pair):
        x, y = causal_pair
        result = granger_causality(x, y)
        assert len(result) == 3, \
            f"granger_causality should return 3 values, got {len(result)}"
        # F-stat should be positive
        assert result[0] > 0, f"F-statistic should be positive, got {result[0]}"
        # p-value in [0, 1]
        assert 0 <= result[1] <= 1, f"p-value out of range: {result[1]}"
