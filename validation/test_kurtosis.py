"""
Kurtosis validation against scipy.stats.

Ground truth: scipy.stats.kurtosis (Fisher definition, excess kurtosis)

Known values:
- Normal distribution: kurtosis ~ 0 (excess)
- Uniform distribution: kurtosis ~ -1.2
- Laplace distribution: kurtosis ~ 3.0
"""
import numpy as np
import pytest

from scipy.stats import kurtosis as scipy_kurtosis

from primitives.individual.statistics import kurtosis


class TestKurtosisVsScipy:
    """Compare against scipy.stats.kurtosis."""

    def test_normal(self, white_noise):
        ours = kurtosis(white_noise)
        theirs = scipy_kurtosis(white_noise, fisher=True)
        assert abs(ours - theirs) < 0.01, f"ours={ours:.4f} scipy={theirs:.4f}"

    def test_uniform(self):
        rng = np.random.RandomState(42)
        data = rng.uniform(0, 1, 10000)
        ours = kurtosis(data)
        theirs = scipy_kurtosis(data, fisher=True)
        assert abs(ours - theirs) < 0.01, f"ours={ours:.4f} scipy={theirs:.4f}"


class TestKurtosisAnalytical:
    """Test against known distribution properties."""

    def test_normal_near_zero(self, white_noise):
        k = kurtosis(white_noise)
        assert abs(k) < 0.5, f"Normal excess kurtosis should be ~0, got {k:.4f}"

    def test_uniform_negative(self):
        rng = np.random.RandomState(42)
        data = rng.uniform(0, 1, 50000)
        k = kurtosis(data)
        assert k < 0, f"Uniform kurtosis should be negative (~-1.2), got {k:.4f}"
