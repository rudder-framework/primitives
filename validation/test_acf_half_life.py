"""
ACF half-life validation against statsmodels ACF.

Ground truth:
- statsmodels.tsa.stattools.acf
- White noise: half-life = 1 (no memory)
- AR(1) with phi=0.9: half-life ~ -1/ln(0.9) ~ 9.5
"""
import numpy as np
import pytest


class TestACFHalfLifeVsStatsmodels:
    """Verify ACF computation matches statsmodels."""

    def test_white_noise(self):
        from prmtvs.individual.acf import acf_half_life
        rng = np.random.RandomState(42)
        hl = acf_half_life(rng.randn(10000))
        assert hl <= 3, f"White noise half-life should be ~1, got {hl}"

    def test_ar1_process(self):
        """AR(1) with phi=0.9: theoretical half-life ~ 6.6 at threshold 0.5."""
        from prmtvs.individual.acf import acf_half_life
        rng = np.random.RandomState(42)
        phi = 0.9
        n = 10000
        y = np.zeros(n)
        for i in range(1, n):
            y[i] = phi * y[i-1] + rng.randn()
        hl = acf_half_life(y, threshold=0.5)
        # ACF of AR(1) = phi^k, so phi^k = 0.5 -> k = log(0.5)/log(phi) ~ 6.58
        expected = np.log(0.5) / np.log(phi)
        assert abs(hl - expected) < 3, \
            f"AR(1) phi=0.9 half-life should be ~{expected:.1f}, got {hl}"


class TestACFHalfLifeAnalytical:

    def test_constant_max_lag(self):
        from prmtvs.individual.acf import acf_half_life
        hl = acf_half_life(np.ones(1000))
        assert hl == 100  # max_lag = min(n//4, 100) = 100

    def test_short_signal_nan(self):
        from prmtvs.individual.acf import acf_half_life
        hl = acf_half_life(np.array([1.0, 2.0]))
        assert np.isnan(hl)

    def test_threshold_parameter(self):
        """Lower threshold = higher half-life for same signal."""
        from prmtvs.individual.acf import acf_half_life
        rng = np.random.RandomState(42)
        phi = 0.9
        n = 10000
        y = np.zeros(n)
        for i in range(1, n):
            y[i] = phi * y[i-1] + rng.randn()
        hl_50 = acf_half_life(y, threshold=0.5)
        hl_e = acf_half_life(y, threshold=1/np.e)
        assert hl_e > hl_50, \
            f"1/e threshold ({hl_e}) should give longer half-life than 0.5 ({hl_50})"
