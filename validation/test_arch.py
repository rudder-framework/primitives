"""
ARCH test validation against statsmodels.

Ground truth: statsmodels.stats.diagnostic.het_arch
"""
import numpy as np
import pytest


class TestArchVsStatsmodels:

    def test_garch_process(self):
        """Simulated GARCH(1,1): should detect ARCH effects."""
        from primitives.stat_tests.volatility import arch_test
        from statsmodels.stats.diagnostic import het_arch

        rng = np.random.RandomState(42)
        n = 2000
        y = np.zeros(n)
        sigma2 = np.zeros(n)
        sigma2[0] = 1.0
        for i in range(1, n):
            sigma2[i] = 0.1 + 0.3 * y[i-1]**2 + 0.5 * sigma2[i-1]
            y[i] = rng.randn() * np.sqrt(sigma2[i])

        ours = arch_test(y)
        theirs = het_arch(y, nlags=12)

        assert ours['pvalue'] < 0.05, f"Should detect ARCH: p={ours['pvalue']:.4f}"
        assert theirs[1] < 0.05, f"Statsmodels should agree: p={theirs[1]:.4f}"

    def test_white_noise_no_arch(self):
        """White noise: no ARCH effects."""
        from primitives.stat_tests.volatility import arch_test

        rng = np.random.RandomState(42)
        y = rng.randn(2000)
        result = arch_test(y)
        assert result['pvalue'] > 0.05, \
            f"White noise should have no ARCH: p={result['pvalue']:.4f}"


class TestArchAnalytical:

    def test_constant_no_arch(self):
        from primitives.stat_tests.volatility import arch_test
        result = arch_test(np.ones(500))
        assert result['pvalue'] > 0.05 or np.isnan(result['pvalue'])

    def test_short_signal(self):
        from primitives.stat_tests.volatility import arch_test
        result = arch_test(np.array([1.0, 2.0, 3.0]))
        assert np.isnan(result['statistic'])
