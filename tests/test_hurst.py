"""Tests for Hurst exponent."""
import numpy as np
import pytest


def test_random_walk_persistent():
    """Cumulative sum of random should have H > 0.5."""
    np.random.seed(42)
    signal = np.cumsum(np.random.randn(1000))

    from prmtvs.complexity import hurst_exponent
    h = hurst_exponent(signal)
    assert 0.5 < h <= 1.0, f"Expected H > 0.5 for random walk, got {h}"


def test_white_noise_around_half():
    """White noise should have H ~ 0.5."""
    np.random.seed(42)
    signal = np.random.randn(2000)

    from prmtvs.complexity import hurst_exponent
    h = hurst_exponent(signal)
    assert 0.3 < h < 0.7, f"Expected H ~ 0.5 for white noise, got {h}"


def test_short_signal_returns_nan():
    """Signal shorter than rs_min_k returns NaN."""
    from prmtvs.complexity import hurst_exponent
    h = hurst_exponent(np.array([1.0, 2.0, 3.0]))
    assert np.isnan(h)


def test_constant_signal_returns_nan():
    """Constant signal (zero std) returns NaN."""
    from prmtvs.complexity import hurst_exponent
    h = hurst_exponent(np.ones(100))
    assert np.isnan(h)


def test_nan_values_stripped():
    """NaN values in signal are ignored."""
    np.random.seed(42)
    signal = np.cumsum(np.random.randn(500))
    signal[10] = np.nan
    signal[200] = np.nan

    from prmtvs.complexity import hurst_exponent
    h = hurst_exponent(signal)
    assert 0.0 <= h <= 1.0


def test_result_clipped():
    """Result is always in [0, 1]."""
    np.random.seed(42)
    from prmtvs.complexity import hurst_exponent
    for _ in range(10):
        signal = np.cumsum(np.random.randn(200))
        h = hurst_exponent(signal)
        if not np.isnan(h):
            assert 0.0 <= h <= 1.0
