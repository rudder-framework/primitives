"""Tests for permutation entropy."""
import numpy as np
import pytest


def test_periodic_low_entropy():
    """Periodic signal should have low permutation entropy."""
    signal = np.sin(np.linspace(0, 20 * np.pi, 1000))

    from primitives.complexity import permutation_entropy
    pe = permutation_entropy(signal, order=3, delay=1)
    assert 0.0 <= pe < 0.7, f"Expected low PE for periodic, got {pe}"


def test_random_high_entropy():
    """Random signal should have high permutation entropy."""
    np.random.seed(42)
    signal = np.random.randn(1000)

    from primitives.complexity import permutation_entropy
    pe = permutation_entropy(signal, order=3, delay=1)
    assert pe > 0.9, f"Expected high PE for random, got {pe}"


def test_normalized_range():
    """Normalized PE should be in [0, 1]."""
    np.random.seed(42)
    signal = np.random.randn(500)

    from primitives.complexity import permutation_entropy
    pe = permutation_entropy(signal, order=3, delay=1, normalize=True)
    assert 0.0 <= pe <= 1.0


def test_short_signal_returns_nan():
    """Signal too short for given order returns NaN."""
    from primitives.complexity import permutation_entropy
    pe = permutation_entropy(np.array([1.0, 2.0]), order=3, delay=1)
    assert np.isnan(pe)


def test_constant_signal():
    """Constant signal — all patterns identical, entropy = 0."""
    from primitives.complexity import permutation_entropy
    pe = permutation_entropy(np.ones(100), order=3, delay=1)
    assert pe == 0.0 or np.isclose(pe, 0.0)
