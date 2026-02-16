"""Tests for sample entropy."""
import numpy as np
import pytest


def test_periodic_low_entropy():
    """Periodic signal should have low sample entropy."""
    signal = np.sin(np.linspace(0, 20 * np.pi, 500))

    from pmtvs.complexity import sample_entropy
    se = sample_entropy(signal, m=2)
    assert se < 1.0, f"Expected low SE for periodic, got {se}"


def test_random_higher_entropy():
    """Random signal should have higher sample entropy than periodic."""
    np.random.seed(42)
    periodic = np.sin(np.linspace(0, 20 * np.pi, 500))
    random_sig = np.random.randn(500)

    from pmtvs.complexity import sample_entropy
    se_periodic = sample_entropy(periodic, m=2)
    se_random = sample_entropy(random_sig, m=2)

    # Both should be finite
    assert np.isfinite(se_periodic)
    assert np.isfinite(se_random)
    assert se_random > se_periodic


def test_constant_returns_nan():
    """Constant signal (zero std, r=0) returns NaN."""
    from pmtvs.complexity import sample_entropy
    se = sample_entropy(np.ones(100), m=2)
    assert np.isnan(se)


def test_short_signal_returns_nan():
    """Signal shorter than m+2 returns NaN."""
    from pmtvs.complexity import sample_entropy
    se = sample_entropy(np.array([1.0, 2.0]), m=2)
    assert np.isnan(se)
