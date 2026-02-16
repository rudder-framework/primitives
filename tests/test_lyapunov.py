"""Tests for Lyapunov exponent estimation."""
import numpy as np
import pytest


def test_returns_tuple():
    """lyapunov_rosenstein returns (float, array, array)."""
    np.random.seed(42)
    signal = np.random.randn(200)

    from pmtvs.dynamics import lyapunov_rosenstein
    result = lyapunov_rosenstein(signal)
    assert len(result) == 3
    lam, div, iters = result
    assert isinstance(lam, float)


def test_short_signal_returns_nan():
    """Signal too short returns NaN."""
    from pmtvs.dynamics import lyapunov_rosenstein
    lam, _, _ = lyapunov_rosenstein(np.array([1.0, 2.0, 3.0]))
    assert np.isnan(lam)


def test_periodic_near_zero():
    """Periodic signal should have Lyapunov near zero or negative."""
    signal = np.sin(np.linspace(0, 40 * np.pi, 2000))

    from pmtvs.dynamics import lyapunov_rosenstein
    lam, _, _ = lyapunov_rosenstein(signal, dimension=3, delay=5)
    # Periodic signals have non-positive Lyapunov
    if np.isfinite(lam):
        assert lam < 0.5, f"Expected lam < 0.5 for periodic, got {lam}"


def test_kantz_returns_tuple():
    """lyapunov_kantz returns (float, array)."""
    np.random.seed(42)
    signal = np.random.randn(200)

    from pmtvs.dynamics import lyapunov_kantz
    result = lyapunov_kantz(signal)
    assert len(result) == 2
    lam, div = result
    assert isinstance(lam, float)
