"""Tests for optimal delay estimation."""
import numpy as np
import pytest


def test_autocorr_method():
    """Autocorrelation method returns reasonable delay."""
    np.random.seed(42)
    signal = np.sin(np.linspace(0, 20 * np.pi, 1000))

    from primitives.embedding.delay import optimal_delay
    tau = optimal_delay(signal, method='autocorr')
    assert isinstance(tau, int)
    assert tau >= 1


def test_mutual_info_method():
    """Mutual information method returns reasonable delay."""
    np.random.seed(42)
    signal = np.sin(np.linspace(0, 20 * np.pi, 1000))

    from primitives.embedding.delay import optimal_delay
    tau = optimal_delay(signal, method='mutual_info')
    assert isinstance(tau, int)
    assert tau >= 1


def test_autocorr_e_method():
    """1/e decay method returns reasonable delay."""
    np.random.seed(42)
    signal = np.sin(np.linspace(0, 20 * np.pi, 1000))

    from primitives.embedding.delay import optimal_delay
    tau = optimal_delay(signal, method='autocorr_e')
    assert isinstance(tau, int)
    assert tau >= 1


def test_constant_signal():
    """Constant signal returns delay=1."""
    from primitives.embedding.delay import optimal_delay
    tau = optimal_delay(np.ones(100))
    assert tau == 1


def test_short_signal():
    """Very short signal returns delay=1."""
    from primitives.embedding.delay import optimal_delay
    tau = optimal_delay(np.array([1.0, 2.0, 3.0]))
    assert tau == 1


def test_time_delay_embedding():
    """Time delay embedding produces correct shape."""
    signal = np.arange(100, dtype=np.float64)

    from primitives.embedding.delay import time_delay_embedding
    emb = time_delay_embedding(signal, dimension=3, delay=2)
    assert emb.shape == (96, 3)
    # First point: [0, 2, 4]
    np.testing.assert_array_equal(emb[0], [0, 2, 4])


def test_optimal_dimension():
    """Optimal dimension returns integer >= 2."""
    np.random.seed(42)
    signal = np.random.randn(500)

    from primitives.embedding.delay import optimal_dimension
    dim = optimal_dimension(signal, delay=1, max_dim=5)
    assert isinstance(dim, int)
    assert dim >= 1
