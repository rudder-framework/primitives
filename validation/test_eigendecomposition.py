"""
Eigendecomposition validation against numpy.linalg.

Ground truth: numpy.linalg.eigh (LAPACK, gold standard)

Known values:
- Identity matrix: all eigenvalues = 1
- Diagonal matrix: eigenvalues = diagonal entries
- Rank-1 matrix:   one nonzero eigenvalue
"""
import numpy as np
import pytest

from prmtvs.matrix.decomposition import eigendecomposition


class TestEigenVsNumpy:
    """Compare against numpy.linalg.eigh."""

    def test_identity(self):
        eigenvalues, eigenvectors = eigendecomposition(np.eye(5))
        expected = np.ones(5)
        np.testing.assert_allclose(sorted(eigenvalues), sorted(expected), atol=1e-10)

    def test_diagonal(self):
        diag_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        m = np.diag(diag_vals)
        eigenvalues, eigenvectors = eigendecomposition(m)
        np.testing.assert_allclose(sorted(eigenvalues), sorted(diag_vals), atol=1e-10)

    def test_random_symmetric_vs_numpy(self):
        rng = np.random.RandomState(42)
        a = rng.randn(10, 10)
        m = a @ a.T  # guaranteed symmetric positive semi-definite
        ours = sorted(eigendecomposition(m)[0])
        theirs = sorted(np.linalg.eigh(m)[0])
        np.testing.assert_allclose(ours, theirs, atol=1e-8)

    def test_rank_one(self):
        v = np.array([1.0, 2.0, 3.0])
        m = np.outer(v, v)  # rank 1
        eigs = sorted(eigendecomposition(m)[0])
        # Should have one large eigenvalue and rest ~ 0
        assert eigs[-1] > 10  # sum of squares = 14
        assert all(abs(e) < 1e-10 for e in eigs[:-1])
