"""
Graph Laplacian spectrum validation against numpy.

Ground truth:
- Complete graph K_n: known eigenvalues
- Path graph P_n: known Fiedler value
- Disconnected graph: n_components matches block structure
- Star graph: known spectrum
"""
import numpy as np
import pytest

from pmtvs.matrix.graph import graph_laplacian_spectrum


class TestGraphLaplacianVsNumpy:
    """Compare against direct numpy eigenvalue computation."""

    def test_eigenvalues_match_numpy(self):
        """Our eigenvalues should match np.linalg.eigh on the Laplacian."""
        rng = np.random.RandomState(42)
        A = np.abs(rng.randn(6, 6))
        A = (A + A.T) / 2
        np.fill_diagonal(A, 0)

        # Manual Laplacian (normalized)
        degrees = A.sum(axis=1)
        D = np.diag(degrees)
        L = D - A
        d_inv_sqrt = np.where(degrees > 1e-12, 1.0 / np.sqrt(degrees), 0.0)
        D_inv_sqrt = np.diag(d_inv_sqrt)
        L_norm = D_inv_sqrt @ L @ D_inv_sqrt

        expected_eigs = np.sort(np.linalg.eigh(L_norm)[0])
        expected_eigs = np.maximum(expected_eigs, 0.0)

        result = graph_laplacian_spectrum(A, normalized=True)
        np.testing.assert_allclose(result['eigenvalues'], expected_eigs, atol=1e-10)

    def test_unnormalized_eigenvalues_match_numpy(self):
        """Unnormalized Laplacian eigenvalues should match numpy."""
        rng = np.random.RandomState(42)
        A = np.abs(rng.randn(5, 5))
        A = (A + A.T) / 2
        np.fill_diagonal(A, 0)

        degrees = A.sum(axis=1)
        L = np.diag(degrees) - A
        expected_eigs = np.sort(np.linalg.eigh(L)[0])
        expected_eigs = np.maximum(expected_eigs, 0.0)

        result = graph_laplacian_spectrum(A, normalized=False)
        np.testing.assert_allclose(result['eigenvalues'], expected_eigs, atol=1e-10)


class TestGraphLaplacianAnalytical:
    """Test against mathematically known values."""

    def test_complete_graph_k5_normalized(self):
        """K_5 normalized: eigenvalues [0, 5/4, 5/4, 5/4, 5/4]."""
        n = 5
        A = np.ones((n, n)) - np.eye(n)
        result = graph_laplacian_spectrum(A, normalized=True)
        eigs = result['eigenvalues']
        assert abs(eigs[0]) < 1e-10
        for i in range(1, n):
            assert abs(eigs[i] - n / (n - 1)) < 1e-10, \
                f"K_{n} normalized eigenvalue should be {n/(n-1):.4f}, got {eigs[i]:.4f}"
        assert result['n_components'] == 1
        assert abs(result['algebraic_connectivity'] - n / (n - 1)) < 1e-10

    def test_complete_graph_k5_unnormalized(self):
        """K_5 unnormalized: eigenvalues [0, 5, 5, 5, 5]."""
        n = 5
        A = np.ones((n, n)) - np.eye(n)
        result = graph_laplacian_spectrum(A, normalized=False)
        eigs = result['eigenvalues']
        assert abs(eigs[0]) < 1e-10
        for i in range(1, n):
            assert abs(eigs[i] - n) < 1e-10

    def test_star_graph_unnormalized(self):
        """Star graph S_4 (hub + 3 leaves) unnormalized: eigenvalues [0, 1, 1, 4]."""
        # Star with hub=0 and leaves=1,2,3
        A = np.array([
            [0, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ], dtype=float)
        result = graph_laplacian_spectrum(A, normalized=False)
        eigs = result['eigenvalues']
        assert abs(eigs[0]) < 1e-10
        assert abs(eigs[1] - 1.0) < 1e-10, f"S_4 eigenvalue 1 should be 1, got {eigs[1]}"
        assert abs(eigs[2] - 1.0) < 1e-10, f"S_4 eigenvalue 2 should be 1, got {eigs[2]}"
        assert abs(eigs[3] - 4.0) < 1e-10, f"S_4 eigenvalue 3 should be 4, got {eigs[3]}"

    def test_cycle_graph_c4_unnormalized(self):
        """Cycle C_4 unnormalized: eigenvalues [0, 2, 2, 4]."""
        A = np.array([
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ], dtype=float)
        result = graph_laplacian_spectrum(A, normalized=False)
        eigs = result['eigenvalues']
        assert abs(eigs[0]) < 1e-10
        assert abs(eigs[1] - 2.0) < 1e-10
        assert abs(eigs[2] - 2.0) < 1e-10
        assert abs(eigs[3] - 4.0) < 1e-10

    def test_disconnected_three_components(self):
        """Three isolated components → n_components = 3."""
        A = np.zeros((6, 6))
        # Component 1: nodes 0,1
        A[0, 1] = A[1, 0] = 1.0
        # Component 2: nodes 2,3
        A[2, 3] = A[3, 2] = 1.0
        # Component 3: nodes 4,5
        A[4, 5] = A[5, 4] = 1.0
        result = graph_laplacian_spectrum(A)
        assert result['n_components'] == 3
        assert result['algebraic_connectivity'] == 0.0

    def test_path_graph_p4_fiedler(self):
        """Path P_4 unnormalized: Fiedler value = 2 - sqrt(2) ≈ 0.5858."""
        A = np.array([
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
        ], dtype=float)
        result = graph_laplacian_spectrum(A, normalized=False)
        expected_fiedler = 2 - np.sqrt(2)
        assert abs(result['algebraic_connectivity'] - expected_fiedler) < 1e-10, \
            f"P_4 Fiedler should be {expected_fiedler:.4f}, got {result['algebraic_connectivity']:.4f}"

    def test_weighted_complete_graph(self):
        """Weighted K_3 with equal weights → scaled version of unweighted."""
        w = 2.5
        A = w * (np.ones((3, 3)) - np.eye(3))
        result = graph_laplacian_spectrum(A, normalized=False)
        eigs = result['eigenvalues']
        # Unnormalized K_3 with weight w: eigenvalues [0, 3w, 3w]
        assert abs(eigs[0]) < 1e-10
        assert abs(eigs[1] - 3 * w) < 1e-10
        assert abs(eigs[2] - 3 * w) < 1e-10

    def test_correlation_matrix_as_adjacency(self):
        """Practical: correlation matrix → spectrum reveals structure."""
        rng = np.random.RandomState(42)
        # Two correlated groups
        group1 = rng.randn(3, 100)
        group2 = rng.randn(3, 100)
        signals = np.vstack([group1, group2])
        corr = np.abs(np.corrcoef(signals))

        result = graph_laplacian_spectrum(corr)
        assert result['n_components'] >= 1
        assert result['algebraic_connectivity'] >= 0
        assert len(result['eigenvalues']) == 6
