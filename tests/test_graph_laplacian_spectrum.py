"""
Unit tests for graph_laplacian_spectrum primitive.
"""
import numpy as np
import pytest

from pmtvs.matrix.graph import graph_laplacian_spectrum


class TestGraphLaplacianSpectrumBasic:
    """Basic contract tests."""

    def test_returns_dict(self):
        A = np.array([[0, 1], [1, 0]], dtype=float)
        result = graph_laplacian_spectrum(A)
        assert isinstance(result, dict)

    def test_all_keys_present(self):
        A = np.array([[0, 1], [1, 0]], dtype=float)
        result = graph_laplacian_spectrum(A)
        expected_keys = {
            'eigenvalues', 'algebraic_connectivity', 'spectral_gap',
            'n_components', 'effective_connectivity', 'max_eigenvalue',
        }
        assert set(result.keys()) == expected_keys

    def test_eigenvalues_is_array(self):
        A = np.array([[0, 1], [1, 0]], dtype=float)
        result = graph_laplacian_spectrum(A)
        assert isinstance(result['eigenvalues'], np.ndarray)

    def test_n_components_is_int(self):
        A = np.array([[0, 1], [1, 0]], dtype=float)
        result = graph_laplacian_spectrum(A)
        assert isinstance(result['n_components'], int)


class TestGraphLaplacianSpectrumEdgeCases:
    """Edge case handling."""

    def test_single_node(self):
        result = graph_laplacian_spectrum(np.array([[0.0]]))
        assert result['n_components'] == 1
        assert result['algebraic_connectivity'] == 0.0
        assert result['max_eigenvalue'] == 0.0

    def test_identity_matrix_all_isolated(self):
        """Identity matrix → diagonal zeroed → all isolated nodes."""
        result = graph_laplacian_spectrum(np.eye(5))
        assert result['n_components'] == 5
        assert result['algebraic_connectivity'] == 0.0

    def test_zero_matrix_disconnected(self):
        """All-zero adjacency → disconnected graph."""
        result = graph_laplacian_spectrum(np.zeros((4, 4)))
        assert result['n_components'] == 4
        assert result['algebraic_connectivity'] == 0.0

    def test_nan_in_adjacency(self):
        """NaN values treated as no coupling (0)."""
        A = np.array([[0, np.nan, 1], [np.nan, 0, 1], [1, 1, 0]], dtype=float)
        result = graph_laplacian_spectrum(A)
        assert isinstance(result['n_components'], int)
        assert not np.isnan(result['algebraic_connectivity'])


class TestGraphLaplacianSpectrumValues:
    """Known-value tests."""

    def test_two_coupled_nodes_normalized(self):
        """K_2 normalized: eigenvalues should be [0, 2]."""
        A = np.array([[0, 1], [1, 0]], dtype=float)
        result = graph_laplacian_spectrum(A, normalized=True)
        eigs = result['eigenvalues']
        assert abs(eigs[0]) < 1e-10, "Smallest eigenvalue should be 0"
        assert abs(eigs[1] - 2.0) < 1e-10, "Largest eigenvalue of K_2 normalized should be 2"
        assert result['n_components'] == 1

    def test_two_coupled_nodes_unnormalized(self):
        """K_2 unnormalized: eigenvalues should be [0, 2]."""
        A = np.array([[0, 1], [1, 0]], dtype=float)
        result = graph_laplacian_spectrum(A, normalized=False)
        eigs = result['eigenvalues']
        assert abs(eigs[0]) < 1e-10
        assert abs(eigs[1] - 2.0) < 1e-10

    def test_complete_graph_k4_normalized(self):
        """K_4 normalized Laplacian: eigenvalues [0, 4/3, 4/3, 4/3]."""
        A = np.ones((4, 4)) - np.eye(4)
        result = graph_laplacian_spectrum(A, normalized=True)
        eigs = result['eigenvalues']
        assert abs(eigs[0]) < 1e-10
        # For K_n normalized, non-zero eigenvalues = n/(n-1)
        for i in range(1, 4):
            assert abs(eigs[i] - 4.0/3.0) < 1e-10, \
                f"K_4 normalized eigenvalue {i} should be 4/3, got {eigs[i]}"

    def test_complete_graph_k4_unnormalized(self):
        """K_4 unnormalized Laplacian: eigenvalues [0, 4, 4, 4]."""
        A = np.ones((4, 4)) - np.eye(4)
        result = graph_laplacian_spectrum(A, normalized=False)
        eigs = result['eigenvalues']
        assert abs(eigs[0]) < 1e-10
        for i in range(1, 4):
            assert abs(eigs[i] - 4.0) < 1e-10

    def test_disconnected_two_components(self):
        """Block diagonal adjacency → 2 connected components."""
        A = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ], dtype=float)
        result = graph_laplacian_spectrum(A)
        assert result['n_components'] == 2
        assert result['algebraic_connectivity'] == 0.0

    def test_path_graph_p3(self):
        """Path graph P_3 unnormalized: eigenvalues [0, 1, 3]."""
        A = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ], dtype=float)
        result = graph_laplacian_spectrum(A, normalized=False)
        eigs = result['eigenvalues']
        assert abs(eigs[0]) < 1e-10
        assert abs(eigs[1] - 1.0) < 1e-10, f"Fiedler value of P_3 should be 1, got {eigs[1]}"
        assert abs(eigs[2] - 3.0) < 1e-10, f"Max eigenvalue of P_3 should be 3, got {eigs[2]}"

    def test_eigenvalues_sorted_ascending(self):
        rng = np.random.RandomState(42)
        A = np.abs(rng.randn(5, 5))
        A = (A + A.T) / 2
        result = graph_laplacian_spectrum(A)
        eigs = result['eigenvalues']
        assert np.all(np.diff(eigs) >= -1e-12), "Eigenvalues should be sorted ascending"

    def test_eigenvalues_non_negative(self):
        """Laplacian eigenvalues are always >= 0."""
        rng = np.random.RandomState(42)
        A = np.abs(rng.randn(6, 6))
        A = (A + A.T) / 2
        result = graph_laplacian_spectrum(A)
        assert np.all(result['eigenvalues'] >= 0)

    def test_spectral_gap_connected(self):
        """Connected graph should have spectral_gap > 0."""
        A = np.ones((4, 4)) - np.eye(4)
        result = graph_laplacian_spectrum(A)
        assert result['spectral_gap'] > 0

    def test_spectral_gap_disconnected(self):
        """Disconnected graph should have spectral_gap = 0 or NaN."""
        result = graph_laplacian_spectrum(np.zeros((4, 4)))
        assert result['spectral_gap'] != result['spectral_gap'] or result['spectral_gap'] == 0.0

    def test_asymmetric_input_symmetrized(self):
        """Asymmetric input should be symmetrized."""
        A = np.array([[0, 2, 0], [0, 0, 3], [0, 0, 0]], dtype=float)
        result = graph_laplacian_spectrum(A)
        # Should not crash; the function symmetrizes internally
        assert result['n_components'] >= 1
