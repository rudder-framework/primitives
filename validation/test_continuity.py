"""
Continuity features validation.

Ground truth: analytical expectations.
"""
import numpy as np
import pytest


class TestContinuityAnalytical:

    def test_all_unique(self):
        from prmtvs.individual.continuity import continuity_features
        rng = np.random.RandomState(42)
        feat = continuity_features(rng.randn(1000))
        assert feat['unique_ratio'] == 1.0, \
            f"Random floats should all be unique, got {feat['unique_ratio']}"
        assert feat['is_integer'] is False
        assert feat['sparsity'] == 0.0

    def test_constant(self):
        from prmtvs.individual.continuity import continuity_features
        feat = continuity_features(np.ones(100))
        assert feat['unique_ratio'] == 1 / 100
        assert feat['is_integer'] is True
        assert feat['sparsity'] == 0.0

    def test_all_zeros(self):
        from prmtvs.individual.continuity import continuity_features
        feat = continuity_features(np.zeros(50))
        assert feat['unique_ratio'] == 1 / 50
        assert feat['is_integer'] is True
        assert feat['sparsity'] == 1.0

    def test_integer_signal(self):
        from prmtvs.individual.continuity import continuity_features
        feat = continuity_features(np.array([1, 2, 3, 4, 5], dtype=float))
        assert feat['is_integer'] is True
        assert feat['unique_ratio'] == 1.0

    def test_mixed_sparsity(self):
        from prmtvs.individual.continuity import continuity_features
        data = np.array([0, 1, 0, 2, 0, 3, 0, 4], dtype=float)
        feat = continuity_features(data)
        assert feat['sparsity'] == 0.5, f"Half zeros, got {feat['sparsity']}"

    def test_empty(self):
        from prmtvs.individual.continuity import continuity_features
        feat = continuity_features(np.array([]))
        assert np.isnan(feat['unique_ratio'])
        assert np.isnan(feat['sparsity'])
