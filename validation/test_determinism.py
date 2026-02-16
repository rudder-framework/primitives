"""
Determinism (from signal) validation.

Ground truth: analytical expectations.
- Deterministic signal (sine): high determinism
- White noise: low determinism
- Constant: degenerate (all points recur)
"""
import numpy as np
import pytest


class TestDeterminismAnalytical:

    def test_sine_high_determinism(self):
        """Periodic signal should have high determinism."""
        from pmtvs.dynamical.rqa import determinism_from_signal
        t = np.linspace(0, 4 * np.pi, 500)
        det = determinism_from_signal(np.sin(t), dimension=3, delay=10)
        assert det > 0.7, f"Sine determinism should be high, got {det:.4f}"

    def test_sine_higher_than_noise(self):
        """Deterministic signal should have higher determinism than noise."""
        from pmtvs.dynamical.rqa import determinism_from_signal
        rng = np.random.RandomState(42)
        t = np.linspace(0, 4 * np.pi, 500)
        det_sine = determinism_from_signal(np.sin(t), dimension=3, delay=10)
        det_noise = determinism_from_signal(rng.randn(500), dimension=3, delay=1)
        assert det_sine > det_noise, \
            f"Sine det ({det_sine:.4f}) should exceed noise det ({det_noise:.4f})"

    def test_matches_manual_pipeline(self):
        """determinism_from_signal should match manual recurrence_matrix + determinism."""
        from pmtvs.dynamical.rqa import (
            determinism_from_signal, recurrence_matrix, determinism
        )
        rng = np.random.RandomState(42)
        sig = rng.randn(200)
        # Manual pipeline
        R = recurrence_matrix(sig, dimension=3, delay=1)
        manual = determinism(R, min_line=2)
        # Wrapper
        auto = determinism_from_signal(sig, dimension=3, delay=1)
        assert abs(auto - manual) < 1e-12, \
            f"Wrapper and manual should match: {auto} vs {manual}"

    def test_short_signal(self):
        """Very short signal should return 0 or degenerate gracefully."""
        from pmtvs.dynamical.rqa import determinism_from_signal
        # With dimension=3, delay=1, need at least 5 points for a meaningful RM
        det = determinism_from_signal(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert 0.0 <= det <= 1.0
