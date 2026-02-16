"""
Turning point ratio validation.

Ground truth: Mathematical expectation.
- White noise (iid): E[TPR] = 2/3
- Monotonic ramp: TPR = 0
- Alternating: TPR = 1
"""
import numpy as np
import pytest


class TestTPRAnalytical:

    def test_white_noise_near_two_thirds(self):
        from prmtvs.individual.temporal import turning_point_ratio
        rng = np.random.RandomState(42)
        tpr = turning_point_ratio(rng.randn(10000))
        assert abs(tpr - 2/3) < 0.02, \
            f"White noise TPR should be ~0.667, got {tpr:.4f}"

    def test_monotonic_zero(self):
        from prmtvs.individual.temporal import turning_point_ratio
        tpr = turning_point_ratio(np.arange(100, dtype=float))
        assert tpr == 0.0, f"Monotonic ramp TPR should be 0, got {tpr}"

    def test_alternating_one(self):
        from prmtvs.individual.temporal import turning_point_ratio
        alt = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
        tpr = turning_point_ratio(alt)
        assert tpr == 1.0, f"Alternating TPR should be 1.0, got {tpr}"

    def test_bounded_zero_one(self):
        from prmtvs.individual.temporal import turning_point_ratio
        rng = np.random.RandomState(42)
        tpr = turning_point_ratio(rng.randn(1000))
        assert 0.0 <= tpr <= 1.0

    def test_short_signal_nan(self):
        from prmtvs.individual.temporal import turning_point_ratio
        assert np.isnan(turning_point_ratio(np.array([1.0, 2.0])))

    def test_sine_moderate(self):
        """Sine wave: TPR depends on sampling density."""
        from prmtvs.individual.temporal import turning_point_ratio
        t = np.linspace(0, 10, 1000)
        tpr = turning_point_ratio(np.sin(2 * np.pi * 5 * t))
        # Sine sampled well above Nyquist -> many smooth sections, low TPR
        assert tpr < 0.5, f"Well-sampled sine TPR should be < 0.5, got {tpr:.4f}"
