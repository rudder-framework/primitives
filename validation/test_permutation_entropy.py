"""
Permutation entropy validation against ordpy and known bounds.

Ground truth:
- Ordered signal (sine, ramp): PE ~ 0 (low complexity)
- Random signal:               PE ~ 1 (max complexity, when normalized)

Reference: ordpy.permutation_entropy()
"""
import numpy as np
import pytest

from pmtvs import permutation_entropy

import ordpy


class TestPermEntropyVsOrdpy:
    """Compare against ordpy (published reference)."""

    def test_random(self, white_noise):
        ours = permutation_entropy(white_noise)
        theirs = ordpy.permutation_entropy(white_noise, dx=3, taux=1)
        assert abs(ours - theirs) < 0.05, \
            f"ours={ours:.4f} ordpy={theirs:.4f}"

    def test_sine(self, sine_wave):
        ours = permutation_entropy(sine_wave)
        theirs = ordpy.permutation_entropy(sine_wave, dx=3, taux=1)
        assert abs(ours - theirs) < 0.1, \
            f"ours={ours:.4f} ordpy={theirs:.4f}"


class TestPermEntropyAnalytical:
    """Test against known mathematical properties."""

    def test_random_is_high(self, white_noise):
        pe = permutation_entropy(white_noise)
        assert pe > 0.8, f"Random signal PE should be high, got {pe:.4f}"

    def test_sine_is_low(self, sine_wave):
        pe = permutation_entropy(sine_wave)
        assert pe < 0.5, f"Sine wave PE should be low, got {pe:.4f}"

    def test_monotonic_is_zero(self):
        ramp = np.arange(1000, dtype=float)
        pe = permutation_entropy(ramp)
        assert pe < 0.01, f"Monotonic ramp PE should be ~0, got {pe:.4f}"

    def test_bounded(self, white_noise):
        pe = permutation_entropy(white_noise)
        assert 0.0 <= pe <= 1.0, f"Normalized PE out of [0, 1]: {pe:.4f}"
