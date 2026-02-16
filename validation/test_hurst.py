"""
Hurst exponent validation against nolds and known analytical values.

Ground truth:
- White noise:  H ~ 0.5 (no memory)
- Random walk:  H ~ 1.0 (perfect memory)
- Antipersistent: H < 0.5 (mean-reverting)

Reference: nolds.hurst_rs()
"""
import numpy as np
import pytest

from pmtvs import hurst_exponent

import nolds


class TestHurstVsNolds:
    """Compare our Hurst against nolds (published reference)."""

    def test_white_noise(self, white_noise):
        ours = hurst_exponent(white_noise)
        theirs = nolds.hurst_rs(white_noise)
        # Different R/S implementations use different subseries ranges,
        # so allow slightly wider tolerance for white noise
        assert abs(ours - theirs) < 0.1, f"ours={ours:.4f} nolds={theirs:.4f}"

    def test_random_walk(self, random_walk):
        ours = hurst_exponent(random_walk)
        theirs = nolds.hurst_rs(random_walk)
        assert abs(ours - theirs) < 0.05, f"ours={ours:.4f} nolds={theirs:.4f}"

    def test_sine(self, sine_wave):
        ours = hurst_exponent(sine_wave)
        theirs = nolds.hurst_rs(sine_wave)
        assert abs(ours - theirs) < 0.1, f"ours={ours:.4f} nolds={theirs:.4f}"


class TestHurstAnalytical:
    """Test against mathematically known values."""

    def test_white_noise_near_half(self, white_noise):
        h = hurst_exponent(white_noise)
        assert 0.35 < h < 0.65, f"White noise Hurst should be ~0.5, got {h:.4f}"

    def test_random_walk_near_one(self, random_walk):
        h = hurst_exponent(random_walk)
        assert h > 0.85, f"Random walk Hurst should be ~1.0, got {h:.4f}"

    def test_bounded_zero_to_one(self, white_noise):
        h = hurst_exponent(white_noise)
        assert 0.0 <= h <= 1.0, f"Hurst must be in [0, 1], got {h:.4f}"

    def test_short_signal_doesnt_crash(self):
        short = np.random.randn(20)
        h = hurst_exponent(short)
        assert isinstance(h, float)  # may be NaN, but must not crash

    def test_constant_signal(self, constant):
        h = hurst_exponent(constant)
        assert np.isnan(h) or isinstance(h, float)  # degenerate case
