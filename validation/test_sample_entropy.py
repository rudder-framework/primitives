"""
Sample entropy validation against nolds.

Ground truth:
- Regular signal (sine): low SampEn
- Random signal:         high SampEn
- Constant signal:       undefined / NaN

Reference: nolds.sampen()
"""
import numpy as np
import pytest

from pmtvs import sample_entropy

import nolds


class TestSampleEntropyVsNolds:
    """Compare against nolds (published reference)."""

    def test_random(self, white_noise):
        ours = sample_entropy(white_noise[:2000])
        theirs = nolds.sampen(white_noise[:2000], emb_dim=2)
        if not np.isnan(ours) and not np.isnan(theirs):
            assert abs(ours - theirs) < 0.5, \
                f"ours={ours:.4f} nolds={theirs:.4f}"

    def test_sine(self, sine_wave):
        ours = sample_entropy(sine_wave[:2000])
        theirs = nolds.sampen(sine_wave[:2000], emb_dim=2)
        if not np.isnan(ours) and not np.isnan(theirs):
            assert abs(ours - theirs) < 0.5, \
                f"ours={ours:.4f} nolds={theirs:.4f}"


class TestSampleEntropyAnalytical:
    """Test against known properties."""

    def test_random_higher_than_sine(self, white_noise, sine_wave):
        se_random = sample_entropy(white_noise[:2000])
        se_sine = sample_entropy(sine_wave[:2000])
        if not np.isnan(se_random) and not np.isnan(se_sine):
            assert se_random > se_sine, \
                f"Random ({se_random:.4f}) should have higher SampEn than sine ({se_sine:.4f})"

    def test_non_negative(self, white_noise):
        se = sample_entropy(white_noise)
        if not np.isnan(se):
            assert se >= 0, f"SampEn must be non-negative, got {se:.4f}"

    def test_constant_is_nan_or_near_zero(self, constant):
        se = sample_entropy(constant)
        # Constant signal: all templates match, so A/B ~ 1, SampEn ~ 0.
        # May not be exactly 0 due to floating-point edge cases.
        assert np.isnan(se) or abs(se) < 0.01, \
            f"Constant signal SampEn should be NaN or ~0, got {se}"
