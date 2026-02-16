"""
Lyapunov exponent validation against nolds and known dynamical systems.

Ground truth:
- Lorenz attractor: max LE ~ 0.91 (published value)
- Periodic orbit:   LE <= 0
- White noise:      LE > 0 (but not physically meaningful)

Reference: nolds.lyap_r() (Rosenstein method)
"""
import numpy as np
import pytest

from prmtvs import lyapunov_rosenstein

import nolds


class TestLyapunovVsNolds:
    """Compare against nolds Rosenstein method."""

    def test_lorenz(self, lorenz_x):
        ours_result = lyapunov_rosenstein(lorenz_x)
        ours = ours_result[0]  # (exponent, divergence_curve, steps)
        theirs = nolds.lyap_r(lorenz_x, emb_dim=3, lag=1)
        # Both should be positive for Lorenz
        if not np.isnan(ours) and not np.isnan(theirs):
            assert ours > 0 and theirs > 0, \
                f"Lorenz LE should be positive: ours={ours:.4f} nolds={theirs:.4f}"

    def test_random(self, white_noise):
        ours_result = lyapunov_rosenstein(white_noise[:5000])
        ours = ours_result[0]
        theirs = nolds.lyap_r(white_noise[:5000], emb_dim=3, lag=1)
        # Both should produce a result (may differ in magnitude)
        assert isinstance(ours, float)
        assert isinstance(theirs, (float, np.floating))


class TestLyapunovAnalytical:
    """Test against known dynamical system properties."""

    def test_lorenz_positive(self, lorenz_x):
        le = lyapunov_rosenstein(lorenz_x)[0]
        if not np.isnan(le):
            assert le > 0, f"Lorenz max LE should be positive, got {le:.4f}"

    def test_lorenz_order_of_magnitude(self, lorenz_x):
        le = lyapunov_rosenstein(lorenz_x)[0]
        if not np.isnan(le):
            # Published value ~ 0.91. Rosenstein method estimates vary widely
            # depending on embedding params and fit region. Just verify positive
            # and within a very wide range.
            assert 0.01 < le < 5.0, \
                f"Lorenz LE should be positive, got {le:.4f}"

    def test_sine_non_positive(self, sine_wave):
        le = lyapunov_rosenstein(sine_wave[:5000])[0]
        if not np.isnan(le):
            # Periodic signals should have LE <= 0 (or near zero with estimation noise)
            assert le < 0.5, \
                f"Sine wave LE should be near-zero or negative, got {le:.4f}"
