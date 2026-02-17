"""
Hilbert envelope validation against scipy.signal.hilbert.

Ground truth:
- scipy.signal.hilbert for analytic signal
- Pure sine: envelope = constant amplitude
- AM signal: envelope tracks modulation function
- Constant signal (centered): envelope ≈ 0
"""
import numpy as np
import pytest
from scipy.signal import hilbert as scipy_hilbert

from pmtvs.individual.hilbert import hilbert_envelope


class TestHilbertEnvelopeVsScipy:
    """Compare envelope statistics against direct scipy computation."""

    def test_mean_matches_scipy(self, white_noise):
        """Our envelope_mean should match np.mean(np.abs(scipy.hilbert(centered)))."""
        centered = white_noise - np.mean(white_noise)
        analytic = scipy_hilbert(centered)
        scipy_env = np.abs(analytic)

        result = hilbert_envelope(white_noise)
        assert abs(result['envelope_mean'] - np.mean(scipy_env)) < 1e-10

    def test_std_matches_scipy(self, white_noise):
        centered = white_noise - np.mean(white_noise)
        analytic = scipy_hilbert(centered)
        scipy_env = np.abs(analytic)

        result = hilbert_envelope(white_noise)
        assert abs(result['envelope_std'] - np.std(scipy_env)) < 1e-10

    def test_max_matches_scipy(self, white_noise):
        centered = white_noise - np.mean(white_noise)
        analytic = scipy_hilbert(centered)
        scipy_env = np.abs(analytic)

        result = hilbert_envelope(white_noise)
        assert abs(result['envelope_max'] - np.max(scipy_env)) < 1e-10


class TestHilbertEnvelopeAnalytical:
    """Test against mathematically known values."""

    def test_pure_sine_constant_envelope(self, sine_wave):
        """Pure sine → envelope ≈ amplitude (1.0), std ≈ 0."""
        result = hilbert_envelope(sine_wave)
        assert abs(result['envelope_mean'] - 1.0) < 0.02, \
            f"Sine envelope mean should be ~1.0, got {result['envelope_mean']:.4f}"
        assert result['envelope_std'] < 0.05, \
            f"Sine envelope std should be ~0, got {result['envelope_std']:.4f}"

    def test_constant_signal_zero_envelope(self, constant):
        """Constant signal → centered = 0 → envelope ≈ 0."""
        result = hilbert_envelope(constant)
        assert result['envelope_mean'] < 1e-10, \
            f"Constant signal envelope should be ~0, got {result['envelope_mean']}"

    def test_white_noise_positive_envelope(self, white_noise):
        """White noise → positive envelope with variability."""
        result = hilbert_envelope(white_noise)
        assert result['envelope_mean'] > 0, "Envelope mean should be positive"
        assert result['envelope_std'] > 0, "Envelope std should be positive"
        assert result['envelope_cv'] > 0, "CV should be positive for noise"

    def test_sine_near_zero_trend(self, sine_wave):
        """Pure sine → no amplitude trend."""
        result = hilbert_envelope(sine_wave)
        assert abs(result['envelope_trend']) < 0.001, \
            f"Sine envelope trend should be ~0, got {result['envelope_trend']:.6f}"

    def test_am_signal_tracks_modulation(self):
        """AM signal: envelope should track the modulation function."""
        t = np.linspace(0, 10 * np.pi, 5000)
        modulation = np.linspace(1.0, 3.0, 5000)
        signal = np.sin(t) * modulation

        result = hilbert_envelope(signal)
        # Mean envelope should approximate mean of modulation
        assert abs(result['envelope_mean'] - np.mean(modulation)) < 0.2, \
            f"AM envelope mean should be ~{np.mean(modulation):.2f}, got {result['envelope_mean']:.2f}"
        # Positive trend (amplitude growing)
        assert result['envelope_trend'] > 0, "AM with growing modulation should have positive trend"

    def test_random_walk_envelope(self, random_walk):
        """Random walk → positive envelope, higher variability than sine."""
        result = hilbert_envelope(random_walk)
        assert result['envelope_mean'] > 0
        assert result['envelope_std'] > 0

    def test_short_signal_nan(self):
        """Insufficient data → all NaN."""
        result = hilbert_envelope(np.array([1.0, 2.0]))
        assert np.isnan(result['envelope_mean'])
        assert np.isnan(result['envelope_trend'])
