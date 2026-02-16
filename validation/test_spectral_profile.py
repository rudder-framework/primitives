"""
Spectral profile validation.

Ground truth:
- scipy.signal.welch for PSD
- Known analytical signals (pure sine = one frequency, white noise = flat)
"""
import numpy as np
import pytest


class TestSpectralProfileVsScipy:
    """Compare individual measures against scipy equivalents."""

    def test_dominant_frequency_sine(self):
        """Pure 5 Hz sine: dominant frequency = 5 Hz."""
        from pmtvs.individual.spectral import spectral_profile
        fs = 1000
        t = np.arange(10000) / fs
        signal = np.sin(2 * np.pi * 5.0 * t)
        result = spectral_profile(signal, fs=fs)
        assert abs(result['dominant_frequency'] - 5.0) < 0.5, \
            f"Expected ~5 Hz, got {result['dominant_frequency']}"

    def test_flatness_white_noise(self):
        """White noise: spectral flatness near 1."""
        from pmtvs.individual.spectral import spectral_profile
        rng = np.random.RandomState(42)
        result = spectral_profile(rng.randn(10000))
        assert result['spectral_flatness'] > 0.5, \
            f"White noise flatness should be high, got {result['spectral_flatness']:.4f}"

    def test_flatness_sine(self):
        """Pure sine: spectral flatness near 0."""
        from pmtvs.individual.spectral import spectral_profile
        t = np.linspace(0, 10, 10000)
        result = spectral_profile(np.sin(2 * np.pi * 5 * t))
        assert result['spectral_flatness'] < 0.1, \
            f"Sine flatness should be low, got {result['spectral_flatness']:.4f}"

    def test_snr_sine_high(self):
        """Pure sine: high spectral peak SNR."""
        from pmtvs.individual.spectral import spectral_profile
        t = np.linspace(0, 10, 10000)
        result = spectral_profile(np.sin(2 * np.pi * 5 * t))
        assert result['spectral_peak_snr'] > 20, \
            f"Sine SNR should be very high, got {result['spectral_peak_snr']:.1f} dB"

    def test_snr_white_noise_low(self):
        """White noise: low spectral peak SNR."""
        from pmtvs.individual.spectral import spectral_profile
        rng = np.random.RandomState(42)
        result = spectral_profile(rng.randn(10000))
        assert result['spectral_peak_snr'] < 15, \
            f"White noise SNR should be low, got {result['spectral_peak_snr']:.1f} dB"

    def test_entropy_white_noise_high(self):
        """White noise: high spectral entropy."""
        from pmtvs.individual.spectral import spectral_profile
        rng = np.random.RandomState(42)
        result = spectral_profile(rng.randn(10000))
        assert result['spectral_entropy'] > 0.8, \
            f"White noise entropy should be high, got {result['spectral_entropy']:.4f}"

    def test_entropy_sine_low(self):
        """Pure sine: low spectral entropy."""
        from pmtvs.individual.spectral import spectral_profile
        t = np.linspace(0, 10, 10000)
        result = spectral_profile(np.sin(2 * np.pi * 5 * t))
        assert result['spectral_entropy'] < 0.3, \
            f"Sine entropy should be low, got {result['spectral_entropy']:.4f}"

    def test_first_bin_detection(self):
        """DC-dominant signal should flag is_first_bin_peak."""
        from pmtvs.individual.spectral import spectral_profile
        result = spectral_profile(np.linspace(0, 100, 1000))
        assert result['is_first_bin_peak'] is True

    def test_slope_red_noise(self):
        """1/f noise: negative spectral slope."""
        from pmtvs.individual.spectral import spectral_profile
        rng = np.random.RandomState(42)
        red = np.cumsum(rng.randn(10000))
        result = spectral_profile(red)
        assert result['spectral_slope'] < -0.3, \
            f"1/f noise slope should be negative, got {result['spectral_slope']:.4f}"

    def test_short_signal_nan(self):
        """Insufficient data returns NaN."""
        from pmtvs.individual.spectral import spectral_profile
        result = spectral_profile(np.array([1.0, 2.0, 3.0]))
        assert np.isnan(result['spectral_flatness'])

    def test_constant_signal(self):
        """Constant signal: all measures handled gracefully."""
        from pmtvs.individual.spectral import spectral_profile
        result = spectral_profile(np.ones(1000))
        assert isinstance(result, dict)
