"""
Unit tests for hilbert_envelope primitive.
"""
import numpy as np
import pytest

from pmtvs.individual.hilbert import hilbert_envelope


class TestHilbertEnvelopeBasic:
    """Basic contract tests."""

    def test_returns_dict(self):
        signal = np.sin(np.linspace(0, 10 * np.pi, 500))
        result = hilbert_envelope(signal)
        assert isinstance(result, dict)

    def test_all_keys_present(self):
        signal = np.sin(np.linspace(0, 10 * np.pi, 500))
        result = hilbert_envelope(signal)
        expected_keys = {
            'envelope_mean', 'envelope_std', 'envelope_max',
            'envelope_min', 'envelope_range', 'envelope_trend',
            'envelope_cv',
        }
        assert set(result.keys()) == expected_keys

    def test_all_values_are_float(self):
        signal = np.sin(np.linspace(0, 10 * np.pi, 500))
        result = hilbert_envelope(signal)
        for key, val in result.items():
            assert isinstance(val, (float, np.floating)), \
                f"{key} should be float, got {type(val)}"


class TestHilbertEnvelopeEdgeCases:
    """Edge case handling."""

    def test_short_signal_returns_nan(self):
        result = hilbert_envelope(np.array([1.0]))
        for key, val in result.items():
            assert np.isnan(val), f"{key} should be NaN for short signal"

    def test_length_3_returns_nan(self):
        result = hilbert_envelope(np.array([1.0, 2.0, 3.0]))
        assert np.isnan(result['envelope_mean'])

    def test_length_4_works(self):
        result = hilbert_envelope(np.array([1.0, -1.0, 1.0, -1.0]))
        assert not np.isnan(result['envelope_mean'])

    def test_constant_signal(self):
        result = hilbert_envelope(np.ones(100))
        # After centering, signal is all zeros → envelope ≈ 0
        assert result['envelope_mean'] < 1e-10
        assert result['envelope_std'] < 1e-10

    def test_nan_in_signal(self):
        signal = np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0])
        result = hilbert_envelope(signal)
        # Should handle NaN by dropping, leaving 5 points (>= 4)
        assert not np.isnan(result['envelope_mean'])

    def test_all_nan_returns_nan(self):
        signal = np.array([np.nan, np.nan, np.nan, np.nan])
        result = hilbert_envelope(signal)
        assert np.isnan(result['envelope_mean'])

    def test_empty_array_returns_nan(self):
        result = hilbert_envelope(np.array([]))
        assert np.isnan(result['envelope_mean'])


class TestHilbertEnvelopeValues:
    """Known-value tests."""

    def test_pure_sine_constant_envelope(self):
        """Pure sine wave → constant envelope ≈ 1.0."""
        t = np.linspace(0, 10 * np.pi, 1000)
        signal = np.sin(t)
        result = hilbert_envelope(signal)
        # Envelope of a pure sine (after centering) should be ~1.0
        assert abs(result['envelope_mean'] - 1.0) < 0.05
        # std should be small (nearly constant envelope)
        assert result['envelope_std'] < 0.1
        # trend should be near zero
        assert abs(result['envelope_trend']) < 0.001

    def test_am_signal_positive_trend(self):
        """Amplitude-modulated signal with growing amplitude → positive trend."""
        t = np.linspace(0, 10 * np.pi, 1000)
        modulation = np.linspace(0.5, 2.0, 1000)
        signal = np.sin(t) * modulation
        result = hilbert_envelope(signal)
        assert result['envelope_trend'] > 0, "Growing amplitude should give positive trend"

    def test_am_signal_negative_trend(self):
        """Decaying amplitude → negative trend."""
        t = np.linspace(0, 10 * np.pi, 1000)
        modulation = np.linspace(2.0, 0.5, 1000)
        signal = np.sin(t) * modulation
        result = hilbert_envelope(signal)
        assert result['envelope_trend'] < 0, "Decaying amplitude should give negative trend"

    def test_range_is_max_minus_min(self):
        signal = np.random.RandomState(42).randn(500)
        result = hilbert_envelope(signal)
        assert abs(result['envelope_range'] - (result['envelope_max'] - result['envelope_min'])) < 1e-12

    def test_envelope_non_negative(self):
        """Envelope (absolute value of analytic signal) is always non-negative."""
        signal = np.random.RandomState(42).randn(500)
        result = hilbert_envelope(signal)
        assert result['envelope_min'] >= 0
        assert result['envelope_mean'] >= 0

    def test_cv_is_std_over_mean(self):
        signal = np.sin(np.linspace(0, 20 * np.pi, 1000)) + 2 * np.sin(np.linspace(0, 2 * np.pi, 1000))
        result = hilbert_envelope(signal)
        if result['envelope_mean'] > 1e-12:
            expected_cv = result['envelope_std'] / result['envelope_mean']
            assert abs(result['envelope_cv'] - expected_cv) < 1e-12
