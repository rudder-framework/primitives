"""
Hilbert Transform Primitives (24-27)

Analytic signal, envelope, instantaneous frequency/amplitude.
"""

import numpy as np
from scipy import signal as scipy_signal


def hilbert_transform(signal: np.ndarray) -> np.ndarray:
    """
    Compute Hilbert transform (analytic signal).

    Parameters
    ----------
    signal : np.ndarray
        Input signal (real)

    Returns
    -------
    np.ndarray
        Complex analytic signal

    Notes
    -----
    z(t) = x(t) + i*H[x](t)
    where H is the Hilbert transform.
    """
    signal = np.asarray(signal)
    return scipy_signal.hilbert(signal)


def envelope(signal: np.ndarray) -> np.ndarray:
    """
    Compute signal envelope (amplitude modulation).

    Parameters
    ----------
    signal : np.ndarray
        Input signal

    Returns
    -------
    np.ndarray
        Envelope (instantaneous amplitude)

    Notes
    -----
    A(t) = |z(t)| = sqrt(x² + H[x]²)
    Useful for detecting amplitude modulation.
    """
    analytic = hilbert_transform(signal)
    return np.abs(analytic)


def instantaneous_amplitude(signal: np.ndarray) -> np.ndarray:
    """
    Compute instantaneous amplitude.

    Parameters
    ----------
    signal : np.ndarray
        Input signal

    Returns
    -------
    np.ndarray
        Instantaneous amplitude (same as envelope)
    """
    return envelope(signal)


def instantaneous_frequency(
    signal: np.ndarray,
    fs: float = 1.0
) -> np.ndarray:
    """
    Compute instantaneous frequency.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    fs : float
        Sampling frequency

    Returns
    -------
    np.ndarray
        Instantaneous frequency

    Notes
    -----
    f(t) = (1/2π) * d(phase)/dt
    where phase = angle(z(t))
    """
    analytic = hilbert_transform(signal)
    phase = np.unwrap(np.angle(analytic))
    inst_freq = np.gradient(phase, 1/fs) / (2 * np.pi)
    return inst_freq


def instantaneous_phase(signal: np.ndarray) -> np.ndarray:
    """
    Compute instantaneous phase.

    Parameters
    ----------
    signal : np.ndarray
        Input signal

    Returns
    -------
    np.ndarray
        Instantaneous phase (unwrapped)
    """
    analytic = hilbert_transform(signal)
    return np.unwrap(np.angle(analytic))


def hilbert_envelope(signal: np.ndarray) -> dict:
    """
    Compute instantaneous amplitude envelope statistics of a 1D signal.

    Parameters
    ----------
    signal : np.ndarray
        1D array of real values. Length >= 4.

    Returns
    -------
    dict with keys:
        envelope_mean : float     — mean of envelope (average amplitude)
        envelope_std : float      — std of envelope (amplitude variability)
        envelope_max : float      — peak envelope value
        envelope_min : float      — minimum envelope value
        envelope_range : float    — max - min (dynamic range of amplitude modulation)
        envelope_trend : float    — slope of linear fit to envelope (amplitude growing/shrinking)
        envelope_cv : float       — coefficient of variation (std/mean, normalized variability)

    Notes
    -----
    analytic_signal = signal + j * hilbert(signal)
    envelope = |analytic_signal|

    The signal is mean-centered before computing the analytic signal
    to avoid DC component dominating the envelope.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()

    nan_result = {
        'envelope_mean': np.nan,
        'envelope_std': np.nan,
        'envelope_max': np.nan,
        'envelope_min': np.nan,
        'envelope_range': np.nan,
        'envelope_trend': np.nan,
        'envelope_cv': np.nan,
    }

    if len(signal) < 4:
        return nan_result

    # Drop NaN values
    signal = signal[np.isfinite(signal)]
    if len(signal) < 4:
        return nan_result

    # Remove mean to avoid DC component dominating envelope
    centered = signal - np.mean(signal)

    # Analytic signal via Hilbert transform
    analytic = scipy_signal.hilbert(centered)
    env = np.abs(analytic)

    env_mean = float(np.mean(env))
    env_std = float(np.std(env))
    env_max = float(np.max(env))
    env_min = float(np.min(env))

    # Linear trend of envelope
    t = np.arange(len(env), dtype=np.float64)
    if len(env) > 1:
        coeffs = np.polyfit(t, env, 1)
        trend = float(coeffs[0])
    else:
        trend = 0.0

    cv = float(env_std / env_mean) if env_mean > 1e-12 else np.nan

    return {
        'envelope_mean': env_mean,
        'envelope_std': env_std,
        'envelope_max': env_max,
        'envelope_min': env_min,
        'envelope_range': float(env_max - env_min),
        'envelope_trend': trend,
        'envelope_cv': cv,
    }
