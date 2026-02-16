"""
Spectral Primitives (17-23)

FFT, PSD, and frequency-domain analysis.
"""

import numpy as np
from scipy import signal as scipy_signal
from typing import Tuple, Optional

from prmtvs._config import USE_RUST as _USE_RUST_SPECTRAL

if _USE_RUST_SPECTRAL:
    try:
        from prmtvs._rust import (
            psd as _psd_rs,
            spectral_entropy as _spectral_entropy_rs,
            spectral_centroid as _spectral_centroid_rs,
        )
    except ImportError:
        _USE_RUST_SPECTRAL = False


def fft(signal: np.ndarray) -> np.ndarray:
    """
    Compute Fast Fourier Transform.

    Parameters
    ----------
    signal : np.ndarray
        Input signal (time domain)

    Returns
    -------
    np.ndarray
        Complex FFT coefficients

    Notes
    -----
    X[k] = sum_{n=0}^{N-1} x[n] * exp(-2πi*k*n/N)
    """
    signal = np.asarray(signal)
    return np.fft.fft(signal)


def psd(
    signal: np.ndarray,
    fs: float = 1.0,
    nperseg: int = None,
    method: str = 'welch'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Power Spectral Density.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    fs : float
        Sampling frequency
    nperseg : int, optional
        Segment length for Welch method
    method : str
        'welch' (averaged) or 'periodogram' (raw)

    Returns
    -------
    tuple
        (frequencies, power_spectral_density)

    Notes
    -----
    Welch method reduces variance by averaging.
    """
    signal = np.asarray(signal).flatten()

    if nperseg is None:
        nperseg = min(256, len(signal))

    if _USE_RUST_SPECTRAL and method == 'welch':
        f, p = _psd_rs(signal, fs, nperseg)
        return np.asarray(f), np.asarray(p)

    if method == 'welch':
        freqs, Pxx = scipy_signal.welch(signal, fs=fs, nperseg=nperseg)
    else:
        freqs, Pxx = scipy_signal.periodogram(signal, fs=fs)

    return freqs, Pxx


def dominant_frequency(
    signal: np.ndarray,
    fs: float = 1.0
) -> float:
    """
    Find dominant (peak) frequency.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    fs : float
        Sampling frequency

    Returns
    -------
    float
        Frequency with maximum power

    Notes
    -----
    f_dominant = argmax(PSD)
    """
    freqs, Pxx = psd(signal, fs)
    return float(freqs[np.argmax(Pxx)])


def spectral_centroid(
    signal: np.ndarray,
    fs: float = 1.0
) -> float:
    """
    Compute spectral centroid.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    fs : float
        Sampling frequency

    Returns
    -------
    float
        Spectral centroid (center of mass of spectrum)

    Notes
    -----
    f_c = sum(f * P(f)) / sum(P(f))
    "Brightness" of the signal.
    """
    if _USE_RUST_SPECTRAL:
        return _spectral_centroid_rs(np.asarray(signal).flatten(), fs)

    freqs, Pxx = psd(signal, fs)
    total_power = np.sum(Pxx)
    if total_power == 0:
        return 0.0
    return float(np.sum(freqs * Pxx) / total_power)


def spectral_bandwidth(
    signal: np.ndarray,
    fs: float = 1.0,
    p: int = 2
) -> float:
    """
    Compute spectral bandwidth.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    fs : float
        Sampling frequency
    p : int
        Order (2 for variance-based)

    Returns
    -------
    float
        Spectral bandwidth

    Notes
    -----
    BW = (sum(|f - f_c|^p * P(f)) / sum(P(f)))^(1/p)
    Spread of spectrum around centroid.
    """
    freqs, Pxx = psd(signal, fs)
    centroid = spectral_centroid(signal, fs)
    total_power = np.sum(Pxx)
    if total_power == 0:
        return 0.0
    return float((np.sum(np.abs(freqs - centroid)**p * Pxx) / total_power) ** (1/p))


def spectral_entropy(
    signal: np.ndarray,
    fs: float = 1.0,
    normalize: bool = True
) -> float:
    """
    Compute spectral entropy.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    fs : float
        Sampling frequency
    normalize : bool
        If True, normalize to [0, 1]

    Returns
    -------
    float
        Spectral entropy

    Notes
    -----
    H = -sum(P_norm * log(P_norm))
    Measures flatness/complexity of spectrum.
    High: white noise (flat spectrum)
    Low: pure tone (peaked spectrum)
    """
    if _USE_RUST_SPECTRAL:
        return _spectral_entropy_rs(np.asarray(signal).flatten(), fs, normalize)

    freqs, Pxx = psd(signal, fs)

    # Normalize to probability distribution
    total_power = np.sum(Pxx)
    if total_power == 0:
        return 0.0

    P_norm = Pxx / total_power
    P_norm = P_norm[P_norm > 0]  # Avoid log(0)

    entropy = -np.sum(P_norm * np.log2(P_norm))

    if normalize:
        max_entropy = np.log2(len(Pxx))
        if max_entropy > 0:
            entropy = entropy / max_entropy

    return float(entropy)


def wavelet_coeffs(
    signal: np.ndarray,
    wavelet: str = 'db4',
    level: int = None
) -> list:
    """
    Compute discrete wavelet transform coefficients.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    wavelet : str
        Wavelet name (e.g., 'db4', 'haar', 'sym5')
    level : int, optional
        Decomposition level

    Returns
    -------
    list
        [cA_n, cD_n, cD_{n-1}, ..., cD_1]
        Approximation and detail coefficients

    Notes
    -----
    DWT decomposes signal into frequency bands.
    cA: low-frequency approximation
    cD: high-frequency details
    """
    try:
        import pywt
        signal = np.asarray(signal)

        if level is None:
            level = pywt.dwt_max_level(len(signal), wavelet)
            level = min(level, 6)  # Cap at 6

        coeffs = pywt.wavedec(signal, wavelet, level=level)
        return coeffs
    except ImportError:
        # Return simple high/low frequency split
        signal = np.asarray(signal)
        n = len(signal)
        fft_signal = np.fft.fft(signal)
        mid = n // 2

        # Low frequency (approximation)
        low_freq = fft_signal.copy()
        low_freq[mid//2:-mid//2] = 0
        cA = np.real(np.fft.ifft(low_freq))

        # High frequency (detail)
        high_freq = fft_signal - low_freq
        cD = np.real(np.fft.ifft(high_freq))

        return [cA, cD]


def spectral_profile(data: np.ndarray, fs: float = 1.0, fft_size: int = None) -> dict:
    """
    Compute all spectral measures from a single FFT.

    One FFT, all measures. This is what Prime calls for typology_raw.

    Args:
        data: 1-D signal array
        fs: Sampling frequency (Hz). Default 1.0 (normalized).
        fft_size: FFT length. Default None (use signal length).

    Returns:
        dict with keys:
            spectral_flatness: float
            spectral_slope: float
            spectral_peak_snr: float
            dominant_frequency: float
            harmonic_noise_ratio: float
            spectral_entropy: float
            spectral_centroid: float
            spectral_rolloff: float
            is_first_bin_peak: bool

    Insufficient data -> all NaN.
    """
    data = np.asarray(data, dtype=float).ravel()
    n = len(data)

    nan_result = {
        'spectral_flatness': np.nan,
        'spectral_slope': np.nan,
        'spectral_peak_snr': np.nan,
        'dominant_frequency': np.nan,
        'harmonic_noise_ratio': np.nan,
        'spectral_entropy': np.nan,
        'spectral_centroid': np.nan,
        'spectral_rolloff': np.nan,
        'is_first_bin_peak': False,
    }

    if n < 8:
        return nan_result

    # --- Single FFT ---
    if fft_size is None:
        fft_size = n
    windowed = data - np.mean(data)
    fft_vals = np.fft.rfft(windowed, n=fft_size)
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(fft_size, 1.0 / fs)

    total_power = np.sum(power)
    if total_power < 1e-20:
        return nan_result

    # --- Dominant frequency ---
    peak_idx = np.argmax(power)
    dominant_freq = float(freqs[peak_idx])
    is_first_bin_peak = bool(peak_idx <= 1)

    # --- Spectral flatness (geometric mean / arithmetic mean) ---
    power_pos = power[power > 0]
    if len(power_pos) > 0:
        log_mean = np.mean(np.log(power_pos))
        geo_mean = np.exp(log_mean)
        arith_mean = np.mean(power_pos)
        sf = float(geo_mean / arith_mean) if arith_mean > 0 else np.nan
    else:
        sf = np.nan

    # --- Spectral slope (log-log regression) ---
    pos_mask = (freqs > 0) & (power > 0)
    if np.sum(pos_mask) > 2:
        log_f = np.log10(freqs[pos_mask])
        log_p = np.log10(power[pos_mask])
        coeffs = np.polyfit(log_f, log_p, 1)
        slope = float(coeffs[0])
    else:
        slope = np.nan

    # --- Spectral peak SNR (dB above median) ---
    median_power = np.median(power[power > 0]) if np.any(power > 0) else 1e-20
    peak_power = power[peak_idx]
    snr = float(10 * np.log10(peak_power / median_power)) if median_power > 0 else np.nan

    # --- Harmonic noise ratio ---
    non_peak_power = total_power - peak_power
    hnr = float(peak_power / non_peak_power) if non_peak_power > 0 else np.nan

    # --- Spectral entropy ---
    p_norm = power / total_power
    p_pos = p_norm[p_norm > 0]
    entropy = -np.sum(p_pos * np.log2(p_pos))
    max_entropy = np.log2(len(power))
    se = float(entropy / max_entropy) if max_entropy > 0 else 0.0

    # --- Spectral centroid ---
    sc = float(np.sum(freqs * power) / total_power)

    # --- Spectral rolloff (85% cumulative power) ---
    cumsum = np.cumsum(power) / total_power
    rolloff_idx = np.searchsorted(cumsum, 0.85)
    sr = float(freqs[min(rolloff_idx, len(freqs) - 1)])

    return {
        'spectral_flatness': sf,
        'spectral_slope': slope,
        'spectral_peak_snr': snr,
        'dominant_frequency': dominant_freq,
        'harmonic_noise_ratio': hnr,
        'spectral_entropy': se,
        'spectral_centroid': sc,
        'spectral_rolloff': sr,
        'is_first_bin_peak': is_first_bin_peak,
    }
