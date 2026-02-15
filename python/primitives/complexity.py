"""Python fallback implementations for complexity primitives."""
import numpy as np
from math import factorial
from typing import Optional


def hurst_exponent(signal: np.ndarray, method: str = 'rs') -> float:
    """
    Hurst exponent via R/S analysis.

    H < 0.5: anti-persistent (mean-reverting)
    H = 0.5: random walk
    H > 0.5: persistent (trending)
    """
    signal = np.asarray(signal, dtype=np.float64).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    RS_MIN_K = 10

    if n < RS_MIN_K:
        return np.nan

    if method == 'dfa':
        return _dfa(signal)

    # Rescaled Range (R/S) method
    max_k = min(int(n * 0.5), 100)
    k_values = []
    rs_values = []

    for k in range(RS_MIN_K, max_k):
        n_subseries = n // k
        rs_sum = 0

        for i in range(n_subseries):
            subseries = signal[i * k:(i + 1) * k]
            mean = np.mean(subseries)
            Y = np.cumsum(subseries - mean)
            R = np.max(Y) - np.min(Y)
            S = np.std(subseries, ddof=1)

            if S > 0:
                rs_sum += R / S

        if n_subseries > 0:
            rs_avg = rs_sum / n_subseries
            if rs_avg > 0:
                k_values.append(np.log(k))
                rs_values.append(np.log(rs_avg))

    if len(k_values) < 3:
        return np.nan

    H, _ = np.polyfit(k_values, rs_values, 1)
    return float(np.clip(H, 0, 1))


def _dfa(signal: np.ndarray, order: int = 1) -> float:
    """Detrended Fluctuation Analysis."""
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    if n < 20:
        return np.nan

    Y = np.cumsum(signal - np.mean(signal))

    min_scale = 10
    max_scale = min(int(n * 0.25), 100)

    scales = np.unique(np.logspace(
        np.log10(min_scale), np.log10(max_scale), 20
    ).astype(int))

    fluctuations = []

    for scale in scales:
        n_segments = n // scale
        if n_segments < 2:
            continue

        F_sq = []
        for i in range(n_segments):
            segment = Y[i * scale:(i + 1) * scale]
            x = np.arange(scale)
            coeffs = np.polyfit(x, segment, order)
            trend = np.polyval(coeffs, x)
            F_sq.append(np.mean((segment - trend) ** 2))

        if F_sq:
            fluctuations.append(np.sqrt(np.mean(F_sq)))

    if len(fluctuations) < 3:
        return np.nan

    log_scales = np.log(scales[:len(fluctuations)])
    log_fluct = np.log(fluctuations)
    alpha, _ = np.polyfit(log_scales, log_fluct, 1)

    return float(alpha)


def sample_entropy(signal: np.ndarray, m: int = 2, r: Optional[float] = None) -> float:
    """
    Sample entropy.

    SampEn = -log(A/B) where:
    - B = count of template matches of length m
    - A = count of template matches of length m+1
    """
    signal = np.asarray(signal, dtype=np.float64).flatten()
    n = len(signal)

    if r is None:
        r = 0.2 * np.std(signal)

    if r == 0 or n < m + 2:
        return np.nan

    def count_matches(template_len):
        count = 0
        templates = np.array([signal[i:i + template_len] for i in range(n - template_len)])
        for i in range(len(templates)):
            for j in range(i + 1, len(templates)):
                if np.max(np.abs(templates[i] - templates[j])) < r:
                    count += 1
        return count

    B = count_matches(m)
    A = count_matches(m + 1)

    if B == 0:
        return np.nan

    return -np.log(A / B) if A > 0 else np.nan


def permutation_entropy(
    signal: np.ndarray,
    order: int = 3,
    delay: int = 1,
    normalize: bool = True,
) -> float:
    """
    Permutation entropy.

    Measures complexity via ordinal pattern distribution.
    """
    signal = np.asarray(signal, dtype=np.float64).flatten()
    n = len(signal)

    if n < order * delay:
        return np.nan

    n_patterns = n - (order - 1) * delay
    patterns = np.zeros((n_patterns, order))

    for i in range(n_patterns):
        for j in range(order):
            patterns[i, j] = signal[i + j * delay]

    perms = np.argsort(patterns, axis=1)
    perm_strings = [''.join(map(str, p)) for p in perms]
    unique, counts = np.unique(perm_strings, return_counts=True)

    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs))

    if normalize:
        max_entropy = np.log2(factorial(order))
        if max_entropy > 0:
            entropy = entropy / max_entropy

    return float(entropy)
