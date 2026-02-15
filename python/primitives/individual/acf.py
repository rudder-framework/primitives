"""ACF-derived features."""
import numpy as np


def acf_half_life(data: np.ndarray, max_lag: int = None, threshold: float = 0.5) -> float:
    """
    Lag at which autocorrelation drops below threshold.

    Args:
        data: 1-D signal array
        max_lag: Maximum lag to check. Default: min(n//4, 100).
        threshold: ACF value to cross. Default 0.5 (half-life).
                   Use 1/e ~ 0.368 for e-folding time.

    Returns:
        float — lag at which |ACF| first drops below threshold.
        Returns max_lag if ACF never drops below threshold (strong memory).
        Returns NaN if signal too short.
    """
    data = np.asarray(data, dtype=float).ravel()
    n = len(data)

    if n < 10:
        return np.nan

    if max_lag is None:
        max_lag = min(n // 4, 100)

    mean = np.mean(data)
    var = np.var(data)

    if var < 1e-20:
        return float(max_lag)  # constant signal = infinite memory

    for lag in range(1, max_lag + 1):
        cov = np.mean((data[:-lag] - mean) * (data[lag:] - mean))
        acf_val = cov / var
        if abs(acf_val) < threshold:
            return float(lag)

    return float(max_lag)
