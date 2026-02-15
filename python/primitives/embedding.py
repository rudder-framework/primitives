"""Python fallback implementations for embedding primitives."""
import numpy as np
from typing import Optional


def time_delay_embedding(
    signal: np.ndarray,
    dimension: int = 3,
    delay: int = 1,
) -> np.ndarray:
    """
    Construct time-delay embedding.

    Returns (n_points, dimension) array.
    """
    signal = np.asarray(signal, dtype=np.float64).flatten()
    n = len(signal)
    n_points = n - (dimension - 1) * delay

    if n_points <= 0:
        raise ValueError(
            f"Signal too short for embedding: {n} samples, "
            f"need at least {(dimension - 1) * delay + 1}"
        )

    embedded = np.zeros((n_points, dimension))
    for d in range(dimension):
        start = d * delay
        embedded[:, d] = signal[start:start + n_points]

    return embedded


def optimal_delay(
    signal: np.ndarray,
    max_lag: Optional[int] = None,
    method: str = 'mutual_info',
) -> int:
    """
    Estimate optimal time delay for embedding.

    Methods:
        'mutual_info': First minimum of mutual information (default)
        'autocorr': First zero crossing of autocorrelation
        'autocorr_e': First 1/e decay of autocorrelation
    """
    signal = np.asarray(signal, dtype=np.float64).flatten()
    n = len(signal)

    if max_lag is None:
        max_lag = n // 4
    max_lag = min(max_lag, n // 2)

    if n < 4:
        return 1

    # Constant signal — no meaningful delay
    if np.ptp(signal) < 1e-15:
        return 1

    if method == 'autocorr':
        signal_centered = signal - np.mean(signal)
        var = np.var(signal_centered)
        if var == 0:
            return 1
        for lag in range(1, max_lag):
            acf = np.mean(signal_centered[:-lag] * signal_centered[lag:]) / var
            if acf <= 0:
                return lag
        return max_lag

    elif method == 'autocorr_e':
        signal_centered = signal - np.mean(signal)
        var = np.var(signal_centered)
        if var == 0:
            return 1
        threshold = 1 / np.e
        for lag in range(1, max_lag):
            acf = np.mean(signal_centered[:-lag] * signal_centered[lag:]) / var
            if acf <= threshold:
                return lag
        return max_lag

    elif method == 'mutual_info':
        mi_values = []
        for lag in range(1, max_lag):
            mi = _lagged_mutual_info(signal, lag, bins=16)
            mi_values.append(mi)

            if len(mi_values) >= 3:
                if mi_values[-2] < mi_values[-1] and mi_values[-2] < mi_values[-3]:
                    return lag - 1

        if mi_values:
            initial_mi = mi_values[0]
            for lag, mi in enumerate(mi_values, 1):
                if mi < 0.5 * initial_mi:
                    return lag

        return max(1, max_lag // 4)

    else:
        raise ValueError(f"Unknown method: {method}")


def optimal_dimension(
    signal: np.ndarray,
    delay: Optional[int] = None,
    max_dim: int = 10,
    method: str = 'fnn',
    threshold: float = 0.01,
) -> int:
    """
    Estimate optimal embedding dimension via FNN or Cao's method.
    """
    signal = np.asarray(signal, dtype=np.float64).flatten()
    n = len(signal)
    tau = delay if delay is not None else 1

    if n < (max_dim + 1) * tau + 2:
        return 2

    if method == 'fnn':
        return _fnn_dimension(signal, tau, max_dim, threshold)
    elif method == 'cao':
        return _cao_dimension(signal, tau, max_dim, threshold)
    else:
        raise ValueError(f"Unknown method: {method}")


def _lagged_mutual_info(signal: np.ndarray, lag: int, bins: int = 16) -> float:
    """Compute mutual information between signal and its lagged version."""
    x = signal[:-lag]
    y = signal[lag:]

    hist_xy, _, _ = np.histogram2d(x, y, bins=bins)
    pxy = hist_xy / hist_xy.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if pxy[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += pxy[i, j] * np.log(pxy[i, j] / (px[i] * py[j]))
    return mi


def _fnn_dimension(signal, delay, max_dim, threshold):
    """False Nearest Neighbors method."""
    for dim in range(1, max_dim):
        n_points = len(signal) - (dim + 1) * delay
        if n_points < 10:
            return max(dim, 2)

        embed = np.array([
            [signal[i + d * delay] for d in range(dim)]
            for i in range(n_points)
        ])

        fnn_count = 0
        total = 0
        n_check = min(n_points, 1000)

        for i in range(n_check):
            dists = np.linalg.norm(embed - embed[i], axis=1)
            dists[i] = np.inf
            best_j = np.argmin(dists)
            best_dist = dists[best_j]

            if best_dist > 1e-10:
                r_d1 = abs(signal[i + dim * delay] - signal[best_j + dim * delay])
                if r_d1 / best_dist > 10.0:
                    fnn_count += 1
                total += 1

        fnn_ratio = fnn_count / total if total > 0 else 0.0
        if fnn_ratio < threshold:
            return dim + 1

    return max_dim


def _cao_dimension(signal, delay, max_dim, threshold):
    """Cao's method for optimal dimension."""
    prev_e = 0.0

    for dim in range(1, max_dim + 1):
        n_points = len(signal) - dim * delay
        if n_points < 10:
            return max(dim, 2)

        prev_dim = max(dim - 1, 1)
        n_prev = len(signal) - prev_dim * delay
        n_pts = min(n_points, n_prev)

        embed_prev = np.array([
            [signal[i + d * delay] for d in range(prev_dim)]
            for i in range(n_pts)
        ])

        n_check = min(n_pts, 500)
        e_sum = 0.0
        count = 0

        for i in range(n_check):
            dists = np.linalg.norm(embed_prev - embed_prev[i], axis=1)
            dists[i] = np.inf
            best_j = np.argmin(dists)
            best_dist = dists[best_j]

            if best_dist > 1e-10:
                embed_i = np.array([signal[i + d * delay] for d in range(dim)])
                embed_j = np.array([signal[best_j + d * delay] for d in range(dim)])
                r_d1 = np.linalg.norm(embed_i - embed_j)
                e_sum += r_d1 / best_dist
                count += 1

        e = e_sum / count if count > 0 else 1.0

        if dim > 1 and prev_e > 1e-15:
            e1 = e / prev_e
            if abs(e1 - 1.0) < threshold:
                return dim

        prev_e = e

    return max_dim
