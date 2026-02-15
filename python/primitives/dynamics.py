"""Python fallback implementations for dynamics primitives."""
import numpy as np
from typing import Optional, Tuple
from scipy.spatial.distance import pdist, squareform


def lyapunov_rosenstein(
    signal: np.ndarray,
    dimension: Optional[int] = None,
    delay: Optional[int] = None,
    min_tsep: Optional[int] = None,
    max_iter: Optional[int] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Rosenstein's method for largest Lyapunov exponent.

    Returns (lambda_max, divergence_curve, iterations).
    """
    signal = np.asarray(signal, dtype=np.float64).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    if n < 50:
        return np.nan, np.array([]), np.array([])

    dim = dimension if dimension is not None else 3
    tau = delay if delay is not None else 1
    tsep = min_tsep if min_tsep is not None else tau * dim
    max_it = max_iter if max_iter is not None else min(n // 10, 500)

    # Embed
    n_vectors = n - (dim - 1) * tau
    if n_vectors < 2 * tsep:
        return np.nan, np.array([]), np.array([])

    embedded = np.array([
        signal[i:i + dim * tau:tau]
        for i in range(n_vectors)
    ])

    # Find nearest neighbors (excluding temporal neighbors)
    distances = squareform(pdist(embedded))
    for i in range(len(distances)):
        for j in range(max(0, i - tsep), min(len(distances), i + tsep + 1)):
            distances[i, j] = np.inf

    nearest = np.argmin(distances, axis=1)

    # Track divergence
    divergences = []
    for step in range(1, min(max_it, n_vectors - int(np.max(nearest)) - 1)):
        step_divs = []
        for i in range(len(nearest)):
            j = nearest[i]
            if i + step < n_vectors and j + step < n_vectors:
                d0 = distances[i, j]
                d1 = np.linalg.norm(embedded[i + step] - embedded[j + step])
                if d0 > 0 and d1 > 0:
                    step_divs.append(np.log(d1 / d0))
        if step_divs:
            divergences.append(np.mean(step_divs))

    if len(divergences) < 2:
        return np.nan, np.array([]), np.array([])

    steps = np.arange(1, len(divergences) + 1, dtype=np.float64)
    slope, _ = np.polyfit(steps, divergences, 1)

    return float(slope), np.array(divergences), steps


def lyapunov_kantz(
    signal: np.ndarray,
    dimension: Optional[int] = None,
    delay: Optional[int] = None,
    min_tsep: Optional[int] = None,
    epsilon: Optional[float] = None,
    max_iter: Optional[int] = None,
) -> Tuple[float, np.ndarray]:
    """
    Kantz's method for largest Lyapunov exponent.

    Returns (max_lyapunov, divergence_curve).
    """
    signal = np.asarray(signal, dtype=np.float64).flatten()
    n = len(signal)

    dim = dimension if dimension is not None else 3
    tau = delay if delay is not None else 1
    tsep = min_tsep if min_tsep is not None else dim * tau

    n_points = n - (dim - 1) * tau
    if n_points < tsep + 2:
        return np.nan, np.array([])

    embedded = np.array([
        signal[i:i + dim * tau:tau]
        for i in range(n_points)
    ])

    eps = epsilon if epsilon is not None else np.std(signal) * 0.1
    max_it = min(max_iter if max_iter is not None else n_points // 4, n_points - 1)

    divergence = np.zeros(max_it)
    counts = np.zeros(max_it, dtype=int)

    for i in range(n_points):
        for j in range(n_points):
            if abs(i - j) < tsep:
                continue
            dist = np.linalg.norm(embedded[i] - embedded[j])
            if 0 < dist < eps:
                for k in range(max_it):
                    ii, jj = i + k, j + k
                    if ii >= n_points or jj >= n_points:
                        break
                    d = np.linalg.norm(embedded[ii] - embedded[jj])
                    if d > 0:
                        divergence[k] += np.log(d)
                        counts[k] += 1

    div_curve = []
    for k in range(max_it):
        if counts[k] > 0:
            div_curve.append(divergence[k] / counts[k])

    div_curve = np.array(div_curve)

    if len(div_curve) < 2:
        return np.nan, div_curve

    fit_len = max(2, min(len(div_curve) // 10, len(div_curve)))
    x = np.arange(fit_len, dtype=np.float64)
    slope, _ = np.polyfit(x, div_curve[:fit_len], 1)

    return float(slope), div_curve


def ftle_local_linearization(
    trajectory: np.ndarray,
    time_horizon: int = 10,
    n_neighbors: int = 10,
    epsilon: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    FTLE via local linearization of the flow.

    Returns (ftle, confidence) arrays.
    """
    from scipy.spatial import KDTree

    trajectory = np.asarray(trajectory, dtype=np.float64)
    if trajectory.ndim == 1:
        raise ValueError("Trajectory must be 2D (n_points, dimension)")

    n_points, dim = trajectory.shape
    n_valid = n_points - time_horizon

    if n_valid < 10:
        return np.full(n_points, np.nan), np.full(n_points, 0.0)

    tree = KDTree(trajectory)

    if epsilon is None:
        sample_idx = np.random.choice(n_points, min(100, n_points), replace=False)
        dists = []
        for i in sample_idx:
            for j in sample_idx:
                if i != j:
                    dists.append(np.linalg.norm(trajectory[i] - trajectory[j]))
        epsilon = np.percentile(dists, 20) if dists else 1.0

    ftle = np.full(n_points, np.nan)
    confidence = np.full(n_points, 0.0)

    for i in range(n_valid):
        indices = tree.query_ball_point(trajectory[i], epsilon)
        valid_neighbors = [j for j in indices if j != i and j + time_horizon < n_points]

        if len(valid_neighbors) < n_neighbors:
            _, indices = tree.query(trajectory[i], k=n_neighbors + 1)
            valid_neighbors = [j for j in indices[1:] if j + time_horizon < n_points]

        if len(valid_neighbors) < dim + 1:
            continue

        delta_x0 = np.array([trajectory[j] - trajectory[i] for j in valid_neighbors])
        delta_xT = np.array([
            trajectory[j + time_horizon] - trajectory[i + time_horizon]
            for j in valid_neighbors
        ])

        try:
            Phi_T, residuals, rank, s = np.linalg.lstsq(delta_x0, delta_xT, rcond=None)
            Phi = Phi_T.T

            singular_values = np.linalg.svd(Phi, compute_uv=False)
            sigma_max = singular_values[0]

            if sigma_max > 0:
                ftle[i] = (1.0 / time_horizon) * np.log(sigma_max)

            if residuals.size > 0 and np.sum(delta_xT ** 2) > 0:
                r2 = 1 - np.sum(residuals) / np.sum(delta_xT ** 2)
                confidence[i] = max(0, min(1, r2))
            else:
                confidence[i] = 0.5
        except (np.linalg.LinAlgError, ValueError):
            continue

    return ftle, confidence


def ftle_direct_perturbation(
    signal: np.ndarray,
    dimension: int = 3,
    delay: int = 1,
    time_horizon: int = 10,
    perturbation: float = 1e-6,
    n_perturbations: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    FTLE via direct perturbation of delay-embedded signal.

    Returns (ftle, stretching_directions).
    """
    signal = np.asarray(signal, dtype=np.float64).flatten()
    n = len(signal)

    n_points = n - (dimension - 1) * delay
    if n_points < time_horizon + 2:
        return np.array([]), np.array([])

    embedded = np.zeros((n_points, dimension))
    for d in range(dimension):
        embedded[:, d] = signal[d * delay:d * delay + n_points]

    n_valid = n_points - time_horizon
    ftle = np.full(n_points, np.nan)
    stretching_dirs = np.full((n_points, dimension), np.nan)

    for i in range(n_valid):
        max_stretch = 0.0
        best_dir = np.zeros(dimension)

        for _ in range(n_perturbations):
            direction = np.random.randn(dimension)
            direction /= np.linalg.norm(direction)

            x_pert = embedded[i] + perturbation * direction
            dists = np.linalg.norm(embedded - x_pert, axis=1)
            nearest_idx = np.argmin(dists)

            if nearest_idx + time_horizon >= n_points:
                continue

            x_final = embedded[i + time_horizon]
            x_pert_final = embedded[nearest_idx + time_horizon]
            final_dist = np.linalg.norm(x_final - x_pert_final)

            if final_dist > max_stretch:
                max_stretch = final_dist
                best_dir = direction

        if max_stretch > 0 and perturbation > 0:
            ftle[i] = (1.0 / time_horizon) * np.log(max_stretch / perturbation)
            stretching_dirs[i] = best_dir

    return ftle, stretching_dirs
