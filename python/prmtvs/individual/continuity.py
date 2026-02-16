"""Continuity and data-quality features."""
import numpy as np


def continuity_features(data: np.ndarray) -> dict:
    """
    Compute data-quality / continuity features for a 1-D signal.

    These features characterise the "shape" of a signal's value
    distribution rather than its temporal dynamics: how many unique
    values exist, whether they are integer-valued, and how sparse
    the non-zero content is.

    Args:
        data: 1-D numeric array

    Returns:
        dict with:
            unique_ratio: float — fraction of values that are unique
            is_integer: bool — True if every finite value is integer
            sparsity: float — fraction of values that are exactly zero
    """
    data = np.asarray(data, dtype=float).ravel()
    n = len(data)

    if n == 0:
        return {'unique_ratio': np.nan, 'is_integer': False, 'sparsity': np.nan}

    # unique_ratio: |unique values| / n
    unique_ratio = float(len(np.unique(data)) / n)

    # is_integer: every finite value is an integer
    finite = data[np.isfinite(data)]
    is_integer = bool(len(finite) > 0 and np.all(finite == np.floor(finite)))

    # sparsity: fraction of values that are exactly zero
    sparsity = float(np.sum(data == 0.0) / n)

    return {
        'unique_ratio': unique_ratio,
        'is_integer': is_integer,
        'sparsity': sparsity,
    }
