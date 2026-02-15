"""
Primitives — Rust-accelerated signal analysis.

Standalone compute package. Both Prime and Manifold depend on this.
No circular dependencies.

Usage:
    from primitives import hurst_exponent, permutation_entropy
    from primitives import optimal_delay, lyapunov_rosenstein
"""
import os

_USE_RUST = os.environ.get("PRIMITIVES_USE_RUST", "1") != "0"

# Try Rust imports
if _USE_RUST:
    try:
        from primitives._rust import (
            hurst_exponent,
            permutation_entropy,
            sample_entropy,
            lyapunov_rosenstein,
            lyapunov_kantz,
            ftle_local_linearization,
            ftle_direct_perturbation,
            optimal_delay,
            time_delay_embedding,
            optimal_dimension,
        )
        BACKEND = "rust"
    except ImportError:
        _USE_RUST = False

# Python fallbacks
if not _USE_RUST:
    from primitives.complexity import (
        hurst_exponent,
        permutation_entropy,
        sample_entropy,
    )
    from primitives.dynamics import (
        lyapunov_rosenstein,
        lyapunov_kantz,
        ftle_local_linearization,
        ftle_direct_perturbation,
    )
    from primitives.embedding import (
        optimal_delay,
        time_delay_embedding,
        optimal_dimension,
    )
    BACKEND = "python"

__all__ = [
    "hurst_exponent",
    "permutation_entropy",
    "sample_entropy",
    "lyapunov_rosenstein",
    "lyapunov_kantz",
    "ftle_local_linearization",
    "ftle_direct_perturbation",
    "optimal_delay",
    "time_delay_embedding",
    "optimal_dimension",
    "BACKEND",
]
