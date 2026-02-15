"""
Primitives — Rust-accelerated signal analysis.

Standalone compute package. Both Prime and Manifold depend on this.
No circular dependencies.

Usage:
    from primitives import hurst_exponent, permutation_entropy
    from primitives import optimal_delay, lyapunov_rosenstein

    # Or import by category:
    from primitives.individual.statistics import mean, std
    from primitives.dynamical.lyapunov import lyapunov_spectrum
    from primitives.stat_tests.hypothesis import t_test
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
    from primitives.embedding.delay import (
        optimal_delay,
        time_delay_embedding,
        optimal_dimension,
    )
    BACKEND = "python"

# Subpackages (lazy — imported on access, not at startup)
from primitives import individual  # noqa: F401
from primitives import pairwise  # noqa: F401
from primitives import dynamical  # noqa: F401
from primitives import matrix  # noqa: F401
from primitives import embedding  # noqa: F401
from primitives import information  # noqa: F401
from primitives import network  # noqa: F401
from primitives import topology  # noqa: F401
from primitives import stat_tests  # noqa: F401

__all__ = [
    # Tier 1 Rust-accelerated (top-level convenience exports)
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
    # Subpackages
    "individual",
    "pairwise",
    "dynamical",
    "matrix",
    "embedding",
    "information",
    "network",
    "topology",
    "stat_tests",
]
