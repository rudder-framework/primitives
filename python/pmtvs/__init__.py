"""
pmtvs — Rust-accelerated signal analysis.

Standalone compute package. Both Prime and Manifold depend on this.
No circular dependencies.

Usage:
    from pmtvs import hurst_exponent, permutation_entropy
    from pmtvs import optimal_delay, lyapunov_rosenstein

    # Or import by category:
    from pmtvs.individual.statistics import mean, std
    from pmtvs.dynamical.lyapunov import lyapunov_spectrum
    from pmtvs.stat_tests.hypothesis import t_test
"""
__version__ = "0.2.0"

import os

_USE_RUST = os.environ.get("PMTVS_USE_RUST", "1") != "0"

# Try Rust imports
if _USE_RUST:
    try:
        from pmtvs._rust import (
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
    from pmtvs.complexity import (
        hurst_exponent,
        permutation_entropy,
        sample_entropy,
    )
    from pmtvs.dynamics import (
        lyapunov_rosenstein,
        lyapunov_kantz,
        ftle_local_linearization,
        ftle_direct_perturbation,
    )
    from pmtvs.embedding.delay import (
        optimal_delay,
        time_delay_embedding,
        optimal_dimension,
    )
    BACKEND = "python"

# lyapunov_exponent: always Python (Rust not yet ported to match fixed defaults)
from pmtvs.individual.dynamics import lyapunov_exponent  # noqa: E402

# Subpackages (lazy — imported on access, not at startup)
from pmtvs import individual  # noqa: F401
from pmtvs import pairwise  # noqa: F401
from pmtvs import dynamical  # noqa: F401
from pmtvs import matrix  # noqa: F401
from pmtvs import embedding  # noqa: F401
from pmtvs import information  # noqa: F401
from pmtvs import network  # noqa: F401
from pmtvs import topology  # noqa: F401
from pmtvs import stat_tests  # noqa: F401

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
    "lyapunov_exponent",
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
