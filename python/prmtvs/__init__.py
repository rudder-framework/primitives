"""
prmtvs — Rust-accelerated signal analysis.

Standalone compute package. Both Prime and Manifold depend on this.
No circular dependencies.

Usage:
    from prmtvs import hurst_exponent, permutation_entropy
    from prmtvs import optimal_delay, lyapunov_rosenstein

    # Or import by category:
    from prmtvs.individual.statistics import mean, std
    from prmtvs.dynamical.lyapunov import lyapunov_spectrum
    from prmtvs.stat_tests.hypothesis import t_test
"""
import os

_USE_RUST = os.environ.get("PRMTVS_USE_RUST", "1") != "0"

# Try Rust imports
if _USE_RUST:
    try:
        from prmtvs._rust import (
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
    from prmtvs.complexity import (
        hurst_exponent,
        permutation_entropy,
        sample_entropy,
    )
    from prmtvs.dynamics import (
        lyapunov_rosenstein,
        lyapunov_kantz,
        ftle_local_linearization,
        ftle_direct_perturbation,
    )
    from prmtvs.embedding.delay import (
        optimal_delay,
        time_delay_embedding,
        optimal_dimension,
    )
    BACKEND = "python"

# lyapunov_exponent: always Python (Rust not yet ported to match fixed defaults)
from prmtvs.individual.dynamics import lyapunov_exponent  # noqa: E402

# Subpackages (lazy — imported on access, not at startup)
from prmtvs import individual  # noqa: F401
from prmtvs import pairwise  # noqa: F401
from prmtvs import dynamical  # noqa: F401
from prmtvs import matrix  # noqa: F401
from prmtvs import embedding  # noqa: F401
from prmtvs import information  # noqa: F401
from prmtvs import network  # noqa: F401
from prmtvs import topology  # noqa: F401
from prmtvs import stat_tests  # noqa: F401

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
