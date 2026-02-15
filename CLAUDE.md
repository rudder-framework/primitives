# CLAUDE.md — Primitives

Primitives is a Rust+Python library of 281 pure mathematical functions
for signal analysis and dynamical systems. Arrays in, numbers out. Nothing else.

**Primitives is a calculator.** You hand it numbers, it hands numbers back.
It has no idea where the numbers came from. It never sees a file, a DataFrame,
a signal_id, a cohort, or a config object.

---

## Repositories

GitHub org: `rudder-framework`

```
rudder-framework/primitives    → THIS REPO — pure math (Rust + Python)
rudder-framework/manifold      → compute engine (pipeline, parquet I/O)
rudder-framework/prime         → interpreter (classification, SQL, explorer)
```

Primitives is the **leaf dependency**. It depends on nothing.
Both Prime and Manifold depend on it. Neither owns it.

```
primitives   ← THIS REPO (no dependencies on Prime or Manifold)
   ↑    ↑
   |    |
Prime  Manifold
```

### How each consumer uses primitives

```python
# Prime calls primitives ONCE per signal (typology):
primitives.hurst_exponent(entire_signal)     → "what kind of signal is this?"

# Manifold calls primitives PER WINDOW per signal (thousands of times):
primitives.hurst_exponent(window_of_signal)  → "how is this signal changing?"
```

Same function. Different scale. Same result.

---

## How to Build

### Prerequisites

```bash
# Install Rust (one time)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
rustc --version
```

### Dev build (installs in current venv)

```bash
cd ~/primitives
pip install maturin
maturin develop --release
```

This compiles the Rust code and makes `primitives` importable in Python immediately.

### Verify

```bash
python -c "from primitives import hurst_exponent; print('OK')"
python -c "from primitives import BACKEND; print(f'Backend: {BACKEND}')"
```

### Build wheels for distribution

```bash
maturin build --release
# Output: target/wheels/primitives-*.whl
```

---

## Project Structure

```
primitives/
├── Cargo.toml                    # Rust config
├── pyproject.toml                # maturin build config
├── LICENSE.md                    # MIT license
│
├── src/                          # Rust source (fast path — Tier 1 only)
│   ├── lib.rs                    # PyO3 module entry point
│   ├── complexity/
│   │   ├── mod.rs
│   │   ├── hurst.rs              # hurst_exponent (R/S + DFA)
│   │   └── entropy.rs            # permutation_entropy, sample_entropy
│   ├── dynamics/
│   │   ├── mod.rs
│   │   ├── lyapunov.rs           # lyapunov_rosenstein, lyapunov_kantz
│   │   └── ftle.rs               # ftle_local_linearization, ftle_direct_perturbation
│   └── embedding/
│       ├── mod.rs
│       └── delay.rs              # optimal_delay, time_delay_embedding, optimal_dimension
│
├── python/                       # Python implementations (all 281 primitives)
│   └── primitives/
│       ├── __init__.py           # PRIMITIVES_USE_RUST toggle, re-exports
│       ├── _config.py            # USE_RUST env var reader
│       ├── config.py             # Centralized default parameters
│       ├── complexity.py         # Backward compat (original flat file)
│       ├── dynamics.py           # Backward compat (original flat file)
│       ├── individual/           # Single-signal (20 modules): statistics, spectral, entropy, fractal, acf, continuity, etc.
│       ├── pairwise/             # Two-signal: correlation, causality, distance, etc.
│       ├── dynamical/            # Chaos: lyapunov, dimension, rqa, ftle, saddle
│       ├── matrix/               # All-signal: covariance, decomposition, dmd, graph
│       ├── embedding/            # Phase space: delay embedding, optimal params
│       ├── information/          # Info theory: entropy, divergence, mutual info, transfer
│       ├── network/              # Graph metrics: centrality, community, paths, structure
│       ├── topology/             # Persistent homology: persistence, distance
│       └── stat_tests/           # Statistical tests: hypothesis, bootstrap, stationarity, volatility
│
├── tests/                        # Unit tests (58 tests)
│   ├── test_hurst.py
│   ├── test_permutation_entropy.py
│   ├── test_sample_entropy.py
│   ├── test_lyapunov.py
│   ├── test_optimal_delay.py
│   └── test_rust_python_parity.py
│
└── validation/                   # Ground-truth validation suite (90 tests)
    ├── conftest.py               # Shared fixtures (white noise, random walk, Lorenz, etc.)
    ├── test_hurst.py             # vs nolds
    ├── test_permutation_entropy.py  # vs ordpy
    ├── test_sample_entropy.py    # vs nolds
    ├── test_lyapunov.py          # vs nolds
    ├── test_optimal_delay.py     # analytical
    ├── test_eigendecomposition.py # vs numpy
    ├── test_kurtosis.py          # vs scipy
    ├── test_spectral.py          # vs scipy
    ├── test_spectral_profile.py  # vs scipy
    ├── test_acf.py               # vs statsmodels
    ├── test_acf_half_life.py     # analytical + AR(1)
    ├── test_adf.py               # vs statsmodels
    ├── test_arch.py              # vs statsmodels
    ├── test_granger.py           # analytical
    ├── test_turning_point_ratio.py  # analytical (E[TPR]=2/3)
    ├── test_determinism.py       # analytical + pipeline match
    └── test_continuity.py        # analytical
```

### Rust-accelerated primitives (Tier 1)

| Function | Rust | Python | Parity |
|----------|------|--------|--------|
| hurst_exponent (R/S) | ✅ | ✅ | ✅ < 1e-14 |
| hurst_exponent (DFA) | ✅ | ✅ | ✅ < 1e-13 |
| permutation_entropy | ✅ | ✅ | ✅ < 1e-14 |
| sample_entropy | ✅ | ✅ | ✅ < 1e-14 |
| lyapunov_rosenstein | ✅ | ✅ | ⚠️ mismatch — Rust needs re-port to match manifold Python |
| lyapunov_kantz | ✅ | ✅ | ⚠️ mismatch — Rust needs re-port to match manifold Python |
| ftle_local_linearization | ✅ | ✅ | ✅ |
| ftle_direct_perturbation | ✅ | ✅ | ✅ |
| optimal_delay | ✅ | ✅ | ✅ exact |
| time_delay_embedding | ✅ | ✅ | ✅ |
| optimal_dimension | ✅ | ✅ | ✅ exact |

All other ~270 primitives are Python-only.

---

## Import Patterns

```python
# Top-level (Rust-accelerated when available)
from primitives import hurst_exponent, BACKEND

# Category imports (always Python)
from primitives.individual.statistics import kurtosis
from primitives.individual.spectral import spectral_entropy, spectral_profile
from primitives.individual.acf import acf_half_life
from primitives.individual.temporal import turning_point_ratio
from primitives.individual.continuity import continuity_features
from primitives.pairwise.causality import granger_causality
from primitives.dynamical.rqa import recurrence_rate, determinism_from_signal
from primitives.stat_tests.volatility import arch_test
from primitives.matrix.decomposition import eigendecomposition
from primitives.information.entropy import shannon_entropy
from primitives.network.centrality import centrality_betweenness
from primitives.topology.persistence import betti_numbers
from primitives.stat_tests.stationarity_tests import adf_test

# Backward compat (still works)
from primitives.complexity import hurst_exponent
from primitives.dynamics import lyapunov_rosenstein
```

---

## Cargo.toml

```toml
[package]
name = "primitives"
version = "0.3.0"
edition = "2021"

[lib]
name = "primitives"
crate-type = ["cdylib"]

[dependencies]
pyo3 = "0.27"
numpy = "0.27"

# Ready for Tier 2+ Rust ports (uncomment as needed):
# ndarray = "0.16"
# ndarray-linalg = "0.16"
# ndarray-stats = "0.6"
# rustfft = "6.2"
# rayon = "1.10"
# num-complex = "0.4"
# statrs = "0.17"
# kiddo = "4.2"
```

---

## The One Rule

Every function in this repo follows the same contract:

```python
def some_primitive(data: np.ndarray, **params) -> float | np.ndarray | dict
```

**Input:** numpy array(s) and scalar parameters.
**Output:** float, numpy array, or dict of floats/arrays.
**Never:** file paths, DataFrames, config objects, signal_id, window_id, cohort.

If a function needs to know about files or pipelines, it does not belong here.

---

## Rust / Python Toggle

Environment variable: `PRIMITIVES_USE_RUST`

```bash
# Rust backend (default — fast)
python -c "from primitives import BACKEND; print(BACKEND)"
# → "rust"

# Python fallback (debugging, or Rust not installed)
PRIMITIVES_USE_RUST=0 python -c "from primitives import BACKEND; print(BACKEND)"
# → "python"
```

The toggle is read ONCE at import time. Both Prime and Manifold get the same
backend from the same env var.

Individual Python modules also check `_config.USE_RUST` for per-function
Rust acceleration when calling directly (e.g. `from primitives.individual.fractal import hurst_exponent`).

---

## Rules

### 1. PURE FUNCTIONS ONLY
numpy in, numbers out. No DataFrames. No pandas. No polars. No file I/O.
No signal_id. No cohort. No config objects. No window_id. Stateless.

### 2. RUST MUST MATCH PYTHON
Rust output must match Python within 1e-10. Differences at 1e-14 are acceptable
(floating point path differences). Test on REAL data (FD004 signals), not just
random arrays.

### 3. INSUFFICIENT DATA → NaN
If there aren't enough samples for the math to work, return NaN. Never skip.
Never crash. Never return a made-up number.

### 4. ONE FUNCTION PER CONCERN
`hurst_exponent()` computes Hurst. Period. Combining Hurst + entropy + spectral
into one function is an engine's job (Manifold), not a primitive's.

### 5. NO ALGORITHM CHANGES DURING RUST PORT
Port the Python algorithm line-by-line. Do not optimize. Do not "improve."
Match the output first. Optimize in a separate PR after verification.

### 6. BRIDGE PATTERN
Every Rust primitive has a Python fallback. The toggle in `__init__.py`
decides which runs. Users without Rust installed get Python automatically.

### 7. ONE PRIMITIVE PER RUST FILE
`src/complexity/hurst.rs` contains `hurst_exponent`. That's it.
Keep it auditable.

### 8. SHOW YOUR WORK
Before modifying any file:
1. Show the existing Python implementation you're porting
2. Show the Rust equivalent
3. Show the parity test result on real data
4. Get approval before changing function signatures

---

## Testing

### Run all tests

```bash
cd ~/primitives
python -m pytest tests/ -v          # 58 unit tests
python -m pytest validation/ -v     # 90 ground-truth validation tests
```

### Parity test suite

`tests/test_rust_python_parity.py` tests every Rust primitive against its Python
fallback on 5 signal types (random walk, white noise, trending, periodic, short).

### Validation suite

`validation/` tests primitives against reference libraries (nolds, ordpy, scipy,
statsmodels) and known analytical solutions. Requires extra deps:

```bash
pip install nolds antropy ordpy statsmodels
```

### Manual parity check

```python
import numpy as np
from primitives._rust import hurst_exponent as rs_hurst
from primitives.individual.fractal import hurst_exponent as py_hurst

y = np.cumsum(np.random.RandomState(42).randn(500))
diff = abs(rs_hurst(y) - py_hurst(y))
assert diff < 1e-10
```

---

## Known Gaps

| Gap | Status |
|-----|--------|
| Lyapunov Rust/Python parity | Rust was ported from old dynamics.py; manifold version diverged. Re-port needed. |
| CI/CD wheel builds | Not set up. GitHub Actions needed for cross-platform wheels. |
| adf_test Rust port | Python works (in tests/stationarity_tests.py). Tier 1 Rust port pending. |

---

## What Does NOT Belong Here

| Thing | Where it goes | Why |
|-------|---------------|-----|
| Reading parquet files | Manifold | I/O is not math |
| Window sliding | Manifold | Orchestration is not math |
| Classification logic | Prime | Decisions are not math |
| Signal typology decisions | Prime | "hurst 0.87 = DRIFTING" is classification |
| SQL queries | Prime | Queries are not math |
| Config/manifest handling | Prime | Configuration is not math |
| DataFrame operations | Manifold | DataFrames are not arrays |
| Parallel cohort processing | Manifold | Parallelism is orchestration |
| Visualization / HTML | Prime | Rendering is not math |
