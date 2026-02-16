# pmtvs

281 pure math functions for signal analysis and dynamical systems.
Rust-accelerated. numpy in, scalars out.

## Install

```
pip install pmtvs
```

## Usage

```python
from pmtvs import lyapunov_exponent, hurst_exponent, permutation_entropy, BACKEND
import numpy as np

signal = np.cumsum(np.random.randn(1000))

le = lyapunov_exponent(signal)      # chaos detection
h = hurst_exponent(signal)          # long-range dependence
pe = permutation_entropy(signal)    # complexity
print(BACKEND)                      # "rust" or "python"
```

## What's inside

Lyapunov exponents, Hurst, FTLE, sample/permutation/transfer entropy,
RQA, phase space embedding, DFA, Granger causality, mutual information,
persistent homology, stationarity tests, spectral analysis, wavelet
decomposition, DTW, KL/JS divergence, and 250+ more.

One import. No config objects. No DataFrames.

## Rust backend

10 performance-critical functions are accelerated via Rust/PyO3.
Falls back to pure Python automatically if the Rust extension
isn't available. Set `PMTVS_USE_RUST=0` to force Python.

## Citing

If you use pmtvs in your research, please cite:

```bibtex
@software{pmtvs,
  author = {Rudder, Jason},
  title = {pmtvs: Signal Analysis Primitives},
  year = {2026},
  url = {https://github.com/rudder-framework/primitives}
}
```

## License

MIT

Part of the [Rudder Framework](https://github.com/rudder-framework).
