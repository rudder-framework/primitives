# pmtvs

281 pure math functions for signal analysis and dynamical systems. Rust-accelerated. numpy in, scalars out.

```bash
pip install pmtvs
```

```python
from pmtvs import lyapunov_exponent, hurst_exponent, permutation_entropy, BACKEND

le = lyapunov_exponent(signal)  # chaos detection
h = hurst_exponent(signal)      # long-range dependence  
pe = permutation_entropy(signal) # complexity
print(BACKEND)                   # "rust" or "python"
```

**What's inside:** Lyapunov exponents, Hurst, FTLE, sample/permutation/transfer entropy, RQA, phase space embedding, DFA, Granger causality, mutual information, persistent homology, stationarity tests, spectral analysis, wavelet decomposition, DTW, KL/JS divergence, and 250+ more. One import, no config objects, no DataFrames.

Part of the [Rudder Framework](https://github.com/rudder-framework).
