# pmtvs

281 pure math functions for signal analysis and dynamical systems. Rust-accelerated. numpy in, scalars out.

```bash
pip install pmtvs
```

```python
from pmtvs import lyapunov_exponent, hurst_exponent, ftle_local_linearization, permutation_entropy

le = lyapunov_exponent(signal)
```

Part of the [Rudder Framework](https://github.com/rudder-framework).
