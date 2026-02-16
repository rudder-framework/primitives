# pmtvs

Rust-accelerated primitives for signal analysis and dynamical systems.

281 pure mathematical functions. Arrays in, numbers out.

## Install

```bash
pip install pmtvs
```

## Usage

```python
from pmtvs import hurst_exponent, permutation_entropy, BACKEND

import numpy as np
y = np.cumsum(np.random.randn(1000))

print(hurst_exponent(y))       # ~0.98 (persistent random walk)
print(permutation_entropy(y))  # ~0.95 (high complexity)
print(BACKEND)                 # "rust" or "python"
```

## License

MIT
