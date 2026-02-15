"""
Shared test signals with known properties.

Every signal here has a mathematically provable characteristic
that ground truth libraries agree on. No ambiguous cases.
"""
import numpy as np
import pytest


@pytest.fixture
def white_noise():
    """White noise: Hurst ~ 0.5, high perm_entropy, stationary."""
    rng = np.random.RandomState(42)
    return rng.randn(10000)


@pytest.fixture
def random_walk():
    """Random walk: Hurst ~ 1.0, non-stationary, ADF fails to reject."""
    rng = np.random.RandomState(42)
    return np.cumsum(rng.randn(10000))


@pytest.fixture
def sine_wave():
    """Pure sine: low perm_entropy, dominant frequency known exactly."""
    t = np.linspace(0, 10, 10000)
    return np.sin(2 * np.pi * 5.0 * t)  # 5 Hz


@pytest.fixture
def constant():
    """Constant signal: zero variance, kurtosis undefined, entropy = 0."""
    return np.ones(1000) * 3.14


@pytest.fixture
def linear_trend():
    """Linear ramp: Hurst ~ 1.0, non-stationary, zero spectral entropy."""
    return np.linspace(0, 100, 5000)


@pytest.fixture
def lorenz_x():
    """Lorenz attractor x-component. Known max Lyapunov ~ 0.91.

    Integrated with standard parameters: sigma=10, rho=28, beta=8/3.
    """
    from scipy.integrate import solve_ivp

    def lorenz(t, state):
        x, y, z = state
        return [10 * (y - x), x * (28 - z) - y, x * y - (8/3) * z]

    sol = solve_ivp(lorenz, [0, 50], [1.0, 1.0, 1.0],
                    t_eval=np.linspace(0, 50, 50000), rtol=1e-10)
    return sol.y[0]


@pytest.fixture
def two_frequencies():
    """Two known frequencies: 3 Hz and 7 Hz. For spectral tests."""
    t = np.linspace(0, 10, 10000)
    return np.sin(2 * np.pi * 3.0 * t) + 0.5 * np.sin(2 * np.pi * 7.0 * t)


@pytest.fixture
def correlated_pair():
    """Two signals with known correlation ~ 0.8."""
    rng = np.random.RandomState(42)
    x = rng.randn(1000)
    y = 0.8 * x + 0.6 * rng.randn(1000)  # r ~ 0.8
    return x, y


@pytest.fixture
def uncorrelated_pair():
    """Two independent signals. Correlation ~ 0, Granger = no."""
    rng = np.random.RandomState(42)
    x = rng.randn(1000)
    rng2 = np.random.RandomState(99)
    y = rng2.randn(1000)
    return x, y


@pytest.fixture
def causal_pair():
    """X causes Y with 1-step lag. Granger should detect."""
    rng = np.random.RandomState(42)
    n = 2000
    x = rng.randn(n)
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.7 * x[i-1] + 0.3 * rng.randn()
    return x, y
