"""
Test that every Rust primitive matches its Python fallback.

Compares Rust (_rust) output to Python (individual/, dynamical/, embedding/)
output on multiple signal types. Tolerance: 1e-10 for floats.
"""
import numpy as np
import pytest
import importlib

SIGNALS = {
    'random_walk': np.cumsum(np.random.RandomState(42).randn(500)),
    'white_noise': np.random.RandomState(42).randn(500),
    'trending': np.cumsum(np.random.RandomState(42).randn(500)) + np.linspace(0, 10, 500),
    'periodic': np.sin(np.linspace(0, 20 * np.pi, 500)),
    'short': np.random.RandomState(42).randn(50),
}

# (func_name, rust_module, python_module, kwargs)
SCALAR_PAIRS = [
    ("hurst_exponent", "prmtvs._rust", "prmtvs.individual.fractal", {}),
    ("hurst_exponent", "prmtvs._rust", "prmtvs.individual.fractal", {"method": "dfa"}),
    ("permutation_entropy", "prmtvs._rust", "prmtvs.individual.entropy", {}),
    ("sample_entropy", "prmtvs._rust", "prmtvs.individual.entropy", {}),
]

# Functions that return tuples — compare element by element
TUPLE_PAIRS = [
    ("lyapunov_rosenstein", "prmtvs._rust", "prmtvs.dynamical.lyapunov", {}),
    ("lyapunov_kantz", "prmtvs._rust", "prmtvs.dynamical.lyapunov", {}),
]

# Functions that return int
INT_PAIRS = [
    ("optimal_delay", "prmtvs._rust", "prmtvs.embedding.delay", {}),
    ("optimal_dimension", "prmtvs._rust", "prmtvs.embedding.delay", {}),
]


def _get_fn(module_path, func_name):
    mod = importlib.import_module(module_path)
    return getattr(mod, func_name)


@pytest.mark.parametrize("signal_name", SIGNALS.keys())
@pytest.mark.parametrize("func_name,rust_mod,py_mod,kwargs", SCALAR_PAIRS,
                         ids=[f"{p[0]}({p[3]})" if p[3] else p[0] for p in SCALAR_PAIRS])
def test_scalar_parity(func_name, rust_mod, py_mod, kwargs, signal_name):
    """Test scalar-returning primitives match between Rust and Python."""
    rust_fn = _get_fn(rust_mod, func_name)
    py_fn = _get_fn(py_mod, func_name)

    y = SIGNALS[signal_name]
    rs = rust_fn(y, **kwargs)
    py = py_fn(y, **kwargs)

    if np.isnan(rs) and np.isnan(py):
        return  # Both NaN is OK
    assert abs(rs - py) < 1e-10, (
        f"{func_name}({kwargs}) on {signal_name}: rust={rs} py={py} diff={abs(rs - py)}"
    )


@pytest.mark.xfail(reason="Rust lyapunov was ported from old dynamics.py; manifold dynamical/lyapunov.py diverged. Re-port needed.")
@pytest.mark.parametrize("signal_name", ["random_walk", "white_noise", "trending"])
@pytest.mark.parametrize("func_name,rust_mod,py_mod,kwargs", TUPLE_PAIRS,
                         ids=[p[0] for p in TUPLE_PAIRS])
def test_tuple_parity(func_name, rust_mod, py_mod, kwargs, signal_name):
    """Test tuple-returning primitives: compare first element (the exponent)."""
    rust_fn = _get_fn(rust_mod, func_name)
    py_fn = _get_fn(py_mod, func_name)

    y = SIGNALS[signal_name]
    rs_result = rust_fn(y, **kwargs)
    py_result = py_fn(y, **kwargs)

    rs_val = rs_result[0]
    py_val = py_result[0]

    if np.isnan(rs_val) and np.isnan(py_val):
        return
    assert abs(rs_val - py_val) < 1e-10, (
        f"{func_name} on {signal_name}: rust={rs_val} py={py_val} diff={abs(rs_val - py_val)}"
    )


@pytest.mark.parametrize("signal_name", ["random_walk", "white_noise", "periodic"])
@pytest.mark.parametrize("func_name,rust_mod,py_mod,kwargs", INT_PAIRS,
                         ids=[p[0] for p in INT_PAIRS])
def test_int_parity(func_name, rust_mod, py_mod, kwargs, signal_name):
    """Test integer-returning primitives match exactly."""
    rust_fn = _get_fn(rust_mod, func_name)
    py_fn = _get_fn(py_mod, func_name)

    y = SIGNALS[signal_name]
    rs = rust_fn(y, **kwargs)
    py = py_fn(y, **kwargs)

    assert rs == py, f"{func_name} on {signal_name}: rust={rs} py={py}"
