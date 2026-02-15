# Validation Suite

Tests that verify primitives produce **correct** results,
not just consistent results.

## What this tests

Every test compares our implementation against EITHER:
1. A published reference library (scipy, nolds, statsmodels, etc.)
2. A signal with a mathematically provable answer

## When to run

- After any Rust port (new implementation must match ground truth)
- After any algorithm change
- Before any release
- When in doubt

## How to run

```bash
cd ~/primitives

# Install reference libraries (one time)
pip install nolds antropy ordpy statsmodels scipy

# Run all validation
python -m pytest validation/ -v

# Run one primitive
python -m pytest validation/test_hurst.py -v

# Run only analytical tests (no external dependencies)
python -m pytest validation/ -v -k "Analytical"
```

## Tolerance

- Reference library comparison: varies per primitive (see each test)
- Analytical known-answer: generous bounds (estimators have variance)
- If a test fails, our code is wrong until proven otherwise

## Adding new validation tests

1. Find a reference library that implements the same algorithm
2. Find a signal with a known analytical answer
3. Write both types of tests
4. Run against both Rust and Python backends
