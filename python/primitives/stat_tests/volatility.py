"""ARCH/GARCH volatility tests."""
import numpy as np


def arch_test(data: np.ndarray, nlags: int = 12) -> dict:
    """
    Engle's ARCH test for heteroscedasticity.

    Tests whether residual variance is time-varying.
    H0: No ARCH effects (homoscedastic).
    Low p-value -> reject -> volatility clustering exists.

    Args:
        data: 1-D signal array (typically returns or residuals)
        nlags: Number of lags for the test

    Returns:
        dict with:
            statistic: float
            pvalue: float
            is_heteroscedastic: bool — True if p < 0.05
    """
    data = np.asarray(data, dtype=float).ravel()
    n = len(data)

    if n < nlags + 10:
        return {'statistic': np.nan, 'pvalue': np.nan, 'is_heteroscedastic': False}

    # Demean
    resid = data - np.mean(data)
    sq_resid = resid ** 2

    # OLS regression: sq_resid_t = c + sum(a_i * sq_resid_{t-i})
    y = sq_resid[nlags:]
    X = np.column_stack([
        sq_resid[nlags - i - 1: n - i - 1] for i in range(nlags)
    ])
    X = np.column_stack([np.ones(len(y)), X])

    try:
        # OLS: beta = (X'X)^{-1} X'y
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_hat = X @ beta
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot < 1e-20:
            return {'statistic': 0.0, 'pvalue': 1.0, 'is_heteroscedastic': False}

        r_squared = 1 - ss_res / ss_tot
        stat = float(len(y) * r_squared)

        # Chi-squared distribution
        from scipy.stats import chi2
        pvalue = float(1 - chi2.cdf(stat, nlags))

        return {
            'statistic': stat,
            'pvalue': pvalue,
            'is_heteroscedastic': pvalue < 0.05,
        }
    except (np.linalg.LinAlgError, ValueError):
        return {'statistic': np.nan, 'pvalue': np.nan, 'is_heteroscedastic': False}
