"""
stratlab/kite_stats.py — the Augmented Dickey-Fuller test used by the Varsity pair-trade (PTM2) pair filter.

Same algorithm and defaults as statsmodels.tsa.stattools.adfuller(x) (regression="c", autolag="AIC",
maxlag = ceil(12 (nobs/100)^(1/4))), with MacKinnon (1994, 2010) approximate p-values for one variable
(statsmodels.tsa.adfvalues.mackinnonp(N=1, regression="c")). statsmodels is not a dependency of the app, so the
few lines are reproduced here; tests_kite.py checks them against values computed with statsmodels 0.14.
"""
from __future__ import annotations

import math

import numpy as np

# MacKinnon p-value surface, regression "c", N = 1 (statsmodels.tsa.adfvalues)
_TAU_MAX, _TAU_MIN, _TAU_STAR = 2.74, -18.83, -1.61
_SMALLP = (2.1659, 1.4412, 0.038269)
_LARGEP = (1.7339, 9.3202 * 1e-1, -1.2745 * 1e-1, -1.0368 * 1e-2)


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def mackinnonp(tau: float) -> float:
    if tau > _TAU_MAX:
        return 1.0
    if tau < _TAU_MIN:
        return 0.0
    c = _SMALLP if tau <= _TAU_STAR else _LARGEP
    return _ncdf(sum(k * tau ** i for i, k in enumerate(c)))


def _ols(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ssr = float(resid @ resid)
    return beta, ssr


def _lagmat(dx, maxlag):
    """Rows t = maxlag..len-1: [dx_t, dx_{t-1}, ..., dx_{t-maxlag}] (statsmodels lagmat trim='both', original='in')."""
    n = len(dx)
    return np.column_stack([dx[maxlag - k: n - k] for k in range(maxlag + 1)])


def adf(x, maxlag: int | None = None) -> tuple[float, float, int]:
    """(ADF statistic, MacKinnon p-value, lags used) with a constant and AIC lag selection (statsmodels default)."""
    x = np.asarray(x, float)
    nobs0 = len(x)
    if maxlag is None:
        maxlag = int(math.ceil(12.0 * (nobs0 / 100.0) ** 0.25))
        maxlag = min(nobs0 // 2 - 1 - 1, maxlag)
    dx = np.diff(x)
    xd = _lagmat(dx, maxlag)
    nobs = xd.shape[0]
    xd[:, 0] = x[-nobs - 1:-1]                      # lagged level replaces the current difference
    y = dx[-nobs:]
    full = np.column_stack([np.ones(nobs), xd])     # [const, x_{t-1}, dx_{t-1}, ..., dx_{t-maxlag}]
    best, best_lag = None, 0
    for k in range(2, 2 + maxlag + 1):              # columns used: const + level + (k-2) lagged differences
        _b, ssr = _ols(y, full[:, :k])
        llf = -nobs / 2.0 * (math.log(2 * math.pi) + math.log(ssr / nobs) + 1.0)
        aic = -2.0 * llf + 2.0 * k
        if best is None or aic < best:
            best, best_lag = aic, k - 2
    xd = _lagmat(dx, best_lag) if best_lag > 0 else dx[:, None].copy()
    nobs = xd.shape[0]
    xd[:, 0] = x[-nobs - 1:-1]
    y = dx[-nobs:]
    X = np.column_stack([xd[:, :best_lag + 1], np.ones(nobs)])
    beta, ssr = _ols(y, X)
    dof = nobs - X.shape[1]
    s2 = ssr / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    t = float(beta[0] / math.sqrt(cov[0, 0]))
    return t, mackinnonp(t), best_lag
