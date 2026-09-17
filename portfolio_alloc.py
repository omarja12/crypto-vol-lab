"""
Portfolio allocators for the vol-premium cross-section.

Every allocator returns weights that sum to 1 (or to the gross budget under a
long-short setting). Which one to trust is an empirical question answered by
walk-forward comparison, not by theory - see portfolio_backtest.py. The honest
default expectation at this breadth is that equal-weight is hard to beat.

EXPECTED RETURNS
----------------
For the vol cross-section the expected return per bucket is the variance risk
premium: model-free implied variance minus your forecast of realised variance.
This is an actual observable estimate, not a historical mean, which is the one
respect in which mean-variance is better founded here than in equity allocation.
It does not make the estimate correct - it makes it honest about what it is.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def equal_weight(n):
    """The benchmark every optimiser has to beat to justify its complexity."""
    return np.ones(n) / n


def inverse_variance(cov):
    """Weights proportional to 1/variance. Ignores correlations by design."""
    v = np.diag(cov)
    w = 1.0 / v
    return w / w.sum()


def minimum_variance(cov, long_only=True, w_max=None):
    """
    Minimise w' cov w subject to the weights summing to 1.

    Uses no expected returns at all, so none of the estimation error in the
    means propagates into it. That is exactly why it is the right robustness
    check against mean-variance: if the two disagree sharply, the disagreement
    is being driven by the return estimates, and you should ask how much you
    trust them.
    """
    n = cov.shape[0]
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0 if long_only else None, w_max) for _ in range(n)]

    res = minimize(lambda w: float(w @ cov @ w), equal_weight(n),
                   method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-12})
    return res.x


def mean_variance(mu, cov, risk_aversion=1.0, long_only=True, w_max=None):
    """
    Maximise  mu' w - (risk_aversion/2) w' cov w  subject to weights summing to 1.

    Higher risk_aversion tilts toward minimum-variance; lower chases the means
    harder and is more exposed to their estimation error. At small breadth,
    prefer a higher value than feels natural.
    """
    n = cov.shape[0]
    mu = np.asarray(mu, float)

    def neg_util(w):
        return -(mu @ w - 0.5 * risk_aversion * (w @ cov @ w))

    def neg_grad(w):
        return -(mu - risk_aversion * cov @ w)

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0 if long_only else None, w_max) for _ in range(n)]

    res = minimize(neg_util, equal_weight(n), jac=neg_grad,
                   method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-12})
    return res.x


def risk_parity(cov, tol=1e-8, max_iter=50000):
    """
    Equal risk contribution: each asset contributes the same share of total
    portfolio variance. Needs no expected returns and no matrix inversion, so
    it is robust to exactly the errors that break mean-variance.

    Solved by the multiplicative fixed-point iteration of Spinu / Griveau-Billion
    et al., which converges for any positive-definite covariance. Note the
    method assumes a long-only solution exists; when assets are strong mutual
    hedges (large negative correlations) equal risk contribution can be
    ill-posed, and the returned weights should be sanity-checked with
    `risk_contributions` below.
    """
    n = cov.shape[0]
    w = equal_weight(n)
    b = np.ones(n) / n                          # target risk budget

    for _ in range(max_iter):
        cov_w = cov @ w
        # Multiplicative update: w_i <- b_i / (cov_w)_i, then renormalise in the
        # covariance metric. Monotone and convergent for PD cov.
        w_new = b / np.maximum(cov_w, 1e-15)
        w_new = w_new / np.sqrt(w_new @ cov @ w_new)
        w_new = w_new / w_new.sum()
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new

    return w


def risk_contributions(w, cov):
    """Fractional risk contribution per asset. Should be ~equal after risk_parity."""
    w = np.asarray(w, float)
    cov_w = cov @ w
    return w * cov_w / float(w @ cov_w)


def mean_cvar(returns_scenarios, mu=None, alpha=0.95, target_return=None,
              long_only=True, w_max=None):
    """
    Minimise Conditional Value at Risk (expected loss beyond the alpha quantile)
    via the Rockafellar-Uryasev linear formulation.

    For skewed payoffs - which delta-hedged short vol positions are - CVaR is a
    more honest objective than variance, because it prices the tail directly
    rather than assuming a symmetric spread. This is the allocator to prefer
    when the return distribution is visibly asymmetric.

    `returns_scenarios` is (n_scenarios x n_assets): empirical or simulated
    joint returns. More scenarios give a better tail estimate; the tail is the
    whole point, so do not starve it.
    """
    R = np.asarray(returns_scenarios, float)
    S, n = R.shape

    # Variables: [w (n), VaR (1), z (S)] with z_s >= -R_s.w - VaR, z_s >= 0.
    # Objective: VaR + 1/((1-alpha)S) sum z_s.
    from scipy.optimize import linprog

    c = np.concatenate([np.zeros(n), [1.0], np.ones(S) / ((1 - alpha) * S)])

    # -R.w - VaR - z <= 0
    A_ub = np.hstack([-R, -np.ones((S, 1)), -np.eye(S)])
    b_ub = np.zeros(S)

    A_eq = np.concatenate([np.ones(n), [0.0], np.zeros(S)])[None, :]
    b_eq = [1.0]

    if target_return is not None and mu is not None:
        mu = np.asarray(mu, float)
        row = np.concatenate([-mu, [0.0], np.zeros(S)])
        A_ub = np.vstack([A_ub, row])
        b_ub = np.append(b_ub, -target_return)

    w_bounds = [(0.0 if long_only else None, w_max) for _ in range(n)]
    bounds = w_bounds + [(None, None)] + [(0, None)] * S

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"CVaR LP failed: {res.message}")
    return res.x[:n]


def portfolio_stats(w, mu, cov):
    """Expected return, volatility, and Sharpe for a weight vector."""
    w = np.asarray(w, float)
    ret = float(mu @ w) if mu is not None else np.nan
    vol = float(np.sqrt(w @ cov @ w))
    return {"return": ret, "vol": vol,
            "sharpe": ret / vol if vol > 0 and np.isfinite(ret) else np.nan}
