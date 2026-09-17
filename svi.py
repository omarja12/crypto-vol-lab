"""
Stage 4: raw SVI slice calibration.
Stage 5: butterfly and calendar arbitrage diagnostics.

PARAMETERISATION
----------------
Raw SVI, on TOTAL implied variance w(k) = sigma^2 * T:

    w(k) = a + b * ( rho*(k - m) + sqrt((k - m)^2 + s^2) )

    a    vertical level
    b    overall angle between the asymptotes
    rho  counter-clockwise rotation, i.e. skew
    m    horizontal translation
    s    smoothness at the vertex

NOTE: `s` here is the SVI shape parameter. It is NOT an implied volatility.
The literature calls it sigma, which collides with the volatility symbol and
causes real confusion, so it is named `s` throughout this module.

WING CONSTRAINT
---------------
Lee's moment formula bounds total variance growth: w(k)/|k| -> beta <= 2.
For raw SVI, as k -> +inf, w(k) ~ a + b(1+rho)(k-m), so the right wing slope is
b(1+rho) and the left is b(1-rho). Hence

    b * (1 + |rho|) <= 2

Beware when comparing against other implementations: a bound of 4/T appears in
sources that parameterise implied VARIANCE rather than TOTAL variance. This
module works in total variance throughout, so the bound is 2.

CALIBRATION
-----------
The five-parameter problem is non-convex and will find local minima. Use the
quasi-explicit reduction (Zeliade Systems): substitute y = (k - m)/s, giving

    w = a_tilde + d*y + c*sqrt(y^2 + 1)

with c = b*s and d = rho*b*s. For FIXED (m, s) this is linear in
(a_tilde, d, c), so the inner problem is a small constrained least squares
solved exactly. The search therefore reduces from five dimensions to two.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.optimize import minimize

LEE_WING_BOUND = 2.0
MIN_POINTS = 5


@dataclass
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    s: float

    def as_dict(self):
        return asdict(self)


@dataclass
class SVIFit:
    params: SVIParams | None
    rmse_w: float           # total variance units
    rmse_vol: float         # volatility points, the interpretable one
    n_points: int
    T: float
    converged: bool
    reason: str = ""

    def as_dict(self):
        d = {"rmse_w": self.rmse_w, "rmse_vol": self.rmse_vol,
             "n_points": self.n_points, "T": self.T,
             "converged": self.converged, "reason": self.reason}
        if self.params:
            d.update(self.params.as_dict())
        return d


# --------------------------------------------------------------------------
# The curve and its derivatives
# --------------------------------------------------------------------------

def svi_w(k, p: SVIParams):
    """Total implied variance."""
    k = np.asarray(k, float)
    z = k - p.m
    return p.a + p.b * (p.rho * z + np.sqrt(z * z + p.s * p.s))


def svi_dw(k, p: SVIParams):
    """First derivative in k. Analytic; do not use finite differences here."""
    k = np.asarray(k, float)
    z = k - p.m
    return p.b * (p.rho + z / np.sqrt(z * z + p.s * p.s))


def svi_d2w(k, p: SVIParams):
    """Second derivative in k."""
    k = np.asarray(k, float)
    z = k - p.m
    return p.b * p.s * p.s / np.power(z * z + p.s * p.s, 1.5)


def svi_vol(k, p: SVIParams, T: float):
    """Implied volatility, for plotting and comparison against quotes."""
    w = svi_w(k, p)
    return np.sqrt(np.maximum(w, 0.0) / T)


# --------------------------------------------------------------------------
# Stage 4: calibration
# --------------------------------------------------------------------------

def _inner_solve(y, w, weights, s, w_max):
    """
    Constrained linear least squares in (a_tilde, d, c) for fixed (m, s).

    Constraints, following Zeliade, expressed so that the recovered raw
    parameters automatically satisfy |rho| < 1 and the Lee wing bound:

        0 <= c
        |d| <= c                        <=>  |rho| <= 1
        c + |d| <= LEE_WING_BOUND * s   <=>  b(1 + |rho|) <= LEE_WING_BOUND
        0 <= a_tilde <= max(w)
    """
    basis = np.column_stack([np.ones_like(y), y, np.sqrt(y * y + 1.0)])
    sw = np.sqrt(weights)
    A = basis * sw[:, None]
    target = w * sw

    def objective(x):
        r = A @ x - target
        return float(r @ r)

    def grad(x):
        return 2.0 * A.T @ (A @ x - target)

    cons = [
        {"type": "ineq", "fun": lambda x: x[2]},                                  # c >= 0
        {"type": "ineq", "fun": lambda x: x[2] - abs(x[1])},                      # |d| <= c
        {"type": "ineq", "fun": lambda x: LEE_WING_BOUND * s - x[2] - abs(x[1])}, # wing bound
        {"type": "ineq", "fun": lambda x: x[0]},                                  # a >= 0
        {"type": "ineq", "fun": lambda x: w_max - x[0]},                          # a <= max(w)
    ]

    # Unconstrained least squares as the starting point, projected into the
    # feasible region so SLSQP starts somewhere legal.
    x0, *_ = np.linalg.lstsq(A, target, rcond=None)
    x0[2] = max(x0[2], 1e-8)
    x0[1] = np.clip(x0[1], -x0[2], x0[2])
    x0[0] = np.clip(x0[0], 0.0, w_max)

    res = minimize(objective, x0, jac=grad, constraints=cons,
                   method="SLSQP", options={"maxiter": 200, "ftol": 1e-14})
    return res.x, float(res.fun)


def calibrate_svi(k, iv, T, iv_spread=None, n_grid=18):
    """
    Fit one expiry slice.

    Parameters
    ----------
    k : array
        Log-moneyness, ln(K/F). Must come from the stage 2 forward, not spot.
    iv : array
        Implied volatilities (decimal, not percent).
    T : float
        Year fraction.
    iv_spread : array, optional
        Implied vol bid-ask per strike. Used for weighting, so that the fit
        follows the strikes you could actually trade rather than the wings
        where one tick moves implied vol by several points. Uniform weights
        if omitted, which is a worse fit to the tradeable surface.

    Returns
    -------
    SVIFit
    """
    k = np.asarray(k, float)
    iv = np.asarray(iv, float)

    ok = np.isfinite(k) & np.isfinite(iv) & (iv > 0)
    k, iv = k[ok], iv[ok]
    if k.size < MIN_POINTS:
        return SVIFit(None, np.nan, np.nan, k.size, T, False,
                      f"only {k.size} usable points")

    w = iv * iv * T

    if iv_spread is not None:
        sp = np.asarray(iv_spread, float)[ok]
        # Convert the vol spread into a total-variance spread via dw/dsigma,
        # so the weighting is expressed in the units actually being fitted.
        w_spread = 2.0 * iv * T * np.abs(sp)
        w_spread = np.where(np.isfinite(w_spread) & (w_spread > 0),
                            w_spread, np.nanmedian(w_spread))
        weights = 1.0 / np.maximum(w_spread, 1e-10) ** 2
    else:
        weights = np.ones_like(w)
    weights = weights / weights.sum() * len(weights)

    w_max = float(np.max(w))

    # Outer search over (m, s). Two dimensions, so a coarse grid followed by
    # a local refinement is reliable and cheap.
    m_grid = np.linspace(k.min() - 0.1, k.max() + 0.1, n_grid)
    s_grid = np.exp(np.linspace(np.log(0.005), np.log(2.0), n_grid))

    best = (np.inf, None, None, None)
    for m in m_grid:
        for s in s_grid:
            y = (k - m) / s
            x, sse = _inner_solve(y, w, weights, s, w_max)
            if sse < best[0]:
                best = (sse, m, s, x)

    def outer(v):
        m, log_s = v
        s = np.exp(log_s)
        if s <= 0:
            return 1e12
        y = (k - m) / s
        _, sse = _inner_solve(y, w, weights, s, w_max)
        return sse

    res = minimize(outer, [best[1], np.log(best[2])],
                   method="Nelder-Mead",
                   options={"maxiter": 400, "xatol": 1e-8, "fatol": 1e-14})

    m = float(res.x[0])
    s = float(np.exp(res.x[1]))
    y = (k - m) / s
    x, sse = _inner_solve(y, w, weights, s, w_max)
    a_t, d, c = x

    if c <= 1e-12:
        return SVIFit(None, np.nan, np.nan, k.size, T, False,
                      "degenerate fit: c collapsed to zero")

    p = SVIParams(a=float(a_t), b=float(c / s), rho=float(d / c),
                  m=m, s=s)

    fitted_w = svi_w(k, p)
    resid_w = fitted_w - w
    rmse_w = float(np.sqrt(np.mean(resid_w ** 2)))
    fitted_vol = np.sqrt(np.maximum(fitted_w, 0) / T)
    rmse_vol = float(np.sqrt(np.mean((fitted_vol - iv) ** 2)))

    return SVIFit(p, rmse_w, rmse_vol, k.size, T, True)


# --------------------------------------------------------------------------
# Stage 5: butterfly arbitrage (Durrleman's condition)
# --------------------------------------------------------------------------

def durrleman_g(k, p: SVIParams):
    """
    Durrleman's function. Non-negative everywhere <=> non-negative risk-neutral
    density <=> no butterfly arbitrage.

        g(k) = (1 - k w'/(2w))^2 - (w'^2/4)(1/w + 1/4) + w''/2

    A well-fitting SVI slice can and does produce negative density in the
    wings. Compute this; never assume the fit respects it.
    """
    k = np.asarray(k, float)
    w = svi_w(k, p)
    dw = svi_dw(k, p)
    d2w = svi_d2w(k, p)

    with np.errstate(divide="ignore", invalid="ignore"):
        term1 = (1.0 - k * dw / (2.0 * w)) ** 2
        term2 = (dw ** 2 / 4.0) * (1.0 / w + 0.25)
        g = term1 - term2 + d2w / 2.0

    return np.where(w > 0, g, -np.inf)


def check_butterfly(p: SVIParams, k_lo, k_hi, n=400, margin=0.05):
    """
    Evaluate g on a dense grid spanning the traded range plus a margin.

    Returns (passed, worst_g, k_at_worst).
    """
    grid = np.linspace(k_lo - margin, k_hi + margin, n)
    g = durrleman_g(grid, p)
    i = int(np.nanargmin(g))
    return bool(g[i] >= 0), float(g[i]), float(grid[i])


def check_lee_bound(p: SVIParams):
    """The wing condition. A fit resting exactly on the bound is being forced."""
    slack = LEE_WING_BOUND - p.b * (1.0 + abs(p.rho))
    return bool(slack >= 0), float(slack)


def min_total_variance(p: SVIParams):
    """Minimum of w over all k. Must be non-negative."""
    return p.a + p.b * p.s * np.sqrt(max(1.0 - p.rho ** 2, 0.0))


# --------------------------------------------------------------------------
# Stage 5: calendar arbitrage
# --------------------------------------------------------------------------

def check_calendar(fits: list[tuple[float, SVIParams]], k_lo=-0.5, k_hi=0.5, n=200):
    """
    Total variance must be non-decreasing in maturity at every log-moneyness.

    `fits` is [(T, params), ...]; it is sorted here, so order does not matter.

    Returns (passed, worst_violation_depth, k_at_worst, (T1, T2)).
    Depth is reported rather than a bare boolean, because a violation of 1e-9
    and one of 0.01 call for different responses.
    """
    ordered = sorted([f for f in fits if f[1] is not None], key=lambda x: x[0])
    if len(ordered) < 2:
        return True, 0.0, np.nan, None

    grid = np.linspace(k_lo, k_hi, n)
    worst, worst_k, worst_pair = 0.0, np.nan, None

    for (T1, p1), (T2, p2) in zip(ordered[:-1], ordered[1:]):
        diff = svi_w(grid, p2) - svi_w(grid, p1)   # must be >= 0
        i = int(np.argmin(diff))
        if diff[i] < worst:
            worst, worst_k, worst_pair = float(diff[i]), float(grid[i]), (T1, T2)

    return worst >= 0, worst, worst_k, worst_pair


def slice_report(fit: SVIFit, k_lo, k_hi):
    """One row of stage 4 and 5 diagnostics for an expiry."""
    if not fit.converged or fit.params is None:
        return {"converged": False, "reason": fit.reason}

    p = fit.params
    bf_ok, bf_worst, bf_k = check_butterfly(p, k_lo, k_hi)
    lee_ok, lee_slack = check_lee_bound(p)

    return {
        "converged": True,
        "T": fit.T,
        "n_points": fit.n_points,
        "rmse_vol_pts": fit.rmse_vol * 100,
        "a": p.a, "b": p.b, "rho": p.rho, "m": p.m, "s": p.s,
        "butterfly_ok": bf_ok,
        "durrleman_min": bf_worst,
        "durrleman_min_k": bf_k,
        "lee_ok": lee_ok,
        "lee_slack": lee_slack,
        "min_total_var": min_total_variance(p),
    }
