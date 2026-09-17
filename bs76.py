"""
Black-76 pricing and implied volatility inversion.

Implements the numerics required by stage 3 of the pipeline specification.
Everything is expressed in terms of the FORWARD, not spot, because the forward
is what stage 2 extracts and because it removes any need for a dividend or
carry assumption.

Conventions
-----------
F   forward price to expiry
K   strike
T   time to expiry in year fractions (ACT/365)
D   discount factor to expiry
cp  +1 for a call, -1 for a put
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

# Inversion is considered converged when the price residual falls below this,
# in the same currency units as the input price.
PRICE_TOL = 1e-10
MAX_NEWTON_ITER = 100

# Bracketing interval for the Brent fallback. 1000% vol is far beyond anything
# a real quote implies; if the root is outside this the quote is bad, not the
# solver.
SIGMA_LO = 1e-6
SIGMA_HI = 10.0


def black76_price(F, K, T, sigma, cp, D=1.0):
    """Undiscounted-forward Black-76 price, then discounted by D."""
    F, K, T, sigma, cp = map(np.asarray, (F, K, T, sigma, cp))

    with np.errstate(divide="ignore", invalid="ignore"):
        sqrt_t = np.sqrt(T)
        d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t
        price = cp * (F * norm.cdf(cp * d1) - K * norm.cdf(cp * d2))

    # Degenerate limits: zero vol or zero time collapse to intrinsic.
    intrinsic = np.maximum(cp * (F - K), 0.0)
    degenerate = (sigma <= 0) | (T <= 0)
    price = np.where(degenerate, intrinsic, price)

    return D * price


def black76_vega(F, K, T, sigma, D=1.0):
    """Sensitivity of price to a unit change in volatility (not per 1%)."""
    F, K, T, sigma = map(np.asarray, (F, K, T, sigma))

    with np.errstate(divide="ignore", invalid="ignore"):
        sqrt_t = np.sqrt(T)
        d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_t)
        vega = D * F * norm.pdf(d1) * sqrt_t

    return np.where((sigma <= 0) | (T <= 0), 0.0, vega)


def _seed_sigma(price, F, K, T, D):
    """
    Brenner-Subrahmanyam approximation, exact at the money and a decent
    starting point elsewhere. A bad seed costs iterations; it does not cost
    correctness, because Brent picks up anything Newton drops.
    """
    undiscounted = price / D
    seed = np.sqrt(2.0 * np.pi / T) * undiscounted / F
    return float(np.clip(seed, 0.01, 5.0))


def implied_vol(price, F, K, T, cp, D=1.0):
    """
    Invert a single option price to implied volatility.

    Returns NaN rather than raising when the price is not invertible. Callers
    are expected to count and report NaNs: per the spec, a rising failure rate
    at moderate moneyness is the signature of a bad forward from stage 2, not
    of a solver problem.
    """
    if not np.isfinite(price) or price <= 0 or T <= 0 or F <= 0 or K <= 0:
        return np.nan

    # No-arbitrage bounds. Outside these no volatility reproduces the price.
    intrinsic = D * max(cp * (F - K), 0.0)
    upper = D * (F if cp > 0 else K)
    if price <= intrinsic + PRICE_TOL or price >= upper - PRICE_TOL:
        return np.nan

    sigma = _seed_sigma(price, F, K, T, D)

    # Newton on vega. Fast where vega is meaningful, which is most of the
    # tradeable surface.
    for _ in range(MAX_NEWTON_ITER):
        diff = float(black76_price(F, K, T, sigma, cp, D)) - price
        if abs(diff) < PRICE_TOL:
            return sigma
        vega = float(black76_vega(F, K, T, sigma, D))
        if vega < 1e-12:
            break  # flat objective, hand over to Brent
        step = diff / vega
        sigma_new = sigma - step
        if sigma_new <= SIGMA_LO or sigma_new >= SIGMA_HI or not np.isfinite(sigma_new):
            break
        sigma = sigma_new

    # Brent fallback. Guaranteed to converge given a sign change, which the
    # arbitrage bounds above have already established.
    def objective(s):
        return float(black76_price(F, K, T, s, cp, D)) - price

    try:
        return brentq(objective, SIGMA_LO, SIGMA_HI, xtol=1e-12, maxiter=200)
    except (ValueError, RuntimeError):
        return np.nan


def implied_vol_vector(prices, F, strikes, T, cps, D=1.0):
    """Convenience wrapper. Deliberately a loop: correctness over speed here.

    This is the function to move into Numba once you are rebuilding history
    rather than processing one snapshot. It is not the bottleneck for a single
    day's chain.
    """
    return np.array(
        [
            implied_vol(p, F, k, T, c, D)
            for p, k, c in zip(np.asarray(prices), np.asarray(strikes), np.asarray(cps))
        ]
    )
