"""
Stage 2: forward and discount factor extraction by put-call parity regression.

    C(K) - P(K) = D(T) * (F(T) - K)

This is linear in K, so a regression of (C - P) on K gives

    slope     = -D(T)
    intercept =  D(T) * F(T)

and therefore both the discount factor and the forward, implied by the option
market itself. No external rate curve, no carry assumption.

This is the highest-leverage stage in the pipeline. An error in the forward is
a horizontal shift in log-moneyness, which presents as a systematic skew tilt.
That tilt is stable, looks tradeable, and survives backtesting. See
test_synthetic.py, which demonstrates the effect on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


# Points whose residual exceeds this many median absolute deviations are
# dropped before the refit. One stale quote on one leg of one pair is enough
# to tilt the whole forward under plain least squares.
TRIM_MAD = 3.0
MIN_PAIRS = 4


@dataclass
class ForwardFit:
    """Result of one expiry's extraction, with the diagnostics the spec requires."""

    forward: float
    discount: float
    implied_rate: float          # continuously compounded, from the discount factor
    n_pairs_used: int
    n_pairs_trimmed: int
    r_squared: float
    residual_std: float          # currency units
    ok: bool
    reason: str = ""

    def as_dict(self):
        return asdict(self)


def _ols_line(x, y):
    """Least squares fit of y = a + b*x. Returns (intercept, slope, residuals)."""
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    return coef[0], coef[1], resid


def extract_forward(strikes, call_prices, put_prices, T, moneyness_band=0.10, spot=None):
    """
    Extract F(T) and D(T) for a single expiry.

    Parameters
    ----------
    strikes, call_prices, put_prices : array-like
        Matched arrays. Only strikes where BOTH legs survived stage 1 filtering
        should be passed in.
    T : float
        Year fraction to settlement.
    moneyness_band : float
        Restrict the regression to strikes within this fraction of an initial
        forward estimate. Near-money pairs are the liquid ones; far strikes add
        noise, not information. Widened automatically if too few pairs survive.
    spot : float, optional
        Used only to seed the moneyness band when supplied.

    Returns
    -------
    ForwardFit
    """
    K = np.asarray(strikes, dtype=float)
    C = np.asarray(call_prices, dtype=float)
    P = np.asarray(put_prices, dtype=float)

    finite = np.isfinite(K) & np.isfinite(C) & np.isfinite(P)
    K, C, P = K[finite], C[finite], P[finite]

    if K.size < MIN_PAIRS:
        return ForwardFit(np.nan, np.nan, np.nan, K.size, 0, np.nan, np.nan,
                          False, f"only {K.size} usable pairs")

    y = C - P

    # First pass over all pairs, used only to locate the money.
    intercept, slope, _ = _ols_line(K, y)
    if slope >= 0:
        return ForwardFit(np.nan, np.nan, np.nan, K.size, 0, np.nan, np.nan,
                          False, "non-negative slope: parity violated across the chain")
    f_seed = intercept / -slope if slope != 0 else (spot if spot else np.median(K))

    # Restrict to the near-money band, widening if it leaves too little.
    band = moneyness_band
    while band <= 1.0:
        sel = np.abs(K / f_seed - 1.0) <= band
        if sel.sum() >= MIN_PAIRS:
            break
        band *= 2
    else:
        sel = np.ones_like(K, dtype=bool)

    Ks, ys = K[sel], y[sel]

    # Second pass, then trim outliers by MAD and refit.
    intercept, slope, resid = _ols_line(Ks, ys)
    mad = np.median(np.abs(resid - np.median(resid)))
    n_trimmed = 0
    if mad > 0:
        keep = np.abs(resid - np.median(resid)) <= TRIM_MAD * mad
        if keep.sum() >= MIN_PAIRS:
            n_trimmed = int((~keep).sum())
            Ks, ys = Ks[keep], ys[keep]
            intercept, slope, resid = _ols_line(Ks, ys)

    if slope >= 0:
        return ForwardFit(np.nan, np.nan, np.nan, len(Ks), n_trimmed, np.nan, np.nan,
                          False, "non-negative slope after trimming")

    D = -slope
    F = intercept / D

    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    resid_std = float(np.std(resid, ddof=2)) if len(resid) > 2 else np.nan

    rate = -np.log(D) / T if (D > 0 and T > 0) else np.nan

    # Acceptance criterion from the spec. Below 0.999 means something in the
    # chain is stale or mispaired; do not silently continue.
    ok = bool(np.isfinite(r2) and r2 >= 0.999 and D > 0 and F > 0)
    reason = "" if ok else f"r2={r2:.6f} below acceptance threshold 0.999"

    return ForwardFit(float(F), float(D), float(rate), len(Ks), n_trimmed,
                      float(r2), resid_std, ok, reason)


def basis_check(fit: ForwardFit, index_price: float, T: float):
    """
    Annualised basis of the extracted forward over the index.

    On a crypto venue this replaces the implied dividend of the equity case,
    and it has an independent cross-check: the listed futures curve. If the
    two disagree by more than a few basis points, stage 2 failed for that
    expiry and everything downstream of it is contaminated.
    """
    if not np.isfinite(fit.forward) or index_price <= 0 or T <= 0:
        return np.nan
    return (fit.forward / index_price - 1.0) / T
