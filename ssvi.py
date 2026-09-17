"""
SSVI: the remediation path when per-slice raw SVI produces calendar crossings.

Raw SVI fits each expiry independently, so nothing stops adjacent slices from
crossing. SSVI (Gatheral and Jacquier) ties them together through a single
shape function and is arbitrage-free by construction under stated conditions:

    w(k, theta) = (theta/2) * ( 1 + rho*phi(theta)*k
                                + sqrt( (phi(theta)*k + rho)^2 + (1 - rho^2) ) )

where theta = theta(T) is at-the-money total variance, and phi is a shape
function. This module uses the power law

    phi(theta) = eta / ( theta^gamma * (1 + theta)^(1 - gamma) )

CONDITIONS (Gatheral-Jacquier)
------------------------------
No calendar arbitrage:  theta(T) non-decreasing in T.
No butterfly arbitrage: theta*phi(theta)*(1 + |rho|) < 4
                    and theta*phi(theta)^2*(1 + |rho|) <= 4

Both are enforced during fitting rather than checked afterwards.

THE TRADE-OFF IS REAL
---------------------
SSVI has three global parameters plus the theta term structure, against five
per slice for raw SVI. Per-slice fit quality gets worse. Accept it. A surface
with slightly larger residuals and no arbitrage is usable; a tight fit with
negative density is not.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.optimize import minimize

from svi import SVIParams


@dataclass
class SSVIParams:
    rho: float
    eta: float
    gamma: float
    thetas: np.ndarray      # ATM total variance, one per expiry
    Ts: np.ndarray          # matching maturities

    def as_dict(self):
        d = asdict(self)
        d["thetas"] = list(self.thetas)
        d["Ts"] = list(self.Ts)
        return d


def phi_power_law(theta, eta, gamma):
    theta = np.maximum(np.asarray(theta, float), 1e-12)
    return eta / (theta ** gamma * (1.0 + theta) ** (1.0 - gamma))


def ssvi_w(k, theta, rho, eta, gamma):
    """Total variance under SSVI."""
    k = np.asarray(k, float)
    phi = phi_power_law(theta, eta, gamma)
    x = phi * k
    return 0.5 * theta * (1.0 + rho * x + np.sqrt((x + rho) ** 2 + (1.0 - rho ** 2)))


def butterfly_conditions(theta, rho, eta, gamma):
    """Returns (c1_slack, c2_slack); both must be >= 0."""
    phi = phi_power_law(theta, eta, gamma)
    c1 = 4.0 - theta * phi * (1.0 + abs(rho))
    c2 = 4.0 - theta * phi ** 2 * (1.0 + abs(rho))
    return float(np.min(c1)), float(np.min(c2))


def monotonise(thetas):
    """
    Enforce non-decreasing ATM total variance.

    A running maximum is the minimal change that removes calendar arbitrage,
    and it makes the adjustment visible: compare against the input to see
    which expiries were moved and by how much.
    """
    return np.maximum.accumulate(np.asarray(thetas, float))


def atm_total_variance(slices):
    """
    Read theta directly from the quotes.

    At k = 0 the SSVI expression collapses to w = theta exactly, so theta IS
    the at-the-money total variance. It is an observable, not a free parameter.
    Fitting it anyway turns a well-conditioned 3-parameter problem into a badly
    conditioned (3 + n) one, for no gain.
    """
    out = []
    for T, k, iv, _ in slices:
        k, iv = np.asarray(k, float), np.asarray(iv, float)
        good = np.isfinite(k) & np.isfinite(iv) & (iv > 0)
        if good.sum() > 1:
            order = np.argsort(k[good])
            atm = float(np.interp(0.0, k[good][order], iv[good][order]))
        else:
            atm = float(np.nanmean(iv))
        out.append(atm ** 2 * T)
    return np.array(out)


def fit_ssvi(slices, atm_thetas=None, refine_thetas=False):
    """
    Joint fit across expiries.

    Parameters
    ----------
    slices : list of (T, k_array, iv_array, weight_array or None)
        One entry per expiry.
    atm_thetas : array, optional
        ATM total variance per expiry. Read from the quotes when omitted.
    refine_thetas : bool
        Allow theta to move during fitting. Off by default: theta is observable
        and letting the optimiser adjust it trades conditioning for nothing.
        Turn it on only if the ATM quotes for some expiry are unreliable.

    Returns
    -------
    (SSVIParams, diagnostics dict)
    """
    Ts = np.array([s[0] for s in slices], float)
    order = np.argsort(Ts)
    slices = [slices[i] for i in order]
    Ts = Ts[order]

    if atm_thetas is None:
        atm_thetas = atm_total_variance(slices)
    else:
        atm_thetas = np.asarray(atm_thetas, float)[order]

    # Monotonising here is what removes calendar arbitrage. Compare the output
    # against the input to see which expiries were moved and by how much.
    thetas_fixed = monotonise(atm_thetas)

    n_theta = len(thetas_fixed) if refine_thetas else 0
    if refine_thetas:
        inc = np.maximum(np.diff(np.concatenate([[0.0], thetas_fixed])), 1e-10)
        x0 = np.concatenate([[0.0, 0.0, 0.0], np.log(inc)])
    else:
        x0 = np.array([0.0, 0.0, 0.0])

    def unpack(x):
        rho = np.tanh(x[0])                       # |rho| < 1 structurally
        eta = np.exp(x[1])                        # eta > 0
        gamma = 1.0 / (1.0 + np.exp(-x[2]))       # gamma in (0, 1)
        thetas = np.cumsum(np.exp(x[3:])) if n_theta else thetas_fixed
        return rho, eta, gamma, thetas

    def objective(x):
        rho, eta, gamma, thetas = unpack(x)

        sse = 0.0
        for (T, k, iv, wgt), theta in zip(slices, thetas):
            k, iv = np.asarray(k, float), np.asarray(iv, float)
            good = np.isfinite(k) & np.isfinite(iv) & (iv > 0)
            if good.sum() < 3:
                continue
            w_obs = iv[good] ** 2 * T
            w_fit = ssvi_w(k[good], theta, rho, eta, gamma)
            weights = np.ones(good.sum()) if wgt is None else np.asarray(wgt, float)[good]
            sse += float(np.sum(weights * (w_fit - w_obs) ** 2))

        # Butterfly conditions as a steep penalty. Cheaper and more robust than
        # a hard constraint here, and the final fit is verified explicitly.
        c1, c2 = butterfly_conditions(thetas, rho, eta, gamma)
        penalty = 0.0
        if c1 < 0:
            penalty += 1e6 * c1 ** 2
        if c2 < 0:
            penalty += 1e6 * c2 ** 2

        return sse + penalty

    # Multi-start over the shape parameters. The objective is smooth but not
    # convex in eta, and a single start occasionally settles in a poor basin.
    best = (np.inf, x0)
    for rho_seed in (-1.0, 0.0, 1.0):
        for eta_seed in (-0.7, 0.0, 0.7, 1.4):
            start = x0.copy()
            start[0], start[1] = rho_seed, eta_seed
            res = minimize(objective, start, method="Nelder-Mead",
                           options={"maxiter": 8000, "maxfev": 8000,
                                    "xatol": 1e-10, "fatol": 1e-16})
            if res.fun < best[0]:
                best = (res.fun, res.x)

    res_x = best[1]
    res = minimize(objective, res_x, method="Nelder-Mead",
                   options={"maxiter": 20000, "maxfev": 20000,
                            "xatol": 1e-12, "fatol": 1e-18})

    rho, eta, gamma, thetas = unpack(res.x)
    c1, c2 = butterfly_conditions(thetas, rho, eta, gamma)

    # Per-slice error, reported in volatility points because that is the unit
    # anyone can reason about.
    rmses = []
    for (T, k, iv, _), theta in zip(slices, thetas):
        k, iv = np.asarray(k, float), np.asarray(iv, float)
        good = np.isfinite(k) & np.isfinite(iv) & (iv > 0)
        if good.sum() < 3:
            rmses.append(np.nan)
            continue
        w_fit = ssvi_w(k[good], theta, rho, eta, gamma)
        vol_fit = np.sqrt(np.maximum(w_fit, 0) / T)
        rmses.append(float(np.sqrt(np.mean((vol_fit - iv[good]) ** 2)) * 100))

    params = SSVIParams(rho=rho, eta=eta, gamma=gamma, thetas=thetas, Ts=Ts)
    diagnostics = {
        "rho": rho, "eta": eta, "gamma": gamma,
        "butterfly_c1_slack": c1,
        "butterfly_c2_slack": c2,
        "arbitrage_free": bool(c1 >= 0 and c2 >= 0),
        "calendar_free": bool(np.all(np.diff(thetas) >= 0)),
        "rmse_vol_pts_by_expiry": rmses,
        "mean_rmse_vol_pts": float(np.nanmean(rmses)),
        "optimiser_success": bool(res.success),
    }
    return params, diagnostics


def ssvi_slice_to_svi(theta, rho, eta, gamma) -> SVIParams:
    """
    Express one SSVI slice in raw SVI parameters.

    Lets the SSVI output flow through the same plotting, density and
    diagnostic code paths as a per-slice fit, so nothing downstream needs to
    know which parameterisation produced the surface.
    """
    phi = float(phi_power_law(theta, eta, gamma))
    return SVIParams(
        a=0.5 * theta * (1.0 - rho ** 2),
        b=0.5 * theta * phi,
        rho=rho,
        m=-rho / phi,
        s=np.sqrt(1.0 - rho ** 2) / phi,
    )
