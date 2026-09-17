"""
Tests for stages 4 and 5.

Round-trip recovery, then deliberate arbitrage construction to confirm the
gates actually fire. A gate that never rejects anything is not a gate.
"""

from __future__ import annotations

import numpy as np

from ssvi import (
    butterfly_conditions,
    fit_ssvi,
    monotonise,
    ssvi_slice_to_svi,
    ssvi_w,
)
from svi import (
    SVIParams,
    calibrate_svi,
    check_butterfly,
    check_calendar,
    check_lee_bound,
    durrleman_g,
    svi_w,
)

RNG = np.random.default_rng(11)


def report(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}\n       {detail}")
    return ok


# --------------------------------------------------------------------------

def test_svi_round_trip():
    """
    Recover a known slice.

    Compare CURVES, not parameters. SVI parameters are near-degenerate in
    places, so two very different parameter sets can describe the same surface
    to within noise. Surface distance is the meaningful metric, which is also
    why the spec monitors it rather than parameter distance day over day.
    """
    T = 45 / 365
    truth = SVIParams(a=0.008, b=0.15, rho=-0.45, m=0.02, s=0.12)
    k = np.linspace(-0.35, 0.35, 25)
    w_true = svi_w(k, truth)
    iv = np.sqrt(w_true / T)

    fit = calibrate_svi(k, iv, T)
    if not fit.converged:
        return report("SVI round trip on exact data", False, fit.reason)

    w_fit = svi_w(k, fit.params)
    max_w_err = float(np.max(np.abs(w_fit - w_true)))
    ok = fit.rmse_vol * 100 < 0.05
    return report(
        "SVI round trip on exact data",
        ok,
        f"RMSE {fit.rmse_vol * 100:.4f} vol points, max total-variance error "
        f"{max_w_err:.2e}",
    )


def test_svi_under_noise():
    T = 45 / 365
    truth = SVIParams(a=0.008, b=0.15, rho=-0.45, m=0.02, s=0.12)
    k = np.linspace(-0.35, 0.35, 25)
    iv = np.sqrt(svi_w(k, truth) / T)
    noisy = iv + RNG.normal(0, 0.003, iv.shape)      # 0.3 vol point noise

    fit = calibrate_svi(k, noisy, T)
    ok = fit.converged and fit.rmse_vol * 100 < 0.5
    return report(
        "SVI fit with 0.3 vol point quote noise",
        ok,
        f"RMSE {fit.rmse_vol * 100:.3f} vol points (acceptance: under 0.5)",
    )


def test_weighting_follows_liquidity():
    """The fit should track tight strikes and tolerate error in wide ones."""
    T = 45 / 365
    truth = SVIParams(a=0.008, b=0.15, rho=-0.45, m=0.02, s=0.12)
    k = np.linspace(-0.35, 0.35, 25)
    iv = np.sqrt(svi_w(k, truth) / T)

    # Wings are wide and corrupted; the near-money strikes are tight and clean.
    spread = np.where(np.abs(k) > 0.2, 0.08, 0.004)
    corrupted = iv.copy()
    corrupted[np.abs(k) > 0.2] += RNG.normal(0, 0.02, int((np.abs(k) > 0.2).sum()))

    unweighted = calibrate_svi(k, corrupted, T)
    weighted = calibrate_svi(k, corrupted, T, iv_spread=spread)

    near = np.abs(k) <= 0.2
    err_u = np.sqrt(np.mean((np.sqrt(svi_w(k[near], unweighted.params) / T) - iv[near]) ** 2))
    err_w = np.sqrt(np.mean((np.sqrt(svi_w(k[near], weighted.params) / T) - iv[near]) ** 2))

    ok = err_w <= err_u
    return report(
        "Spread weighting improves the near-money fit",
        ok,
        f"near-money RMSE: unweighted {err_u * 100:.3f} vs weighted "
        f"{err_w * 100:.3f} vol points",
    )


# --------------------------------------------------------------------------

def test_durrleman_passes_clean_slice():
    good = SVIParams(a=0.008, b=0.15, rho=-0.45, m=0.02, s=0.12)
    ok, worst, at_k = check_butterfly(good, -0.4, 0.4)
    return report(
        "Durrleman condition holds on a well-behaved slice",
        ok,
        f"min g = {worst:.6f} at k = {at_k:.3f}",
    )


def test_durrleman_catches_arbitrage():
    """
    Construct a slice with negative density and confirm the gate fires.

    A very small vertex smoothness with a large angle produces a sharp kink,
    which is exactly where the density goes negative. This is not a contrived
    edge case; it is what an unconstrained fit to noisy wings tends to produce.
    """
    bad = SVIParams(a=0.001, b=0.9, rho=-0.92, m=0.0, s=0.005)
    ok, worst, at_k = check_butterfly(bad, -0.4, 0.4)
    detected = (not ok) and worst < 0
    return report(
        "Durrleman condition rejects an arbitrageable slice",
        detected,
        f"min g = {worst:.4f} at k = {at_k:.3f} (negative, correctly rejected)",
    )


def test_lee_bound_enforced_by_calibration():
    """The calibrated fit must never violate the wing bound."""
    T = 30 / 365
    k = np.linspace(-0.5, 0.5, 30)
    # Steep wings, the shape that pushes against the bound.
    iv = 0.5 + 0.8 * np.abs(k) + RNG.normal(0, 0.002, k.shape)

    fit = calibrate_svi(k, iv, T)
    ok_lee, slack = check_lee_bound(fit.params)
    return report(
        "Calibration respects the Lee wing bound",
        ok_lee,
        f"b(1+|rho|) = {fit.params.b * (1 + abs(fit.params.rho)):.4f}, "
        f"bound 2.0, slack {slack:.4f}",
    )


def test_calendar_detection():
    """Two slices that cross must be caught."""
    early = SVIParams(a=0.020, b=0.15, rho=-0.4, m=0.0, s=0.10)
    late = SVIParams(a=0.012, b=0.15, rho=-0.4, m=0.0, s=0.10)  # lower: crosses
    ok, worst, at_k, pair = check_calendar([(0.10, early), (0.25, late)])
    detected = (not ok) and worst < 0
    return report(
        "Calendar arbitrage detected between crossing slices",
        detected,
        f"worst violation {worst:.5f} in total variance at k = {at_k:.3f}",
    )


def test_calendar_passes_ordered_slices():
    early = SVIParams(a=0.008, b=0.15, rho=-0.4, m=0.0, s=0.10)
    late = SVIParams(a=0.020, b=0.18, rho=-0.4, m=0.0, s=0.10)
    ok, worst, _, _ = check_calendar([(0.10, early), (0.25, late)])
    return report(
        "Calendar check passes properly ordered slices",
        ok,
        f"minimum gap {worst:.6f} (non-negative)",
    )


# --------------------------------------------------------------------------

def test_ssvi_recovery():
    """Fit SSVI to data generated from SSVI."""
    rho_t, eta_t, gamma_t = -0.55, 1.2, 0.45
    Ts = np.array([14, 35, 70, 140]) / 365
    thetas_t = np.array([0.30, 0.32, 0.34, 0.36]) ** 2 * Ts * (365 / 365)
    thetas_t = np.array([0.0035, 0.0095, 0.0200, 0.0420])

    slices = []
    for T, th in zip(Ts, thetas_t):
        k = np.linspace(-0.4, 0.4, 21)
        w = ssvi_w(k, th, rho_t, eta_t, gamma_t)
        slices.append((T, k, np.sqrt(w / T), None))

    params, diag = fit_ssvi(slices)
    ok = diag["arbitrage_free"] and diag["calendar_free"] and diag["mean_rmse_vol_pts"] < 0.5
    return report(
        "SSVI joint fit recovers a generated surface",
        ok,
        f"mean RMSE {diag['mean_rmse_vol_pts']:.4f} vol points, "
        f"arbitrage-free {diag['arbitrage_free']}, calendar-free {diag['calendar_free']}",
    )


def test_ssvi_removes_calendar_crossing():
    """
    The remediation path.

    Build slices that cross, fit per-slice SVI (which will keep the crossing),
    then fit SSVI and confirm the crossing is gone.
    """
    Ts = np.array([21, 42, 84]) / 365
    k = np.linspace(-0.35, 0.35, 21)

    # Middle expiry deliberately too low. Note this needs a LARGE vol drop:
    # total variance is sigma^2 * T, so a modest fall in vol at a longer
    # maturity still leaves w increasing. Naive test constructions miss this.
    atm_vols = [0.85, 0.38, 0.60]
    per_slice, ssvi_input = [], []
    for T, v in zip(Ts, atm_vols):
        iv = v + 0.35 * k ** 2 - 0.15 * k
        fit = calibrate_svi(k, iv, T)
        per_slice.append((T, fit.params))
        ssvi_input.append((T, k, iv, None))

    before_ok, before_worst, _, _ = check_calendar(per_slice)

    params, diag = fit_ssvi(ssvi_input)
    after = [
        (T, ssvi_slice_to_svi(th, params.rho, params.eta, params.gamma))
        for T, th in zip(params.Ts, params.thetas)
    ]
    after_ok, after_worst, _, _ = check_calendar(after)

    ok = (not before_ok) and after_ok
    return report(
        "SSVI removes a calendar crossing that per-slice SVI leaves",
        ok,
        f"before: violation {before_worst:.5f}; after: {after_worst:.6f}. "
        f"Cost is fit quality: mean RMSE now {diag['mean_rmse_vol_pts']:.2f} vol points",
    )


def test_monotonise():
    raw = np.array([0.004, 0.009, 0.007, 0.020])
    fixed = monotonise(raw)
    ok = np.all(np.diff(fixed) >= 0) and fixed[2] == 0.009
    return report(
        "Theta monotonisation is minimal and visible",
        ok,
        f"{list(np.round(raw, 4))} -> {list(np.round(fixed, 4))} "
        f"(only index 2 moved)",
    )


if __name__ == "__main__":
    results = [
        test_svi_round_trip(),
        test_svi_under_noise(),
        test_weighting_follows_liquidity(),
        test_durrleman_passes_clean_slice(),
        test_durrleman_catches_arbitrage(),
        test_lee_bound_enforced_by_calibration(),
        test_calendar_detection(),
        test_calendar_passes_ordered_slices(),
        test_ssvi_recovery(),
        test_ssvi_removes_calendar_crossing(),
        test_monotonise(),
    ]
    print(f"\n{sum(results)}/{len(results)} checks passed")
