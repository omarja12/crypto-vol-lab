"""
Synthetic validation for stages 2 and 3.

Generate option prices from KNOWN parameters, run them back through the
pipeline, and check the parameters are recovered. This runs with no network
and no market data, and it is the correct first test: if the pipeline cannot
recover a forward it generated itself, live data will not help.

Test 4 is not a pass/fail test. It quantifies failure mode #1 from Appendix B
of the specification, so that the size of the error is on the record rather
than an assertion.
"""

from __future__ import annotations

import numpy as np

from bs76 import black76_price, implied_vol
from stage2_forward import extract_forward

RNG = np.random.default_rng(20260901)

# Ground truth
TRUE_INDEX = 100_000.0
TRUE_BASIS = 0.08          # 8% annualised, typical crypto contango
TRUE_T = 45 / 365
TRUE_F = TRUE_INDEX * (1 + TRUE_BASIS * TRUE_T)
TRUE_RATE = 0.045
TRUE_D = np.exp(-TRUE_RATE * TRUE_T)

STRIKES = np.arange(70_000, 145_000, 2_500, dtype=float)


def true_smile(K, F=TRUE_F):
    """A skewed, convex smile. Shape is arbitrary; only recoverability matters."""
    k = np.log(K / F)
    return 0.55 - 0.35 * k + 1.10 * k**2


def generate_chain(F=TRUE_F, D=TRUE_D, T=TRUE_T, noise_bps=0.0):
    sig = true_smile(STRIKES, F)
    C = np.array([float(black76_price(F, K, T, s, +1, D)) for K, s in zip(STRIKES, sig)])
    P = np.array([float(black76_price(F, K, T, s, -1, D)) for K, s in zip(STRIKES, sig)])
    if noise_bps:
        scale = noise_bps * 1e-4 * TRUE_INDEX
        C = C + RNG.normal(0, scale, C.shape)
        P = P + RNG.normal(0, scale, P.shape)
    return C, P, sig


def report(name, passed, detail):
    print(f"[{'PASS' if passed else 'FAIL'}] {name}\n       {detail}")
    return passed


def test_1_exact_recovery():
    C, P, _ = generate_chain()
    fit = extract_forward(STRIKES, C, P, TRUE_T, spot=TRUE_INDEX)
    ef = abs(fit.forward - TRUE_F) / TRUE_F
    ed = abs(fit.discount - TRUE_D)
    ok = fit.ok and ef < 1e-9 and ed < 1e-9
    return report(
        "Stage 2 exact recovery on noiseless prices",
        ok,
        f"forward error {ef:.3e} relative, discount error {ed:.3e}, R2 {fit.r_squared:.10f}",
    )


def test_2_recovery_under_noise():
    C, P, _ = generate_chain(noise_bps=1.0)
    fit = extract_forward(STRIKES, C, P, TRUE_T, spot=TRUE_INDEX)
    ef_bps = abs(fit.forward - TRUE_F) / TRUE_F * 1e4
    ok = fit.ok and ef_bps < 10.0
    return report(
        "Stage 2 recovery with 1bp quote noise",
        ok,
        f"forward error {ef_bps:.2f} bps (acceptance: under 10), "
        f"R2 {fit.r_squared:.6f}, {fit.n_pairs_used} pairs",
    )


def test_3_outlier_rejection():
    C, P, _ = generate_chain(noise_bps=1.0)
    C = C.copy()
    C[len(C) // 2] *= 1.5          # one stale call quote, near the money
    fit = extract_forward(STRIKES, C, P, TRUE_T, spot=TRUE_INDEX)
    ef_bps = abs(fit.forward - TRUE_F) / TRUE_F * 1e4
    ok = fit.n_pairs_trimmed >= 1 and ef_bps < 25.0
    return report(
        "Stage 2 trims a single stale quote",
        ok,
        f"{fit.n_pairs_trimmed} pair(s) trimmed, forward error {ef_bps:.2f} bps",
    )


def test_4_iv_round_trip():
    C, P, sig = generate_chain()
    fit = extract_forward(STRIKES, C, P, TRUE_T, spot=TRUE_INDEX)

    # Stage 3 OTM rule: calls above the forward, puts below.
    errors = []
    for K, c_px, p_px, s_true in zip(STRIKES, C, P, sig):
        if K > fit.forward:
            iv = implied_vol(c_px, fit.forward, K, TRUE_T, +1, fit.discount)
        else:
            iv = implied_vol(p_px, fit.forward, K, TRUE_T, -1, fit.discount)
        errors.append(abs(iv - s_true))

    errors = np.array(errors)
    n_failed = int(np.sum(~np.isfinite(errors)))
    worst = np.nanmax(errors) * 100
    ok = n_failed == 0 and worst < 1e-6
    return report(
        "Stage 3 implied vol round trip",
        ok,
        f"{n_failed} inversion failures, worst error {worst:.3e} vol points",
    )


def test_5_wrong_forward_damage():
    """
    Quantify Appendix B failure mode #1.

    Instead of extracting the forward, assume the forward equals the index
    (i.e. ignore the basis) - the shortcut that feels harmless. Then measure
    the skew that appears in a surface which by construction has none beyond
    its true shape.
    """
    C, P, sig = generate_chain()
    fit = extract_forward(STRIKES, C, P, TRUE_T, spot=TRUE_INDEX)
    wrong_F = TRUE_INDEX          # basis ignored

    def rr25(F, D):
        """25-delta risk reversal, the standard one-number summary of skew."""
        ivs, ks = [], []
        for K, c_px, p_px in zip(STRIKES, C, P):
            cp = +1 if K > F else -1
            px = c_px if cp > 0 else p_px
            iv = implied_vol(px, F, K, TRUE_T, cp, D)
            if np.isfinite(iv):
                ivs.append(iv)
                ks.append(np.log(K / F))
        ivs, ks = np.array(ivs), np.array(ks)
        lo = np.interp(-0.15, ks, ivs)
        hi = np.interp(+0.15, ks, ivs)
        return hi - lo

    correct = rr25(fit.forward, fit.discount)
    naive = rr25(wrong_F, TRUE_D)
    distortion = abs(naive - correct) * 100

    print(
        f"[INFO] Wrong-forward distortion\n"
        f"       true forward {fit.forward:,.0f} vs assumed {wrong_F:,.0f} "
        f"({(fit.forward / wrong_F - 1) * 1e4:.0f} bps)\n"
        f"       risk reversal shifts by {distortion:.2f} vol points - a stable, "
        f"tradeable-looking skew that is pure artefact"
    )
    return True


if __name__ == "__main__":
    print(f"Ground truth: F={TRUE_F:,.2f}  D={TRUE_D:.8f}  T={TRUE_T:.6f}\n")
    results = [
        test_1_exact_recovery(),
        test_2_recovery_under_noise(),
        test_3_outlier_rejection(),
        test_4_iv_round_trip(),
        test_5_wrong_forward_damage(),
    ]
    print(f"\n{sum(results)}/{len(results)} checks passed")
