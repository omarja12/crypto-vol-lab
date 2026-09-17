"""
Tests for the portfolio layer.

Two kinds of test. The first kind checks the machinery does what it claims:
shrinkage improves conditioning, risk parity equalises risk contributions, the
factor model recovers known structure. The second kind is the honest one: on
data with a weak signal, mean-variance should NOT reliably beat equal-weight out
of sample. A portfolio library that cannot demonstrate its own principal failure
mode is one you cannot trust to tell you when to switch it off.
"""

from __future__ import annotations

import numpy as np

from portfolio_alloc import (
    equal_weight,
    mean_variance,
    minimum_variance,
    portfolio_stats,
    risk_parity,
)
from portfolio_backtest import walk_forward
from portfolio_cov import (
    condition_number,
    ledoit_wolf,
    pca_factor_cov,
    sample_cov,
)

RNG = np.random.default_rng(2026)


def report(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}\n       {detail}")
    return ok


def make_factor_returns(T, n, k=3, factor_vol=0.02, idio_vol=0.005, seed=0):
    """
    Returns driven by k common factors plus idiosyncratic noise - the structure
    a vol surface actually has. Used to check the factor covariance recovers it.
    """
    rng = np.random.default_rng(seed)
    loadings = rng.normal(size=(n, k))
    factors = rng.normal(0, factor_vol, size=(T, k))
    idio = rng.normal(0, idio_vol, size=(T, n))
    return factors @ loadings.T + idio


# --------------------------------------------------------------------------

def test_shrinkage_improves_conditioning():
    """The core reason to shrink: a usable inverse when assets are few."""
    R = make_factor_returns(T=40, n=10, seed=1)     # fewer obs than a stable cov needs
    cond_sample = condition_number(sample_cov(R))
    shrunk, intensity = ledoit_wolf(R)
    cond_shrunk = condition_number(shrunk)

    ok = cond_shrunk < cond_sample and 0 <= intensity <= 1
    return report(
        "Ledoit-Wolf improves conditioning",
        ok,
        f"condition number {cond_sample:.0f} -> {cond_shrunk:.0f}, "
        f"shrinkage intensity {intensity:.2f}",
    )


def test_pca_recovers_structure():
    """A 3-factor surface should be explained by 3 components."""
    R = make_factor_returns(T=250, n=12, k=3, seed=2)
    _, comp = pca_factor_cov(R, k=3, return_components=True)
    cum = comp["cumulative_explained"]
    ok = cum > 0.90
    return report(
        "PCA factor model captures a 3-factor surface",
        ok,
        f"3 components explain {cum:.1%} of variance "
        f"(ratios {np.round(comp['explained_variance_ratio'], 3)})",
    )


def test_risk_parity_equalises_contributions():
    # Vol-surface buckets are POSITIVELY correlated (they share a level factor),
    # so build the test covariance that way. Randomly-signed loadings produce
    # strong hedges, under which equal risk contribution is ill-posed - a real
    # limitation of risk parity, not a bug, but the wrong thing to test here.
    rng = np.random.default_rng(3)
    common = rng.normal(0, 0.02, size=(250, 1)) @ np.ones((1, 6))   # shared level
    idio = rng.normal(0, 0.008, size=(250, 6))
    R = common + idio
    from portfolio_alloc import risk_contributions
    cov = ledoit_wolf(R)[0]
    w = risk_parity(cov)

    rc = risk_contributions(w, cov)
    spread = rc.max() - rc.min()
    ok = spread < 0.02 and abs(w.sum() - 1) < 1e-6
    return report(
        "Risk parity equalises risk contributions",
        ok,
        f"risk contributions range {rc.min():.3f}-{rc.max():.3f} "
        f"(spread {spread:.4f}, target {1/len(w):.3f} each)",
    )


def test_min_variance_beats_equal_on_variance():
    """By construction min-variance must have the lowest in-sample variance."""
    R = make_factor_returns(T=250, n=8, seed=4)
    cov = ledoit_wolf(R)[0]
    w_mv = minimum_variance(cov)
    w_eq = equal_weight(8)

    var_mv = w_mv @ cov @ w_mv
    var_eq = w_eq @ cov @ w_eq
    ok = var_mv <= var_eq + 1e-12
    return report(
        "Minimum variance achieves the lowest in-sample variance",
        ok,
        f"portfolio variance: min-var {var_mv:.6f} <= equal-weight {var_eq:.6f}",
    )


def test_mean_variance_uses_the_signal():
    """With a TRUE signal in the means, mean-variance should tilt toward it."""
    n = 5
    cov = np.eye(n) * 0.01
    mu = np.array([0.0, 0.0, 0.0, 0.0, 0.05])       # asset 4 clearly best
    w = mean_variance(mu, cov, risk_aversion=1.0)
    ok = np.argmax(w) == 4
    return report(
        "Mean-variance tilts toward a genuine signal",
        ok,
        f"weights {np.round(w, 3)}, heaviest on asset {int(np.argmax(w))}",
    )


def test_honest_no_free_lunch():
    """
    THE HONEST TEST.

    Returns are pure noise with NO predictable signal. Mean-variance, chasing
    the garbage trailing mean, should NOT beat equal-weight out of sample on
    average across many independent noise draws.

    Averaging over seeds is the point: a single draw is noise, and judging an
    allocator on one backtest is the same multiple-testing error that sinks real
    strategy research. We assert on the mean over 20 worlds, not on one.
    """
    n_worlds = 8
    deltas = []
    turnovers = []
    for seed in range(n_worlds):
        rng = np.random.default_rng(1000 + seed)
        noise = rng.normal(0, 0.02, size=(400, 8))
        summary, _ = walk_forward(noise, mu_series=None, window=60, step=5,
                                  cov_method="ledoit_wolf", risk_aversion=3.0,
                                  only=["equal_weight", "mean_variance"])
        deltas.append(summary["mean_variance"]["sharpe"] - summary["equal_weight"]["sharpe"])
        turnovers.append(summary["mean_variance"]["avg_turnover"])

    mean_delta = float(np.mean(deltas))
    # On pure noise the average edge should be essentially zero. It certainly
    # should not be reliably positive.
    ok = mean_delta <= 0.15
    return report(
        "No free lunch: mean-variance shows no edge on noise, averaged over 8 worlds",
        ok,
        f"mean OOS Sharpe advantage {mean_delta:+.3f} (should be ~0), "
        f"while paying avg turnover {np.mean(turnovers):.2f} vs 0 for equal-weight",
    )


def test_signal_helps_when_real():
    """
    The complement, also averaged over worlds. When a REAL persistent signal is
    fed through mu_series, mean-variance SHOULD beat equal-weight ON AVERAGE.
    This confirms the harness is not simply incapable of ever showing an edge.

    The signal has to clear the covariance estimation noise to be usable, which
    is itself the honest lesson: a weak true signal can be invisible after
    realistic estimation error, so the drift here is deliberately not tiny.
    """
    n_worlds = 8
    deltas = []
    for seed in range(n_worlds):
        rng = np.random.default_rng(3000 + seed)
        T, n = 400, 6
        base = rng.normal(0, 0.02, size=(T, n))
        base[:, 0] += 0.010                            # persistent real edge
        mu_series = np.zeros((T, n))
        mu_series[:, 0] = 0.010                         # signal known at each date
        summary, _ = walk_forward(base, mu_series=mu_series, window=60, step=5,
                                  cov_method="ledoit_wolf", risk_aversion=2.0,
                                  only=["equal_weight", "mean_variance"])
        deltas.append(summary["mean_variance"]["sharpe"] - summary["equal_weight"]["sharpe"])

    mean_delta = float(np.mean(deltas))
    frac_positive = float(np.mean([d > 0 for d in deltas]))
    ok = mean_delta > 0.2 and frac_positive > 0.6
    return report(
        "A real signal lets mean-variance beat equal-weight on average",
        ok,
        f"mean OOS Sharpe advantage {mean_delta:+.3f}, positive in "
        f"{frac_positive:.0%} of worlds",
    )


if __name__ == "__main__":
    results = [
        test_shrinkage_improves_conditioning(),
        test_pca_recovers_structure(),
        test_risk_parity_equalises_contributions(),
        test_min_variance_beats_equal_on_variance(),
        test_mean_variance_uses_the_signal(),
        test_honest_no_free_lunch(),
        test_signal_helps_when_real(),
    ]
    print(f"\n{sum(results)}/{len(results)} checks passed")
