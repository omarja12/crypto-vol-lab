"""
Walk-forward comparison of allocators.

This is the module that matters. The allocators all look impressive in a single
in-sample fit; the only question that counts is whether any of them beats
equal-weight OUT OF SAMPLE, on data the optimiser did not see when choosing the
weights. At this breadth the honest prior is that they mostly do not, and a
harness that cannot surface that is worse than useless.

METHOD
------
Rolling window. At each rebalance, estimate mu and cov from a trailing window,
choose weights, then hold them over the next (out-of-sample) period and record
the realised return. Never let the estimation window overlap the evaluation
period - that overlap is the most common way a backtest lies to you.

No costs are modelled here. Turnover is reported instead, because at your
account size rebalancing cost is a first-order drag and a high-turnover
optimiser can be worse than equal-weight even when its gross returns look
better. Fold your own per-trade cost model in before believing any of it.
"""

from __future__ import annotations

import numpy as np

from portfolio_alloc import (
    equal_weight,
    inverse_variance,
    mean_variance,
    minimum_variance,
    risk_parity,
)
from portfolio_cov import ledoit_wolf, pca_factor_cov, sample_cov


def _annualise(returns_per_step, steps_per_year):
    r = np.asarray(returns_per_step, float)
    mean = r.mean() * steps_per_year
    vol = r.std(ddof=1) * np.sqrt(steps_per_year)
    return mean, vol, (mean / vol if vol > 0 else np.nan)


def walk_forward(returns, mu_series=None, window=60, step=5,
                 cov_method="ledoit_wolf", risk_aversion=3.0,
                 steps_per_year=252, only=None):
    """
    Compare allocators out of sample.

    Parameters
    ----------
    returns : (T x n)
        Per-period bucket returns (VRP returns or delta-hedged P&L).
    mu_series : (T x n), optional
        Expected-return estimate available AT each date, e.g. the measured
        variance risk premium. If omitted, the trailing sample mean is used,
        which is precisely the weak estimate that makes mean-variance
        disappoint - so passing a real signal here is the whole game.
    window : int
        Trailing estimation window.
    step : int
        Rebalance frequency, also the out-of-sample holding length.
    cov_method : {"sample", "ledoit_wolf", "pca_factor"}
    only : list of allocator names, optional
        Restrict to these allocators. Skips the slower optimisers when a caller
        only needs to compare a couple, which matters for repeated backtests.

    Returns
    -------
    dict keyed by allocator name, each with realised out-of-sample series,
    annualised stats, and average turnover.
    """
    R = np.asarray(returns, float)
    T, n = R.shape

    def estimate_cov(win):
        if cov_method == "sample":
            return sample_cov(win)
        if cov_method == "pca_factor":
            return pca_factor_cov(win, k=min(3, n - 1))
        return ledoit_wolf(win)[0]

    allocators = {
        "equal_weight": lambda mu, cov: equal_weight(n),
        "inverse_variance": lambda mu, cov: inverse_variance(cov),
        "minimum_variance": lambda mu, cov: minimum_variance(cov),
        "risk_parity": lambda mu, cov: risk_parity(cov),
        "mean_variance": lambda mu, cov: mean_variance(mu, cov, risk_aversion),
    }
    if only is not None:
        allocators = {k: v for k, v in allocators.items() if k in only}

    results = {name: {"returns": [], "weights": []} for name in allocators}

    start = window
    while start + step <= T:
        win = R[start - window:start]
        cov = estimate_cov(win)

        if mu_series is not None:
            mu = np.asarray(mu_series, float)[start - 1]
        else:
            mu = win.mean(axis=0)

        oos = R[start:start + step]

        for name, alloc in allocators.items():
            try:
                w = alloc(mu, cov)
            except Exception:
                w = equal_weight(n)
            # Realised return of held weights over the out-of-sample block.
            block = float(np.mean(oos @ w))
            results[name]["returns"].append(block)
            results[name]["weights"].append(w)

        start += step

    summary = {}
    for name, data in results.items():
        rets = np.array(data["returns"])
        weights = np.array(data["weights"])
        turnover = (np.mean(np.sum(np.abs(np.diff(weights, axis=0)), axis=1))
                    if len(weights) > 1 else 0.0)
        mean, vol, sharpe = _annualise(rets, steps_per_year / step)
        summary[name] = {
            "ann_return": mean,
            "ann_vol": vol,
            "sharpe": sharpe,
            "avg_turnover": float(turnover),
            "n_rebalances": len(rets),
            "cum_return": float(np.prod(1 + rets) - 1),
        }

    return summary, results


def print_summary(summary, benchmark="equal_weight"):
    """Table sorted by Sharpe, with the margin over the benchmark made explicit."""
    bench_sharpe = summary.get(benchmark, {}).get("sharpe", np.nan)

    print(f"{'allocator':<20}{'ann.ret':>9}{'ann.vol':>9}{'Sharpe':>8}"
          f"{'vs bench':>10}{'turnover':>10}")
    print("-" * 76)
    for name, s in sorted(summary.items(), key=lambda x: -(x[1]["sharpe"] if np.isfinite(x[1]["sharpe"]) else -np.inf)):
        delta = s["sharpe"] - bench_sharpe if np.isfinite(bench_sharpe) else np.nan
        flag = "  <- benchmark" if name == benchmark else ""
        print(f"{name:<20}{s['ann_return']:>8.1%}{s['ann_vol']:>9.1%}"
              f"{s['sharpe']:>8.2f}{delta:>+10.2f}{s['avg_turnover']:>10.2f}{flag}")

    print("\nRead this honestly: if nothing clears the benchmark by a margin that")
    print("survives your per-trade costs and the turnover column, equal-weight is")
    print("the correct choice and that is a finding worth writing up.")
