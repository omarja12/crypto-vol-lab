"""
Offline tests for the historical loaders.

Neither loader can be tested against the live services from a restricted
network, so these build synthetic files in the real wire formats and exercise
every parsing and normalisation path. What they cannot catch is the remote
services changing their schema; run the live commands once and compare.
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from bs76 import black76_price, implied_vol
from deribit_history import load_trades, liquidity_map, effective_spread_stats
from stage2_forward import extract_forward
from tardis_free import (
    estimate_index_from_slope,
    free_tier_dates,
    read_chain_day,
    to_normalised,
)

INDEX = 100_000.0
RATE = 0.045


def build_tardis_csv(path: Path, n_snapshots=3):
    """A gzipped CSV in Tardis's Deribit options_chain schema."""
    base = dt.datetime(2026, 3, 1, 8, 0, tzinfo=dt.timezone.utc)
    rows = []

    for snap in range(n_snapshots):
        ts = base + dt.timedelta(minutes=30 * snap)
        for days, vol0, basis in [(21, 0.50, 0.06), (49, 0.55, 0.09)]:
            expiry = (base + dt.timedelta(days=days)).replace(hour=8)
            T = (expiry - ts).total_seconds() / (365 * 86400)
            F = INDEX * (1 + basis * T)
            D = np.exp(-RATE * T)
            tag = expiry.strftime("%d%b%y").upper().lstrip("0")

            for K in np.arange(80_000, 130_000, 2_500, dtype=float):
                k = np.log(K / F)
                sigma = vol0 - 0.30 * k + 0.90 * k**2
                for cp, word, letter in ((1, "call", "C"), (-1, "put", "P")):
                    usd = float(black76_price(F, K, T, sigma, cp, D))
                    native = usd / INDEX
                    spread = max(native * 0.02, 0.0005)
                    rows.append(
                        {
                            "exchange": "deribit",
                            "symbol": f"BTC-{tag}-{int(K)}-{letter}",
                            "timestamp": int(ts.timestamp() * 1_000_000),
                            "local_timestamp": int(ts.timestamp() * 1_000_000),
                            "type": word,
                            "strike_price": K,
                            "expiration": int(expiry.timestamp() * 1_000_000),
                            "open_interest": 50,
                            "last_price": native,
                            "bid_price": native - spread / 2,
                            "bid_amount": 5.0,
                            "bid_iv": sigma * 100,
                            "ask_price": native + spread / 2,
                            "ask_amount": 5.0,
                            "ask_iv": sigma * 100,
                            "mark_price": native,
                            "mark_iv": sigma * 100,
                            "underlying_index": "btc_usd",
                            "underlying_price": F,
                            "delta": np.nan, "gamma": np.nan,
                            "vega": np.nan, "theta": np.nan, "rho": np.nan,
                        }
                    )

    df = pd.DataFrame(rows)
    with gzip.open(path, "wt") as fh:
        df.to_csv(fh, index=False)
    return len(rows)


def build_trades_jsonl(path: Path, n=500):
    rng = np.random.default_rng(7)
    base = dt.datetime(2026, 3, 1, tzinfo=dt.timezone.utc)
    with path.open("w") as fh:
        for i in range(n):
            mark = float(rng.uniform(0.005, 0.06))
            direction = "buy" if rng.random() > 0.5 else "sell"
            edge = mark * 0.01 * (1 if direction == "buy" else -1)
            fh.write(json.dumps({
                "trade_id": f"T{i}",
                "instrument_name": f"BTC-27MAR26-{int(rng.choice([90000, 100000, 110000]))}-C",
                "timestamp": int((base + dt.timedelta(seconds=i * 30)).timestamp() * 1000),
                "price": mark + edge,
                "mark_price": mark,
                "index_price": INDEX,
                "iv": 55.0,
                "direction": direction,
                "amount": float(rng.choice([0.1, 0.5, 1.0])),
            }) + "\n")


def report(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}\n       {detail}")
    return ok


def test_free_tier_dates():
    days = free_tier_dates(2024, 2026)
    ok = all(d.day == 1 for d in days) and len(days) > 24
    return report("Free-tier date enumeration", ok,
                  f"{len(days)} days, all first-of-month: {ok}")


def test_tardis_parse_and_extract():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "chain.csv.gz"
        n = build_tardis_csv(path)

        raw = read_chain_day(path, resample_minutes=30, currency="BTC")
        norm = to_normalised(raw, index_price=INDEX)

        # Same stage 2 code as live capture, no branching.
        errs = []
        for (snap, expiry), grp in norm.groupby(["snapshot_start", "expiry"]):
            calls = grp[grp["cp"] == 1].set_index("strike")
            puts = grp[grp["cp"] == -1].set_index("strike")
            common = sorted(set(calls.index) & set(puts.index))
            fit = extract_forward(
                np.array(common),
                calls.loc[common, "mid_usd"].to_numpy(),
                puts.loc[common, "mid_usd"].to_numpy(),
                float(grp["T"].iloc[0]),
                spot=INDEX,
            )
            venue_f = float(grp["forward_hint"].iloc[0])
            errs.append(abs(fit.forward / venue_f - 1) * 1e4)

        worst = max(errs)
        ok = len(norm) > 0 and worst < 5.0
        return report(
            "Tardis parse -> normalise -> stage 2",
            ok,
            f"{n} rows in, {len(norm)} normalised, {len(errs)} expiry-snapshots, "
            f"worst forward error {worst:.2f} bps vs the file's own forward",
        )


def test_index_estimation_fallback():
    """The no-external-index fallback, and how much it actually costs."""
    T = 30 / 365
    F = INDEX * (1 + 0.08 * T)
    D = np.exp(-RATE * T)
    strikes = np.arange(85_000, 120_000, 2_500, dtype=float)

    calls, puts = [], []
    for K in strikes:
        s = 0.52 - 0.30 * np.log(K / F) + 0.90 * np.log(K / F) ** 2
        calls.append(float(black76_price(F, K, T, s, +1, D)) / INDEX)
        puts.append(float(black76_price(F, K, T, s, -1, D)) / INDEX)

    # Deliberately wrong rate assumption. In practice you do NOT know the rate;
    # that uncertainty is the whole cost of this fallback, so testing with the
    # true rate would pass trivially and tell you nothing.
    est = estimate_index_from_slope(strikes, calls, puts, assumed_rate=0.02, T=T)
    err_bps = abs(est / INDEX - 1) * 1e4

    # Propagate to what it actually costs: a scale error on every premium.
    K_atm = strikes[np.argmin(np.abs(strikes - F))]
    s_true = 0.52 - 0.30 * np.log(K_atm / F) + 0.90 * np.log(K_atm / F) ** 2
    px = float(black76_price(F, K_atm, T, s_true, +1, D))
    iv_wrong = implied_vol(px * (est / INDEX), F, K_atm, T, +1, D)
    vol_pts = abs(iv_wrong - s_true) * 100

    ok = np.isfinite(est) and err_bps < 50
    return report(
        "Index estimation from parity slope (fallback path)",
        ok,
        f"rate assumed 2.0% vs true 4.5%: index off by {err_bps:.1f} bps, "
        f"costing {vol_pts:.3f} vol points at the money",
    )


def test_trades_loader():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trades_2026-03-01.jsonl"
        build_trades_jsonl(path, 500)

        trades = load_trades(path.parent)
        liq = liquidity_map(trades, 5)
        spread = effective_spread_stats(trades)

        ok = len(trades) == 500 and not liq.empty and not spread.empty
        median_rel = trades.assign(
            rel=(trades["price"] - trades["mark_price"]).abs() / trades["mark_price"]
        )["rel"].median()
        return report(
            "Trade history loader and cost statistics",
            ok,
            f"{len(trades)} trades, {len(liq)} contracts mapped, "
            f"median distance from mark {median_rel:.2%}",
        )


if __name__ == "__main__":
    results = [
        test_free_tier_dates(),
        test_tardis_parse_and_extract(),
        test_index_estimation_fallback(),
        test_trades_loader(),
    ]
    print(f"\n{sum(results)}/{len(results)} checks passed")
