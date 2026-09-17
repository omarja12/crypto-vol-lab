"""
End-to-end run: capture -> filter -> forward extraction -> implied vols -> smile.

    python run_smile.py BTC            live capture from Deribit
    python run_smile.py --offline      synthetic payload, no network

Offline mode builds a payload with the same shape Deribit returns, so the
normalisation, filtering, pairing and extraction paths are all exercised
without a connection. Use it to check the plumbing, then switch to live.
"""

from __future__ import annotations

import sys
import datetime as dt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bs76 import black76_price, implied_vol
from deribit_capture import fetch_chain, save_raw, normalise, apply_filters, rejection_summary
from stage2_forward import extract_forward, basis_check


# --------------------------------------------------------------------------
# Stages 2 and 3 over a whole chain
# --------------------------------------------------------------------------

def process_expiry(grp: pd.DataFrame):
    """One expiry: pair legs, extract the forward, invert the OTM wing."""
    calls = grp[grp["cp"] == +1].set_index("strike")
    puts = grp[grp["cp"] == -1].set_index("strike")
    common = sorted(set(calls.index) & set(puts.index))
    if len(common) < 4:
        return None, None

    T = float(grp["T"].iloc[0])
    index_price = float(grp["index_price"].iloc[0])

    fit = extract_forward(
        strikes=np.array(common),
        call_prices=calls.loc[common, "mid_usd"].to_numpy(),
        put_prices=puts.loc[common, "mid_usd"].to_numpy(),
        T=T,
        spot=index_price,
    )
    if not np.isfinite(fit.forward):
        return fit, None

    # Stage 3: OTM only. Calls above the forward, puts below.
    out = []
    for _, row in grp.iterrows():
        otm = (row["cp"] == +1 and row["strike"] > fit.forward) or (
            row["cp"] == -1 and row["strike"] < fit.forward
        )
        if not otm:
            continue
        iv_mid = implied_vol(row["mid_usd"], fit.forward, row["strike"], T, row["cp"], fit.discount)
        iv_bid = implied_vol(row["bid_usd"], fit.forward, row["strike"], T, row["cp"], fit.discount)
        iv_ask = implied_vol(row["ask_usd"], fit.forward, row["strike"], T, row["cp"], fit.discount)
        out.append(
            {
                "expiry": row["expiry"],
                "T": T,
                "strike": row["strike"],
                "cp": row["cp"],
                "forward": fit.forward,
                # Fitting coordinates. All downstream work lives here.
                "k": np.log(row["strike"] / fit.forward),
                "iv_mid": iv_mid,
                "iv_bid": iv_bid,
                "iv_ask": iv_ask,
                "iv_spread": (iv_ask - iv_bid) if np.isfinite(iv_ask) and np.isfinite(iv_bid) else np.nan,
                "w": iv_mid**2 * T if np.isfinite(iv_mid) else np.nan,
                "iv_venue": row["mark_iv_venue"] / 100.0 if row["mark_iv_venue"] else np.nan,
            }
        )

    return fit, pd.DataFrame(out)


def run(df: pd.DataFrame):
    kept, quarantined = apply_filters(df)

    print("Stage 1 rejections")
    summary = rejection_summary(quarantined)
    print(summary.to_string() if not summary.empty else "  none")
    print(f"  kept {len(kept)} of {len(df)} quotes\n")

    fits, surface = [], []
    for expiry, grp in kept.groupby("expiry"):
        fit, ivs = process_expiry(grp)
        if fit is None:
            continue
        basis = basis_check(fit, float(grp["index_price"].iloc[0]), float(grp["T"].iloc[0]))
        venue_fwd = float(grp["forward_hint"].iloc[0])
        fits.append(
            {
                "expiry": expiry,
                "days": float(grp["days_to_expiry"].iloc[0]),
                "forward": fit.forward,
                "venue_forward": venue_fwd,
                # Cross-check. Disagreement beyond a few bps means stage 2
                # failed for this expiry and everything downstream is suspect.
                "fwd_diff_bps": (fit.forward / venue_fwd - 1) * 1e4 if venue_fwd else np.nan,
                "discount": fit.discount,
                "implied_rate": fit.implied_rate,
                "basis_annual": basis,
                "r2": fit.r_squared,
                "pairs": fit.n_pairs_used,
                "trimmed": fit.n_pairs_trimmed,
                "ok": fit.ok,
            }
        )
        if ivs is not None and not ivs.empty:
            surface.append(ivs)

    fits_df = pd.DataFrame(fits).sort_values("days")
    surface_df = pd.concat(surface, ignore_index=True) if surface else pd.DataFrame()

    print("Stage 2 diagnostics")
    cols = ["days", "forward", "fwd_diff_bps", "implied_rate", "basis_annual", "r2", "pairs", "trimmed", "ok"]
    print(fits_df[cols].to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    if not surface_df.empty:
        n_fail = int(surface_df["iv_mid"].isna().sum())
        print(f"\nStage 3: {len(surface_df)} OTM quotes, {n_fail} inversion failures "
              f"({n_fail / len(surface_df):.1%}; acceptance is under 1%)")

        # Independent check against the venue's own IV. Never an input to the
        # pipeline, only a validation that the conventions were handled right.
        both = surface_df.dropna(subset=["iv_mid", "iv_venue"])
        if not both.empty:
            err = (both["iv_mid"] - both["iv_venue"]).abs()
            print(f"Convention check vs venue mark IV: median {err.median() * 100:.3f} "
                  f"vol points, worst {err.max() * 100:.3f}")
            if err.median() > 0.01:
                print("  WARNING: median above 1 vol point. Suspect the USD conversion "
                      "in deribit_capture.normalise before going further.")

    return fits_df, surface_df


def plot_smiles(surface_df: pd.DataFrame, path="smiles.png", max_expiries=4):
    if surface_df.empty:
        return None
    expiries = sorted(surface_df["expiry"].unique())[:max_expiries]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for e in expiries:
        s = surface_df[(surface_df["expiry"] == e)].dropna(subset=["iv_mid"]).sort_values("k")
        if s.empty:
            continue
        days = s["T"].iloc[0] * 365
        ax.plot(s["k"], s["iv_mid"] * 100, marker="o", ms=3, label=f"{days:.0f}d")
        ax.fill_between(s["k"], s["iv_bid"] * 100, s["iv_ask"] * 100, alpha=0.15)
    ax.axvline(0, color="k", lw=0.6, alpha=0.4)
    ax.set_xlabel("log-moneyness  k = ln(K/F)")
    ax.set_ylabel("implied volatility (%)")
    ax.set_title("Deribit smiles, forward extracted by put-call parity")
    ax.legend(title="expiry")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    return path


# --------------------------------------------------------------------------
# Offline payload, shaped like the real API response
# --------------------------------------------------------------------------

def synthetic_snapshot():
    now = dt.datetime.now(dt.timezone.utc)
    index = 100_000.0
    result = []

    for days, base_vol, basis in [(14, 0.48, 0.05), (35, 0.52, 0.08), (70, 0.56, 0.10)]:
        expiry = (now + dt.timedelta(days=days)).replace(hour=8, minute=0, second=0, microsecond=0)
        T = (expiry - now).total_seconds() / (365 * 86400)
        F = index * (1 + basis * T)
        D = np.exp(-0.045 * T)
        tag = expiry.strftime("%d%b%y").upper().lstrip("0")

        for K in np.arange(70_000, 145_000, 5_000, dtype=float):
            k = np.log(K / F)
            sigma = base_vol - 0.30 * k + 0.90 * k**2
            for cp, letter in ((+1, "C"), (-1, "P")):
                usd = float(black76_price(F, K, T, sigma, cp, D))
                native = usd / index
                spread = max(native * 0.02, 0.0005)
                result.append(
                    {
                        "instrument_name": f"BTC-{tag}-{int(K)}-{letter}",
                        "bid_price": round(native - spread / 2, 4),
                        "ask_price": round(native + spread / 2, 4),
                        "mark_price": native,
                        "mark_iv": sigma * 100,
                        "underlying_price": F,
                        "estimated_delivery_price": index,
                        "open_interest": 100,
                        "volume": 10,
                    }
                )

    return {
        "currency": "BTC",
        "snapshot_start": now.isoformat(),
        "snapshot_end": now.isoformat(),
        "window_seconds": 0.0,
        "result": result,
    }


if __name__ == "__main__":
    offline = "--offline" in sys.argv
    if offline:
        print("OFFLINE MODE: synthetic payload, no network\n")
        snap = synthetic_snapshot()
    else:
        currency = next((a for a in sys.argv[1:] if not a.startswith("-")), "BTC")
        snap = fetch_chain(currency)
        print(f"captured {len(snap['result'])} instruments in "
              f"{snap['window_seconds']:.2f}s -> {save_raw(snap)}\n")

    df = normalise(snap)
    fits_df, surface_df = run(df)
    out = plot_smiles(surface_df)
    if out:
        print(f"\nsmiles written to {out}")
