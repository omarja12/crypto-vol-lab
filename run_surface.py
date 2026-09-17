"""
Full run: capture -> filter -> forward -> implied vols -> SVI -> arbitrage gates.

    python run_surface.py --offline    synthetic payload, no network
    python run_surface.py BTC          live capture

Stage 5 is a HARD GATE. If the per-slice fit produces butterfly arbitrage or
calendar crossings, the script refits jointly under SSVI and reports what that
cost in fit quality. It does not pass an arbitrageable surface downstream.
"""

from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deribit_capture import fetch_chain, normalise, save_raw
from run_smile import run, synthetic_snapshot
from ssvi import fit_ssvi, ssvi_slice_to_svi
from svi import (
    calibrate_svi,
    check_calendar,
    durrleman_g,
    slice_report,
    svi_vol,
)


def fit_surface(surface_df: pd.DataFrame):
    """Stage 4 across every expiry, then the stage 5 gates."""
    if surface_df.empty:
        return pd.DataFrame(), {}, []

    rows, fits, ssvi_input = [], [], []

    for expiry, grp in surface_df.groupby("expiry"):
        g = grp.dropna(subset=["iv_mid"]).sort_values("k")
        if len(g) < 5:
            continue

        T = float(g["T"].iloc[0])
        k = g["k"].to_numpy()
        iv = g["iv_mid"].to_numpy()
        spread = g["iv_spread"].to_numpy()

        fit = calibrate_svi(k, iv, T, iv_spread=spread)
        rep = slice_report(fit, k.min(), k.max())
        rep["expiry"] = expiry
        rep["days"] = T * 365
        rows.append(rep)

        if fit.converged:
            fits.append((T, fit.params))
            ssvi_input.append((T, k, iv, 1.0 / np.maximum(spread, 1e-4) ** 2))

    slices_df = pd.DataFrame(rows)

    cal_ok, cal_worst, cal_k, cal_pair = check_calendar(fits)
    bf_ok = bool(slices_df["butterfly_ok"].all()) if "butterfly_ok" in slices_df else False

    gates = {
        "butterfly_ok": bf_ok,
        "calendar_ok": cal_ok,
        "calendar_worst": cal_worst,
        "calendar_worst_k": cal_k,
        "calendar_pair": cal_pair,
        "passed": bool(bf_ok and cal_ok),
    }

    return slices_df, gates, ssvi_input


def remediate(ssvi_input):
    """Joint SSVI refit. Arbitrage-free by construction, at a cost in fit."""
    params, diag = fit_ssvi(ssvi_input)
    fits = [
        (T, ssvi_slice_to_svi(th, params.rho, params.eta, params.gamma))
        for T, th in zip(params.Ts, params.thetas)
    ]
    cal_ok, cal_worst, _, _ = check_calendar(fits)
    diag["calendar_ok_after"] = cal_ok
    diag["calendar_worst_after"] = cal_worst
    return params, diag, fits


def plot_surface(surface_df, slices_df, path="surface.png", max_expiries=4):
    """Fitted smiles over the quotes, plus Durrleman's function beneath."""
    if surface_df.empty or slices_df.empty:
        return None

    expiries = sorted(slices_df[slices_df["converged"]]["expiry"].unique())[:max_expiries]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [2.2, 1]})

    from svi import SVIParams

    for e in expiries:
        row = slices_df[slices_df["expiry"] == e].iloc[0]
        q = surface_df[surface_df["expiry"] == e].dropna(subset=["iv_mid"]).sort_values("k")
        if q.empty:
            continue

        p = SVIParams(row["a"], row["b"], row["rho"], row["m"], row["s"])
        T = float(row["T"])
        grid = np.linspace(q["k"].min() - 0.05, q["k"].max() + 0.05, 300)

        line, = ax1.plot(grid, svi_vol(grid, p, T) * 100, lw=1.6,
                         label=f"{row['days']:.0f}d  (rmse {row['rmse_vol_pts']:.2f}pt)")
        ax1.scatter(q["k"], q["iv_mid"] * 100, s=12, alpha=0.65, color=line.get_color())
        ax2.plot(grid, durrleman_g(grid, p), lw=1.3, color=line.get_color())

    ax1.axvline(0, color="k", lw=0.6, alpha=0.35)
    ax1.set_ylabel("implied volatility (%)")
    ax1.set_title("SVI fit over quotes, with Durrleman's condition below")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.25)

    ax2.axhline(0, color="crimson", lw=1.0)
    ax2.axvline(0, color="k", lw=0.6, alpha=0.35)
    ax2.set_xlabel("log-moneyness  k = ln(K/F)")
    ax2.set_ylabel("g(k)")
    ax2.set_title("g(k) < 0 means negative density: butterfly arbitrage", fontsize=9)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    return path


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

    print("\n" + "=" * 70)
    print("Stage 4: SVI calibration")
    print("=" * 70)

    slices_df, gates, ssvi_input = fit_surface(surface_df)
    if slices_df.empty:
        print("no expiry had enough points to fit")
        sys.exit(0)

    cols = ["days", "n_points", "rmse_vol_pts", "a", "b", "rho", "m", "s",
            "butterfly_ok", "durrleman_min", "lee_slack"]
    print(slices_df[[c for c in cols if c in slices_df]]
          .to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    print("\n" + "=" * 70)
    print("Stage 5: arbitrage gates")
    print("=" * 70)
    print(f"  butterfly (Durrleman g >= 0 everywhere): "
          f"{'PASS' if gates['butterfly_ok'] else 'FAIL'}")
    print(f"  calendar (w non-decreasing in T):        "
          f"{'PASS' if gates['calendar_ok'] else 'FAIL'}"
          f"   worst {gates['calendar_worst']:.6f}")

    if not gates["passed"]:
        print("\n  GATE FAILED -> refitting jointly under SSVI")
        params, diag, _ = remediate(ssvi_input)
        print(f"  rho {diag['rho']:.4f}  eta {diag['eta']:.4f}  gamma {diag['gamma']:.4f}")
        print(f"  arbitrage-free: {diag['arbitrage_free']}   "
              f"calendar-free: {diag['calendar_ok_after']}")
        print(f"  cost: mean RMSE now {diag['mean_rmse_vol_pts']:.3f} vol points")
        print("  Accept the worse fit. An arbitrageable surface is not usable.")
    else:
        print("\n  Both gates passed. Surface is safe to derive from.")

    out = plot_surface(surface_df, slices_df)
    if out:
        print(f"\nsurface written to {out}")
