"""
Loader for Tardis.dev free-tier Deribit options chain data.

WHAT IS FREE
------------
Tardis distributes the FIRST DAY OF EACH MONTH without an API key, for every
dataset they carry. For Deribit that includes `options_chain`: full quote
snapshots across every listed strike, which is exactly what stages 2 to 5 need
and exactly what no free source otherwise provides.

That is roughly twelve real quote-level days per year, going back several years.
Sparse, but enough to build and validate SVI calibration and the arbitrage gates
now, instead of waiting months for your own archive to mature.

Confirm the free-tier arrangement still stands before planning around it; it
dates back a few years and terms change. If it has ended, everything else in
this module still works against paid downloads or your own recorded snapshots.

THE INDEX PROBLEM
-----------------
Deribit option prices are in BTC. Converting to USD needs the spot index, which
the options_chain dataset does not carry - it gives `underlying_price`, which is
the FORWARD for that expiry, not spot.

Three ways to resolve it, in order of preference:

  1. `index_price` argument, from a source you trust.
  2. Binance BTCUSDT klines, free and unlimited, joined on timestamp. You will
     be downloading these anyway for realised variance in phase 4, so this
     costs nothing extra. The Deribit index is a composite of major spot
     venues and Binance tracks it to a few basis points.
  3. Estimated from the parity regression slope, assuming a rate. See
     `estimate_index_from_slope`. Carries roughly a quarter vol point of error.
     A fallback, not a plan.

Note that the FORWARD itself does not need the index at all. In native units
the parity regression gives slope = -D/I and intercept = D*F/I, so the index
cancels in F = intercept / -slope. Only the price conversion needs it.
"""

from __future__ import annotations

import datetime as dt
import gzip
import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests

TARDIS_BASE = "https://datasets.tardis.dev/v1"
TIMEOUT = 120

# Tardis timestamps are microseconds since epoch.
US = 1_000_000


def free_tier_dates(start_year: int, end_year: int):
    """The days available without an API key: the first of each month."""
    out = []
    today = dt.date.today()
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            d = dt.date(y, m, 1)
            if d < today:
                out.append(d)
    return out


def download_chain_day(date: dt.date, exchange="deribit", cache_dir="data/tardis"):
    """
    Download one day of options chain snapshots. Cached; safe to re-run.

    The `OPTIONS` symbol is Tardis's aggregate covering every listed contract
    for that day.
    """
    cache = Path(cache_dir) / exchange
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"options_chain_{date:%Y-%m-%d}.csv.gz"

    if path.exists():
        return path

    url = f"{TARDIS_BASE}/{exchange}/options_chain/{date:%Y/%m/%d}/OPTIONS.csv.gz"
    resp = requests.get(url, timeout=TIMEOUT)
    if resp.status_code == 403:
        raise RuntimeError(
            f"{date} is not free-tier. Only the first day of each month is "
            f"available without an API key."
        )
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path


def read_chain_day(path, resample_minutes=30, currency="BTC"):
    """
    Parse a downloaded day into raw quote rows.

    A full day of chain snapshots is large and highly redundant. `resample_minutes`
    keeps one snapshot per interval, which is ample for surface work and keeps
    memory sane. Set to None to keep everything.
    """
    with gzip.open(path, "rb") as fh:
        df = pd.read_csv(io.BytesIO(fh.read()))

    df = df[df["symbol"].str.startswith(f"{currency}-")].copy()
    if df.empty:
        return df

    df["ts"] = pd.to_datetime(df["timestamp"], unit="us", utc=True)
    df["expiry_ts"] = pd.to_datetime(df["expiration"], unit="us", utc=True)

    if resample_minutes:
        bucket = df["ts"].dt.floor(f"{resample_minutes}min")
        # Take the last quote in each bucket for each instrument, so every
        # instrument in a snapshot is as close in time as the data allows.
        df["bucket"] = bucket
        df = df.sort_values("ts").groupby(["bucket", "symbol"], as_index=False).last()
        df["ts"] = df["bucket"]

    return df


def estimate_index_from_slope(strikes, call_native, put_native, assumed_rate=0.05, T=None):
    """
    Fallback index estimate. In native units the parity slope is -D/I, so with
    an assumed discount factor the index follows.

    Introduces roughly a quarter of a volatility point of error near the money.
    Use only when nothing better is available, and record that you used it.
    """
    K = np.asarray(strikes, float)
    y = np.asarray(call_native, float) - np.asarray(put_native, float)
    ok = np.isfinite(K) & np.isfinite(y)
    if ok.sum() < 4:
        return np.nan
    A = np.column_stack([np.ones(ok.sum()), K[ok]])
    coef, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
    slope = coef[1]
    if slope >= 0:
        return np.nan
    D = np.exp(-assumed_rate * T) if T else 1.0
    return float(D / -slope)


def index_from_binance_klines(klines: pd.DataFrame, timestamps: pd.Series):
    """
    Join a spot index onto snapshot timestamps from Binance 1m klines.

    `klines` needs columns `open_time` (UTC datetime) and `close` (float).
    Free and unlimited from data.binance.vision; you need this data anyway for
    realised variance in phase 4.
    """
    k = klines[["open_time", "close"]].sort_values("open_time")
    target = pd.DataFrame({"ts": pd.to_datetime(timestamps, utc=True)}).sort_values("ts")
    merged = pd.merge_asof(target, k, left_on="ts", right_on="open_time",
                           direction="nearest", tolerance=pd.Timedelta("5min"))
    return merged["close"].to_numpy()


def to_normalised(df: pd.DataFrame, index_price=None, assumed_rate=0.05):
    """
    Tardis rows -> the normalised frame the pipeline consumes.

    Produces the same columns as `deribit_capture.normalise`, so stages 1 to 3
    are shared between live capture and historical replay with no branching.

    `index_price` may be a scalar, an array aligned to `df`, or None to fall
    back to the slope estimate per snapshot.
    """
    if df.empty:
        return df

    d = df.copy()
    d["cp"] = np.where(d["type"].str.lower().str.startswith("c"), 1, -1)
    d["T"] = (d["expiry_ts"] - d["ts"]).dt.total_seconds() / (365.0 * 86400.0)

    if index_price is None:
        idx = np.full(len(d), np.nan)
        for (snap, exp), grp in d.groupby(["ts", "expiry_ts"]):
            calls = grp[grp["cp"] == 1].set_index("strike_price")
            puts = grp[grp["cp"] == -1].set_index("strike_price")
            common = sorted(set(calls.index) & set(puts.index))
            if len(common) < 4:
                continue
            est = estimate_index_from_slope(
                common,
                calls.loc[common, "mark_price"].to_numpy(),
                puts.loc[common, "mark_price"].to_numpy(),
                assumed_rate,
                float(grp["T"].iloc[0]),
            )
            if np.isfinite(est):
                idx[d.index.isin(grp.index)] = est
        d["index_price"] = idx
        d["index_source"] = "slope_estimate"
    else:
        d["index_price"] = index_price
        d["index_source"] = "external"

    out = pd.DataFrame(
        {
            "instrument": d["symbol"],
            "expiry": d["expiry_ts"],
            "T": d["T"],
            "days_to_expiry": d["T"] * 365.0,
            "strike": d["strike_price"].astype(float),
            "cp": d["cp"],
            "bid_usd": d["bid_price"] * d["index_price"],
            "ask_usd": d["ask_price"] * d["index_price"],
            "mark_usd": d["mark_price"] * d["index_price"],
            "bid_native": d["bid_price"],
            "ask_native": d["ask_price"],
            "index_price": d["index_price"],
            "index_source": d["index_source"],
            "forward_hint": d["underlying_price"],
            "mark_iv_venue": d["mark_iv"],
            "open_interest": d.get("open_interest"),
            "volume": np.nan,
            "snapshot_start": d["ts"].astype(str),
        }
    )
    out["mid_usd"] = (out["bid_usd"] + out["ask_usd"]) / 2.0
    return out.reset_index(drop=True)


def load_free_history(dates, currency="BTC", resample_minutes=30,
                      index_lookup=None, cache_dir="data/tardis"):
    """
    Download and normalise a set of free-tier days.

    `index_lookup` is an optional callable taking a timestamp Series and
    returning index prices, e.g. a partial of `index_from_binance_klines`.
    """
    frames = []
    for date in dates:
        try:
            path = download_chain_day(date, cache_dir=cache_dir)
        except Exception as exc:
            print(f"  {date}: skipped ({exc})")
            continue
        raw = read_chain_day(path, resample_minutes, currency)
        if raw.empty:
            continue
        idx = index_lookup(raw["ts"]) if index_lookup else None
        frames.append(to_normalised(raw, index_price=idx))
        print(f"  {date}: {len(raw)} quotes across "
              f"{raw['ts'].nunique()} snapshots")

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
