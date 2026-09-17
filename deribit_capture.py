"""
Stages 0 and 1: Deribit option chain capture, normalisation and filtering.

THE CONVENTION TRAP
-------------------
Deribit's BTC and ETH options are inverse: quotes are in units of the base
currency, not dollars. A quote of 0.0210 on a BTC option means 0.0210 BTC.

The USD payoff, however, is vanilla. A call pays max(S_T - K, 0) / S_T in BTC,
which converted at S_T is max(S_T - K, 0) USD. So once the premium is converted
to USD at the CURRENT index, standard Black-76 applies against the forward, and
nothing else about the pipeline needs to change.

Converting at the index and not the forward matters: the premium is paid now.
Using the forward here introduces an error of exactly the basis, which test 5
in test_synthetic.py shows is worth several volatility points of fake skew.

VERIFY THIS BEFORE TRUSTING ANY OUTPUT. Pick one liquid contract, take its
mark price from this module, and compare against the USD price Deribit shows in
its own interface. If they disagree, fix it here before building anything on
top. Do not skip this because the code looks reasonable.

USDC-settled (linear) instruments are quoted in USD already. They are detected
and excluded rather than silently mishandled.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests

API = "https://www.deribit.com/api/v2/public"
TIMEOUT = 20

# Stage 1 filter thresholds. These belong in configuration, versioned with the
# pipeline, so that a threshold change shows up as a discontinuity you can
# explain rather than a mystery you cannot.
MAX_REL_SPREAD = 1.00        # relative to mid
MIN_DAYS_TO_EXPIRY = 5.0
MIN_STRIKES_PER_SIDE = 5

INSTRUMENT_RE = re.compile(r"^(?P<ccy>[A-Z]+)-(?P<exp>\d{1,2}[A-Z]{3}\d{2})-(?P<strike>[\d.]+)-(?P<cp>[CP])$")


# --------------------------------------------------------------------------
# Stage 0: capture
# --------------------------------------------------------------------------

def fetch_chain(currency: str = "BTC") -> dict:
    """
    Single-call snapshot of the whole chain.

    get_book_summary_by_currency returns every instrument in one response,
    which keeps the snapshot window short. Per the spec that window should be
    under five seconds; one request comfortably satisfies it, which is a real
    advantage over venues that require per-instrument polling.
    """
    started = dt.datetime.now(dt.timezone.utc)
    resp = requests.get(
        f"{API}/get_book_summary_by_currency",
        params={"currency": currency, "kind": "option"},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    finished = dt.datetime.now(dt.timezone.utc)

    payload = resp.json()
    if "result" not in payload:
        raise RuntimeError(f"unexpected response: {payload}")

    return {
        "currency": currency,
        "snapshot_start": started.isoformat(),
        "snapshot_end": finished.isoformat(),
        "window_seconds": (finished - started).total_seconds(),
        "result": payload["result"],
    }


def save_raw(snapshot: dict, root: str = "data/L0") -> Path:
    """
    L0 is immutable. Write once, never edit, never overwrite.

    When you find a bug in stage 4 six months from now - and you will - this
    directory is the only thing that lets you rebuild a corrected history.
    """
    day = snapshot["snapshot_start"][:10]
    out = Path(root) / snapshot["currency"] / day
    out.mkdir(parents=True, exist_ok=True)
    stamp = snapshot["snapshot_start"].replace(":", "").replace("-", "")
    path = out / f"chain_{stamp}.json"
    path.write_text(json.dumps(snapshot))
    return path


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------

def parse_instrument(name: str):
    """BTC-26JUN26-120000-C -> (expiry date, strike, +1/-1). None if unparseable."""
    m = INSTRUMENT_RE.match(name)
    if not m:
        return None
    try:
        expiry = dt.datetime.strptime(m.group("exp"), "%d%b%y").replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None
    # Deribit settles at 08:00 UTC. Using the calendar date instead costs a
    # third of a day at the front expiry, which is where T matters most.
    expiry = expiry.replace(hour=8)
    return expiry, float(m.group("strike")), (+1 if m.group("cp") == "C" else -1)


def normalise(snapshot: dict) -> pd.DataFrame:
    """Raw payload -> tidy frame with USD prices and year fractions."""
    rows = []
    asof = dt.datetime.fromisoformat(snapshot["snapshot_start"])

    for item in snapshot["result"]:
        name = item.get("instrument_name", "")
        if "_" in name:      # USDC-settled linear instrument
            continue
        parsed = parse_instrument(name)
        if parsed is None:
            continue
        expiry, strike, cp = parsed

        index = item.get("estimated_delivery_price") or item.get("underlying_index_price")
        forward_hint = item.get("underlying_price")
        if not index or not forward_hint:
            continue

        T = (expiry - asof).total_seconds() / (365.0 * 86400.0)

        rows.append(
            {
                "instrument": name,
                "expiry": expiry,
                "T": T,
                "days_to_expiry": T * 365.0,
                "strike": strike,
                "cp": cp,
                # Quoted in base currency; convert at the index, not the forward.
                "bid_usd": (item["bid_price"] * index) if item.get("bid_price") else np.nan,
                "ask_usd": (item["ask_price"] * index) if item.get("ask_price") else np.nan,
                "mark_usd": (item["mark_price"] * index) if item.get("mark_price") else np.nan,
                "bid_native": item.get("bid_price"),
                "ask_native": item.get("ask_price"),
                "index_price": index,
                "forward_hint": forward_hint,   # Deribit's own forward, for cross-check only
                "mark_iv_venue": item.get("mark_iv"),  # ditto - never an input
                "open_interest": item.get("open_interest"),
                "volume": item.get("volume"),
                "snapshot_start": snapshot["snapshot_start"],
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["mid_usd"] = (df["bid_usd"] + df["ask_usd"]) / 2.0
    return df


# --------------------------------------------------------------------------
# Stage 1: filtering
# --------------------------------------------------------------------------

def apply_filters(df: pd.DataFrame):
    """
    Returns (kept, quarantined). Nothing is deleted - rejected quotes are
    retained with a reason code, because rejection rates by code are a leading
    indicator that something upstream has broken.
    """
    if df.empty:
        return df, df

    d = df.copy()
    d["reject"] = ""

    def mark(mask, code):
        hit = mask & (d["reject"] == "")
        d.loc[hit, "reject"] = code

    mark(d["bid_usd"].isna() | (d["bid_usd"] <= 0), "NO_BID")
    mark(d["ask_usd"].isna() | (d["ask_usd"] <= 0), "NO_ASK")
    mark(d["bid_usd"] > d["ask_usd"], "CROSSED")
    mark(d["days_to_expiry"] < MIN_DAYS_TO_EXPIRY, "NEAR_EXPIRY")
    mark(d["T"] <= 0, "EXPIRED")

    rel_spread = (d["ask_usd"] - d["bid_usd"]) / d["mid_usd"].replace(0, np.nan)
    mark(rel_spread > MAX_REL_SPREAD, "WIDE")

    # Intrinsic uses the venue's forward purely as a screen. Real intrinsic
    # testing happens after stage 2 gives us our own forward.
    intrinsic = np.maximum(d["cp"] * (d["forward_hint"] - d["strike"]), 0.0)
    mark(d["ask_usd"] < intrinsic * 0.98, "BELOW_INTRINSIC")

    kept = d[d["reject"] == ""].drop(columns=["reject"])
    quarantined = d[d["reject"] != ""]

    # An expiry too thin on either side cannot support a parity regression.
    thin = []
    for expiry, grp in kept.groupby("expiry"):
        if (grp["cp"] == +1).sum() < MIN_STRIKES_PER_SIDE or (grp["cp"] == -1).sum() < MIN_STRIKES_PER_SIDE:
            thin.append(expiry)
    if thin:
        moved = kept[kept["expiry"].isin(thin)].copy()
        moved["reject"] = "THIN_EXPIRY"
        quarantined = pd.concat([quarantined, moved], ignore_index=True)
        kept = kept[~kept["expiry"].isin(thin)]

    return kept.reset_index(drop=True), quarantined.reset_index(drop=True)


def rejection_summary(quarantined: pd.DataFrame) -> pd.Series:
    """Track this as a daily time series. A break here precedes a break elsewhere."""
    if quarantined.empty:
        return pd.Series(dtype=int)
    return quarantined["reject"].value_counts()
