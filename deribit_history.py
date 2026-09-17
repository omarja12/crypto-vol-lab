"""
Scraper for Deribit's History API: full historical options trade data, free.

Deribit serves every trade ever printed on its options market through
history.deribit.com, with no API key and no charge. Expect roughly 10GB for the
complete BTC options history and a couple of hours to pull it.

WHAT THIS IS AND IS NOT GOOD FOR
--------------------------------
Trades are sparse across strikes. On any given minute only a handful of the
several hundred listed contracts have printed, so you CANNOT build a clean
quote-based surface from this. Stages 2 to 5 need quotes; use tardis_free.py
or your own recorded snapshots for those.

What trade data is good for:

  - Realised trading costs. Every print carries its direction, so you can
    measure where trades actually happen relative to mid. This is the empirical
    basis of the cost model that phase 4 backtesting depends on, and it is
    otherwise pure guesswork.
  - Liquidity mapping. Which strikes and expiries actually trade, by time of
    day. Determines what is realistically tradeable at your size.
  - Volume-weighted implied vol by bucket, as a sanity check on your fitted
    surface in periods where you have no quote data.
  - Flow analysis: trade direction against subsequent surface moves.

Each trade carries `index_price` and Deribit's own `iv`, so the rows are
self-contained and need no external join for USD conversion.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import pandas as pd
import requests

HISTORY_API = "https://history.deribit.com/api/v2/public"
TIMEOUT = 30

# The generic API documents a 1000 row cap; the history host has been observed
# accepting 10000. Start high and fall back rather than assuming either.
PAGE_SIZE = 10_000
FALLBACK_PAGE_SIZE = 1_000

# Deribit's public rate limits are generous but not unlimited. This pacing has
# no trouble; raise it if you get 429s.
SLEEP_BETWEEN_PAGES = 0.15


def _get(endpoint, params):
    resp = requests.get(f"{HISTORY_API}/{endpoint}", params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()
    if "error" in payload:
        raise RuntimeError(payload["error"])
    return payload["result"]


def fetch_trades_window(currency, start, end, kind="option", page_size=PAGE_SIZE):
    """
    All trades in [start, end). Paginates forward by timestamp.

    Timestamps are milliseconds since epoch, as Deribit expects them.
    """
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    out, cursor, size = [], start_ms, page_size
    while cursor < end_ms:
        try:
            result = _get(
                "get_last_trades_by_currency_and_time",
                {
                    "currency": currency,
                    "kind": kind,
                    "start_timestamp": cursor,
                    "end_timestamp": end_ms,
                    "count": size,
                    "sorting": "asc",
                    "include_old": "true",
                },
            )
        except requests.HTTPError as exc:
            if size > FALLBACK_PAGE_SIZE:
                size = FALLBACK_PAGE_SIZE      # host rejected the page size
                continue
            raise exc

        trades = result.get("trades", [])
        if not trades:
            break

        out.extend(trades)
        last_ts = trades[-1]["timestamp"]

        # Advance past the last timestamp. Trades share milliseconds, so a
        # naive cursor of last_ts loops forever; +1 can in principle drop a
        # simultaneous trade, which is why duplicates are removed by trade_id
        # at the end rather than assumed absent.
        if last_ts <= cursor:
            cursor += 1
        else:
            cursor = last_ts

        if not result.get("has_more", len(trades) == size):
            break

        time.sleep(SLEEP_BETWEEN_PAGES)

    return out


def fetch_trades_by_day(currency, start_date, end_date, kind="option",
                        out_dir="data/L0_trades"):
    """
    Day by day, written to JSONL as it goes.

    Resumable: an existing file for a day is left alone, so an interrupted pull
    picks up where it stopped. Written under L0 because it is raw and immutable.
    """
    root = Path(out_dir) / currency
    root.mkdir(parents=True, exist_ok=True)

    day = start_date
    while day < end_date:
        path = root / f"trades_{day:%Y-%m-%d}.jsonl"
        if path.exists():
            day += dt.timedelta(days=1)
            continue

        start = dt.datetime.combine(day, dt.time.min, tzinfo=dt.timezone.utc)
        end = start + dt.timedelta(days=1)
        trades = fetch_trades_window(currency, start, end, kind)

        with path.open("w") as fh:
            for t in trades:
                fh.write(json.dumps(t) + "\n")
        print(f"  {day}: {len(trades)} trades")

        day += dt.timedelta(days=1)

    return root


def load_trades(path_or_dir) -> pd.DataFrame:
    """Read scraped JSONL into a tidy frame with USD prices."""
    p = Path(path_or_dir)
    files = sorted(p.glob("*.jsonl")) if p.is_dir() else [p]

    rows = []
    for f in files:
        with f.open() as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).drop_duplicates(subset=["trade_id"])
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    # Trades carry their own index price, so no external join is needed.
    df["price_usd"] = df["price"] * df["index_price"]
    df["iv"] = df.get("iv", pd.Series(index=df.index, dtype=float)) / 100.0
    return df.sort_values("ts").reset_index(drop=True)


def effective_spread_stats(trades: pd.DataFrame):
    """
    Where trades actually print relative to mark, by direction.

    This is the empirical input to the cost model. Backtesting against an
    assumed haircut instead is how strategies that look profitable on paper
    turn out not to be.
    """
    if trades.empty or "mark_price" not in trades:
        return pd.DataFrame()

    d = trades.copy()
    # Signed cost in volatility-neutral terms is hard; in price terms it is not.
    d["slippage_native"] = (d["price"] - d["mark_price"]) * d["direction"].map(
        {"buy": 1.0, "sell": -1.0}
    )
    d["slippage_usd"] = d["slippage_native"] * d["index_price"]
    # Relative to the option's own price, which is the comparable measure
    # across strikes of wildly different premium.
    d["slippage_rel"] = d["slippage_native"] / d["mark_price"].replace(0, float("nan"))

    return d.groupby("direction")[["slippage_usd", "slippage_rel"]].describe()


def liquidity_map(trades: pd.DataFrame, top_n=20):
    """Which contracts actually trade. Determines what you can realistically hold."""
    if trades.empty:
        return pd.DataFrame()
    g = (
        trades.groupby("instrument_name")
        .agg(trades=("trade_id", "count"),
             volume=("amount", "sum"),
             notional_usd=("price_usd", "sum"))
        .sort_values("trades", ascending=False)
    )
    return g.head(top_n)


if __name__ == "__main__":
    import sys

    currency = sys.argv[1] if len(sys.argv) > 1 else "BTC"
    days_back = int(sys.argv[2]) if len(sys.argv) > 2 else 7

    end = dt.date.today()
    start = end - dt.timedelta(days=days_back)
    print(f"fetching {currency} option trades, {start} to {end}")
    root = fetch_trades_by_day(currency, start, end)

    trades = load_trades(root)
    print(f"\n{len(trades)} trades loaded")
    if not trades.empty:
        print("\nMost traded contracts:")
        print(liquidity_map(trades, 10).to_string())
