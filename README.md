# crypto-vol-lab — Crypto Volatility Surface Research Pipeline

**Stages 0 to 5 of the volatility surface specification, plus a portfolio construction layer.** Forward extraction, implied volatility inversion, arbitrage-free SVI/SSVI calibration, and mean-variance-family allocation, for Deribit crypto options.

Build order follows Appendix A of the spec: forward extraction comes first,
because it validates the data source before anything is built on top of it.

---

## 1. Requirements

Python 3.10 or later.

```
pip install numpy scipy pandas matplotlib requests
```

No API key needed. Everything here uses Deribit's public endpoints. You do not
need an account to capture data — only to trade.

---

## 2. Quickstart

Run these in order. Do not skip step 1.

```
python test_synthetic.py         # (a) validate the numerics          -> 5/5
python test_loaders.py           # (b) validate the loaders           -> 4/4
python test_svi.py               # (c) validate calibration and gates -> 11/11
python test_portfolio.py         # (d) validate the allocation layer  -> 7/7
python run_surface.py --offline  # (e) validate the plumbing, stages 0-5
python run_surface.py BTC        # (f) live capture
```

Steps (a) to (e) need no network. Only (f) touches the API.

**(a) `test_synthetic.py`** generates option prices from known parameters, runs
them back through stages 2 and 3, and checks the parameters are recovered. If
this fails, something is wrong with your environment, not with the market.
Expect `5/5 checks passed`.

**(b) `test_loaders.py`** exercises the Tardis free-tier parser, the date
enumeration, the index-estimation fallback and the trade-history cost
statistics. Expect `4/4 checks passed`.

**(c) `test_svi.py`** checks that calendar arbitrage is detected between
crossing slices, that a generated surface is recovered, and that the SSVI joint
fit removes a crossing per-slice SVI leaves behind — reporting the cost in fit
quality rather than hiding it. Expect `11/11 checks passed`.

**(d) `test_portfolio.py`** checks the allocation layer, including a deliberate
negative control: on pure noise, mean-variance must show *no* out-of-sample
edge over equal weight while still paying turnover. A method that looks good on
noise is measuring nothing. Expect `7/7 checks passed`.

**(e) `run_surface.py --offline`** builds a payload shaped like a real Deribit
response and runs stages 0 to 5 end to end: normalise, filter, pair legs,
extract forwards, invert, calibrate, gate. This exercises every code path
except the HTTP call itself, and writes `surface.png`.

**(f) `run_surface.py BTC`** hits the live API. Substitute `ETH` for ether.
Writes an immutable raw snapshot to `data/L0/BTC/<date>/` and a plot.

---

## 3. Reading the output

```
Stage 1 rejections
NO_BID    5
WIDE      1
  kept 84 of 90 quotes
```

Rejection counts by reason code. Track these as a daily time series. A sudden
change means either the market moved or your feed broke, and you need to know
which. Rejected quotes are quarantined, never deleted.

```
Stage 2 diagnostics
   days      forward  fwd_diff_bps  implied_rate  basis_annual     r2  pairs  trimmed   ok
14.3217 100,195.8525       -0.0335        0.0459        0.0499 1.0000      4        0 True
```

| Column | What it tells you |
|---|---|
| `forward` | Your extracted forward, from put-call parity |
| `fwd_diff_bps` | Gap versus Deribit's own forward. **Should be within a few bps.** Wider means stage 2 failed for this expiry |
| `implied_rate` | Continuously compounded USD rate implied by the discount factor |
| `basis_annual` | Annualised premium of forward over index — the crypto analogue of an implied dividend |
| `r2` | Parity regression fit. **Acceptance threshold is 0.999** |
| `pairs` / `trimmed` | Strike pairs used, and how many were dropped as outliers |
| `ok` | Whether the expiry passed its acceptance criteria |

```
Stage 3: 39 OTM quotes, 0 inversion failures (0.0%; acceptance is under 1%)
Convention check vs venue mark IV: median 0.025 vol points, worst 0.333
```

A rising inversion failure rate at moderate moneyness is the signature of a bad
forward from stage 2, not a solver problem. Look upstream.

---

## 4. Verify this before trusting live output

Deribit's BTC and ETH options are **inverse**: quotes are in the base currency,
not dollars. A quote of 0.0210 means 0.0210 BTC.

`deribit_capture.normalise` converts to USD at the **index** price, not the
forward, because the premium is paid now. Getting this wrong shifts every
forward by the basis, which `test_synthetic.py` test 5 shows is worth several
volatility points of entirely fake skew.

**Do this by hand once:** pick a liquid contract, take its `mark_usd` from this
module, and compare against the USD price shown in Deribit's own interface.
Confirm they agree before building anything on top.

`run_smile.py` also runs this check automatically, comparing your inverted vols
against Deribit's published mark IV, and warns if the median gap exceeds one
volatility point. The venue's IV is validation only and is never an input.

### Stage 4 and 5 output

```
   days  n_points  rmse_vol_pts       a      b     rho       m      s  butterfly_ok  durrleman_min  lee_slack
14.3092         9        0.1388 -0.0000 0.0312 -0.6389 -0.0967 0.3367          True         0.4920     1.9488
```

| Column | What it tells you |
|---|---|
| `rmse_vol_pts` | Weighted fit error. **Acceptance is under 0.5 vol points** |
| `a b rho m s` | Raw SVI parameters. `s` is the SVI shape parameter, NOT a volatility |
| `durrleman_min` | Minimum of g(k). **Must be non-negative** — negative means negative risk-neutral density |
| `lee_slack` | Headroom on the wing bound. A fit resting at zero is being forced |

Stage 5 is a hard gate. On failure the script refits jointly under SSVI and
reports the cost in fit quality. Accept the worse fit: an arbitrage-free surface
with larger residuals is usable, a tight fit with negative density is not.

**Conventions that trip people up.** This module works in TOTAL variance
`w = sigma^2 * T`, so Lee's wing bound is `b(1+|rho|) <= 2`. Sources that
parameterise implied variance instead quote `4/T`. Also, under SSVI
`w(0, theta) = theta` exactly, so theta is the *observed* ATM total variance
rather than a fitted parameter — treating it as free turns a well-conditioned
3-parameter problem into a badly conditioned one.

---

## 5. Free historical data

Everything here costs nothing. No API key, no account, no subscription.

### 5.1 Quote snapshots — `tardis_free.py`

Tardis.dev serves the **first day of each month** without an API key, including
Deribit's full options chain. Roughly twelve real quote-level days per year,
going back several years. Enough to build and validate stages 4 and 5 now
rather than waiting months for your own archive.

```python
from tardis_free import free_tier_dates, load_free_history

dates = free_tier_dates(2023, 2026)          # first of every month
df = load_free_history(dates, currency="BTC", resample_minutes=30)
```

Confirm the free tier still stands before planning around it; the arrangement
dates back a few years.

**The index problem.** Chain quotes are in BTC and the dataset carries the
forward, not spot. Resolve it one of three ways, best first:

1. Pass `index_price` from a source you trust.
2. Join Binance BTCUSDT 1m klines via `index_from_binance_klines`. Free from
   data.binance.vision, and you need them anyway for realised variance in
   phase 4.
3. Let it fall back to `estimate_index_from_slope`. `test_loaders.py` measures
   what this costs: a 2.5 point error in the assumed rate moves the index by
   about 21 bps and the at-the-money vol by about 0.11 points. Acceptable as a
   fallback, not as a plan.

The **forward** never needs the index. In native units the parity slope is
`-D/I` and the intercept `D*F/I`, so the index cancels in `F = intercept/-slope`.

### 5.2 Trade history — `deribit_history.py`

The full options trade archive, free. Around 10GB for BTC, a couple of hours.

```
python deribit_history.py BTC 30      # last 30 days
```

Resumable — an existing day file is skipped, so an interrupted pull continues.

Trades are **sparse across strikes**, so you cannot build a surface from them.
What they are for:

- **Cost model.** Every print carries its direction, so you can measure where
  trades actually happen relative to mark. This is the empirical basis of the
  slippage assumptions phase 4 backtesting rests on. Assuming a flat haircut
  instead is how a strategy looks profitable on paper and is not.
- **Liquidity map.** Which strikes and expiries actually trade, and when.
- Volume-weighted IV as a sanity check where you have no quote data.

### 5.3 The rest of the free stack

| Need | Source | Cost |
|---|---|---|
| Live chain snapshots | Deribit public API | free |
| Historical quotes | Tardis free tier, first of month | free |
| Historical trades | Deribit History API | free |
| DVOL history (stage 6 ground truth) | CryptoDataDownload | free |
| Spot history for realised variance | data.binance.vision | free |
| Compute | Oracle Cloud free ARM tier, or any old machine | free |

---

## 6. Building your own archive

Free historical *quote* snapshots do not exist for any options market. Start
recording now; the archive compounds from day one.

```
*/30 * * * * cd /path/to/crypto-vol-lab && /usr/bin/python3 run_smile.py BTC >> capture.log 2>&1
```

Every 30 minutes, 24/7, gives roughly 48 snapshots a day against about 13 for a
cash equity market. Point it at ETH as well — same code, different argument.

`data/L0/` is immutable. Write once, never edit. When you find a bug in stage 4
six months from now, it is the only thing that lets you rebuild a corrected
history.

---

## 7. Troubleshooting

| Symptom | Likely cause |
|---|---|
| `fwd_diff_bps` in the hundreds | USD conversion wrong — see section 4 |
| Convention check warns above 1 vol point | Same |
| Many `NEAR_EXPIRY` rejections | Normal. Threshold is 5 days; short-dated numerics are unreliable |
| `r2` below 0.999 on one expiry | Stale quotes, or too few liquid pairs. Check `pairs` |
| High inversion failure rate | Bad forward upstream, not the solver |
| Most quotes rejected `WIDE` | Illiquid expiry, or you are running during a thin period |
| `non-negative slope` | Legs mispaired, or the chain is badly stale |
| HTTP 429 | Rate limited. Space out captures |

---

## 8. Why Deribit and not a free equity source

Free equity option chains (Yahoo and similar) give you the *current* chain only,
with no reliable per-quote timestamps, a delayed underlying, and no quote sizes.
Stage 2 needs option quotes and the underlying captured at the same instant; a
stale underlying against fresh option quotes produces precisely the forward
error that test 5 quantifies. There is also no history, so there is nothing to
research.

Deribit publishes full historical trade data and per-contract OHLCV for free,
along with history for DVOL, its VIX analogue — which gives you an independent
ground-truth series to validate the model-free implied variance calculation
against when you reach stage 6. It runs 24/7, has no dividends to complicate
stage 2, and is tradeable at a size you can actually afford.

The trade-offs are real: an inverse settlement convention that the textbook does
not cover, only two liquid underlyings so no cross-sectional work, and free
history that covers trades rather than quotes.

---

## 9. Portfolio construction (phase 4)

Turns per-bucket variance-risk-premium signals into target positions. Lives
after the VRP signal exists and before execution.

```
python test_portfolio.py
```

**The one rule this layer cannot enforce for you:** feed it variance-risk-premium
returns or delta-hedged P&L, NEVER raw option returns. Raw option returns make
sample variance understate the risk of short options, and any optimiser fed them
walks straight into short gamma. This is the standard way mean-variance goes
wrong on options.

Three covariance estimators (`portfolio_cov.py`): sample as the baseline,
Ledoit-Wolf shrinkage as the minimum defensible choice, and a PCA factor model
that reconstructs covariance from the level/slope/curvature factors your surface
actually lives on. Watch the condition number across them to see what shrinkage
buys.

Five allocators (`portfolio_alloc.py`): equal-weight (the benchmark), inverse-
variance, minimum-variance, risk parity, mean-variance, plus mean-CVaR for when
the payoff distribution is visibly skewed.

`portfolio_backtest.walk_forward` compares them out of sample on a rolling
window. Example output with a real signal fed through `mu_series`:

```
allocator             ann.ret  ann.vol  Sharpe  vs bench  turnover
mean_variance          29.3%     4.9%    5.92     +5.49      0.00
inverse_variance        2.1%     4.6%    0.45     +0.02      0.04
equal_weight            2.0%     4.6%    0.43     +0.00      0.00  <- benchmark
risk_parity             2.0%     4.6%    0.42     -0.00      0.03
minimum_variance        0.7%     5.0%    0.14     -0.29      0.44
```

Mean-variance wins **only because a genuine signal was supplied**. Run it with
`mu_series=None` (trailing sample mean, i.e. no real signal) and the edge
vanishes — `test_portfolio.py` asserts exactly this, averaged over 8 noise
worlds. The turnover column is not decoration: at your account size an allocator
that rebalances heavily can lose to equal-weight after costs even when its gross
Sharpe looks better. Fold your per-trade cost model in before believing any row.

Two caveats specific to your situation. Your effective breadth is ~3 (the tenor
buckets collapse onto three factors), so the optimiser has little to work with;
careful budgeting across three factors captures most of the benefit. And at $200
the continuous weights are unimplementable against Deribit minimums — treat this
as a research and risk-analysis tool, not a live position sizer, until the
account is larger.

---

## 10. Not yet implemented

Stages 6 through 8: Breeden-Litzenberger density, model-free implied variance,
constant-maturity series, factor decomposition, and the daily validation suite
(reprice-within-spread, held-out strikes, surface distance monitoring).

The signal itself: realised variance estimators (two-scale, realised kernels),
HAR forecasting, and the variance risk premium that feeds `mu_series` in the
portfolio layer. That is where edge lives; everything else is infrastructure.

Do not start stage 6 until stages 2 and 5 pass their acceptance criteria on
twenty consecutive days of live data. The portfolio layer is built but has
nothing real to allocate until the VRP signal exists.
