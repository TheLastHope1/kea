# Research log

Every strategy variant evaluated, including the ones that were discarded. The
running total sets the multiple-testing haircut on "beats buy and hold" (the
deflated Sharpe ratio): the more variants tried, the higher the bar. Record it in
the config's `[report] trials`, or pass `kea compare --trials N`.

Benchmarks (buy-and-hold, 60/40) are not trials.

## ETF portfolios: running total 9

| # | Date | Variant | Data | Result |
|---|---|---|---|---|
| 1 | 2026-10-07 | `trend`, original universe (SPY, QQQ, IWM, EFA, EEM, VNQ, TLT, IEF, LQD, GLD, DBC; BIL) | Nasdaq 2016-2026, dividends missing for NYSE listings | development run, returns understated |
| 2 | 2026-10-07 | `momentum`, original universe | same | |
| 3 | 2026-10-07 | `ml`, original universe | same | AUC 0.473 |
| 4 | 2026-10-07 | `ensemble` (trend + momentum + ml), original universe | same | |
| 5 | 2026-10-07 | `trend`, Nasdaq universe (VONE, VTWO, VXUS, TLT, IEF, VCIT, GLD, PDBC; SHY) | Nasdaq 2016-2026, full dividends; window 2021-2026 | 1.6% a year, Sharpe 0.03 |
| 6 | 2026-10-07 | `momentum`, Nasdaq universe | same | 8.7% a year, Sharpe 0.71 |
| 7 | 2026-10-07 | `ml`, Nasdaq universe | same | AUC 0.475; 0.4% a year |
| 8 | 2026-10-07 | `ensemble` (trend + momentum), Nasdaq universe | same | 6.1% a year, Sharpe 0.56: the default |
| 9 | 2026-10-07 | `trend_filter`, 200-day average on VONE, cash in SHY | Nasdaq 2016-2026; window 2017-2026 | 4.9% a year against 14.7% |

## Century study (Kenneth French indices, 1927-2026): running total 5

| # | Date | Variant | Result |
|---|---|---|---|
| 1 | 2026-10-07 | `trend` | 8.1% a year, worst fall 54% (market 84%) |
| 2 | 2026-10-07 | `momentum` (12-1 month, must beat T-bills) | 9.4% a year, worst fall 50% |
| 3 | 2026-10-07 | `ml` | AUC 0.520, Brier worse than the base rate |
| 4 | 2026-10-07 | `trend_filter` (200-day average) | 8.9% a year, worst fall 62% |
| 5 | 2026-10-07 | `ensemble` (trend + momentum + trend_filter) | 8.9% a year, worst year -17.9% |

## Crypto (research only, Binance 2017-2026): running total 5

| # | Date | Variant | Result |
|---|---|---|---|
| 1 | 2026-10-07 | `trend` | coin list chosen with hindsight |
| 2 | 2026-10-07 | `momentum` | |
| 3 | 2026-10-07 | `ml` | AUC 0.483 |
| 4 | 2026-10-07 | `ensemble` (trend + momentum + ml) | |
| 5 | 2026-10-07 | `ensemble` (trend + momentum) | |

## Decisions

* **2026-10-07. Switched the default universe to Nasdaq-listed ETFs** (VONE, VTWO,
  VXUS, TLT, IEF, VCIT, PDBC, plus GLD, which pays no dividends) with SHY as cash.
  Yahoo blocks cloud servers, and Nasdaq only publishes dividends for Nasdaq
  listings. The exposures match the original universe; the switch was made before
  evaluating the new one.
* **2026-10-07. Took the ML forecaster out of the default ensemble.** Its
  out-of-sample AUC was 0.473-0.520 on three independent datasets, with calibration
  worse than the base rate each time.
* **2026-10-07. Changed the headline statistic** from "probability the Sharpe ratio
  beats zero" to "probability it beats buy-and-hold's". Any long-only equity
  strategy beats zero over a long sample; that measures the equity premium, not skill.
