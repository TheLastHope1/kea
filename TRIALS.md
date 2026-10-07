# Research log

Every strategy variant evaluated, including the ones that were discarded. The count
feeds the Deflated Sharpe Ratio (`kea compare --trials N`): the more variants tried,
the higher the bar a result must clear before it counts as skill rather than luck.

Benchmarks (buy-and-hold, 60/40) are not trials. Parameters below are the literature
defaults in `config/default.toml` unless stated.

## ETF universe (11 ETFs + BIL, monthly)

| # | Date | Variant | Data | Notes |
|---|---|---|---|---|
| 1 | 2026-10-07 | `trend` | Nasdaq, 2016-2026, partly price-only | development run; returns understated |
| 2 | 2026-10-07 | `momentum` | same | |
| 3 | 2026-10-07 | `ml` (gbm) | same | out-of-sample AUC 0.473: no skill |
| 4 | 2026-10-07 | `ensemble` (trend + momentum + ml) | same | default |

Running total: **4**

## Crypto universe (research only, weekly)

| # | Date | Variant | Data | Notes |
|---|---|---|---|---|
| 1 | 2026-10-07 | `trend` | Binance, 2017-2026 | coin list chosen with hindsight |
| 2 | 2026-10-07 | `momentum` | same | |
| 3 | 2026-10-07 | `ml` (gbm) | same | out-of-sample AUC 0.483: no skill |
| 4 | 2026-10-07 | `ensemble` (trend + momentum + ml) | same | |
| 5 | 2026-10-07 | `ensemble` (trend + momentum) | same | |

Running total: **5**
