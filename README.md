# vrp-research

**Status: work in progress (Phase 1 of 7). Results below are placeholders until the final phase; nothing here is a finished claim yet.**

## Question

Everyone knows selling S&P 500 variance earns a premium on average — the variance risk premium (VRP) is the persistently positive spread between risk-neutral expected variance (VIX²) and subsequently realized variance, and it exists because short-vol positions lose badly in crashes. The question this repo asks is narrower and harder: **does a realized-volatility forecast add economic value on top of naively always being short vol, after realistic option transaction costs?** A rigorous "no" is an acceptable answer and will be reported as such.

## Components

1. **RV forecasting engine** — HAR-RV family, GARCH family, and a LightGBM check, against non-optional baselines (random walk, expanding mean, EWMA λ=0.94, and the VIX itself), evaluated strictly walk-forward with QLIKE/MSE, Diebold–Mariano tests (HAC), Mincer–Zarnowitz regressions, and a Model Confidence Set.
2. **VRP strategy study** — a forecast-conditioned synthetic 30-day variance swap vs an always-short benchmark, with an explicit vol-point half-spread cost model and a breakeven-cost analysis as the headline chart.

## Data

| Series | Source | Notes |
|---|---|---|
| VIX, VIX9D, VIX3M, VIX6M daily OHLC | CBOE (`cdn.cboe.com`, free) | VIX from 1990 under the current (2003) methodology; the Oct-2014 addition of SPX weeklys is noted as a regime caveat, not a series break |
| S&P 500 (^GSPC), SPY daily OHLCV | Yahoo Finance v8 chart API | 1985+ pulled, sample starts 1990 with VIX |
| S&P 500 (^SPX) daily | Stooq | independent source used only to cross-check closes |

Realized variance is computed from **daily** data. The gold-standard 5-minute RV (Oxford-Man realized library) was discontinued; if a usable mirror is found it will be used to validate that conclusions do not flip. The payoff-relevant estimator is the sum of squared **close-to-close** daily log returns — deliberately, because that is how a variance swap's floating leg is defined. Range-based estimators (Parkinson, Garman–Klass, Rogers–Satchell, Yang–Zhang) are computed as conditioning features and robustness checks; they exclude (or, for YZ, separately model) the overnight gap and are documented as such. This choice and its limitations are disclosed prominently; daily-based RV is noisier than intraday RV and that noise is why QLIKE (robust to proxy noise) is the primary loss.

Raw pulls are cached to `data/raw/` with a `manifest.json` recording URL, sha256, row counts, and timestamps. `src/` never touches the network.

## Reproduce

```bash
uv sync                          # pinned environment (uv.lock committed)
python3 scripts/download_raw.py  # stdlib-only, writes data/raw/ + manifest
make all                         # every number and figure in this README
make test lint                   # pytest, ruff, mypy
```

CI runs lint, type-checks, and the (hermetic, no-network) test suite on Python 3.11/3.12.

## Honesty infrastructure

- Walk-forward evaluation only; a final holdout period is never touched until the very end (the README will state how many times it was evaluated).
- Every configuration evaluated is logged; the total count feeds a deflated Sharpe ratio for the headline result.
- Lookahead audit: unit tests assert that the target at date *t* is computed strictly from *t+1..t+22* and is immune to poisoning of all data ≤ *t* (`tests/test_alignment.py`).
- All standard errors for overlapping 22-day horizons use HAC/Newey–West corrections.
- Sharpe is never reported without drawdown and CVaR; a short-vol strategy's Sharpe is misleading in isolation.

*(Full results, the breakeven-cost chart, crisis event studies, and the limitations section land in Phase 7.)*
