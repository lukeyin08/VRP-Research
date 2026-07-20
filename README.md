# Volatility forecasting and the variance risk premium

**Question.** Everyone knows selling S&P 500 variance earns a premium on average. The narrower question this repo answers: *does a realized-volatility forecast add economic value on top of naively always being short vol, after realistic option transaction costs?*

**Answer, honestly.** The variance risk premium itself is real and survives realistic costs: always-short earns a net Sharpe of **0.47** on the 2000–2018 development period and **0.19** on the untouched 2019–2026 holdout at a 0.5 vol-point half-spread. The forecast layer raises gross Sharpe (1.02 vs 0.74) and its best variant beats always-short net in *point estimate* in both periods (0.64 vs 0.47 dev; 0.49 vs 0.19 holdout) — but that increment is **not statistically significant** (HAC p = 0.42 dev, p = 0.34 holdout), and its deflated Sharpe ratio against the 66 configurations this project evaluated is a coin flip (DSR = 0.51). What the forecast *demonstrably* buys is not average return but tail shape: on the holdout the conditioned strategy's max drawdown was −17.7% vs −34.2% for always-short, skew −3.3 vs −7.5, and it sidestepped the August 2024 yen-carry unwind entirely (+1% vs −19%). Conclusion: **the VRP is harvestable; the claim that this forecast adds *alpha* on top is not defensible in this sample; its defensible contribution is risk control.**

Everything below is walk-forward, evaluated out of sample, with the final holdout touched exactly once.

---

## 1. Data

| Series | Source | Notes |
|---|---|---|
| VIX, VIX9D, VIX3M, VIX6M daily OHLC | CBOE (`cdn.cboe.com`, free) | VIX from 1990-01-02 under the current (2003) methodology; the Oct-2014 inclusion of SPX weeklys is a regime caveat, not a series break |
| S&P 500 (^GSPC), SPY daily OHLCV | Yahoo v8 chart API | pulled from 1985; joint sample starts 1990 with VIX |
| S&P 500 (^SPX) | Stooq | independent close cross-check only (optional; degrades gracefully) |

`python3 scripts/download_raw.py` (stdlib-only) writes `data/raw/` with a manifest recording URL, sha256, row counts, and timestamps. `src/` never touches the network. 9,198 joint trading days, 1990-01-02 → 2026-07-17.

**Realized variance is computed from daily data.** The gold standard is 5-minute RV; the Oxford-Man realized library that provided it free was discontinued, and free intraday history is limited to a ~60-day yfinance window, so this project runs on daily data and says so. Two deliberate choices follow:

- The forecast target and strategy payoff use **close-to-close squared daily log returns** — precisely because a variance swap's floating leg is defined on close-to-close returns. The proxy is noisy but unbiased *and* payoff-exact.
- **QLIKE is the primary loss** because it is robust to noise in the volatility proxy (Patton 2011): model rankings under QLIKE with a conditionally unbiased noisy proxy are consistent with rankings against true variance, which MSE's are not.

Range-based estimators (Parkinson, Garman–Klass, Rogers–Satchell, Yang–Zhang) are computed as conditioning features and cross-checks only. Data archaeology the validation layer surfaced, all handled explicitly in code:

- CBOE's own published VIX history contains rows violating OHLC relations (e.g. 1992-02-11 open 19.24 > high 18.57; 47 rows in VIX, 1 each in VIX3M/6M). We consume closes only; those unused fields are counted warnings, not silently accepted or silently dropped.
- Yahoo ^GSPC **opens are synthetic (= prior close) on 50–98% of days per year through 2005**, clean only from ~2008. Open-dependent estimators (GK/RS/YZ) are therefore untrustworthy early; every row carries a `spx_open_synthetic` flag, and the ML feature set uses only Parkinson (high/low) among range estimators. Parkinson vs close-to-close means (14.6 vs 18.0 vol pts) line up with the expected overnight-variance gap.
- Only calendar anomaly: the 9/11 closure. 33 VIX-only and 4 SPX-only dates dropped explicitly. **Nothing is forward-filled anywhere.**
- SPX closed exactly flat on 5 days in 36 years; log-space transforms floor at quote resolution ((0.32 vol pts)² annualized, disclosed), and QLIKE evaluation floors the target at (0.1 vol pt)² — binding on 44 of 4,757 dev observations at h=1, zero at h=22.

Sanity check the whole project rests on — implied vs subsequently realized vol, and the ex-post VRP (mean **+0.0113** variance units, positive on **85.6%** of days, positive every decade):

![VIX vs realized](reports/figures/phase1_vix_vs_forward_rv.png)
![Ex-post VRP](reports/figures/phase1_vrp_expost.png)

## 2. Evaluation protocol

- **Walk-forward, expanding window,** refit every 22 trading days. A training row whose 22-day target window is not fully observed by the forecast date is excluded (`position(s) ≤ position(t) − 22`); unit tests poison the future and assert bit-identical forecasts.
- **Development period 2000–2018; final holdout 2019 → 2026-06.** Dev evaluation dates are trimmed so no dev target window even overlaps the holdout. A `holdout_guard` raises if any Phase 2–6 code touches it. The holdout was scored by the frozen pipeline exactly once, in Phase 7; subsequent re-runs regenerate that same evaluation (the only changes after first holdout contact were figure cosmetics).
- Every standard error and test accounts for the overlapping-window autocorrelation that 22-day targets on daily data create: Diebold–Mariano with Newey–West long-run variance at lag 44, Mincer–Zarnowitz with HAC(44), Model Confidence Set with block bootstrap (block 44).
- All 66 model/strategy configurations ever evaluated are logged in `reports/config_log.csv` and feed the deflated Sharpe ratio (Bailey & López de Prado).

## 3. Forecasting results (h = 22 days, annualized variance)

Development period (n = 4,757), sorted by QLIKE; DM test vs HAR(levels):

| model | QLIKE | MSE ×1e4 | OOS R² vs exp. mean | MZ β | MZ p(α=0,β=1) | DM p vs HAR |
|---|---|---|---|---|---|---|
| **GJR-GARCH(1,1)** | **0.2544** | 19.77 | 0.556 | 0.91 | 0.42 | **0.026** |
| HAR-log-IV | 0.2571 | 21.96 | 0.506 | 1.07 | 0.17 | 0.107 |
| GJR-t | 0.2615 | 20.66 | 0.535 | 0.84 | 0.36 | 0.190 |
| HAR-IV | 0.2659 | 24.75 | 0.444 | 0.96 | 0.09 | 0.215 |
| EGARCH | 0.2683 | 22.69 | 0.490 | 1.49 | 0.19 | 0.346 |
| GARCH(1,1) | 0.2696 | 22.56 | 0.493 | 0.84 | 0.34 | 0.133 |
| VIX (market's own forecast) | 0.2751 | 22.83 | 0.487 | 0.89 | **0.0002** | 0.503 |
| HAR (levels) | 0.2877 | 23.63 | 0.469 | 0.95 | 0.394 | — |
| HAR (log) | 0.2897 | 22.98 | 0.483 | 1.24 | 0.52 | 0.873 |
| HAR-J | 0.2910 | 23.38 | 0.475 | 0.86 | 0.22 | 0.338 |
| HAR-CJ | 0.2928 | 23.43 | 0.473 | 0.87 | 0.22 | 0.172 |
| LightGBM | 0.3175 | 32.93 | 0.260 | 0.92 | 0.018 | 0.330 |
| EWMA (λ=0.94) | 0.3338 | 23.96 | 0.461 | 0.76 | 0.05 | 0.192 |
| Random walk | 0.4107 | 25.73 | 0.422 | 0.70 | 0.002 | 0.007 |
| Expanding mean | 0.6818 | 44.48 | 0.000 | −1.34 | 0.02 | 0.004 |

What a skeptic should take from this table:

- **The leverage effect is the one robust modeling gain.** GJR-GARCH is the only model that beats HAR at conventional significance (p = 0.026) — fit on returns alone, no RV features. Symmetric GARCH does not.
- **The VIX is as hard a benchmark as advertised.** It ranks above HAR on QLIKE while *failing* MZ unbiasedness (β = 0.89, p = 0.0002): it systematically over-forecasts because it embeds the premium itself, and QLIKE's asymmetry (under-forecasting punished more) forgives that direction of bias. HAR-levels is the only model that passes MZ cleanly.
- **Jumps add nothing** at daily frequency (HAR-J/CJ ≤ HAR), consistent with the coarse daily bipower proxy — reported, not dropped.
- **LightGBM loses to HAR** on identical information (significantly vs HAR-log-IV, p = 0.0013) and its dev MCS p-value (0.06) excludes it from the 90% set. Reported, shipped anyway, complexity not adopted.
- Model Confidence Set (90%): {GJR, HAR-log-IV, VIX, EGARCH, GJR-t, HAR-IV, GARCH, HAR, HAR-J, HAR-CJ, HAR-log, and — a caution on test power — expanding mean at p = 0.101}. The 75% set: {GJR-n, HAR-log-IV, VIX, EGARCH, GJR-t, HAR-IV}.
- On the **holdout** (n = 1,873, includes COVID), QLIKE levels roughly double for everyone and the ranking shuffles *within* the indistinguishable set (VIX 0.442, HAR 0.452, EGARCH 0.455, GJR 0.457); only random walk is significantly worse than HAR (p = 0.003); LightGBM is again clearly worse (0.600). Conclusions do not flip: `reports/tables/phase7_holdout_h22.md`.
- Multi-step aggregation is done by summing the forecast variance path, not scaling the 1-day forecast; with fitted persistence ≈0.99 the naive ×22 shortcut errs by only a few percent on average but by up to ~±15% exactly after shocks (`reports/figures/phase3_agg_check.png`).

Secondary horizons (1d, 5d) in `reports/tables/phase3_h{1,5}.md`. One pathology worth naming: HAR-IV *in levels* produces negative fitted variances on 14% of days at h=1 (IV is biased high for 1-day RV, so the regression crosses zero in calm regimes) — a specification failure of levels-plus-IV at short horizons, reported as-is.

## 4. The strategy: does the forecast beat always-short, net of costs?

**Instrument.** Daily-rolled constant-maturity synthetic 30-day variance swap, short. Daily P&L per unit notional = carry (yesterday's strike amortized vs today's realized squared return) + mark-to-market on the unexpired book from VIX² changes; with a flat IV path this telescopes exactly to the swap payoff (unit-tested). Signals decided at close *t* earn from *t+1* (poisoning-tested).

**Costs are parameterized in vol points**, not % of premium: crossing at half-spread *h* gives up (σ+h)² − σ² variance units per notional traded, charged on the daily 1/22 book roll plus every position change. Sizing: vol-target 10% (estimated from the *unit* P&L, no self-reference), hard cap 1.5× notional, −5% month-to-date stop that flattens until month end. All parameters in `BacktestParams`.

**The breakeven chart is the headline result:**

![Breakeven](reports/figures/phase5_breakeven.png)

Development period, net at 0.5 vp half-spread (gross in parentheses):

| strategy | net Sharpe | net ann. ret | max DD | skew | CVaR99 | ann. cost |
|---|---|---|---|---|---|---|
| always-short | 0.47 (0.74) | 4.9% | −23.3% | −8.3 | −3.8% | 2.8% |
| naive VRP binary (trailing RV) | 0.34 (0.76) | 3.5% | −24.7% | −9.0 | −3.8% | 4.2% |
| model binary (GJR) | 0.39 (0.84) | 4.1% | −28.0% | −8.6 | −3.8% | 4.5% |
| model linear, daily rebal. | 0.46 (1.12) | 4.4% | −22.7% | −6.6 | −3.6% | 6.2% |
| **model linear, weekly rebal.** | **0.64 (1.02)** | 6.1% | −22.1% | −7.3 | −3.6% | 3.4% |

The pattern that decides the question: conditioning **widens the gross edge and pays it straight back in turnover**. Binary conditioning and the naive VRP are strictly worse than doing nothing sophisticated. The one variant that survives — proportional sizing, weekly rebalanced — beats always-short by +1.2%/yr net… with an HAC t-stat of 0.81 (**p = 0.42**). Its deflated Sharpe against N = 66 logged trials: **DSR = 0.51**, i.e. even odds that its true Sharpe exceeds the zero-skill multiple-testing benchmark (always-short: DSR = 0.28). Breakeven half-spreads: conditioned-daily variants die at ~0.9–1.0 vp; always-short and the weekly variant survive to ~1.4–1.5 vp; nothing survives 1.5 vp. Institutional SPX 1-month variance/at-the-money spreads are roughly 0.5–1.0 vp — inside exactly the band where the answer flips.

**Holdout (2019 → 2026-06, evaluated once):** always-short net Sharpe 0.19 (max DD −34.2%); model-linear-weekly 0.49 net / 0.86 gross (max DD −17.7%, skew −3.3 vs −7.5); the *daily*-rebalanced linear variant went negative (−0.08) — churn again. Net edge +3.3%/yr, HAC **p = 0.34**. Same verdict as dev: point estimate favors the forecast, significance does not arrive.

![Full equity](reports/figures/phase7_equity_full.png)

## 5. Crisis anatomy — this strategy sells insurance

Sharpe alone is actively misleading here: daily skew ≈ −7 to −8, kurtosis > 100, and the losses concentrate exactly when everything else is losing. Event studies (net P&L paths, per unit capital):

![Dev events](reports/figures/phase6_events_dev.png)
![Holdout events](reports/figures/phase7_events_holdout.png)

- **GFC (Sep–Dec 2008):** always-short −15%; the linear-sized strategy finished ≈ +3% — vol-targeting had already cut exposure as VIX rose, and the VRP signal kept it short only when the spread was extreme.
- **Volmageddon (Feb 2018):** nobody dodged it. The VRP was positive and vol was low on Feb 2; the spike was a *vol event from calm*, unforecastable by construction here. Binary/always −22% in days; linear −10% (smaller position, same hit); stops then flattened the book — visible as flat lines.
- **COVID (Feb–Apr 2020, holdout):** always-short −21% at trough; linear-weekly −12% with a faster recovery.
- **Yen-carry unwind (Aug 2024, holdout):** the clearest win for conditioning — linear-weekly was nearly flat *into* Aug 5 (VRP had compressed) and finished the window ≈ +1%; always-short and binary took −19% in three days.

Worst drawdown episodes and full tail tables: `reports/tables/phase6_*.md`. The stop-loss and vol-target do real work in the tails; they do not manufacture the (insignificant) mean edge — gross Sharpes order the same way.

## 6. Lookahead audit

The subtle trap in this design is alignment: VIX at close *t* must pair with RV over *t+1..t+22*, never a window reaching back into *t*. Defenses, all in `tests/`:

- `realized_var_forward` is provably immune to poisoning of all data ≤ *t* (bit-identical forecasts), and the assembled feature frame is re-checked end-to-end (`test_alignment.py`).
- The walk-forward engine excludes training rows with unobserved target windows; poisoning target values inside (*t−22, t*] changes nothing, poisoning the last *observable* row does (`test_walkforward.py`, both OLS and generic-model paths).
- The backtest earns P&L strictly from the day after the signal; flipping the signal at *s* cannot alter P&L through *s* (`test_backtest.py`).
- HAR features hand-computed; estimator formulas anchored to hand-calculated constants; cost arithmetic hand-computed; the rolled swap telescopes to the true payoff under flat IV.

54 tests, hermetic (no network), run in CI on 3.11/3.12 with ruff and mypy clean.

## 7. What this backtest does not capture

- **Crisis spread widening.** The half-spread is constant in vol points; real spreads blow out exactly in the events of §5, so crisis P&L for any *trading* during those windows is optimistic. (The always-short book trades least; the comparison direction is conservative for the conditioned strategies.)
- The mark on unexpired swaps assumes a flat 30-day term structure (VIX for all residual maturities); no margin, financing, or collateral haircuts; no capacity or market-impact modeling.
- VIX² is used as the variance-swap strike; the true strike trades at a small basis to VIX².
- Daily-frequency RV proxy throughout — with 5-minute RV the forecast rankings could tighten (QLIKE's proxy-robustness protects rankings, not levels).
- Single asset, single instrument; no cross-sectional diversification of the premium.
- The weekly-rebalance variants were added *after* observing daily-variant churn on the dev period — a mild form of iterative selection. That is exactly why the headline strategy is best-of-8, why all 66 configurations are logged, and why the deflated Sharpe and both p-values are reported next to every claim. Treat the 0.64 accordingly.

## 8. Repository and reproduction

```
data/{raw,processed}/     # gitignored; raw pulls carry a sha256 manifest
src/data|features|models|evaluation|backtest|plotting
src/run_phase1..7.py      # one runner per phase
tests/                    # 54 hermetic tests incl. anti-lookahead poisoning
reports/{figures,tables}  # every artifact in this README, regenerated by make
```

From a clean clone (Python ≥ 3.10; `uv`):

```bash
uv sync                          # pinned environment (uv.lock committed)
python3 scripts/download_raw.py  # stdlib-only raw pull (~1 min)
make all                         # phases 1-7: every number and figure above
make test lint                   # pytest, ruff, mypy
```

Determinism: single global seed (1990) recorded in `src/config.py`; EGARCH simulation seeds and refit schedules are keyed to absolute block indices so chunked/resumed runs are bit-identical. GARCH walk-forwards cache to `data/processed/` and resume automatically. `make phase7` exists to *reproduce* the one-time holdout evaluation, not to iterate on it.

## References

Corsi (2009) HAR-RV; Patton (2011) volatility proxies and loss functions; Diebold & Mariano (1995); Hansen, Lunde & Nason (2011) MCS; Barndorff-Nielsen & Shephard (2004) bipower variation; Carr & Wu (2009) and Bollerslev, Tauchen & Zhou (2009) on the VRP; Bailey & López de Prado (2014) deflated Sharpe.
