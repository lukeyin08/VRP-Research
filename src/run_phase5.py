"""Phase 5: VRP signal + variance-swap backtest with the cost model (dev only).

Signal: VRP[t] = IV30[t] - E_t[RV(t+1..t+22)].
Physical forecast = GJR-GARCH(1,1) - the best dev-period QLIKE among physical
models (selection rule fixed on dev data only; har_log_iv, statistically
indistinguishable, is run as robustness). Naive version uses trailing 22d RV.

Strategies (weight rules fixed a priori, all logged):
  always_short   w=1                       <- the benchmark to beat
  naive_binary   w=1{IV - trailing RV > 0}
  model_binary   w=1{IV - forecast > 0}
  model_linear   w=clip(VRP / expanding median past positive VRP, 0, 1.5)

Headline cost: 0.5 vol-point half-spread; the breakeven chart sweeps 0..3 vp.

Outputs: reports/tables/phase5_perf.md, phase5_breakeven.csv,
figures phase5_breakeven.png + phase5_equity.png,
data/processed/strategy_dev_pnl.parquet (for Phase 6).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestParams, run_strategy
from src.backtest.metrics import perf_summary, sharpe
from src.backtest.signals import always_short, binary_weights, linear_weights, vrp_signal
from src.config import DEV_END, EVAL_START, PROCESSED_DIR, TABLES_DIR
from src.evaluation.config_log import log_config, total_configs
from src.evaluation.walkforward import holdout_guard
from src.features.build import load_features
from src.plotting.phase5 import plot_breakeven, plot_equity

EVAL_WINDOW = f"{EVAL_START}..{DEV_END}"
MODEL_FORECAST = "gjr_n"  # best dev QLIKE among physical-measure models (Phase 3/4)
ROBUSTNESS_FORECAST = "har_log_iv"
HEADLINE_SPREAD_VP = 0.5
SPREAD_GRID = [round(0.25 * i, 2) for i in range(13)]  # 0.0 .. 3.0 vol pts


def build_base_weights(df: pd.DataFrame, fc: pd.DataFrame) -> dict[str, tuple[pd.Series, int]]:
    """name -> (base weights, rebalance_every). Weekly variants exist because the
    conditioned signals churn daily; trading them weekly is a standard
    implementation choice fixed a priori, not a tuned parameter."""
    idx = fc.index
    iv = df["iv30"].reindex(idx)
    vrp_model = vrp_signal(iv, fc[MODEL_FORECAST])
    vrp_naive = vrp_signal(iv, df["rv_cc_trail_22"].reindex(idx))
    vrp_robust = vrp_signal(iv, fc[ROBUSTNESS_FORECAST])
    return {
        "always_short": (always_short(idx), 1),
        "always_short_5d": (always_short(idx), 5),
        "naive_binary": (binary_weights(vrp_naive), 1),
        "model_binary": (binary_weights(vrp_model), 1),
        "model_binary_5d": (binary_weights(vrp_model), 5),
        "model_linear": (linear_weights(vrp_model), 1),
        "model_linear_5d": (linear_weights(vrp_model), 5),
        "robust_binary": (binary_weights(vrp_robust), 1),
    }


def main() -> int:
    df = load_features()
    fc = pd.read_parquet(PROCESSED_DIR / "forecasts_dev_h22.parquet")
    holdout_guard(pd.DatetimeIndex(fc.index))
    weights = build_base_weights(df, fc)

    # breakeven sweep
    curves: dict[str, list[float]] = {name: [] for name in weights}
    for h in SPREAD_GRID:
        for name, (base, reb) in weights.items():
            params = BacktestParams(half_spread_vp=h, rebalance_every=reb)
            res = run_strategy(df, base, params)
            curves[name].append(sharpe(res["pnl_net"]))
    breakeven = pd.DataFrame(curves, index=pd.Index(SPREAD_GRID, name="half_spread_vp"))

    # headline runs, gross + net
    rows = []
    pnl_store: dict[str, pd.Series] = {}
    for name, (base, reb) in weights.items():
        params = BacktestParams(half_spread_vp=HEADLINE_SPREAD_VP, rebalance_every=reb)
        res = run_strategy(df, base, params)
        pnl_store[name] = res["pnl_net"]
        pnl_store[f"{name}__gross"] = res["pnl_gross"]
        g = perf_summary(res["pnl_gross"], name)
        n_ = perf_summary(res["pnl_net"], name)
        rows.append(
            {
                "strategy": name,
                "gross_ann_ret": g["ann_return"],
                "gross_sharpe": g["sharpe"],
                "net_ann_ret": n_["ann_return"],
                "net_sharpe": n_["sharpe"],
                "net_max_dd": n_["max_dd"],
                "net_skew": n_["skew"],
                "net_CVaR99": n_["CVaR99"],
                "avg_weight": float(res["w"].mean()),
                "ann_cost": float(res["cost"].mean() * 252),
            }
        )
        log_config(
            "5",
            f"strategy_{name}",
            22,
            {**params.as_dict(), "signal": name, "physical_forecast": MODEL_FORECAST},
            EVAL_WINDOW,
        )
    perf = pd.DataFrame(rows).set_index("strategy")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    perf.round(4).to_markdown(TABLES_DIR / "phase5_perf.md")
    breakeven.round(3).to_csv(TABLES_DIR / "phase5_breakeven.csv")
    pd.DataFrame(pnl_store).to_parquet(PROCESSED_DIR / "strategy_dev_pnl.parquet")

    p1 = plot_breakeven(breakeven)
    p2 = plot_equity({k: v for k, v in pnl_store.items() if "__gross" not in k})

    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(f"=== phase5 | {EVAL_WINDOW} | headline half-spread {HEADLINE_SPREAD_VP} vp ===")
        print(perf)
        print("\nNet Sharpe by half-spread (vol pts):")
        print(breakeven.round(2))
    # breakeven points (first spread where net sharpe <= 0), linear interpolation
    for name in weights:
        s = breakeven[name]
        below = s[s <= 0]
        be = below.index[0] if len(below) else np.nan
        print(f"breakeven half-spread {name}: {be if pd.notna(be) else '>3.0'} vp")
    print(f"figures: {p1}, {p2}")
    print(f"configurations logged to date: {total_configs()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
