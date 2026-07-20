"""Phase 7: the single evaluation on the final holdout (2019 -> sample end).

Everything before this phase used only 2000-2018. This runner scores the
frozen model set and the frozen strategy rules ONCE on the holdout - including
COVID-2020 and the Aug-2024 yen-carry unwind - and regenerates the stitched
full-period figures for the README. Re-running reproduces the same single
evaluation; nothing here may feed back into model or strategy selection.

Outputs: reports/tables/phase7_holdout_h22.md, phase7_strategy_holdout.md,
phase7_significance_holdout.md, figures phase7_events_holdout.png,
phase7_equity_full.png.
"""

from __future__ import annotations

import pandas as pd

from src.backtest.engine import BacktestParams, run_strategy
from src.backtest.metrics import perf_summary
from src.config import HOLDOUT_START, PROCESSED_DIR, TABLES_DIR
from src.evaluation.walkforward import WalkForwardSpec, holdout_eval_index
from src.features.build import load_features
from src.models import baselines
from src.models.garch import GARCH_SPECS
from src.models.garch_cache import ensure_garch, load_garch
from src.models.har import HARForecaster
from src.models.lgbm import lgbm_forecast
from src.plotting.phase6 import plot_event_studies
from src.plotting.phase7 import plot_full_equity
from src.run_phase3 import score
from src.run_phase5 import HEADLINE_SPREAD_VP, build_base_weights
from src.run_phase6 import BENCH, HEADLINE, net_edge_test

HOLDOUT_EVENTS = {
    "COVID (Feb-Apr 2020)": ("2020-02-14", "2020-04-30"),
    "Yen-carry unwind (Jul-Aug 2024)": ("2024-07-25", "2024-08-20"),
}


def holdout_forecasts(df: pd.DataFrame, eval_index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    spec = WalkForwardSpec(horizon=22)
    out: dict[str, pd.Series] = {
        "random_walk": baselines.random_walk(df, 22).reindex(eval_index),
        "expanding_mean": baselines.expanding_mean(df).reindex(eval_index),
        "ewma_riskmetrics": baselines.ewma_riskmetrics(df).reindex(eval_index),
        "vix": baselines.vix_as_forecast(df).reindex(eval_index),
    }
    for m in (
        HARForecaster(spec=spec),
        HARForecaster(log_space=True, spec=spec),
        HARForecaster(use_iv=True, spec=spec),
        HARForecaster(log_space=True, use_iv=True, spec=spec),
        HARForecaster(jump_mode="j", spec=spec),
        HARForecaster(jump_mode="cj", spec=spec),
    ):
        out[m.name] = m.forecast(df, eval_index)
    for gname in GARCH_SPECS:
        ensure_garch(gname, "holdout")
        out[gname] = load_garch(gname, "holdout")[22].reindex(eval_index)
    out["lightgbm"] = lgbm_forecast(df, eval_index, horizon=22)
    return out


def main() -> int:
    df = load_features()
    idx = pd.DatetimeIndex(df.index)
    hold_idx = holdout_eval_index(idx, 22)

    # ---- forecasts on the holdout, scored once -----------------------------
    fc_hold = holdout_forecasts(df, hold_idx)
    table, _losses = score(df, fc_hold, 22, hold_idx)

    # ---- strategies: stitched dev+holdout signal, frozen rules -------------
    fc_dev = pd.read_parquet(PROCESSED_DIR / "forecasts_dev_h22.parquet")
    common = [c for c in fc_dev.columns if c in fc_hold]
    fc_full = pd.concat(
        [fc_dev[common], pd.DataFrame({k: fc_hold[k] for k in common})]
    ).sort_index()
    fc_full = fc_full[~fc_full.index.duplicated(keep="first")]

    weights = build_base_weights(df, fc_full)
    params_by_name = {
        name: BacktestParams(half_spread_vp=HEADLINE_SPREAD_VP, rebalance_every=reb)
        for name, (_, reb) in weights.items()
    }
    pnl_full: dict[str, pd.Series] = {}
    rows = []
    for name, (base, _reb) in weights.items():
        res = run_strategy(df, base, params_by_name[name])
        pnl_full[name] = res["pnl_net"]
        hold_slice = res["pnl_net"].loc[HOLDOUT_START:]
        gross_slice = res["pnl_gross"].loc[HOLDOUT_START:]
        s = perf_summary(hold_slice, name)
        s["gross_sharpe"] = perf_summary(gross_slice, name)["sharpe"]
        rows.append(s)
    strat = pd.DataFrame(rows).set_index("strategy")

    edge = net_edge_test(
        pnl_full[HEADLINE].loc[HOLDOUT_START:], pnl_full[BENCH].loc[HOLDOUT_START:]
    )
    sig = pd.DataFrame(
        [
            {
                "comparison": f"{HEADLINE} - {BENCH} net, holdout only (HAC lag 22)",
                "ann_diff": -edge.mean_diff * 252,
                "t_stat": -edge.stat,
                "p_value": edge.pvalue,
                "n_days": edge.n,
            }
        ]
    )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.round(4).to_markdown(TABLES_DIR / "phase7_holdout_h22.md")
    strat.round(4).to_markdown(TABLES_DIR / "phase7_strategy_holdout.md")
    sig.round(5).to_markdown(TABLES_DIR / "phase7_significance_holdout.md", index=False)

    pnl_frame = pd.DataFrame(pnl_full)
    pnl_frame.to_parquet(PROCESSED_DIR / "strategy_full_pnl.parquet")
    p1 = plot_event_studies(
        pnl_frame, [BENCH, HEADLINE, "model_binary"], HOLDOUT_EVENTS, "phase7_events_holdout.png"
    )
    p2 = plot_full_equity(pnl_frame[[BENCH, "naive_binary", "model_binary", HEADLINE]])

    with pd.option_context("display.width", 220, "display.float_format", "{:.4f}".format):
        print(f"=== phase7 HOLDOUT {HOLDOUT_START}..{hold_idx.max().date()} - evaluated once ===")
        print(f"forecasts (n={table.attrs['n_obs']}, floored={table.attrs['n_target_floored']}):")
        print(table)
        print("\nstrategies on holdout (net, 0.5 vp; gross_sharpe alongside):")
        print(strat)
        print("\nholdout significance:")
        print(sig)
    print(f"figures: {p1}, {p2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
