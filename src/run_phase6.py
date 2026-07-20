"""Phase 6: risk analysis + development-period crisis event studies.

This strategy sells insurance; Sharpe alone is actively misleading for it, so
this phase reports the full tail profile, tabulates the worst drawdowns, tests
whether the conditioned strategy's net edge over always-short is statistically
distinguishable (HAC), and computes the deflated Sharpe ratio against the total
number of configurations ever logged.

Event studies here cover dev-period crises (GFC 2008, Volmageddon Feb-2018).
COVID-2020 and the Aug-2024 yen-carry unwind sit in the holdout and are
examined exactly once, in Phase 7.
"""

from __future__ import annotations

import pandas as pd

from src.backtest.metrics import deflated_sharpe, drawdown_episodes, perf_summary
from src.config import PROCESSED_DIR, TABLES_DIR
from src.evaluation.config_log import total_configs
from src.evaluation.dm import DMResult, dm_test
from src.plotting.phase6 import plot_drawdowns, plot_event_studies

HEADLINE = "model_linear_5d"
BENCH = "always_short"
KEY_STRATEGIES = [
    "always_short",
    "naive_binary",
    "model_binary",
    "model_binary_5d",
    "model_linear",
    "model_linear_5d",
]

DEV_EVENTS = {
    "GFC (Sep-Dec 2008)": ("2008-09-01", "2008-12-31"),
    "Volmageddon (Jan-Mar 2018)": ("2018-01-15", "2018-03-15"),
}


def net_edge_test(pnl_a: pd.Series, pnl_b: pd.Series, lag: int = 22) -> DMResult:
    """HAC test on the daily net P&L differential (a minus b; positive favors a).

    Uses the DM machinery with losses = -pnl, so mean_diff < 0 still means 'a
    better', consistent with the forecast tables.
    """
    return dm_test(-pnl_a, -pnl_b, lag=lag)


def main() -> int:
    pnl = pd.read_parquet(PROCESSED_DIR / "strategy_dev_pnl.parquet")

    # full tail profile, net at the headline spread
    risk = pd.DataFrame([perf_summary(pnl[s].dropna(), s) for s in KEY_STRATEGIES]).set_index(
        "strategy"
    )

    # worst drawdown episodes for benchmark and headline strategy
    dd_bench = drawdown_episodes(pnl[BENCH].dropna())
    dd_head = drawdown_episodes(pnl[HEADLINE].dropna())

    # is the net edge statistically distinguishable?
    edge = net_edge_test(pnl[HEADLINE].dropna(), pnl[BENCH].dropna())
    edge_binary = net_edge_test(pnl["model_binary"].dropna(), pnl[BENCH].dropna())

    # deflated Sharpe for the headline strategy, penalized by every config logged
    n_trials = total_configs()
    ds = deflated_sharpe(pnl[HEADLINE].dropna(), n_trials=n_trials)
    ds_bench = deflated_sharpe(pnl[BENCH].dropna(), n_trials=n_trials)

    sig = pd.DataFrame(
        [
            {
                "comparison": f"{HEADLINE} - {BENCH} (net, HAC lag 22)",
                "mean_daily_diff": -edge.mean_diff,
                "ann_diff": -edge.mean_diff * 252,
                "t_stat": -edge.stat,
                "p_value": edge.pvalue,
            },
            {
                "comparison": f"model_binary - {BENCH} (net, HAC lag 22)",
                "mean_daily_diff": -edge_binary.mean_diff,
                "ann_diff": -edge_binary.mean_diff * 252,
                "t_stat": -edge_binary.stat,
                "p_value": edge_binary.pvalue,
            },
            {
                "comparison": f"deflated Sharpe {HEADLINE} (N={n_trials} trials)",
                "mean_daily_diff": ds.sharpe_ann,
                "ann_diff": ds.sr_benchmark_ann,
                "t_stat": float("nan"),
                "p_value": 1.0 - ds.dsr,
            },
            {
                "comparison": f"deflated Sharpe {BENCH} (N={n_trials} trials)",
                "mean_daily_diff": ds_bench.sharpe_ann,
                "ann_diff": ds_bench.sr_benchmark_ann,
                "t_stat": float("nan"),
                "p_value": 1.0 - ds_bench.dsr,
            },
        ]
    )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    risk.round(4).to_markdown(TABLES_DIR / "phase6_risk.md")
    dd_bench.round(4).to_markdown(TABLES_DIR / "phase6_drawdowns_always_short.md")
    dd_head.round(4).to_markdown(TABLES_DIR / "phase6_drawdowns_headline.md")
    sig.round(5).to_markdown(TABLES_DIR / "phase6_significance.md", index=False)

    p1 = plot_drawdowns({s: pnl[s].dropna() for s in (BENCH, HEADLINE)})
    p2 = plot_event_studies(pnl, [BENCH, HEADLINE, "model_binary"], DEV_EVENTS)

    with pd.option_context("display.width", 220, "display.float_format", "{:.4f}".format):
        print("=== phase6 risk profile (net, 0.5 vp half-spread, dev period) ===")
        print(risk)
        print(f"\nworst drawdowns - {BENCH}:")
        print(dd_bench)
        print(f"\nworst drawdowns - {HEADLINE}:")
        print(dd_head)
        print("\nsignificance / deflation:")
        print(sig)
    print(f"\nfigures: {p1}, {p2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
