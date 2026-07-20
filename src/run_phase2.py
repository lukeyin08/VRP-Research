"""Phase 2: baselines + HAR, walk-forward on the development period only.

Models (h = 22 primary; 1 and 5 secondary): random walk, expanding mean,
EWMA(0.94), VIX-as-forecast, HAR in levels, HAR in logs. Scored with QLIKE and
MSE on a common non-NaN sample; DM tests vs HAR(levels) with NW lag 44;
Mincer-Zarnowitz with HAC errors; Campbell-Thompson OOS R^2 vs expanding mean.

Outputs: reports/tables/phase2_h{1,5,22}.md (+.csv),
reports/figures/phase2_forecasts_ts.png, phase2_mz_scatter.png
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from src.config import DEV_END, EVAL_START, TABLES_DIR
from src.evaluation.config_log import log_config, total_configs
from src.evaluation.dm import dm_test
from src.evaluation.losses import TARGET_EVAL_FLOOR, clip_forecasts, mse, oos_r2, qlike
from src.evaluation.mz import mz_regression
from src.evaluation.walkforward import WalkForwardSpec, dev_eval_index, holdout_guard
from src.features.build import load_features
from src.models import baselines
from src.models.har import HARForecaster
from src.plotting.phase2 import plot_forecast_ts, plot_mz_scatter

EVAL_WINDOW = f"{EVAL_START}..{DEV_END}"
DM_BENCHMARK = "har"


def collect_forecasts(
    df: pd.DataFrame, horizon: int, eval_index: pd.DatetimeIndex
) -> dict[str, pd.Series]:
    spec = WalkForwardSpec(horizon=horizon)
    full: dict[str, Callable[[], pd.Series]] = {
        "random_walk": lambda: baselines.random_walk(df, horizon),
        "expanding_mean": lambda: baselines.expanding_mean(df),
        "ewma_riskmetrics": lambda: baselines.ewma_riskmetrics(df),
        "vix": lambda: baselines.vix_as_forecast(df),
    }
    out = {name: fn().reindex(eval_index) for name, fn in full.items()}
    for log_space in (False, True):
        m = HARForecaster(log_space=log_space, spec=spec)
        out[m.name] = m.forecast(df, eval_index)
    return out


def score_horizon(df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    eval_index = dev_eval_index(pd.DatetimeIndex(df.index), horizon)
    holdout_guard(eval_index)
    forecasts = collect_forecasts(df, horizon, eval_index)
    y_all = df[f"target_rv_{horizon}"].reindex(eval_index)

    # common sample: identical dates for every model, so losses are comparable
    mask = y_all.notna()
    for f in forecasts.values():
        mask &= f.notna()
    y = y_all[mask]

    clipped: dict[str, pd.Series] = {}
    n_clipped: dict[str, int] = {}
    for name, f in forecasts.items():
        clipped[name], n_clipped[name] = clip_forecasts(f[mask])

    # QLIKE target floored at quote resolution (counted); MSE/MZ use raw target
    y_q = y.clip(lower=TARGET_EVAL_FLOOR)
    n_floored = int((y < TARGET_EVAL_FLOOR).sum())
    ql = {name: qlike(y_q, f) for name, f in clipped.items()}
    rows = []
    for name, f in clipped.items():
        m = mz_regression(y, f)
        d = dm_test(ql[name], ql[DM_BENCHMARK]) if name != DM_BENCHMARK else None
        rows.append(
            {
                "model": name,
                "QLIKE": ql[name].mean(),
                "MSE_x1e4": mse(y, f).mean() * 1e4,
                "OOS_R2_vs_expmean": oos_r2(y, f, clipped["expanding_mean"]),
                "MZ_alpha": m.alpha,
                "MZ_beta": m.beta,
                "MZ_R2": m.r2,
                "MZ_p_joint": m.p_joint,
                "DM_stat_vs_har": d.stat if d else float("nan"),
                "DM_p_vs_har": d.pvalue if d else float("nan"),
                "n_clipped": n_clipped[name],
            }
        )
        log_config(
            phase="2",
            model=name,
            horizon=horizon,
            params={"refit_every": 22, "min_train": 252},
            eval_window=EVAL_WINDOW,
        )
    table = pd.DataFrame(rows).set_index("model").sort_values("QLIKE")
    table.attrs["n_obs"] = int(mask.sum())
    table.attrs["n_target_floored"] = n_floored
    return table, {**clipped, "_target": y}


def main() -> int:
    df = load_features()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[int, pd.DataFrame] = {}
    fc22: dict[str, pd.Series] = {}
    for horizon in (22, 5, 1):
        table, fc = score_horizon(df, horizon)
        results[horizon] = table
        if horizon == 22:
            fc22 = fc
        table.round(4).to_csv(TABLES_DIR / f"phase2_h{horizon}.csv")
        table.round(4).to_markdown(TABLES_DIR / f"phase2_h{horizon}.md")

    best_har = min(("har", "har_log"), key=lambda m: float(results[22]["QLIKE"].loc[m]))
    p1 = plot_forecast_ts(fc22["_target"], fc22[best_har], fc22["vix"], best_har)
    p2 = plot_mz_scatter(fc22["_target"], fc22[best_har], best_har)

    for horizon in (22, 5, 1):
        t = results[horizon]
        print(
            f"\n=== horizon {horizon}d | n = {t.attrs['n_obs']} "
            f"| targets floored for QLIKE: {t.attrs['n_target_floored']} | {EVAL_WINDOW} ==="
        )
        with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
            print(t)
    print(f"\nfigures: {p1}, {p2}")
    print(f"configurations logged to date: {total_configs()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
