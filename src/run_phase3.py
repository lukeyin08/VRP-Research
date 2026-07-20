"""Phase 3: GARCH family + remaining HAR variants; full comparison at h=22.

Adds to the Phase 2 set: HAR-RV-IV (levels and log), HAR-RV-J, HAR-RV-CJ,
GARCH(1,1), GJR-GARCH, GJR-t, EGARCH. GARCH variants also report h=1 and h=5
(same recursion, partial sums). Everything walk-forward, dev period only.

Outputs: reports/tables/phase3_h{22,5,1}.md/.csv, phase3 forecasts parquet
(cached for Phase 4 MCS and Phase 5), reports/figures/phase3_agg_check.png.
"""

from __future__ import annotations

import pandas as pd

from src.config import DEV_END, EVAL_START, PROCESSED_DIR, TABLES_DIR
from src.evaluation.config_log import log_config, total_configs
from src.evaluation.dm import dm_test
from src.evaluation.losses import TARGET_EVAL_FLOOR, clip_forecasts, mse, oos_r2, qlike
from src.evaluation.mz import mz_regression
from src.evaluation.walkforward import WalkForwardSpec, dev_eval_index, holdout_guard
from src.features.build import load_features
from src.models import baselines
from src.models.garch import GARCH_SPECS, spec_params
from src.models.garch_cache import GARCH_KW, ensure_garch_dev, load_garch_dev
from src.models.har import HARForecaster
from src.plotting.phase3 import plot_garch_aggregation_check

EVAL_WINDOW = f"{EVAL_START}..{DEV_END}"
DM_BENCHMARK = "har"


def all_forecasts(
    df: pd.DataFrame, horizon: int, eval_index: pd.DatetimeIndex
) -> dict[str, pd.Series]:
    spec = WalkForwardSpec(horizon=horizon)
    out: dict[str, pd.Series] = {
        "random_walk": baselines.random_walk(df, horizon).reindex(eval_index),
        "expanding_mean": baselines.expanding_mean(df).reindex(eval_index),
        "ewma_riskmetrics": baselines.ewma_riskmetrics(df).reindex(eval_index),
        "vix": baselines.vix_as_forecast(df).reindex(eval_index),
    }
    har_variants = [
        HARForecaster(spec=spec),
        HARForecaster(log_space=True, spec=spec),
        HARForecaster(use_iv=True, spec=spec),
        HARForecaster(log_space=True, use_iv=True, spec=spec),
        HARForecaster(jump_mode="j", spec=spec),
        HARForecaster(jump_mode="cj", spec=spec),
    ]
    for m in har_variants:
        out[m.name] = m.forecast(df, eval_index)
    return out


def score(
    df: pd.DataFrame,
    forecasts: dict[str, pd.Series],
    horizon: int,
    eval_index: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (score table, aligned QLIKE loss frame on the common sample)."""
    y_all = df[f"target_rv_{horizon}"].reindex(eval_index)
    mask = y_all.notna()
    for f in forecasts.values():
        mask &= f.notna()
    y = y_all[mask]
    y_q = y.clip(lower=TARGET_EVAL_FLOOR)
    n_floored = int((y < TARGET_EVAL_FLOOR).sum())

    clipped: dict[str, pd.Series] = {}
    rows = []
    ql_frame: dict[str, pd.Series] = {}
    for name, f in forecasts.items():
        cf, _ = clip_forecasts(f[mask])
        clipped[name] = cf
        ql_frame[name] = qlike(y_q, cf)
    for name, cf in clipped.items():
        m = mz_regression(y, cf)
        d = dm_test(ql_frame[name], ql_frame[DM_BENCHMARK]) if name != DM_BENCHMARK else None
        rows.append(
            {
                "model": name,
                "QLIKE": ql_frame[name].mean(),
                "MSE_x1e4": mse(y, cf).mean() * 1e4,
                "OOS_R2_vs_expmean": oos_r2(y, cf, clipped["expanding_mean"]),
                "MZ_alpha": m.alpha,
                "MZ_beta": m.beta,
                "MZ_R2": m.r2,
                "MZ_p_joint": m.p_joint,
                "DM_stat_vs_har": d.stat if d else float("nan"),
                "DM_p_vs_har": d.pvalue if d else float("nan"),
            }
        )
    table = pd.DataFrame(rows).set_index("model").sort_values("QLIKE")
    table.attrs["n_obs"] = int(mask.sum())
    table.attrs["n_target_floored"] = n_floored
    losses = pd.DataFrame(ql_frame)
    return table, losses


def main() -> int:
    df = load_features()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # GARCH forecasts: resumable disk cache; one pass per spec covers all horizons
    garch_by_h: dict[int, dict[str, pd.Series]] = {1: {}, 5: {}, 22: {}}
    eval22 = dev_eval_index(pd.DatetimeIndex(df.index), 22)
    holdout_guard(eval22)
    for gname in GARCH_SPECS:
        ensure_garch_dev(gname)
        gfc = load_garch_dev(gname)
        for h in (1, 5, 22):
            garch_by_h[h][gname] = gfc[h]
        log_config("3", gname, 22, {**spec_params(gname), **GARCH_KW}, EVAL_WINDOW)

    saved: dict[int, pd.DataFrame] = {}
    for horizon in (22, 5, 1):
        eval_index = dev_eval_index(pd.DatetimeIndex(df.index), horizon)
        holdout_guard(eval_index)
        fc = all_forecasts(df, horizon, eval_index)
        for gname, g in garch_by_h[horizon].items():
            fc[gname] = g.reindex(eval_index)
        for name in fc:
            if name not in GARCH_SPECS:
                log_config("3", name, horizon, {"refit_every": 22, "min_train": 252}, EVAL_WINDOW)
        table, losses = score(df, fc, horizon, eval_index)
        saved[horizon] = table
        table.round(4).to_csv(TABLES_DIR / f"phase3_h{horizon}.csv")
        table.round(4).to_markdown(TABLES_DIR / f"phase3_h{horizon}.md")
        if horizon == 22:
            # cache aligned forecasts + losses for Phases 4 and 5
            pd.DataFrame(fc).to_parquet(PROCESSED_DIR / "forecasts_dev_h22.parquet")
            losses.to_parquet(PROCESSED_DIR / "qlike_dev_h22.parquet")

    # figure: correct aggregation vs naive 22x scaling for GARCH(1,1)
    g11 = load_garch_dev("garch_n")
    p = plot_garch_aggregation_check(g11[1], g11[22])

    for horizon in (22, 5, 1):
        t = saved[horizon]
        print(
            f"\n=== phase3 horizon {horizon}d | n = {t.attrs['n_obs']} "
            f"| floored targets: {t.attrs['n_target_floored']} ==="
        )
        with pd.option_context("display.width", 220, "display.float_format", "{:.4f}".format):
            print(t)
    print(f"\nfigure: {p}")
    print(f"configurations logged to date: {total_configs()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
