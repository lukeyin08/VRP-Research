"""Phase 4: LightGBM on the HAR information set + Model Confidence Set.

The ML layer exists to answer one question: does nonlinear ML beat HAR on
identical information? The MCS then identifies, across ALL h=22 models, the
set statistically indistinguishable from the best at 90%/75% confidence.

Outputs: reports/tables/phase4_h22.md, phase4_mcs.md, phase4_lgbm_importance.md;
updates data/processed/forecasts_dev_h22.parquet with the lightgbm column.
"""

from __future__ import annotations

import pandas as pd

from src.config import DEV_END, EVAL_START, PROCESSED_DIR, TABLES_DIR
from src.evaluation.config_log import log_config, total_configs
from src.evaluation.dm import dm_test
from src.evaluation.losses import TARGET_EVAL_FLOOR, clip_forecasts, mse, oos_r2, qlike
from src.evaluation.mcs import model_confidence_set
from src.evaluation.mz import mz_regression
from src.evaluation.walkforward import dev_eval_index, holdout_guard
from src.features.build import load_features
from src.models.lgbm import LGBM_FEATURES, LGBM_PARAMS, lgbm_feature_importance, lgbm_forecast

EVAL_WINDOW = f"{EVAL_START}..{DEV_END}"


def main() -> int:
    df = load_features()
    eval_index = dev_eval_index(pd.DatetimeIndex(df.index), 22)
    holdout_guard(eval_index)

    fc_all = pd.read_parquet(PROCESSED_DIR / "forecasts_dev_h22.parquet")
    lgbm = lgbm_forecast(df, eval_index, horizon=22)
    fc_all["lightgbm"] = lgbm.reindex(fc_all.index)
    fc_all.to_parquet(PROCESSED_DIR / "forecasts_dev_h22.parquet")
    log_config("4", "lightgbm", 22, {**LGBM_PARAMS, "features": list(LGBM_FEATURES)}, EVAL_WINDOW)

    y = df["target_rv_22"].reindex(fc_all.index)
    mask = y.notna() & fc_all.notna().all(axis=1)
    y_m = y[mask]
    y_q = y_m.clip(lower=TARGET_EVAL_FLOOR)

    ql: dict[str, pd.Series] = {}
    rows = []
    for name in fc_all.columns:
        cf, _ = clip_forecasts(fc_all.loc[mask, name])
        ql[name] = qlike(y_q, cf)
        m = mz_regression(y_m, cf)
        rows.append(
            {
                "model": name,
                "QLIKE": ql[name].mean(),
                "MSE_x1e4": mse(y_m, cf).mean() * 1e4,
                "OOS_R2_vs_expmean": oos_r2(y_m, cf, fc_all.loc[mask, "expanding_mean"]),
                "MZ_beta": m.beta,
                "MZ_p_joint": m.p_joint,
            }
        )
    losses = pd.DataFrame(ql)

    # DM: LightGBM vs the HAR variants it must beat to justify its complexity
    for bench in ("har", "har_log_iv"):
        d = dm_test(losses["lightgbm"], losses[bench])
        rows_idx = [r["model"] for r in rows].index("lightgbm")
        rows[rows_idx][f"DM_p_vs_{bench}"] = d.pvalue
        rows[rows_idx][f"DM_stat_vs_{bench}"] = d.stat

    table = pd.DataFrame(rows).set_index("model").sort_values("QLIKE")
    mcs = model_confidence_set(losses)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(TABLES_DIR / "phase4_h22.csv")
    table.round(4).to_markdown(TABLES_DIR / "phase4_h22.md")
    mcs.round(4).to_markdown(TABLES_DIR / "phase4_mcs.md")

    imp = lgbm_feature_importance(df, 22, pd.Timestamp(DEV_END))
    imp.to_frame().to_markdown(TABLES_DIR / "phase4_lgbm_importance.md")

    losses.to_parquet(PROCESSED_DIR / "qlike_dev_h22.parquet")

    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(f"=== phase4 h=22 | n = {int(mask.sum())} | {EVAL_WINDOW} ===")
        print(table)
        print("\nMODEL CONFIDENCE SET (block bootstrap, QLIKE)")
        print(mcs)
        print("\nLightGBM split-importance (single dev fit, diagnostic only)")
        print(imp)
    print(f"\nconfigurations logged to date: {total_configs()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
