"""LightGBM on the same information set as HAR-IV plus jump/range features.

Included specifically to test whether nonlinear ML beats HAR out of sample; in
this literature it usually does not by much, and either outcome is reported.
One FIXED hyperparameter configuration, chosen a priori at modest complexity
and logged - never tuned on evaluation data. Trained on log-RV (skewed target)
with Duan smearing on the retransform, same as HAR-log.

Feature note: only long-history, lag-safe features. Parkinson trails are valid
over the whole sample (high/low only); open-dependent estimators (GK/RS/YZ)
are excluded because pre-2008 Yahoo opens are synthetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.config import SEED
from src.evaluation.walkforward import WalkForwardSpec, walkforward_model
from src.models.har import LOG_FLOOR_ANN

LGBM_FEATURES = (
    "rv_cc_trail_1",
    "rv_cc_trail_5",
    "rv_cc_trail_22",
    "bv_trail_22",
    "jump_22",
    "iv30",
    "rv_pk_trail_5",
    "rv_pk_trail_22",
)

# Single fixed configuration (logged); never tuned on evaluation data.
LGBM_PARAMS: dict[str, object] = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": SEED,
    "verbosity": -1,
}


def make_lgbm() -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=SEED,
        verbosity=-1,
    )


def lgbm_design(df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.Series]:
    x = df[list(LGBM_FEATURES)].copy()
    xv = np.log(np.clip(x.to_numpy(dtype=float), LOG_FLOOR_ANN, None))
    x = pd.DataFrame(xv, index=x.index, columns=x.columns)
    y_raw = df[f"target_rv_{horizon}"]
    y = pd.Series(
        np.log(np.clip(y_raw.to_numpy(dtype=float), LOG_FLOOR_ANN, None)),
        index=y_raw.index,
        name=y_raw.name,
    )
    return x, y


def lgbm_forecast(
    df: pd.DataFrame, eval_index: pd.DatetimeIndex, horizon: int = 22, refit_every: int = 22
) -> pd.Series:
    x, y = lgbm_design(df, horizon)
    spec = WalkForwardSpec(horizon=horizon, refit_every=refit_every, min_train=756)
    f = walkforward_model(x, y, eval_index, spec, model_factory=make_lgbm, log_space=True)
    return f.rename("lightgbm")


def lgbm_feature_importance(df: pd.DataFrame, horizon: int, train_end: pd.Timestamp) -> pd.Series:
    """Gain importance from a single fit on the dev training set (diagnostic only)."""
    x, y = lgbm_design(df, horizon)
    m = pd.concat([x, y], axis=1).loc[:train_end].dropna()
    model = make_lgbm()
    model.fit(m[list(LGBM_FEATURES)].to_numpy(), m[str(y.name)].to_numpy())
    imp = pd.Series(model.feature_importances_, index=list(LGBM_FEATURES), name="importance")
    return imp.sort_values(ascending=False)
