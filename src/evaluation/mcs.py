"""Model Confidence Set (Hansen, Lunde & Nason 2011) via arch's implementation.

Given a T x k frame of losses, the MCS procedure repeatedly eliminates the
worst model until the null of equal predictive ability cannot be rejected,
producing a p-value per model: the set of models with p >= alpha is the
confidence set of models statistically indistinguishable from the best. Block
bootstrap handles the serial dependence from overlapping horizons.
"""

from __future__ import annotations

import pandas as pd
from arch.bootstrap import MCS

from src.config import SEED


def model_confidence_set(
    losses: pd.DataFrame, reps: int = 1000, block_size: int = 44
) -> pd.DataFrame:
    clean = losses.dropna()
    mcs = MCS(clean, size=0.1, reps=reps, block_size=block_size, method="R", seed=SEED)
    mcs.compute()
    out = mcs.pvalues.rename(columns={"Pvalue": "mcs_pvalue"})
    out["in_set_90"] = out["mcs_pvalue"] >= 0.10
    out["in_set_75"] = out["mcs_pvalue"] >= 0.25
    return out.sort_values("mcs_pvalue", ascending=False)
