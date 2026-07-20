"""Registry of every model configuration ever evaluated.

Multiple-testing inflation is the main reason backtests die out of sample, so
every (model, params, horizon, eval window) combination that gets scored is
appended here, deduplicated, and committed to the repo. The total count feeds
the deflated Sharpe ratio for the headline strategy result.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from src.config import REPORTS_DIR

LOG_PATH = REPORTS_DIR / "config_log.csv"
_COLUMNS = ["phase", "model", "horizon", "params", "eval_window", "first_logged_utc"]


def log_config(
    phase: str, model: str, horizon: int, params: dict[str, object], eval_window: str
) -> int:
    """Record a configuration; returns the total distinct count so far."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "phase": phase,
        "model": model,
        "horizon": horizon,
        "params": json.dumps(params, sort_keys=True),
        "eval_window": eval_window,
    }
    if LOG_PATH.exists():
        df = pd.read_csv(LOG_PATH)
    else:
        df = pd.DataFrame(columns=_COLUMNS)
    key_cols = ["phase", "model", "horizon", "params", "eval_window"]
    exists = bool(((df[key_cols] == pd.Series(row)).all(axis=1)).any()) if len(df) else False
    if not exists:
        row["first_logged_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(LOG_PATH, index=False)
    return len(df)


def total_configs() -> int:
    if not LOG_PATH.exists():
        return 0
    return len(pd.read_csv(LOG_PATH))
