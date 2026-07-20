"""Paths, constants, and global configuration. No logic."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

# --- conventions -------------------------------------------------------------
# All variance quantities in this project are ANNUALIZED variances of daily log
# returns, in decimal units (e.g. 0.04 = (20 vol pts)^2). VIX is quoted as
# annualized vol in percentage points, so implied variance = (VIX/100)^2.
TRADING_DAYS_PER_YEAR = 252
HORIZONS = (1, 5, 22)
PRIMARY_HORIZON = 22  # ~30 calendar days, matching the VIX horizon

SAMPLE_START = "1990-01-02"  # first VIX print (current methodology, back-computed)

# --- evaluation protocol ------------------------------------------------------
# Walk-forward forecasts are produced from EVAL_START (first ten years are the
# initial training window). Phases 2-4 evaluate ONLY on [EVAL_START, DEV_END].
# [HOLDOUT_START, ...] is the final untouched test set, evaluated exactly once
# in Phase 7; src.evaluation.walkforward.holdout_guard enforces this in code.
EVAL_START = "2000-01-03"
DEV_END = "2018-12-31"
HOLDOUT_START = "2019-01-01"

# Single global seed, recorded here; every stochastic step must use it.
SEED = 1990
