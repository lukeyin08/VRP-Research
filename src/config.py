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

# Single global seed, recorded here; every stochastic step must use it.
SEED = 1990
