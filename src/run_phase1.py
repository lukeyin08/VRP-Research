"""Phase 1: build processed data, validate, and sanity-check the VRP.

Outputs
-------
- data/processed/features.parquet
- reports/figures/phase1_vix_vs_forward_rv.png
- reports/figures/phase1_vrp_expost.png
- reports/tables/phase1_summary.md, phase1_estimators.md
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import TABLES_DIR
from src.data.validate import report as report_issues
from src.features.build import build_features
from src.plotting.phase1 import plot_vix_vs_forward_rv, plot_vrp_expost


def _summary_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["iv30", "target_rv_22"]).copy()
    d["decade"] = (pd.DatetimeIndex(d.index).year // 10 * 10).astype(str) + "s"
    rows = []
    for label, grp in [("full sample", d), *list(d.groupby("decade"))]:
        rows.append(
            {
                "period": label,
                "days": len(grp),
                "mean VIX (vol pts)": grp["vix_close"].mean(),
                "mean fwd 22d realized vol (vol pts)": (np.sqrt(grp["target_rv_22"]) * 100).mean(),
                "mean IV (var units)": grp["iv30"].mean(),
                "mean fwd RV (var units)": grp["target_rv_22"].mean(),
                "mean VRP (var units)": grp["vrp_expost"].mean(),
                "share VRP > 0": (grp["vrp_expost"] > 0).mean(),
            }
        )
    return pd.DataFrame(rows).set_index("period")


def _estimator_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "cc (close-to-close)": "dv_cc",
        "parkinson": "dv_parkinson",
        "garman-klass": "dv_garman_klass",
        "rogers-satchell": "dv_rogers_satchell",
        "yang-zhang (22d)": "dv_yang_zhang22",
    }
    d = df[list(cols.values())].dropna()
    out = pd.DataFrame(
        {
            "mean ann. vol (pts)": {
                k: float(np.sqrt(d[v].mean() * 252) * 100) for k, v in cols.items()
            },
            "corr with cc (daily)": {k: float(d[v].corr(d["dv_cc"])) for k, v in cols.items()},
            "corr with cc (22d sums)": {
                k: float(d[v].rolling(22).sum().corr(d["dv_cc"].rolling(22).sum()))
                for k, v in cols.items()
            },
        }
    )
    out.index.name = "estimator"
    return out


def main() -> int:
    df, align_rep, issues = build_features(strict=True)

    print("=" * 72)
    print("ALIGNMENT")
    print(align_rep)
    print("=" * 72)
    print("VALIDATION (warnings and info; errors would have raised)")
    print(report_issues([i for i in issues if i.severity != "error"]))
    print("=" * 72)

    summary = _summary_table(df)
    est = _estimator_table(df)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    summary.round(4).to_markdown(TABLES_DIR / "phase1_summary.md")
    est.round(4).to_markdown(TABLES_DIR / "phase1_estimators.md")

    p1 = plot_vix_vs_forward_rv(df)
    p2 = plot_vrp_expost(df)

    with pd.option_context("display.width", 160, "display.float_format", "{:.4f}".format):
        print("PHASE 1 SUMMARY (annualized variance units unless noted)")
        print(summary)
        print()
        print("RV ESTIMATOR COMPARISON")
        print(est)
    print()
    print(f"figures: {p1}, {p2}")
    mean_vrp = df["vrp_expost"].mean()
    share_pos = (df["vrp_expost"].dropna() > 0).mean()
    print(
        f"\nVRP check: mean ex-post VRP = {mean_vrp:.5f} var units "
        f"({np.sqrt(df['iv30'].mean()) * 100:.1f} vol pts implied vs "
        f"{np.sqrt(df['target_rv_22'].mean()) * 100:.1f} realized), "
        f"positive on {share_pos:.1%} of days"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
