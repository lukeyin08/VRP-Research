"""Explicit trading-calendar alignment.

Master calendar = intersection of SPX and VIX trading dates from SAMPLE_START.
Nothing is forward-filled. Later-launched term-structure series (VIX9D/3M/6M)
carry NaN before their first print. Every dropped date is counted and reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pandas as pd

from src.config import SAMPLE_START


@dataclass
class AlignmentReport:
    n_joint: int
    first: str
    last: str
    dropped_spx_only: int  # dates SPX traded but no VIX print
    dropped_vix_only: int
    term_coverage: dict[str, str]  # series -> first available date
    crosscheck: str

    def __str__(self) -> str:
        lines = [
            f"joint calendar: {self.n_joint} days, {self.first} .. {self.last}",
            f"dates dropped: {self.dropped_spx_only} SPX-only, {self.dropped_vix_only} VIX-only",
            "term structure first dates: "
            + ", ".join(f"{k}={v}" for k, v in self.term_coverage.items()),
            f"cross-source close check: {self.crosscheck}",
        ]
        return "\n".join(lines)


def build_joint(
    spx: pd.DataFrame,
    vix: pd.DataFrame,
    term: dict[str, pd.DataFrame | None],
    spx_check: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, AlignmentReport]:
    spx = spx[spx.index >= SAMPLE_START]
    vix = vix[vix.index >= SAMPLE_START]

    joint_idx = spx.index.intersection(vix.index)
    dropped_spx = len(spx.index.difference(joint_idx))
    dropped_vix = len(vix.index.difference(joint_idx))

    out = pd.DataFrame(index=joint_idx)
    for col in ("open", "high", "low", "close"):
        out[f"spx_{col}"] = spx[col].reindex(joint_idx)
    out["vix_close"] = vix["close"].reindex(joint_idx)

    coverage: dict[str, str] = {}
    for name, df in term.items():
        if df is None or df.empty:
            coverage[name] = "absent"
            continue
        s = df["close"].reindex(joint_idx)  # NaN outside its own history; no fill
        out[f"{name}_close"] = s
        fv = s.first_valid_index()
        coverage[name] = str(cast(pd.Timestamp, fv).date()) if fv is not None else "absent"

    crosscheck = "skipped (no second source)"
    if spx_check is not None:
        both = out[["spx_close"]].join(spx_check["close"].rename("alt"), how="inner").dropna()
        if len(both):
            rel = (both["spx_close"] / both["alt"] - 1.0).abs()
            crosscheck = (
                f"{len(both)} overlapping days vs stooq, median |diff| {rel.median():.2e}, "
                f"max {rel.max():.2e}, days > 20bps: {(rel > 0.002).sum()}"
            )

    rep = AlignmentReport(
        n_joint=len(out),
        first=str(joint_idx.min().date()),
        last=str(joint_idx.max().date()),
        dropped_spx_only=dropped_spx,
        dropped_vix_only=dropped_vix,
        term_coverage=coverage,
        crosscheck=crosscheck,
    )
    return out, rep
