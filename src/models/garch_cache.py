"""Resumable on-disk cache for walk-forward GARCH forecasts.

Fitting four GARCH variants monthly over decades takes minutes; this cache
lets the computation resume across interrupted runs and spares re-runs. The
refit schedule and simulation seeds are keyed to the ABSOLUTE block index
within each window, so a chunked computation is bit-identical to a single pass.

Windows: "dev" (Phases 2-5) and "holdout" (Phase 7 only).

CLI: python -m src.models.garch_cache <model> <dev|holdout> [max_blocks]
"""

from __future__ import annotations

import sys

import pandas as pd

from src.config import PROCESSED_DIR
from src.evaluation.walkforward import dev_eval_index, holdout_eval_index, holdout_guard
from src.features.build import load_features
from src.models.garch import garch_forecasts

HORIZONS = (1, 5, 22)
REFIT_EVERY = 22
GARCH_KW = {"min_train": 1250, "simulations": 200}
# distinct seed streams for the two windows so holdout sims are independent
_BLOCK_OFFSET = {"dev": 0, "holdout": 10_000}


def _eval_index(df: pd.DataFrame, which: str) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(df.index)
    if which == "dev":
        out = dev_eval_index(idx, 22)
        holdout_guard(out)
        return out
    if which == "holdout":
        return holdout_eval_index(idx, 22)
    raise ValueError(f"unknown window {which!r}")


def _blocks(eval_index: pd.DatetimeIndex) -> list[pd.DatetimeIndex]:
    return [eval_index[i : i + REFIT_EVERY] for i in range(0, len(eval_index), REFIT_EVERY)]


def load_cache(name: str, which: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"garch_{which}_{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=[f"h{h}" for h in HORIZONS])


def ensure_garch(name: str, which: str, max_blocks: int | None = None) -> tuple[int, int]:
    """Compute up to `max_blocks` missing refit-blocks; returns (done, total)."""
    df = load_features()
    eval_index = _eval_index(df, which)
    blocks = _blocks(eval_index)
    cache = load_cache(name, which)

    missing = [(bi, b) for bi, b in enumerate(blocks) if not b.isin(cache.index).all()]
    todo = missing if max_blocks is None else missing[:max_blocks]
    for bi, block in todo:
        fc = garch_forecasts(
            df["ret_cc"],
            block,
            name,
            horizons=HORIZONS,
            refit_every=REFIT_EVERY,
            block_offset=_BLOCK_OFFSET[which] + bi,
            **GARCH_KW,
        )
        add = pd.DataFrame({f"h{h}": fc[h] for h in HORIZONS})
        prev = cache[~cache.index.isin(add.index)]
        cache = add if prev.empty else pd.concat([prev, add]).sort_index()
        cache.to_parquet(PROCESSED_DIR / f"garch_{which}_{name}.parquet")
    done = len(blocks) - (len(missing) - len(todo))
    return done, len(blocks)


def load_garch(name: str, which: str) -> dict[int, pd.Series]:
    """Load a completed cache as horizon -> forecast series (raises if incomplete)."""
    df = load_features()
    eval_index = _eval_index(df, which)
    cache = load_cache(name, which)
    if not eval_index.isin(cache.index).all():
        raise RuntimeError(f"garch cache {which}/{name} incomplete - run ensure_garch")
    cache = cache.reindex(eval_index)
    return {h: cache[f"h{h}"].rename(name) for h in HORIZONS}


# Backwards-compatible dev-window aliases (used by run_phase3)
def ensure_garch_dev(name: str, max_blocks: int | None = None) -> tuple[int, int]:
    return ensure_garch(name, "dev", max_blocks)


def load_garch_dev(name: str) -> dict[int, pd.Series]:
    return load_garch(name, "dev")


def main() -> int:
    name, which = sys.argv[1], sys.argv[2]
    max_blocks = int(sys.argv[3]) if len(sys.argv) > 3 else None
    done, total = ensure_garch(name, which, max_blocks)
    print(f"{which}/{name}: {done}/{total} blocks cached")
    return 0 if done == total else 3  # 3 => call again


if __name__ == "__main__":
    raise SystemExit(main())
