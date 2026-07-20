"""Resumable on-disk cache for the walk-forward GARCH forecasts (dev period).

Fitting four GARCH variants monthly over 19 years takes minutes; this cache
lets the computation resume across interrupted runs and spares re-runs. The
refit schedule and simulation seeds are keyed to the ABSOLUTE block index, so a
chunked computation is bit-identical to a single pass.

CLI (used by make phase3 indirectly and for chunked runs):
    python -m src.models.garch_cache <model> [max_blocks]
"""

from __future__ import annotations

import sys

import pandas as pd

from src.config import PROCESSED_DIR
from src.evaluation.walkforward import dev_eval_index, holdout_guard
from src.features.build import load_features
from src.models.garch import garch_forecasts

HORIZONS = (1, 5, 22)
REFIT_EVERY = 22
GARCH_KW = {"min_train": 1250, "simulations": 200}


def _blocks(eval_index: pd.DatetimeIndex) -> list[pd.DatetimeIndex]:
    return [eval_index[i : i + REFIT_EVERY] for i in range(0, len(eval_index), REFIT_EVERY)]


def load_cache(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"garch_dev_{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=[f"h{h}" for h in HORIZONS])


def ensure_garch_dev(name: str, max_blocks: int | None = None) -> tuple[int, int]:
    """Compute up to `max_blocks` missing refit-blocks; returns (done, total)."""
    df = load_features()
    eval_index = dev_eval_index(pd.DatetimeIndex(df.index), 22)
    holdout_guard(eval_index)
    blocks = _blocks(eval_index)
    cache = load_cache(name)

    missing = [(bi, b) for bi, b in enumerate(blocks) if not b.isin(cache.index).all()]
    todo = missing if max_blocks is None else missing[:max_blocks]
    for bi, block in todo:
        fc = garch_forecasts(
            df["ret_cc"],
            block,
            name,
            horizons=HORIZONS,
            refit_every=REFIT_EVERY,
            block_offset=bi,
            **GARCH_KW,
        )
        add = pd.DataFrame({f"h{h}": fc[h] for h in HORIZONS})
        prev = cache[~cache.index.isin(add.index)]
        cache = add if prev.empty else pd.concat([prev, add]).sort_index()
        cache.to_parquet(PROCESSED_DIR / f"garch_dev_{name}.parquet")
    done = len(blocks) - (len(missing) - len(todo))
    return done, len(blocks)


def load_garch_dev(name: str) -> dict[int, pd.Series]:
    """Load the completed cache as horizon -> forecast series (raises if incomplete)."""
    df = load_features()
    eval_index = dev_eval_index(pd.DatetimeIndex(df.index), 22)
    cache = load_cache(name)
    if not eval_index.isin(cache.index).all():
        raise RuntimeError(f"garch cache for {name} incomplete - run ensure_garch_dev")
    cache = cache.reindex(eval_index)
    return {h: cache[f"h{h}"].rename(name) for h in HORIZONS}


def main() -> int:
    name = sys.argv[1]
    max_blocks = int(sys.argv[2]) if len(sys.argv) > 2 else None
    done, total = ensure_garch_dev(name, max_blocks)
    print(f"{name}: {done}/{total} blocks cached")
    return 0 if done == total else 3  # 3 => call again


if __name__ == "__main__":
    raise SystemExit(main())
