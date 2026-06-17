"""Parquet fetch-cache for full-market daily pulls.

A backfill of years × endpoints is many thousands of calls; caching each pull to
disk means it is paid for once and every later gauntlet run reads from parquet.
"""

from collections.abc import Callable
from pathlib import Path

import pandas as pd


def read_or_fetch(path: str | Path, fetch_fn: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    """Return the parquet at ``path`` if it exists, else call ``fetch_fn``, cache
    its result to ``path`` (creating parent dirs), and return it.
    """
    path = Path(path)
    if path.exists():
        return pd.read_parquet(path)
    df = fetch_fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df
