"""Tests for the parquet fetch-cache.

Each (endpoint, trade_date) full-market pull is cached to parquet and reused, so
a backfill is paid for once and re-runs of the gauntlet read from disk instead of
re-hitting the mirror.
"""

import pandas as pd

from ashare_gauntlet.data.cache import read_or_fetch


def test_read_or_fetch_fetches_once_then_serves_from_cache(tmp_path):
    calls: list[int] = []

    def fetch() -> pd.DataFrame:
        calls.append(1)
        return pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    path = tmp_path / "daily" / "20240105.parquet"

    df1 = read_or_fetch(path, fetch)  # miss -> fetch + write (creates parent dir)
    df2 = read_or_fetch(path, fetch)  # hit -> read from disk, no second fetch

    assert calls == [1]
    assert path.exists()
    assert list(df2["a"]) == [1, 2]
    pd.testing.assert_frame_equal(df1, df2)
