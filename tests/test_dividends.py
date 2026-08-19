"""dividends 股息叠加(展示层)测试:缺失标 MISSING/UNAVAILABLE 不当 0。"""
from pathlib import Path

import pandas as pd
import pytest

from ashare_gauntlet import dividends as dv


def _cache(tmp_path: Path) -> str:
    return str(tmp_path / "data" / "cache")


def _write_daily_basic(tmp_path: Path, as_of: str, rows: list[dict]) -> None:
    directory = tmp_path / "data" / "cache" / "daily_basic"
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(directory / f"{as_of}.parquet")


def test_reads_dv_ttm_and_marks_missing(tmp_path: Path) -> None:
    _write_daily_basic(tmp_path, "20260818", [
        {"ts_code": "600000.SH", "dv_ttm": 4.2, "dv_ratio": 3.9},
        {"ts_code": "000001.SZ", "dv_ttm": float("nan"), "dv_ratio": 1.1},
    ])
    out = dv.dividend_yields(
        ["600000.SH", "000001.SZ", "600999.SH"], "20260818", cache_dir=_cache(tmp_path))
    assert out["600000.SH"] == {"dv_ttm": 4.2, "dv_ratio": 3.9}
    # NaN → MISSING(None),不伪造 0
    assert out["000001.SZ"]["dv_ttm"] is None
    assert out["000001.SZ"]["dv_ratio"] == 1.1
    # 分区里没有的 code → 两字段皆 None
    assert out["600999.SH"] == {"dv_ttm": None, "dv_ratio": None}


def test_missing_partition_fail_loud(tmp_path: Path) -> None:
    with pytest.raises(dv.DividendDataUnavailable):
        dv.dividend_yields(["600000.SH"], "20260818", cache_dir=_cache(tmp_path))


def test_degraded_partition_without_dividend_columns(tmp_path: Path) -> None:
    # 退化 schema:分区存在但不含 dv 列 → 显式不可用,不静默返回全 None 冒充有数据
    _write_daily_basic(tmp_path, "20260818", [{"ts_code": "600000.SH", "pe": 10.0}])
    with pytest.raises(dv.DividendDataUnavailable):
        dv.dividend_yields(["600000.SH"], "20260818", cache_dir=_cache(tmp_path))


def test_only_dv_ttm_column_present(tmp_path: Path) -> None:
    # 只带 dv_ttm(实测个别历史分区如此):dv_ratio 该字段 None,dv_ttm 正常
    _write_daily_basic(tmp_path, "20260818", [{"ts_code": "600000.SH", "dv_ttm": 2.5}])
    out = dv.dividend_yields(["600000.SH"], "20260818", cache_dir=_cache(tmp_path))
    assert out["600000.SH"]["dv_ttm"] == 2.5
    assert out["600000.SH"]["dv_ratio"] is None


def test_empty_codes_returns_empty(tmp_path: Path) -> None:
    _write_daily_basic(tmp_path, "20260818", [{"ts_code": "600000.SH", "dv_ttm": 4.2}])
    assert dv.dividend_yields([], "20260818", cache_dir=_cache(tmp_path)) == {}


def test_indicative_ttm_cash() -> None:
    assert dv.indicative_ttm_cash(4.2, 10000) == 420.0
    assert dv.indicative_ttm_cash(None, 10000) is None
    assert dv.indicative_ttm_cash(4.2, None) is None
    assert dv.indicative_ttm_cash(float("inf"), 10000) is None
    assert dv.indicative_ttm_cash(float("nan"), 10000) is None
