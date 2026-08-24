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


def test_all_null_dividend_columns_fail_loud(tmp_path: Path) -> None:
    # 2026-08-24 实测形态:分区存在、列存在,但 dv_ttm/dv_ratio 全市场整列 NULL
    # (上游镜像字段退化)→ 必须 fail-loud,不得静默返回全 None 冒充"无分红"
    _write_daily_basic(tmp_path, "20260824", [
        {"ts_code": "600000.SH", "dv_ttm": None, "dv_ratio": None, "pe": 10.0},
        {"ts_code": "000001.SZ", "dv_ttm": None, "dv_ratio": None, "pe": 8.0},
    ])
    with pytest.raises(dv.DividendDataDegraded):
        dv.dividend_yields(["600000.SH"], "20260824", cache_dir=_cache(tmp_path))


def test_partial_null_market_is_ok(tmp_path: Path) -> None:
    # 部分股票无分红是正常 None(20260821 分区约 1800 只如此),不得误报退化——
    # 即使**本次查询的 code 恰好全是 None**,只要全市场存在非空值就是 OK
    _write_daily_basic(tmp_path, "20260821", [
        {"ts_code": "600000.SH", "dv_ttm": 4.2, "dv_ratio": 3.9},
        {"ts_code": "000001.SZ", "dv_ttm": None, "dv_ratio": None},
        {"ts_code": "300750.SZ", "dv_ttm": None, "dv_ratio": None},
    ])
    out = dv.dividend_yields(
        ["000001.SZ", "300750.SZ"], "20260821", cache_dir=_cache(tmp_path))
    assert out["000001.SZ"] == {"dv_ttm": None, "dv_ratio": None}
    assert out["300750.SZ"] == {"dv_ttm": None, "dv_ratio": None}


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
