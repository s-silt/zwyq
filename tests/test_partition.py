"""Tests for data.partition —— 日分区文件枚举(防污染)+ adj_factor 完整性断言。

背景是真实事故不是假设:data/cache/daily/ 曾混入 <code>_<start>_<end>.parquet
形态的整段拉取文件(已人工隔离)。直接 glob *.parquet 的脚本会把它们当日分区读进
面板 → 交易日历/横截面被污染。统一入口只认 ^\\d{8}\\.parquet$,混入文件 warning
列出(surface 不静默)。
"""
import os

import pandas as pd
import pytest

from ashare_gauntlet.data.partition import assert_adj_complete, date_partition_files


def _mk(tmp_path, endpoint: str, names: list[str]) -> None:
    d = tmp_path / endpoint
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        pd.DataFrame({"ts_code": ["600000.SH"], "trade_date": ["20260101"],
                      "close": [10.0]}).to_parquet(d / n, index=False)


# ---------- date_partition_files:只认 ^\d{8}\.parquet$,升序完整路径 ----------

def test_date_partition_files_returns_only_date_named_sorted(tmp_path):
    _mk(tmp_path, "daily", ["20260102.parquet", "20260101.parquet"])
    out = date_partition_files(str(tmp_path), "daily")
    assert [os.path.basename(f) for f in out] == ["20260101.parquet", "20260102.parquet"]
    assert all(os.path.isfile(f) for f in out)   # 返回可直接 read_parquet 的完整路径


def test_date_partition_files_filters_pollution_and_warns(tmp_path, capsys):
    # 真实事故重演:daily/ 混入整段拉取的 <code>_<start>_<end>.parquet
    _mk(tmp_path, "daily", ["20260101.parquet",
                            "600000.SH_20220101_20260101.parquet",
                            "notes.parquet"])
    out = date_partition_files(str(tmp_path), "daily")
    assert [os.path.basename(f) for f in out] == ["20260101.parquet"]
    err = capsys.readouterr().err
    # warning 必须点名列出混入文件(surface 不静默)
    assert "600000.SH_20220101_20260101.parquet" in err
    assert "notes.parquet" in err


def test_date_partition_files_clean_dir_no_warning(tmp_path, capsys):
    _mk(tmp_path, "daily", ["20260101.parquet"])
    date_partition_files(str(tmp_path), "daily")
    assert capsys.readouterr().err == ""


def test_date_partition_files_empty_or_missing_dir_returns_empty(tmp_path):
    assert date_partition_files(str(tmp_path), "daily") == []


def test_date_partition_files_rejects_wrong_digit_count(tmp_path):
    # 7 位 / 9 位数字都不是日分区(严格 8 位,^\d{8}\.parquet$ 全匹配)
    _mk(tmp_path, "daily", ["2026010.parquet", "202601011.parquet", "20260101.parquet"])
    out = date_partition_files(str(tmp_path), "daily")
    assert [os.path.basename(f) for f in out] == ["20260101.parquet"]


# ---------- assert_adj_complete:daily×adj 左合并后 adj_factor 缺行即 raise ----------

def _px(*adj_vals: float) -> pd.DataFrame:
    dates = ["20260101", "20260102", "20260103"][: len(adj_vals)]
    return pd.DataFrame({"ts_code": ["A"] * len(adj_vals), "trade_date": dates,
                         "close": [10.0] * len(adj_vals), "adj_factor": list(adj_vals)})


def test_assert_adj_complete_passes_when_full():
    assert assert_adj_complete(_px(1.0, 1.1, 1.2)) is None


def test_assert_adj_complete_raises_listing_missing_dates():
    # 缺 adj_factor 会让该行 adj_close=NaN 被下游 dropna 静默吞掉 → 必须 fail-loud
    with pytest.raises(SystemExit, match="20260102"):
        assert_adj_complete(_px(1.0, float("nan"), 1.2))


# ---------- 脚本接线:直接 glob daily/ 的入口已改走统一 iterator ----------

def test_screen_load_ignores_polluted_files(tmp_path):
    from scripts.screen import _load
    _mk(tmp_path, "daily", ["20260101.parquet", "600000.SH_20220101_20260101.parquet"])
    df = _load(str(tmp_path), "daily")
    assert len(df) == 1          # 污染文件的行没有混进面板
    assert df["trade_date"].iloc[0] == "20260101"


def test_entry_check_trade_dates_ignores_polluted_files(tmp_path):
    from scripts.entry_check import _trade_dates
    _mk(tmp_path, "daily", ["20260102.parquet", "20260101.parquet",
                            "600000.SH_20220101_20260101.parquet"])
    # 交易日历只能来自日分区文件名;污染文件名前 8 位("600000.S")绝不能进日历
    assert _trade_dates(str(tmp_path)) == ["20260101", "20260102"]
