"""X-05 namechange PIT 名称面板:历史逐日 ST 状态取代"最终名称近似"(composite 残余偏差)。"""
from __future__ import annotations

import pandas as pd
import pytest


def _changes() -> pd.DataFrame:
    # 乱序输入(函数须自行排序);A 经历 戴帽→摘帽,B 从未变更过一次(单条记录)
    return pd.DataFrame({
        "ts_code": ["000001.SZ", "000001.SZ", "000001.SZ", "600000.SH"],
        "name": ["万科", "ST万科", "万科A", "浦发"],
        "start_date": ["20200101", "20220301", "20230501", "20190101"],
    })


def test_name_asof_is_point_in_time():
    from ashare_gauntlet.namechange import name_asof

    nm = name_asof(_changes(), "20220401")
    assert nm["000001.SZ"] == "ST万科"       # 戴帽期内=当时名称,非最终名称
    assert nm["600000.SH"] == "浦发"
    assert name_asof(_changes(), "20200601")["000001.SZ"] == "万科"
    assert name_asof(_changes(), "20230601")["000001.SZ"] == "万科A"   # 摘帽后
    assert name_asof(_changes(), "20180101").empty                     # 早于一切记录


def test_st_codes_asof_with_fallback():
    from ashare_gauntlet.namechange import st_codes_asof

    # fallback=stock_basic 现名:namechange 无记录的票名称从未变过,现名即历史名
    fallback = pd.Series({"000001.SZ": "万科A", "600000.SH": "浦发", "000002.SZ": "*ST国安"})
    assert st_codes_asof(_changes(), "20220401", fallback) == {"000001.SZ", "000002.SZ"}
    assert st_codes_asof(_changes(), "20230601", fallback) == {"000002.SZ"}


def test_load_namechange_fails_loud_on_bad_start_date(tmp_path):
    from ashare_gauntlet.namechange import load_namechange

    bad = pd.DataFrame({"ts_code": ["A"], "name": ["ST甲"], "start_date": [None]})
    p = tmp_path / "namechange"
    p.mkdir()
    bad.to_parquet(p / "all.parquet")
    with pytest.raises(ValueError):
        load_namechange(str(tmp_path))       # 无效 start_date 不可定位时间轴,拒绝静默丢

    with pytest.raises(FileNotFoundError):
        load_namechange(str(tmp_path / "nowhere"))   # 缓存缺失=先跑 backfill,不静默退化
