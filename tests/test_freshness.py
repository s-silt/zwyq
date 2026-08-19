"""EOD 缓存挂钟新鲜度:防"决策建在上周价格上"的静默降级。"""
from __future__ import annotations

import pytest

from ashare_gauntlet import freshness as fr


def test_weekdays_between_skips_weekend():
    # 20260814(周五)→ 20260817(周一):只有周一 1 个工作日
    assert fr.weekdays_between("20260814", "20260817") == 1
    assert fr.weekdays_between("20260817", "20260817") == 0
    # 跨整周:周一→次周一 = 5 个工作日
    assert fr.weekdays_between("20260810", "20260817") == 5


def test_fresh_when_no_weekday_gap():
    r = fr.classify_cache_freshness("20260818", "20260818")
    assert r["status"] == "FRESH" and r["weekday_gap"] == 0
    # 周五缓存 + 周六看 → 无工作日缺口
    assert fr.classify_cache_freshness("20260814", "20260815")["status"] == "FRESH"


def test_suspect_one_or_two_weekdays():
    r = fr.classify_cache_freshness("20260814", "20260817")   # 缺 1 个工作日
    assert r["status"] == "SUSPECT"
    assert "确认" in str(r["detail"])


def test_stale_three_or_more_weekdays():
    r = fr.classify_cache_freshness("20260810", "20260817")   # 缺 5 个工作日
    assert r["status"] == "STALE" and r["weekday_gap"] == 5
    assert "refresh" in str(r["detail"])


def test_missing_cache_never_fresh():
    r = fr.classify_cache_freshness(None, "20260818")
    assert r["status"] == "MISSING" and r["weekday_gap"] is None


def test_future_cache_and_bad_input_fail_loud():
    with pytest.raises(fr.FreshnessError):
        fr.classify_cache_freshness("20260820", "20260818")   # 缓存晚于今天
    with pytest.raises(fr.FreshnessError):
        fr.classify_cache_freshness("2026818", "20260818")    # 非 8 位
    with pytest.raises(fr.FreshnessError):
        fr.classify_cache_freshness("20261332", "20260818")   # 非真实日期
