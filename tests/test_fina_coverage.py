"""P0③ 财务因子退市覆盖率审计(吸纳终榜 2026-07-05 P0 第3条)。

价格侧 survivorship 已修(ffill 退出),但财务表侧从未量化:若"后来退市的股"在
历史横截面上系统性缺财务行(数据商不回填退市股财报),ACC/EP 等财务因子的 IC 样本
就有幸存者偏差——崩盘前的差公司恰好不进样本,因子表现被高估。
审计口径:对每个历史时点,把"当时有价格的宇宙"按"今天仍在上市名单"切两组,
分别报 PIT 财务可得率;差距=财务侧幸存者偏差的直接读数。
"""
import math

from scripts.fina_coverage_audit import coverage_split


def test_coverage_split_basic():
    uni = ["a", "b", "c", "d"]          # 当期有价格的宇宙
    have = {"a", "b", "c"}              # PIT 财务可得
    listed_now = {"a", "b"}             # 今天仍上市(c、d 已消失=退市/换牌)
    r = coverage_split(uni, have, listed_now)
    assert r["n"] == 4 and r["n_gone"] == 2
    assert r["cov_listed"] == 1.0       # a、b 全有
    assert r["cov_gone"] == 0.5         # c 有、d 无
    assert abs(r["gap"] - 0.5) < 1e-12  # 缺口=既存组覆盖率−消失组覆盖率


def test_coverage_split_no_gone_group_nan():
    # 某期宇宙全部延续至今 → 消失组覆盖率无定义(NaN),不伪造 0 或 1
    r = coverage_split(["a", "b"], {"a"}, {"a", "b"})
    assert r["n_gone"] == 0
    assert math.isnan(r["cov_gone"]) and math.isnan(r["gap"])


def test_coverage_split_empty_universe_nan():
    r = coverage_split([], set(), set())
    assert r["n"] == 0 and math.isnan(r["cov_listed"])
