"""X-07 标签触发退出:tag_exit_step 纯函数(持仓涨出 🎰/TREND 顶格是否该卖)。

实盘归因动机:2026-07-14 金诚信(🎰+TREND0.85)与紫金按"涨过头"标签卖出,
卖后 30 日 +18%/+21%——该规则从未过实证检验,X-07 补测。
"""
from __future__ import annotations

import pytest


def test_tag_exit_drops_flagged_members_only():
    from scripts.composite_backtest import tag_exit_step

    prev = {"A", "B", "C"}
    d10 = {"A", "B", "C", "D"}          # D 是当期新入档
    flagged = {"B"}                      # B 涨出标签
    tradable = {"A", "B", "C", "D"}
    members = tag_exit_step(prev, d10, flagged, tradable)
    assert members == {"A", "C", "D"}    # B 被标签踢出,其余照常


def test_tag_exit_never_blocks_entry_only_exit():
    """标签只作用于**持仓退出**,不改变入池——新入档的带标签票仍不进(生产已由
    spec_crowd 在候选层拦截),但本函数只管退出侧,不得反向把非持仓票加进来。"""
    from scripts.composite_backtest import tag_exit_step

    members = tag_exit_step(set(), {"X", "Y"}, {"X"}, {"X", "Y"})
    assert members == {"Y"}              # 新档成员里带标签的同样不留


def test_tag_exit_respects_tradable():
    from scripts.composite_backtest import tag_exit_step

    # 不可交易(停牌/一字板)的票不得凭空留在组合里
    assert tag_exit_step({"A"}, {"A", "B"}, set(), {"A"}) == {"A"}


def test_tag_exit_empty_flags_equals_d10():
    from scripts.composite_backtest import tag_exit_step

    d10 = {"A", "B", "C"}
    assert tag_exit_step({"A"}, d10, set(), d10) == d10


def test_tag_exit_rejects_non_set_inputs():
    from scripts.composite_backtest import tag_exit_step

    with pytest.raises(TypeError):
        tag_exit_step(["A"], {"A"}, set(), {"A"})   # 列表会让集合运算静默变形
