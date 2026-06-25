"""Tests for compute_holdscore —— 持有分(质地优先排序分),叠在 tier 之上量化质地。"""
from ashare_gauntlet.record import compute_holdscore


def _rec(grade, np_yoy, dedt, rev, ncr, pe, peg=None):
    return {
        "tier": {"grade": grade},
        "fundamental": {"np_yoy": np_yoy, "dedt_yoy": dedt, "rev_yoy": rev},
        "quality": {"net_cash_ratio": ncr},
        "valuation": {"pe_ttm": pe, "peg": peg},
    }


def test_clean_cheap_triple_growth_scores_high():
    # 🟢 三增 + 扣非≥净利(真) + 现金正 + 便宜 PE13 → 高分
    s = compute_holdscore(_rec("🟢", 40, 45, 30, 1.2, 13.0, 0.4))
    assert s >= 75


def test_landmine_scores_near_zero():
    s = compute_holdscore(_rec("⛔", -50, -60, -20, -0.9, 50.0))
    assert s <= 10


def test_low_base_illusion_is_penalized():
    # 净利暴增但营收没跟上 = 低基数幻觉 → 比同档真三增低
    real = compute_holdscore(_rec("🟢", 40, 42, 35, 1.1, 20.0))
    lowbase = compute_holdscore(_rec("🟢", 260, 250, 8, 1.1, 20.0))
    assert lowbase < real


def test_non_recurring_profit_is_penalized():
    # 净利远超扣非(靠非经常)→ 比扣非≥净利的低
    clean = compute_holdscore(_rec("🟢", 30, 35, 20, 1.0, 20.0))
    dirty = compute_holdscore(_rec("🟢", 50, 12, 20, 1.0, 20.0))
    assert dirty < clean


def test_expensive_valuation_drags_score():
    cheap = compute_holdscore(_rec("🟢", 30, 32, 20, 1.0, 13.0))
    pricey = compute_holdscore(_rec("🟢", 30, 32, 20, 1.0, 70.0))
    assert pricey < cheap


def test_negative_cashflow_penalized_vs_positive():
    pos = compute_holdscore(_rec("🟢", 30, 32, 20, 1.5, 20.0))
    neg = compute_holdscore(_rec("🟢", 30, 32, 20, -0.5, 20.0))
    assert neg < pos


def test_tier_ordering_dominates():
    # 同样数据,🟢 > 🟡 > 🔴(质地基座主导)
    g = compute_holdscore(_rec("🟢", 20, 20, 15, 1.0, 20.0))
    y = compute_holdscore(_rec("🟡", 20, 20, 15, 1.0, 20.0))
    r = compute_holdscore(_rec("🔴", 20, 20, 15, 1.0, 20.0))
    assert g > y > r


def test_handles_missing_fields_gracefully():
    # 缺字段不报错,返回数值
    s = compute_holdscore({"tier": {"grade": "🟢"}, "fundamental": {}, "quality": {}, "valuation": {}})
    assert isinstance(s, (int, float))
