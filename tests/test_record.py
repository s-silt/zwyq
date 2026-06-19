"""TDD:D 结构化数据层 record.py 纯函数(tier_of 规则化四层 / diff_records / merge_factcheck / build_record 装配)。

手工小 fixture,不读 data/cache、不联网。tier_of 偏保守、严格优先级 ⛔→🔴→🟡→🟢。
quality 复用已入库 lit_factors(年报口径),不重复实现。
"""
import pandas as pd
import pytest

from ashare_gauntlet.record import build_record, diff_records, merge_factcheck, tier_of


# ---------------------------------------------------------------------------
# tier_of:传入"半成品 record"(已装配 fundamental/quality/balance/status/flags)
# ---------------------------------------------------------------------------
def _rec(
    *,
    profitable: bool | None = True,
    np_yoy: float | None = 20.0,
    dedt_yoy: float | None = 15.0,
    rev_yoy: float | None = 12.0,
    np_yi: float | None = 5.0,
    dedt_yi: float | None = 4.0,
    ocf: float | None = 6.0,
    goodwill: float | None = 1.0,
    net_assets: float | None = 50.0,
    is_st: bool | None = False,
    flags: list[dict] | None = None,
):
    return {
        "fundamental": {
            "profitable": profitable,
            "np_yoy": np_yoy,
            "dedt_yoy": dedt_yoy,
            "rev_yoy": rev_yoy,
            "np_yi": np_yi,
            "dedt_yi": dedt_yi,
        },
        "quality": {"op_cashflow_yi": ocf},
        "balance": {"goodwill_yi": goodwill, "net_assets_yi": net_assets},
        "status": {"is_st": is_st},
        "flags": flags or [],
    }


def test_tier_green_three_up_clean():
    t = tier_of(_rec())
    assert t["grade"] == "🟢"
    assert t["needs_human"] is False


def test_tier_green_survives_neutral_flag():
    # 提示级旗标(解禁)不应阻断 🟢;只有"警示"级才阻断
    t = tier_of(_rec(flags=[{"type": "解禁", "severity": "提示"}]))
    assert t["grade"] == "🟢"


def test_tier_mine_loss():
    assert tier_of(_rec(profitable=False, np_yi=-1.0))["grade"] == "⛔"


def test_tier_mine_double_severe_decline():
    # 净利&扣非同比双 ≤ -40%(仍盈利,但重度恶化)
    assert tier_of(_rec(np_yoy=-45.0, dedt_yoy=-50.0))["grade"] == "⛔"


def test_tier_mine_goodwill_over_net_assets():
    # 商誉/净资产 = 30/50 = 60% > 50%
    assert tier_of(_rec(goodwill=30.0, net_assets=50.0))["grade"] == "⛔"


def test_tier_mine_st():
    assert tier_of(_rec(is_st=True))["grade"] == "⛔"


def test_tier_red_dedt_divergence():
    # 净利为正但扣非绝对值为负 —— 题材背离
    t = tier_of(_rec(np_yi=3.0, dedt_yi=-1.0))
    assert t["grade"] == "🔴"
    assert t["needs_human"] is True


def test_tier_red_revenue_up_profit_down():
    # 增收不增利:营收+10% 但净利 -5%(扣非绝对值仍为正,排除背离路径)
    t = tier_of(_rec(rev_yoy=10.0, np_yoy=-5.0, dedt_yoy=-8.0, np_yi=2.0, dedt_yi=1.0))
    assert t["grade"] == "🔴"
    assert t["needs_human"] is True


def test_tier_yellow_negative_cashflow():
    # 三增但经营现金流 < 0
    t = tier_of(_rec(ocf=-2.0))
    assert t["grade"] == "🟡"


def test_tier_yellow_negative_cashflow_profitable_none_no_prefix():
    # [A3] profitable=None(净利绝对值缺失)+ ocf<0:不得断言"盈利但",无依据
    t = tier_of(_rec(profitable=None, np_yi=None, ocf=-2.0))
    assert t["grade"] == "🟡"
    assert any("经营现金流<0" in r for r in t["reasons"])
    assert not any("盈利但" in r for r in t["reasons"])


def test_tier_yellow_negative_cashflow_profitable_true_keeps_prefix():
    # [A3] profitable=True 时仍保留"盈利但"前缀(有依据)
    t = tier_of(_rec(profitable=True, ocf=-2.0))
    assert any("盈利但经营现金流<0" in r for r in t["reasons"])


def test_tier_yellow_low_base():
    # 疑低基数:净利 +50% 但营收 +5%
    t = tier_of(_rec(np_yoy=50.0, rev_yoy=5.0))
    assert t["grade"] == "🟡"
    assert t["needs_human"] is True


def test_tier_yellow_high_pledge_warn_flag():
    t = tier_of(_rec(flags=[{"type": "质押", "severity": "警示"}]))
    assert t["grade"] == "🟡"


def test_tier_yellow_missing_data_downgrade():
    # 缺 dedt_yoy → 不得判 🟢,降级 🟡 + 数据缺失 + needs_human
    t = tier_of(_rec(dedt_yoy=None))
    assert t["grade"] == "🟡"
    assert t["needs_human"] is True
    assert any("数据缺失" in r for r in t["reasons"])


def test_tier_dedt_abs_missing_blocks_green():
    # [A1] 净利绝对值为正但扣非绝对值缺失(None)→ 无法核扣非背离,
    # 不得凭 dedt_yoy>0 直达 🟢;保守降级 🟡 + needs_human 并标注无法核
    t = tier_of(_rec(np_yi=5.0, dedt_yi=None))
    assert t["grade"] == "🟡"
    assert t["needs_human"] is True
    assert any("扣非绝对值缺失" in r and "无法核" in r for r in t["reasons"])


def test_tier_dedt_abs_present_negative_still_red():
    # [A1] 守卫不得误伤:扣非绝对值确为负时仍走 🔴 背离,不被新降级吞掉
    t = tier_of(_rec(np_yi=5.0, dedt_yi=-1.0))
    assert t["grade"] == "🔴"


def test_tier_np_abs_missing_no_spurious_dedt_downgrade():
    # [A1] 净利绝对值缺失(np_yi=None)时,扣非背离判据本就 N/A,
    # 不应因 dedt_yi 缺失而触发"无法核扣非背离"降级(背离前提是 np_yi>0)
    t = tier_of(_rec(np_yi=None, dedt_yi=None))
    assert not any("无法核扣非背离" in r for r in t["reasons"])


def test_tier_priority_mine_beats_red():
    # 既亏损(⛔)又扣非负(🔴)→ ⛔ 优先
    assert tier_of(_rec(profitable=False, np_yi=-2.0, dedt_yi=-3.0))["grade"] == "⛔"


# ---------------------------------------------------------------------------
# [A5] tier_of 边界/优先级(characterization:锁住现有正确行为,非 red-first)
# ---------------------------------------------------------------------------
def test_double_decline_boundary_minus40():
    # 净利&扣非同比恰好 = -40%(<= DECLINE_SEVERE)→ ⛔(边界含等号)
    t = tier_of(_rec(np_yoy=-40.0, dedt_yoy=-40.0))
    assert t["grade"] == "⛔"
    assert any("双降" in r for r in t["reasons"])


def test_double_decline_needs_both():
    # 仅净利 -45% 而扣非同比缺失(None)→ 不判 ⛔(双降需两者俱在)。
    # rev_yoy 也置 None 以隔离 🔴"增收不增利"路径(rev_yoy>0 才触发),
    # 从而验证落点为 🟡 数据缺失降级而非 ⛔。
    t = tier_of(_rec(np_yoy=-45.0, dedt_yoy=None, rev_yoy=None))
    assert t["grade"] == "🟡"
    assert t["needs_human"] is True
    assert any("数据缺失" in r for r in t["reasons"])


def test_mine_beats_yellow():
    # ST(⛔)+ 经营现金流<0(🟡 候选)→ ⛔ 优先,严格 ⛔>🟡
    t = tier_of(_rec(is_st=True, ocf=-2.0))
    assert t["grade"] == "⛔"
    assert any("ST" in r for r in t["reasons"])


def test_tier_yellow_fallback_describes_failure():
    # 净利/扣非/营收均下滑但未及 ⛔/🔴 阈值(np-37/dedt-39/rev-5,扣非绝对值仍正)→ 🟡 兜底
    # reason 必须说明缺口(诚实面板要可读),而非泛泛"未满足全条件"
    t = tier_of(_rec(np_yoy=-37.0, dedt_yoy=-39.0, rev_yoy=-5.0, dedt_yi=0.5, ocf=3.0))
    assert t["grade"] == "🟡"
    assert t["needs_human"] is True
    assert any("净利未增" in r for r in t["reasons"])


# ---------------------------------------------------------------------------
# [A6] missing-data 降级参数化(characterization:缺任一 🟢 必需输入 → 🟡 + needs_human)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("field", ["np_yoy", "rev_yoy", "ocf"])
def test_tier_missing_required_input_downgrades(field):
    t = tier_of(_rec(**{field: None}))
    assert t["grade"] == "🟡"
    assert t["needs_human"] is True
    assert any("数据缺失" in r for r in t["reasons"])


# ---------------------------------------------------------------------------
# diff_records
# ---------------------------------------------------------------------------
def test_diff_records_tier_and_flags_and_field():
    old = {
        "ts_code": "601138.SH",
        "tier": {"grade": "🟢"},
        "entry": {"grade": "A"},
        "fundamental": {"np_yoy": 20.0},
        "valuation": {"pe_ttm": 30.0},
        "flags": [],
    }
    new = {
        "ts_code": "601138.SH",
        "tier": {"grade": "🟡"},
        "entry": {"grade": "B"},
        "fundamental": {"np_yoy": -5.0},
        "valuation": {"pe_ttm": 31.0},
        "flags": [{"type": "减持", "severity": "提示"}],
    }
    d = diff_records(old, new)
    assert d["tier_change"] == ["🟢", "🟡"]
    assert d["entry_change"] == ["A", "B"]
    assert "减持" in d["new_flags"]
    assert any(c["path"] == "fundamental.np_yoy" for c in d["field_changes"])


def test_diff_records_non_numeric_string_value_no_raise():
    # [A4] 回读的历史/外部 cards JSON 在 DIFF_FIELDS 塞了非数值字符串("N/A")
    # 不得抛 ValueError;转不出 float 的按"变化"记入 field_changes
    old = {
        "ts_code": "601138.SH",
        "tier": {"grade": "🟢"}, "entry": {"grade": "A"},
        "fundamental": {"np_yoy": "N/A", "dedt_yoy": 15.0},
        "valuation": {"pe_ttm": 30.0},
        "flags": [],
    }
    new = {
        "ts_code": "601138.SH",
        "tier": {"grade": "🟢"}, "entry": {"grade": "A"},
        "fundamental": {"np_yoy": 20.0, "dedt_yoy": 15.0},
        "valuation": {"pe_ttm": 30.0},
        "flags": [],
    }
    d = diff_records(old, new)  # 不抛
    # np_yoy: "N/A"->20.0 算变化;dedt_yoy/pe_ttm 不变
    paths = {c["path"] for c in d["field_changes"]}
    assert "fundamental.np_yoy" in paths
    assert "fundamental.dedt_yoy" not in paths
    assert "valuation.pe_ttm" not in paths


def test_diff_records_both_non_numeric_equal_string_no_change():
    # [A4] 两侧同为非数值字符串("N/A"=="N/A")→ 转不出 float,但相等,不应记为变化
    old = {
        "ts_code": "X", "tier": {"grade": "🟢"}, "entry": {"grade": "A"},
        "fundamental": {"np_yoy": "N/A", "dedt_yoy": 15.0},
        "valuation": {"pe_ttm": 30.0}, "flags": [],
    }
    new = {
        "ts_code": "X", "tier": {"grade": "🟢"}, "entry": {"grade": "A"},
        "fundamental": {"np_yoy": "N/A", "dedt_yoy": 15.0},
        "valuation": {"pe_ttm": 30.0}, "flags": [],
    }
    d = diff_records(old, new)
    assert all(c["path"] != "fundamental.np_yoy" for c in d["field_changes"])


def test_diff_records_no_change():
    rec = {
        "ts_code": "X",
        "tier": {"grade": "🟢"},
        "entry": {"grade": "A"},
        "fundamental": {"np_yoy": 20.0},
        "valuation": {"pe_ttm": 30.0},
        "flags": [],
    }
    d = diff_records(rec, rec)
    assert d["tier_change"] is None
    assert d["entry_change"] is None
    assert d["new_flags"] == []
    assert d["dropped_flags"] == []


def test_diff_dropped_flags():
    # [A7] 旧有旗、新无旗 → dropped_flags 非空(且 new_flags 空)
    old = {
        "ts_code": "X", "tier": {"grade": "🟢"}, "entry": {"grade": "A"},
        "fundamental": {"np_yoy": 20.0}, "valuation": {"pe_ttm": 30.0},
        "flags": [{"type": "质押", "severity": "警示"}],
    }
    new = {
        "ts_code": "X", "tier": {"grade": "🟢"}, "entry": {"grade": "A"},
        "fundamental": {"np_yoy": 20.0}, "valuation": {"pe_ttm": 30.0},
        "flags": [],
    }
    d = diff_records(old, new)
    assert "质押" in d["dropped_flags"]
    assert d["new_flags"] == []


def test_diff_field_none_transition():
    # [A7] 字段 None↔有值 记入 field_changes(任一为 None 即变化,不抛)
    old = {
        "ts_code": "X", "tier": {"grade": "🟢"}, "entry": {"grade": "A"},
        "fundamental": {"np_yoy": None, "dedt_yoy": 15.0},
        "valuation": {"pe_ttm": 30.0}, "flags": [],
    }
    new = {
        "ts_code": "X", "tier": {"grade": "🟢"}, "entry": {"grade": "A"},
        "fundamental": {"np_yoy": 20.0, "dedt_yoy": None},
        "valuation": {"pe_ttm": 30.0}, "flags": [],
    }
    d = diff_records(old, new)
    paths = {c["path"]: (c["old"], c["new"]) for c in d["field_changes"]}
    assert paths["fundamental.np_yoy"] == (None, 20.0)
    assert paths["fundamental.dedt_yoy"] == (15.0, None)


# ---------------------------------------------------------------------------
# merge_factcheck(铁律:绝不覆盖接口口径数字,只填 factcheck 键)
# ---------------------------------------------------------------------------
def test_merge_factcheck_does_not_overwrite_interface_numbers():
    record = {
        "ts_code": "601138.SH",
        "fundamental": {"np_yi": 5.0, "np_yoy": 20.0},
        "valuation": {"pe_ttm": 30.0},
        "factcheck": None,
    }
    fc = {
        "confirmed": True,
        "q1_net_profit_yi": 3.2,
        "disputes": ["x"],
        "news": ["y"],
        "verified_at": "20260619",
    }
    out = merge_factcheck(record, fc)
    # 接口数字原封不动
    assert out["fundamental"]["np_yi"] == 5.0
    assert out["valuation"]["pe_ttm"] == 30.0
    # 核实结论进独立 factcheck 键,净利只进 q1_net_profit_yi(与 fundamental.np_yi 并列)
    assert out["factcheck"]["confirmed"] is True
    assert out["factcheck"]["q1_net_profit_yi"] == 3.2
    # 不就地改原对象
    assert record["factcheck"] is None


def test_merge_factcheck_none_resets():
    # [A7] fc=None 把旧 factcheck 重置为 None;不就地改原对象;不动 tier
    record = {
        "ts_code": "601138.SH",
        "fundamental": {"np_yi": 5.0},
        "tier": {"grade": "🟢"},
        "factcheck": {"confirmed": True, "q1_net_profit_yi": 3.2},
    }
    out = merge_factcheck(record, None)
    assert out["factcheck"] is None
    # tier 不被触碰
    assert out["tier"] == {"grade": "🟢"}
    # 不 mutate 原对象
    assert record["factcheck"] == {"confirmed": True, "q1_net_profit_yi": 3.2}
    assert out is not record


# ---------------------------------------------------------------------------
# build_record 集成(最小 fixture,验证装配 + 复用 lit_factors 年报口径 + meta)
# ---------------------------------------------------------------------------
def _daily_adj(code="601138.SH", n=25):
    dates = [f"202605{i:02d}" for i in range(1, n + 1)]
    closes = [100.0 + i for i in range(n)]
    daily = pd.DataFrame({"ts_code": code, "trade_date": dates, "close": closes, "amount": [1e6] * n})
    adj = pd.DataFrame({"ts_code": code, "trade_date": dates, "adj_factor": [1.0] * n})
    return daily, adj


def _fund_tables():
    income = pd.DataFrame(
        [
            {"end_date": "20251231", "total_revenue": 80e8, "n_income_attr_p": 8e8},
            {"end_date": "20260331", "total_revenue": 20e8, "n_income_attr_p": 2e8},
        ]
    )
    fina = pd.DataFrame(
        [
            {"end_date": "20251231", "or_yoy": 10.0, "netprofit_yoy": 18.0,
             "dt_netprofit_yoy": 16.0, "grossprofit_margin": 29.0, "profit_dedt": 7.5e8},
            {"end_date": "20260331", "or_yoy": 12.0, "netprofit_yoy": 20.0,
             "dt_netprofit_yoy": 15.0, "grossprofit_margin": 30.0, "profit_dedt": 1.8e8},
        ]
    )
    cashflow = pd.DataFrame([{"end_date": "20251231", "n_cashflow_act": 10e8}])
    balancesheet = pd.DataFrame(
        [
            {"end_date": "20251231", "total_assets": 200e8, "total_hldr_eqy_exc_min_int": 120e8,
             "goodwill": 5e8, "accounts_receiv": 10e8, "money_cap": 30e8},
            {"end_date": "20260331", "total_assets": 210e8, "total_hldr_eqy_exc_min_int": 122e8,
             "goodwill": 5e8, "accounts_receiv": 11e8, "money_cap": 28e8},
        ]
    )
    empty = pd.DataFrame()
    return {
        "income": income,
        "fina_indicator": fina,
        "balancesheet": balancesheet,
        "cashflow": cashflow,
        "share_float": empty,
        "pledge_stat": empty,
        "stk_holdertrade": empty,
        "namechange": empty,
        "forecast": empty,
        "express": empty,
    }


def test_build_record_assembles_and_reuses_lit_factors():
    code = "601138.SH"
    daily, adj = _daily_adj(code)
    from ashare_gauntlet.factsheet import market_returns

    mr = market_returns(daily, adj, (5, 20))
    db_row = {"pe_ttm": 20.0, "pb": 2.0, "total_mv": 500000.0}  # 万元 -> 50 亿
    rec = build_record(
        code,
        name="测试股",
        industry="电池",
        as_of="20260618",
        daily_sub=daily,
        adj_sub=adj,
        mr=mr,
        fund_tables=_fund_tables(),
        db_row=db_row,
    )
    # 结构
    for k in ("ts_code", "name", "industry", "as_of", "tier", "entry", "technical",
              "valuation", "fundamental", "balance", "quality", "status", "flags", "factcheck", "meta"):
        assert k in rec
    assert rec["factcheck"] is None
    # 基本面(最新季 20260331)
    assert rec["fundamental"]["end_date"] == "20260331"
    assert rec["fundamental"]["np_yi"] == pytest.approx(2.0)
    assert rec["fundamental"]["dedt_yi"] == pytest.approx(1.8)
    # 估值
    assert rec["valuation"]["pe_ttm"] == pytest.approx(20.0)
    assert rec["valuation"]["mv_yi"] == pytest.approx(50.0)
    assert rec["valuation"]["peg"] == pytest.approx(1.0)  # 20 / 20
    # quality 复用 lit_factors 年报口径:净现比=10/8=1.25,应计=(8-10)/200=-0.01,OCF年报=10
    assert rec["quality"]["net_cash_ratio"] == pytest.approx(1.25)
    assert rec["quality"]["accrual"] == pytest.approx(-0.01)
    assert rec["quality"]["op_cashflow_yi"] == pytest.approx(10.0)
    # tier:三增 + 现金流>0 + 无警示 → 🟢
    assert rec["tier"]["grade"] == "🟢"
    # 口径标注:每个非空数值叶子有 meta
    assert "valuation.pe_ttm" in rec["meta"]
    assert rec["meta"]["valuation.pe_ttm"]["source"]
    assert "quality.net_cash_ratio" in rec["meta"]
    # [A2] technical.vol_ratio 是被渲染的数值叶子(render_md "量比"),口径标注不得漏
    assert "technical.vol_ratio" in rec["meta"]
    vr_meta = rec["meta"]["technical.vol_ratio"]
    assert vr_meta["unit"]
    assert vr_meta["as_of"]
    assert vr_meta["source"]


# ---------------------------------------------------------------------------
# review follow-up #1:负净资产(资不抵债)→ ⛔(不被 na>0 守卫跳过)
# ---------------------------------------------------------------------------
def test_tier_mine_negative_net_assets():
    # 净资产≤0(资不抵债)比"商誉占净资产>50%"更该出局;原 na>0 守卫会把它漏掉
    t = tier_of(_rec(net_assets=-3.0, goodwill=1.0))
    assert t["grade"] == "⛔"
    assert any("资不抵债" in r or "净资产" in r for r in t["reasons"])


def test_tier_missing_net_assets_not_fabricated_mine():
    # 净资产缺失(None)不得伪造资不抵债;其余指标正常 → 不应判 ⛔
    t = tier_of(_rec(net_assets=None, goodwill=None))
    assert t["grade"] != "⛔"


# ---------------------------------------------------------------------------
# review follow-up #3:质押旗标带结构化数值 value(renderer 不再反解析字符串)
# ---------------------------------------------------------------------------
def test_build_flags_pledge_carries_structured_value():
    from ashare_gauntlet.record import _build_flags

    empty = pd.DataFrame()
    fund_tables = {
        "pledge_stat": pd.DataFrame([{"end_date": "20260331", "pledge_ratio": 60.0}]),
        "share_float": empty,
        "stk_holdertrade": empty,
        "forecast": empty,
        "express": empty,
    }
    flags = _build_flags(fund_tables, "20260618", "20260331")
    pledge = next(f for f in flags if f["type"] == "质押")
    assert pledge["severity"] == "警示"  # 60% >= 50 阈值
    assert pledge["value"] == pytest.approx(60.0)
