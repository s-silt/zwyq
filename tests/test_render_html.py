"""TDD:多股 HTML dashboard 观察名单总览 render_dashboard 纯函数(无 IO,返回自含单文件 HTML)。

断言结构性质而非像素:含 <table、含排序 JS(sort/onclick)、含过滤控件(tier/行业)、
含组合层摘要(行业集中度文字 / 市值分桶 / tier 分布 / 口径边界且无资金分配数字)、
含全部 41 行(真实底座)、None 值占位("—"/"N/A")而非 0、离散档文字(色盲安全)、
事件中性(无红色恐吓触发器)、每只行内嵌 B 的 render_svg_card 可钻取、结论不裸出小数 score。

手工小 fixture + 真实底座抽样,纯渲染、不联网。
"""
import json
import os
import re

import pytest

from ashare_gauntlet.render_html import render_dashboard

CARDS = os.path.join("data", "cards", "20260618.json")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def _rec(
    *,
    ts_code: str = "601138.SH",
    name: str = "工业富联",
    industry: str = "通信设备",
    grade: str = "🟢",
    tier_words: str = "强且干净",
    entry_grade: str = "A",
    score: float = 91.4,
    close: float | None = 78.08,
    dist60: float | None = -3.66,
    pct20: float | None = 91.4,
    ret20: float | None = 19.2,
    trend: str = "多头",
    rsi: float | None = 61.0,
    vol_ratio: float | None = 1.75,
    pe_ttm: float | None = 38.1,
    pb: float | None = 8.79,
    peg: float | None = 0.37,
    mv_yi: float | None = 15494.2,
    np_yoy: float | None = 102.5,
    dedt_yoy: float | None = 109.0,
    net_cash_ratio: float | None = 1.2,
    accrual: float | None = -0.02,
    recv_pct: float | None = 290.0,
    flags: list[dict] | None = None,
    needs_human: bool = False,
) -> dict:
    return {
        "ts_code": ts_code,
        "name": name,
        "industry": industry,
        "as_of": "20260618",
        "tier": {"grade": grade, "reasons": [tier_words], "needs_human": needs_human},
        "entry": {"grade": entry_grade, "score": score, "tag": "趋势内"},
        "technical": {
            "close": close, "dist60": dist60, "pct20": pct20, "trend": trend,
            "rsi": rsi, "ret20": ret20, "vol_ratio": vol_ratio,
        },
        "valuation": {"pe_ttm": pe_ttm, "pb": pb, "peg": peg, "mv_yi": mv_yi},
        "fundamental": {
            "end_date": "20260331", "rev_yi": 2510.7, "rev_yoy": 56.5,
            "np_yi": 105.9, "np_yoy": np_yoy, "dedt_yoy": dedt_yoy, "dedt_yi": 102.4,
            "gross_margin": 7.35, "profitable": True,
        },
        "balance": {
            "accounts_receiv_yi": 1025.0, "goodwill_yi": 3.28, "money_cap_yi": 1021.3,
            "net_assets_yi": 1762.1, "total_assets_yi": 4571.8,
            "recv_to_annual_net_pct": recv_pct,
        },
        "quality": {"op_cashflow_yi": 52.3, "net_cash_ratio": net_cash_ratio, "accrual": accrual},
        "status": {"is_st": False, "ever_st": False, "current_name": name},
        "flags": flags if flags is not None else [
            {"type": "质押", "severity": "提示",
             "fact": "控股股东体系质押0.3%", "date": "20260618", "source": "pledge_stat接口"},
        ],
        "meta": {
            "technical.close": {"unit": "元", "as_of": "20260618", "source": "daily前复权"},
            "valuation.pe_ttm": {"unit": "倍", "as_of": "20260618", "source": "daily_basic接口"},
            "fundamental.np_yoy": {"unit": "%", "as_of": "20260331", "source": "fina_indicator"},
            "quality.net_cash_ratio": {"unit": "倍", "as_of": "20251231", "source": "cashflow(年报)"},
        },
        "factcheck": None,
        "theme": "",
        "tags": [],
    }


def _mixed_cohort() -> list[dict]:
    """覆盖四档 + 多行业 + 一条缺失值记录,供组合摘要/过滤/占位断言。"""
    return [
        _rec(ts_code="601138.SH", name="工业富联", industry="通信设备", grade="🟢", tier_words="强干净"),
        _rec(ts_code="002463.SZ", name="沪电股份", industry="元器件", grade="🟢", tier_words="强干净", score=97.0),
        _rec(ts_code="000001.SZ", name="某瑕疵", industry="元器件", grade="🟡", tier_words="盈利瑕疵", score=70.0),
        _rec(ts_code="000002.SZ", name="某背离", industry="软件服务", grade="🔴", tier_words="题材背离", score=55.0),
        _rec(
            ts_code="000003.SZ", name="某地雷", industry="互联网", grade="⛔", tier_words="地雷出局",
            score=10.0, needs_human=True,
            # 全数值缺失:验证 None 占位
            close=None, dist60=None, pct20=None, ret20=None, rsi=None, vol_ratio=None,
            pe_ttm=None, pb=None, peg=None, mv_yi=None, np_yoy=None, dedt_yoy=None,
            net_cash_ratio=None, accrual=None, recv_pct=None,
            flags=[],
        ),
    ]


# ---------------------------------------------------------------------------
# 自含单文件 HTML 基本结构
# ---------------------------------------------------------------------------
def test_returns_self_contained_html_document():
    html = render_dashboard(_mixed_cohort())
    assert isinstance(html, str)
    low = html.lower()
    assert "<!doctype html" in low
    assert "<html" in low and "</html>" in low
    # 自含:内联 <style> 与 <script>,不强依赖外链 css/js
    assert "<style" in low
    assert "<script" in low


def test_has_sortable_table_with_onclick_sort_js():
    html = render_dashboard(_mixed_cohort())
    assert "<table" in html.lower()
    # 排序:表头可点 + JS 含 sort 关键字
    assert "onclick" in html.lower()
    assert "sort" in html.lower()
    # 表头(<th>)存在
    assert "<th" in html.lower()


def test_has_filter_controls_for_tier_and_industry():
    html = render_dashboard(_mixed_cohort())
    low = html.lower()
    # 过滤控件:select / option / 或 data 属性驱动的过滤
    assert ("<select" in low) or ("filter" in low)
    # 行业过滤须能选到出现过的行业名
    assert "通信设备" in html
    assert "元器件" in html
    # tier 过滤须有四档文字之一可选
    assert ("质地" in html) or ("tier" in low)


# ---------------------------------------------------------------------------
# 列内容:三条分项条 + sparkline + 旗标数 + 离散档
# ---------------------------------------------------------------------------
def test_table_has_three_dimension_bars_tech_fund_flow():
    html = render_dashboard(_mixed_cohort())
    # 三条分项条(技术/基本/资金)——文字标注须出现
    assert "技术" in html
    assert "基本" in html
    assert "资金" in html


def test_table_has_flag_count_column():
    html = render_dashboard(_mixed_cohort())
    assert ("旗" in html) or ("flag" in html.lower())


def test_tier_discrete_grade_with_text_label_colorblind_safe():
    """色盲安全:每档 emoji 图标 + 文字档名(颜色之外必有文字)。"""
    html = render_dashboard(_mixed_cohort())
    for g in ("🟢", "🟡", "🔴", "⛔"):
        assert g in html
    # 文字档名(非仅靠颜色)
    assert ("强" in html) or ("强干净" in html)
    assert "地雷" in html


def test_entry_grade_letter_present_not_raw_score():
    """结论用离散档,不把 entry.score 小数当结论裸出。"""
    html = render_dashboard(_mixed_cohort())
    assert "A" in html  # entry 档字母
    # 91.4 不应作为结论裸出现(允许在 svg 钻取的 title/score 之外不出现)
    assert "91.4" not in html


# ---------------------------------------------------------------------------
# 组合层摘要(顶部,展示层)
# ---------------------------------------------------------------------------
def test_portfolio_summary_industry_concentration_text():
    html = render_dashboard(_mixed_cohort())
    # 行业集中度:须含"集中度"或"行业"占比文字 + top 行业名
    assert ("集中度" in html) or ("行业分布" in html) or ("行业" in html)
    # top 行业(元器件 在 mixed 中出现 2 次,为并列最多之一)应被点名
    assert "元器件" in html


def test_portfolio_summary_mv_buckets():
    html = render_dashboard(_mixed_cohort())
    # 市值分布分桶:须含"市值"且有分桶语义(亿 / 桶 / 区间)
    assert "市值" in html
    assert "亿" in html


def test_portfolio_summary_tier_distribution():
    html = render_dashboard(_mixed_cohort())
    # tier 分布:四档计数。mixed 中 🟢2 🟡1 🔴1 ⛔1
    # 强化:断言具体 chip 片段(🟢 强干净 <b>2</b>),而非 "2" in html(几乎恒真,捕不到计数错)
    assert "🟢 强干净 <b>2</b>" in html
    assert "🟡 盈利瑕疵 <b>1</b>" in html
    assert "🔴 题材背离 <b>1</b>" in html
    assert "⛔ 地雷出局 <b>1</b>" in html


def test_portfolio_summary_mv_bucket_counts_specific():
    """市值桶计数须落到具体 <b>N</b> 片段(mixed:4 只 mv_yi=15494.2 → 大盘桶,1 只缺失)。"""
    html = render_dashboard(_mixed_cohort())
    # 4 只大市值入 ≥3000亿(大盘)桶
    assert "≥3000亿(大盘) <b>4</b>" in html
    # 1 只全缺失 → 市值缺失计 1
    assert "市值缺失 <b>1</b>" in html


def test_portfolio_summary_emits_no_capital_allocation_numbers():
    """展示层不得出现可照抄的分配数字。mixed 有 2 只 🟢×entryA,旧口径会渲染
    「60,000 元预算等权占满 2 只,单只 30,000 元」——那份分配绕过 D10/composite/
    fact-check/治理否决/行业上限,且选股依据是 §11 已否决的择时透镜。"""
    html = render_dashboard(_mixed_cohort())
    assert "30,000 元" not in html
    assert "60,000" not in html
    assert "等权占满" not in html
    assert "本页不做资金分配" in html


def test_portfolio_summary_points_buy_sizing_to_frozen_snapshot():
    """口径边界须点名唯一出处(data/decisions 冻结快照)并保留非建议措辞;
    "决策台"命名收回——"决策"二字只属于冻结快照。"""
    html = render_dashboard(_mixed_cohort())
    assert "data/decisions" in html
    assert "建议" in html          # 页脚「不构成选股、择时或仓位建议」
    assert "决策台" not in html


def test_empty_records_page_also_drops_decision_desk_naming():
    """空 records 分支是另一条渲染路径,改名不能只改一半。"""
    assert "决策台" not in render_dashboard([])


# ---------------------------------------------------------------------------
# 事件中性(不做买入触发器、不用红色恐吓)
# ---------------------------------------------------------------------------
def test_events_neutral_no_buy_trigger_language():
    html = render_dashboard(_mixed_cohort())
    # 不得出现买入触发器式措辞
    assert "买入信号" not in html
    assert "立即买入" not in html
    # 中性提示语义存在
    assert ("提示" in html) or ("中性" in html)


# ---------------------------------------------------------------------------
# 钻取:复用 B 的 render_svg_card 内联 <svg>
# ---------------------------------------------------------------------------
def test_drilldown_embeds_svg_card_per_row():
    html = render_dashboard(_mixed_cohort())
    # 复用 B 的单股 SVG 卡:内联 <svg>(钻取)
    assert "<svg" in html.lower()
    # 每只都钻取:至少与记录数同量级的 svg(这里 5 只)
    assert html.lower().count("<svg") >= 5
    # 雷达免责随卡带出(B 卡含"低分≠坏")
    assert "低分≠坏" in html


# ---------------------------------------------------------------------------
# 缺失值占位(None -> 占位,不当 0、不省略)
# ---------------------------------------------------------------------------
def test_none_values_render_placeholder_not_zero():
    html = render_dashboard(_mixed_cohort())
    # 含全缺失记录的那只 -> 占位符出现
    assert ("—" in html) or ("N/A" in html)


def test_empty_records_does_not_raise():
    html = render_dashboard([])
    assert "<!doctype html" in html.lower()
    # 空集也要诚实成文,而非崩
    assert ("<table" in html.lower()) or ("无" in html)


def test_xss_safety_name_is_escaped():
    """名称含尖括号不得破坏 DOM 结构(HTML 转义)。"""
    rec = _rec(name="恶意<script>x</script>", ts_code="999999.SZ")
    html = render_dashboard([rec])
    # 注入的脚本标签不得以裸 <script>x</script> 形式出现在文档里
    assert "<script>x</script>" not in html
    # 转义后实体应在
    assert "&lt;script&gt;" in html


# ---------------------------------------------------------------------------
# 真实底座:全部 41 行 + None 占位 + 离散档
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# C4:render_svg_card 契约"永不抛";真到这里的是编程错误,须向上抛、不吞成红字
# ---------------------------------------------------------------------------
def test_svg_card_real_error_propagates_not_swallowed(monkeypatch):
    """坏 record 触发 render_svg_card 内部异常(编程错误)应向上抛,而非被吞成 <p class=err>。"""
    import ashare_gauntlet.render_html as rh

    def _boom(rec, cohort=None):
        raise KeyError("missing_key")

    monkeypatch.setattr(rh, "render_svg_card", _boom)
    with pytest.raises(KeyError):
        rh.render_dashboard(_mixed_cohort())


def test_no_err_swallow_string_in_normal_render():
    """正常 record 渲染不应出现兜底吞异常的红字片段(该兜底已移除)。"""
    html = render_dashboard(_mixed_cohort())
    assert "单股卡渲染失败" not in html


# ---------------------------------------------------------------------------
# C5/C6:概览表旗标列——per-flag type·fact·date(severity) 进 title;
#         中性 _FLAG_ICON 按 type,不用随计数放大的红旗 🚩×N
# ---------------------------------------------------------------------------
def _flag_cell_html(html: str, ts_code: str) -> str:
    """抽出某只行内的旗标 <td class="flags" ...> 单元格。"""
    # 行以 aria-label 含 ts_code 标识;简单起见从含该 code 的位置向后找 flags td
    i = html.index(ts_code)
    seg = html[i:]
    j = seg.index('<td class="flags"')
    end = seg.index("</td>", j)
    return seg[j:end]


def test_flag_cell_title_enumerates_per_flag_fact_type_date():
    rec = _rec(
        ts_code="600519.SH", name="贵州茅台", industry="白酒",
        flags=[
            {"type": "减持", "severity": "提示", "fact": "近一年减持2笔", "date": "20251223"},
            {"type": "解禁", "severity": "警示", "fact": "首发解禁8.2%", "date": "20260701"},
        ],
    )
    html = render_dashboard([rec])
    cell = _flag_cell_html(html, "600519.SH")
    # title 须含每条旗的 type·fact·date(severity),色盲/读屏不展开行也能拿到事件文字
    assert "减持" in cell
    assert "近一年减持2笔" in cell
    assert "20251223" in cell
    assert "解禁" in cell
    assert "首发解禁8.2%" in cell
    assert "警示" in cell


def test_flag_cell_uses_neutral_type_icon_not_red_flag_count():
    """单一类型旗标行渲染该 type 的中性 _FLAG_ICON,而非 🚩×N(红旗随量放大)。"""
    from ashare_gauntlet.render_html import _FLAG_ICON

    rec = _rec(
        ts_code="600519.SH", name="贵州茅台", industry="白酒",
        flags=[
            {"type": "减持", "severity": "提示", "fact": "减持A", "date": "20251223"},
            {"type": "减持", "severity": "提示", "fact": "减持B", "date": "20251224"},
        ],
    )
    html = render_dashboard([rec])
    cell = _flag_cell_html(html, "600519.SH")
    assert _FLAG_ICON["减持"] in cell           # 📉 中性 type 图标
    assert "🚩" not in cell                       # 不用红旗随计数放大


def test_no_flags_row_has_no_red_flag_icon():
    rec = _rec(ts_code="600000.SH", name="无旗", industry="银行", flags=[])
    html = render_dashboard([rec])
    cell = _flag_cell_html(html, "600000.SH")
    assert "🚩" not in cell


# ---------------------------------------------------------------------------
# C7:负市值(异常,实务不可达)计入 mv_na 而非小盘桶,避免误导
# ---------------------------------------------------------------------------
def test_negative_market_cap_not_counted_as_small_cap():
    recs = [
        _rec(ts_code="000004.SZ", name="负市值", industry="测试", mv_yi=-5.0),
        _rec(ts_code="000005.SZ", name="正常小盘", industry="测试", mv_yi=50.0),
    ]
    html = render_dashboard(recs)
    # 负 mv 不入小盘桶:小盘桶应只计 1(那只 50 亿),不是 2
    assert "<100亿(小盘) <b>1</b>" in html
    # 负 mv 计入市值缺失/异常
    assert "市值缺失 <b>1</b>" in html


def test_num_rejects_nan():
    """_num 须把 NaN 视作缺失(None),与 record._num 口径一致,防 NaN 混入排序/分桶。"""
    from ashare_gauntlet.render_html import _num

    assert _num({"valuation": {"mv_yi": float("nan")}}, "valuation.mv_yi") is None
    assert _num({"valuation": {"mv_yi": 12.5}}, "valuation.mv_yi") == 12.5


@pytest.mark.skipif(not os.path.exists(CARDS), reason="无真实底座")
def test_real_cards_all_41_rows_present():
    with open(CARDS, encoding="utf-8") as fh:
        data = json.load(fh)
    html = render_dashboard(data)
    # 全部记录的 ts_code 都出现(逐只在表/钻取里)
    for r in data:
        assert r["ts_code"] in html, f"缺 {r['ts_code']}"
    # 41 只
    assert len(data) == 41


@pytest.mark.skipif(not os.path.exists(CARDS), reason="无真实底座")
def test_real_cards_industry_concentration_top_is_named():
    with open(CARDS, encoding="utf-8") as fh:
        data = json.load(fh)
    html = render_dashboard(data)
    # 元器件 26/41 为最集中行业,组合摘要须点名
    assert "元器件" in html


@pytest.mark.skipif(not os.path.exists(CARDS), reason="无真实底座")
def test_real_cards_well_formed_no_unescaped_ampersand():
    with open(CARDS, encoding="utf-8") as fh:
        data = json.load(fh)
    html = render_dashboard(data)
    # HTML5:<script> 为 raw-text,其内 && / < / > 是合法 JS,不参与转义判定;
    # 但 script 体内不得出现会提前闭合元素的 "</script"。
    scripts = re.findall(r"<script\b[^>]*>(.*?)</script>", html, flags=re.DOTALL | re.IGNORECASE)
    assert scripts, "应有内联 <script>"
    for s in scripts:
        assert "</script" not in s.lower(), "script 体不得含会提前闭合的 </script"
    # 标记区(去掉 script/style raw-text 后):不得有未转义裸 &(实体除外),防属性注入破坏结构。
    markup = re.sub(r"<script\b[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    markup = re.sub(r"<style\b[^>]*>.*?</style>", "", markup, flags=re.DOTALL | re.IGNORECASE)
    assert not re.search(r"&(?!amp;|lt;|gt;|quot;|apos;|#)", markup)


# ---------------------------------------------------------------------------
# data_coverage:事件表空=未取到,dashboard 要 surface(内嵌 SVG 卡 + 概览旗标 title)
# ---------------------------------------------------------------------------
def test_data_coverage_unknown_surfaced_in_dashboard():
    rec = _rec()
    rec["data_coverage"] = {
        "share_float": "present", "pledge_stat": "present",
        "stk_holdertrade": "unknown", "forecast": "present", "express": "unknown",
    }
    html = render_dashboard([rec])
    assert "数据未取到" in html  # 经内嵌 render_svg_card 卡 / 概览旗标单元格 title


def test_data_coverage_all_present_no_caveat_dashboard():
    rec = _rec()
    rec["data_coverage"] = {
        k: "present" for k in ("share_float", "pledge_stat", "stk_holdertrade", "forecast", "express")
    }
    html = render_dashboard([rec])
    assert "数据未取到" not in html
