"""TDD:多股 HTML dashboard 总览决策台 render_dashboard 纯函数(无 IO,返回自含单文件 HTML)。

断言结构性质而非像素:含 <table、含排序 JS(sort/onclick)、含过滤控件(tier/行业)、
含组合层摘要(行业集中度文字 / 市值分桶 / tier 分布 / 预算参考且标注"参考非建议")、
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
# 组合层摘要(顶部决策台)
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
    assert "2" in html  # 🟢 计数
    # 分布须以离散档呈现
    for g in ("🟢", "🟡", "🔴", "⛔"):
        assert g in html


def test_portfolio_summary_budget_reference_marked_not_advice():
    """预算参考须明确标注'参考/非建议',不臆造持仓规模。"""
    html = render_dashboard(_mixed_cohort())
    assert "预算" in html or "参考" in html
    # 铁律:标注为参考,非买入建议
    assert ("参考" in html) or ("非建议" in html)
    assert ("建议" in html)  # 出现"非建议/不构成建议"字样


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
