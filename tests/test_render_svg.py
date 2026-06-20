"""TDD:单股 SVG 卡 render_svg_card 纯函数(无 IO,返回完整自含 <svg>)。

断言结构性质而非像素:含 <svg、tier 文字档名+图标(色盲安全)、雷达 6 轴标签齐、
"低分≠坏" 直觉免责、归一化口径文字、子弹图对比集与时点标注、缺失轴降级("—")不抛异常、
结论用离散档(无裸 score 小数当结论)、事件旗标含 <title>/aria-label 文字。

手工小 fixture + 真实底座抽样,纯渲染、不联网。
"""
import json
import os
import re

import pytest

from ashare_gauntlet.render_svg import render_svg_card

CARDS = os.path.join("data", "cards", "20260618.json")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def _rec(
    *,
    grade: str = "🟢",
    tier_label_words: str = "强且干净",
    entry_grade: str = "A",
    industry: str = "通信设备",
    pe_ttm: float | None = 30.0,
    np_yoy: float | None = 20.0,
    net_cash_ratio: float | None = 1.2,
    accrual: float | None = -0.02,
    pct20: float | None = 90.0,
    dedt_yoy: float | None = 15.0,
    recv_pct: float | None = 300.0,
    flags: list[dict] | None = None,
) -> dict:
    return {
        "ts_code": "601138.SH",
        "name": "工业富联",
        "industry": industry,
        "as_of": "20260618",
        "tier": {"grade": grade, "reasons": [tier_label_words], "needs_human": False},
        "entry": {"grade": entry_grade, "score": 91.4, "tag": "趋势内"},
        "technical": {"close": 78.08, "pct20": pct20, "trend": "多头", "rsi": 61.0},
        "valuation": {"pe_ttm": pe_ttm, "pb": 8.79, "peg": 0.37, "mv_yi": 15494.2},
        "fundamental": {
            "end_date": "20260331", "rev_yoy": 56.5, "np_yoy": np_yoy,
            "dedt_yoy": dedt_yoy, "gross_margin": 7.35, "profitable": True,
        },
        "balance": {"recv_to_annual_net_pct": recv_pct, "goodwill_yi": 3.28},
        "quality": {
            "op_cashflow_yi": 52.4, "net_cash_ratio": net_cash_ratio, "accrual": accrual,
        },
        "status": {"is_st": False, "ever_st": False, "current_name": "工业富联"},
        "flags": flags if flags is not None else [
            {"type": "质押", "severity": "提示",
             "fact": "控股股东体系质押0.3%", "date": "20260618", "source": "pledge_stat接口"},
        ],
        "meta": {},
        "theme": "",
        "tags": [],
    }


def _cohort(n: int = 6) -> list[dict]:
    out: list[dict] = []
    for i in range(n):
        out.append(_rec(
            pe_ttm=20.0 + i * 10,
            np_yoy=-50.0 + i * 30,
            net_cash_ratio=-2.0 + i * 1.5,
            accrual=-0.10 + i * 0.04,
            pct20=80.0 + i * 3,
            dedt_yoy=-40.0 + i * 25,
            recv_pct=100.0 + i * 150,
        ))
    return out


# ---------------------------------------------------------------------------
# 基本结构
# ---------------------------------------------------------------------------
def test_returns_self_contained_svg():
    svg = render_svg_card(_rec(), cohort=_cohort())
    assert isinstance(svg, str)
    assert svg.lstrip().startswith("<svg")
    assert "</svg>" in svg
    # 自含:声明 xmlns,不依赖外部样式表/图片
    assert "xmlns" in svg
    assert "<link" not in svg and "<image" not in svg


def test_header_has_tier_words_icon_entry_name_code_industry():
    svg = render_svg_card(_rec(grade="🟢", tier_label_words="强且干净", entry_grade="A"), cohort=_cohort())
    # 色盲安全:emoji 图标 + 文字档名(颜色之外必有文字)
    assert "🟢" in svg
    assert "强且干净" in svg
    # entry 档
    assert "A" in svg
    # 名称(代码)+ 行业
    assert "工业富联" in svg
    assert "601138.SH" in svg
    assert "通信设备" in svg


def test_tier_color_is_colorblind_safe_text_present_for_each_grade():
    for g, words in [("🟢", "强且干净"), ("🟡", "盈利瑕疵"), ("🔴", "题材背离"), ("⛔", "地雷")]:
        svg = render_svg_card(_rec(grade=g, tier_label_words=words), cohort=_cohort())
        assert g in svg          # 图标
        assert words in svg      # 文字档名(颜色之外的可读信息)


def test_badge_shows_short_tier_name_not_long_reason_overflow():
    # 真实数据 reason 很长(如"净利&扣非&营收三增·现金流>0·无警示");徽章固定 250px,
    # 必须显示由 grade 派生的短档名,长 reason 只进 tooltip,否则文字溢出徽章撞 entry 档。
    long_reason = "增收不增利营收增但净利与扣非下滑需配涨幅人工复核额外冗长占位以触发溢出风险"
    svg = render_svg_card(_rec(grade="🔴", tier_label_words=long_reason), cohort=_cohort())
    assert 'font-size="22" fill="#ffffff">🔴 题材背离</text>' in svg   # 22px 徽章=短档名(防溢出)
    assert long_reason in svg                                          # 完整 reason 仍可达(tooltip;此串无需转义字符)


# ---------------------------------------------------------------------------
# 雷达 / 雪花图:固定 6 轴
# ---------------------------------------------------------------------------
def test_radar_has_all_six_axis_labels_in_fixed_order():
    svg = render_svg_card(_rec(), cohort=_cohort())
    for axis in ["估值", "成长", "盈利质量", "现金质量", "风险", "动量"]:
        assert axis in svg, f"缺轴标签 {axis}"
    # 固定轴序:估值 在 动量 之前
    assert svg.index("估值") < svg.index("动量")


def test_radar_has_low_score_not_bad_disclaimer_and_norm_caption():
    svg = render_svg_card(_rec(), cohort=_cohort())
    assert "低分≠坏" in svg
    # 归一化口径:百分位 / cohort
    assert ("百分位" in svg) or ("分位" in svg)
    assert "cohort" in svg or "对比集" in svg
    # 明确"非买卖信号 / 偏科直觉"
    assert ("非买卖信号" in svg) or ("非买卖" in svg)


# ---------------------------------------------------------------------------
# 子弹图:PE / 扣非增速 / 净现比 vs 对比集中位
# ---------------------------------------------------------------------------
def test_bullet_has_three_metrics_and_cohort_median_caption():
    svg = render_svg_card(_rec(), cohort=_cohort())
    assert "PE" in svg
    assert "扣非" in svg          # dedt_yoy
    assert "净现比" in svg        # net_cash_ratio
    # 对比集 + 中位 + 口径时点标注(诚实:用 cohort 中位,非全市场)
    assert "中位" in svg
    assert "对比集" in svg
    # 时点(交易日 / 季 / 年报 之一须出现作口径)
    assert ("交易日" in svg) or ("@2026" in svg) or ("年报" in svg) or ("季" in svg)


def test_bullet_uses_same_industry_cohort_when_ge_three_else_full():
    # 同行业 ≥3:对比集口径文字应体现"同行业"
    same_ind = [_rec(industry="通信设备", pe_ttm=20.0 + i * 5) for i in range(4)]
    svg = render_svg_card(_rec(industry="通信设备"), cohort=same_ind)
    assert ("同行业" in svg) or ("通信设备" in svg)
    # 同行业 <3:回退全 cohort,口径文字应体现"全 cohort"或"全部"
    mixed = [_rec(industry="互联网"), _rec(industry="半导体")]
    svg2 = render_svg_card(_rec(industry="通信设备"), cohort=mixed)
    assert ("全" in svg2) or ("cohort" in svg2)


# ---------------------------------------------------------------------------
# 事件旗标行:中性提示 + 事实 + 日期 + a11y 文字
# ---------------------------------------------------------------------------
def test_flag_row_neutral_fact_date_with_title_for_a11y():
    svg = render_svg_card(
        _rec(flags=[{"type": "减持", "severity": "提示",
                     "fact": "近一年减持2笔", "date": "20251223", "source": "stk_holdertrade接口"}]),
        cohort=_cohort(),
    )
    assert "减持" in svg
    assert "近一年减持2笔" in svg
    assert "20251223" in svg
    # a11y:<title> 或 aria-label
    assert ("<title" in svg) or ("aria-label" in svg)


def test_no_flags_renders_without_error_and_states_none():
    svg = render_svg_card(_rec(flags=[]), cohort=_cohort())
    assert svg.lstrip().startswith("<svg")
    # 无旗标也要诚实说明,而非空白
    assert ("无" in svg)


# ---------------------------------------------------------------------------
# 缺失值降级:灰条 + "—",不抛异常、不伪造 0
# ---------------------------------------------------------------------------
def test_missing_axis_values_degrade_no_exception_no_fake_zero():
    rec = _rec(pe_ttm=None, np_yoy=None, net_cash_ratio=None,
               accrual=None, pct20=None, dedt_yoy=None, recv_pct=None)
    svg = render_svg_card(rec, cohort=_cohort())  # 不应抛
    assert svg.lstrip().startswith("<svg")
    # 占位符出现(— 或 N/A),而非 0
    assert ("—" in svg) or ("N/A" in svg)


def test_missing_bullet_metric_shows_placeholder():
    rec = _rec(pe_ttm=None, dedt_yoy=None)
    svg = render_svg_card(rec, cohort=_cohort())
    assert ("—" in svg) or ("N/A" in svg)


def test_none_cohort_does_not_raise():
    svg = render_svg_card(_rec(), cohort=None)
    assert svg.lstrip().startswith("<svg")


def test_empty_cohort_does_not_raise():
    svg = render_svg_card(_rec(), cohort=[])
    assert svg.lstrip().startswith("<svg")


# ---------------------------------------------------------------------------
# 产品铁律:不把小数 score 当结论(结论用离散档)
# ---------------------------------------------------------------------------
def test_does_not_present_raw_decimal_score_as_verdict():
    # entry.score=91.4 不应作为"结论"裸出现在卡里(结论用 A/B/C 与 tier emoji)
    svg = render_svg_card(_rec(), cohort=_cohort())
    assert "91.4" not in svg
    # 但离散档必须在
    assert "🟢" in svg


# ---------------------------------------------------------------------------
# 真实底座抽样:🟢 与 ⛔ 各一,cohort 传全量
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not os.path.exists(CARDS), reason="无真实底座")
def test_real_cards_green_and_mine_render():
    with open(CARDS, encoding="utf-8") as fh:
        data = json.load(fh)
    green = next(r for r in data if r["tier"]["grade"] == "🟢")
    mine = next(r for r in data if r["tier"]["grade"] == "⛔")
    for r in (green, mine):
        svg = render_svg_card(r, cohort=data)
        assert svg.lstrip().startswith("<svg")
        assert "</svg>" in svg
        assert r["ts_code"] in svg
        assert r["tier"]["grade"] in svg
        assert "低分≠坏" in svg


@pytest.mark.skipif(not os.path.exists(CARDS), reason="无真实底座")
def test_real_cards_no_raw_attribute_injection_breaks_svg():
    # name 含特殊字符时不应破坏 SVG(XML 转义);抽全量确认无裸 & 造成非法 XML
    with open(CARDS, encoding="utf-8") as fh:
        data = json.load(fh)
    svg = render_svg_card(data[0], cohort=data)
    # 能被 XML parser 解析即结构合法;用 expat 并关闭实体扩展(防 XXE/billion-laughs),
    # 渲染输出是自产字符串而非不可信输入,这里只验证 well-formed。
    import xml.parsers.expat as expat

    parser = expat.ParserCreate()
    parser.DefaultHandler = lambda _data: None
    try:
        parser.UseForeignDTD(True)  # 不解析外部 DTD
    except (AttributeError, expat.error):
        pass
    parser.Parse(svg, True)  # well-formed 即不抛 ExpatError
    # 不得有未转义裸 & (除实体)
    assert not re.search(r"&(?!amp;|lt;|gt;|quot;|apos;|#)", svg)


# ---------------------------------------------------------------------------
# review follow-up #3:风险轴优先读结构化 value,不再反解析展示字符串
# ---------------------------------------------------------------------------
def test_pledge_pct_prefers_structured_value_over_parsing_fact():
    from ashare_gauntlet.render_svg import _pledge_pct

    rec = {"flags": [{"type": "质押", "value": 42.0, "fact": "无数字的纯文字描述"}]}
    assert _pledge_pct(rec) == pytest.approx(42.0)


def test_pledge_pct_falls_back_to_fact_when_value_absent():
    from ashare_gauntlet.render_svg import _pledge_pct

    rec = {"flags": [{"type": "质押", "fact": "控股股东体系质押12.9%"}]}
    assert _pledge_pct(rec) == pytest.approx(12.9)


# ---------------------------------------------------------------------------
# C1:四类旗标(质押/解禁/减持/超预期)全渲染时,末条 y 坐标不被画布裁剪
# ---------------------------------------------------------------------------
def _four_flags() -> list[dict]:
    return [
        {"type": "质押", "severity": "提示", "fact": "控股股东体系质押12.9%", "date": "20260618"},
        {"type": "解禁", "severity": "警示", "fact": "首发限售解禁8.2%", "date": "20260701"},
        {"type": "减持", "severity": "提示", "fact": "近一年减持2笔", "date": "20251223"},
        {"type": "超预期", "severity": "提示", "fact": "业绩预告上修", "date": "20260415"},
    ]


def _svg_height(svg: str) -> float:
    m = re.search(r'<svg[^>]*\bheight="([0-9.]+)"', svg)
    assert m, "缺 <svg height>"
    return float(m.group(1))


def test_four_flags_last_row_y_within_canvas_height():
    """四类旗标全出时,最后一条文字基线 ly 必须 <= 画布高 - 12(不被底框裁剪)。"""
    svg = render_svg_card(_rec(flags=_four_flags()), cohort=_cohort())
    h = _svg_height(svg)
    # 取所有旗标文字 y(事件提示旗标题行之后的 <text ... y=...>),末条须留 12px 余量
    ys = [float(y) for y in re.findall(r'<text x="16" y="([0-9.]+)" font-size="12" fill="#222222" aria-label=', svg)]
    assert ys, "应渲染出旗标文字行"
    assert max(ys) <= h - 12, f"末条旗标 y={max(ys)} 超出画布 h-12={h - 12}"


def test_svg_height_grows_with_more_flags():
    """画布高随旗标数增大(动态 H),保证多旗不被裁。"""
    h1 = _svg_height(render_svg_card(_rec(flags=_four_flags()[:1]), cohort=_cohort()))
    h4 = _svg_height(render_svg_card(_rec(flags=_four_flags()), cohort=_cohort()))
    assert h4 > h1


def test_svg_height_matches_viewbox():
    """动态 H 时 viewBox 高度须与 height 一致(否则缩放错位)。"""
    svg = render_svg_card(_rec(flags=_four_flags()), cohort=_cohort())
    h = _svg_height(svg)
    m = re.search(r'viewBox="0 0 \d+ ([0-9.]+)"', svg)
    assert m and float(m.group(1)) == h


# ---------------------------------------------------------------------------
# C2:负 PE(亏损股)估值轴降级为缺失,不混入 cohort 百分位排序
# ---------------------------------------------------------------------------
def test_negative_pe_valuation_axis_degrades_not_raises():
    """pe_ttm<=0(亏损)→ 估值轴归一化为缺失/灰点,不抛、不算负 EP 进百分位。"""
    from ashare_gauntlet.render_svg import AXES

    val_getter = AXES[0][1]  # 估值轴 getter
    assert val_getter(_rec(pe_ttm=-10.0)) is None
    assert val_getter(_rec(pe_ttm=0.0)) is None
    # 正常 PE 仍返回正 EP
    assert val_getter(_rec(pe_ttm=20.0)) == pytest.approx(0.05)
    # 整卡不抛
    svg = render_svg_card(_rec(pe_ttm=-10.0), cohort=_cohort())
    assert svg.lstrip().startswith("<svg")


# ---------------------------------------------------------------------------
# C3:severity==警示 视觉/读屏可辨(中性强调,非红非恐吓);提示不强调
# ---------------------------------------------------------------------------
def _flags_group(svg: str) -> str:
    """抽出事件旗标 <g>...</g> 段(标题文字之后到该 g 末),用于只看旗标区强调。"""
    i = svg.index("事件提示旗")
    return svg[i:]


def test_severity_warn_is_emphasized_hint_is_not():
    svg_warn = render_svg_card(
        _rec(flags=[{"type": "解禁", "severity": "警示", "fact": "解禁8.2%", "date": "20260701"}]),
        cohort=_cohort(),
    )
    svg_hint = render_svg_card(
        _rec(flags=[{"type": "减持", "severity": "提示", "fact": "减持2笔", "date": "20251223"}]),
        cohort=_cohort(),
    )
    warn_g = _flags_group(svg_warn)
    hint_g = _flags_group(svg_hint)
    # 警示:旗标区出现中性强调标记(前置中性符 ‼ 或对 severity 加粗 font-weight=700),非红非恐吓
    assert "警示" in warn_g
    assert ("‼" in warn_g) or ('font-weight="700"' in warn_g)
    # 提示:旗标区无该强调(确认是 severity 触发,而非恒定渲染)
    assert "‼" not in hint_g
    assert 'font-weight="700"' not in hint_g


# ---------------------------------------------------------------------------
# data_coverage:事件表空=未取到,卡底部 surface(中性灰字,且高度容纳不被裁)
# ---------------------------------------------------------------------------
def test_data_coverage_unknown_surfaced_in_card():
    rec = _rec()
    rec["data_coverage"] = {
        "share_float": "present", "pledge_stat": "present",
        "stk_holdertrade": "unknown", "forecast": "present", "express": "unknown",
    }
    svg = render_svg_card(rec, cohort=_cohort())
    assert "数据未取到" in svg
    assert "未确认" in svg
    assert svg.lstrip().startswith("<svg") and "</svg>" in svg


def test_data_coverage_grows_card_height_not_clipped():
    flags = [
        {"type": "质押", "severity": "提示", "fact": "质押10%", "date": "20260618"},
        {"type": "解禁", "severity": "提示", "fact": "解禁5%", "date": "20260701"},
    ]
    base = render_svg_card(_rec(flags=flags), cohort=_cohort())
    rec = _rec(flags=flags)
    rec["data_coverage"] = {
        "share_float": "present", "pledge_stat": "present",
        "stk_holdertrade": "present", "forecast": "present", "express": "unknown",
    }
    withcov = render_svg_card(rec, cohort=_cohort())
    hb = int(re.search(r'height="(\d+)"', base).group(1))
    hc = int(re.search(r'height="(\d+)"', withcov).group(1))
    assert hc > hb  # 未取到 caveat 多一行 → 卡更高,不被底框裁


def test_data_coverage_missing_no_crash_svg():
    svg = render_svg_card(_rec(), cohort=_cohort())  # _rec 无 data_coverage
    assert "数据未取到" not in svg and svg.lstrip().startswith("<svg")
