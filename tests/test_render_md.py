"""TDD for the Markdown 诚实面板 renderer (ashare_gauntlet.render_md.render_md).

Asserts structural properties, not exact bytes: the三层 (overview table / per-name
<details> / 口径脚注) all appear; discrete tiers (🟢🟡🔴⛔) drive conclusions rather
than raw decimals; missing values render as the "—" placeholder (never 0); the
sparkline block-chars show up; factcheck (when present) is surfaced as 独立核实;
events render as neutral 提示旗 with fact + date.

The renderer is a pure function (records -> str, no IO); IO lives in scripts/panel.py.
"""

from __future__ import annotations

from typing import Any

import pytest

from ashare_gauntlet.render_md import render_md


def _meta(as_of_val: str, fund: str = "20260331", cash: str = "20251231") -> dict[str, Any]:
    return {
        "technical.close": {"unit": "元", "as_of": as_of_val, "source": "daily前复权"},
        "valuation.pe_ttm": {"unit": "倍", "as_of": as_of_val, "source": "daily_basic接口"},
        "fundamental.rev_yi": {"unit": "亿元", "as_of": fund, "source": "income.total_revenue"},
        "quality.op_cashflow_yi": {"unit": "亿元", "as_of": cash, "source": "cashflow(年报)"},
    }


def _rec(
    ts_code: str,
    name: str,
    grade: str,
    *,
    needs_human: bool = False,
    reasons: list[str] | None = None,
    peg: float | None = 0.5,
    pe_ttm: float | None = 30.0,
    flags: list[dict[str, Any]] | None = None,
    factcheck: dict[str, Any] | None = None,
    score: float = 90.0,
) -> dict[str, Any]:
    return {
        "ts_code": ts_code,
        "name": name,
        "industry": "半导体",
        "as_of": "20260618",
        "entry": {"grade": "A", "score": score, "tag": "趋势内"},
        "technical": {
            "close": 50.0,
            "dist60": -3.5,
            "pct20": 88.0,
            "trend": "多头",
            "rsi": 60.0,
            "ret20": 12.0,
            "vol_ratio": 1.2,
        },
        "valuation": {"pe_ttm": pe_ttm, "pb": 5.0, "peg": peg, "mv_yi": 500.0},
        "fundamental": {
            "end_date": "20260331",
            "rev_yi": 100.0,
            "rev_yoy": 30.0,
            "np_yi": 10.0,
            "np_yoy": 40.0,
            "dedt_yoy": 35.0,
            "dedt_yi": 9.0,
            "gross_margin": 25.0,
            "profitable": True,
        },
        "balance": {
            "accounts_receiv_yi": 40.0,
            "goodwill_yi": None,
            "money_cap_yi": 30.0,
            "net_assets_yi": 130.0,
            "total_assets_yi": 280.0,
            "recv_to_annual_net_pct": 300.0,
        },
        "quality": {"op_cashflow_yi": 20.0, "net_cash_ratio": 1.5, "accrual": -0.02},
        "status": {"is_st": False, "ever_st": False, "current_name": name},
        "flags": flags or [],
        "meta": _meta("20260618"),
        "tier": {"grade": grade, "reasons": reasons or ["测试理由"], "needs_human": needs_human},
        "factcheck": factcheck,
        "theme": "",
        "tags": [],
    }


@pytest.fixture
def sample() -> list[dict[str, Any]]:
    return [
        _rec("601138.SH", "工业富联", "🟢", reasons=["净利&扣非&营收三增·现金流>0·无警示"]),
        _rec(
            "000100.SZ",
            "TCL科技",
            "🟡",
            needs_human=True,
            reasons=["疑低基数(净利+54% 营收+8%)"],
            peg=None,  # 缺失值 -> 占位
            flags=[
                {
                    "type": "解禁",
                    "severity": "提示",
                    "fact": "解禁 占流通4.7%",
                    "date": "20260710",
                    "source": "share_float接口",
                }
            ],
        ),
        _rec(
            "000063.SZ",
            "中兴通讯",
            "🔴",
            needs_human=True,
            reasons=["增收不增利(营收增但净利/扣非降)"],
            factcheck={
                "confirmed": True,
                "q1_net_profit_yi": 9.13,
                "disputes": ["某争议点"],
                "news": ["某新闻"],
                "verified_at": "20260618",
            },
        ),
        _rec("001229.SZ", "魅视科技", "⛔", reasons=["亏损", "净利&扣非双降≤-40%"]),
    ]


def test_returns_str(sample: list[dict[str, Any]]) -> None:
    out = render_md(sample)
    assert isinstance(out, str)
    assert out.strip()


def test_has_overview_table_header(sample: list[dict[str, Any]]) -> None:
    out = render_md(sample)
    # Markdown 概览表:表头分隔行(--- | ---)必须存在
    assert "---" in out and "|" in out
    # 概览每只一行:四只都出现在概览(代码出现)
    for code in ("601138.SH", "000100.SZ", "000063.SZ", "001229.SZ"):
        assert code in out


def test_has_collapsible_details_per_name(sample: list[dict[str, Any]]) -> None:
    out = render_md(sample)
    assert "<details>" in out
    assert "<summary>" in out
    # 每只一个 <details>:四只 -> 至少四个
    assert out.count("<details>") >= len(sample)


def test_footnote_with_caliber_time_source(sample: list[dict[str, Any]]) -> None:
    out = render_md(sample)
    # 口径脚注:时点 + 来源汇总(从 meta 提炼)
    assert "20260618" in out  # 交易日时点
    assert "20260331" in out  # 基本面最新季时点
    assert "20251231" in out or "2025年报" in out  # 现金质量年报时点
    assert "tushare" in out or "接口" in out  # 来源


def test_stop_tier_text_appears(sample: list[dict[str, Any]]) -> None:
    out = render_md(sample)
    # 色盲安全:⛔ 之外必有文字档名
    assert "⛔" in out
    assert "地雷" in out or "出局" in out or "排除" in out


def test_all_four_tier_emojis_render(sample: list[dict[str, Any]]) -> None:
    out = render_md(sample)
    for emoji in ("🟢", "🟡", "🔴", "⛔"):
        assert emoji in out


def test_none_value_renders_as_placeholder_not_zero(sample: list[dict[str, Any]]) -> None:
    out = render_md(sample)
    # TCL科技 peg=None 且 goodwill_yi=None;占位符 "—" 或 N/A 必须出现
    assert "—" in out or "N/A" in out
    # 缺失绝不渲染成 0:不得出现 "PEG 0.00" / "PEG: 0" 形式当结论
    assert "PEG0.00" not in out
    assert "PEG 0.00" not in out


def test_sparkline_block_chars_present(sample: list[dict[str, Any]]) -> None:
    out = render_md(sample)
    # pct20 sparkline 用区块字符之一
    assert any(ch in out for ch in "▁▂▃▄▅▆▇█")


def test_needs_human_surfaced(sample: list[dict[str, Any]]) -> None:
    out = render_md(sample)
    # needs_human=True 的标的要展示需人工复核信号
    assert "人工" in out or "复核" in out or "needs_human" in out.lower()


def test_tier_reasons_surfaced(sample: list[dict[str, Any]]) -> None:
    out = render_md(sample)
    assert "净利&扣非&营收三增" in out  # 🟢 reason
    assert "增收不增利" in out  # 🔴 reason


def test_factcheck_surfaced_when_present(sample: list[dict[str, Any]]) -> None:
    out = render_md(sample)
    # factcheck 非空要展示,并标注"独立核实"
    assert "独立核实" in out
    assert "9.13" in out  # q1_net_profit_yi 与接口数字并列


def test_event_flag_renders_neutral_with_fact_and_date(sample: list[dict[str, Any]]) -> None:
    out = render_md(sample)
    # 事件=提示旗 + 事实 + 日期;不做买入触发器,不用红色恐吓
    assert "解禁 占流通4.7%" in out
    assert "20260710" in out
    # 中性提示:出现"提示"字样而非"卖出/危险/触发"
    assert "提示" in out


def test_no_naked_decimal_score_as_conclusion(sample: list[dict[str, Any]]) -> None:
    out = render_md(sample)
    # 铁律①:不把 entry.score 小数当结论。score=90.0 不应作为独立"结论档"出现,
    # 结论用 entry 档 A/B/C。至少 entry 档字母要出现。
    assert "A" in out


def test_empty_records_returns_str() -> None:
    out = render_md([])
    assert isinstance(out, str)
