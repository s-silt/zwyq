"""Markdown 诚实面板渲染(纯函数,无 IO)。

`render_md(records) -> str` 把 D 层 cards 渲染成三层渐进的 Markdown:
  ① 顶部概览表 —— 每只一行 ≤8 列(质地档 | 名称(代码) | entry档 | 现价/距60高 |
     近20分位+sparkline | PE/PEG | 旗标数 | 趋势),按 tier ⛔→🔴→🟡→🟢 分组、
     组内按 entry.score 降序。
  ② 每只一节 <details> 折叠,分组小表(技术/估值/基本面/现金质量/资产/事件),
     值旁注单位;展示 tier.reasons 与 needs_human;factcheck 非空时与接口数字并列,
     标"独立核实"。
  ③ 口径脚注 —— 从 meta 提炼时点+来源汇总。

两条产品铁律:① 一切结论用离散档(tier emoji+文字档名 / entry A/B/C),不把小数
score 当结论;每个数值就近标注口径/时点/来源。② 资金/事件(flags)只做中性提示旗
(emoji+一句事实+日期),绝不做买入触发器、不用红色恐吓;颜色之外必有图标/文字。

缺失值(None)一律渲染为占位 "—",绝不当 0 或省略。
"""

from __future__ import annotations

import html as _html
from typing import Any

# tier 档:emoji -> (展示顺序权重小=靠前, 文字档名)。色盲安全:emoji 已含图标,补文字。
_TIER_ORDER: dict[str, int] = {"⛔": 0, "🔴": 1, "🟡": 2, "🟢": 3}
_TIER_LABEL: dict[str, str] = {
    "🟢": "🟢 强干净",
    "🟡": "🟡 盈利瑕疵",
    "🔴": "🔴 题材背离",
    "⛔": "⛔ 地雷出局",
}

# pct20 百分位 -> sparkline 区块字符(单字符 sparkline,配当前数值一起展示)。
_BLOCKS: str = "▁▂▃▄▅▆▇█"

_NA: str = "—"


def _md(s: object) -> str:
    """文本进 markup 前转义:① HTML 实体(< > &,本面板内嵌 <details> 会透传 raw HTML,
    防 stored-XSS);② Markdown 结构字符 `|`(表格列分隔)转义、换行折成空格。

    用于一切非受控自由文本(股票名/行业/趋势/tier 理由/旗标 fact/factcheck 文本)。
    数值经 _fmt 已是受控字符串,无需经此。
    """
    return (
        _html.escape(str(s), quote=False)
        .replace("\\", "\\\\")  # 先转义反斜杠,避免与下面的 `\|` 混淆
        .replace("|", "\\|")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def _fmt(value: Any, *, suffix: str = "", decimals: int = 2, pct_sign: bool = False) -> str:
    """格式化数值;None/非数 -> 占位 "—"(绝不当 0)。

    pct_sign=True 时在数字后补 "%";suffix 为额外单位/标记(如 "倍"、"亿")。
    """
    if value is None or not isinstance(value, (int, float)):
        return _NA
    body = f"{value:+.{decimals}f}" if pct_sign else f"{value:.{decimals}f}"
    tail = "%" if pct_sign else ""
    return f"{body}{tail}{suffix}"


def _spark(pct: Any) -> str:
    """百分位(0–100)-> 单个区块字符。None -> 占位。"""
    if pct is None or not isinstance(pct, (int, float)):
        return _NA
    idx = int(max(0.0, min(100.0, float(pct))) / 100.0 * (len(_BLOCKS) - 1) + 0.5)
    return _BLOCKS[idx]


def _name_code(rec: dict[str, Any]) -> str:
    return f"{_md(rec.get('name', '?'))}({_md(rec.get('ts_code', '?'))})"


def _sorted_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 tier ⛔→🔴→🟡→🟢 分组,组内按 entry.score 降序。"""

    def key(rec: dict[str, Any]) -> tuple[int, float]:
        tier = rec.get("tier") or {}
        entry = rec.get("entry") or {}
        order = _TIER_ORDER.get(tier.get("grade", ""), 99)
        score = entry.get("score")
        score_f = float(score) if isinstance(score, (int, float)) else -1.0
        return (order, -score_f)

    return sorted(records, key=key)


def _overview_table(records: list[dict[str, Any]]) -> list[str]:
    """① 顶部概览表(每只一行 ≤8 列)。"""
    header = (
        "| 质地档 | 名称(代码) | entry | 现价/距60高 | 近20分位 | PE / PEG | 旗 | 趋势 |"
    )
    sep = "| --- | --- | :---: | --- | --- | --- | :---: | :---: |"
    rows: list[str] = [header, sep]
    for rec in records:
        tier = rec.get("tier") or {}
        entry = rec.get("entry") or {}
        tech = rec.get("technical") or {}
        val = rec.get("valuation") or {}
        flags = rec.get("flags") or []

        label = _TIER_LABEL.get(tier.get("grade", ""), tier.get("grade", _NA))
        human = " ⚑需人工" if tier.get("needs_human") else ""
        close = _fmt(tech.get("close"), decimals=2)
        dist60 = _fmt(tech.get("dist60"), pct_sign=True)
        pct20 = tech.get("pct20")
        pct_cell = f"{_spark(pct20)} {_fmt(pct20, decimals=0)}" if pct20 is not None else _NA
        pe = _fmt(val.get("pe_ttm"), decimals=1)
        peg = _fmt(val.get("peg"), decimals=2)
        flag_cell = str(len(flags)) if flags else "·"
        trend = _md(tech.get("trend") or _NA)

        rows.append(
            f"| {label}{human} | {_name_code(rec)} | {entry.get('grade', _NA)} "
            f"| {close} / {dist60} | {pct_cell} | {pe} / {peg} | {flag_cell} | {trend} |"
        )
    return rows


def _kv_table(title: str, pairs: list[tuple[str, str]]) -> list[str]:
    """一个分组小表:两列(指标 | 值,值旁已注单位)。"""
    out: list[str] = [f"**{title}**", "", "| 指标 | 值 |", "| --- | --- |"]
    out += [f"| {k} | {v} |" for k, v in pairs]
    out.append("")
    return out


def _detail_section(rec: dict[str, Any]) -> list[str]:
    """② 单只 <details> 折叠节。"""
    tier = rec.get("tier") or {}
    tech = rec.get("technical") or {}
    val = rec.get("valuation") or {}
    fund = rec.get("fundamental") or {}
    bal = rec.get("balance") or {}
    qual = rec.get("quality") or {}
    status = rec.get("status") or {}
    flags = rec.get("flags") or []
    fc = rec.get("factcheck")

    label = _TIER_LABEL.get(tier.get("grade", ""), tier.get("grade", _NA))
    out: list[str] = ["<details>", f"<summary>{label} · {_name_code(rec)}</summary>", ""]

    # 质地档理由 + 需人工复核
    reasons = tier.get("reasons") or []
    if reasons:
        out.append("质地理由:" + "；".join(_md(r) for r in reasons))
    out.append(
        "人工复核:需要(needs_human)" if tier.get("needs_human") else "人工复核:免(规则可判)"
    )
    st_bits: list[str] = []
    if status.get("is_st"):
        st_bits.append("当前 ST")
    if status.get("ever_st"):
        st_bits.append("曾 ST")
    if st_bits:
        out.append("状态:" + "、".join(st_bits))
    out.append("")

    out += _kv_table(
        "技术（@交易日）",
        [
            ("现价", _fmt(tech.get("close"), suffix=" 元")),
            ("距60日高", _fmt(tech.get("dist60"), pct_sign=True)),
            ("近20日收益", _fmt(tech.get("ret20"), pct_sign=True)),
            ("近20分位", f"{_spark(tech.get('pct20'))} {_fmt(tech.get('pct20'), decimals=1)}"),
            ("RSI14", _fmt(tech.get("rsi"), decimals=1)),
            ("量比", _fmt(tech.get("vol_ratio"))),
            ("趋势", _md(tech.get("trend") or _NA)),
        ],
    )
    out += _kv_table(
        "估值（@交易日）",
        [
            ("PE(TTM)", _fmt(val.get("pe_ttm"), suffix=" 倍", decimals=1)),
            ("PB", _fmt(val.get("pb"), suffix=" 倍")),
            ("PEG", _fmt(val.get("peg"))),
            ("总市值", _fmt(val.get("mv_yi"), suffix=" 亿", decimals=0)),
        ],
    )
    end_date = fund.get("end_date") or "最新季"
    out += _kv_table(
        f"基本面（@{end_date}）",
        [
            ("营收", _fmt(fund.get("rev_yi"), suffix=" 亿")),
            ("营收同比", _fmt(fund.get("rev_yoy"), pct_sign=True)),
            ("归母净利", _fmt(fund.get("np_yi"), suffix=" 亿")),
            ("净利同比", _fmt(fund.get("np_yoy"), pct_sign=True)),
            ("扣非同比", _fmt(fund.get("dedt_yoy"), pct_sign=True)),
            ("扣非净利", _fmt(fund.get("dedt_yi"), suffix=" 亿")),
            ("毛利率", _fmt(fund.get("gross_margin"), pct_sign=True)),
            ("盈利", "是" if fund.get("profitable") else ("否" if fund.get("profitable") is False else _NA)),
        ],
    )
    out += _kv_table(
        "现金质量（@2025年报）",
        [
            ("经营现金流", _fmt(qual.get("op_cashflow_yi"), suffix=" 亿")),
            ("净现比", _fmt(qual.get("net_cash_ratio"), suffix=" 倍")),
            ("应计", _fmt(qual.get("accrual"), suffix=" 倍")),
        ],
    )
    out += _kv_table(
        "资产（@最新季）",
        [
            ("应收账款", _fmt(bal.get("accounts_receiv_yi"), suffix=" 亿")),
            ("商誉", _fmt(bal.get("goodwill_yi"), suffix=" 亿")),
            ("货币资金", _fmt(bal.get("money_cap_yi"), suffix=" 亿")),
            ("净资产", _fmt(bal.get("net_assets_yi"), suffix=" 亿")),
            ("总资产", _fmt(bal.get("total_assets_yi"), suffix=" 亿")),
            ("应收/年报净利", _fmt(bal.get("recv_to_annual_net_pct"), pct_sign=True)),
        ],
    )

    # 事件:中性提示旗 + 一句事实 + 日期(绝不买入触发/不恐吓)
    out.append("**事件（中性提示旗）**")
    out.append("")
    if flags:
        for f in flags:
            ftype = _md(f.get("type", ""))
            sev = _md(f.get("severity", "提示"))
            fact = _md(f.get("fact", ""))
            date = _md(f.get("date", _NA))
            src = _md(f.get("source", ""))
            out.append(f"- 🚩 [{sev}] {ftype}:{fact}（{date}{('，' + src) if src else ''}）")
    else:
        out.append(f"- {_NA} 无事件提示")
    out.append("")

    # factcheck:非空则与接口数字并列,标"独立核实"
    if isinstance(fc, dict):
        out.append("**独立核实（人工/外部,非接口直出）**")
        out.append("")
        confirmed = fc.get("confirmed")
        conf_s = "已确认" if confirmed else ("存疑" if confirmed is False else _NA)
        out.append(f"- 结论:{conf_s}")
        out.append(f"- 核实Q1归母净利:{_fmt(fc.get('q1_net_profit_yi'), suffix=' 亿')}（独立核实,与接口 income 并列）")
        disputes = fc.get("disputes") or []
        if disputes:
            out.append("- 争议:" + "；".join(_md(d) for d in disputes))
        news = fc.get("news") or []
        if news:
            out.append("- 消息:" + "；".join(_md(n) for n in news))
        out.append(f"- 核实时点:{_md(fc.get('verified_at', _NA))}")
        out.append("")

    out += ["</details>", ""]
    return out


def _footnote(records: list[dict[str, Any]]) -> list[str]:
    """③ 口径脚注:从 meta 提炼时点 + 来源汇总。"""
    trade_dates: set[str] = set()
    fund_dates: set[str] = set()
    cash_dates: set[str] = set()
    sources: set[str] = set()

    for rec in records:
        meta = rec.get("meta") or {}
        for path, info in meta.items():
            if not isinstance(info, dict):
                continue
            as_of_v = info.get("as_of")
            src = info.get("source")
            if isinstance(src, str) and src:
                sources.add(src)
            if not isinstance(as_of_v, str):
                continue
            if path.startswith(("technical.", "valuation.")):
                trade_dates.add(as_of_v)
            elif path.startswith(("fundamental.", "balance.")):
                fund_dates.add(as_of_v)
            elif path.startswith("quality."):
                cash_dates.add(as_of_v)

    def _join(s: set[str]) -> str:
        return "/".join(sorted(s)) if s else _NA

    src_kind = "tushare 接口" if sources else _NA

    lines: list[str] = [
        "---",
        "",
        "**口径脚注（数值就近标注;此处汇总时点与来源）**",
        "",
        f"- 估值/技术 @ 交易日 {_join(trade_dates)}",
        f"- 基本面/资产 @ 最新季 {_join(fund_dates)}",
        f"- 现金质量 @ 年报 {_join(cash_dates)}",
        f"- 数据来源:{src_kind}",
        (
            "- 一切结论用离散档(质地 🟢🟡🔴⛔ / entry A·B·C),小数 score 仅供组内排序、"
            "不作结论;事件为中性提示旗,非买入触发器。缺失值以「—」占位。"
        ),
    ]
    return lines


def render_md(records: list[dict[str, Any]]) -> str:
    """渲染三层渐进诚实面板(纯函数,返回 Markdown 字符串)。"""
    ordered = _sorted_records(records)

    counts: dict[str, int] = {}
    for rec in ordered:
        g = (rec.get("tier") or {}).get("grade", _NA)
        counts[g] = counts.get(g, 0) + 1
    tally = " · ".join(
        f"{_TIER_LABEL.get(g, g)} {counts[g]}"
        for g in sorted(counts, key=lambda x: _TIER_ORDER.get(x, 99))
    )

    lines: list[str] = [
        "# A股选股诚实面板",
        "",
        f"标的 {len(ordered)} 只 · {tally if tally else '无'}",
        "",
        "## 一、概览（质地档分组 ⛔→🔴→🟡→🟢，组内按 entry 分数排）",
        "",
    ]
    lines += _overview_table(ordered)
    lines += ["", "## 二、逐只展开（折叠）", ""]
    for rec in ordered:
        lines += _detail_section(rec)
    lines += _footnote(ordered)

    return "\n".join(lines) + "\n"
