"""多股 HTML dashboard 总览决策台(纯渲染,无 IO)。

``render_dashboard(records) -> str`` 返回自含单文件 HTML(内联 CSS/JS,零外链):

  顶部组合层摘要(决策台):tier 分布(离散档计数)、行业集中度(top 占比文字)、
    市值分桶分布、预算参考(memory trading-constraints:可买沪深主板、预算约 6 万、
    逐步加仓;**不臆造持仓规模**,仅"若等权选中 🟢/entryA 若干只"的占用参考,
    明确标注为参考、非买入建议)。
  可排序/过滤表:列 [质地档(emoji+文字) | 名称(代码) 行业 | entry 档 |
    技术/基本/资金 三条分项条 | 价格/营收/资金流 sparkline small-multiples | 旗标数]。
    表头点击排序、按 tier/行业过滤(纯原生 JS,无框架)。
  点行展开钻取:内联 B 的 ``render_svg.render_svg_card(record, cohort=records)``;
    hover sparkline 出 tooltip。

两条产品铁律:① 结论只用离散档(tier emoji+文字档名 / entry A·B·C),绝不把 entry.score
小数当结论;每个数值就近标注口径/时点/来源。② 资金/事件(flags)只做中性提示旗
(图标+一句事实+日期),绝不做买入触发器、不用红色恐吓;色盲安全:颜色之外必有图标/文字。

缺失值(None)一律渲染为占位 "—"(绝不当 0 或省略)。
"""

from __future__ import annotations

import html as _html
from typing import Any, Optional

from ashare_gauntlet.render_svg import render_svg_card

# ---------------------------------------------------------------------------
# 常量(色盲安全:颜色仅辅助,文字档名+图标承载语义)
# ---------------------------------------------------------------------------
NA: str = "—"

# 预算参考(memory trading-constraints):约 6 万、逐步加仓;不臆造持仓规模。
BUDGET_YUAN: int = 60000

_TIER_ORDER: dict[str, int] = {"🟢": 0, "🟡": 1, "🔴": 2, "⛔": 3}
_TIER_LABEL: dict[str, str] = {
    "🟢": "强干净",
    "🟡": "盈利瑕疵",
    "🔴": "题材背离",
    "⛔": "地雷出局",
}
_TIER_FILL: dict[str, str] = {
    "🟢": "#1b7f37",
    "🟡": "#9a7d0a",
    "🔴": "#b23b3b",
    "⛔": "#6b6b6b",
}

_FLAG_ICON: dict[str, str] = {"解禁": "🔓", "减持": "📉", "质押": "🔗", "超预期": "✨"}

_BLOCKS: str = "▁▂▃▄▅▆▇█"

# 市值分桶(亿元):上界递增,最后一桶为 +∞。
_MV_BUCKETS: list[tuple[float, str]] = [
    (100.0, "<100亿(小盘)"),
    (300.0, "100–300亿"),
    (1000.0, "300–1000亿"),
    (3000.0, "1000–3000亿"),
    (float("inf"), "≥3000亿(大盘)"),
]


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _esc(s: object) -> str:
    """HTML 文本/属性转义(防 < > & " ' 破坏 DOM / 属性注入)。"""
    return _html.escape(str(s), quote=True)


def _num(rec: dict[str, Any], dotted: str) -> Optional[float]:
    """按点号路径取数值;缺失/非数/bool 返回 None(绝不 0)。"""
    cur: Any = rec
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    if cur is None or isinstance(cur, bool):
        return None
    try:
        return float(cur)
    except (TypeError, ValueError):
        return None


def _fmt(v: Optional[float], digits: int = 2, suffix: str = "", *, plus: bool = False) -> str:
    """数值格式化;None -> 占位 "—"(不伪造 0、不省略)。"""
    if v is None:
        return NA
    body = f"{v:+.{digits}f}" if plus else f"{v:.{digits}f}"
    return f"{body}{suffix}"


def _spark1(pct: Optional[float]) -> str:
    """单字符 sparkline 区块(百分位 0–100 映射到 ▁..█);None -> 占位。"""
    if pct is None:
        return NA
    p = max(0.0, min(100.0, pct))
    return _BLOCKS[int(p / 100.0 * (len(_BLOCKS) - 1) + 0.5)]


def _percentile(values: list[float], x: float) -> Optional[float]:
    """x 在 values 中的百分位(0–100,含自身,中点法);values 空 -> None。"""
    n = len(values)
    if n == 0:
        return None
    below = sum(1 for v in values if v < x)
    equal = sum(1 for v in values if v == x)
    return 100.0 * (below + 0.5 * equal) / n


def _collect(records: list[dict[str, Any]], dotted: str) -> list[float]:
    out: list[float] = []
    for r in records:
        v = _num(r, dotted)
        if v is not None:
            out.append(v)
    return out


# ---------------------------------------------------------------------------
# 三条分项条:技术 / 基本 / 资金(均 cohort 百分位归一化,0–100;低分≠坏)
# ---------------------------------------------------------------------------
def _dim_scores(
    rec: dict[str, Any],
    pcts: dict[str, list[float]],
) -> dict[str, Optional[float]]:
    """技术=动量分位(pct20);基本=扣非增速分位;资金=净现比分位(现金质量代理)。

    预先算好的 cohort 值池 pcts 传入,避免逐行重复收集。缺失维度返回 None(灰条+—)。
    """
    out: dict[str, Optional[float]] = {}
    tech = _num(rec, "technical.pct20")
    out["技术"] = _percentile(pcts["technical.pct20"], tech) if tech is not None else None
    fund = _num(rec, "fundamental.dedt_yoy")
    out["基本"] = _percentile(pcts["fundamental.dedt_yoy"], fund) if fund is not None else None
    flow = _num(rec, "quality.net_cash_ratio")
    out["资金"] = _percentile(pcts["quality.net_cash_ratio"], flow) if flow is not None else None
    return out


def _bar_cell(label: str, score: Optional[float], raw_txt: str) -> str:
    """一条分项条(内联 bar + 文字;缺失灰条 + —)。data-排序键放在 td 上。"""
    sort_v = -1.0 if score is None else score
    if score is None:
        inner = (
            f'<div class="bar"><i class="fill na" style="width:100%"></i></div>'
            f'<span class="bv">{NA}</span>'
        )
    else:
        inner = (
            f'<div class="bar"><i class="fill" style="width:{score:.0f}%"></i></div>'
            f'<span class="bv">{score:.0f}</span>'
        )
    title = f"{label}分位 {NA if score is None else f'{score:.0f}'}/100 · {raw_txt}(cohort 百分位;低分≠坏,非买卖信号)"
    return f'<td class="dim" data-v="{sort_v:.2f}" title="{_esc(title)}">{inner}</td>'


# ---------------------------------------------------------------------------
# 组合层摘要(决策台顶部)
# ---------------------------------------------------------------------------
def _summary(records: list[dict[str, Any]]) -> str:
    n = len(records)

    # tier 分布(离散档计数)
    tier_counts: dict[str, int] = {}
    for r in records:
        g = str((r.get("tier") or {}).get("grade", ""))
        tier_counts[g] = tier_counts.get(g, 0) + 1
    tier_bits: list[str] = []
    for g in ("🟢", "🟡", "🔴", "⛔"):
        c = tier_counts.get(g, 0)
        lab = _TIER_LABEL.get(g, g)
        tier_bits.append(
            f'<span class="chip" style="border-color:{_TIER_FILL.get(g, "#999")}">'
            f"{g} {lab} <b>{c}</b></span>"
        )
    tier_html = "".join(tier_bits)

    # 行业集中度(top 占比文字)
    ind_counts: dict[str, int] = {}
    for r in records:
        ind = str(r.get("industry") or NA)
        ind_counts[ind] = ind_counts.get(ind, 0) + 1
    top_inds = sorted(ind_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:4]
    if n and top_inds:
        ind_bits = "、".join(
            f"{_esc(ind)} {c}只({100.0 * c / n:.0f}%)" for ind, c in top_inds
        )
        top_ind, top_c = top_inds[0]
        conc_note = (
            f"行业集中度:最集中「{_esc(top_ind)}」{top_c}只占 {100.0 * top_c / n:.0f}%;"
            f"Top{len(top_inds)} = {ind_bits}。"
        )
    else:
        conc_note = "行业集中度:无标的。"

    # 市值分桶分布
    mv_counts: list[int] = [0] * len(_MV_BUCKETS)
    mv_na = 0
    for r in records:
        mv = _num(r, "valuation.mv_yi")
        if mv is None:
            mv_na += 1
            continue
        for i, (hi, _lab) in enumerate(_MV_BUCKETS):
            if mv < hi:
                mv_counts[i] += 1
                break
    mv_bits = " · ".join(
        f"{lab} <b>{mv_counts[i]}</b>" for i, (_hi, lab) in enumerate(_MV_BUCKETS) if mv_counts[i]
    )
    if mv_na:
        mv_bits += f" · 市值缺失 <b>{mv_na}</b>"
    if not mv_bits:
        mv_bits = NA

    # 预算参考(不臆造持仓规模;仅"若等权选中 🟢/entryA 若干只"占用参考)
    green_a = [
        r for r in records
        if str((r.get("tier") or {}).get("grade", "")) == "🟢"
        and str((r.get("entry") or {}).get("grade", "")) == "A"
    ]
    k = len(green_a)
    if k:
        per = BUDGET_YUAN / k
        budget_note = (
            f"预算参考(非建议):若以约 {BUDGET_YUAN:,} 元预算等权占满当前 🟢 强干净 × entry A "
            f"{k} 只,单只约 {per:,.0f} 元;逐步加仓、不一次性满仓。"
        )
    else:
        budget_note = (
            f"预算参考(非建议):约 {BUDGET_YUAN:,} 元预算;当前无 🟢×entryA 标的,留位、暂不占用。"
        )

    return (
        '<section class="summary" aria-label="组合层摘要">'
        "<h2>组合层摘要 · 决策台<span class=\"sub\">（参考非建议；结论用离散档，非买卖信号）</span></h2>"
        f'<div class="srow"><span class="slabel">质地档分布</span>'
        f'<div class="chips">{tier_html}</div></div>'
        f'<div class="srow"><span class="slabel">行业集中度</span>'
        f'<div class="stext">{conc_note}</div></div>'
        f'<div class="srow"><span class="slabel">市值分布</span>'
        f'<div class="stext">{mv_bits}（按总市值 mv_yi 分桶，@交易日）</div></div>'
        f'<div class="srow"><span class="slabel">预算参考</span>'
        f'<div class="stext note">{_esc(budget_note)}</div></div>'
        "</section>"
    )


# ---------------------------------------------------------------------------
# 过滤控件
# ---------------------------------------------------------------------------
def _controls(records: list[dict[str, Any]]) -> str:
    inds = sorted({str(r.get("industry") or NA) for r in records})
    ind_opts = '<option value="">行业:全部</option>' + "".join(
        f'<option value="{_esc(i)}">{_esc(i)}</option>' for i in inds
    )
    tier_opts = (
        '<option value="">质地档:全部</option>'
        '<option value="🟢">🟢 强干净</option>'
        '<option value="🟡">🟡 盈利瑕疵</option>'
        '<option value="🔴">🔴 题材背离</option>'
        '<option value="⛔">⛔ 地雷出局</option>'
    )
    return (
        '<div class="controls" role="group" aria-label="过滤">'
        '<label>质地 <select id="fTier" onchange="applyFilter()">'
        f"{tier_opts}</select></label>"
        '<label>行业 <select id="fInd" onchange="applyFilter()">'
        f"{ind_opts}</select></label>"
        '<span class="hint">点表头排序 · 点行展开单股卡 · 悬停分项条看口径</span>'
        '<span id="shown" class="hint"></span>'
        "</div>"
    )


# ---------------------------------------------------------------------------
# 表
# ---------------------------------------------------------------------------
_COLS: list[tuple[str, str, str]] = [
    # (表头文字, 排序类型 num|text, 提示)
    ("质地档", "num", "tier 离散档（⛔→🟢 排序键）"),
    ("名称(代码) · 行业", "text", "名称/代码/行业"),
    ("entry", "text", "入场离散档 A/B/C"),
    ("技术", "num", "动量分位（pct20，cohort 百分位）"),
    ("基本", "num", "扣非增速分位（dedt_yoy，cohort 百分位）"),
    ("资金", "num", "现金质量分位（净现比，cohort 百分位）"),
    ("价格/营收/资金流", "text", "small-multiples sparkline（价距60高 / 营收同比 / 净现比）"),
    ("旗", "num", "中性提示旗数量"),
    ("趋势", "text", "均线趋势"),
]


def _spark_multiples(rec: dict[str, Any]) -> str:
    """价格/营收/资金流 small-multiples:三枚单字符 sparkline + a11y title。缺失 -> —。"""
    # 价格强度:用距60日高(越接近 0 越强,映射到 0..100 = 100+dist60 截断)
    dist60 = _num(rec, "technical.dist60")
    price_p = None if dist60 is None else max(0.0, min(100.0, 100.0 + dist60))
    rev_yoy = _num(rec, "fundamental.rev_yoy")
    rev_p = None if rev_yoy is None else max(0.0, min(100.0, 50.0 + rev_yoy / 4.0))
    ncr = _num(rec, "quality.net_cash_ratio")
    flow_p = None if ncr is None else max(0.0, min(100.0, 50.0 + ncr * 20.0))
    title = (
        f"价(距60高 {_fmt(dist60, 1, '%', plus=True)}) · "
        f"营收同比 {_fmt(rev_yoy, 1, '%', plus=True)} · "
        f"净现比 {_fmt(ncr, 2, '倍')}"
    )
    return (
        f'<span class="sparkm" title="{_esc(title)}">'
        f"{_spark1(price_p)}{_spark1(rev_p)}{_spark1(flow_p)}</span>"
    )


def _row(rec: dict[str, Any], idx: int, cohort: list[dict[str, Any]], pcts: dict[str, list[float]]) -> str:
    tier = rec.get("tier") or {}
    grade = str(tier.get("grade", "")) or NA
    reasons = tier.get("reasons") or []
    words = str(reasons[0]) if reasons else _TIER_LABEL.get(grade, "未定档")
    fill = _TIER_FILL.get(grade, "#999")
    tier_sort = _TIER_ORDER.get(grade, 99)
    needs = bool(tier.get("needs_human"))

    entry = rec.get("entry") or {}
    eg = str(entry.get("grade", NA))
    name = str(rec.get("name", NA))
    code = str(rec.get("ts_code", NA))
    industry = str(rec.get("industry", NA))
    tech = rec.get("technical") or {}
    trend = str(tech.get("trend") or NA)
    flags = rec.get("flags") or []
    nflags = len(flags)

    dims = _dim_scores(rec, pcts)
    human_badge = ' <span class="human" title="规则未定，待人工复核">⚑人工</span>' if needs else ""

    tier_cell = (
        f'<td class="tier" data-v="{tier_sort}" data-tier="{_esc(grade)}">'
        f'<span class="badge" style="background:{fill}" '
        f'title="{_esc(grade)} {_esc(words)}{"（待人工复核）" if needs else ""}">'
        f"{grade} {_esc(words)}</span>{human_badge}</td>"
    )
    name_cell = (
        f'<td class="name" data-v="{_esc(name)}" data-ind="{_esc(industry)}">'
        f'<b>{_esc(name)}</b> <span class="code">({_esc(code)})</span>'
        f'<span class="ind">{_esc(industry)}</span></td>'
    )
    entry_cell = f'<td class="entry" data-v="{_esc(eg)}"><span class="eg">{_esc(eg)}</span></td>'

    bars = (
        _bar_cell("技术", dims["技术"], f"动量分位 pct20={_fmt(_num(rec,'technical.pct20'),0)}")
        + _bar_cell("基本", dims["基本"], f"扣非同比 {_fmt(_num(rec,'fundamental.dedt_yoy'),1,'%',plus=True)}")
        + _bar_cell("资金", dims["资金"], f"净现比 {_fmt(_num(rec,'quality.net_cash_ratio'),2,'倍')}")
    )
    spark_cell = f'<td class="spark" data-v="{_esc(name)}">{_spark_multiples(rec)}</td>'
    flag_cell = (
        f'<td class="flags" data-v="{nflags}" title="中性提示旗（非买卖信号）">'
        f'{("🚩×" + str(nflags)) if nflags else "·"}</td>'
    )
    trend_cell = f'<td class="trend" data-v="{_esc(trend)}">{_esc(trend)}</td>'

    # 钻取详情行:内联 B 的单股 SVG 卡(cohort=全量)
    try:
        svg = render_svg_card(rec, cohort=cohort)
    except Exception as exc:  # noqa: BLE001 — 不吞:把渲染失败显式标到 UI,便于定位坏 record
        svg = f'<p class="err">单股卡渲染失败:{_esc(type(exc).__name__)}: {_esc(exc)}</p>'
    detail = (
        f'<tr class="detail" id="d{idx}" hidden><td colspan="{len(_COLS)}">'
        f'<div class="card">{svg}</div></td></tr>'
    )

    main = (
        f'<tr class="r" data-tier="{_esc(grade)}" data-ind="{_esc(industry)}" '
        f'onclick="toggle({idx})" tabindex="0" '
        f'aria-label="{_esc(name)} {_esc(code)} {_esc(grade)} {_esc(words)}">'
        f"{tier_cell}{name_cell}{entry_cell}{bars}{spark_cell}{flag_cell}{trend_cell}</tr>"
    )
    return main + detail


def _table(records: list[dict[str, Any]]) -> str:
    # 预收集 cohort 分位值池(三条分项条共用,避免逐行重复)
    pcts = {
        "technical.pct20": _collect(records, "technical.pct20"),
        "fundamental.dedt_yoy": _collect(records, "fundamental.dedt_yoy"),
        "quality.net_cash_ratio": _collect(records, "quality.net_cash_ratio"),
    }
    # 排序:与 md 一致 ⛔→🔴→🟡→🟢?这里默认 🟢 在前(决策台优先看强档),组内 entry.score 降序。
    def _key(r: dict[str, Any]) -> tuple[int, float]:
        g = str((r.get("tier") or {}).get("grade", ""))
        s = (r.get("entry") or {}).get("score")
        sf = float(s) if isinstance(s, (int, float)) else -1.0
        return (_TIER_ORDER.get(g, 99), -sf)

    ordered = sorted(records, key=_key)

    ths: list[str] = []
    for i, (label, kind, tip) in enumerate(_COLS):
        ths.append(
            f'<th data-kind="{kind}" onclick="sortBy({i})" title="{_esc(tip)}" '
            f'tabindex="0" role="columnheader" aria-sort="none">'
            f'{_esc(label)}<span class="arr"></span></th>'
        )
    head = "<thead><tr>" + "".join(ths) + "</tr></thead>"

    body_rows = "".join(_row(r, i, ordered, pcts) for i, r in enumerate(ordered))
    body = f"<tbody id=\"tb\">{body_rows}</tbody>"
    return f'<table id="grid" aria-label="多股总览表">{head}{body}</table>'


# ---------------------------------------------------------------------------
# 内联 CSS / JS(零外链)
# ---------------------------------------------------------------------------
_CSS: str = """
:root{--ink:#222;--mute:#666;--line:#e3e3e3;--bg:#fff;--accent:#4a78c8;--track:#eee;}
*{box-sizing:border-box}
body{margin:0;font-family:'Segoe UI','Microsoft YaHei',system-ui,sans-serif;color:var(--ink);background:#f7f7f8;font-size:14px}
.wrap{max-width:1180px;margin:0 auto;padding:18px}
h1{font-size:22px;margin:0 0 2px}
.lead{color:var(--mute);margin:0 0 14px;font-size:13px}
.summary{background:var(--bg);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin-bottom:14px}
.summary h2{font-size:16px;margin:0 0 10px}
.summary .sub{font-weight:400;color:var(--mute);font-size:12px;margin-left:8px}
.srow{display:flex;gap:12px;align-items:baseline;padding:4px 0;border-top:1px dashed var(--line)}
.srow:first-of-type{border-top:none}
.slabel{flex:0 0 88px;color:var(--mute);font-size:12px}
.stext{flex:1;line-height:1.5}
.stext.note{color:#5a4a00;background:#fff8e1;border-radius:6px;padding:4px 8px}
.chips{display:flex;flex-wrap:wrap;gap:8px}
.chip{border:2px solid #999;border-radius:14px;padding:2px 10px;font-size:13px;background:#fafafa}
.chip b{font-size:14px}
.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-bottom:8px}
.controls select{font:inherit;padding:3px 6px;border:1px solid #ccc;border-radius:6px}
.hint{color:var(--mute);font-size:12px}
table{width:100%;border-collapse:collapse;background:var(--bg);border:1px solid var(--line);border-radius:10px;overflow:hidden}
th,td{padding:7px 9px;text-align:left;border-bottom:1px solid var(--line);vertical-align:middle}
th{position:sticky;top:0;background:#f0f1f3;cursor:pointer;user-select:none;font-size:12px;white-space:nowrap}
th .arr{margin-left:4px;color:var(--mute);font-size:10px}
tr.r{cursor:pointer}
tr.r:hover{background:#f3f7ff}
tr.r:focus{outline:2px solid var(--accent);outline-offset:-2px}
.badge{color:#fff;border-radius:6px;padding:2px 8px;font-size:13px;white-space:nowrap}
.human{color:#9a7d0a;font-size:11px}
.code{color:var(--mute);font-size:12px}
.ind{display:block;color:var(--mute);font-size:11px}
.eg{display:inline-block;min-width:20px;text-align:center;font-weight:700;border:1px solid #ccd;border-radius:5px;padding:1px 7px;background:#eef2fb}
td.dim{min-width:96px}
.bar{display:inline-block;width:66px;height:9px;background:var(--track);border-radius:5px;overflow:hidden;vertical-align:middle}
.bar .fill{display:block;height:100%;background:var(--accent)}
.bar .fill.na{background:#cfcfcf}
.bv{margin-left:6px;font-size:11px;color:var(--mute)}
.sparkm{font-size:16px;letter-spacing:1px;font-family:'Cascadia Mono',Consolas,monospace}
td.flags{text-align:center}
tr.detail>td{background:#fbfbfd;padding:10px}
.card{display:flex;justify-content:center}
.card svg{max-width:100%;height:auto}
.err{color:#b23b3b}
footer{color:var(--mute);font-size:12px;margin-top:14px;line-height:1.6}
"""

# JS:排序(数值/文本,toggle 升降)、过滤(tier+行业,联动隐藏 detail 行)、行展开。
# 纯原生,无框架,无外链。
_JS: str = r"""
var asc = {};
function rows(){return Array.prototype.slice.call(document.querySelectorAll('#tb tr.r'));}
function detailOf(tr){var n=tr.nextElementSibling;return (n&&n.classList.contains('detail'))?n:null;}
function sortBy(col){
  var tb=document.getElementById('tb');
  var ths=document.querySelectorAll('#grid th');
  var kind=ths[col].getAttribute('data-kind');
  asc[col]=!asc[col]; var dir=asc[col]?1:-1;
  for(var i=0;i<ths.length;i++){ths[i].setAttribute('aria-sort','none');var a=ths[i].querySelector('.arr');if(a)a.textContent='';}
  ths[col].setAttribute('aria-sort',asc[col]?'ascending':'descending');
  var arr=ths[col].querySelector('.arr'); if(arr)arr.textContent=asc[col]?'▲':'▼';
  var rs=rows();
  rs.sort(function(a,b){
    var ca=a.children[col], cb=b.children[col];
    var va=ca.getAttribute('data-v'), vb=cb.getAttribute('data-v');
    if(kind==='num'){return (parseFloat(va)-parseFloat(vb))*dir;}
    return String(va).localeCompare(String(vb),'zh')*dir;
  });
  rs.forEach(function(tr){var d=detailOf(tr);tb.appendChild(tr);if(d)tb.appendChild(d);});
}
function toggle(i){
  var d=document.getElementById('d'+i);
  if(d){d.hidden=!d.hidden;}
}
function applyFilter(){
  var t=document.getElementById('fTier').value;
  var ind=document.getElementById('fInd').value;
  var shown=0, tot=0;
  rows().forEach(function(tr){
    tot++;
    var ok=(!t||tr.getAttribute('data-tier')===t)&&(!ind||tr.getAttribute('data-ind')===ind);
    tr.style.display=ok?'':'none';
    var d=detailOf(tr);
    if(d){d.hidden=true; d.style.display=ok?'':'none';}
    if(ok)shown++;
  });
  var s=document.getElementById('shown');
  if(s)s.textContent='显示 '+shown+' / '+tot+' 只';
}
document.addEventListener('DOMContentLoaded',applyFilter);
"""


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def render_dashboard(records: list[dict[str, Any]]) -> str:
    """渲染多股 HTML dashboard 决策台(纯函数,返回自含单文件 HTML 字符串)。

    Args:
        records: 全量 card records(schema 见 data/cards/<as_of>.json)。任一数值可能为 None。

    铁律:结论只用离散档(tier emoji+文字档名 / entry 字母),不把 score 小数当结论;
    数值就近标注口径时点;flags 仅中性提示旗、非买入触发器;色盲安全(图标+文字);
    缺失值降级占位「—」,不伪造 0;HTML 转义防注入;零外链、内联 CSS/JS。
    """
    as_of = NA
    for r in records:
        a = r.get("as_of")
        if isinstance(a, str) and a:
            as_of = a
            break

    n = len(records)
    if n:
        summary = _summary(records)
        controls = _controls(records)
        table = _table(records)
        body = summary + controls + table
    else:
        # 空集诚实成文,不崩
        body = (
            '<section class="summary"><h2>组合层摘要 · 决策台'
            '<span class="sub">（参考非建议）</span></h2>'
            "<div class=\"srow\"><div class=\"stext\">无标的（空 records）。</div></div></section>"
            '<table id="grid"><thead><tr><th>质地档</th></tr></thead>'
            '<tbody id="tb"></tbody></table>'
        )

    return (
        "<!DOCTYPE html>\n"
        '<html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>A股选股总览决策台 · {_esc(as_of)}</title>"
        f"<style>{_CSS}</style></head><body><div class=\"wrap\">"
        f"<h1>A股选股总览决策台<span class=\"sub\"></span></h1>"
        f'<p class="lead">标的 {n} 只 · 截面 {_esc(as_of)} · '
        "结论用离散档(质地 🟢🟡🔴⛔ / entry A·B·C),非买卖信号;"
        "事件为中性提示旗;缺失值以「—」占位;数值就近标注口径/时点/来源。</p>"
        f"{body}"
        '<footer>口径:估值/技术 @ 交易日 · 基本面/资产 @ 最新季 · 现金质量 @ 年报;'
        "三条分项条与雷达均为 cohort 百分位归一化,低分≠坏、仅偏科直觉,非买卖信号。"
        "组合摘要的预算占用为参考、非买入建议,不臆造持仓规模。"
        "<br>数据来源 tushare 接口直出 + 规则定档;本页纯本地渲染、不联网。</footer>"
        f"</div><script>{_JS}</script></body></html>"
    )
