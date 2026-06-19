"""单股 SVG 卡(深挖重点票)——纯渲染函数,无 IO,返回完整自含 ``<svg ...>...</svg>``。

设计遵循两条产品铁律:
  ① 一切结论用离散档(tier 🟢🟡🔴⛔ / entry A/B/C),绝不把 entry.score 小数当结论;
     每个数值就近标注口径/时点(估值@交易日 / 基本面@最新季 / 现金质量@年报)。
  ② 资金/事件面(flags)只做中性提示旗 + 一句事实 + 日期,绝不当买入触发器、不用红色恐吓;
     色盲安全:颜色之外必有图标/文字(tier emoji 含图标,另补文字档名)。

雷达雪花图固定 6 轴(轴序固定):
  估值(EP=1/pe_ttm) | 成长(np_yoy) | 盈利质量(net_cash_ratio)
  | 现金质量(-accrual,越大越干净) | 风险(-质押% 或 -recv占比,越大越安全) | 动量(pct20)
各轴用 cohort 百分位归一化到 0-100;**低分≠坏,仅偏科直觉,非买卖信号**。

子弹图:PE / 扣非增速(dedt_yoy) / 净现比(net_cash_ratio) vs 对比集中位。
对比集=cohort 中同行业(≥3 只)否则全 cohort;**明确标注对比集与口径时点**
(诚实:无全市场行业中位数据,用 cohort 中位)。

缺失值的轴/子弹降级显示(灰条 + "—"),不报错、不伪造 0。
"""
from __future__ import annotations

from typing import Any, Callable, Optional

# ---------------------------------------------------------------------------
# 常量:画布 / 配色(色盲安全:颜色仅作辅助,文字档名+图标承载语义)
# ---------------------------------------------------------------------------
W: int = 640
H: int = 760
PLACEHOLDER: str = "—"

# tier 档:emoji 图标 + 中性配色(色盲安全靠图标+文字,不靠颜色区分)
TIER_FILL: dict[str, str] = {
    "🟢": "#1b7f37",  # 绿
    "🟡": "#9a7d0a",  # 琥珀
    "🔴": "#b23b3b",  # 暗红(中性,不刺眼)
    "⛔": "#6b6b6b",  # 灰(地雷=回避,非恐吓红)
}
GRAY: str = "#bdbdbd"  # 缺失/降级灰条
INK: str = "#222222"
MUTE: str = "#666666"
TRACK: str = "#e8e8e8"  # 子弹图底轨
GRID: str = "#d8d8d8"   # 雷达网格

# flags 类型 -> 中性提示图标(绝不用恐吓红/爆炸)
FLAG_ICON: dict[str, str] = {
    "解禁": "🔓",
    "减持": "📉",
    "质押": "🔗",
    "超预期": "✨",
}


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _esc(s: object) -> str:
    """XML 文本/属性转义(防裸 & < > " ' 破坏 SVG)。"""
    t = str(s)
    return (
        t.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _num(record: dict[str, Any], dotted: str) -> Optional[float]:
    """按点号路径取数值;缺失/非数返回 None(绝不 0)。"""
    cur: Any = record
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


def _fmt(v: Optional[float], digits: int = 1, suffix: str = "") -> str:
    """数值格式化;None -> 占位符(不伪造 0、不省略)。"""
    if v is None:
        return PLACEHOLDER
    return f"{v:.{digits}f}{suffix}"


def _percentile(values: list[float], x: float) -> float:
    """x 在 values 中的百分位(0-100),含自身;values 非空。"""
    n = len(values)
    if n == 0:
        return 50.0
    below = sum(1 for v in values if v < x)
    equal = sum(1 for v in values if v == x)
    # 中点法:更稳,单点 cohort 给 50
    return 100.0 * (below + 0.5 * equal) / n


# ---------------------------------------------------------------------------
# 轴定义:label + 取值器(把 record 映射到"越大越好"的方向)
# 风险轴用 -质押%(无质押则 -recv占比),方向统一为"越大越安全/越好"。
# ---------------------------------------------------------------------------
def _pledge_pct(record: dict[str, Any]) -> Optional[float]:
    """从 flags 解析控股股东体系质押百分比(中性事实,仅用于风险轴归一化)。"""
    for fl in record.get("flags", []) or []:
        if isinstance(fl, dict) and fl.get("type") == "质押":
            fact = str(fl.get("fact", ""))
            # 形如 "控股股东体系质押12.9%"
            digits = ""
            seen_dot = False
            for ch in fact:
                if ch.isdigit():
                    digits += ch
                elif ch == "." and not seen_dot and digits:
                    digits += ch
                    seen_dot = True
                elif digits:
                    break
            if digits:
                try:
                    return float(digits)
                except ValueError:
                    return None
    return None


def _risk_value(record: dict[str, Any]) -> Optional[float]:
    """风险轴(越大越安全):优先 -质押%;无质押旗标则 -应收/年净利占比。"""
    p = _pledge_pct(record)
    if p is not None:
        return -p
    recv = _num(record, "balance.recv_to_annual_net_pct")
    return None if recv is None else -recv


# (label, 取值器, 该轴口径脚注)
AXES: list[tuple[str, Callable[[dict[str, Any]], Optional[float]], str]] = [
    ("估值", lambda r: (None if (_num(r, "valuation.pe_ttm") in (None, 0.0))
                        else 1.0 / float(_num(r, "valuation.pe_ttm") or 0.0)), "EP=1/PE"),
    ("成长", lambda r: _num(r, "fundamental.np_yoy"), "归母净利同比"),
    ("盈利质量", lambda r: _num(r, "quality.net_cash_ratio"), "净现比"),
    ("现金质量", lambda r: (None if _num(r, "quality.accrual") is None
                        else -float(_num(r, "quality.accrual") or 0.0)), "-应计"),
    ("风险", _risk_value, "-质押%/-应收占比"),
    ("动量", lambda r: _num(r, "technical.pct20"), "20日收益分位"),
]


# ---------------------------------------------------------------------------
# 对比集:同行业 ≥3 只则同行业,否则全 cohort
# ---------------------------------------------------------------------------
def _cohort_for_bullets(
    record: dict[str, Any], cohort: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], str]:
    """返回 (对比集, 口径文字)。诚实标注:无全市场行业中位数据,用 cohort 中位。"""
    if not cohort:
        return [record], "对比集:仅本股(无 cohort)"
    ind = record.get("industry")
    same = [r for r in cohort if r.get("industry") == ind] if ind else []
    if len(same) >= 3:
        return same, f"对比集:cohort 同行业「{_esc(ind)}」{len(same)}只 · 中位"
    return cohort, f"对比集:全 cohort {len(cohort)}只 · 中位(同行业不足3只)"


def _median(values: list[float]) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def _collect(cohort: list[dict[str, Any]], getter: Callable[[dict[str, Any]], Optional[float]]) -> list[float]:
    out: list[float] = []
    for r in cohort:
        v = getter(r)
        if v is not None:
            out.append(v)
    return out


# ---------------------------------------------------------------------------
# 片段渲染
# ---------------------------------------------------------------------------
def _header(record: dict[str, Any]) -> str:
    tier = record.get("tier", {}) or {}
    grade = str(tier.get("grade", "")) or "⛔"
    reasons = tier.get("reasons") or []
    words = str(reasons[0]) if reasons else "未定档"
    fill = TIER_FILL.get(grade, GRAY)
    entry = record.get("entry", {}) or {}
    eg = str(entry.get("grade", PLACEHOLDER))
    name = str(record.get("name", PLACEHOLDER))
    code = str(record.get("ts_code", PLACEHOLDER))
    industry = str(record.get("industry", PLACEHOLDER))
    needs = bool(tier.get("needs_human"))
    flag_human = " · 待人工复核" if needs else ""

    return (
        f'<g font-family="Segoe UI,Microsoft YaHei,sans-serif">'
        # tier 徽章(色盲安全:emoji 图标 + 文字档名,颜色仅辅助)
        f'<rect x="16" y="16" rx="8" ry="8" width="250" height="42" fill="{fill}" '
        f'aria-label="质地档 {grade} {_esc(words)}"><title>质地档 {grade} {_esc(words)}{_esc(flag_human)}</title></rect>'
        f'<text x="28" y="44" font-size="22" fill="#ffffff">{grade} 档 {_esc(words)}</text>'
        # entry 档(离散字母,非小数 score)
        f'<rect x="278" y="16" rx="8" ry="8" width="120" height="42" fill="#33373d">'
        f'<title>入场档 {_esc(eg)}(离散档,非小数评分)</title></rect>'
        f'<text x="290" y="44" font-size="18" fill="#ffffff">入场 {_esc(eg)} 档</text>'
        # 名称(代码)+ 行业
        f'<text x="16" y="86" font-size="24" font-weight="700" fill="{INK}">'
        f'{_esc(name)} <tspan font-size="16" font-weight="400" fill="{MUTE}">({_esc(code)})</tspan></text>'
        f'<text x="16" y="110" font-size="14" fill="{MUTE}">行业 {_esc(industry)}'
        f'{_esc(flag_human)} · 结论用离散档,数值仅供口径参考</text>'
        f"</g>"
    )


def _radar(record: dict[str, Any], cohort: list[dict[str, Any]]) -> str:
    """6 轴雷达/雪花图。归一化=cohort 百分位(0-100)。缺失轴灰点 + '—'。"""
    cx, cy, R = 180, 300, 120
    pool = cohort if cohort else [record]
    n = len(AXES)
    import math

    # 预算各轴该股归一化分 + 百分位文字
    norm: list[Optional[float]] = []
    for _, getter, _foot in AXES:
        x = getter(record)
        if x is None:
            norm.append(None)
            continue
        vals = _collect(pool, getter)
        norm.append(_percentile(vals, x) if vals else 50.0)

    parts: list[str] = [
        '<g font-family="Segoe UI,Microsoft YaHei,sans-serif">'
        f'<text x="180" y="170" font-size="15" font-weight="700" fill="{INK}" text-anchor="middle">'
        f'偏科雷达(单股直觉)</text>'
    ]

    # 网格圈
    for frac in (0.25, 0.5, 0.75, 1.0):
        parts.append(
            f'<circle cx="{cx}" cy="{cy}" r="{R * frac:.1f}" fill="none" stroke="{GRID}" stroke-width="1"/>'
        )

    # 轴线 + 标签 + 数据点坐标
    pts: list[Optional[tuple[float, float]]] = []
    for i, (label, _g, foot) in enumerate(AXES):
        ang = -math.pi / 2 + 2 * math.pi * i / n
        ux, uy = math.cos(ang), math.sin(ang)
        ex, ey = cx + R * ux, cy + R * uy
        parts.append(f'<line x1="{cx}" y1="{cy}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="{GRID}" stroke-width="1"/>')
        # 标签(略外移)
        lx, ly = cx + (R + 26) * ux, cy + (R + 26) * uy
        anchor = "middle"
        if ux > 0.3:
            anchor = "start"
        elif ux < -0.3:
            anchor = "end"
        v = norm[i]
        vtxt = PLACEHOLDER if v is None else f"{v:.0f}"
        parts.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="12" fill="{INK}" text-anchor="{anchor}">'
            f'{_esc(label)}<tspan font-size="10" fill="{MUTE}"> {vtxt}</tspan>'
            f'<title>{_esc(label)} 百分位 {vtxt}(口径 {_esc(foot)};低分≠坏)</title></text>'
        )
        if v is None:
            pts.append(None)
        else:
            rr = R * (v / 100.0)
            pts.append((cx + rr * ux, cy + rr * uy))

    # 数据多边形(缺失轴回落圆心,但单独画灰点提示降级)
    poly_pts = [(p if p is not None else (cx, cy)) for p in pts]
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in poly_pts)
    parts.append(
        f'<polygon points="{poly}" fill="#4a78c8" fill-opacity="0.22" stroke="#4a78c8" stroke-width="1.5"/>'
    )
    for i, p in enumerate(pts):
        if p is None:
            # 缺失轴:灰点落在轴中段 + "—"
            ang = -math.pi / 2 + 2 * math.pi * i / n
            gx, gy = cx + R * 0.5 * math.cos(ang), cy + R * 0.5 * math.sin(ang)
            parts.append(f'<circle cx="{gx:.1f}" cy="{gy:.1f}" r="3.5" fill="{GRAY}"/>')
            parts.append(
                f'<text x="{gx:.1f}" y="{gy - 6:.1f}" font-size="10" fill="{MUTE}" text-anchor="middle">{PLACEHOLDER}</text>'
            )
        else:
            parts.append(f'<circle cx="{p[0]:.1f}" cy="{p[1]:.1f}" r="3" fill="#2f57a6"/>')

    # 归一化口径 + 免责(铁律:低分≠坏、非买卖信号)
    parts.append(
        f'<text x="16" y="448" font-size="11" fill="{MUTE}">'
        f'归一化:各轴=cohort 百分位(0–100,n={len(pool)});风险轴=-质押%/-应收占比(越大越安全)。</text>'
    )
    parts.append(
        f'<text x="16" y="464" font-size="11" font-weight="700" fill="{INK}">'
        f'低分≠坏,仅偏科直觉,非买卖信号;单股直觉不做多股叠加。</text>'
    )
    parts.append("</g>")
    return "".join(parts)


def _bullets(record: dict[str, Any], cohort: list[dict[str, Any]]) -> str:
    """三条子弹图:PE / 扣非增速 / 净现比 vs 对比集中位。缺失降级灰条 + '—'。"""
    pool, caption = _cohort_for_bullets(record, cohort)

    # (标签, 口径时点, 取值器, 区间下界, 区间上界, 单位, digits)
    specs: list[tuple[str, str, Callable[[dict[str, Any]], Optional[float]], float, float, str, int]] = [
        ("PE(TTM)", "估值@交易日", lambda r: _num(r, "valuation.pe_ttm"), 0.0, 100.0, "倍", 1),
        ("扣非增速", "基本面@最新季", lambda r: _num(r, "fundamental.dedt_yoy"), -100.0, 200.0, "%", 1),
        ("净现比", "现金质量@年报", lambda r: _num(r, "quality.net_cash_ratio"), -3.0, 5.0, "倍", 2),
    ]

    x0, bw, y = 16, W - 32 - 130, 500  # 条左 x、可用宽、首行 y
    row_h = 56
    parts: list[str] = [
        '<g font-family="Segoe UI,Microsoft YaHei,sans-serif">'
        f'<text x="16" y="488" font-size="15" font-weight="700" fill="{INK}">子弹图 · 本股 vs 对比集中位</text>'
    ]

    for k, (label, asof, getter, lo, hi, unit, dg) in enumerate(specs):
        ry = y + k * row_h
        val = getter(record)
        med = _median(_collect(pool, getter))

        def _x(v: float, lo: float = lo, hi: float = hi) -> float:
            t = 0.0 if hi == lo else (v - lo) / (hi - lo)
            t = max(0.0, min(1.0, t))
            return x0 + bw * t

        # 标签 + 口径时点
        parts.append(
            f'<text x="16" y="{ry + 12}" font-size="12" fill="{INK}">{_esc(label)} '
            f'<tspan fill="{MUTE}" font-size="10">({_esc(asof)})</tspan></text>'
        )
        # 底轨
        parts.append(
            f'<rect x="{x0}" y="{ry + 20}" width="{bw}" height="12" rx="3" fill="{TRACK}"/>'
        )

        if val is None:
            # 降级:灰条 + "—",不伪造 0
            parts.append(
                f'<rect x="{x0}" y="{ry + 20}" width="{bw}" height="12" rx="3" fill="{GRAY}" fill-opacity="0.5">'
                f'<title>{_esc(label)} 数据缺失</title></rect>'
            )
            parts.append(
                f'<text x="{x0 + bw / 2:.0f}" y="{ry + 30}" font-size="11" fill="{MUTE}" text-anchor="middle">{PLACEHOLDER}</text>'
            )
        else:
            bx = _x(val)
            parts.append(
                f'<rect x="{x0}" y="{ry + 20}" width="{bx - x0:.1f}" height="12" rx="3" fill="#4a78c8">'
                f'<title>{_esc(label)} {_fmt(val, dg, unit)}</title></rect>'
            )
            # 本股数值标签
            parts.append(
                f'<text x="{min(bx + 6, x0 + bw):.0f}" y="{ry + 30}" font-size="11" fill="{INK}">'
                f'{_fmt(val, dg, unit)}</text>'
            )
        # 对比集中位刻度线
        if med is not None:
            mx = _x(med)
            parts.append(
                f'<line x1="{mx:.1f}" y1="{ry + 16}" x2="{mx:.1f}" y2="{ry + 36}" stroke="{INK}" stroke-width="2">'
                f'<title>对比集中位 {_fmt(med, dg, unit)}</title></line>'
            )
        med_txt = _fmt(med, dg, unit) if med is not None else PLACEHOLDER
        parts.append(
            f'<text x="{x0 + bw + 8}" y="{ry + 30}" font-size="10" fill="{MUTE}">中位 {med_txt}</text>'
        )

    # 对比集与口径文字(诚实:用 cohort 中位,无全市场行业中位数据)
    parts.append(
        f'<text x="16" y="{y + 3 * row_h + 4}" font-size="11" fill="{MUTE}">'
        f'{caption} · 竖线=中位刻度;无全市场行业中位数据,口径以 cohort 为准。</text>'
    )
    parts.append("</g>")
    return "".join(parts)


def _flags(record: dict[str, Any]) -> str:
    """底部事件旗标行:中性提示 emoji + 事实 + 日期 + a11y <title>。绝不当买入触发器。"""
    flags = record.get("flags") or []
    y = 706
    parts: list[str] = [
        '<g font-family="Segoe UI,Microsoft YaHei,sans-serif">'
        f'<line x1="16" y1="{y - 18}" x2="{W - 16}" y2="{y - 18}" stroke="{GRID}" stroke-width="1"/>'
        f'<text x="16" y="{y}" font-size="12" font-weight="700" fill="{INK}">事件提示旗(中性事实,非买卖信号)</text>'
    ]
    if not flags:
        parts.append(
            f'<text x="16" y="{y + 20}" font-size="12" fill="{MUTE}">无质押/解禁/减持/超预期旗标</text>'
        )
    else:
        ly = y + 20
        for fl in flags:
            if not isinstance(fl, dict):
                continue
            ftype = str(fl.get("type", ""))
            icon = FLAG_ICON.get(ftype, "•")
            fact = str(fl.get("fact", ""))
            date = str(fl.get("date", PLACEHOLDER))
            sev = str(fl.get("severity", "提示"))
            label = f"{ftype} · {fact} · {date}({sev})"
            parts.append(
                f'<text x="16" y="{ly}" font-size="12" fill="{INK}" aria-label="{_esc(label)}">'
                f'{icon} {_esc(ftype)} · {_esc(fact)} · <tspan fill="{MUTE}">{_esc(date)}({_esc(sev)})</tspan>'
                f'<title>{_esc(label)}</title></text>'
            )
            ly += 18
    parts.append("</g>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def render_svg_card(record: dict[str, Any], cohort: Optional[list[dict[str, Any]]] = None) -> str:
    """渲染单股 SVG 卡(纯函数,无 IO)。返回完整自含 ``<svg ...>...</svg>`` 字符串。

    Args:
        record: 单条 card record(schema 见 data/cards/<as_of>.json)。任一数值可能为 None。
        cohort: 对比集(通常传全量 records);None/空时雷达以本股自身为池、子弹图回退仅本股。

    铁律:结论只用离散档(tier emoji + 文字档名 / entry 字母),不把 score 小数当结论;
    数值就近标注口径时点;flags 仅中性提示;缺失值降级占位,不伪造 0、不抛异常。
    """
    pool: list[dict[str, Any]] = list(cohort) if cohort else []
    code = _esc(record.get("ts_code", ""))
    name = _esc(record.get("name", ""))

    body = (
        _header(record)
        + _radar(record, pool)
        + _bullets(record, pool)
        + _flags(record)
    )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" role="img" '
        f'aria-label="单股质地卡 {name} {code}">'
        f"<title>单股质地卡 {name}({code})</title>"
        f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>'
        f'<rect x="6" y="6" width="{W - 12}" height="{H - 12}" rx="14" fill="none" stroke="#ececec" stroke-width="2"/>'
        f"{body}"
        f"</svg>"
    )
