"""D 结构化数据层(纯函数)——一次算、多处用(A/B/C 渲染器共同底座)。

把技术面(factsheet)、接口基本面(fundamentals)、A股有效因子(lit_factors)装配成
单只股票的结构化 record(值与口径分离),并按"规则化人工四层"给定性质地档 tier。

铁律:
- 纯函数,无 IO/不联网,DataFrame 由调用方传入(IO/CLI 在 scripts/cards.py)。
- 一切结论用离散档(tier 🟢🟡🔴⛔ / entry A/B/C),不把小数 score 当结论;每个数值叶子在
  meta 标 {unit, as_of, source}。
- 资金/事件面只做"中性提示旗"(severity 提示/警示),不做买入触发器。
- 数据缺失返回 None 并标注,绝不伪造;真错误(如 KeyError)向上抛,不静默吞。
- quality 复用已入库 lit_factors(年报口径),不重复实现。
"""
from __future__ import annotations

import copy
from typing import Any, cast

import pandas as pd

from ashare_gauntlet.factsheet import daily_tech_facts, entry_rank
from ashare_gauntlet.fundamentals import (
    balance_facts,
    cashflow_facts,
    latest_express,
    latest_forecast,
    latest_quarter,
    peg,
    pledge_ratio,
    receivables_ratio,
    recent_holder_trades,
    st_status,
    upcoming_unlocks,
)
from ashare_gauntlet.lit_factors import ANNUAL, accrual_ratio, net_cash_ratio

# ---- 可调常量(离散档阈值) -------------------------------------------------
ENTRY_GRADE_CUTS: tuple[float, float] = (70.0, 40.0)  # A: >=70, B: >=40, C: <40
DECLINE_SEVERE = -40.0   # ⛔:净利&扣非同比双 <= 此值
GOODWILL_WARN = 50.0     # ⛔:商誉/净资产(%) > 此值
LOW_BASE_NP = 40.0       # 🟡 低基数:净利同比 > 此值
LOW_BASE_REV = 10.0      # 🟡 低基数:而营收同比 < 此值
PLEDGE_WARN = 50.0       # 质押旗标升"警示"的比例(%)
DIFF_FIELDS = ("fundamental.np_yoy", "fundamental.dedt_yoy", "valuation.pe_ttm")


def _num(value: Any) -> float | None:
    """转 float;None / 非数 / NaN -> None。"""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _value_at(df: pd.DataFrame, col: str, end: str) -> float | None:
    """取 ``end_date==end`` 行的 ``col`` 值(多行按 ann_date 取最新);缺失返回 None。

    用于读 balancesheet(总资产/归母净资产)与 fina_indicator(扣非绝对值/扣非增速)
    在指定报告期的值。不静默吞:列存在但值非数 -> None(数据缺失,非错误)。
    """
    if df is None or df.empty or col not in df.columns or "end_date" not in df.columns:
        return None
    d = df.copy()
    d["end_date"] = d["end_date"].astype(str)
    rows = cast(pd.DataFrame, d[d["end_date"] == str(end)])
    if rows.empty:
        return None
    if "ann_date" in rows.columns:
        rows = rows.sort_values("ann_date")
    return _num(rows.iloc[-1][col])


def _entry_grade(score: float) -> str:
    a, b = ENTRY_GRADE_CUTS
    if score >= a:
        return "A"
    if score >= b:
        return "B"
    return "C"


def tier_of(rec: dict[str, Any]) -> dict[str, Any]:
    """规则化人工四层,偏保守,严格优先级 ⛔ -> 🔴 -> 🟡 -> 否则 🟢。

    读取已装配的 fundamental / quality / balance / status / flags;判 🟢 所需输入缺失则
    保守降级 🟡 并标"数据缺失"。返回 {grade, reasons, needs_human}。
    """
    f = rec.get("fundamental") or {}
    q = rec.get("quality") or {}
    b = rec.get("balance") or {}
    s = rec.get("status") or {}
    flags = rec.get("flags") or []

    profitable = f.get("profitable")
    np_yoy, dedt_yoy, rev_yoy = f.get("np_yoy"), f.get("dedt_yoy"), f.get("rev_yoy")
    np_yi, dedt_yi = f.get("np_yi"), f.get("dedt_yi")
    ocf = q.get("op_cashflow_yi")
    gw, na = b.get("goodwill_yi"), b.get("net_assets_yi")
    is_st = bool(s.get("is_st"))
    warn_pledge = any(fl.get("type") == "质押" and fl.get("severity") == "警示" for fl in flags)

    # ⛔ 地雷/重度恶化
    mine: list[str] = []
    if is_st:
        mine.append("当前ST")
    if profitable is False:
        mine.append("亏损")
    if np_yoy is not None and dedt_yoy is not None and np_yoy <= DECLINE_SEVERE and dedt_yoy <= DECLINE_SEVERE:
        mine.append(f"净利&扣非双降≤{DECLINE_SEVERE:.0f}%")
    if na is not None and na <= 0:
        mine.append("资不抵债(净资产≤0)")
    if gw is not None and na is not None and na > 0 and gw / na * 100 > GOODWILL_WARN:
        mine.append(f"商誉占净资产>{GOODWILL_WARN:.0f}%")
    if mine:
        return {"grade": "⛔", "reasons": mine, "needs_human": False}

    # 🔴 题材背离·警示(需配涨幅,人工复核)
    red: list[str] = []
    if np_yi is not None and np_yi > 0 and dedt_yi is not None and dedt_yi < 0:
        red.append("净利为正但扣非为负(扣非背离)")
    if rev_yoy is not None and rev_yoy > 0 and (
        (np_yoy is not None and np_yoy < 0) or (dedt_yoy is not None and dedt_yoy < 0)
    ):
        red.append("增收不增利(营收增但净利/扣非降)")
    if red:
        return {"grade": "🔴", "reasons": red, "needs_human": True}

    # 判 🟢 必需输入的缺失
    needed = {"np_yoy": np_yoy, "dedt_yoy": dedt_yoy, "rev_yoy": rev_yoy, "op_cashflow_yi": ocf}
    missing = [k for k, v in needed.items() if v is None]

    # 🟡 盈利但瑕疵
    yellow: list[str] = []
    needs_human = False
    if np_yi is not None and np_yi > 0 and dedt_yi is None:
        # 净利为正但扣非绝对值缺失:无法核扣非真实正负(背离),不得凭 dedt_yoy 直达 🟢
        yellow.append("扣非绝对值缺失·无法核扣非背离")
        needs_human = True
    if ocf is not None and ocf < 0:
        yellow.append("经营现金流<0(年报)" if profitable is not True else "盈利但经营现金流<0(年报)")
    if np_yoy is not None and rev_yoy is not None and np_yoy > LOW_BASE_NP and rev_yoy < LOW_BASE_REV:
        yellow.append(f"疑低基数(净利+{np_yoy:.0f}% 营收+{rev_yoy:.0f}%)")
        needs_human = True
    if warn_pledge:
        yellow.append("高质押(警示)")
    if missing:
        yellow.append("数据缺失:" + ",".join(missing))
        needs_human = True
    if yellow:
        return {"grade": "🟡", "reasons": yellow, "needs_human": needs_human}

    # 🟢 强且干净:净利&扣非&营收三增 + 经营现金流>0 + 无警示旗标
    if (
        np_yoy is not None and np_yoy > 0
        and dedt_yoy is not None and dedt_yoy > 0
        and rev_yoy is not None and rev_yoy > 0
        and ocf is not None and ocf > 0
    ):
        return {"grade": "🟢", "reasons": ["净利&扣非&营收三增·现金流>0·无警示"], "needs_human": False}

    # 兜底:未达 🟢 全条件但未触发上面任一明确档(如净利/营收/扣非下滑但未及阈值)-> 保守 🟡。
    # 明列缺口(诚实面板要可读),而非泛泛"未满足全条件"。
    fail: list[str] = []
    if not (np_yoy is not None and np_yoy > 0):
        fail.append("净利未增")
    if not (dedt_yoy is not None and dedt_yoy > 0):
        fail.append("扣非未增")
    if not (rev_yoy is not None and rev_yoy > 0):
        fail.append("营收未增")
    if not (ocf is not None and ocf > 0):
        fail.append("现金流≤0")
    return {"grade": "🟡", "reasons": ["未达🟢(三增/现金流):" + "·".join(fail)], "needs_human": True}


def _build_flags(fund_tables: dict[str, pd.DataFrame], as_of: str, q_end: str) -> list[dict[str, Any]]:
    """事件/资金面"中性提示旗"。severity 仅 提示/警示;颜色之外自带文字事实+日期+来源。"""
    flags: list[dict[str, Any]] = []

    pr = pledge_ratio(fund_tables["pledge_stat"], as_of)
    if pr is not None and pr > 0:
        flags.append({
            "type": "质押",
            "severity": "警示" if pr >= PLEDGE_WARN else "提示",
            "fact": f"控股股东体系质押{pr:.1f}%",
            "value": pr,  # 结构化数值:渲染器读这个,勿反解析 fact 字符串
            "date": as_of,
            "source": "pledge_stat接口",
        })

    unlocks = upcoming_unlocks(fund_tables["share_float"], as_of, 180) if as_of else []
    if unlocks:
        u = unlocks[0]
        ratio = u.get("float_ratio")
        fact = f"解禁 占流通{ratio:.1f}%" if ratio is not None else "限售解禁"
        flags.append({"type": "解禁", "severity": "提示", "fact": fact, "date": u.get("float_date"), "source": "share_float接口"})

    trades = recent_holder_trades(fund_tables["stk_holdertrade"], as_of, 365) if as_of else []
    reds = [t for t in trades if t.get("direction") == "减持"]
    if reds:
        flags.append({
            "type": "减持", "severity": "提示",
            "fact": f"近一年减持{len(reds)}笔", "date": reds[0].get("ann_date"), "source": "stk_holdertrade接口",
        })

    fc = latest_forecast(fund_tables["forecast"])
    ex = latest_express(fund_tables["express"])
    pmin = fc.get("p_change_min")
    eyoy = ex.get("yoy_net_profit_pct")
    if fc and q_end and str(fc.get("end_date", "")) > str(q_end) and pmin is not None and pmin >= 50:
        flags.append({
            "type": "超预期", "severity": "提示",
            "fact": f"预告{fc.get('type', '')} 净利+{pmin:.0f}%起", "date": fc.get("end_date"), "source": "forecast接口",
        })
    elif ex and q_end and str(ex.get("end_date", "")) > str(q_end) and eyoy is not None and eyoy >= 50:
        flags.append({
            "type": "超预期", "severity": "提示",
            "fact": f"快报净利同比+{eyoy:.0f}%", "date": ex.get("end_date"), "source": "express接口",
        })
    return flags


def build_record(
    ts_code: str,
    *,
    name: str,
    industry: str,
    as_of: str,
    daily_sub: pd.DataFrame,
    adj_sub: pd.DataFrame,
    mr: dict[int, pd.Series],
    fund_tables: dict[str, pd.DataFrame],
    db_row: Any,
) -> dict[str, Any]:
    """装配单只股票的完整 record。

    daily_sub/adj_sub:该 code 的日线/复权因子行;mr:market_returns 全市场 dict;
    fund_tables:10 张接口表(键缺失即 KeyError,响亮失败);db_row:daily_basic 单行
    (Series/dict,含 pe_ttm/pb/total_mv)或 None。
    """
    income = fund_tables["income"]
    fina = fund_tables["fina_indicator"]
    bs = fund_tables["balancesheet"]
    cashflow = fund_tables["cashflow"]

    # 技术面(前复权,描述性)
    tech = daily_tech_facts(ts_code, daily_sub, adj_sub, mr)
    technical: dict[str, Any] = {
        "close": tech.get("close"),
        "dist60": tech.get("dist_60d_high_pct"),
        "pct20": tech.get("pct20"),
        "trend": tech.get("trend"),
        "rsi": tech.get("rsi"),
        "ret20": tech.get("ret20_pct"),
        "vol_ratio": tech.get("vol_ratio"),
    }

    # 入场纪律分 -> 离散档
    score, tag = entry_rank(tech)
    entry: dict[str, Any] = {"grade": _entry_grade(score), "score": round(score, 1), "tag": tag}

    # 基本面(最新季)
    q = latest_quarter(income, fina)
    end = q.get("end_date")
    dedt_yoy = _value_at(fina, "dt_netprofit_yoy", end) if end else None
    dedt_abs = _value_at(fina, "profit_dedt", end) if end else None
    fundamental: dict[str, Any] = {
        "end_date": end,
        "rev_yi": q.get("revenue_yi"),
        "rev_yoy": q.get("revenue_yoy_pct"),
        "np_yi": q.get("net_profit_yi"),
        "np_yoy": q.get("net_profit_yoy_pct"),
        "dedt_yoy": dedt_yoy,
        "dedt_yi": dedt_abs / 1e8 if dedt_abs is not None else None,
        "gross_margin": q.get("gross_margin_pct"),
        "profitable": q.get("profitable"),
    }

    # 估值
    pe = _num(db_row.get("pe_ttm")) if db_row is not None else None
    pb = _num(db_row.get("pb")) if db_row is not None else None
    mv = _num(db_row.get("total_mv")) if db_row is not None else None
    valuation: dict[str, Any] = {
        "pe_ttm": pe,
        "pb": pb,
        "peg": peg(pe, fundamental["np_yoy"]),
        "mv_yi": mv / 1e4 if mv is not None else None,
    }

    # 资产负债(取 balance_facts 所用报告期的归母净资产/总资产)
    bf = balance_facts(bs)
    bend = bf.get("end_date")
    na_raw = _value_at(bs, "total_hldr_eqy_exc_min_int", bend) if bend else None
    ta_raw = _value_at(bs, "total_assets", bend) if bend else None
    balance: dict[str, Any] = {
        "accounts_receiv_yi": bf.get("accounts_receiv_yi"),
        "goodwill_yi": bf.get("goodwill_yi"),
        "money_cap_yi": bf.get("money_cap_yi"),
        "net_assets_yi": na_raw / 1e8 if na_raw is not None else None,
        "total_assets_yi": ta_raw / 1e8 if ta_raw is not None else None,
        "recv_to_annual_net_pct": receivables_ratio(bs, income),
    }

    # 现金质量(年报口径,复用 lit_factors,不重复实现)
    quality: dict[str, Any] = {
        "op_cashflow_yi": cashflow_facts(cashflow, ANNUAL).get("op_cashflow_yi"),
        "net_cash_ratio": net_cash_ratio(income, cashflow),
        "accrual": accrual_ratio(income, cashflow, bs),
    }

    # ST 状态
    st = st_status(fund_tables["namechange"]) or {}
    status: dict[str, Any] = {
        "is_st": bool(st.get("is_st")),
        "ever_st": bool(st.get("ever_st")),
        "current_name": st.get("current_name") or name,
    }

    flags = _build_flags(fund_tables, as_of, str(end) if end else "")

    # 口径标注:每个非空数值叶子 -> {unit, as_of, source}
    meta: dict[str, dict[str, str]] = {}

    def _m(path: str, val: Any, unit: str, asof: Any, source: str) -> None:
        if val is not None:
            meta[path] = {"unit": unit, "as_of": str(asof), "source": source}

    qend = str(end) if end else ""
    bstr = str(bend) if bend else ""
    _m("technical.close", technical["close"], "元", as_of, "daily前复权")
    _m("technical.dist60", technical["dist60"], "%", as_of, "距60日高(前复权)")
    _m("technical.pct20", technical["pct20"], "百分位", as_of, "全市场20日收益排名")
    _m("technical.rsi", technical["rsi"], "0-100", as_of, "RSI14(前复权)")
    _m("technical.ret20", technical["ret20"], "%", as_of, "近20日收益(前复权)")
    _m("technical.vol_ratio", technical["vol_ratio"], "倍", as_of, "量/20日均量(前复权)")
    _m("valuation.pe_ttm", pe, "倍", as_of, "daily_basic接口")
    _m("valuation.pb", pb, "倍", as_of, "daily_basic接口")
    _m("valuation.peg", valuation["peg"], "倍", as_of, "PE_TTM/净利同比增速")
    _m("valuation.mv_yi", valuation["mv_yi"], "亿元", as_of, "daily_basic.total_mv")
    _m("fundamental.rev_yi", fundamental["rev_yi"], "亿元", qend, "income.total_revenue")
    _m("fundamental.rev_yoy", fundamental["rev_yoy"], "%", qend, "fina_indicator.or_yoy")
    _m("fundamental.np_yi", fundamental["np_yi"], "亿元", qend, "income.n_income_attr_p")
    _m("fundamental.np_yoy", fundamental["np_yoy"], "%", qend, "fina_indicator.netprofit_yoy")
    _m("fundamental.dedt_yoy", fundamental["dedt_yoy"], "%", qend, "fina_indicator.dt_netprofit_yoy")
    _m("fundamental.dedt_yi", fundamental["dedt_yi"], "亿元", qend, "fina_indicator.profit_dedt")
    _m("fundamental.gross_margin", fundamental["gross_margin"], "%", qend, "fina_indicator.grossprofit_margin")
    _m("balance.accounts_receiv_yi", balance["accounts_receiv_yi"], "亿元", bstr, "balancesheet.accounts_receiv")
    _m("balance.goodwill_yi", balance["goodwill_yi"], "亿元", bstr, "balancesheet.goodwill")
    _m("balance.money_cap_yi", balance["money_cap_yi"], "亿元", bstr, "balancesheet.money_cap")
    _m("balance.net_assets_yi", balance["net_assets_yi"], "亿元", bstr, "balancesheet.total_hldr_eqy_exc_min_int")
    _m("balance.total_assets_yi", balance["total_assets_yi"], "亿元", bstr, "balancesheet.total_assets")
    _m("balance.recv_to_annual_net_pct", balance["recv_to_annual_net_pct"], "%", qend, "应收/最近年报归母净利")
    _m("quality.op_cashflow_yi", quality["op_cashflow_yi"], "亿元", ANNUAL, "cashflow.n_cashflow_act(年报)")
    _m("quality.net_cash_ratio", quality["net_cash_ratio"], "倍", ANNUAL, "经营现金流/归母净利(年报)")
    _m("quality.accrual", quality["accrual"], "倍", ANNUAL, "(归母净利-经营现金流)/总资产(年报)")

    record: dict[str, Any] = {
        "ts_code": ts_code,
        "name": name,
        "industry": industry,
        "as_of": as_of,
        "entry": entry,
        "technical": technical,
        "valuation": valuation,
        "fundamental": fundamental,
        "balance": balance,
        "quality": quality,
        "status": status,
        "flags": flags,
        "meta": meta,
    }
    record["tier"] = tier_of(record)
    record["factcheck"] = None
    return record


def _get(d: dict[str, Any], path: str) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def diff_records(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """同一 ts_code 跨交易日变化摘要:tier/entry 档变动 + 关键值变化 + 旗标增减。"""
    old_tier, new_tier = _get(old, "tier.grade"), _get(new, "tier.grade")
    old_entry, new_entry = _get(old, "entry.grade"), _get(new, "entry.grade")

    field_changes: list[dict[str, Any]] = []
    for path in DIFF_FIELDS:
        ov, nv = _get(old, path), _get(new, path)
        if ov is None and nv is None:
            continue
        # 回读的历史/外部 cards JSON 可能塞了非数值字符串(如 "N/A"):用 _num 各转 float|None,
        # 不抛 ValueError。两侧都能转成 float -> 比数值;否则(任一转不出)按原值是否相等判变化。
        fo, fn = _num(ov), _num(nv)
        if fo is not None and fn is not None:
            changed = abs(fo - fn) > 1e-6
        else:
            changed = ov != nv
        if changed:
            field_changes.append({"path": path, "old": ov, "new": nv})

    old_ft = {fl.get("type") for fl in (old.get("flags") or []) if fl.get("type")}
    new_ft = {fl.get("type") for fl in (new.get("flags") or []) if fl.get("type")}
    return {
        "ts_code": new.get("ts_code"),
        "tier_change": [old_tier, new_tier] if old_tier != new_tier else None,
        "entry_change": [old_entry, new_entry] if old_entry != new_entry else None,
        "field_changes": field_changes,
        "new_flags": sorted(new_ft - old_ft),
        "dropped_flags": sorted(old_ft - new_ft),
    }


def merge_factcheck(record: dict[str, Any], fc: dict[str, Any] | None) -> dict[str, Any]:
    """把独立 factcheck 结论并入 record(铁律:绝不覆盖任何接口口径数字)。

    返回 record 深拷贝,仅设置 ``factcheck`` 键;fc 即便带净利数字,也只进
    ``factcheck.q1_net_profit_yi``(独立核实值,与 fundamental.np_yi 并列展示);不重算 tier。
    """
    out = copy.deepcopy(record)
    if fc is None:
        out["factcheck"] = None
        return out
    out["factcheck"] = {
        "confirmed": fc.get("confirmed"),
        "q1_net_profit_yi": fc.get("q1_net_profit_yi"),
        "disputes": fc.get("disputes") or [],
        "news": fc.get("news") or [],
        "verified_at": fc.get("verified_at"),
    }
    return out
