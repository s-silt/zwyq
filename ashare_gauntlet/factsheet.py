"""Factual-layer indicators + single-stock factsheet.

Everything here is *descriptive* — computed from real closes — never a prediction
or a trade call. EMA/RSI/Bollinger are the tested versions of the numbers a TA
report quotes; we compute them ourselves rather than trust an unverified source.
"""

import math
from collections.abc import Sequence
from typing import Any, cast

import pandas as pd

# 沪深主板代码集 = 600/601/603/605(沪)+ 000/001/002/003(深);
# 不含创业板(300/301)、科创板(688/689)、北交所(8xx)、B股。横截面分位的对比集
# 必须只对主板 cohort 计算(契约C2),否则被科创/创业的高波动股污染分位口径。
_MAIN_BOARD_PREFIXES = ("600", "601", "603", "605", "000", "001", "002", "003")


def _num(value: Any) -> float | None:
    """转 float;None / 非数 / NaN -> None(缺失即缺失,不伪造默认值)。"""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def is_main_board(ts_code: str) -> bool:
    """ts_code 是否属沪深主板(对比集纯净度:横截面分位只在主板内排名)。"""
    code = str(ts_code).partition(".")[0]
    return code[:3] in _MAIN_BOARD_PREFIXES


def _main_board_cohort(returns: pd.Series) -> pd.Series:
    """把一条 ``{ts_code: return}`` Series 过滤到沪深主板 cohort。"""
    mask = [is_main_board(c) for c in returns.index]
    return cast(pd.Series, returns[mask])


def ema(close: pd.Series, span: int) -> pd.Series:
    """Exponential moving average (adjust=False, the conventional TA form)."""
    return cast(pd.Series, close.ewm(span=span, adjust=False).mean())


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI. 100 = only gains over the window, 0 = only losses."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False).mean()
    return cast(pd.Series, 100 - 100 / (1 + avg_gain / avg_loss))


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> tuple[float, float, float]:
    """Latest Bollinger band triple (lower, mid, upper) over the last ``n`` closes;
    mid is the SMA, bands are mid ± k·(sample std)."""
    window = close.iloc[-n:]
    mid = float(window.mean())
    sd = float(window.std())  # ddof=1
    return mid - k * sd, mid, mid + k * sd


# LANDMINE: tushare's ``moneyflow_hsgt`` REUSES the columns hgt/sgt/north_money
# for two different quantities. BEFORE this date they are daily NET inflow (百万元,
# signed); ON/AFTER it the exchanges stopped disclosing net flow and the same
# columns silently became gross TURNOVER (成交额). Same names, ~50x scale jump,
# never negative. Reading north_money as net flow post-cutoff is a catastrophic
# ~50x fabrication — every reader of this field must respect the cutoff.
NORTH_FLOW_SEMANTICS_CUTOFF = "20240819"


def north_turnover(moneyflow_hsgt: pd.DataFrame, as_of: str) -> dict[str, float]:
    """北向(沪股通+深股通)当日**总成交额**(亿元) from a cached ``moneyflow_hsgt`` pull.

    Returns TURNOVER only, and refuses dates before
    :data:`NORTH_FLOW_SEMANTICS_CUTOFF` so the pre-cutoff NET-inflow semantics can
    never be misread as turnover (see the landmine note above). ``hgt``/``sgt``/
    ``north_money`` arrive as strings in 百万元; converted to 亿元 (``/100``).
    Cross-verified: 2026-06-04 ``north_money`` == 东财「共成交 3651.69 亿」exactly.
    """
    if as_of < NORTH_FLOW_SEMANTICS_CUTOFF:
        raise ValueError(
            f"north_money 在 {as_of} 是'净流入'语义(< {NORTH_FLOW_SEMANTICS_CUTOFF}),"
            "非成交额;本函数只返回成交额,拒绝换义前的日期以防 ~50x 误报"
        )
    row = moneyflow_hsgt[moneyflow_hsgt["trade_date"] == as_of]
    if row.empty:
        raise KeyError(f"moneyflow_hsgt 无 {as_of} 数据")
    r = row.iloc[0]
    return {
        "hgt_yi": float(r["hgt"]) / 100.0,
        "sgt_yi": float(r["sgt"]) / 100.0,
        "total_yi": float(r["north_money"]) / 100.0,
    }


def north_flow_disclosure() -> str:
    """The standing, honest fact about northbound daily-flow availability.

    Daily northbound NET inflow disclosure was discontinued on 2024-08-19 (per-
    stock 北向持股 moved to quarterly); only the aggregate turnover + the top-10
    active names remain daily. Returned as a one-liner so every report self-
    carries the limit instead of relying on a manual footnote — the factual
    layer must never quote a daily net figure that no longer officially exists.
    """
    return (
        "北向净流入: 自 2024-08-19 制度性停披露(个股持股改季度), 无官方日频净额; "
        "日频仅总成交额+十大成交活跃股 —— 事实层不提供北向净流入(非遗漏,是数据缺口)"
    )


def build_factsheet(
    ts_code: str,
    daily_all: pd.DataFrame,
    adj_all: pd.DataFrame,
    hk_all: pd.DataFrame | None = None,
    ema_short: int = 5,
    ema_long: int = 20,
    rsi_n: int = 14,
    boll_n: int = 20,
) -> dict[str, object]:
    """Assemble the factual layer for one A-share from cached full-market data.

    Price/indicators use the FORWARD-adjusted (前复权) close so the levels sit on
    the same scale as the current quote (latest 前复权 == raw last price) while
    splits/dividends still don't create phantom moves — the right basis for a
    descriptive level display (the gauntlet uses 后复权 for returns). Northbound
    holding (北向持股占比), if present, is a *real* data point — never invented.
    """
    one = daily_all[daily_all["ts_code"] == ts_code].merge(
        adj_all[["ts_code", "trade_date", "adj_factor"]],
        on=["ts_code", "trade_date"],
        how="left",
    )
    one = one.sort_values("trade_date")
    adj = one["adj_factor"].to_numpy()
    raw = one["close"].to_numpy()
    qfq = raw * adj / adj[-1]  # 前复权: latest == raw quote, history dividend-adjusted
    close = pd.Series(qfq, index=one["trade_date"].to_numpy())

    lower, mid, upper = bollinger(close, boll_n)
    fs: dict[str, object] = {
        "ts_code": ts_code,
        "as_of": str(one["trade_date"].iloc[-1]),
        "close_raw": float(raw[-1]),
        "pct_chg_1d_pct": float(raw[-1] / raw[-2] - 1) * 100 if len(raw) > 1 else float("nan"),
        "amount": float(one["amount"].iloc[-1]) if "amount" in one.columns else None,
        "high_20d": float(close.iloc[-20:].max()),
        "low_20d": float(close.iloc[-20:].min()),
        "high_60d": float(close.iloc[-60:].max()),
        "low_60d": float(close.iloc[-60:].min()),
        "ema_short": float(ema(close, ema_short).iloc[-1]),
        "ema_long": float(ema(close, ema_long).iloc[-1]),
        "rsi": float(rsi(close, rsi_n).iloc[-1]),
        "boll": (lower, mid, upper),
    }

    if hk_all is not None:
        h = cast(pd.DataFrame, hk_all[hk_all["ts_code"] == ts_code].copy())
        if len(h):
            h["ratio"] = pd.to_numeric(h["ratio"], errors="coerce")
            h = h.sort_values("trade_date")
            ratios = h["ratio"].to_numpy()
            fs["north_ratio"] = float(ratios[-1])
            fs["north_ratio_chg_5"] = float(ratios[-1] - ratios[-6]) if len(ratios) > 5 else float("nan")
    return fs


def market_returns(
    daily_all: pd.DataFrame,
    adj_all: pd.DataFrame,
    horizons: Sequence[int] = (5, 20),
) -> dict[int, pd.Series]:
    """Latest h-session back-adjusted return for every stock, per horizon.

    Used to rank one stock cross-sectionally against the whole market (returns
    are scale-invariant, so 后复权 is fine here)."""
    piv = daily_all.merge(
        adj_all[["ts_code", "trade_date", "adj_factor"]], on=["ts_code", "trade_date"], how="left"
    )
    piv["hfq"] = piv["close"] * piv["adj_factor"]
    wide = piv.pivot_table(index="trade_date", columns="ts_code", values="hfq").sort_index()
    out: dict[int, pd.Series] = {}
    for h in horizons:
        if len(wide) > h:
            out[h] = (wide.iloc[-1] / wide.iloc[-(h + 1)] - 1).dropna()
    return out


def daily_tech_facts(
    ts_code: str,
    daily_all: pd.DataFrame,
    adj_all: pd.DataFrame,
    market_rets: dict[int, pd.Series] | None = None,
    ema_short: int = 5,
    ema_long: int = 20,
    rsi_n: int = 14,
) -> dict[str, object]:
    """Richer per-stock factual analysis: trend label, momentum + direction,
    Bollinger position, distance from the 60-session high, recent returns and —
    if ``market_rets`` is given — the cross-sectional percentile vs the whole
    market. Descriptive only.
    """
    one = daily_all[daily_all["ts_code"] == ts_code].merge(
        adj_all[["ts_code", "trade_date", "adj_factor"]], on=["ts_code", "trade_date"], how="left"
    )
    one = one.sort_values("trade_date")
    adj = one["adj_factor"].to_numpy()
    raw = one["close"].to_numpy()
    q = pd.Series(raw * adj / adj[-1], index=one["trade_date"].to_numpy())  # 前复权

    cur = float(q.iloc[-1])
    e_s = float(ema(q, ema_short).iloc[-1])
    e_l = float(ema(q, ema_long).iloc[-1])
    rsi_series = rsi(q, rsi_n)
    lo, mid, up = bollinger(q, 20)
    high60 = float(q.iloc[-60:].max())
    amt = one["amount"].to_numpy()

    def ret(h: int) -> float:
        return float(q.iloc[-1] / q.iloc[-(h + 1)] - 1) if len(q) > h else math.nan

    ret5, ret20 = ret(5), ret(20)
    trend = "多头" if cur > e_s > e_l else ("空头" if cur < e_s < e_l else "纠缠")
    fs: dict[str, object] = {
        "ts_code": ts_code,
        "as_of": str(one["trade_date"].iloc[-1]),
        "close": cur,
        "trend": trend,
        "ema_short": e_s,
        "ema_long": e_l,
        "rsi": float(rsi_series.iloc[-1]),
        "rsi_dir": "↑" if rsi_series.iloc[-1] >= rsi_series.iloc[-6] else "↓",
        "boll": (lo, mid, up),
        "dist_60d_high_pct": (cur / high60 - 1) * 100,
        "ret5_pct": ret5 * 100,
        "ret20_pct": ret20 * 100,
        "vol_ratio": float(amt[-1] / amt[-20:].mean()),
    }
    if market_rets:
        # 横截面分位只对沪深主板 cohort 计算(契约C2:对比集纯净)。cohort 为空
        # (无主板可比股)→ 分位无依据,不写该键(下游按缺失/None 处理,不伪造)。
        if 5 in market_rets:
            cohort5 = _main_board_cohort(market_rets[5])
            if not cohort5.empty:
                fs["pct5"] = float((cohort5 < ret5).mean() * 100)
        if 20 in market_rets:
            cohort20 = _main_board_cohort(market_rets[20])
            if not cohort20.empty:
                fs["pct20"] = float((cohort20 < ret20).mean() * 100)
    return fs


def entry_rank(facts: dict[str, Any]) -> tuple[float | None, str]:
    """Transparent 'buying-discipline' score + tag for one stock — NOT a return
    prediction (momentum & reversal both tested NO_GO in this market/window).

    The stated rule, organizing by a disciplined entry lens:
      base   = 20-day cross-sectional relative strength (pct20, 0-100)
      −60    if 空头 (downtrend) — don't catch a falling knife
      −20    if RSI > 70 — don't chase an overbought move
      +15    if 多头 and price within −5%..+3% of EMA20 — a pullback to support
             inside an uptrend
    Higher = better fits the lens. Tag summarizes the entry condition.

    口径边界(methodology §11,2026-07 定谳):"回踩=更好买点"已被实证否决——D10 池内挑
    破均线/近期回调的"更弱的"是负筛选(R_DMA20 −0.39% t−2.7、R_RET5 −0.34% t−2.9,
    entry_readiness 子集 21 日净 −0.67%/期 t−2.9)。本函数只作技术面展示透镜保留,
    生产 BUY 恒 entry_model_version='research-only',不得用本分做选股、择时或资金分配。

    缺关键技术输入(横截面分位 pct20 / 现价 close)时**不补 0/50/close 默认**
    伪造一个分(契约C2,数据源纯净):返回 ``(None, '数据缺失·无入场分')``。
    """
    rs = _num(facts.get("pct20"))
    close = _num(facts.get("close"))
    if rs is None or close is None:
        missing = [n for n, v in (("近20分位", rs), ("现价", close)) if v is None]
        return None, "数据缺失·无入场分(" + "/".join(missing) + ")"

    trend = facts.get("trend")
    rsi_val = float(facts.get("rsi", 50.0))
    ema_long = _num(facts.get("ema_long")) or close
    dist_ema_pct = (close / ema_long - 1) * 100 if ema_long else 0.0

    score = rs
    tags: list[str] = []
    if trend == "空头":
        score -= 60
        tags.append("下跌趋势·勿接")
    if rsi_val > 70:
        score -= 20
        tags.append("超买·勿追")
    if trend == "多头" and -5.0 <= dist_ema_pct <= 3.0:
        score += 15
        tags.append("回踩支撑·趋势内")
    if not tags:
        tags.append("趋势内" if trend == "多头" else "中性")
    return score, "·".join(tags)
