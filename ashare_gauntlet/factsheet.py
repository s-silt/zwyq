"""Factual-layer indicators + single-stock factsheet.

Everything here is *descriptive* — computed from real closes — never a prediction
or a trade call. EMA/RSI/Bollinger are the tested versions of the numbers a TA
report quotes; we compute them ourselves rather than trust an unverified source.
"""

import math
from collections.abc import Sequence
from typing import cast

import pandas as pd


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
        if 5 in market_rets:
            fs["pct5"] = float((market_rets[5] < ret5).mean() * 100)
        if 20 in market_rets:
            fs["pct20"] = float((market_rets[20] < ret20).mean() * 100)
    return fs
