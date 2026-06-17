"""Factual-layer indicators + single-stock factsheet.

Everything here is *descriptive* — computed from real closes — never a prediction
or a trade call. EMA/RSI/Bollinger are the tested versions of the numbers a TA
report quotes; we compute them ourselves rather than trust an unverified source.
"""

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
