"""EOD 缓存挂钟新鲜度(纯函数,只读):把"缓存悄悄没更新"从静默降级变成显式提醒。

补的缺口(深读 R3 头号风险):读端 as_of 直接取本地 daily 分区的 max,**只比"财务 vs
as_of",从不比"as_of vs 今天"**。若 refresh 因 token 耗尽/网络中断没补上最新日,
factor_rank/buy_list 会用上周价格算出一份**看起来完全正常**的决策快照——实盘风控里
最危险的静默降级。

判定口径(诚实边界):本地 daily 分区只记录到缓存最新日,**之后是否开市本地无从得知**
(交易日历要联网 trade_cal)。因此本模块用**工作日(周一至周五)**计数做保守代理:
- 0 个工作日缺口 → FRESH
- 1~2 个 → SUSPECT(可能是节假日,也可能是刷新漏跑;要人确认,不当作正常)
- ≥3 个 → STALE(A股连续休市上限≈国庆/春节 ~9 自然日≈5~7 工作日,但连挂 3 个工作日
  仍未更新已足够可疑,宁可误报也不静默)
误报方向是"多提醒",不是"少提醒"——未知不解释为安全。

import 阶段无 I/O 副作用。
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

SUSPECT_WEEKDAYS = 1   # ≥ 此工作日缺口 → SUSPECT
STALE_WEEKDAYS = 3     # ≥ 此工作日缺口 → STALE


class FreshnessError(ValueError):
    """输入日期非法——不猜不静默。"""


def _parse8(value: str, field: str) -> date:
    if not isinstance(value, str) or len(value) != 8 or not value.isdigit():
        raise FreshnessError(f"{field} 必须是 8 位 YYYYMMDD,得到 {value!r}")
    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except ValueError as exc:
        raise FreshnessError(f"{field} 不是真实日期: {value!r}") from exc


def weekdays_between(start: str, end: str) -> int:
    """(start, end] 区间内的工作日数(周一至周五;不含 start 当日,含 end 当日)。

    end 早于 start → 负向不定义,抛错(缓存日期晚于今天=系统时钟或缓存有问题)。
    """
    d0 = _parse8(start, "start")
    d1 = _parse8(end, "end")
    if d1 < d0:
        raise FreshnessError(f"end {end} 早于 start {start}——缓存日期晚于今天?核对系统时钟")
    count = 0
    cur = d0 + timedelta(days=1)
    while cur <= d1:
        if cur.weekday() < 5:
            count += 1
        cur += timedelta(days=1)
    return count


def classify_cache_freshness(latest_cached: str | None,
                             today: str) -> dict[str, object]:
    """判定本地 EOD 缓存相对挂钟的新鲜度。

    返回 {status, weekday_gap, latest_cached, today, detail}。
    status ∈ {MISSING, FRESH, SUSPECT, STALE}——MISSING(无缓存)绝不当 FRESH。
    """
    if not latest_cached:
        return {"status": "MISSING", "weekday_gap": None, "latest_cached": None,
                "today": today,
                "detail": "本地 daily 缓存为空——先跑 backfill/refresh,不可在无行情上决策"}
    gap = weekdays_between(latest_cached, today)
    if gap >= STALE_WEEKDAYS:
        status = "STALE"
        detail = (f"缓存最新日 {latest_cached} 距今 {gap} 个工作日未更新"
                  "——决策可能建在旧价格上;先跑 refresh/backfill 再看决策")
    elif gap >= SUSPECT_WEEKDAYS:
        status = "SUSPECT"
        detail = (f"缓存最新日 {latest_cached} 距今 {gap} 个工作日"
                  "——可能是节假日,也可能是刷新漏跑;请确认(工具不替你判定)")
    else:
        status = "FRESH"
        detail = f"缓存最新日 {latest_cached} 与今日无工作日缺口"
    return {"status": status, "weekday_gap": gap, "latest_cached": latest_cached,
            "today": today, "detail": detail}
