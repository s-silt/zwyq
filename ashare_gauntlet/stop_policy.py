"""止损政策一致性检查(纯函数,只读):核对持仓 stop 是否落在双仓制政策带内。

补的缺口(深读 R5 头号风险):
- `trade_record --buy` 落账后 position 的 stop 恒为 None,盘中哨兵 alert_level 对
  stop=None 的仓**完全跳过 BREACH/NEAR**——建仓到人工补止损之间是"无止损警报裸窗口"。
- stop 是人工填的绝对价,写反方向/少个 0 没有任何守卫。

边界(严守"只算不判/终判永久人工"):本模块**只 surface 不改**——绝不自动写 stop、
绝不替用户决定卖出。政策带的两条腿口径不同,如实说明(对抗复核 P2:此处曾笼统写
"全部来自 entry_check.BUCKETS、不新造常数",与代码不符):
- **短线**引用 entry_check.BUCKETS["short"]["stop_frac"](硬止损 -7% 单点)+ tolerance
  容差(人工填价误差,非新阈值);
- **长线**用本模块 LONG_STOP_RANGE = 制度区间 -12%~-15% 的端点。该区间在 entry_check
  里只以散文注释存在、无可导入表示,故此处是它唯一的可执行副本——**entry_check 若改
  区间,本常数不会自动跟随**(登记为后续:把区间提成 BUCKETS["long"]["stop_range"])。

import 阶段无 I/O 副作用。
"""
from __future__ import annotations

import math
from typing import Any, Iterable

# 制度参数唯一来源:双仓制 stop_frac(短线 0.07 / 长线 0.13)。此处**引用**不复制,
# 避免与 entry_check 漂移(scripts 层导入是既有惯例:trade_record 亦 import 其常量)。
from scripts.entry_check import BUCKETS

# bucket 归一复用 account_state 的**唯一权威实现**(此前本模块自带第二份别名表,
# 是跨层双轨制的一部分;归一必须全仓一处,否则各层对同一仓的判定会漂移)
from ashare_gauntlet.account_state import normalize_bucket as _normalize_bucket
DEFAULT_TOLERANCE = 0.02   # 短线硬止损的人工填价容差(绝对百分点),非评分阈值

# 长线止损是**制度区间**而非单点:-12% ~ -15%(README「双仓制止损宽度」/
# entry_check「制度区间-12%~-15%取中值」)。此前用"中值 -13% ± 2pp"生成带,
# 算术上是 -11%~-15%,把制度区间外的 -11% 判成 OK(跨层审计),故改为直接
# 引用区间端点。短线是"硬止损 -7%"单点,给 ±tolerance 容差覆盖人工填价误差。
LONG_STOP_RANGE = (0.12, 0.15)   # (最紧, 最松) 距成本跌幅


class StopPolicyError(ValueError):
    """输入结构非法——不静默跳过(未知不解释为安全)。"""


def _finite_positive(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number > 0 else None


def policy_band(bucket: Any, cost: float,
                tolerance: float = DEFAULT_TOLERANCE) -> tuple[float, float] | None:
    """给定仓别与成本价,返回 stop 的政策带 (下沿, 上沿);仓别未知/制度前 → None。

    - 长线:直接用制度区间端点 → [cost×0.85, cost×0.88](= 距成本 −15% ~ −12%)。
      不用"中值±容差",否则会把制度区间外的 −11% 判成 OK(跨层审计)。
    - 短线:硬止损 −7% 单点 + ±tolerance 人工填价容差 → 默认 [cost×0.91, cost×0.95]。
    """
    key = _normalize_bucket(bucket)
    if key == "long":
        tight, loose = LONG_STOP_RANGE
        return round(cost * (1.0 - loose), 4), round(cost * (1.0 - tight), 4)
    if key == "short":
        frac = float(BUCKETS["short"]["stop_frac"])
        return round(cost * (1.0 - frac - tolerance), 4), round(cost * (1.0 - frac + tolerance), 4)
    return None   # legacy(制度前)/未知:不套新规,由调用方标 UNKNOWN


def check_position(position: dict[str, Any],
                   tolerance: float = DEFAULT_TOLERANCE) -> dict[str, Any]:
    """单仓止损一致性判定。返回 {ts_code, status, detail, stop, cost, band, bucket}。

    status:
      MISSING_STOP —— stop 未填(哨兵对该仓不做 BREACH/NEAR,裸窗口)
      ABOVE_COST   —— stop ≥ 成本价(方向可疑:止损不应在成本上方,疑似写反)
      OUT_OF_BAND  —— stop 在政策带外(偏紧/偏松,或少写一位)
      OK           —— 落在政策带内
      UNKNOWN      —— 仓别或成本缺失/非法,无法判定(不当作 OK)
    """
    if not isinstance(position, dict):
        raise StopPolicyError("position 必须是对象")
    code = str(position.get("ts_code") or "")
    bucket = position.get("bucket")
    cost = _finite_positive(position.get("cost"))
    stop_raw = position.get("stop")
    stop = _finite_positive(stop_raw)
    base = {"ts_code": code, "bucket": bucket, "cost": cost, "stop": stop, "band": None}

    if cost is None:
        return {**base, "status": "UNKNOWN", "detail": "成本价缺失/非正,无法判定止损带"}
    band = policy_band(bucket, cost, tolerance)
    base["band"] = band
    if stop is None:
        detail = "stop 未填——盘中哨兵对该仓跳过 BREACH/NEAR(无止损警报);人工补止损价"
        if stop_raw is not None:
            detail = f"stop={stop_raw!r} 非正有限数——同样无警报保护;人工修正"
        return {**base, "status": "MISSING_STOP", "detail": detail}
    if stop >= cost:
        return {**base, "status": "ABOVE_COST",
                "detail": f"stop {stop} ≥ 成本 {cost}——止损在成本上方,疑似方向写反或抄错"}
    if band is None:
        return {**base, "status": "UNKNOWN",
                "detail": f"仓别 {bucket!r} 未知,无政策带可比(短线/长线之外不判定)"}
    low, high = band
    if low <= stop <= high:
        return {**base, "status": "OK",
                "detail": f"落在政策带 [{low}, {high}] 内(距成本 {(stop / cost - 1) * 100:+.1f}%)"}
    side = "偏紧(离成本太近)" if stop > high else "偏松(离成本太远)"
    return {**base, "status": "OUT_OF_BAND",
            "detail": f"stop {stop} 在政策带 [{low}, {high}] 外·{side}"
                      f"(距成本 {(stop / cost - 1) * 100:+.1f}%)"}


def check_positions(positions: Iterable[dict[str, Any]],
                    tolerance: float = DEFAULT_TOLERANCE) -> list[dict[str, Any]]:
    """批量判定,保持输入顺序。"""
    return [check_position(p, tolerance) for p in positions]


def summarize(results: list[dict[str, Any]]) -> dict[str, int]:
    """按 status 计数(供 CLI 退出码与简报使用)。"""
    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return counts


def needs_attention(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """需要人工处理的行(非 OK);UNKNOWN 也算——未知不解释为安全。"""
    return [r for r in results if r["status"] != "OK"]


# 短线仓时间止损窗口:10 交易日。**本行是该制度常数的唯一定义处**,全仓没有第二份
# (trade_journal.RECENT_N 数值同为 10 但单位是「笔」= 展示行数,改它不动风控;
#  曾被本注释误称为"出处",单位都不同,照它去改常数会改错文件)。
# 计数单位 = holdings.held_trading_days 口径:含 entry 当日,entry 当天记第 1 日。
# 故触发日 = entry 之后第 9 个交易日;同一笔在 trade_journal(T+1 口径,不含 entry
# 当日)只记 9 —— 复盘看到短线 avg_hold_days≈9 是口径差 1,不是纪律执行过紧。
SHORT_TIME_STOP_DAYS_INCLUSIVE = 10


def check_time_stop(record: dict[str, Any],
                    limit: int = SHORT_TIME_STOP_DAYS_INCLUSIVE) -> dict[str, Any] | None:
    """短线仓时间止损提醒:held_days ≥ 10 交易日 → surface(只提示,不替你卖)。

    record = holdings_watch/build_position_record 的持仓估值行(含 bucket/held_days)。
    held_days 只认 holdings.held_trading_days 的含 entry 当日口径;此处**不做 ±1 换算**
    ——两套口径一旦在判定层混用,"第几天该卖"就再也对不上账(复盘侧的 T+1 口径由
    trade_journal 自己声明,换算责任留在读数一侧)。
    只对短线仓有意义;非短线、held_days 缺失(停牌无行情等)→ None(不伪造判定,
    缺失由 stop 检查那条线的 UNKNOWN 覆盖,不在这里重复报)。
    """
    if not isinstance(record, dict):
        raise StopPolicyError("record 必须是对象")
    if _normalize_bucket(record.get("bucket")) != "short":
        return None
    held = record.get("held_days")
    if isinstance(held, bool) or not isinstance(held, int) or held < 0:
        return None
    if held < limit:
        return None
    return {
        "ts_code": str(record.get("ts_code") or ""),
        "status": "TIME_STOP",
        "held_days": held,
        "limit": limit,
        "detail": f"短线仓已持有 {held} 个交易日 ≥ {limit}(时间止损窗口)"
                  "——按双仓制审视是否了结(人工终判)",
    }


def check_time_stops(records: Iterable[dict[str, Any]],
                     limit: int = SHORT_TIME_STOP_DAYS_INCLUSIVE) -> list[dict[str, Any]]:
    """批量时间止损提醒,只返回命中的行。"""
    out = []
    for r in records:
        hit = check_time_stop(r, limit)
        if hit is not None:
            out.append(hit)
    return out


def conditional_order_coverage(account: dict[str, Any]) -> dict[str, Any]:
    """持仓 × 条件单覆盖核查(只读):回答"破线保护到底确认了没有"。

    背景(2026-08-19 用户拍板"哨兵无用"后):盘中零自动监控,破线保护**只剩券商
    条件单一条防线**,而系统无法验证券商端真实状态。此函数不假装能验证券商——
    它只回答账本能回答的部分:结构化条件单是否 verified、哪些持仓**没有** active
    SELL 单覆盖。其余(价格是否仍匹配、单子是否被券商撤销/过期)只有你能核对。

    account = normalize_account_state 的输出。返回:
      {status, verified, orders_format, covered, uncovered, note}
      status ∈ {VERIFIED, UNVERIFIED, INVALID, NO_POSITIONS}
    绝不因"没有条件单数据"就说安全——未知一律记入 uncovered(未知不解释为安全)。
    """
    positions = [p for p in account.get("positions", []) if isinstance(p, dict)]
    co = account.get("conditional_orders") or {}
    co_status = str(co.get("status") or "missing")
    held = [str(p.get("ts_code")) for p in positions if p.get("ts_code")]
    if not held:
        return {"status": "NO_POSITIONS", "verified": True, "orders_format": co.get("format"),
                "covered": [], "uncovered": [], "note": "无持仓"}

    # 只有 structured v2 且 verified 时,才可能逐仓判定覆盖;自由文本条件单无法机读
    raw_orders = co.get("orders") if isinstance(co.get("orders"), list) else None
    if co_status != "verified" or raw_orders is None:
        return {
            "status": "INVALID" if co_status == "invalid" else "UNVERIFIED",
            "verified": False, "orders_format": co.get("format"),
            "covered": [], "uncovered": held,
            "note": f"条件单状态={co_status}——系统无法确认任何持仓有破线保护;"
                    "盘中无自动监控,请自行到券商核对条件单是否仍有效",
        }
    protected = {
        str(o.get("ts_code")) for o in raw_orders
        if isinstance(o, dict) and str(o.get("side", "")).upper() == "SELL"
        and str(o.get("status", "")).lower() == "active"
    }
    covered = [c for c in held if c in protected]
    uncovered = [c for c in held if c not in protected]
    return {
        "status": "VERIFIED", "verified": True, "orders_format": co.get("format"),
        "covered": covered, "uncovered": uncovered,
        "note": ("全部持仓有 active SELL 条件单(仍需你核对券商端价格/有效期)"
                 if not uncovered else
                 f"{len(uncovered)} 只持仓无 active SELL 条件单——盘中无自动监控,这些仓"
                 "破线时不会有任何提醒或自动保护"),
    }
