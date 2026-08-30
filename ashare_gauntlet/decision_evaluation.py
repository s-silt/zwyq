"""历史决策链的冻结输出审计与反事实可执行性评估。

本模块只消费调用方传入的快照和行情表，不读文件、不联网、不打印。历史人工
fact-check、账户和政策没有完整版本化，因此这里只审计当时已经冻结在 decision
snapshot 里的状态；未知永远不补成 clear，也不把建议数量解释为真实成交。
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, time
import math
import re
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from ashare_gauntlet.backtest import newey_west_tstat
from ashare_gauntlet.costs import round_trip_cost_rate
from scripts.factor_backtest import one_word_limit_down, one_word_limit_up

DATE_RE = re.compile(r"^\d{8}$")
STATES = {"BUY", "WAIT", "HOLD", "EXIT"}
FACTCHECK_CODES = {
    "FACTCHECK_CLEAR": "clear_as_recorded",
    "GOVERNANCE_RED": "red_as_recorded",
    "FACTCHECK_EXPIRED": "expired_as_recorded",
    "FACTCHECK_REQUIRED": "missing_as_recorded",
    "FACTCHECK_AFTER_AS_OF": "future_evidence_rejected",
}
DEFAULT_HORIZONS = (5, 10, 21, 42)
COMMISSION_RATE = 0.00025
SLIPPAGE_RATE = 0.0015


class DecisionEvaluationError(ValueError):
    """决策快照或行情契约矛盾，不能安全评估。"""


def _is_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _is_date8(value: Any) -> bool:
    if not isinstance(value, str) or not DATE_RE.fullmatch(value):
        return False
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError:
        return False
    return True


def factcheck_status(reason_codes: Iterable[str]) -> str:
    """从冻结 reason code 提取当时的 fact-check 状态；不读取当前覆盖文件。"""
    found = {FACTCHECK_CODES[code] for code in reason_codes if code in FACTCHECK_CODES}
    if len(found) > 1:
        raise DecisionEvaluationError(f"fact-check reason codes 互相矛盾: {sorted(found)}")
    return next(iter(found)) if found else "unknown"


def audit_snapshot(snapshot: Any, file_date: str, factor_rows: Any | None,
                   *, factor_integrity_status: str = "unverified_legacy") -> dict[str, Any]:
    """审计一份冻结快照，返回结构化错误而不是因单个坏文件中断整批审计。

    legacy snapshot 没有内容摘要；文件名和日期只能证明“当前文件可读”，不能证明其
    内容自生成后未被覆盖。因此缺摘要时明确标为 ``unverified_legacy``。
    """
    errors: list[str] = []
    warnings: list[str] = []
    if not _is_date8(file_date):
        return _audit_result(file_date, errors=["文件日期必须是真实 YYYYMMDD"])
    if not isinstance(snapshot, dict):
        return _audit_result(file_date, errors=["snapshot 顶层必须是对象"])

    as_of = snapshot.get("as_of")
    if not _is_date8(as_of):
        errors.append("as_of 必须是真实 YYYYMMDD")
    elif as_of != file_date:
        errors.append(f"文件日期 {file_date} 与 as_of {as_of} 不一致")
    if snapshot.get("data_status") != "complete":
        errors.append("data_status 必须为 complete")
    generated_at = snapshot.get("generated_at")
    try:
        generated_dt = datetime.fromisoformat(generated_at) if isinstance(generated_at, str) else None
        if generated_dt is None or generated_dt.tzinfo is None:
            raise ValueError
    except ValueError:
        generated_dt = None
        errors.append("generated_at 必须是带时区的 ISO-8601 时间")
    if generated_dt is not None and _is_date8(as_of):
        generated_local = generated_dt.astimezone(ZoneInfo("Asia/Shanghai"))
        if generated_local.strftime("%Y%m%d") < as_of:
            errors.append("generated_at 不能早于 as_of")
        elif (generated_local.strftime("%Y%m%d") == as_of
              and generated_local.timetz().replace(tzinfo=None) < time(15, 0)):
            errors.append("generated_at 早于 as_of 收盘，不能声称已使用当日 EOD")

    factor_path = snapshot.get("factor_snapshot")
    if not isinstance(factor_path, str):
        errors.append("factor_snapshot 必须是路径字符串")
    else:
        m = re.search(r"(?:^|[/\\])(\d{8})_factor\.json$", factor_path)
        if not m or (isinstance(as_of, str) and m.group(1) != as_of):
            errors.append("factor_snapshot 文件日期必须与 as_of 一致")
    if factor_rows is None:
        errors.append("factor_snapshot 不存在或不可读取")
    elif not isinstance(factor_rows, list):
        errors.append("factor_snapshot 顶层必须是列表")

    account_as_of = snapshot.get("account_as_of")
    account_schema = snapshot.get("account_source_schema")
    provenance_complete = account_as_of is not None and account_schema is not None
    if not provenance_complete:
        warnings.append("legacy snapshot 缺账户 provenance，完整输入不可重算")
    elif account_as_of != as_of:
        errors.append("account_as_of 必须与 as_of 一致")

    decisions = snapshot.get("decisions")
    if not isinstance(decisions, list):
        errors.append("decisions 必须是列表")
        decisions = []
    seen: set[str] = set()
    factor_by_code: dict[str, dict[str, Any]] = {}
    if isinstance(factor_rows, list):
        for row in factor_rows:
            if isinstance(row, dict) and isinstance(row.get("ts_code"), str):
                code = row["ts_code"]
                if code in factor_by_code:
                    errors.append(f"factor snapshot 重复 ts_code {code}")
                factor_by_code[code] = row

    encoded = 0
    for i, decision in enumerate(decisions):
        prefix = f"decisions[{i}]"
        if not isinstance(decision, dict):
            errors.append(f"{prefix} 必须是对象")
            continue
        code = decision.get("ts_code")
        if not isinstance(code, str) or not code:
            errors.append(f"{prefix}.ts_code 非法")
            continue
        if code in seen:
            errors.append(f"重复 ts_code {code}")
        seen.add(code)
        state = decision.get("state")
        if state not in STATES:
            errors.append(f"{prefix}.state 非法: {state!r}")
        reasons = decision.get("reason_codes")
        evidence = decision.get("evidence")
        execution = decision.get("execution")
        invalidations = decision.get("invalidations")
        if not isinstance(reasons, list) or not all(isinstance(x, str) for x in reasons):
            errors.append(f"{prefix}.reason_codes 必须是字符串列表")
            reasons = []
        if not isinstance(evidence, dict):
            errors.append(f"{prefix}.evidence 必须是对象")
            evidence = {}
        if not isinstance(execution, dict):
            errors.append(f"{prefix}.execution 必须是对象")
            execution = {}
        elif state == "BUY" and execution.get("eligible_from") != "NEXT_TRADING_DAY":
            errors.append(f"{prefix}.execution.eligible_from 必须为 NEXT_TRADING_DAY")
        if not isinstance(invalidations, list):
            errors.append(f"{prefix}.invalidations 必须是列表")
        try:
            fc = factcheck_status(reasons)
            if fc != "unknown":
                encoded += 1
        except DecisionEvaluationError as exc:
            errors.append(f"{prefix}: {exc}")
            fc = "unknown"

        shares = execution.get("shares")
        if (not isinstance(shares, int) or isinstance(shares, bool) or shares < 0):
            errors.append(f"{prefix}.execution.shares 必须是非负整数")
        max_entry = execution.get("max_entry_price")
        if max_entry is not None and (not _is_number(max_entry) or float(max_entry) <= 0):
            errors.append(f"{prefix}.execution.max_entry_price 必须为正有限数或 null")

        if state == "BUY":
            decile = evidence.get("decile")
            # X-14:生产候选池 = 当期 D10(D10 码)∪ B8 带保留成员(B8_BAND 码);decile
            # 与来源码不一致=证据自相矛盾,按错误上报
            if not ((decile == 10 and "D10" in reasons)
                    or (decile in (8, 9) and "B8_BAND" in reasons)):
                errors.append(f"BUY {code} 不在生产候选池(decile={decile!r} 与 D10/B8_BAND 码不一致)")
            if "FACTCHECK_CLEAR" not in reasons:
                errors.append(f"BUY {code} 缺 FACTCHECK_CLEAR")
            if fc != "clear_as_recorded":
                errors.append(f"BUY {code} 的 fact-check 冻结状态不是 clear")
            factor_row = factor_by_code.get(code)
            if factor_row is None:
                errors.append(f"BUY {code} 在 factor snapshot 中缺行")
            elif factor_row.get("decile") not in (8, 9, 10):
                errors.append(f"BUY {code} 在 factor snapshot 中不在 D8+ 池(decile="
                              f"{factor_row.get('decile')!r})")
        elif state in {"HOLD", "EXIT"} and code not in factor_by_code:
            warnings.append(f"{state} {code} 已掉出 factor snapshot；保留冻结持仓状态")

    total = len(decisions)
    result = _audit_result(file_date, errors=errors, warnings=warnings)
    result.update({
        "as_of": as_of,
        "generated_at": generated_at,
        "factor_snapshot": factor_path,
        "decision_count": total,
        "factcheck_encoded_count": encoded,
        "factcheck_encoded_coverage": encoded / total if total else None,
        "frozen_output_replayability": "invalid_snapshot" if errors else "frozen_output_readable",
        "content_integrity_status": "invalid" if errors else factor_integrity_status,
        "pit_evidence_status": "invalid" if errors else (
            "complete" if provenance_complete and encoded == total else "partial"),
        "full_input_recomputability": "not_recomputable",
    })
    return result


def _audit_result(file_date: str, *, errors: list[str], warnings: list[str] | None = None) -> dict[str, Any]:
    return {
        "file_date": file_date,
        "valid": not errors,
        "errors": errors,
        "warnings": warnings or [],
    }


def extract_buy_episodes(snapshots: list[dict[str, Any]],
                         known_trade_days: "Iterable[str] | None" = None) -> list[dict[str, Any]]:
    """把连续 BUY 压成 episode；``_unknown_boundary`` 禁止跨坏快照合并。

    ``known_trade_days``(可选)提供本地已知交易日全集:相邻两份快照之间若存在
    已知交易日却没有任何快照文件,该缺口视同 unknown boundary——那天可能生成过
    WAIT/EXIT 快照后丢失,直接把后一份的 BUY 当延续会少计 episode 并漏掉左删失
    标记(codex review P1-3)。未知不解释为"未运行"。
    """
    known = sorted(set(str(d) for d in known_trade_days)) if known_trade_days else []
    episodes: list[dict[str, Any]] = []
    previous: dict[str, str] = {}
    boundary_known = False
    prev_as_of: str | None = None
    for snapshot in snapshots:
        cur_as_of = str(snapshot.get("as_of") or "") or None
        if prev_as_of and cur_as_of and any(
                prev_as_of < day < cur_as_of for day in known):
            previous = {}
            boundary_known = False
        if cur_as_of:
            prev_as_of = cur_as_of
        if snapshot.get("_unknown_boundary"):
            previous = {}
            boundary_known = False
            continue
        current: dict[str, str] = {}
        for decision in snapshot["decisions"]:
            code = str(decision["ts_code"])
            state = str(decision["state"])
            current[code] = state
            if state == "BUY" and previous.get(code) != "BUY":
                execution = decision.get("execution", {})
                episodes.append({
                    "episode_id": f"{snapshot['as_of']}:{code}",
                    "as_of": snapshot["as_of"],
                    "generated_at": snapshot.get("generated_at"),
                    "ts_code": code,
                    "name": decision.get("name", code),
                    "shares_advised": execution.get("shares", 0),
                    "max_entry_price": execution.get("max_entry_price"),
                    "factcheck_status": factcheck_status(decision.get("reason_codes", [])),
                    "left_censored": not boundary_known,
                    "actual_execution": False,
                })
        previous = current
        boundary_known = True
    return episodes


def build_market_tables(daily: pd.DataFrame, adj_factor: pd.DataFrame) -> tuple[list[str], dict[str, pd.DataFrame]]:
    """构造评估所需的逐日行情；任一 daily 行缺复权因子即 fail-loud。"""
    need_daily = {"ts_code", "trade_date", "open", "high", "low", "close"}
    need_adj = {"ts_code", "trade_date", "adj_factor"}
    missing = (need_daily - set(daily.columns)) | (need_adj - set(adj_factor.columns))
    if missing:
        raise DecisionEvaluationError(f"行情输入缺列 {sorted(missing)}")
    if daily.empty:
        raise DecisionEvaluationError("daily 行情为空")
    px = daily[list(need_daily)].copy()
    adj = adj_factor[list(need_adj)].copy()
    px["ts_code"] = px["ts_code"].astype(str)
    px["trade_date"] = px["trade_date"].astype(str)
    adj["ts_code"] = adj["ts_code"].astype(str)
    adj["trade_date"] = adj["trade_date"].astype(str)
    px = px.merge(adj, on=["ts_code", "trade_date"], how="left", validate="one_to_one")
    miss = px.loc[px["adj_factor"].isna(), "trade_date"].unique()
    if len(miss):
        raise DecisionEvaluationError(f"adj_factor 缺交易日 {sorted(miss)[:4]}")
    numeric = ["open", "high", "low", "close", "adj_factor"]
    for column in numeric:
        converted = pd.to_numeric(px[column], errors="coerce")
        invalid = converted.notna() & (~converted.map(math.isfinite) | (converted <= 0))
        # daily 中 NaN open 表示停牌，允许；非空脏文本或非正/无穷值不是停牌。
        coerced = px[column].notna() & converted.isna()
        if bool((invalid | coerced).any()):
            bad = px.loc[invalid | coerced, ["ts_code", "trade_date"]].head(3).to_dict("records")
            raise DecisionEvaluationError(f"行情字段 {column} 含非法值: {bad}")
        px[column] = converted
    ohlc_present = px[["open", "high", "low", "close"]].notna().sum(axis=1)
    incomplete_ohlc = (ohlc_present > 0) & (ohlc_present < 4)
    if bool(incomplete_ohlc.any()):
        raise DecisionEvaluationError("daily OHLC 必须四价全空（停牌）或全齐")
    traded = ohlc_present == 4
    bad_ohlc = traded & ((px["low"] > px["high"]) | (px["open"] < px["low"])
                         | (px["open"] > px["high"]) | (px["close"] < px["low"])
                         | (px["close"] > px["high"]))
    if bool(bad_ohlc.any()):
        raise DecisionEvaluationError("daily OHLC 关系非法")
    px["adj_open"] = px["open"] * px["adj_factor"]
    px["adj_close"] = px["close"] * px["adj_factor"]
    trade_days = sorted(str(x) for x in px["trade_date"].unique())
    return trade_days, {d: g.reset_index(drop=True) for d, g in px.groupby("trade_date")}


def _require_limit_day(limit_by_date: dict[str, pd.DataFrame], date: str) -> pd.DataFrame:
    frame = limit_by_date.get(date)
    if frame is None or frame.empty:
        raise DecisionEvaluationError(f"stk_limit/{date} 缺失或为空")
    return frame


def _limit_value(limit: pd.DataFrame, code: str, column: str, date: str) -> float:
    rows = limit[limit["ts_code"].astype(str) == code]
    if len(rows) != 1:
        raise DecisionEvaluationError(f"stk_limit/{date} {code} 的 {column} 必须恰有一行")
    value = rows.iloc[0].get(column)
    if not _is_number(value) or float(value) <= 0:
        raise DecisionEvaluationError(f"stk_limit/{date} {code} 的 {column} 非正有限数")
    return float(value)


def _first_eligible_open(as_of: str, generated_at: str | None, trade_days: list[str]) -> str | None:
    candidates = [d for d in trade_days if d > as_of]
    if not candidates:
        return None
    if generated_at is None:
        return candidates[0]
    generated = datetime.fromisoformat(generated_at).astimezone(ZoneInfo("Asia/Shanghai"))
    generated_date = generated.strftime("%Y%m%d")
    generated_time = generated.timetz().replace(tzinfo=None)
    for date in candidates:
        # 同日 09:30 前生成才可使用当日开盘；盘中/盘后生成必须等下一交易日。
        if date > generated_date or (date == generated_date and generated_time < time(9, 30)):
            return date
    return None


def _entry_for_code(code: str, as_of: str, trade_days: list[str],
                    market_by_date: dict[str, pd.DataFrame],
                    limit_by_date: dict[str, pd.DataFrame],
                    max_entry_price: float | None = None,
                    generated_at: str | None = None) -> dict[str, Any]:
    entry_date = _first_eligible_open(as_of, generated_at, trade_days)
    if entry_date is None:
        return {"status": "entry_not_mature", "entry_date": None}
    daily = market_by_date[entry_date]
    limit = _require_limit_day(limit_by_date, entry_date)
    row = daily[daily["ts_code"] == code]
    if row.empty or pd.isna(row.iloc[0]["adj_open"]):
        return {"status": "suspended_next_day", "entry_date": entry_date}
    _limit_value(limit, code, "up_limit", entry_date)
    if code in one_word_limit_up(daily, limit, [code]):
        return {"status": "one_word_limit_up", "entry_date": entry_date}
    raw_open = float(row.iloc[0]["open"])
    if max_entry_price is not None and raw_open > max_entry_price + 1e-9:
        return {"status": "above_max_entry_price", "entry_date": entry_date,
                "open": raw_open, "max_entry_price": max_entry_price}
    return {"status": "fillable_next_open", "entry_date": entry_date,
            "entry_price": float(row.iloc[0]["adj_open"])}


def evaluate_code_horizon(code: str, as_of: str, horizon: int, trade_days: list[str],
                          market_by_date: dict[str, pd.DataFrame],
                          limit_by_date: dict[str, pd.DataFrame],
                          *, max_entry_price: float | None = None,
                          generated_at: str | None = None,
                          commission_rate: float = COMMISSION_RATE,
                          slippage_rate: float = SLIPPAGE_RATE) -> dict[str, Any]:
    """评估一个冻结信号；买入不顺延，卖出受阻则顺延到首个可卖开盘。"""
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
        raise DecisionEvaluationError(f"horizon 必须为正整数，得到 {horizon!r}")
    entry = _entry_for_code(code, as_of, trade_days, market_by_date, limit_by_date,
                            max_entry_price=max_entry_price, generated_at=generated_at)
    out: dict[str, Any] = {"horizon": horizon, **entry}
    if entry["status"] != "fillable_next_open":
        out["outcome_status"] = entry["status"]
        return out

    entry_idx = trade_days.index(entry["entry_date"])
    target_idx = entry_idx + horizon
    if target_idx >= len(trade_days):
        out["outcome_status"] = "insufficient_maturity"
        return out

    target_date = trade_days[target_idx]
    out["target_exit_date"] = target_date
    for pos in range(target_idx, len(trade_days)):
        date = trade_days[pos]
        daily = market_by_date[date]
        row = daily[daily["ts_code"] == code]
        if row.empty or pd.isna(row.iloc[0]["adj_open"]):
            continue
        limit = _require_limit_day(limit_by_date, date)
        _limit_value(limit, code, "down_limit", date)
        if code in one_word_limit_down(daily, limit, [code]):
            continue
        exit_price = float(row.iloc[0]["adj_open"])
        gross = exit_price / float(entry["entry_price"]) - 1.0
        cost = round_trip_cost_rate(entry["entry_date"], commission_rate, slippage_rate,
                                    sell_date=date)
        out.update({
            "outcome_status": "resolved",
            "exit_date": date,
            "exit_price": exit_price,
            "exit_deferred_days": pos - target_idx,
            "gross_return": gross,
            "round_trip_cost_rate": cost,
            "net_return": gross - cost,
        })
        return out
    out["outcome_status"] = "unresolved_exit"
    return out


def interval_return(index_daily: pd.DataFrame | None, start: str, end: str) -> float | None:
    """沪深300同一 entry-open → exit-open 区间收益；缺端点时显式缺失。"""
    if index_daily is None or index_daily.empty:
        return None
    if not {"trade_date", "open"}.issubset(index_daily.columns):
        raise DecisionEvaluationError("指数行情缺 trade_date/open，不能构造同持有时点基准")
    d = index_daily.copy()
    d["trade_date"] = d["trade_date"].astype(str)
    values = pd.to_numeric(d["open"], errors="coerce")
    bad = d["open"].notna() & (values.isna() | ~values.map(math.isfinite) | (values <= 0))
    if bool(bad.any()) or d["trade_date"].duplicated().any():
        raise DecisionEvaluationError("指数 open 非正有限数或日期重复")
    s = values[d["trade_date"] == start].dropna()
    e = values[d["trade_date"] == end].dropna()
    if len(s) != 1 or len(e) != 1:
        return None
    return float(e.iloc[0] / s.iloc[0] - 1.0)


def _mean(values: list[float]) -> float | None:
    return float(pd.Series(values, dtype=float).mean()) if values else None


def evaluate_episodes(episodes: list[dict[str, Any]], factors_by_date: dict[str, list[dict[str, Any]]],
                      trade_days: list[str], market_by_date: dict[str, pd.DataFrame],
                      limit_by_date: dict[str, pd.DataFrame], horizons: tuple[int, ...] = DEFAULT_HORIZONS,
                      index_daily: pd.DataFrame | None = None,
                      *, commission_rate: float = COMMISSION_RATE,
                      slippage_rate: float = SLIPPAGE_RATE) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """评估 BUY episode，并给出同日 D10 可执行等权基线。"""
    enriched: list[dict[str, Any]] = []
    for episode in episodes:
        event = dict(episode)
        event["outcomes"] = {}
        factor_rows = factors_by_date.get(str(event["as_of"]), [])
        d10 = sorted({str(r["ts_code"]) for r in factor_rows
                      if isinstance(r, dict) and r.get("decile") == 10 and r.get("ts_code")})
        for horizon in horizons:
            outcome = evaluate_code_horizon(
                event["ts_code"], event["as_of"], horizon, trade_days, market_by_date,
                limit_by_date, max_entry_price=event.get("max_entry_price"),
                generated_at=event.get("generated_at"), commission_rate=commission_rate,
                slippage_rate=slippage_rate)
            bench_net: list[float] = []
            bench_gross: list[float] = []
            bench_statuses: Counter[str] = Counter()
            if outcome.get("outcome_status") == "resolved":
                for code in d10:
                    member = evaluate_code_horizon(
                        code, event["as_of"], horizon, trade_days, market_by_date, limit_by_date,
                        generated_at=event.get("generated_at"), commission_rate=commission_rate,
                        slippage_rate=slippage_rate)
                    status = str(member.get("outcome_status"))
                    bench_statuses[status] += 1
                    if status == "resolved":
                        bench_gross.append(float(member["gross_return"]))
                        bench_net.append(float(member["net_return"]))
                idx_ret = interval_return(index_daily, outcome["entry_date"], outcome["exit_date"])
                outcome["hs300_open_return"] = idx_ret
                outcome["excess_vs_hs300"] = (
                    float(outcome["net_return"]) - idx_ret if idx_ret is not None else None)
                outcome["d10_attempted_count"] = len(d10)
                outcome["d10_outcome_counts"] = dict(sorted(bench_statuses.items()))
                outcome["d10_fillable_count"] = sum(
                    count for status, count in bench_statuses.items()
                    if status not in {"one_word_limit_up", "suspended_next_day", "entry_not_mature"})
                outcome["d10_resolved_count"] = len(bench_net)
                incomplete = any(status in {"unresolved_exit", "insufficient_maturity"}
                                 for status in bench_statuses)
                outcome["d10_benchmark_status"] = (
                    "incomplete_exit_coverage" if incomplete else "complete")
                # 未解退出属于已买入成员，不能事后从分母删掉后仍发布“完整”基准。
                outcome["d10_gross_return"] = None if incomplete else _mean(bench_gross)
                outcome["d10_net_return"] = None if incomplete else _mean(bench_net)
                outcome["increment_vs_d10"] = (
                    float(outcome["net_return"]) - float(outcome["d10_net_return"])
                    if outcome["d10_net_return"] is not None else None)
            event["outcomes"][str(horizon)] = outcome
        enriched.append(event)
    metrics = aggregate_metrics(enriched, horizons)
    # calendar-time 组合规则就位后回撤可实算(替换 aggregate_metrics 的
    # not_computed 默认);aggregate_metrics 自身无行情输入,保持纯 events 层。
    for horizon in horizons:
        metrics[str(horizon)].update(
            calendar_time_drawdown(enriched, horizon, trade_days, market_by_date))
    return enriched, metrics


def calendar_time_drawdown(events: list[dict[str, Any]], horizon: int,
                           trade_days: list[str],
                           market_by_date: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """calendar-time 等权组合的最大回撤(Fama 1998 / Mitchell-Stafford 2000 惯例)。

    规则(预先指定,用户批准 2026-08-12,纯测量层不改生产分配):每日组合收益 =
    当日活跃(已按次日开盘成交入场、尚未按开盘退出)episode 日收益的等权均值
    (每日再平衡);无活跃 episode 的日子净值不变。episode 逐日路径:入场日
    entry_open→close、中间日 close→close、退出日 prev_close→exit_open;停牌日
    收益记 0(净值冻结),复牌跳跃计入复牌日。未成交(一字涨停/停牌/超限价)
    episode 不占仓;未到期/未能退出的 episode 持有至数据尾——不事后剔除。
    回撤按毛价格路径计(成本在出入场两端一次性,不摊日频)。
    """
    daily_returns: dict[str, list[float]] = {}
    filled = 0
    for event in events:
        outcome = event.get("outcomes", {}).get(str(horizon), {})
        if outcome.get("status") != "fillable_next_open":
            continue
        filled += 1
        code = str(event["ts_code"])
        entry_date = str(outcome["entry_date"])
        exit_date = outcome.get("exit_date")
        idx0 = trade_days.index(entry_date)
        idx1 = trade_days.index(str(exit_date)) if exit_date else len(trade_days) - 1
        prev = float(outcome["entry_price"])
        for pos in range(idx0, idx1 + 1):
            date = trade_days[pos]
            row = market_by_date[date]
            match = row[row["ts_code"] == code]
            if pos == idx1 and exit_date is not None:
                price = float(outcome["exit_price"])
            elif match.empty or pd.isna(match.iloc[0]["adj_close"]):
                daily_returns.setdefault(date, []).append(0.0)   # 停牌冻结
                continue
            else:
                price = float(match.iloc[0]["adj_close"])
            daily_returns.setdefault(date, []).append(price / prev - 1.0)
            prev = price
    if not filled:
        return {"max_drawdown": None, "max_drawdown_status": "no_filled_episodes"}
    nav = peak = 1.0
    drawdown = 0.0
    for date in sorted(daily_returns):
        returns = daily_returns[date]
        nav *= 1.0 + sum(returns) / len(returns)
        peak = max(peak, nav)
        drawdown = min(drawdown, nav / peak - 1.0)
    return {
        "max_drawdown": drawdown,
        "max_drawdown_status": "computed_calendar_time_equal_weight",
        "max_drawdown_basis": "gross_price_path_entry_open_to_exit_open",
        "active_portfolio_days": len(daily_returns),
    }


def aggregate_metrics(events: list[dict[str, Any]], horizons: tuple[int, ...]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for horizon in horizons:
        outcomes = [e["outcomes"][str(horizon)] for e in events]
        resolved = [o for o in outcomes if o.get("outcome_status") == "resolved"]
        net = [float(o["net_return"]) for o in resolved]
        excess = [float(o["excess_vs_hs300"]) for o in resolved
                  if o.get("excess_vs_hs300") is not None]
        dated_increments = [
            (str(event["as_of"]), float(event["outcomes"][str(horizon)]["increment_vs_d10"]))
            for event in events
            if event["outcomes"][str(horizon)].get("increment_vs_d10") is not None
        ]
        increment_by_date = (pd.DataFrame(dated_increments, columns=["as_of", "value"])
                             .groupby("as_of")["value"].mean()
                             if dated_increments else pd.Series(dtype=float))
        increments = [float(x) for x in increment_by_date]
        # 统计单位是信号日横截面均值，而非同日每只股票；至少用 horizon-1 的 HAC lag
        # 吸收重叠持有窗口。样本太少时 newey_west_tstat 按既有契约返回 NaN。
        required_lag = max(horizon - 1, 0)
        minimum_dates = max(4, required_lag + 2)
        if len(increment_by_date) >= minimum_dates:
            _, nw_t, nw_lag = newey_west_tstat(increment_by_date, lag=required_lag)
            nw_status = "computed"
        else:
            nw_t, nw_lag = math.nan, required_lag
            nw_status = "insufficient_signal_dates_for_required_lag"
        metrics[str(horizon)] = {
            "episode_count": len(outcomes),
            "resolved_count": len(resolved),
            "fillable_count": sum(o.get("status") == "fillable_next_open" for o in outcomes),
            "immature_count": sum(o.get("outcome_status") in {"entry_not_mature", "insufficient_maturity"}
                                  for o in outcomes),
            "unresolved_exit_count": sum(o.get("outcome_status") == "unresolved_exit" for o in outcomes),
            "one_word_limit_up_count": sum(o.get("outcome_status") == "one_word_limit_up" for o in outcomes),
            "suspended_next_day_count": sum(o.get("outcome_status") == "suspended_next_day" for o in outcomes),
            "above_max_entry_price_count": sum(
                o.get("outcome_status") == "above_max_entry_price" for o in outcomes),
            "mean_net_return": _mean(net),
            "median_net_return": float(pd.Series(net).median()) if net else None,
            "positive_rate": sum(x > 0 for x in net) / len(net) if net else None,
            "mean_excess_vs_hs300": _mean(excess),
            "mean_increment_vs_d10": _mean(increments),
            "increment_signal_date_count": len(increment_by_date),
            "increment_nw_t": None if math.isnan(nw_t) else float(nw_t),
            "increment_nw_lag": nw_lag,
            "increment_nw_status": nw_status,
            "max_drawdown": None,
            "max_drawdown_status": "not_computed_without_calendar_time_portfolio",
        }
    return metrics


# X-09 预注册参数(docs/experiments.md,用户批准 2026-08-12;零新常数:
# 主评窗=生产月频 21 日,最小信号日=aggregate_metrics 既有 NW lag 契约 max(4, lag+2))
X09_PRIMARY_HORIZON = 21
X09_MIN_SIGNAL_DATES = max(4, (X09_PRIMARY_HORIZON - 1) + 2)
X09_T_THRESHOLD = 2.0


def increment_verdict(metrics: dict[str, Any], *,
                      horizon: int = X09_PRIMARY_HORIZON,
                      min_signal_dates: int = X09_MIN_SIGNAL_DATES,
                      t_threshold: float = X09_T_THRESHOLD) -> dict[str, Any]:
    """X-09:决策链 BUY vs 同日 D10 可执行等权的增量方向判定(预注册判据)。

    只输出判定与建议处置文本,绝不自动改生产——三分支处置(开启置信度连续化
    预注册 / 复审四关过滤 / 继续积累)永远由人工执行。判据在样本积累前预先
    指定,防事后挑窗挑门槛;NW t 与均值必然同号,分支只看 t。
    """
    base = {"experiment_id": "X-09", "primary_horizon": horizon,
            "min_signal_dates": min_signal_dates, "t_threshold": t_threshold,
            "production_action": "none_automatic"}
    row = metrics.get(str(horizon))
    if row is None:
        return {**base, "status": "primary_horizon_not_evaluated",
                "disposition": "在 horizons 中加入主评窗后重跑"}
    count = row.get("increment_signal_date_count") or 0
    t = row.get("increment_nw_t")
    base.update({"signal_date_count": count, "increment_nw_t": t,
                 "mean_increment_vs_d10": row.get("mean_increment_vs_d10")})
    if count < min_signal_dates:
        return {**base, "status": "insufficient_sample",
                "disposition": "继续积累,不下结论"}
    if t is None:
        return {**base, "status": "not_computable",
                "disposition": "样本达标但 NW t 不可算——检查增量序列退化"}
    if t >= t_threshold:
        return {**base, "status": "positive_direction_evidence",
                "disposition": "人工:登记置信度连续化预注册实验"}
    if t <= -t_threshold:
        return {**base, "status": "negative_direction_evidence",
                "disposition": "人工:复审四关+factcheck 是否过滤掉好股票"}
    return {**base, "status": "not_significant",
            "disposition": "继续积累,不下结论"}


def snapshot_summary(valid_snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    states: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    codes: set[str] = set()
    for snapshot in valid_snapshots:
        for decision in snapshot["decisions"]:
            states[str(decision["state"])] += 1
            reasons.update(str(x) for x in decision.get("reason_codes", []))
            codes.add(str(decision["ts_code"]))
    return {
        "snapshot_count": len(valid_snapshots),
        "decision_count": sum(states.values()),
        "unique_stock_count": len(codes),
        "state_counts": dict(sorted(states.items())),
        "reason_code_counts": dict(sorted(reasons.items())),
    }
