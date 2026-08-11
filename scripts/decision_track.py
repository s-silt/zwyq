"""冻结决策快照的 PIT 可回放性审计与 T+1 反事实效果评估。

只读历史 decisions、其引用的 factor snapshot 和本地 EOD/指数缓存；不会读取当前
holdings、policy 或 factcheck overrides 补历史，也不会改写生产决策。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import tempfile
from datetime import datetime
from typing import Any

import pandas as pd

from ashare_gauntlet.config import CACHE_DIR, HOLDSCORE_DIR
from ashare_gauntlet.decision_evaluation import (
    COMMISSION_RATE,
    DEFAULT_HORIZONS,
    SLIPPAGE_RATE,
    DecisionEvaluationError,
    audit_snapshot,
    build_market_tables,
    evaluate_episodes,
    extract_buy_episodes,
    snapshot_summary,
)

DECISION_DIR = "data/decisions"
DEFAULT_OUTPUT = f"{HOLDSCORE_DIR}/decision_chain_evaluation.json"
DECISION_RE = re.compile(r"^(\d{8})_buy_decisions\.json$")


def _inside_root(root: Path, path: Path) -> Path:
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise DecisionEvaluationError(f"路径逃出项目根: {path}") from exc
    return resolved


def _json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _factor_path(root: Path, value: Any) -> Path | None:
    if not isinstance(value, str):
        return None
    candidate = Path(value)
    if candidate.is_absolute():
        return _inside_root(root, candidate)
    return _inside_root(root, root / candidate)


def discover_snapshots(root: Path, start: str | None = None,
                       end: str | None = None) -> list[Path]:
    directory = _inside_root(root, root / DECISION_DIR)
    found: list[tuple[str, Path]] = []
    for path in directory.glob("*_buy_decisions.json") if directory.exists() else []:
        match = DECISION_RE.fullmatch(path.name)
        if not match:
            continue
        date = match.group(1)
        if end and date > end:
            continue
        found.append((date, _inside_root(root, path)))
    found.sort()
    if not start:
        return [path for _, path in found]
    in_range = [(date, path) for date, path in found if date >= start]
    before = [(date, path) for date, path in found if date < start]
    # 最近一份前置快照只提供 episode 左边界，不进入区间覆盖/效果统计。
    return ([before[-1][1]] if before else []) + [path for _, path in in_range]


def _load_market(root: Path, first_as_of: str) -> tuple[
        list[str], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    cache = _inside_root(root, root / CACHE_DIR)
    endpoints = ("daily", "adj_factor", "daily_basic", "stk_limit")
    date_sets = {
        endpoint: {p.stem for p in _inside_root(root, cache / endpoint).glob("????????.parquet")
                   if p.stem.isdigit() and p.stem >= first_as_of}
        for endpoint in endpoints
    }
    dates = sorted(date_sets["daily"])
    if not dates:
        raise DecisionEvaluationError(f"{first_as_of} 之后无 daily 日分区")
    # 以四核心端点日期集合互校：任一端点独有/缺失的交易日都会 fail-loud，不能
    # 让下一个日期被误当 T+1。严格交易日历另有历史覆盖缺口，报告中单独披露。
    union = set().union(*date_sets.values())
    mismatches = {endpoint: sorted(union - dates_) for endpoint, dates_ in date_sets.items()
                  if union - dates_}
    if mismatches:
        raise DecisionEvaluationError(f"四核心 EOD 日期集合不一致: {mismatches}")

    daily_frames: list[pd.DataFrame] = []
    adj_frames: list[pd.DataFrame] = []
    limits: dict[str, pd.DataFrame] = {}
    for date in dates:
        daily_path = _inside_root(root, cache / "daily" / f"{date}.parquet")
        adj_path = _inside_root(root, cache / "adj_factor" / f"{date}.parquet")
        basic_path = _inside_root(root, cache / "daily_basic" / f"{date}.parquet")
        limit_path = _inside_root(root, cache / "stk_limit" / f"{date}.parquet")
        daily_day = pd.read_parquet(
            daily_path, columns=["ts_code", "trade_date", "open", "high", "low", "close"])
        adj_day = pd.read_parquet(adj_path, columns=["ts_code", "trade_date", "adj_factor"])
        basic_day = pd.read_parquet(basic_path, columns=["ts_code", "trade_date", "total_mv"])
        limit_day = pd.read_parquet(
            limit_path, columns=["ts_code", "trade_date", "up_limit", "down_limit"])
        for endpoint, frame in (("daily", daily_day), ("adj_factor", adj_day),
                                ("daily_basic", basic_day), ("stk_limit", limit_day)):
            if frame.empty or set(frame["trade_date"].astype(str)) != {date}:
                raise DecisionEvaluationError(f"{endpoint}/{date} 为空或内容日期错配")
        daily_frames.append(daily_day)
        adj_frames.append(adj_day)
        limits[date] = limit_day
    daily = pd.concat(daily_frames, ignore_index=True)
    adj = pd.concat(adj_frames, ignore_index=True)
    trade_days, market = build_market_tables(daily, adj)
    return trade_days, market, limits


def _load_index_cache(root: Path) -> pd.DataFrame | None:
    path = _inside_root(root, root / CACHE_DIR / "index_daily" / "000300.SH.parquet")
    if not path.exists():
        return None
    return pd.read_parquet(path, columns=["trade_date", "open"])


def _calendar_coverage(root: Path, trade_days: list[str]) -> dict[str, Any]:
    """核对已归档 trade_cal；历史日历拉取不全时只标 partial，不伪称完整。"""
    if not trade_days:
        return {"status": "unavailable", "reason": "no_trade_days"}
    directory = _inside_root(root, root / CACHE_DIR / "trade_cal")
    paths = [_inside_root(root, path) for path in directory.glob("*.parquet")] if directory.exists() else []
    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            frame = pd.read_parquet(path, columns=["cal_date", "is_open"])
        except (OSError, ValueError, KeyError):
            continue
        frames.append(frame)
    if not frames:
        return {"status": "unavailable", "reason": "trade_cal_cache_missing"}
    cal = pd.concat(frames, ignore_index=True)
    cal["cal_date"] = cal["cal_date"].astype(str)
    cal = cal[(cal["cal_date"] >= trade_days[0]) & (cal["cal_date"] <= trade_days[-1])]
    cal = cal.drop_duplicates("cal_date", keep="last")
    expected_calendar = {d.strftime("%Y%m%d") for d in pd.date_range(trade_days[0], trade_days[-1])}
    covered = set(cal["cal_date"])
    open_days = set(cal.loc[pd.to_numeric(cal["is_open"], errors="coerce") == 1, "cal_date"])
    missing_calendar = sorted(expected_calendar - covered)
    missing_open_partitions = sorted(open_days - set(trade_days))
    unexpected_partitions = sorted(set(trade_days) - open_days) if not missing_calendar else []
    complete = not missing_calendar and not missing_open_partitions and not unexpected_partitions
    return {
        "status": "complete" if complete else "partial",
        "covered_calendar_days": len(covered),
        "expected_calendar_days": len(expected_calendar),
        "missing_calendar_dates": missing_calendar,
        "open_days_without_four_core_partitions": missing_open_partitions,
        "partition_days_not_marked_open": unexpected_partitions,
    }


def _coverage(audits: list[dict[str, Any]], snapshots: list[dict[str, Any]],
              events: list[dict[str, Any]], left_censored: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(audits)
    valid = sum(bool(x["valid"]) for x in audits)
    decisions = sum(int(x.get("decision_count", 0)) for x in audits)
    encoded = sum(int(x.get("factcheck_encoded_count", 0)) for x in audits)
    return {
        "snapshot_total": total,
        "snapshot_valid": valid,
        "snapshot_invalid": total - valid,
        "frozen_output_readability_coverage": valid / total if total else None,
        "content_integrity_verified_coverage": 0.0 if total else None,
        "factcheck_status_encoded_coverage": encoded / decisions if decisions else None,
        "full_input_recomputability": 0.0 if total else None,
        "valid_decision_count": sum(len(x["decisions"]) for x in snapshots),
        "buy_episode_count": len(events),
        "left_censored_buy_count": len(left_censored),
    }


def build_report(root: Path, *, start: str | None = None, end: str | None = None,
                 horizons: tuple[int, ...] = DEFAULT_HORIZONS,
                 commission_rate: float = COMMISSION_RATE,
                 slippage_rate: float = SLIPPAGE_RATE) -> dict[str, Any]:
    paths = discover_snapshots(root, start, end)
    if not paths:
        raise DecisionEvaluationError("指定区间没有 decision snapshot")

    audits: list[dict[str, Any]] = []
    valid_snapshots: list[dict[str, Any]] = []
    episode_sequence: list[dict[str, Any]] = []
    factors_by_date: dict[str, list[dict[str, Any]]] = {}
    snapshot_file_dates: set[str] = set()
    earliest_valid_as_of: str | None = None
    for path in paths:
        file_date = DECISION_RE.fullmatch(path.name).group(1)  # discover 已验证
        snapshot_file_dates.add(file_date)
        in_scope = (start is None or file_date >= start) and (end is None or file_date <= end)
        try:
            snapshot = _json(path)
        except (OSError, json.JSONDecodeError) as exc:
            audit = {
                "file_date": file_date, "valid": False,
                "errors": [f"snapshot 不可读取: {type(exc).__name__}: {exc}"],
                "warnings": [], "frozen_output_replayability": "invalid_snapshot",
                "content_integrity_status": "invalid", "pit_evidence_status": "invalid",
                "full_input_recomputability": "not_recomputable", "in_requested_scope": in_scope,
            }
            if in_scope:
                audits.append(audit)
            episode_sequence.append({"_unknown_boundary": True, "as_of": file_date})
            continue
        factor_rows = None
        path_error: str | None = None
        try:
            factor_path = _factor_path(root, snapshot.get("factor_snapshot")) if isinstance(snapshot, dict) else None
        except DecisionEvaluationError as exc:
            factor_path = None
            path_error = str(exc)
        if factor_path is not None and factor_path.exists():
            try:
                factor_rows = _json(factor_path)
            except (OSError, json.JSONDecodeError):
                factor_rows = None
        audit = audit_snapshot(snapshot, file_date, factor_rows)
        if path_error:
            audit["valid"] = False
            audit["errors"].append(path_error)
            audit["frozen_output_replayability"] = "invalid_snapshot"
            audit["content_integrity_status"] = "invalid"
        audit["in_requested_scope"] = in_scope
        if in_scope:
            audits.append(audit)
        if audit["valid"]:
            episode_sequence.append(snapshot)
            factors_by_date[str(snapshot["as_of"])] = factor_rows
            snap_as_of = str(snapshot["as_of"])
            if earliest_valid_as_of is None or snap_as_of < earliest_valid_as_of:
                earliest_valid_as_of = snap_as_of
            if in_scope:
                valid_snapshots.append(snapshot)
        else:
            episode_sequence.append({"_unknown_boundary": True, "as_of": file_date})

    scoped_episodes: list[dict[str, Any]] = []
    left_censored: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    enriched: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {str(h): {"episode_count": 0} for h in horizons}
    calendar_coverage: dict[str, Any] = {"status": "unavailable", "reason": "no_valid_snapshots"}
    benchmark_coverage = {"status": "not_checked"}
    generated_through = (max(str(x.get("as_of")) for x in valid_snapshots)
                         if valid_snapshots else (audits[-1]["file_date"] if audits else paths[-1].stem[:8]))
    if valid_snapshots:
        # 行情从最早可用快照(含区间前置锚)加载,快照间缺口检测才覆盖前置段。
        first_as_of = earliest_valid_as_of or min(str(x["as_of"]) for x in valid_snapshots)
        trade_days, market, limits = _load_market(root, first_as_of)
        generated_through = trade_days[-1]
        calendar_coverage = _calendar_coverage(root, trade_days)
        # codex P1-2a:归档日历完整却有开市日缺全部四核心分区 → T+1 映射必错
        # (下一个有分区的日期会被误当次日开盘),确凿数据缺口不得继续评估。
        if (not calendar_coverage.get("missing_calendar_dates")
                and calendar_coverage.get("open_days_without_four_core_partitions")):
            raise DecisionEvaluationError(
                "归档日历证明存在开市日缺全部四核心分区: "
                f"{calendar_coverage['open_days_without_four_core_partitions'][:5]}"
                "——T+1 映射不可信,先补齐 EOD 再评估")
        # codex P1-3:已知交易日 = EOD 分区日 ∪ 快照文件日;相邻快照之间隔着
        # 已知交易日却无快照文件 → unknown boundary,不跨缺口合并 episode。
        known_days = set(trade_days) | snapshot_file_dates
        scoped_episodes = [event for event in
                           extract_buy_episodes(episode_sequence, known_days)
                           if (start is None or event["as_of"] >= start)
                           and (end is None or event["as_of"] <= end)]
        left_censored = [event for event in scoped_episodes if event.get("left_censored")]
        # 左边界未知的首个 BUY 可能是早已持续的 episode，只保留覆盖计数，不纳入主效果。
        episodes = [event for event in scoped_episodes if not event.get("left_censored")]
        index_daily = _load_index_cache(root)
        benchmark_coverage = {
            "status": "available" if index_daily is not None else "unavailable",
            "basis": "open_to_open" if index_daily is not None else None,
        }
        enriched, metrics = evaluate_episodes(
            episodes, factors_by_date, trade_days, market, limits, horizons,
            index_daily, commission_rate=commission_rate,
            slippage_rate=slippage_rate)
        # codex P1-2b:T+1 判定的依据逐窗口显式标注——归档日历不完整时,交易日
        # 序列只来自现存分区,无法排除"四端点共同缺日",读数须按未验对待。
        t_plus_one_basis = (
            "verified_against_complete_calendar"
            if calendar_coverage.get("status") == "complete"
            else "partition_dates_only_unverified")
        for row in metrics.values():
            row["t_plus_one_basis"] = t_plus_one_basis

    return {
        "schema": "decision_chain_evaluation.v1",
        "generated_through": generated_through,
        "horizons": list(horizons),
        "scope": {
            "source": "frozen_decision_snapshots",
            "actual_execution": False,
            "production_policy_changed": False,
            "current_manual_state_used_for_history": False,
            "entry_assumption": "next_trading_day_open_counterfactual",
            "warning": "建议信号反事实评估，不是真实成交账户，也不是个股上涨概率",
        },
        "cost_assumptions": {
            "commission_rate_one_way": commission_rate,
            "slippage_rate_one_way": slippage_rate,
            "minimum_commission_included": False,
        },
        "calendar_coverage": calendar_coverage,
        "benchmark_coverage": benchmark_coverage,
        "coverage": _coverage(audits, valid_snapshots, enriched, left_censored),
        "decision_summary": snapshot_summary(valid_snapshots),
        "snapshot_audits": audits,
        "left_censored_events": left_censored,
        "events": enriched,
        "metrics": metrics,
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp",
                                         delete=False) as handle:
            tmp_name = handle.name
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if tmp_name and os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _parse_horizons(value: str) -> tuple[int, ...]:
    try:
        horizons = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("horizons 必须是逗号分隔的正整数") from exc
    if not horizons or any(x <= 0 for x in horizons) or len(set(horizons)) != len(horizons):
        raise argparse.ArgumentTypeError("horizons 必须是互异的正整数")
    return horizons


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--horizons", type=_parse_horizons,
                        default=DEFAULT_HORIZONS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--root", default=".", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    for flag, value in (("--start", args.start), ("--end", args.end)):
        if value is None:
            continue
        try:
            datetime.strptime(value, "%Y%m%d")
        except (TypeError, ValueError):
            parser.error(f"{flag} 必须是真实 YYYYMMDD")
    if args.start and args.end and args.start > args.end:
        parser.error("--start 不能晚于 --end")

    root = Path(args.root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output = _inside_root(root, output)
    report = build_report(root, start=args.start, end=args.end, horizons=args.horizons)
    atomic_write_json(output, report)
    coverage = report["coverage"]
    print(f"决策链评估: snapshots {coverage['snapshot_valid']}/{coverage['snapshot_total']} 有效; "
          f"BUY episodes={coverage['buy_episode_count']}; through={report['generated_through']}")
    for horizon in report["horizons"]:
        row = report["metrics"][str(horizon)]
        print(f"  {horizon}日: resolved={row.get('resolved_count', 0)}/"
              f"{row.get('episode_count', 0)} mean_net={row.get('mean_net_return')}")
    print(f"→ {output}(只读历史反事实评估;非真实成交/非推荐概率)")


if __name__ == "__main__":
    main()
