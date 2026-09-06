"""Safe service layer exposed by the local ashare-gauntlet MCP server.

The MCP boundary is intentionally narrower than a shell: every executable
workflow is allow-listed, paths are confined to the repository, and trade
state is read-only.  A recommendation is never recorded as an execution.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from ashare_gauntlet.governance import audit_opinion, controller_pledge
from ashare_gauntlet.intraday import fetch_quotes
from ashare_gauntlet.account_state import normalize_account_state, classify_account_freshness
from ashare_gauntlet.decision_snapshot import require_decision_snapshot_ready

_DATE8 = re.compile(r"^\d{8}$")
_TS_CODE = re.compile(r"^\d{6}\.(?:SH|SZ)$", re.IGNORECASE)
_DATED_FACTOR = re.compile(r"^(\d{8})_factor\.json$")
_DATED_DECISION = re.compile(r"^(\d{8})_buy_decisions\.json$")
_BACKFILL_REPORT_PREFIX = "BACKFILL_RESULT_JSON="
_DECISION_STATES = frozenset({"BUY", "WAIT", "HOLD", "EXIT"})
_CORE_MARKET_ENDPOINTS = ("daily", "adj_factor", "daily_basic", "stk_limit")
_STATE_ORDER = ("BUY", "WAIT", "HOLD", "EXIT")
_SHANGHAI = ZoneInfo("Asia/Shanghai")

ALLOWED_MODULES = frozenset({
    "scripts.backfill",
    "scripts.factor_rank",
    "scripts.buy_list",
    "scripts.aggressive_pick",
    "scripts.market_temp",
})


def project_root() -> Path:
    """Return the configured repository root (default: package parent)."""
    raw = os.environ.get("ASHARE_GAUNTLET_ROOT")
    return Path(raw).resolve() if raw else Path(__file__).resolve().parents[1]


def _inside_root(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"path escapes project root: {resolved}")
    return resolved


def read_json(relative: str, root: Path | None = None) -> Any:
    root = (root or project_root()).resolve()
    path = _inside_root(root / relative, root)
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _latest_file(directory: Path, pattern: re.Pattern[str]) -> Path:
    matches = [(m.group(1), p) for p in directory.glob("*.json")
               if (m := pattern.fullmatch(p.name))]
    if not matches:
        raise FileNotFoundError(f"no dated snapshot in {directory}")
    return max(matches, key=lambda item: item[0])[1]


def latest_factor_path(root: Path | None = None) -> Path:
    root = (root or project_root()).resolve()
    return _latest_file(root / "data" / "holdscore", _DATED_FACTOR)


def latest_decision_path(root: Path | None = None) -> Path:
    root = (root or project_root()).resolve()
    return _latest_file(root / "data" / "decisions", _DATED_DECISION)


def validate_codes(codes: Sequence[str], *, maximum: int = 100) -> list[str]:
    clean = [str(c).upper() for c in codes]
    if not clean or len(clean) > maximum:
        raise ValueError(f"codes must contain 1..{maximum} symbols")
    bad = [c for c in clean if not _TS_CODE.fullmatch(c)]
    if bad:
        raise ValueError(f"invalid ts_code: {bad}")
    return list(dict.fromkeys(clean))


def validate_date(value: str) -> str:
    if not _DATE8.fullmatch(value):
        raise ValueError(f"date must be YYYYMMDD, got {value!r}")
    datetime.strptime(value, "%Y%m%d")
    return value


def run_module(module: str, args: Sequence[str] = (), *, timeout: int = 900,
               root: Path | None = None) -> dict[str, Any]:
    """Run one allow-listed project CLI and return bounded UTF-8 output."""
    if module not in ALLOWED_MODULES:
        raise ValueError(f"module is not allow-listed: {module}")
    root = (root or project_root()).resolve()
    python = Path(sys.executable).resolve()
    if not python.exists():
        raise FileNotFoundError(f"current interpreter missing: {python}")
    env = os.environ.copy()
    env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
    try:
        proc = subprocess.run(
            [str(python), "-m", module, *map(str, args)], cwd=root, env=env,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {"ok": False, "module": module, "returncode": None,
                "stdout": (exc.stdout or "")[-20000:],
                "stderr": f"timeout after {timeout}s"}
    return {"ok": proc.returncode == 0, "module": module,
            "returncode": proc.returncode, "stdout": proc.stdout[-20000:],
            "stderr": proc.stderr[-10000:]}


def _latest_partition_date(root: Path, endpoint: str) -> str | None:
    directory = root / "data" / "cache" / endpoint
    dates = [p.stem for p in directory.glob("*.parquet") if _DATE8.fullmatch(p.stem)]
    return max(dates) if dates else None


def _number(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def _recommendation_readiness(root: Path) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    endpoint_dates = {ep: _latest_partition_date(root, ep) for ep in _CORE_MARKET_ENDPOINTS}
    observed_dates = {d for d in endpoint_dates.values() if d is not None}
    eod_as_of = next(iter(observed_dates)) if len(observed_dates) == 1 else None
    eod_status = "ready" if eod_as_of and all(endpoint_dates.values()) else "misaligned"
    if eod_status != "ready":
        blockers.append("CORE_EOD_MISSING_OR_MISALIGNED")

    factor_as_of: str | None = None
    decision_as_of: str | None = None
    decision_data_status: str | None = None
    c2_status: str | None = None
    factor_status = "missing"
    decision_status = "missing"
    try:
        factor_as_of = latest_factor_path(root).name[:8]
        factor_status = "ready" if factor_as_of == eod_as_of else "stale"
    except FileNotFoundError:
        pass
    if factor_status != "ready":
        blockers.append("FACTOR_NOT_ALIGNED")
    try:
        decision_path = latest_decision_path(root)
        decision_snapshot = read_json(str(decision_path.relative_to(root)), root)
        if isinstance(decision_snapshot, dict):
            raw_data_status = decision_snapshot.get("data_status")
            if isinstance(raw_data_status, str):
                decision_data_status = raw_data_status
            c2_state = decision_snapshot.get("c2_state")
            if isinstance(c2_state, dict) and isinstance(c2_state.get("status"), str):
                c2_status = c2_state["status"]
        decision = latest_decisions(root)
        decision_as_of = decision["as_of"]
        decision_status = "ready" if decision_as_of == factor_as_of == eod_as_of else "stale"
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        decision_status = "invalid"
    if decision_status != "ready":
        blockers.append("DECISION_NOT_ALIGNED")
    if decision_data_status == "degraded":
        blockers.append("DECISION_SNAPSHOT_DEGRADED")
    if c2_status == "REVIEW_BLOCKED_DATA":
        blockers.append("C2_REVIEW_BLOCKED_DATA")
    elif c2_status == "UNAVAILABLE":
        blockers.append("C2_REVIEW_UNAVAILABLE")

    account_status = "missing"
    holdings_as_of = None
    holdings_freshness = None
    short_slot: dict[str, Any] | None = None
    order_status = "missing"
    order_format = None
    try:
        account = account_snapshot(root)
        holdings_as_of = str(account.get("as_of")) if account.get("as_of") else None
        account_status = account["data_status"]
        short_slot = account["short_slot"]
        # 条件单只解析展示,missing/invalid/legacy 一律不挡荐股/配股/落账,
        # 也不得把未核验状态伪报为 verified。
        order_status = account["conditional_orders"]["status"]
        if account_status != "complete":
            blockers.append("ACCOUNT_STATE_INCOMPLETE")
        # 严格 freshness 必须以本次机器决策日期重算；账户查询本身不预设日期。
        holdings_freshness = (
            classify_account_freshness(holdings_as_of, decision_as_of)
            if decision_as_of else None
        )
        if holdings_freshness and holdings_freshness != "aligned":
            blockers.append(holdings_freshness)
        if short_slot["violation"]:
            blockers.append("SHORT_SLOT_VIOLATION")
    except (FileNotFoundError, ValueError, TypeError, json.JSONDecodeError):
        blockers.append("ACCOUNT_STATE_UNAVAILABLE")

    future_factchecks = 0
    if decision_as_of:
        try:
            overrides = read_json("data/factcheck_overrides.json", root).get("overrides", [])
            future_factchecks = sum(
                1 for item in overrides if str(item.get("as_of") or "") > decision_as_of
            )
        except (FileNotFoundError, AttributeError, json.JSONDecodeError):
            pass
    if future_factchecks:
        warnings.append("FACTCHECK_NEWER_THAN_DECISION_IGNORED")

    blockers = list(dict.fromkeys(blockers))
    return {
        "ready": not blockers,
        "status": "ready" if not blockers else "blocked",
        "as_of": decision_as_of or factor_as_of or eod_as_of,
        "components": {
            "eod": {"status": eod_status, "as_of": eod_as_of,
                    "endpoint_dates": endpoint_dates},
            "factor": {"status": factor_status, "as_of": factor_as_of},
            "decision": {
                "status": decision_status,
                "as_of": decision_as_of,
                "data_status": decision_data_status,
                "c2_status": c2_status,
            },
            "holdings": {"status": account_status, "as_of": holdings_as_of,
                         "freshness": holdings_freshness},
            "short_slot": short_slot,
            "conditional_orders": {"status": order_status},
            "factcheck": {"future_rows_ignored": future_factchecks},
        },
        "blockers": blockers,
        "warnings": warnings,
    }


def healthcheck(root: Path | None = None) -> dict[str, Any]:
    root = (root or project_root()).resolve()
    required = ["data/holdings.json", "data/trading_policy.json",
                "data/profile.json", "data/factcheck_overrides.json"]
    files = {p: (root / p).exists() for p in required}
    env_path = root / ".env.local"
    try:
        factor = latest_factor_path(root).name
    except FileNotFoundError:
        factor = None
    try:
        decision = latest_decision_path(root).name
    except FileNotFoundError:
        decision = None
    operational = all(files.values()) and Path(sys.executable).exists()
    readiness = _recommendation_readiness(root) if operational else {
        "ready": False, "status": "blocked", "as_of": None,
        "components": {}, "blockers": ["SERVICE_NOT_OPERATIONAL"], "warnings": [],
    }
    return {
        "ok": operational,
        "service_status": "ok" if operational else "error",
        "root": str(root), "today": date.today().isoformat(), "files": files,
        "env_file_present": env_path.exists(), "secrets_returned": False,
        "latest_factor": factor, "latest_decision": decision,
        "recommendation_readiness": readiness,
    }


def account_snapshot(root: Path | None = None) -> dict[str, Any]:
    """读取 holdings.json 并归一化为 account_state.v1(薄层:read_json + normalize)。

    MCP 默认不返回条件单 raw(MCP 面向用户,raw 仅内部 Python 调用暴露)。
    """
    root = (root or project_root()).resolve()
    holdings = read_json("data/holdings.json", root)
    if not isinstance(holdings, dict) or not isinstance(holdings.get("positions"), list):
        raise ValueError("holdings must be an object with a positions list")
    normalized = normalize_account_state(holdings, include_raw_orders=False)
    # MCP 兼容:外层字段保留 data_status/as_of/cash/market_value/total_assets 等历史名称
    return {
        "data_status": normalized["data_status"],
        "missing_fields": normalized["missing_fields"],
        "invalid_fields": normalized["invalid_fields"],
        "as_of": normalized["as_of"],
        "cash": normalized["cash"],
        "market_value": normalized["market_value"],
        "total_assets": normalized["total_assets"],
        "position_count": normalized["position_count"],
        "short_position": normalized["short_position"],
        "short_slot": normalized["short_slot"],
        "industry_weights": normalized["industry_weights"],
        "conditional_orders": normalized["conditional_orders"],
        "positions": normalized["positions"],
        "source": normalized["source"],
        # account_state.v1 新增字段供 MCP 消费者
        "schema": normalized["schema"],
        "source_schema": normalized["source_schema"],
        "freshness": normalized.get("freshness"),
        "warnings": normalized["warnings"],
    }


def strategy_context(root: Path | None = None) -> dict[str, Any]:
    """Return bounded policy context without dumping manual source files."""
    root = (root or project_root()).resolve()
    policy = read_json("data/trading_policy.json", root)
    profile = read_json("data/profile.json", root)
    factchecks = read_json("data/factcheck_overrides.json", root)
    trigger_bands = read_json("data/trigger_bands.json", root)
    if not all(isinstance(value, dict) for value in
               (policy, profile, factchecks, trigger_bands)):
        raise ValueError("strategy context files must contain JSON objects")
    overrides = factchecks.get("overrides")
    items = trigger_bands.get("items")
    retired = trigger_bands.get("retired")
    if not isinstance(overrides, list) or not isinstance(items, list):
        raise ValueError("factcheck overrides and trigger band items must be lists")
    if retired is not None and not isinstance(retired, list):
        raise ValueError("retired trigger bands must be a list when present")
    verdict_counts = {
        verdict: sum(isinstance(row, dict) and row.get("verdict") == verdict
                     for row in overrides)
        for verdict in ("clear", "red")
    }
    dated = [str(row.get("as_of")) for row in overrides
             if isinstance(row, dict) and row.get("as_of")]
    return {
        "policy": {key: policy.get(key) for key in (
            "policy_version", "target_positions", "target_weight",
            "industry_cap", "lot_size", "min_cash",
        )},
        "profile": {
            "as_of": profile.get("as_of"),
            "excluded_industries": profile.get("excluded_industries"),
        },
        "factchecks": {
            "override_count": len(overrides),
            "verdict_counts": verdict_counts,
            "latest_as_of": max(dated) if dated else None,
            "details_included": False,
        },
        "trigger_bands": {
            "active_count": len(items),
            "retired_count": len(retired or []),
            "details_included": False,
        },
    }


def _validate_decision_snapshot(snapshot: Any, path: Path) -> dict[str, Any]:
    if not isinstance(snapshot, dict):
        raise ValueError(f"decision snapshot must be an object: {path}")
    result = dict(snapshot)
    as_of = result.get("as_of")
    match = _DATED_DECISION.fullmatch(path.name)
    if not isinstance(as_of, str) or not _DATE8.fullmatch(as_of):
        raise ValueError(f"decision snapshot has invalid as_of: {path}")
    if match is None or match.group(1) != as_of:
        raise ValueError(f"decision filename/as_of mismatch: {path.name} vs {as_of}")
    require_decision_snapshot_ready(result, source=f"decision snapshot: {path}")
    decisions = result["decisions"]

    for index, decision in enumerate(decisions):
        state = decision.get("state")
        if not isinstance(decision.get("execution"), dict):
            raise ValueError(f"decision[{index}] has invalid execution: {path}")
        reasons = decision.get("reason_codes")
        evidence = decision.get("evidence")
        invalidations = decision.get("invalidations")
        execution = decision["execution"]
        if not isinstance(reasons, list) or not all(isinstance(x, str) for x in reasons):
            raise ValueError(f"decision[{index}] has invalid reason_codes: {path}")
        if not isinstance(evidence, dict):
            raise ValueError(f"decision[{index}] has invalid evidence: {path}")
        if not isinstance(invalidations, list):
            raise ValueError(f"decision[{index}] has invalid invalidations: {path}")
        shares = execution.get("shares")
        if (not isinstance(shares, int) or isinstance(shares, bool) or shares < 0):
            raise ValueError(f"decision[{index}] has invalid execution.shares: {path}")
        max_entry = execution.get("max_entry_price")
        if (max_entry is not None
                and (not isinstance(max_entry, (int, float)) or isinstance(max_entry, bool)
                     or not math.isfinite(float(max_entry)) or float(max_entry) <= 0)):
            raise ValueError(
                f"decision[{index}] has invalid execution.max_entry_price: {path}")
        if state == "BUY":
            if execution.get("eligible_from") != "NEXT_TRADING_DAY":
                raise ValueError(
                    f"decision[{index}] BUY has invalid eligible_from: {path}")
            decile = evidence.get("decile")
            # X-14:生产候选池 = 当期 D10(D10 码)∪ B8 带保留成员(B8_BAND 码,decile 8/9)
            in_pool = ((decile == 10 and "D10" in reasons)
                       or (decile in (8, 9) and "B8_BAND" in reasons))
            if not in_pool or "FACTCHECK_CLEAR" not in reasons:
                raise ValueError(
                    f"decision[{index}] BUY lacks D10/B8_BAND/FACTCHECK_CLEAR evidence: {path}")
    result["source_file"] = str(path)
    return result


def _decision_compact(decision: dict[str, Any]) -> dict[str, Any]:
    evidence = decision.get("evidence") or {}
    execution = decision.get("execution") or {}
    return {
        "ts_code": decision.get("ts_code"),
        "name": decision.get("name"),
        "state": decision.get("state"),
        "reason_codes": decision.get("reason_codes", []),
        "evidence": {key: evidence.get(key) for key in
                     ("score", "industry", "last", "decile", "size_bucket")
                     if key in evidence},
        "execution": {
            "max_entry_price": execution.get("max_entry_price"),
            "shares": execution.get("shares"),
        },
    }


def latest_decisions(
    root: Path | None = None,
    *,
    states: Sequence[str] | None = None,
    offset: int = 0,
    limit: int = 50,
    compact: bool = True,
    summary_only: bool = False,
) -> dict[str, Any]:
    root = (root or project_root()).resolve()
    if isinstance(offset, bool) or offset < 0:
        raise ValueError("offset must be >= 0")
    if isinstance(limit, bool) or not 1 <= limit <= 100:
        raise ValueError("limit must be 1..100")
    wanted: set[str] | None = None
    if states is not None:
        wanted = {str(state).upper() for state in states}
        if not wanted or not wanted <= _DECISION_STATES:
            raise ValueError(f"states must contain values from {sorted(_DECISION_STATES)}")

    path = latest_decision_path(root)
    snapshot = _validate_decision_snapshot(
        read_json(str(path.relative_to(root)), root), path
    )
    all_items = snapshot["decisions"]
    counts = {state: sum(item["state"] == state for item in all_items)
              for state in _STATE_ORDER}
    filtered = [item for item in all_items if wanted is None or item["state"] in wanted]
    page = [] if summary_only else filtered[offset:offset + limit]
    if compact:
        page = [_decision_compact(item) for item in page]
    returned = len(page)
    next_offset = offset + returned
    return {
        "as_of": snapshot["as_of"],
        "generated_at": snapshot.get("generated_at"),
        "data_status": snapshot["data_status"],
        "c2_state": dict(snapshot["c2_state"]),
        "c2_status": snapshot["c2_state"]["status"],
        "source_file": str(path.relative_to(root)),
        "summary": {"total": len(all_items), "state_counts": counts},
        "page": {
            "offset": offset, "limit": limit, "returned": returned,
            "filtered_total": len(filtered),
            "has_more": False if summary_only else next_offset < len(filtered),
            "next_offset": None if summary_only or next_offset >= len(filtered) else next_offset,
        },
        "compact": compact,
        "summary_only": summary_only,
        "decisions": page,
    }


def latest_decision_for_code(code: str, *, root: Path | None = None) -> dict[str, Any]:
    root = (root or project_root()).resolve()
    code = validate_codes([code], maximum=1)[0]
    path = latest_decision_path(root)
    snapshot = _validate_decision_snapshot(
        read_json(str(path.relative_to(root)), root), path
    )
    decision = next((d for d in snapshot["decisions"] if d["ts_code"] == code), None)
    return {
        "status": "found" if decision is not None else "missing",
        "ts_code": code,
        "snapshot_as_of": snapshot["as_of"],
        "source_file": str(path.relative_to(root)),
        "reason": None if decision is not None else "NOT_IN_LATEST_DECISION_SNAPSHOT",
        "decision": decision,
    }


def _actionable_view(lookup: dict[str, Any]) -> dict[str, Any]:
    decision = lookup.get("decision")
    if not isinstance(decision, dict):
        return {
            "machine_state": None,
            "user_action": "WAIT",
            "actionable": False,
            "reason": "MACHINE_DECISION_MISSING",
            "buy_range": None,
            "limit_price": None,
            "stop_price": None,
            "max_entry_price": None,
            "planned_shares": None,
        }

    state = decision["state"]
    execution = decision["execution"]
    max_entry_price = execution.get("max_entry_price")
    shares = execution.get("shares")
    if state != "BUY":
        reason = f"MACHINE_STATE_{state}"
        user_action = state
        actionable = state in {"HOLD", "EXIT"}
    elif max_entry_price is None:
        reason = "VERIFIED_ENTRY_PRICE_MISSING"
        user_action = "WAIT"
        actionable = False
    elif not shares:
        reason = "EXECUTABLE_SHARES_MISSING"
        user_action = "WAIT"
        actionable = False
    else:
        reason = None
        user_action = "BUY"
        actionable = True
    return {
        "machine_state": state,
        "user_action": user_action,
        "actionable": actionable,
        "reason": reason,
        "buy_range": None,
        "limit_price": max_entry_price if actionable else None,
        "stop_price": None,
        "max_entry_price": max_entry_price,
        "planned_shares": shares,
    }


def _factor_compact(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row.get(key) for key in (
        "ts_code", "name", "industry", "tier", "decile", "score", "last",
        "spec_crowd", "spike_limit", "poll_mark",
    ) if key in row}


def _latest_factor_snapshot(root: Path) -> tuple[Path, list[dict[str, Any]]]:
    path = latest_factor_path(root)
    rows = read_json(str(path.relative_to(root)), root)
    if not isinstance(rows, list):
        raise ValueError(f"factor snapshot must be a list: {path}")
    return path, rows


def latest_factor_for_code(code: str, *, root: Path | None = None) -> dict[str, Any]:
    root = (root or project_root()).resolve()
    code = validate_codes([code], maximum=1)[0]
    path, rows = _latest_factor_snapshot(root)
    row = next((item for item in rows if item.get("ts_code") == code), None)
    return {"as_of": path.name[:8], "source_file": str(path.relative_to(root)), "row": row}


def latest_factor_rows(
    *,
    deciles: Sequence[int] | None = None,
    states: Sequence[int] | None = None,
    offset: int = 0,
    limit: int = 20,
    compact: bool = True,
    root: Path | None = None,
) -> dict[str, Any]:
    root = (root or project_root()).resolve()
    if deciles is not None and states is not None:
        raise ValueError("use deciles, not both deciles and legacy states")
    if states is not None:
        deciles = states
    if isinstance(offset, bool) or offset < 0:
        raise ValueError("offset must be >= 0")
    if isinstance(limit, bool) or not 1 <= limit <= 100:
        raise ValueError("limit must be 1..100")
    wanted = None
    if deciles is not None:
        wanted = {int(value) for value in deciles}
        if not wanted or not wanted <= set(range(1, 11)):
            raise ValueError("deciles must contain values 1..10")

    path, rows = _latest_factor_snapshot(root)
    if wanted is not None:
        rows = [row for row in rows if row.get("decile") in wanted]
    rows.sort(key=lambda row: float(row.get("score") or float("-inf")), reverse=True)
    page_rows = rows[offset:offset + limit]
    if compact:
        page_rows = [_factor_compact(row) for row in page_rows]
    returned = len(page_rows)
    next_offset = offset + returned
    return {
        "as_of": path.name[:8], "source_file": str(path.relative_to(root)),
        "count": len(rows),
        "page": {"offset": offset, "limit": limit, "returned": returned,
                 "has_more": next_offset < len(rows),
                 "next_offset": next_offset if next_offset < len(rows) else None},
        "compact": compact,
        "rows": page_rows,
    }


def _market_session(now: datetime) -> str:
    local = now.astimezone(_SHANGHAI)
    if local.weekday() >= 5:
        return "non_trading_day"
    current = local.time()
    if current < time(9, 30):
        return "pre_open"
    if current <= time(11, 30):
        return "morning"
    if current < time(13, 0):
        return "lunch_break"
    if current <= time(15, 0):
        return "afternoon"
    return "closed"


def intraday_quotes(codes: Sequence[str], *, now: datetime | None = None) -> dict[str, Any]:
    clean = validate_codes(codes)
    requested = (now or datetime.now(_SHANGHAI)).astimezone(_SHANGHAI)
    quotes = fetch_quotes(clean)
    missing = [code for code in clean if code not in quotes]
    session = _market_session(requested)
    live_session = session in {"morning", "afternoon"}
    today = requested.strftime("%Y%m%d")
    is_intraday = live_session and not missing and len(quotes) == len(clean) and all(
        quote.get("timestamp_status") == "valid" and quote.get("trade_date") == today
        for quote in quotes.values()
    )
    quote_mode = "live_intraday" if is_intraday else "last_available_snapshot"
    requested_at = requested.isoformat(timespec="seconds")
    return {
        "requested_at": requested_at,
        "fetched_at": requested_at,
        "session": session,
        "quote_mode": quote_mode,
        "is_intraday": is_intraday,
        "source": "Tencent qt.gtimg.cn; latest available quote snapshot; never EOD research",
        "quotes": quotes,
        "missing": missing,
    }


def stock_brief(code: str, *, include_intraday: bool = True,
                root: Path | None = None) -> dict[str, Any]:
    root = (root or project_root()).resolve()
    code = validate_codes([code], maximum=1)[0]
    factor = latest_factor_for_code(code, root=root)
    row = factor["row"]
    holdings = account_snapshot(root)
    held = next((p for p in holdings["positions"] if p.get("ts_code") == code), None)
    machine_decision = latest_decision_for_code(code, root=root)
    decision_as_of = machine_decision["snapshot_as_of"]
    factchecks = [item for item in
                  read_json("data/factcheck_overrides.json", root).get("overrides", [])
                  if item.get("ts_code") == code]
    applicable = [item for item in factchecks if str(item.get("as_of") or "") <= decision_as_of]
    newer = [item for item in factchecks if str(item.get("as_of") or "") > decision_as_of]
    factcheck = max(applicable, key=lambda item: str(item.get("as_of") or "")) if applicable else None
    bands = read_json("data/trigger_bands.json", root).get("items", [])
    band = next((b for b in bands if b.get("ts_code") == code), None)
    actionable = _actionable_view(machine_decision)
    quote: dict[str, Any] | None = None
    quote_error: str | None = None
    if include_intraday:
        try:
            quote = intraday_quotes([code])
        except Exception as exc:  # Surface quote outage without hiding EOD facts.
            quote_error = f"{type(exc).__name__}: {str(exc)[:300]}"
    band_status = "missing" if band is None else ("suspended" if held is not None else "advisory")
    return {
        "ts_code": code,
        "factor_as_of": factor["as_of"],
        "factor": row,
        "holding": held,
        "factcheck": factcheck,
        "factcheck_alignment": {
            "status": "applicable" if factcheck is not None else
                      ("newer_ignored" if newer else "missing"),
            "ignored_newer": [{"as_of": item.get("as_of"),
                                "verdict": item.get("verdict")} for item in newer],
        },
        "machine_decision": machine_decision,
        "actionable_view": actionable,
        "snapshot_alignment": {
            "factor_matches_decision": factor["as_of"] == decision_as_of,
            "decision_as_of": decision_as_of,
        },
        "trigger_band": band,
        "trigger_band_status": band_status,
        "trigger_band_reason": "HELD_POSITION_SAME_SYMBOL" if band_status == "suspended" else None,
        "trigger_band_validity": "unverified" if band is not None else None,
        "trigger_band_can_change_decision": False,
        "intraday": quote,
        "intraday_error": quote_error,
    }


def _extract_backfill_report(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith(_BACKFILL_REPORT_PREFIX):
            try:
                value = json.loads(line[len(_BACKFILL_REPORT_PREFIX):])
            except json.JSONDecodeError:
                return None
            return value if isinstance(value, dict) else None
    return None


def refresh_eod(start: str, end: str, *, root: Path | None = None) -> dict[str, Any]:
    start, end = validate_date(start), validate_date(end)
    ds, de = datetime.strptime(start, "%Y%m%d"), datetime.strptime(end, "%Y%m%d")
    if ds > de:
        raise ValueError("start must be <= end")
    if (de - ds).days > 31:
        raise ValueError("one MCP refresh is limited to 31 calendar days")
    if de.date() > date.today():
        raise ValueError("end date cannot be in the future")
    result = run_module(
        "scripts.backfill",
        [start, end, "--strict-market", "--strict-env", "--report-json"],
        timeout=1800,
        root=root,
    )
    coverage = _extract_backfill_report(result.get("stdout", ""))
    contract_error: str | None = None
    if coverage is None:
        contract_error = "strict backfill did not return a valid coverage report"
    elif tuple(coverage.get("required_endpoints", ())) != _CORE_MARKET_ENDPOINTS:
        contract_error = "coverage report does not contain the four core market endpoints"
    elif coverage.get("calendar_status") != "complete":
        contract_error = "coverage report does not establish a complete trade calendar"
    elif coverage.get("completed_pairs") != coverage.get("expected_pairs"):
        contract_error = "coverage report is incomplete"
    elif coverage.get("failed_pairs") or coverage.get("fatal_error") or not coverage.get("ok"):
        contract_error = "strict backfill reported failures"
    result["coverage"] = coverage
    result["contract_error"] = contract_error
    result["ok"] = bool(result.get("ok")) and contract_error is None
    return result


def refresh_financials(codes: Sequence[str], *, root: Path | None = None) -> dict[str, Any]:
    """Force-refresh current structured fundamentals for at most ten symbols."""
    root = (root or project_root()).resolve()
    clean = validate_codes(codes, maximum=10)
    from ashare_gauntlet.config import CACHE_DIR, tushare_pro
    from ashare_gauntlet.data.fetch import fetch_symbol_table
    endpoints = ("income", "fina_indicator", "balancesheet", "cashflow",
                 "share_float", "pledge_stat", "pledge_detail", "fina_audit",
                 "stk_holdertrade", "namechange", "forecast", "express")
    pro = tushare_pro(env_path=root / ".env.local", strict_env=True)
    cache_dir = root / CACHE_DIR
    out: dict[str, Any] = {}
    for code in clean:
        per: dict[str, Any] = {}
        for endpoint in endpoints:
            try:
                frame = fetch_symbol_table(pro, endpoint, code, cache_dir, force=True)
                date_column = "ann_date" if endpoint == "pledge_detail" else "end_date"
                latest = (str(frame[date_column].dropna().astype(str).max())
                          if date_column in frame and not frame.empty else None)
                per[endpoint] = {"ok": True, "rows": len(frame), "latest": latest}
            except Exception as exc:
                per[endpoint] = {"ok": False,
                                 "error": f"{type(exc).__name__}: {str(exc)[:450]}"}
        out[code] = per
    return {"ok": all(x["ok"] for per in out.values() for x in per.values()),
            "codes": out}


def _maintenance_stage(
    name: str,
    module: str,
    args: Sequence[str],
    *,
    timeout: int,
    root: Path | None = None,
) -> dict[str, Any]:
    """Run exactly one named maintenance stage and return bounded evidence."""
    root = (root or project_root()).resolve()
    raw = run_module(module, args, timeout=timeout, root=root)
    result: dict[str, Any] = {
        "ok": raw["ok"],
        "stage": name,
        "module": raw["module"],
        "returncode": raw["returncode"],
    }
    if not raw["ok"]:
        result["stdout_tail"] = raw.get("stdout", "")[-2000:]
        result["stderr_tail"] = raw.get("stderr", "")[-2000:]
    return result


def generate_factor_snapshot(*, top: int = 20,
                             root: Path | None = None) -> dict[str, Any]:
    """Generate only the production factor snapshot from cached EOD."""
    if isinstance(top, bool) or not isinstance(top, int) or not 1 <= top <= 100:
        raise ValueError("top must be an integer in 1..100")
    return _maintenance_stage(
        "factor_rank", "scripts.factor_rank",
        ["--board", "main", "--top", str(top)], timeout=1200, root=root,
    )


def generate_machine_decisions(*, root: Path | None = None) -> dict[str, Any]:
    """Generate only the four-state machine decision snapshot."""
    result = _maintenance_stage(
        "buy_list", "scripts.buy_list", [], timeout=300, root=root,
    )
    if result["ok"]:
        root = (root or project_root()).resolve()
        result["decision_summary"] = latest_decisions(root, summary_only=True)
        result["recommendation_readiness"] = _recommendation_readiness(root)
    return result


def generate_personal_aggressive_view(*, top: int = 20,
                                      root: Path | None = None) -> dict[str, Any]:
    """Generate only the optional personal aggressive view; not a decision source."""
    if isinstance(top, bool) or not isinstance(top, int) or not 1 <= top <= 100:
        raise ValueError("top must be an integer in 1..100")
    return _maintenance_stage(
        "aggressive_pick", "scripts.aggressive_pick", ["--top", str(top)],
        timeout=300, root=root,
    )


def calculate_market_temperature(*, root: Path | None = None) -> dict[str, Any]:
    """Generate only the cached market-temperature artifact."""
    return _maintenance_stage(
        "market_temp", "scripts.market_temp", [], timeout=300, root=root,
    )


def generate_daily_analysis(*, top: int = 20, root: Path | None = None) -> dict[str, Any]:
    """Compatibility orchestration for direct Python callers; not exposed by MCP."""
    root = (root or project_root()).resolve()
    stages: dict[str, Any] = {}
    for name, fn, kwargs in (
        ("factor_rank", generate_factor_snapshot, {"top": top}),
        ("buy_list", generate_machine_decisions, {}),
        ("aggressive_pick", generate_personal_aggressive_view, {"top": top}),
        ("market_temp", calculate_market_temperature, {}),
    ):
        stage = fn(root=root, **kwargs)
        stages[name] = stage
        if name in {"factor_rank", "buy_list"} and not stage["ok"]:
            return {"ok": False, "status": "failed", "failed_stages": [name],
                    "stages": stages}

    failed_optional = [name for name in ("aggressive_pick", "market_temp")
                       if not stages[name]["ok"]]
    account = account_snapshot(root)
    return {
        "ok": not failed_optional,
        "status": "completed" if not failed_optional else "partial",
        "failed_stages": failed_optional,
        "stages": stages,
        "decision_summary": latest_decisions(root, summary_only=True),
        "account_summary": {
            "data_status": account["data_status"], "as_of": account["as_of"],
            "total_assets": account["total_assets"], "short_slot": account["short_slot"],
            "conditional_orders": {"status": account["conditional_orders"]["status"]},
        },
        "recommendation_readiness": _recommendation_readiness(root),
    }


_GOVERNANCE_COLUMNS = {
    "pledge_detail": frozenset({
        "ann_date", "holder_name", "pledged_amount", "holding_amount", "h_total_ratio",
    }),
    "fina_audit": frozenset({"end_date", "audit_result"}),
}


def _local_governance_table(root: Path, endpoint: str,
                            code: str) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """Read one governance parquet without fetching, writing, or creating paths."""
    path = root / "data" / "cache" / endpoint / f"{code}.parquet"
    relative = path.relative_to(root).as_posix()
    coverage: dict[str, Any] = {
        "status": "uncovered", "endpoint": endpoint, "path": relative,
        "rows": None, "reason": None, "missing_columns": [],
    }
    if not path.is_file():
        coverage["reason"] = "CACHE_MISSING"
        return None, coverage
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        coverage["reason"] = "CACHE_READ_ERROR"
        coverage["error"] = f"{type(exc).__name__}: {str(exc)[:450]}"
        return None, coverage
    coverage["rows"] = len(frame)
    if frame.empty:
        coverage["reason"] = "EMPTY_TABLE"
        return None, coverage
    missing = sorted(_GOVERNANCE_COLUMNS[endpoint] - set(frame.columns))
    if missing:
        coverage["reason"] = "SCHEMA_INCOMPLETE"
        coverage["missing_columns"] = missing
        return None, coverage
    coverage.update({"status": "covered", "reason": None})
    return frame, coverage


def governance_check(codes: Sequence[str], *, force: bool = False,
                     root: Path | None = None) -> dict[str, Any]:
    """Check local governance caches only; never fetch, spawn, or write."""
    if force:
        raise ValueError(
            "check_governance never refreshes data; call refresh_stock_financials first"
        )
    clean = validate_codes(codes, maximum=20)
    root = (root or project_root()).resolve()
    results: dict[str, Any] = {}
    covered_tables = 0
    incomplete_codes: list[str] = []

    for code in clean:
        pledge, pledge_coverage = _local_governance_table(root, "pledge_detail", code)
        audit, audit_coverage = _local_governance_table(root, "fina_audit", code)
        pledge_result: dict[str, Any] = {
            "coverage": pledge_coverage,
            "status": "uncovered",
            "controller_pledge": None,
        }
        audit_result: dict[str, Any] = {
            "coverage": audit_coverage,
            "status": "uncovered",
            "opinion": None,
        }

        if pledge is not None:
            try:
                top = controller_pledge(pledge)
            except Exception as exc:
                pledge_coverage.update({
                    "status": "uncovered", "reason": "ANALYSIS_ERROR",
                    "error": f"{type(exc).__name__}: {str(exc)[:450]}",
                })
            else:
                pledge_result["status"] = "found" if top else "none_current"
                pledge_result["controller_pledge"] = top or None
                covered_tables += 1

        if audit is not None:
            try:
                opinion = audit_opinion(audit)
            except Exception as exc:
                audit_coverage.update({
                    "status": "uncovered", "reason": "ANALYSIS_ERROR",
                    "error": f"{type(exc).__name__}: {str(exc)[:450]}",
                })
            else:
                if opinion:
                    audit_result.update({"status": "found", "opinion": opinion})
                    covered_tables += 1
                else:
                    audit_coverage.update({"status": "uncovered", "reason": "ANALYSIS_ERROR"})

        incomplete = (pledge_coverage["status"] != "covered"
                      or audit_coverage["status"] != "covered")
        if incomplete:
            incomplete_codes.append(code)
        results[code] = {
            "data_status": "incomplete" if incomplete else "complete",
            "incomplete": incomplete,
            "pledge": pledge_result,
            "audit": audit_result,
            "warnings": (["UNCOVERED_DOES_NOT_MEAN_LOW_RISK"] if incomplete else []),
        }

    expected_tables = len(clean) * 2
    incomplete = bool(incomplete_codes)
    return {
        "ok": not incomplete,
        "data_status": "incomplete" if incomplete else "complete",
        "incomplete": incomplete,
        "source": "local_parquet",
        "network_access": False,
        "codes": results,
        "coverage": {
            "requested_codes": len(clean),
            "complete_codes": len(clean) - len(incomplete_codes),
            "incomplete_codes": incomplete_codes,
            "covered_tables": covered_tables,
            "expected_tables": expected_tables,
        },
    }
