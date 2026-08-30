"""每日一屏人类简报(只读聚合器):把散落命令收敛成"今天该干嘛"的一屏。

设计边界(严守 CLAUDE.md):
- **纯只读**。不改任何账户/决策/研究状态,不下单,不写 factcheck verdict,不联网。
- **机器状态逐字来自冻结决策快照**(经 mcp_service._validate_decision_snapshot 校验),
  本层绝不重算 BUY/WAIT/EXIT,也绝不把 WAIT 提升为 BUY。
- **股息只是展示叠加**(dv_ttm):不进 composite、不改机器状态、不当买卖信号。
- **factcheck 到期预警只是提示**:读 overrides 的 expires_on 提前 7 天亮灯,
  到期与否的判定权仍在 candidates.override_status/buy_list,本层不写不续期。
- **fail-loud**:四核心 EOD 缺失/错位、或决策快照非法 → 退出码 1,绝不显示"今日无事";
  辅助数据(股息)不可用 → 标 UNAVAILABLE,不当 0/无分红。
- 决策/持仓风险数字全部读已有机器产物(冻结快照 + holdings_watch 的 account_state 快照),
  本脚本只聚合与呈现,不做任何价格/估值计算。

退出码:0=平静(无待办);2=有需人工处理事项;1=数据/快照失败(系统不可信)。

Usage: E:\\zwyq\\.venv\\Scripts\\python.exe -m scripts.daily_brief [--json]
(Windows 控制台中文输出建议前缀 PYTHONIOENCODING=utf-8)
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ashare_gauntlet import mcp_service as svc
from ashare_gauntlet.config import ACCOUNT_STATE_DIR, CACHE_DIR
from ashare_gauntlet.dividends import (
    DividendDataDegraded,
    DividendDataUnavailable,
    dividend_yields,
    indicative_ttm_cash,
)
from ashare_gauntlet.account_state import normalize_account_state, normalize_bucket
from ashare_gauntlet.candidates import HARD_VETO_CODES
from ashare_gauntlet.freshness import classify_cache_freshness
from ashare_gauntlet.stop_policy import (
    check_positions,
    check_time_stops,
    conditional_order_coverage,
    needs_attention,
)
_C2_CODE = "EXIT_RULE_C2_MONTHLY"
_C2_DEFAULT = {
    "status": "NOT_INITIALIZED",
    "last_valid_review_as_of": None,
    "watch": [],
    "exit_eligible": [],
    "error": None,
}
# 判据单一来源=candidates.HARD_VETO_CODES(含 SPEC_CROWD/SPIKE_LIMIT 等 clear 也解不开的码)
_PENDING_BLOCK_CODES = HARD_VETO_CODES
_ACCT_STATE_RE = re.compile(r"^(\d{8})_account_state\.json$")
_SHANGHAI = timezone(timedelta(hours=8))
_PROFIT_TAKE_MULT = 1.25  # 长线 +25% 减半锁利提示线(展示,与 intraday PROFIT 档同源常数)

# 诚实语境(P2:低波价值取舍),全部来自 docs/methodology.md 既有事实,非信号
_ADVISORY = (
    "组合超额薄且集中在跌市(§10:PROD NW t≈2.91@N=139),近年低波价值风格逆风"
    "——对单只 BUY 别过度自信。",
    "前瞻验证样本未成熟(§10.1 X-09 insufficient_sample),尚无 OOS 证据证明"
    "决策链跑赢 D10 等权。",
    "低波价值=慢与闷是因子取舍不是故障;股息叠加(dv_ttm)只是展示,"
    "不承诺收益、不进机器分。",
    "小账户实得受 5 元佣金地板拖累,回测比率口径会高估净收益(小仓位尤甚)。",
)


def _profit_take_line(bucket, cost) -> float | None:
    """+25% 减半锁利线**只属长线仓**(与 intraday PROFIT 档同判据)。

    对短线仓打印此线会诱导超期持有——短线的纪律是 10 交易日时间止损 + -7% 硬止损
    (跨层审计:此前对所有仓别无差别打印,与盘中哨兵各说各话)。
    """
    if normalize_bucket(bucket) != "long":
        return None
    if isinstance(cost, bool) or not isinstance(cost, (int, float)):
        return None
    return round(float(cost) * _PROFIT_TAKE_MULT, 3)


_GATE_STALE_DAYS = 100   # 季度节律(约 1 个季度 + 缓冲);仅决定"要不要提醒复核",非研究门槛


def _gate_baseline_age(root: Path, now: datetime) -> dict:
    """门禁证据基线的年龄(只读;缺基线不当作健康)。"""
    path = root / "data/holdscore/gate_baseline.json"
    if not path.exists():
        return {"status": "MISSING", "days": None,
                "detail": "尚未冻结门禁基线——跑 scripts.gate_check --freeze 建立"}
    try:
        frozen = json.loads(path.read_text(encoding="utf-8")).get("frozen_at")
        stamp = datetime.fromisoformat(str(frozen))
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        return {"status": "MISSING", "days": None, "detail": "门禁基线不可读/无 frozen_at"}
    days = (now - stamp).days
    stale = days >= _GATE_STALE_DAYS
    return {"status": "STALE" if stale else "FRESH", "days": days,
            "detail": (f"门禁证据基线已 {days} 天未复核(≥{_GATE_STALE_DAYS} 天)"
                       "——按季复跑 factor_backtest/composite_backtest 后跑 gate_check"
                       if stale else f"门禁证据基线 {days} 天前冻结")}


_FACTCHECK_HORIZON_DAYS = 7   # 提示窗口(日历日):先于过期暴露,避免到期静默回退 WAIT


def _factcheck_expiry(root: Path, relevant: dict[str, str], ref_as_of: "str | None",
                      now: datetime) -> dict:
    """即将过期的 clear factcheck 覆盖(只读提示,不写 verdict、不重算状态)。

    relevant: ts_code → 展示名,限定在"过期才有后果"的集合(持仓/BUY/待核/C2 观察)
    ——无关股票的旧覆盖过期不产生噪音。到期判定复用 candidates.override_status
    同一契约;窗口 = 决策日(缺省今天)起 N 个日历日内到期的**仍有效 clear**。
    覆盖文件非法 → INVALID 如实上报并出待办(不当"无覆盖",也不吞错)。
    """
    from scripts.buy_list import load_overrides
    from ashare_gauntlet.candidates import override_status

    ref = ref_as_of or now.strftime("%Y%m%d")
    ref_dt = datetime.strptime(ref, "%Y%m%d")
    horizon = (ref_dt + timedelta(days=_FACTCHECK_HORIZON_DAYS)).strftime("%Y%m%d")
    try:
        overrides = load_overrides(str(root / "data/factcheck_overrides.json"))
    except ValueError as exc:   # JSONDecodeError 是 ValueError 子类;行级契约违规同报
        return {"status": "INVALID", "horizon_days": _FACTCHECK_HORIZON_DAYS,
                "expiring": [], "error": str(exc)}
    except OSError as exc:      # 与 _load_snapshot 同款:IO 异常降级为段内错误,不炸整份简报
        return {"status": "INVALID", "horizon_days": _FACTCHECK_HORIZON_DAYS,
                "expiring": [], "error": f"{type(exc).__name__}: {exc}"}
    expiring: list[dict] = []
    for code, ov in sorted(overrides.items()):
        if code not in relevant or override_status(ov, ref) != "clear":
            continue
        expires_on = str(ov["expires_on"])
        if ref <= expires_on <= horizon:
            days = (datetime.strptime(expires_on, "%Y%m%d") - ref_dt).days
            expiring.append({"ts_code": code, "name": relevant.get(code, code),
                             "expires_on": expires_on, "days_left": days})
    return {"status": "OK", "horizon_days": _FACTCHECK_HORIZON_DAYS, "expiring": expiring}


def _latest_account_state(root: Path) -> dict | None:
    """读 holdings_watch 产出的最新 account_state 快照(只读;缺失/损坏如实标注)。"""
    directory = root / ACCOUNT_STATE_DIR
    if not directory.is_dir():
        return None
    files = sorted(p for p in directory.iterdir() if _ACCT_STATE_RE.match(p.name))
    if not files:
        return None
    path = files[-1]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"path": str(path.relative_to(root)), "error": f"{type(exc).__name__}: {exc}"}
    return {
        "path": str(path.relative_to(root)),
        "as_of": data.get("as_of"),
        "data_status": data.get("data_status"),
        "valuation": data.get("valuation"),
        "positions": data.get("positions") if isinstance(data.get("positions"), list) else [],
    }


def _load_snapshot(root: Path) -> tuple[str, dict | None, str | None]:
    """读并校验最新冻结决策快照。返回 (status, snapshot, error)。

    status ∈ {ok, missing, invalid}:missing=尚未生成(可跑 eod_ops);invalid=契约破坏(失败)。
    """
    try:
        path = svc.latest_decision_path(root)
    except FileNotFoundError:
        return "missing", None, None
    try:
        raw = svc.read_json(str(path.relative_to(root)), root)
        if not isinstance(raw, dict):
            raise ValueError("decision snapshot must be an object")

        # Pre-C2 snapshots are still useful for daily operations. Treat the
        # absent sidecar as an explicit migration state, while malformed or
        # present-but-unavailable C2 metadata remains visible as degraded.
        if "c2_state" not in raw:
            raw = {**raw, "c2_state": dict(_C2_DEFAULT)}
        c2_state = raw.get("c2_state")
        c2_unavailable = isinstance(c2_state, dict) and c2_state.get("status") == "UNAVAILABLE"
        if c2_unavailable:
            if raw.get("data_status") not in {"complete", "degraded"}:
                raise ValueError("decision snapshot has invalid data_status")
            if (
                set(c2_state) != set(_C2_DEFAULT)
                or c2_state.get("last_valid_review_as_of") is not None
                or c2_state.get("watch") != []
                or c2_state.get("exit_eligible") != []
                or not isinstance(c2_state.get("error"), str)
                or not c2_state["error"]
            ):
                raise ValueError("c2_state UNAVAILABLE projection is malformed")
            # The shared ready-validator intentionally rejects UNAVAILABLE.
            # Validate the rest of the snapshot through that same contract by
            # substituting the neutral NOT_INITIALIZED projection temporarily.
            validation_snapshot = {
                **raw,
                "c2_state": dict(_C2_DEFAULT),
                "data_status": "complete",
            }
            svc._validate_decision_snapshot(validation_snapshot, path)
            snapshot = raw
        else:
            snapshot = svc._validate_decision_snapshot(raw, path)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        return "invalid", None, f"{type(exc).__name__}: {exc}"
    return "ok", snapshot, None


def _classify(decisions: list[dict]) -> dict:
    """把冻结决策分桶(逐字读 state/reason_codes,不重算)。"""
    buys, exits, c2, pending = [], [], [], []
    counts = {state: 0 for state in ("BUY", "WAIT", "HOLD", "EXIT")}
    for d in decisions:
        state = d.get("state")
        codes = set(d.get("reason_codes", []))
        if state in counts:
            counts[state] += 1
        if state == "BUY":
            buys.append(d)
        elif state == "EXIT":
            exits.append(d)
        if _C2_CODE in codes:
            # C2 是观察集合不是待办:跌出 D10 只在**有效月度审视日**记 1 期,
            # 日频快照不含 streak(消费方见下方 c2_watch)
            c2.append(d)
        if state == "WAIT" and "FACTCHECK_REQUIRED" in codes and not (codes & _PENDING_BLOCK_CODES):
            pending.append(d)
    return {"counts": counts, "buys": buys, "exits": exits, "c2": c2, "pending": pending}


def _buy_view(decision: dict) -> dict:
    """机器 BUY 行 → 用户动作视图(复用 _actionable_view:缺价/缺股数强制 WAIT)。"""
    view = svc._actionable_view({"decision": decision})
    return {
        "ts_code": decision.get("ts_code"),
        "name": decision.get("name"),
        "user_action": view["user_action"],
        "actionable": view["actionable"],
        "reason": view["reason"],
        "max_entry_price": view["max_entry_price"],
        "planned_shares": view["planned_shares"],
        "reason_codes": decision.get("reason_codes", []),
    }


def _compact_decision(decision: dict) -> dict:
    ev = decision.get("evidence") or {}
    return {
        "ts_code": decision.get("ts_code"),
        "name": decision.get("name"),
        "state": decision.get("state"),
        "reason_codes": decision.get("reason_codes", []),
        "decile": ev.get("decile"),
        "score": ev.get("score"),
    }


def _c2_member_rows(codes: list[str], decisions: list[dict]) -> list[dict]:
    """Project durable C2 codes onto decision rows without inventing signals."""
    by_code = {str(d.get("ts_code")): d for d in decisions}
    rows: list[dict] = []
    for code in codes:
        decision = by_code.get(str(code))
        if decision is not None:
            rows.append(_compact_decision(decision))
        else:
            # A durable WATCH may outlive the current daily decision list.
            rows.append({
                "ts_code": str(code), "name": str(code), "state": "WATCH",
                "reason_codes": [_C2_CODE], "decile": None, "score": None,
            })
    return rows


def _c2_watch_view(c2_state: dict, decisions: list[dict], _legacy_c2: list[dict]) -> dict:
    status = c2_state.get("status")
    watch_codes = list(c2_state.get("watch") or [])
    # A legacy daily observation is not durable C2 state. Until the sidecar is
    # initialized, do not project it as WATCH membership.

    last_valid = c2_state.get("last_valid_review_as_of")
    error = c2_state.get("error")
    if status == "REVIEW_BLOCKED_DATA":
        reason = (f"C2 月度审视数据不可用({error});上次有效审视={last_valid or '—'};"
                  "本次 streak 未推进")
    elif status == "UNAVAILABLE":
        reason = f"C2 数据不可用({error or 'unknown'});不能据此判断无退出"
    elif status == "NOT_INITIALIZED":
        reason = "C2 尚未初始化(迁移/尚未运行 scripts.c2_review)"
    elif watch_codes:
        reason = "C2 WATCH 仅作信息展示;需连续 2 个有效月度审视确认退出"
    else:
        reason = "C2 已可用;当前无 WATCH"
    return {
        "status": status,
        "reason": reason,
        "error": error,
        "last_valid_review_as_of": last_valid,
        "watch": watch_codes,
        "exit_eligible": list(c2_state.get("exit_eligible") or []),
        "members": _c2_member_rows(watch_codes, decisions),
    }


def _attach_dividends(codes, as_of, cache_dir):
    """取股息叠加;分区不可用 → UNAVAILABLE,股息列整列 NULL → DEGRADED(不伪造无分红)。"""
    try:
        table = dividend_yields(codes, as_of, cache_dir=cache_dir)
        return {"status": "OK", "yields": table}
    except DividendDataDegraded as exc:
        return {"status": "DEGRADED", "reason": str(exc), "yields": {}}
    except DividendDataUnavailable as exc:
        return {"status": "UNAVAILABLE", "reason": str(exc), "yields": {}}


def build_brief(root: Path | None = None, *, now: datetime | None = None,
                cache_dir: str | None = None) -> dict:
    """组装一屏简报(只读)。返回结构化 dict,含 exit_code。"""
    root = (root or svc.project_root()).resolve()
    now = now or datetime.now(_SHANGHAI)

    health = svc.healthcheck(root)
    readiness = health.get("recommendation_readiness", {})
    blockers = list(readiness.get("blockers", []))
    warnings = list(readiness.get("warnings", []))
    components = readiness.get("components", {})
    eod_comp = components.get("eod", {})
    holdings_comp = components.get("holdings", {})
    as_of = readiness.get("as_of")

    snap_status, snapshot, snap_error = _load_snapshot(root)
    decisions = snapshot["decisions"] if snapshot else []
    decision_as_of = snapshot.get("as_of") if snapshot else None
    buckets = _classify(decisions)

    account = None
    account_error = None
    try:
        account = svc.account_snapshot(root)
    except (FileNotFoundError, ValueError, TypeError, json.JSONDecodeError) as exc:
        account_error = f"{type(exc).__name__}: {exc}"

    held_codes = [str(p.get("ts_code")) for p in (account or {}).get("positions", [])
                  if p.get("ts_code")]

    # 股息叠加锚定决策日(对齐日 = daily_basic 分区存在);错位时可能 UNAVAILABLE
    div_as_of = decision_as_of or as_of
    div_codes = list(dict.fromkeys(
        held_codes
        + [str(d.get("ts_code")) for d in buckets["buys"] if d.get("ts_code")]
        + [str(d.get("ts_code")) for d in buckets["pending"] if d.get("ts_code")]
    ))
    resolved_cache = cache_dir if cache_dir is not None else str(root / CACHE_DIR)
    dividends = {"status": "SKIPPED", "yields": {}}
    if div_as_of and div_codes:
        dividends = _attach_dividends(div_codes, div_as_of, resolved_cache)
    div_yields = dividends.get("yields", {})

    # 持仓风险:优先用 holdings_watch 的 account_state 快照(含 close/pnl/距止损/MA20/held_days)
    acct_state = _latest_account_state(root)
    acct_state_fresh = bool(
        acct_state and not acct_state.get("error")
        and acct_state.get("as_of") == decision_as_of
        and acct_state.get("data_status") == "complete"   # 缺 data_status=未知,不当新鲜(复核 P2)
    )
    holdings_risk = []
    if acct_state_fresh:
        for rec in (acct_state or {}).get("positions", []):
            code = str(rec.get("ts_code"))
            cost = rec.get("cost")
            close = rec.get("close")
            shares = rec.get("shares")
            mv = (float(shares) * float(close)) if (
                isinstance(shares, (int, float)) and not isinstance(shares, bool)
                and isinstance(close, (int, float)) and not isinstance(close, bool)) else None
            dv = div_yields.get(code, {}).get("dv_ttm")
            holdings_risk.append({
                "ts_code": code, "name": rec.get("name"), "bucket": rec.get("bucket"),
                "shares": shares, "cost": cost, "stop": rec.get("stop"),
                "close": close, "pnl_pct": rec.get("pnl_pct"),
                "dist_stop_pct": rec.get("dist_stop_pct"),
                "ma20": rec.get("ma20"), "held_days": rec.get("held_days"),
                "profit_take_line": _profit_take_line(rec.get("bucket"), cost),
                "dv_ttm": dv,
                "indicative_ttm_cash": indicative_ttm_cash(dv, mv),
                "error": rec.get("error"),
            })
    else:
        for p in (account or {}).get("positions", []):
            code = str(p.get("ts_code"))
            cost = p.get("cost")
            dv = div_yields.get(code, {}).get("dv_ttm")
            holdings_risk.append({
                "ts_code": code, "name": p.get("name"), "bucket": p.get("bucket"),
                "shares": p.get("shares"), "cost": cost, "stop": p.get("stop"),
                "profit_take_line": _profit_take_line(p.get("bucket"), cost),
                "dv_ttm": dv, "risk_numbers": "STALE_OR_MISSING",
            })

    # 止损政策一致性(只读 surface,绝不改 stop):裸窗口/写反/带外都要人工处理
    stop_checks = check_positions((account or {}).get("positions", []))
    stop_alerts = needs_attention(stop_checks)
    # 短线仓时间止损(10 交易日制度窗口):消费 account_state 快照已算好的 held_days。
    # **"未核查"必须与"已查无命中"可区分**——前置数据陈旧时返回空列表会被读成"没有
    # 超期仓",而 held_days 根本没算过(跨层审计 P1;与 commit d10f2c3"未核查显式化"同精神)
    if acct_state_fresh:
        time_stop_state = {"status": "CHECKED", "reason": None,
                           "hits": check_time_stops(holdings_risk)}
    else:
        time_stop_state = {"status": "NOT_CHECKED", "hits": [],
                           "reason": "account_state 快照缺失/陈旧(held_days 未算)"
                                     "——先跑 holdings_watch"}
    time_stops = time_stop_state["hits"]
    # EOD 缓存挂钟新鲜度:防"决策建在上周价格上"的静默降级
    cache_fresh = classify_cache_freshness(eod_comp.get("as_of"), now.strftime("%Y%m%d"))
    # 破线保护覆盖:盘中哨兵已停用(2026-08-19 拍板),条件单是唯一防线且系统无法
    # 验证券商端——必须如实显示"未确认/未覆盖",不能沉默着看起来安全
    # 覆盖判定需要订单明细,而 account_snapshot 按 MCP 约定隐藏 raw orders(codex P1);
    # 这里就地用 include_raw_orders=True 重新归一一次(纯读、不改文件)。取不到就如实
    # 走 NO_DETAIL,绝不因拿不到明细而谎称已覆盖。
    account_with_orders = account
    try:
        raw_holdings = svc.read_json("data/holdings.json", root)
        if isinstance(raw_holdings, dict):
            account_with_orders = normalize_account_state(raw_holdings, include_raw_orders=True)
    except (FileNotFoundError, ValueError, TypeError, json.JSONDecodeError):
        pass
    protection = conditional_order_coverage(account_with_orders or {})
    # 门禁证据年龄:准入证据(五门/组合 t)是一次性复跑的结论,不在 eod_ops 里,
    # 会静默变旧。这里只读基线的冻结时间,提醒按季复核(gate_check),不重算。
    gate_age = _gate_baseline_age(root, now)

    # 机器行视图(计算一次、复用);股息只作展示叠加挂上,不改机器状态
    def _dv(code) -> float | None:
        return div_yields.get(str(code), {}).get("dv_ttm")

    buys_view = []
    for d in buckets["buys"]:
        view = _buy_view(d)
        view["dv_ttm"] = _dv(d.get("ts_code"))
        buys_view.append(view)
    exits_view = [_compact_decision(d) for d in buckets["exits"]]
    c2_state = snapshot.get("c2_state", _C2_DEFAULT) if snapshot else dict(_C2_DEFAULT)
    c2_watch = _c2_watch_view(c2_state, decisions, buckets["c2"])
    pending_view = [{**_compact_decision(d), "dv_ttm": _dv(d.get("ts_code"))}
                    for d in buckets["pending"]]

    # factcheck 到期预警的相关集合 = 持仓 + BUY + 待核 + C2 观察(过期才改变这些票的状态)
    factcheck_relevant: dict[str, str] = {}
    for p in (account or {}).get("positions", []):
        if p.get("ts_code"):
            factcheck_relevant[str(p["ts_code"])] = str(p.get("name") or p["ts_code"])
    for d in [*buckets["buys"], *buckets["pending"]]:
        factcheck_relevant[str(d.get("ts_code"))] = str(d.get("name") or d.get("ts_code"))
    for row in c2_watch.get("members", []):
        factcheck_relevant[str(row.get("ts_code"))] = str(row.get("name") or row.get("ts_code"))
    factcheck_expiry = _factcheck_expiry(root, factcheck_relevant, decision_as_of, now)

    # —— next-actions(有序;每条指向"人跑的命令",从不代跑代判)——
    actions: list[str] = []
    holdings_fresh = holdings_comp.get("freshness")
    if holdings_fresh and holdings_fresh != "aligned" and decision_as_of:
        actions.append(f"① 账户 as_of={holdings_comp.get('as_of')} 未对齐决策日 {decision_as_of}"
                       f"——跑 holdings_confirm {decision_as_of} 推进账户日期")
    if snap_status == "missing":
        actions.append("② 尚无决策快照——跑 eod_ops(或 buy_list)生成机器决策")
    if buckets["exits"]:
        names = ",".join(str(d.get("ts_code")) for d in buckets["exits"])
        actions.append(f"③ EXIT 信号 {len(buckets['exits'])} 只({names})"
                       "——人工决定是否卖出;成交后 trade_record --sell")
    # 此处不设 C2 待办(编号 ④ 留空):C2 是跨期规则,日频没有"今天该审视"这回事
    # ——逐日催办既造成警报疲劳,又会诱导用户当场卖成"立即退出"(见上方 c2_watch)
    actionable_buys = [b for b in buys_view if b["actionable"]]
    if actionable_buys:
        names = ",".join(str(b["ts_code"]) for b in actionable_buys)
        actions.append(f"⑤ 机器 BUY {len(actionable_buys)} 只({names})"
                       "——复核 entry_check;人工终判并下单后 trade_record --buy")
    if buckets["pending"]:
        names = ",".join(str(d.get("ts_code")) for d in buckets["pending"])
        actions.append(f"⑥ 唯一未决项=fact-check 的候选 {len(buckets['pending'])} 只({names})"
                       "——读 probe 报告 + factcheck skill,人工写 override(仍是 WAIT)")
    if not acct_state_fresh and held_codes:
        actions.append("⑦ 持仓风险数字缺/陈旧——跑 holdings_watch 生成 EOD 估值")
    if cache_fresh["status"] in ("STALE", "MISSING"):
        actions.append(f"⓪ EOD 缓存新鲜度 {cache_fresh['status']}——{cache_fresh['detail']}")
    due = [r for r in time_stops if r.get("status") == "TIME_STOP"]
    unknown_age = [r for r in time_stops if r.get("status") == "TIME_STOP_UNKNOWN"]
    if due:
        names = ",".join(f"{r['ts_code']}({r['held_days']}日)" for r in due)
        actions.append(f"⑩ 短线仓达时间止损窗口 {len(due)} 只({names})"
                       "——按双仓制审视是否了结(人工终判)")
    if unknown_age:
        names = ",".join(r["ts_code"] for r in unknown_age)
        actions.append(f"⑩b 短线仓持有日龄未知 {len(unknown_age)} 只({names})"
                       "——时间止损无法判定(多半缺 entry_date),补齐后再看")
    if gate_age["status"] in ("STALE", "MISSING"):
        actions.append(f"⑫ 门禁证据{gate_age['status']}——{gate_age['detail']}")
    if factcheck_expiry["status"] == "INVALID":
        actions.append(f"⑬ factcheck 覆盖文件非法——{factcheck_expiry['error']}"
                       "(buy_list 会整场失败;修复后再荐股)")
    elif factcheck_expiry["expiring"]:
        names = ",".join(f"{r['name']}({r['ts_code']})还{r['days_left']}天"
                         for r in factcheck_expiry["expiring"])
        actions.append(f"⑬ factcheck 即将过期 {len(factcheck_expiry['expiring'])} 只({names})"
                       "——到期自动回 WAIT;要保留资格的先重核(人工写 override)")
    if protection["status"] in ("UNVERIFIED", "INVALID", "NO_DETAIL") or protection["uncovered"]:
        actions.append(f"⑪ 破线保护未确认({protection['status']})——{protection['note']}")
    missing_stop = [r for r in stop_alerts if r["status"] == "MISSING_STOP"]
    if missing_stop:
        names = ",".join(r["ts_code"] for r in missing_stop)
        actions.append(f"⑧ 止损缺失 {len(missing_stop)} 只({names})——哨兵对其跳过 "
                       "BREACH/NEAR(无止损警报),人工补 stop 价")
    other_stop = [r for r in stop_alerts if r["status"] != "MISSING_STOP"]
    if other_stop:
        names = ",".join(f"{r['ts_code']}({r['status']})" for r in other_stop)
        actions.append(f"⑨ 止损与双仓制政策不符 {len(other_stop)} 只({names})"
                       "——人工核对是否抄错/写反(工具只提示不改)")
    # readiness 其余 blocker(数据/条件单等)如实列出
    _surfaced = {
        "ACCOUNT_STATE_INCOMPLETE",
        "DECISION_NOT_ALIGNED",
        # The dedicated C2 view already explains these states; do not turn a
        # monthly review problem into a duplicate daily action.
        "C2_REVIEW_BLOCKED_DATA",
        "C2_REVIEW_UNAVAILABLE",
    }
    for b in blockers:
        if b not in _surfaced and b != "CORE_EOD_MISSING_OR_MISALIGNED":
            actions.append(f"• readiness blocker: {b}(须人工处理后方可正式荐股)")

    # —— 退出码 ——
    system_failed = (
        (readiness.get("ready") is False and bool(blockers))
        or snap_status == "invalid"
        or c2_state.get("status") in {"UNAVAILABLE", "REVIEW_BLOCKED_DATA"}
    )
    if system_failed:
        exit_code = 1
    elif actions:
        exit_code = 2
    else:
        exit_code = 0

    return {
        "schema_version": "daily_brief.v1",
        "generated_at": now.isoformat(),
        "as_of": as_of,
        "decision_as_of": decision_as_of,
        "readiness": {
            "ready": readiness.get("ready"),
            "status": readiness.get("status"),
            "blockers": blockers,
            "warnings": warnings,
            "eod": {"status": eod_comp.get("status"), "as_of": eod_comp.get("as_of"),
                    "endpoint_dates": eod_comp.get("endpoint_dates")},
        },
        "decision_snapshot": {
            "status": snap_status,
            "error": snap_error,
            "source_file": snapshot.get("source_file") if snapshot else None,
            "state_counts": buckets["counts"],
        },
        "machine": {
            "buys": buys_view,
            "exits": exits_view,
            "c2_watch": c2_watch,
            "pending_factcheck": pending_view,
        },
        "account": {
            "error": account_error,
            "as_of": (account or {}).get("as_of"),
            "data_status": (account or {}).get("data_status"),
            "cash": (account or {}).get("cash"),
            "total_assets": (account or {}).get("total_assets"),
            "position_count": (account or {}).get("position_count"),
            "freshness": holdings_fresh,
        },
        "holdings_risk": holdings_risk,
        "holdings_risk_source": "account_state_snapshot" if acct_state_fresh else "holdings_only",
        "stop_policy": {"checks": stop_checks, "alerts": stop_alerts},
        "time_stops": time_stops,
        "time_stop_check": time_stop_state,
        "breach_protection": protection,
        "gate_evidence": gate_age,
        "factcheck_expiry": factcheck_expiry,
        "cache_freshness": cache_fresh,
        "dividends": {"as_of": div_as_of, "status": dividends.get("status"),
                      "reason": dividends.get("reason")},
        "next_actions": actions,
        "advisory": list(_ADVISORY),
        "exit_code": exit_code,
    }


def _fmt(value, suffix="") -> str:
    return "—" if value is None else f"{value}{suffix}"


def render_text(brief: dict) -> str:
    """把结构化简报渲染成一屏文本(缺失一律显示 — 不伪造 0)。"""
    lines: list[str] = []
    rd = brief["readiness"]
    head = f"每日简报  as_of={_fmt(brief['as_of'])}  决策日={_fmt(brief['decision_as_of'])}"
    lines.append(head)
    lines.append("=" * max(len(head), 40))

    status = {0: "平静(无待办)", 1: "系统不可信/失败", 2: "有需人工处理事项"}[brief["exit_code"]]
    lines.append(f"[状态] 退出码 {brief['exit_code']} — {status};"
                 f" readiness={_fmt(rd['status'])} ready={_fmt(rd['ready'])}")
    cf = brief.get("cache_freshness") or {}
    if cf.get("status") and cf["status"] != "FRESH":
        lines.append(f"  ⚠ 缓存新鲜度 {cf['status']}: {cf.get('detail')}")
    if rd["blockers"]:
        lines.append(f"  blockers: {', '.join(rd['blockers'])}")
    if rd["warnings"]:
        lines.append(f"  warnings: {', '.join(rd['warnings'])}")

    acct = brief["account"]
    lines.append("")
    lines.append(f"[账户] as_of={_fmt(acct['as_of'])} 现金={_fmt(acct['cash'])} "
                 f"总资产={_fmt(acct['total_assets'])} 持仓数={_fmt(acct['position_count'])} "
                 f"freshness={_fmt(acct['freshness'])}")

    lines.append("")
    div = brief["dividends"]
    div_note = f"股息叠加={div['status']}"
    if div.get("reason"):
        div_note += f"({div['reason']})"
    lines.append(f"[持仓风险] 来源={brief['holdings_risk_source']} {div_note}")
    for h in brief["holdings_risk"]:
        base = (f"  {_fmt(h.get('name'))}({_fmt(h.get('ts_code'))}) {_fmt(h.get('bucket'))} "
                f"×{_fmt(h.get('shares'))} 成本={_fmt(h.get('cost'))} 止损={_fmt(h.get('stop'))}")
        if brief["holdings_risk_source"] == "account_state_snapshot":
            base += (f" 现价={_fmt(h.get('close'))} 盈亏={_fmt(h.get('pnl_pct'), '%')}"
                     f" 距止损={_fmt(h.get('dist_stop_pct'), '%')} 持有={_fmt(h.get('held_days'))}日")
        base += (f" | 减半线={_fmt(h.get('profit_take_line'))} 股息率={_fmt(h.get('dv_ttm'), '%')}"
                 f" TTM指示现金≈{_fmt(h.get('indicative_ttm_cash'))}")
        lines.append(base)
    if not brief["holdings_risk"]:
        lines.append("  (无持仓)")

    prot = brief.get("breach_protection") or {}
    if prot.get("status") and prot["status"] != "NO_POSITIONS":
        lines.append("")
        icon = "✅" if prot["status"] == "VERIFIED" and not prot.get("uncovered") else "⚠"
        lines.append(f"[破线保护] {icon} {prot['status']}(盘中无自动监控,条件单为唯一防线)")
        lines.append(f"  {prot.get('note')}")
        if prot.get("uncovered"):
            lines.append(f"  无 active SELL 单: {', '.join(prot['uncovered'])}")

    ts_state = brief.get("time_stop_check") or {}
    tstops = brief.get("time_stops") or []
    if ts_state.get("status") == "NOT_CHECKED":
        lines.append("")
        lines.append(f"[短线时间止损] 未核查——{ts_state.get('reason')}")
    elif tstops:
        lines.append("")
        lines.append(f"[短线时间止损] {len(tstops)} 只达 10 交易日窗口"
                     "(含买入当日计数;同一笔在复盘账本记 9 日;提示,不替你卖)")
        for r in tstops:
            if r.get("status") == "TIME_STOP_UNKNOWN":
                lines.append(f"  {_fmt(r.get('ts_code'))} 日龄未知——{r.get('detail')}")
                continue
            # 显式标注计数口径:避免与复盘账本的 T+1 口径(恒少 1)对不上时被误读成
            # "提前催我卖"(sol 政策评审建议)
            lines.append(f"  {_fmt(r.get('ts_code'))} 日龄 {r.get('held_days')}/{r.get('limit')}"
                         "(含买入日;复盘账本 T+1 口径记少 1 日)")

    alerts = brief.get("stop_policy", {}).get("alerts", [])
    if alerts:
        lines.append("")
        lines.append(f"[止损政策核对] {len(alerts)} 只需人工处理(工具只提示不改 stop)")
        for r in alerts:
            lines.append(f"  {r['status']:<12} {_fmt(r.get('ts_code'))} {r.get('detail')}")

    sc = brief["decision_snapshot"]["state_counts"]
    lines.append("")
    lines.append(f"[机器决策] 快照={brief['decision_snapshot']['status']} "
                 f"BUY={sc['BUY']} WAIT={sc['WAIT']} HOLD={sc['HOLD']} EXIT={sc['EXIT']}")
    for b in brief["machine"]["buys"]:
        tag = "可执行" if b["actionable"] else f"强制WAIT({b['reason']})"
        lines.append(f"  BUY {_fmt(b.get('name'))}({_fmt(b.get('ts_code'))}) {tag} "
                     f"max_entry={_fmt(b.get('max_entry_price'))} 股数={_fmt(b.get('planned_shares'))}")
    for e in brief["machine"]["exits"]:
        lines.append(f"  EXIT {_fmt(e.get('name'))}({_fmt(e.get('ts_code'))}) "
                     f"{','.join(e.get('reason_codes', []))}")
    c2 = brief["machine"]["c2_watch"]
    if c2.get("status") != "AVAILABLE" or c2.get("reason"):
        lines.append(f"  C2状态={_fmt(c2.get('status'))}: {c2.get('reason')}")
    if c2["members"]:
        lines.append(f"  C2观察(WATCH) {len(c2['members'])} 只(仅信息展示): "
                     + ",".join(f"{_fmt(c.get('name'))}({_fmt(c.get('ts_code'))})"
                                for c in c2["members"]))
    for p in brief["machine"]["pending_factcheck"]:
        lines.append(f"  待fact-check {_fmt(p.get('name'))}({_fmt(p.get('ts_code'))}) (仍WAIT)")
    fx = brief.get("factcheck_expiry") or {}
    if fx.get("status") == "INVALID":
        lines.append(f"  ⚠ factcheck覆盖文件非法: {fx.get('error')}")
    elif fx.get("expiring"):
        lines.append(f"  factcheck将过期 {len(fx['expiring'])} 只"
                     f"(窗口 {fx.get('horizon_days')} 天,到期自动回 WAIT):")
        for r in fx["expiring"]:
            lines.append(f"    {_fmt(r.get('name'))}({_fmt(r.get('ts_code'))})"
                         f" 到期 {r.get('expires_on')}(还 {r.get('days_left')} 天)")

    lines.append("")
    lines.append("[今天该做什么]")
    if brief["next_actions"]:
        for a in brief["next_actions"]:
            lines.append(f"  {a}")
    else:
        lines.append("  无待办。")

    lines.append("")
    lines.append("[诚实语境]")
    for a in brief["advisory"]:
        lines.append(f"  - {a}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="每日一屏人类简报(只读聚合)")
    ap.add_argument("--json", action="store_true", help="输出结构化 JSON 而非文本")
    args = ap.parse_args(argv)

    brief = build_brief()
    if args.json:
        print(json.dumps(brief, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        print(render_text(brief))
    raise SystemExit(brief["exit_code"])


if __name__ == "__main__":
    main()
