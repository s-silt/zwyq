"""每日四态买入决策清单(spec §9/§10/§11)——JSON 为唯一真相源,终端只呈现。

Usage: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m scripts.buy_list [--as-of YYYYMMDD]

fail-loud(spec §11,任一命中即整场失败,不产出看似正常的快照):
factor snapshot 非最新行情日 / schema 缺生产字段 / holdings 重复或非法 / policy 矛盾。
决策不回写 holdings.json(模型建议≠已成交,spec §10)。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from datetime import datetime, timezone, timedelta

import pandas as pd

from ashare_gauntlet.account_state import (
    normalize_account_state,
    require_account_as_of,
)
from ashare_gauntlet.candidates import candidate_assessment, override_status
from ashare_gauntlet.config import CACHE_DIR as CACHE, HOLDINGS_PATH
from ashare_gauntlet.c2_review import C2ReviewError, eligible_codes, validate_state
from ashare_gauntlet.data.partition import date_partition_files
from ashare_gauntlet.portfolio_decision import decide_states, validate_policy
from scripts.illiq_capacity import BUCKETS, mv_terciles

FACTOR_DIR = "data/holdscore"
DECISION_DIR = "data/decisions"
POLICY_PATH = "data/trading_policy.json"
OVERRIDES_PATH = "data/factcheck_overrides.json"
REQUIRED_ROW_FIELDS = ("ts_code", "name", "industry", "decile", "tier",
                       "spec_crowd", "spike_limit", "score", "last", "mv",
                       "f_EP", "f_BP", "f_IVOL")   # 生产因子字段在场=snapshot 出自现役口径
ENTRY_MODEL_VERSION = "research-only"   # M2 过门前不得宣称择时(spec §13)
SIZE_BUCKET_RANK = {b: i for i, b in enumerate(BUCKETS)}   # 小0/中1/大2(X-08 接线)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant {value}")


def _c2_projection(status: str, *, last_valid_review_as_of: str | None = None,
                   watch: list[str] | None = None, exit_eligible: list[str] | None = None,
                   error: str | None = None) -> dict:
    return {
        "status": status,
        "last_valid_review_as_of": last_valid_review_as_of,
        "watch": sorted(str(code) for code in (watch or [])),
        "exit_eligible": sorted(str(code) for code in (exit_eligible or [])),
        "error": error,
    }


def load_c2_projection(path: str | os.PathLike[str]) -> dict:
    """Load and project validated C2 state without mutating the sidecar."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            state = json.load(fh, parse_constant=_reject_json_constant)
    except FileNotFoundError:
        return _c2_projection("NOT_INITIALIZED")
    except (OSError, UnicodeError):
        return _c2_projection("UNAVAILABLE", error="C2_STATE_UNREADABLE")
    except (json.JSONDecodeError, ValueError):
        return _c2_projection("UNAVAILABLE", error="C2_STATE_INVALID_JSON")

    try:
        validate_state(state)
        eligible = eligible_codes(state)
    except (C2ReviewError, TypeError, AttributeError, KeyError):
        return _c2_projection("UNAVAILABLE", error="C2_STATE_INVALID_SCHEMA")

    positions = state["positions"]
    watch = [code for code, row in positions.items() if row["status"] == "WATCH"]
    last_valid = state["last_valid_review_as_of"]
    reviews = state["reviews"]
    latest = max(enumerate(reviews), key=lambda item: (item[1]["period"], item[1]["as_of"], item[0]),
                 default=None)
    if latest is not None and latest[1]["status"] == "REVIEW_BLOCKED_DATA":
        issues = sorted(str(issue) for issue in latest[1]["issues"])
        return _c2_projection(
            "REVIEW_BLOCKED_DATA",
            last_valid_review_as_of=last_valid,
            watch=watch,
            exit_eligible=eligible,
            error="REVIEW_BLOCKED_DATA:" + ",".join(issues),
        )
    return _c2_projection(
        "AVAILABLE",
        last_valid_review_as_of=last_valid,
        watch=watch,
        exit_eligible=eligible,
    )


def size_tercile_ranks(rows: list[dict]) -> "dict[str, tuple[int, str]]":
    """X-08 生产接线(用户批准 2026-08-08):D10 全档内按 panel mv 三分位 → 排序秩。

    桶在 D10 **全档**上划(ex-ante,与 X-08 修正后口径一致),BUY/WAIT 排序第一键;
    依据=增量(S−PROD)净 +0.601%/期 NW t3.07(贴线,含 ~1/3 β),语义=选股偏好。
    fail-loud:D10 行 mv 非正有限数 → 排序静默失效,整场失败。<3 行=无三分位语义
    (X-04 mv_terciles 同精神)→ 空映射,排序回退 score,属合法退化截面非数据损坏。
    """
    d10 = [r for r in rows if r.get("decile") == 10]
    if len(d10) < 3:
        return {}
    mv: dict[str, float] = {}
    for r in d10:
        v = r.get("mv")
        if (not isinstance(v, (int, float)) or isinstance(v, bool)
                or not math.isfinite(v) or v <= 0):
            raise SystemExit(f"D10 行 {r['ts_code']} mv={v!r} 无效——市值桶排序无法执行,"
                             "不生成决策")
        mv[str(r["ts_code"])] = float(v)
    terc = mv_terciles(pd.Series(mv))
    return {str(c): (SIZE_BUCKET_RANK[str(b)], str(b)) for c, b in terc.items()}


def latest_trade_date(cache: "str | None" = None) -> str:
    files = date_partition_files(cache or CACHE, "daily")   # 调用时读全局(可测性)
    if not files:
        raise SystemExit("行情缓存为空——不生成决策")
    return os.path.basename(files[-1])[:8]


def validate_rows(rows: list[dict], held: "set[str] | None" = None) -> None:
    held = held or set()
    seen: set[str] = set()
    for i, r in enumerate(rows):
        missing = [f for f in REQUIRED_ROW_FIELDS if f not in r]
        if missing:
            raise SystemExit(f"factor snapshot 第{i}行缺生产字段 {missing}——不生成决策")
        ts = str(r["ts_code"])
        if ts in seen:
            raise SystemExit(f"factor snapshot 重复代码 {ts}——不生成决策")
        seen.add(ts)
        # 被消费的行(D10 候选/持仓)score 与 last 必须有限——NaN 会破坏排序确定性
        # 并伪装成合法证据(Codex review §③④)
        if r.get("decile") == 10 or ts in held:
            for f in ("score", "last"):
                v = r.get(f)
                if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
                    raise SystemExit(f"factor snapshot {ts} 字段 {f}={v!r} 无效——不生成决策")


def validate_holdings(h: dict) -> None:
    seen: set[str] = set()
    for p in h["positions"]:
        ts = p["ts_code"]
        if ts in seen:
            raise SystemExit(f"holdings 重复代码 {ts}——不生成决策")
        seen.add(ts)
        sh, cost = p["shares"], float(p["cost"])
        if (not isinstance(sh, (int, float)) or isinstance(sh, bool) or sh <= 0
                or float(sh) != int(sh) or not math.isfinite(cost) or cost <= 0):
            raise SystemExit(f"holdings 非法行 {ts}(shares={sh!r}/cost={cost!r})——不生成决策")
        # 行业上限依赖 industry/mv;缺失=约束静默失效(对抗审查:生产 holdings 正是这个形状)
        for f in ("industry", "mv", "last"):
            if p.get(f) in (None, ""):
                raise SystemExit(f"holdings {ts} 缺 {f} 字段——行业/风险约束无法执行,不生成决策")


def load_overrides(path: "str | None" = None) -> dict[str, dict]:
    try:
        data = json.load(open(path or OVERRIDES_PATH, encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(data, dict) or not isinstance(data.get("overrides"), list):
        raise ValueError("factcheck_overrides 顶层必须含 overrides 列表")
    result: dict[str, dict] = {}
    for index, item in enumerate(data["overrides"]):
        if not isinstance(item, dict):
            raise ValueError(f"factcheck_overrides 第 {index} 行必须是对象")
        required = {"ts_code", "as_of", "verdict", "expires_on"}
        missing = sorted(required - set(item))
        if missing:
            raise ValueError(f"factcheck_overrides 第 {index} 行缺字段 {missing}")
        code = item["ts_code"]
        if (not isinstance(code, str) or len(code) != 9 or code[6] != "."
                or not code[:6].isdigit() or code[7:] not in {"SH", "SZ"}
                or code in result):
            raise ValueError(f"factcheck_overrides 股票代码无效或重复: {code!r}")
        override_status(item, item["as_of"])
        result[code] = item
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", dest="as_of", default=None)
    a = ap.parse_args(argv)

    trade_date = latest_trade_date()
    as_of = a.as_of or trade_date
    snap_path = f"{FACTOR_DIR}/{as_of}_factor.json"
    if not os.path.exists(snap_path):
        raise SystemExit(f"无 factor snapshot {snap_path}——先跑 scripts.factor_rank")
    if as_of != trade_date:
        raise SystemExit(f"snapshot 日期 {as_of} ≠ 最新行情日 {trade_date}——陈旧数据不生成决策")

    # 复权/基础数据分区当日在场(深度完整性检查在上游 factor_rank/assert_adj_complete;
    # 此处防"snapshot 在而底层分区被删/未拉"的错配,spec §11)
    for ep in ("adj_factor", "daily_basic"):
        if not os.path.exists(f"{CACHE}/{ep}/{as_of}.parquet"):
            raise SystemExit(f"{ep}/{as_of} 分区缺失——底层数据不完整,不生成决策")

    rows = json.load(open(snap_path, encoding="utf-8"))
    hold = json.load(open(HOLDINGS_PATH, encoding="utf-8"))
    validate_holdings(hold)
    # P0-1: 账户状态归一化 + 严格日期门禁(任何 missing/invalid/stale/future → fail-loud)
    account = normalize_account_state(hold, expected_as_of=as_of)
    require_account_as_of(account, as_of)
    validate_rows(rows, held={p["ts_code"] for p in hold["positions"]})
    policy = json.load(open(POLICY_PATH, encoding="utf-8"))
    validate_policy(policy)
    overrides = load_overrides()

    held = {p["ts_code"]: p for p in hold["positions"]}
    snap_last = {str(r["ts_code"]): r.get("last") for r in rows}
    # 个人风险线(spec §8:触发 EXIT 但归因与因子模型分离)。价格口径=as_of 当日
    # snapshot 收盘价优先,持仓掉出快照时退回 holdings 手工价(对抗审查 P1:
    # holdings.last 为截图口径可能陈旧,snapshot 在场时不得使用)
    risk_breach: set[str] = set()
    for p in hold["positions"]:
        if not p.get("stop"):
            continue
        px = snap_last.get(p["ts_code"], p.get("last"))
        if px is None or not math.isfinite(float(px)):
            raise SystemExit(f"holdings {p['ts_code']} 无可用价格(snapshot 缺行且 last 无效)——不生成决策")
        if float(px) <= float(p["stop"]):
            risk_breach.add(p["ts_code"])
    manual_exit = {p["ts_code"] for p in hold["positions"] if p.get("logic_fail")}
    cash = hold.get("cash")
    account_value = (sum(float(p["mv"]) for p in hold["positions"]) + float(cash)
                     if cash is not None else None)

    # 相关股票 = D10 全档(BUY 候选与其 WAIT 理由)+ 当前持仓(HOLD/EXIT 判定)
    assessments = [candidate_assessment(r, overrides.get(str(r["ts_code"])), as_of)
                   for r in rows
                   if r.get("decile") == 10 or r["ts_code"] in held]
    # P0 修复(对抗审查):持仓股掉出 snapshot(变ST/🔴/亏损/停牌——恰是最危险情形)
    # 时仍须消费人工红灯覆盖,否则 verdict=red 被静默吞掉、错误输出 HOLD
    in_snap = {a["ts_code"] for a in assessments}
    for ts, p in held.items():
        if ts in in_snap:
            continue
        ov = overrides.get(ts)
        assessments.append({"ts_code": ts, "name": p.get("name", ts),
                            "industry": p.get("industry", "其他"), "score": None,
                            "last": p.get("last"), "decile": None, "spec_crowd": None,
                            "spike_limit": None, "eligible_buy": False,
                            "reason_codes": ["SNAPSHOT_MISSING"],
                            "governance_red": override_status(ov, as_of) == "red"})
    ranks = size_tercile_ranks(rows)                     # X-08:D10 全档划桶(ex-ante)
    for a_ in assessments:
        rk = ranks.get(str(a_["ts_code"]))
        if rk is not None:
            a_["size_rank"], a_["size_bucket"] = rk
    c2_path = os.path.join(DECISION_DIR, "c2_review_state.json")
    c2_state = load_c2_projection(c2_path)
    c2_exit_eligible = (
        set(c2_state["exit_eligible"])
        if c2_state["status"] in {"AVAILABLE", "REVIEW_BLOCKED_DATA"}
        else set()
    )
    decisions = decide_states(assessments, held, policy,
                              account_value=account_value,
                              cash=float(cash) if cash is not None else None,
                              risk_breach=risk_breach, manual_exit=manual_exit,
                              c2_exit_eligible=c2_exit_eligible)

    out = {"as_of": as_of,
           "account_as_of": account["as_of"],
           "account_source_schema": account["source_schema"],
           "generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
           "factor_snapshot": snap_path,
           "policy_version": str(policy["policy_version"]),
           "entry_model_version": ENTRY_MODEL_VERSION,
           "data_status": ("degraded" if c2_state["status"] == "UNAVAILABLE"
                           else "complete"),
           "c2_state": c2_state,
           "decisions": decisions}
    out_path = f"{DECISION_DIR}/{as_of}_buy_decisions.json"
    payload = json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False)
    os.makedirs(DECISION_DIR, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(
        suffix=".json", prefix=".tmp_buy_decisions_", dir=DECISION_DIR,
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, out_path)
    except BaseException:
        try:
            os.close(tmp_fd)
        except OSError:
            pass
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise

    print(f"=== 四态决策(as_of={as_of},entry_model={ENTRY_MODEL_VERSION},"
          f"policy v{policy['policy_version']})===")
    for state in ("BUY", "WAIT", "HOLD", "EXIT"):   # spec §9 CLI 分组序
        group = [d for d in decisions if d["state"] == state]
        if state == "WAIT" and len(group) > 15:
            shown, extra = group[:15], len(group) - 15
        else:
            shown, extra = group, 0
        print(f"[{state}] {len(group)} 只")
        for d in shown:
            ex = d["execution"]
            qty = f" {ex['shares']}股" if ex["shares"] else ""
            # 每行自带数据日期(spec §9:截屏/复制传播时单行不脱离口径)
            bk = d["evidence"].get("size_bucket")
            tag = f" {bk}桶" if bk else ""
            print(f"  {d['name']:　<6}{d['ts_code']}{qty}  {'/'.join(d['reason_codes'])}"
                  f"  [{as_of}]{tag}")
        if extra:
            print(f"  …另 {extra} 只 WAIT(完整名单见 JSON)")
    print(f"→ {out_path}(机器唯一真相源;非荐股,执行须人工确认)")
    if c2_state["status"] == "UNAVAILABLE":
        print(f"! C2 月度审视状态不可用({c2_state['error']});已保留独立硬退出并写入快照")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
