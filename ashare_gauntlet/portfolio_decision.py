"""四态决策与组合分配(spec §7/§8)——组合先于个股,分配确定可复现。

关键约定:
- HOLD:持仓未触发预定义退出即保留;跌出 D10 不手拍退出(spec §7,退出规则
  经组合级实验定为 C2 月度规则)→ HOLD + EXIT_RULE_C2_MONTHLY;
- EXIT 仅由预定义原因产生:GOVERNANCE_RED / RISK_LINE_BREACH(个人风险线,
  归因与因子模型分离)/ MANUAL_LOGIC_FAIL(人工逻辑失效覆盖);
- BUY 分配:先保留 HOLD,再按 score 降序(同分 ts_code 升序)遍历,行业 20% 上限、
  现金不透支、A股整手向下取整;账户状态未知 → shares=0 + ACCOUNT_STATE_MISSING
  (不伪造数量,spec §9);
- 全市场可以没有 BUY,不降门槛。
"""
from __future__ import annotations

import math


def validate_policy(policy: dict) -> None:
    """policy 参数自洽性(spec §11):矛盾即 ValueError,不静默修正。"""
    n, w = int(policy["target_positions"]), float(policy["target_weight"])
    cap, lot = float(policy["industry_cap"]), int(policy["lot_size"])
    if n <= 0 or w <= 0 or lot <= 0:
        raise ValueError(f"policy 非法:target_positions={n} target_weight={w} lot={lot}")
    if n * w > 1.0 + 1e-9:
        raise ValueError(f"policy 矛盾:目标持仓×单票权重={n * w:.2f}>100%")
    if w > cap + 1e-9:
        raise ValueError(f"policy 矛盾:单票权重 {w} 超过行业上限 {cap}")


def _exec(shares: int, weight: "float | None") -> dict:
    return {"eligible_from": "NEXT_TRADING_DAY", "max_entry_price": None,
            "target_weight": weight, "shares": shares}


# X-08 生产接线(用户批准 2026-08-08):BUY 候选排序第一键=D10 内市值桶
# (小0/中1/大2,由 buy_list.size_tercile_ranks 注入;增量净 +0.601%/期 NW t3.07 贴线,
# 语义=选股偏好,四关/治理/行业上限/资金分配不变)。无桶信息的候选按中桶排
# (不奖励不惩罚,旧调用方行为不变)。
SIZE_RANK_NEUTRAL = 1

# spec §9:失效条件(持有/买入后何种事件使决策作废,advisory 列表)
_INVALIDATIONS = {"BUY": ["LEAVE_PRODUCTION_UNIVERSE", "RISK_RED_FLAG", "FACTCHECK_EXPIRED"],
                  "HOLD": ["RISK_RED_FLAG", "MANUAL_LOGIC_FAIL"],
                  "EXIT": [], "WAIT": []}


def _mk(ts: str, name: str, state: str, codes: list[str], a: "dict | None",
        execution: dict) -> dict:
    return {"ts_code": ts, "name": name, "state": state, "reason_codes": codes,
            "evidence": _ev(a), "execution": execution,
            "invalidations": list(_INVALIDATIONS[state])}


def decide_states(assessments: list[dict], held: dict[str, dict], policy: dict,
                  account_value: "float | None", cash: "float | None",
                  risk_breach: "set[str] | None" = None,
                  manual_exit: "set[str] | None" = None) -> list[dict]:
    """候选评估 + 当前持仓 → 每只恰好一个状态的决策列表(确定性排序输出)。"""
    validate_policy(policy)
    risk_breach = risk_breach or set()
    manual_exit = manual_exit or set()
    if len({a["ts_code"] for a in assessments}) != len(assessments):
        raise ValueError("assessments 含重复 ts_code——上游 snapshot 未去重")
    by_code = {a["ts_code"]: a for a in assessments}
    decisions: list[dict] = []

    # —— 持仓:EXIT(仅预定义原因)否则 HOLD ——
    for ts, pos in held.items():
        a = by_code.get(ts)
        codes: list[str] = []
        if a is not None and a.get("governance_red"):
            codes.append("GOVERNANCE_RED")
        if ts in risk_breach:
            codes.append("RISK_LINE_BREACH")
        if ts in manual_exit:
            codes.append("MANUAL_LOGIC_FAIL")
        if codes:
            decisions.append(_mk(ts, pos.get("name", ts), "EXIT", codes, a, _exec(0, None)))
            continue
        hold_codes = ["HELD"]
        if a is None or not a.get("eligible_buy"):
            # 跌出生产资格≠当日退出:生产退出规则=C2(连续2个月度审视仍在D10档外才
            # EXIT,methodology §10 退出规则实验:净+0.33% vs 立即退出+0.30%,换手减半);
            # 月度审视=人工例行(仓库内暂无 c2_streak/审视日跟踪实现),日频快照只挂
            # 语义码不累计——消费方只能当观察集合呈现,不得逐日翻成"待退出"
            hold_codes.append("EXIT_RULE_C2_MONTHLY")
        decisions.append(_mk(ts, pos.get("name", ts), "HOLD", hold_codes, a, _exec(0, None)))

    # —— 未持仓:BUY 分配或 WAIT ——
    buyable = sorted((a for a in assessments if a["ts_code"] not in held and a["eligible_buy"]),
                     key=lambda a: (a.get("size_rank", SIZE_RANK_NEUTRAL),
                                    -(a.get("score") or 0.0), a["ts_code"]))
    slots = int(policy["target_positions"]) - len(held)
    lot = int(policy["lot_size"])
    w_target = float(policy["target_weight"])
    cap = float(policy["industry_cap"])
    # 行业权重 = 已持仓实际权重(账户已知时)+ 本轮已分配的计划权重。账户未知时
    # 持仓权重不可得,但计划权重仍必须记账——否则同行业可连发 BUY 突破上限
    # (Codex review P1-1)
    ind_weight: dict[str, float] = {}
    if account_value:
        for pos in held.values():
            ind = pos.get("industry", "其他")
            ind_weight[ind] = ind_weight.get(ind, 0.0) + float(pos.get("mv", 0.0)) / account_value
    remaining_cash = cash

    for a in buyable:
        wait_codes: list[str] = []
        if slots <= 0:
            wait_codes.append("PORTFOLIO_FULL")
        ind = a.get("industry", "其他")
        if ind_weight.get(ind, 0.0) + w_target > cap + 1e-9:
            wait_codes.append("INDUSTRY_CAP")
        if wait_codes:
            decisions.append(_mk(a["ts_code"], a["name"], "WAIT", wait_codes, a, _exec(0, None)))
            continue
        if account_value is None or cash is None or remaining_cash is None:
            # 账户状态未知:状态仍 BUY(资格成立),数量不伪造(spec §9),计划权重照记
            decisions.append(_mk(a["ts_code"], a["name"], "BUY",
                                 a["reason_codes"] + ["ACCOUNT_STATE_MISSING"], a,
                                 _exec(0, w_target)))
            slots -= 1
            ind_weight[ind] = ind_weight.get(ind, 0.0) + w_target
            continue
        price = float(a["last"])
        shares = math.floor(w_target * (account_value or 0.0) / price / lot) * lot
        cost = shares * price
        downsized = False
        if cost > remaining_cash:                    # 现金不足按现金缩量,不透支
            shares = math.floor(remaining_cash / price / lot) * lot
            cost = shares * price
            downsized = True
        if shares <= 0:
            decisions.append(_mk(a["ts_code"], a["name"], "WAIT",
                                 ["INSUFFICIENT_CASH"], a, _exec(0, None)))
            continue
        remaining_cash -= cost
        slots -= 1
        # 行业记账用实际成交权重(缩量时 ≠ 目标权重,记满额会错误挡住后续候选——审查 P2)
        actual_w = cost / account_value
        ind_weight[ind] = ind_weight.get(ind, 0.0) + actual_w
        codes = list(a["reason_codes"]) + (["DOWNSIZED_BY_CASH"] if downsized else [])
        ex = _exec(shares, round(actual_w, 4) if downsized else w_target)
        decisions.append(_mk(a["ts_code"], a["name"], "BUY", codes, a, ex))

    # —— 未持仓且不合格:WAIT(带否决码)——
    for a in assessments:
        if a["ts_code"] in held or a["eligible_buy"]:
            continue
        decisions.append(_mk(a["ts_code"], a["name"], "WAIT",
                             list(a["reason_codes"]), a, _exec(0, None)))

    # 分组顺序与 CLI 一致(spec §9:BUY/WAIT/HOLD/EXIT)
    order = {"BUY": 0, "WAIT": 1, "HOLD": 2, "EXIT": 3}
    decisions.sort(key=lambda d: (order[d["state"]],
                                  d["evidence"].get("size_rank", SIZE_RANK_NEUTRAL),
                                  -(d["evidence"].get("score") or 0.0), d["ts_code"]))
    return decisions


def _ev(a: "dict | None") -> dict:
    if a is None:
        return {}
    ev = {"score": a.get("score"), "industry": a.get("industry"), "last": a.get("last"),
          "decile": a.get("decile"), "spec_crowd": a.get("spec_crowd"),
          "spike_limit": a.get("spike_limit")}
    if "size_rank" in a:
        ev["size_rank"], ev["size_bucket"] = a["size_rank"], a.get("size_bucket")
    return ev
